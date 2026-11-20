"""End-to-end pipeline test with a scripted LLM and canned sources: no network, no API keys."""

import json

import httpx

from brain.cache import Cache
from brain.draft import check, salutation
from brain.fetch import Fetcher
from brain.llm import LLM, LLMBudgetExceeded, LLMError
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence, SenderProfile
from brain.packs import get_pack
from brain.pipeline import Deps, run_campaign
from brain.settings import Settings
from brain.sources.orcid import Orcid
from brain.sources.websearch import WebSearch
from brain.spec import build_spec
from brain.store import Store

SENDER = SenderProfile(name="Riya Shah", headline="Third-year CS undergraduate at NIT Trichy",
                       summary="Built a GNN-based molecule property predictor reaching 0.91 ROC-AUC on Tox21.",
                       highlights=["GNN molecule property predictor, 0.91 ROC-AUC on Tox21"], skills=["PyTorch"])

BODY_OK = ("Your paper on scalable message passing for molecular graphs is close to what I have been building. "
           "I am a third-year computer science undergraduate at NIT Trichy, and over the past year I built a "
           "graph neural network that predicts molecular properties, reaching 0.91 ROC-AUC on Tox21. Working on "
           "it made me curious about how message passing behaves as graphs grow, which is the question your group "
           "studies. I would like to spend the summer contributing to that line of work, and I can share code and "
           "a short write-up of my results. Would you be open to a brief conversation about a summer research "
           "internship in your group?")


def person(i, org="IIT Ropar", domain="iitrpr.ac.in", email=True, text="Scalable message passing for molecular graphs"):
    c = Contact(name=f"Asha{chr(65 + i)} Rani", org=org, org_domain=domain, country="IN",
                location_text="Rupnagar, Punjab, India", keys=[f"openalex:A{i}"], sources=["openalex"],
                evidence=[Evidence(kind="publication", text=text, year=2026)],
                signals={"matched_works": 2, "senior_positions": 1, "seniority": 0.8, "latest_year": 2026})
    if email:
        c.emails = [EmailCandidate(address=f"asha{chr(97 + i)}@{domain}", confidence=0.95, method="mailto")]
    return c


class FakeOpenAlex:
    requests = 0
    failed_terms = []

    def __init__(self, people):
        self.people = people

    def search(self, spec):
        return [c.model_copy(deep=True) for c in self.people]


class ScriptedLLM:
    """Stands in for the HTTP transport; answers by which prompt it is given."""

    def __init__(self, body=BODY_OK, unsupported=None):
        self.body, self.seen = body, []
        self.unsupported = dict(unsupported or {})  # contact id -> claims, reported once

    def __call__(self, model, system, user, temperature, max_tokens):
        self.seen.append(system[:40])
        if system.startswith("You assess"):
            cards = json.loads(user.split("Candidates:\n", 1)[1])
            return json.dumps({"results": [
                {"id": c["id"], "score": 20 if "Soil" in json.dumps(c["evidence"]) else 85,
                 "reasons": ["works on GNNs for molecules"], "hook": "x", "hook_ref": 0} for c in cards]})
        if system.startswith("You screen"):  # the pre-judge: everyone plausible except the soil chemist
            cards = json.loads(user.split("Candidates:\n", 1)[1])
            return json.dumps({"results": [{"id": c["id"], "score": 15 if "Soil" in json.dumps(c) else 80,
                                            "why": "x"} for c in cards]})
        if system.startswith("You fact-check"):
            ids = [line.split("id: ", 1)[1].strip() for line in user.splitlines() if line.startswith("### id: ")]
            return json.dumps({"results": [{"id": i, "unsupported": list(self.unsupported.pop(i, []))} for i in ids]})
        if system.startswith("You write"):
            return json.dumps({"subject": "Summer internship on message passing", "body": self.body})
        if system.startswith("You configure"):
            return json.dumps({"pack": "academic", "objective": "Summer research internship", "ask": "An internship",
                               "persona": {"description": "faculty in GNNs", "roles": ["professor"]},
                               "topics": ["graph neural networks"], "expanded_terms": ["message passing"],
                               "locations": ["Punjab, India"], "search_queries": ["gnn faculty punjab"], "missing": []})
        raise AssertionError("unexpected prompt")


def make_deps(tmp_path, people, llm_script=None, max_calls=50):
    settings = Settings()
    settings.serper_api_key = settings.tavily_api_key = ""
    settings.llm_max_calls = max_calls
    settings.database_url = f"sqlite:///{tmp_path}/t.db"
    cache = Cache()
    offline = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
    script = llm_script or ScriptedLLM()
    return Deps(settings=settings, cache=cache, llm=LLM(settings, cache, transport=script),
                fetcher=Fetcher(settings, cache, client=offline), search=WebSearch(settings, cache, client=offline),
                orcid=Orcid(settings, cache, client=offline), openalex=FakeOpenAlex(people),
                store=Store(settings.database_url)), script


def spec(n=3):
    return CampaignSpec(id="camp1", pack="academic", objective="Summer research internship",
                        ask="A summer research internship in your group", topics=["message passing molecular graphs"],
                        countries=["IN"], regions=["Punjab"], target_count=n, sender=SENDER, timeline="Summer 2027")


def test_full_run_and_llm_call_budget(tmp_path):
    people = [person(i) for i in range(6)] + [person(6, text="Soil chemistry of wheat"), person(7, email=False)]
    deps, script = make_deps(tmp_path, people)
    result = run_campaign(spec(3), deps)

    f = result.funnel
    # the one without an email is not judged or drafted; the soil chemist is screened out before enrichment
    assert f["unique_people"] == 8 and f["reachable"] == 6 and f["prejudge"] == {"screened": 8, "rejected": 1}
    assert len(result.shortlist) == 3 and all(c.score >= 0.5 for c in result.shortlist)
    assert all("Soil" not in c.evidence[0].text for c in result.shortlist)
    assert f["drafted_ok"] == 3 and f["drafts_need_review"] == 0
    d = result.drafts[result.shortlist[0].id]
    assert d.body.startswith("Dear Professor Rani,") and d.body.endswith("Riya Shah") and d.to.endswith("@iitrpr.ac.in")
    # one batched pre-judge, 2x target (6) judged in ONE batched call, one call per draft, one batched fact-check:
    # 6 calls for 3 emails.
    assert f["cost"]["llm"]["paid_calls"] == 6
    assert f["cost"]["llm"]["by_purpose"] == {"prejudge": 1, "judge": 1, "draft": 3, "verify": 1}
    fc = {k: v for k, v in f["fact_check"].items() if k not in ("verifier", "independent_of_writer")}
    assert fc == {"checked": 3, "flagged": 0, "fixed": 0, "still_flagged": 0}
    assert f["fact_check"]["independent_of_writer"] is True  # the checker is not the model that wrote the drafts


def test_resume_repeats_no_paid_work(tmp_path):
    people = [person(i) for i in range(4)]
    deps, _ = make_deps(tmp_path, people)
    run_campaign(spec(2), deps, upto="judge")
    deps2, script2 = make_deps(tmp_path, [])  # discovery would now find nobody: resume must not need it
    result = run_campaign(spec(2), deps2)
    assert len(result.shortlist) == 2 and len(result.drafts) == 2
    assert not any(s.startswith("You assess") for s in script2.seen)  # judging was not paid for twice


def test_suppression_and_already_contacted_are_enforced(tmp_path):
    people = [person(i) for i in range(4)]
    deps, _ = make_deps(tmp_path, people)
    deps.store.suppress("ashaa@iitrpr.ac.in")
    first = run_campaign(spec(2), deps)
    assert all(c.best_email.address != "ashaa@iitrpr.ac.in" for c in first.shortlist)
    sent = first.shortlist[0]
    deps.store.set_status("camp1", sent.id, "sent")
    s2 = spec(2)
    s2.id = "camp2"
    second = run_campaign(s2, deps)
    assert sent.best_email.address not in [c.best_email.address for c in second.shortlist]


def test_ungrounded_draft_is_flagged_not_sent_silently(tmp_path):
    bad = BODY_OK.replace("0.91 ROC-AUC on Tox21", "0.99 ROC-AUC on Tox21, published at NeurIPS 2025")
    deps, script = make_deps(tmp_path, [person(0), person(1)], ScriptedLLM(body=bad))
    result = run_campaign(spec(1), deps)
    d = list(result.drafts.values())[0]
    assert not d.ok and any("0.99" in i for i in d.issues)
    assert result.funnel["drafts_need_review"] == 1
    assert deps.store.results("camp1")[0]["status"] == "needs_review"
    assert sum(s.startswith("You write") for s in script.seen) == 2  # exactly one retry


def test_draft_checks():
    c = person(0)
    sp = spec()
    assert check(BODY_OK, "Subject", c, sp) == []
    assert any("placeholder" in i for i in check(BODY_OK + " [Your Name]", "S", c, sp))
    assert any("quotes a title" in i for i in check(BODY_OK + ' I enjoyed "A Paper That Does Not Exist At All".', "S", c, sp))
    assert any("filler" in i for i in check("I hope this email finds you well. " + BODY_OK, "S", c, sp))
    assert salutation(c, get_pack("general")) == "Hi AshaA,"


def test_llm_gateway_caches_enforces_budget_and_fails_loudly(tmp_path):
    calls = []

    def transport(model, system, user, temperature, max_tokens):
        calls.append(user)
        return "not json" if user.startswith("bad") else '```json\n{"ok": true}\n```'

    settings = Settings()
    settings.llm_max_calls = 3
    llm = LLM(settings, Cache(), transport=transport)
    assert llm.json(system="s", user="u", purpose="p") == {"ok": True}
    assert llm.json(system="s", user="u", purpose="p") == {"ok": True}
    assert len(calls) == 1 and llm.cached_hits == 1  # second call was free
    try:
        llm.json(system="s", user="bad", purpose="p")
        raise AssertionError("expected LLMError")
    except LLMBudgetExceeded:
        pass  # unparseable output is retried until the budget stops it
    except LLMError:
        pass
    try:
        llm.json(system="s", user="another", purpose="p")
        raise AssertionError("expected budget error")
    except LLMBudgetExceeded:
        pass


def test_llm_parse_failure_is_fed_back_and_repaired():
    seen = []

    def transport(model, system, user, temperature, max_tokens):
        seen.append(user)
        return '{"ok": true}' if "could not be parsed" in user else "Sure! here you go"

    llm = LLM(Settings(), Cache(), transport=transport)
    assert llm.json(system="s", user="u", purpose="p") == {"ok": True}
    assert len(seen) == 2 and seen[0] == "u" and "Sure! here you go" in seen[1]


def test_spec_is_built_in_one_call(tmp_path):
    deps, script = make_deps(tmp_path, [])
    sp, questions = build_spec("I want a summer research internship on GNNs in Punjab, India", SENDER, deps.llm)
    assert sp.countries == ["IN"] and sp.regions == ["Punjab"] and sp.pack == "academic" and questions == []
    assert deps.llm.calls == 1


def test_bring_your_own_list_skips_discovery(tmp_path):
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "Name,Email,Company,Title,Website,Notes\n"
        "Dana Whitfield,dana@acmerobotics.com,Acme Robotics,Head of Engineering,https://acmerobotics.com,"
        "Leads the perception team building warehouse robots\n"
        "Info Desk,info@acmerobotics.com,Acme Robotics,,,\n"
        "Sam Ortega,,Northwind Labs,CTO,https://northwind.dev,Works on reinforcement learning for logistics\n")
    deps, _ = make_deps(tmp_path, [person(0)])
    sp = spec(2)
    sp.pack, sp.countries, sp.regions = "general", [], []
    result = run_campaign(sp, deps, csv_path=str(csv_path))
    assert result.funnel["by_source"] == {"csv": 3}  # publication search was not used
    # Only the person with a real, personal address is drafted for: info@ is a role inbox, Sam has no email.
    assert [c.name for c in result.shortlist] == ["Dana Whitfield"]
    assert list(result.drafts.values())[0].body.startswith("Hi Dana,")


def test_fact_check_rewrites_unsupported_claims(tmp_path):
    people = [person(i) for i in range(3)]
    ids = {c.name: c for c in people}
    from brain.resolve import assign_id
    target = assign_id(people[0])
    deps, script = make_deps(tmp_path, people, ScriptedLLM(unsupported={target: ["your work on privacy"]}))
    result = run_campaign(spec(3), deps)
    fc = {k: v for k, v in result.funnel["fact_check"].items() if k not in ("verifier", "independent_of_writer")}
    assert fc == {"checked": 3, "flagged": 1, "fixed": 1, "still_flagged": 0}
    assert all(d.ok for d in result.drafts.values())
    assert deps.llm.usage["draft_fix"] == 1 and deps.llm.usage["verify"] == 2  # initial batch + re-check of the rewrite


def test_conflation_check_catches_mixed_up_cv_items():
    from brain.draft import conflations
    highlights = ["Sifchain Finance: Architected scalable AI applications with TypeScript and NLP models",
                  "Sifchain Finance: Developed blockchain plugins for Polygon enabling AI agents to use DeFi protocols",
                  "Project: Created a multi-model stock price prediction system using LSTM, GRU and Transformers"]
    bad = "In my role at Sifchain Finance I created a stock prediction system with LSTM and GRU models."
    good = ("At Sifchain Finance I architected AI applications with NLP models. "
            "Separately, I built a stock prediction system using LSTM and GRU models.")
    assert conflations(bad, highlights) and "Sifchain Finance" in conflations(bad, highlights)[0]
    assert conflations(good, highlights) == []
    assert conflations("At Sifchain Finance I architected NLP applications, and in a separate project I built a stock "
                       "prediction system using LSTM and GRU models.", highlights) == []
    # Work that really was done at the employer is never flagged, even in a different bullet.
    assert conflations("At Sifchain Finance I built Polygon plugins so AI agents can use DeFi protocols.", highlights) == []
    # Profiles without labels fall back to 'at <Org>' phrases.
    assert conflations(bad, ["Software developer at Sifchain Finance building NLP apps",
                             "Created a stock prediction system using LSTM and GRU"])


def test_rate_limited_model_falls_back_instead_of_failing():
    from brain.llm import _RateLimited
    used = []

    def transport(model, system, user, temperature, max_tokens):
        used.append(model)
        if model == "big":
            raise _RateLimited("big rate-limited for 600s")
        return '{"ok": true}'

    st = Settings()
    st.llm_model_strong, st.llm_model_fast, st.llm_fallback_models = "big", "small", "backup"
    llm = LLM(st, Cache(), transport=transport)
    assert llm.json(system="s", user="u", purpose="judge", tier="strong") == {"ok": True}
    assert llm.json(system="s", user="v", purpose="judge", tier="strong") == {"ok": True}
    assert used == ["big", "backup", "backup"]  # the capped model is not retried for the rest of the run
    assert llm.calls == 2 and llm.report()["model_switches"] == ["big -> backup"]


def test_model_added_greeting_is_removed():
    from brain.draft import clean_body
    org = Contact(kind="org", name="Reskilll")
    assert clean_body("Reskilll team,\n\nWe run a fest.", org) == "We run a fest."
    assert clean_body("Dear Priya,\nWe run a fest.", Contact(name="Priya Nair")) == "We run a fest."
    assert clean_body("We run a fest, and more.\nSecond line.", org).startswith("We run a fest")


def test_failed_judging_is_retried_on_resume(tmp_path):
    people = [person(i) for i in range(3)]

    class Flaky(ScriptedLLM):
        def __call__(self, model, system, user, temperature, max_tokens):
            if system.startswith("You assess"):
                raise LLMError("rate limited")
            return super().__call__(model, system, user, temperature, max_tokens)

    deps, _ = make_deps(tmp_path, people, Flaky())
    first = run_campaign(spec(2), deps)
    assert first.shortlist == [] and first.funnel["judged"] == 0
    deps2, _ = make_deps(tmp_path, [])
    second = run_campaign(spec(2), deps2)  # same campaign id: resumes, and must redo judging
    assert len(second.shortlist) == 2 and len(second.drafts) == 2

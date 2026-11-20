"""Discovery v2: triage, fit model, signal channels, adaptive exploration, memory, backfill, replay, bench.
No network, no API keys: every external answer is scripted."""

import json

import httpx

from brain.cache import Cache
from brain.context import CampaignContext
from brain.fetch import Fetcher
from brain.fit import anti_pattern_hits, fit_scores, is_qualified, profile_for, promising, reach_factor
from brain.llm import LLM
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence, FitCriterion, ProspectProfile, SenderProfile, TargetPersona
from brain.playbooks import get_playbook
from brain.searchplan import SearchPlanner
from brain.settings import Settings
from brain.sources.websearch import Hit, WebSearch
from brain.store import Store, query_template


def _spec(**kw):
    base = dict(id="v2", pack="sponsorship", objective="Sponsors for a college tech fest", ask="Sponsor our fest",
                persona=TargetPersona(description="tech companies", roles=["Head of Marketing", "Partnerships"]),
                topics=["developer tools", "edtech"], countries=["IN"], regions=["Bangalore"], target_count=5,
                sender=SenderProfile(name="T"))
    base.update(kw)
    return CampaignSpec(**base)


# ── triage ───────────────────────────────────────────────────────────────────
def test_triage_heuristics_and_llm_fallback():
    from brain.orgs import is_aggregator
    from brain.triage import heuristic_kind, triage

    k = lambda url, title="", snip="": heuristic_kind(Hit(url, title, snip), is_aggregator)
    assert k("https://devtoolco.io/") == "own_site"
    assert k("https://devtoolco.io/about") == "own_site"
    assert k("https://inc42.com/top-edtech") == "list"
    assert k("https://example.com/x", "Top 25 edtech startups in India") == "list"
    assert k("https://economictimes.indiatimes.com/tech/startups/x") == "article"
    assert k("https://acme.com/blog/how-we-scaled") == "article"
    assert k("https://hackfest.in/2025/sponsors", "HackFest 2025 Sponsors", "our partners") == "event"
    assert k("https://acme.com/products/widgets/pro") == "unsure"

    class Boom:
        available = True

        def json(self, **kw):
            raise RuntimeError("down")

    spec = _spec()
    hits = [Hit("https://acme.com/products/widgets/pro", "Acme Pro")]
    assert triage(hits, spec, is_aggregator, Boom()) == [(hits[0], "own_site")]  # failure keeps the old behaviour

    class Judge:
        available = True

        def json(self, **kw):
            return {"results": [{"i": 0, "kind": "irrelevant"}]}

    stats = {}
    assert triage(hits, spec, is_aggregator, Judge(), stats)[0][1] == "irrelevant" and stats["triage_unsure"] == 1


# ── fit model ────────────────────────────────────────────────────────────────
def test_profile_merges_playbook_defaults_and_scores_evidence():
    spec = _spec(profile=ProspectProfile(criteria=[FitCriterion(name="runs student developer programs",
                                                                  signal="audience", terms=["student", "developer"])]))
    prof = profile_for(spec, get_playbook("sponsorship"))
    names = [c.name for c in prof.criteria]
    assert names[0].startswith("works on or covers any of: developer tools, edtech") and "has sponsored similar events" in names
    assert "runs student developer programs" in names and "sponsorship agency" in prof.anti_patterns

    sponsor = Contact(kind="org", name="DevToolCo", evidence=[Evidence(kind="page", text="Sponsor or partner of HackFest 2025")],
                      bio="Developer tools and APIs used by 40,000 student developers", signals={"past_sponsorship": 1.0})
    vague = Contact(kind="org", name="Random Bakery", bio="Fresh bread every morning")
    agency = Contact(kind="org", name="Fest Masters", bio="Sponsorship agency for developer tools brands and edtech events")
    f1, per1, _ = fit_scores(sponsor, prof)
    f2, _, _ = fit_scores(vague, prof)
    f3, _, _ = fit_scores(agency, prof)
    assert per1["has sponsored similar events"] >= 0.6 and f1 > 0.4 > f2
    assert anti_pattern_hits(prof, agency) == ["sponsorship agency"] and f3 < f1
    assert is_qualified(sponsor, prof) and not is_qualified(vague, prof)
    assert promising(sponsor, prof) and not promising(agency, prof)


def test_reach_factor_prefers_personal_addresses():
    c = Contact(name="A B")
    assert reach_factor(c) == 0.0
    c.add_email(EmailCandidate(address="info@x.com", confidence=0.7, method="inbox"))
    assert reach_factor(c) == 0.9
    c.emails = [EmailCandidate(address="a@x.com", confidence=0.9, method="mailto")]
    assert reach_factor(c) == 1.0


def test_prescore_uses_proof_of_fit_for_organisations():
    from brain.rank import prescore

    spec = _spec()
    proven = Contact(kind="org", name="DevToolCo", bio="developer tools", signals={"past_sponsorship": 1.0},
                     evidence=[Evidence(kind="page", text="Sponsor or partner of HackFest")])
    plain = Contact(kind="org", name="ToolsInc", bio="developer tools")
    ranked = prescore([plain, proven], spec, get_playbook("sponsorship"))
    assert ranked[0].name == "DevToolCo" and ranked[0].criteria["has sponsored similar events"] >= 0.6


# ── channels ─────────────────────────────────────────────────────────────────
EVENT_HTML = """<html><head><title>HackFest 2025</title></head><body><h1>HackFest 2025</h1>
<section id="sponsors"><h2>Our Sponsors</h2>
  <a href="https://devtoolco.io/"><img src="d.png" alt="DevToolCo logo"></a>
  <a href="https://www.learnloop.in/?ref=hf"><img src="l.png" alt="logo"></a>
  <a href="https://instagram.com/hackfest">Instagram</a>
  <img src="x.png" alt="Byte Cloud">
</section>
<footer><a href="https://hackfest.in/contact">Contact</a></footer></body></html>"""


def test_event_page_sponsors_are_proof_of_past_sponsorship():
    from brain.channels import orgs_from_event_page

    orgs = orgs_from_event_page(EVENT_HTML, "https://hackfest.in/", "HackFest 2025")
    got = {o.name: o.org_domain for o in orgs}
    assert got == {"DevToolCo": "devtoolco.io", "Learnloop": "learnloop.in", "Byte Cloud": None}
    assert all(o.signals["past_sponsorship"] == 1.0 for o in orgs)
    assert orgs[0].evidence[0].text == "Sponsor or partner of HackFest 2025"


def test_ats_tokens_and_job_boards():
    from brain.channels import ats_token, fetch_board

    assert ats_token("https://boards.greenhouse.io/acmeai/jobs/123") == ("greenhouse", "acmeai")
    assert ats_token("https://jobs.lever.co/visionco/abc-def") == ("lever", "visionco")
    assert ats_token("https://jobs.ashbyhq.com/genco") == ("ashby", "genco")
    assert ats_token("https://acme.com/careers") is None

    def router(req):
        u = str(req.url)
        if u.endswith("/boards/acmeai"):
            return httpx.Response(200, json={"name": "Acme AI"})
        if u.endswith("/boards/acmeai/jobs"):
            return httpx.Response(200, json={"jobs": [{"title": "ML Engineer", "location": {"name": "Bengaluru"},
                                                        "absolute_url": "https://x/1", "updated_at": "2026-10-01T00:00"}]})
        return httpx.Response(404)

    st = Settings()
    f = Fetcher(st, Cache(), client=httpx.Client(transport=httpx.MockTransport(router)))
    name, jobs = fetch_board(f, "greenhouse", "acmeai")
    assert name == "Acme AI" and jobs == [{"title": "ML Engineer", "location": "Bengaluru", "url": "https://x/1",
                                           "date": "2026-10-01"}]
    assert f.get_json("https://boards-api.greenhouse.io/v1/boards/acmeai") == {"name": "Acme AI"}  # cached now


def test_podcast_feed_gives_host_and_published_email():
    from brain.channels import parse_feed

    xml = """<?xml version="1.0"?><rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
    <title>AI in India</title><itunes:author>Meera Iyer</itunes:author><description>Weekly AI conversations</description>
    <itunes:owner><itunes:name>Meera Iyer</itunes:name><itunes:email>Meera@AIindia.fm</itunes:email></itunes:owner>
    <item><title>Speech models for Indic languages</title><pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    info = parse_feed(xml)
    assert info["email"] == "meera@aiindia.fm" and info["owner"] == "Meera Iyer"
    assert info["episodes"][0]["title"] == "Speech models for Indic languages"


def test_decision_maker_from_search_titles_and_hiring_signal():
    from brain.orgs import hiring_signal, people_from_serp
    from brain.pages import parse

    hits = [Hit("https://in.linkedin.com/in/x", "Rahul Verma - Software Engineer - Other Co | LinkedIn"),
            Hit("https://in.linkedin.com/in/y", "Priya Nair - Head of Partnerships - DevToolCo | LinkedIn"),
            Hit("https://devtoolco.io/team", "Team | DevToolCo")]
    assert people_from_serp(hits, "DevToolCo", ["Head of Partnerships", "Head of Marketing"]) == \
        ("Priya Nair", "Head of Partnerships", "https://in.linkedin.com/in/y")
    assert people_from_serp(hits[:1], "DevToolCo", ["Head of Partnerships"]) is None  # wrong organisation

    ats = parse('<a href="https://boards.greenhouse.io/devtoolco">Jobs</a>', "https://devtoolco.io/")
    careers = parse('<nav><a href="/careers">Careers</a></nav>', "https://devtoolco.io/")
    assert hiring_signal(ats) == 1.0 and hiring_signal(careers) == 0.5
    assert hiring_signal(parse("<p>hello</p>", "https://x.io/")) == 0.0


def test_salutation_addresses_the_decision_maker_through_the_inbox():
    from brain.draft import salutation

    org = Contact(kind="org", name="Smile Dental", decision_maker={"name": "Anita Rao", "title": "Owner", "source": "u"})
    assert salutation(org, get_playbook("b2b_sales")) == "Hi Anita,"
    assert salutation(Contact(kind="org", name="Smile Dental"), get_playbook("b2b_sales")) == "Hello Smile Dental team,"


# ── exploration ──────────────────────────────────────────────────────────────
def test_tool_choice_follows_criteria_not_timing():
    from brain.explore import choose_tools

    spec = _spec(pack="b2b_sales", topics=["dental clinic"])
    seed, expand = choose_tools(spec, get_playbook("b2b_sales"), profile_for(spec, get_playbook("b2b_sales")),
                                has_seeds=False)
    assert seed == ["memory", "places", "orgs"] and expand == ["lookalike"]  # "growing" (timing) adds no job boards
    # a criterion needing proof that no listed tool gives adds the tool that does
    spec2 = _spec(pack="general", profile=ProspectProfile(criteria=[FitCriterion(name="hiring now", signal="hiring")]))
    seed2, _ = choose_tools(spec2, get_playbook("general"), profile_for(spec2, get_playbook("general")), has_seeds=True)
    assert "jobs" in seed2 and "lookalike" not in seed2  # general has no lookalike tool listed
    seed3, _ = choose_tools(_spec(), get_playbook("sponsorship"), profile_for(_spec(), get_playbook("sponsorship")),
                            has_seeds=True)
    assert seed3[:2] == ["memory", "lookalike"]  # the user's examples come first


def test_explore_moves_budget_to_the_productive_tool_and_expands_from_winners(monkeypatch):
    import brain.explore as ex
    from brain.tools import TOOLS, Tool

    calls = []

    def make(name, n_good, n_bad, per_call_cost):
        def fn(run):
            calls.append((name, run.budget, [w.name for w in run.winners]))
            run.deps.search.calls += per_call_cost
            i = len(calls)
            good = [Contact(kind="org", name=f"{name}{i}-{j}", org_domain=f"{name}{i}{j}.com",
                            bio="developer tools for students", signals={"past_sponsorship": 1.0}) for j in range(n_good)]
            bad = [Contact(kind="org", name=f"{name}bad{i}-{j}", org_domain=f"{name}b{i}{j}.com", bio="bakery")
                   for j in range(n_bad)]
            return good + bad, {"queries": per_call_cost}
        return fn

    fake = {"memory": Tool("memory", "m", "free", lambda run: ([], {}), stage="seed"),
            "events": Tool("events", "e", "search", make("ev", 4, 0, 3), True, ("past_sponsorship",), ("person", "org")),
            "orgs": Tool("orgs", "o", "search", make("or", 0, 5, 3), True, ("topic",), ("org",)),
            "lookalike": Tool("lookalike", "l", "search", make("lk", 2, 0, 2), True, ("topic",), ("org",), "expand")}
    monkeypatch.setattr(ex, "TOOLS", fake)
    st = Settings()
    st.serper_api_key = "x"

    class D:
        settings = st
        search = WebSearch(st, Cache())
        store = None

    ctx = CampaignContext()
    spec = _spec()
    planner = SearchPlanner(spec, get_playbook("sponsorship"), ctx, need=12)
    res = ex.explore(spec, get_playbook("sponsorship"), ctx, D(), planner, budget=30, rounds=3)
    names = [c[0] for c in calls]
    assert names[:2] == ["ev", "or"]  # seed round: primary tool first, with the bigger share
    assert calls[0][1] > calls[1][1]
    assert "lk" in names and calls[names.index("lk")][2]  # expansion ran, seeded with winners
    assert ctx.tool_stats["events"]["qualified"] >= 4 and ctx.tool_stats["orgs"]["qualified"] == 0
    later = names[2:]
    assert "or" not in later  # the tool that produced nothing qualified was not given more budget
    assert res.qualified >= 8 and res.rounds[0]["round"] == 0


# ── memory and learning ──────────────────────────────────────────────────────
def test_query_templates_are_learned_per_playbook(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/m.db")
    spec = _spec()
    assert query_template("edtech sponsors Bangalore", spec.topics, spec.regions) == "{topic} sponsors {place}"
    assert query_template("site:jobs.lever.co edtech", spec.topics, []) is None
    ctx = CampaignContext()
    ctx.log_query("edtech sponsors Bangalore", "search", 10, 6, 5)
    ctx.log_query("random words", "search", 10, 0, 0)
    store.record_query_stats("sponsorship", spec, ctx.query_log)
    assert store.learned_templates("sponsorship") == ["{topic} sponsors {place}"]
    assert store.learned_templates("b2b_sales") == []
    planner = SearchPlanner(_spec(topics=["fintech"]), get_playbook("sponsorship"), CampaignContext(),
                            learned_templates=store.learned_templates("sponsorship"))
    assert "fintech sponsors Bangalore" in planner.initial()


def test_known_prospects_come_back_for_a_fitting_campaign(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/m.db")
    c = Contact(id="c1", kind="org", name="DevToolCo", org_domain="devtoolco.io", country="IN",
                bio="developer tools for student developers", signals={"past_sponsorship": 1.0}, score=0.9,
                evidence=[Evidence(kind="page", text="Sponsor or partner of HackFest")])
    c.add_email(EmailCandidate(address="partnerships@devtoolco.io", confidence=0.7, method="inbox"))
    store.upsert_contacts([c])
    spec = _spec()
    got = store.known_prospects(spec, profile_for(spec, get_playbook("sponsorship")))
    assert [g.name for g in got] == ["DevToolCo"] and got[0].score is None and "memory" in got[0].sources
    other = _spec(countries=["US"], topics=["dental clinic"], pack="b2b_sales")
    assert store.known_prospects(other, profile_for(other, get_playbook("b2b_sales"))) == []


def test_productive_queries_rank_by_qualified_yield():
    ctx = CampaignContext()
    ctx.log_query("many junk", "search", 10, 9, 0)
    ctx.log_query("few good", "search", 10, 3, 3)
    assert [q.query for q in ctx.productive_queries()] == ["few good"]


# ── judge ────────────────────────────────────────────────────────────────────
def test_judge_scores_criteria_and_penalises_missing_requirements():
    from brain.judge import judge

    spec = _spec(profile=ProspectProfile(criteria=[FitCriterion(name="sponsored a hackathon", signal="past_sponsorship",
                                                                  weight=2, must=True)]))
    a = Contact(id="a", kind="org", name="A", evidence=[Evidence(kind="page", text="Sponsor or partner of HackFest")])
    b = Contact(id="b", kind="org", name="B", evidence=[Evidence(kind="page", text="developer tools")])
    for c in (a, b):
        c.add_email(EmailCandidate(address=f"info@{c.id}.com", confidence=0.7, method="inbox"))

    def transport(model, system, user, temperature, max_tokens):
        assert "CRITERIA (confirmed by the user): works on or covers any of: developer tools, edtech; sponsored a hackathon [required]" in user
        return json.dumps({"results": [
            {"id": "a", "score": 80, "criteria": {"sponsored a hackathon": 95}, "reasons": ["sponsor"], "hook_ref": 0},
            {"id": "b", "score": 80, "criteria": {"sponsored a hackathon": 0}, "reasons": ["no proof"], "hook_ref": 0}]})

    st = Settings()
    judge([a, b], spec, LLM(st, Cache(), transport=transport), playbook=get_playbook("sponsorship"))
    assert a.score > 0.6 > b.score and a.criteria["sponsored a hackathon"] == 0.95
    assert any(f.startswith("missing_required") for f in b.flags)


# ── replay and bench ─────────────────────────────────────────────────────────
def test_replay_mode_answers_only_from_the_cache():
    st = Settings()
    st.serper_api_key = "x"
    cache = Cache()
    hits = {"n": 0}

    def router(req):
        hits["n"] += 1
        return httpx.Response(200, json={"organic": [{"link": "https://a.com/", "title": "A", "snippet": "s"}]})

    live = WebSearch(st, cache, client=httpx.Client(transport=httpx.MockTransport(router)))
    assert [h.url for h in live.search("q")] == ["https://a.com/"]
    st2 = Settings()
    st2.serper_api_key, st2.replay = "x", "replay"
    rep = WebSearch(st2, cache, client=httpx.Client(transport=httpx.MockTransport(router)))
    assert [h.url for h in rep.search("q")] == ["https://a.com/"] and rep.search("other") == []
    assert hits["n"] == 1 and rep.calls == 0 and rep.replay_misses == 1
    f = Fetcher(st2, cache, client=httpx.Client(transport=httpx.MockTransport(router)))
    assert f.get("https://b.com/").error.startswith("replay") and hits["n"] == 1


def test_bench_scenarios_load_and_metrics_compute():
    from brain.bench import load_scenarios, metrics, spec_from_scenario, summarize
    from brain.pipeline import RunResult

    scs = load_scenarios()
    assert len(scs) >= 6 and {s["playbook"] for s in scs} >= {"academic", "sponsorship", "b2b_sales", "job_search",
                                                               "fundraising", "pr_media"}
    for sc in scs:
        spec = spec_from_scenario(sc)
        assert spec.topics and spec.target_count > 0 and spec.confirmed
    spec = spec_from_scenario(scs[0])
    good = Contact(name="A B", org="X", score=0.8, signals={"hiring": 1.0})
    good.add_email(EmailCandidate(address="a@x.com", confidence=0.9, method="mailto"))
    meh = Contact(name="C D", org="Y", score=0.55)
    r = RunResult(spec=spec, shortlist=[good, meh], funnel={"reachable": 2, "cost": {"search_calls": 10, "llm": {
        "paid_calls": 3, "est_cost_usd": 0.01}}})
    m = metrics(r, spec, {"good": ["A B"], "bad": ["C D"]})
    assert m["good@K"] == 1 and m["proof_rate"] == 0.5 and m["usd_per_good"] == 0.02
    assert m["label_precision@K"] == 0.5
    assert summarize([dict(m, name="x"), {"name": "y", "error": "boom"}])["failed"] == 1


# ── regressions from the first live benchmark ────────────────────────────────
def test_inbox_must_suit_the_ask_and_the_region():
    from brain.orgs import _inbox_rank

    pref = ["partnerships", "sponsorship", "marketing"]
    assert _inbox_rank("careers@testkube.io", pref) is None  # another function's inbox
    assert _inbox_rank("hello@testkube.io", pref) is not None  # generic is fine
    assert _inbox_rank("info-latam@redhat.com", pref, ["IN"]) is None  # another region's team
    assert _inbox_rank("partnerships-apac@x.com", pref, ["IN"]) == 0


def test_event_sponsors_are_named_properly_and_located():
    from brain.channels import orgs_from_event_page

    html = """<html><title>DevWeek Sponsors</title><body><section class="sponsors"><h2>Sponsors</h2>
      <a href="https://www.databricks.com/">www.databricks.com</a>
      <img src="i.png" alt="Invite MyStartupWave as Sponsor"><img src="l.png" alt="line">
    </section><p>San Francisco, California</p></body></html>"""
    orgs = orgs_from_event_page(html, "https://devweek.com/sponsors", "DevWeek Sponsors", place_terms=["india"])
    assert [o.name for o in orgs] == ["Databricks"]
    assert orgs[0].signals["past_sponsorship"] == 0.5  # proof, but of sponsoring an event elsewhere
    local = orgs_from_event_page(html.replace("San Francisco, California", "Bengaluru, India"),
                                 "https://devweek.com/sponsors", "DevWeek", place_terms=["india"])
    assert local[0].signals["past_sponsorship"] == 1.0 and "target area" in local[0].evidence[0].text


def test_function_addresses_are_never_taken_for_an_owners_address():
    from brain.orgs import _FUNCTION_IN_LOCAL

    for local in ("mediarelations", "factcheck", "pr", "investorrelations", "customersupport"):
        assert _FUNCTION_IN_LOCAL.search(local), local
    for local in ("drsharma", "anita.rao", "rahul"):
        assert not _FUNCTION_IN_LOCAL.search(local), local


def test_judge_sees_the_target_geography():
    from brain.judge import sender_brief

    assert "Target geography: Bangalore, India" in sender_brief(_spec())


def test_free_tools_run_even_when_paid_tools_already_found_enough(monkeypatch):
    import brain.explore as ex
    from brain.tools import Tool

    ran = []

    def tool(name, n, paid):
        def fn(run):
            ran.append(name)
            if paid:
                run.deps.search.calls += 1
            return [Contact(kind="person", name=f"Ann {name.title()}{i}", bio="developer tools edtech") for i in range(n)], {}
        return fn

    fake = {"news": Tool("news", "n", "search", tool("news", 30, True), True, ("recent_coverage",), ("person",)),
            "podcasts": Tool("podcasts", "p", "free", tool("podcasts", 3, False), False, ("audience",))}
    monkeypatch.setattr(ex, "TOOLS", fake)
    monkeypatch.setattr("brain.tools.TOOLS", fake)  # the router reads the registry too
    st = Settings()
    st.serper_api_key, st.use_memory = "x", False

    class D:
        settings = st
        search = WebSearch(st, Cache())
        store = None

    spec = _spec(pack="pr_media", target_kind="person")
    pb = get_playbook("pr_media")
    ex.explore(spec, pb, CampaignContext(), D(), SearchPlanner(spec, pb, CampaignContext(), need=5), budget=20, rounds=1)
    assert ran == ["podcasts", "news"]


def test_organisations_sharing_a_website_are_merged():
    from brain.resolve import resolve

    a = Contact(kind="org", name="Glean", org_domain="glean.com", sources=["jobs"], signals={"hiring": 1.0})
    b = Contact(kind="org", name="Glean Technologies", org_domain="www.glean.com", sources=["orgs"])
    out = resolve([a, b])
    assert len(out) == 1 and set(out[0].sources) == {"jobs", "orgs"} and out[0].signals["hiring"] == 1.0

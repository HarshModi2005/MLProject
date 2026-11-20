"""Discovery v3 phase 2, the precision layer: org-type gate, pre-judge, learned fit model, source priors.
No network, no API keys."""

import json
import random

from brain.discovery.hooks.news import is_byline_person
from brain.discovery.precision import fitmodel, orgtype, prejudge, priors
from brain.fit import profile_for
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence, SenderProfile, TargetPersona
from brain.playbooks import get_playbook
from brain.resolve import merge_into
from brain.store import Store


def _spec(**kw):
    base = dict(id="p2", pack="job_search", objective="ML internship at an AI startup", ask="A call?",
                persona=TargetPersona(description="Founders of AI startups in Bangalore", roles=["Founder", "CTO"],
                                      org_types=["AI startup"]),
                topics=["machine learning"], countries=["IN"], regions=["Bangalore"], target_count=5,
                target_kind="org", sender=SenderProfile(name="T"))
    base.update(kw)
    return CampaignSpec(**base)


def _org(name, domain, bio="", **kw):
    return Contact(kind="org", name=name, org=name, org_domain=domain, bio=bio, **kw)


# ── org-type gate ────────────────────────────────────────────────────────────
def test_org_types_are_classified_from_domain_name_and_text():
    agg = lambda d: d in ("justdial.com",)  # noqa: E731
    k = lambda c: orgtype.classify(c, agg)[0]  # noqa: E731
    assert k(_org("IISc", "iisc.ac.in")) == "academic"
    assert k(_org("Thapar Institute of Engineering", "thapar.edu")) == "academic"
    assert k(_org("Indian Institute of Science", None)) == "academic"
    assert k(_org("Frontiers", "frontiersin.org")) == "publisher"
    assert k(_org("Justdial", "justdial.com")) == "directory"
    assert k(_org("MeitY", "meity.gov.in")) == "government"
    assert k(_org("The Economic Times", "economictimes.indiatimes.com")) == "media"
    assert k(_org("Tech Daily News", "techdaily.in", "A digital media company covering startups")) == "media"
    assert k(_org("Pratham Foundation", "pratham.org")) == "nonprofit"
    assert k(_org("Sarvam AI", "sarvam.ai", "We build Indic language models")) == "company"
    j = Contact(kind="person", name="Anvita Manne", org="Frontiers", org_domain="frontiersin.org")
    assert k(j) == "publisher"  # people are classified by where they work


def test_gate_demotes_unwanted_types_and_drops_directories():
    agg = lambda d: d == "justdial.com"  # noqa: E731
    cs = [_org("Sarvam AI", "sarvam.ai"), _org("IISc", "iisc.ac.in"), _org("Justdial", "justdial.com")]
    kept, counts = orgtype.gate(cs, _spec(), get_playbook("job_search"), agg)
    assert [c.name for c in kept] == ["Sarvam AI", "IISc"] and counts == {"demoted_academic": 1, "dropped_directory": 1}
    assert orgtype.off_type(kept[1]) == "academic" and orgtype.off_type(kept[0]) is None
    # the same university is a proper target when the playbook or the persona asks for academics
    kept, counts = orgtype.gate([_org("IISc", "iisc.ac.in")], _spec(pack="academic"), get_playbook("academic"), agg)
    assert counts == {} and orgtype.off_type(kept[0]) is None
    sp = _spec(persona=TargetPersona(description="CSR heads at foundations and companies"))
    assert "nonprofit" in orgtype.wanted_classes(sp, get_playbook("general"))
    # re-gating replaces old tags instead of piling them up
    kept, _ = orgtype.gate(kept, _spec(), get_playbook("job_search"), agg)
    assert [f for f in kept[0].flags if f.startswith(("org_class:", "off_type:"))] == ["org_class:academic", "off_type:academic"]


def test_prescore_demotes_off_type_candidates():
    from brain.rank import prescore

    a, b = _org("Alpha ML", "alpha.ai", "machine learning startup"), _org("Beta ML", "beta.ai", "machine learning startup")
    b.flags.append("off_type:academic")
    ranked = prescore([b, a], _spec(), get_playbook("job_search"))
    assert ranked[0].name == "Alpha ML" and ranked[1].prescore < 0.5 * ranked[0].prescore


def test_bylines_that_name_organisations_are_rejected():
    assert is_byline_person("Priya Nair") and not is_byline_person("Sarvam AI")
    assert not is_byline_person("ET Bureau") and not is_byline_person("The Hindu", "The Hindu")


# ── pre-judge ────────────────────────────────────────────────────────────────
class FakeLLM:
    available = True

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def json(self, system, user, purpose, tier, max_tokens):
        self.calls.append((purpose, tier))
        if self.fail:
            raise RuntimeError("down")
        cards = json.loads(user.split("Candidates:\n", 1)[1])
        return {"results": [{"id": c["id"], "score": 10 if "bakery" in json.dumps(c).lower() else 80, "why": "fits"}
                            for c in cards]}


def test_prejudge_screens_in_batches_and_orders_the_survivors_first():
    from brain.context import CampaignContext

    cs = [_org(f"Org{i}", f"org{i}.ai", "machine learning startup", id=f"o{i}") for i in range(5)]
    cs.append(_org("Sweet Bakery", "bakery.in", "fresh bread", id="bk"))
    for i, c in enumerate(cs):
        c.prescore = 0.9 - i * 0.1
    cs[5].prescore = 0.95  # lexically top, but the wrong kind of target
    llm, stats, ctx = FakeLLM(), {}, CampaignContext()
    n = prejudge.prejudge(cs, _spec(), llm, profile_for(_spec(), get_playbook("job_search")), batch_size=4,
                          stats=stats, ctx=ctx)
    assert n == 6 and llm.calls == [("prejudge", "fast"), ("prejudge", "fast")] and stats == {"screened": 6, "rejected": 1}
    bakery = cs[5]
    assert bakery.prejudge == 0.1 and "prejudge_reject" in bakery.flags and ctx.rejected_reasons == ["Sweet Bakery: fits"]
    order = prejudge.reorder(cs)
    assert order[-1].name == "Sweet Bakery" and order[0].name == "Org0"
    assert prejudge.prejudge(cs, _spec(), llm) == 0 and len(llm.calls) == 2  # already screened: nothing re-paid


def test_prejudge_failure_rejects_nobody():
    cs = [_org("Org", "org.ai", id="o")]
    assert prejudge.prejudge(cs, _spec(), FakeLLM(fail=True)) == 0
    assert cs[0].prejudge is None and "prejudge_reject" not in cs[0].flags


def test_merging_keeps_the_best_screen():
    a = Contact(name="Priya Nair", org="X", prejudge=0.2, flags=["prejudge_reject"])
    b = Contact(name="Priya Nair", org="X", prejudge=0.8)
    m = merge_into(a, b)
    assert m.prejudge == 0.8 and "prejudge_reject" not in m.flags
    c = merge_into(Contact(name="A B", prejudge=0.9), Contact(name="A B", prejudge=0.1, flags=["prejudge_reject"]))
    assert c.prejudge == 0.9 and "prejudge_reject" not in c.flags


# ── learned fit model ────────────────────────────────────────────────────────
def _examples(n=80, seed=1):
    """Synthetic history: hiring proof decides, topic words alone do not."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        hiring = rnd.random() < 0.5
        topic = rnd.random()
        out.append({"features": {"sig:topic": round(topic, 2), "proof:hiring": 1.0 if hiring else 0.0, "kind_org": 1.0},
                    "judge_score": (0.8 if hiring else 0.3) if i % 7 else None,
                    "label": None if i % 7 else ("good" if hiring else "bad")})
    return out


def test_fit_model_learns_what_the_judge_rewards():
    model = fitmodel.train(_examples())
    assert model is not None and model["n"] == 80 and model["train_recall"] >= 0.85
    w = dict(zip(model["features"], model["weights"]))
    assert w["proof:hiring"] > 1.0 and abs(w["sig:topic"]) < w["proof:hiring"] / 2
    fm = fitmodel.FitModel(model)
    prof = profile_for(_spec(), get_playbook("job_search"))
    hiring = _org("Alpha", "alpha.ai", "machine learning", signals={"hiring": 1.0})
    talker = _org("Beta", "beta.ai", "machine learning machine learning")
    assert fm.prob(hiring, prof) > fm.prob(talker, prof) and fm.qualified(hiring, prof)


def test_fit_model_needs_enough_data_of_both_kinds():
    assert fitmodel.train(_examples(n=30)) is None
    one_sided = [dict(e, judge_score=0.9, label=None) for e in _examples()]
    assert fitmodel.train(one_sided) is None
    assert fitmodel.target({"label": "bad", "judge_score": 0.9}) == 0  # a human overrides the judge
    assert fitmodel.target({"judge_score": 0.6}) is None and fitmodel.target({"prejudge": 0.2}) == 0


def test_store_keeps_examples_labels_and_models(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'm.db'}")
    prof = profile_for(_spec(), get_playbook("job_search"))
    a = _org("Alpha", "alpha.ai", "ml", id="a", signals={"hiring": 1.0})
    a.score = 0.8
    b = _org("Beta", "beta.ai", "ml", id="b")
    b.prejudge = 0.2
    rows = fitmodel.examples_from([a, b, _org("Gamma", "g.ai", id="g")], prof)
    assert [r["contact_id"] for r in rows] == ["a", "b"] and rows[0]["features"]["proof:hiring"] == 1.0
    store.record_fit_examples("job_search", "c1", rows)
    store.record_fit_examples("job_search", "c1", rows)  # idempotent per campaign and candidate
    assert store.label_fit_examples("job_search", good=["Beta"], bad=[]) == 1
    ex = {e["contact_id"]: e for e in store.fit_examples("job_search")}
    assert len(ex) == 2 and ex["b"]["label"] == "good" and ex["a"]["judge_score"] == 0.8
    model = fitmodel.train(_examples())
    store.save_fit_model("job_search", model)
    assert store.load_fit_model("job_search")["threshold"] == model["threshold"] and store.load_fit_model("x") is None


def test_explorer_counts_qualified_with_the_learned_model(tmp_path, monkeypatch):
    from brain.context import CampaignContext
    from brain.explore import explore
    from brain.searchplan import SearchPlanner
    from brain.tools import TOOLS, Tool

    store = Store(f"sqlite:///{tmp_path / 'e.db'}")
    store.save_fit_model("job_search", fitmodel.train(_examples()))
    found = [_org("Alpha", "alpha.ai", "machine learning", signals={"hiring": 1.0}),
             _org("Beta", "beta.ai", "machine learning")]
    monkeypatch.setitem(TOOLS, "fake", Tool("fake", "fake", "free", lambda run: (list(found), {})))

    class Deps:
        def __init__(self):
            self.store, self.llm = store, None
            self.settings = type("S", (), {"use_memory": True, "learned_fit": True})()
            self.search = type("Q", (), {"calls": 0, "remaining": 10, "available": True})()

    spec, pb = _spec(), get_playbook("job_search")
    ctx = CampaignContext()
    res = explore(spec, pb, ctx, Deps(), SearchPlanner(spec, pb, ctx, None), budget=0, rounds=1, need=99, tools=["fake"])
    assert res.fit == "learned" and res.qualified == 1  # topic words alone no longer count


# ── source priors ────────────────────────────────────────────────────────────
def test_source_priors_order_paid_sources_and_feed_the_bandit(tmp_path):
    from brain.explore import _ucb

    store = Store(f"sqlite:///{tmp_path / 'p.db'}")
    for _ in range(2):
        store.record_source_stats("job_search",
                                  {"unique": {"orgs": 20, "jobs": 15}, "reachable": {"orgs": 2, "jobs": 4},
                                   "passed_judge": {"orgs": 0, "jobs": 3}},
                                  {"orgs": {"search_calls": 10}, "jobs": {"search_calls": 6}})
    p = store.source_priors("job_search")
    assert p["jobs"]["campaigns"] == 2 and p["jobs"]["passed"] == 6 and p["orgs"]["search_calls"] == 20
    assert priors.order_seed_tools(["orgs", "jobs"], p) == ["jobs", "orgs"]
    assert priors.order_seed_tools(["orgs", "jobs", "events"], p) == ["orgs", "jobs", "events"]  # events: no history
    assert _ucb({}, 2, priors.pseudo(p["jobs"])) > _ucb({}, 2, priors.pseudo(p["orgs"]))
    assert priors.pseudo({}) == {}

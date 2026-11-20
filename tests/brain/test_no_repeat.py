"""No-repeat bulk outreach: the touch ledger, series batches, standby / rejected / recycle, the send gate.
The invariant under test everywhere: the same person is never put in front of the user, or emailed, twice."""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

from brain import touch
from brain.models import CampaignSpec, Contact, Draft, EmailCandidate
from brain.pipeline import run_campaign
from brain.series import SeriesPaused, batch_spec, more, until
from brain.store import Store, TouchRow
from tests.brain.test_pipeline import ScriptedLLM, make_deps, person, spec as base_spec


def lab(i, **kw):
    """Person i at their own organisation (the per-organisation cap is not what these tests are about)."""
    return person(i, org=f"Lab {i}", domain=f"lab{i}.ac.in", **kw)


def sp(cid, n=3, series="", **kw):
    s = base_spec(n)
    s.id, s.series_id = cid, series
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def addresses(result):
    return {c.best_email.address for c in result.shortlist}


class CardLLM(ScriptedLLM):
    """Records which candidates the judge was asked about, so a test can see what was (not) paid for."""

    def __init__(self):
        super().__init__()
        self.judged = []

    def __call__(self, model, system, user, temperature, max_tokens):
        if system.startswith("You assess"):
            self.judged += [c["id"] for c in json.loads(user.split("Candidates:\n", 1)[1])]
        return super().__call__(model, system, user, temperature, max_tokens)


# ── identity and policy (pure) ───────────────────────────────────────────────
def test_email_normalisation_and_keys():
    assert touch.norm_email("A.B+news@Gmail.com") == "ab@gmail.com"
    assert touch.norm_email("x@googlemail.com") == "x@gmail.com"
    assert touch.norm_email("Prof.Rao@iitrpr.ac.in") == "prof.rao@iitrpr.ac.in"  # dots count outside Gmail
    c = lab(1)
    keys = touch.keys_for(c)
    assert "email:asha" + "b@lab1.ac.in" in keys and "openalex:a1" in keys and any(k.startswith("person:") for k in keys)


def test_loose_names_are_looser_than_entity_resolution():
    assert touch.loose_names_compatible("S. Kumar", "Sanjay Kumar")
    assert touch.loose_names_compatible("Sanjay Rao Kumar", "Sanjay Kumar")  # middle names ignored here
    assert not touch.loose_names_compatible("Sanjay Kumar", "Sunil Kumar")
    assert not touch.loose_names_compatible("Kumar", "Sanjay Kumar")


def test_policy_by_mode():
    now = datetime.utcnow()
    series, workspace = touch.Scope("s1"), touch.Scope("s1", "workspace")
    cool = touch.Scope("s1", "cooldown", 30)
    for state in ("bounced", "unsubscribed", "spam", "external"):  # dead addresses and imported lists: every mode
        assert all(touch.row_blocks(state, "s1", now, sc) == "do_not_contact" for sc in (None, series, workspace, cool))
    assert touch.row_blocks("sent", "other", now, series) == "already_contacted"  # emailed once: any series
    assert touch.row_blocks("surfaced", "other", now, series) is None  # another brief's shortlist is not our business
    assert touch.row_blocks("surfaced", "s1", now, series) == "already_surfaced"
    assert touch.row_blocks("surfaced", "other", now, workspace) == "already_surfaced"
    assert touch.row_blocks("rejected", "s1", now, series) == "rejected_earlier"
    assert touch.row_blocks("standby", "s1", now, series) == "on_standby"
    assert touch.row_blocks("surfaced", "s1", now, None) is None  # unscoped: only hard facts
    old = now - timedelta(days=31)
    assert touch.row_blocks("sent", "s1", old, cool) is None and touch.row_blocks("sent", "s1", now, cool)
    assert touch.row_blocks("rejected", "s1", old, cool) == "rejected_earlier"  # a "no" does not expire
    assert touch.row_blocks("replied", "s1", old, cool) == "already_contacted"


def test_states_never_go_backwards_and_cannot_undo_a_send():
    assert touch.advance("drafted", "surfaced") == "drafted"
    assert touch.advance("surfaced", "approved") == "approved"
    assert touch.advance("sent", "rejected") == "sent" and touch.advance("sent", "standby") == "sent"
    assert touch.advance("approved", "rejected") == "rejected"
    assert touch.advance("standby", "approved") == "standby"  # only recall changes standby
    assert touch.advance("replied", "sent") == "replied"
    assert touch.advance("surfaced", "external") == "external"


# ── the ledger in the store ──────────────────────────────────────────────────
def test_same_person_is_recognised_however_they_are_found(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    scope = touch.Scope("s1", campaign_id="a")
    first = Contact(id="c1", name="Sanjay Kumar", org="IIT Ropar", org_domain="iitrpr.ac.in",
                    emails=[EmailCandidate(address="sanjay.kumar@iitrpr.ac.in", confidence=0.95, method="mailto")])
    kept, lost = store.claim_batch("a", scope, [first])
    assert kept == [first] and not lost
    later = touch.Scope("s1", campaign_id="b")
    # the same address with a +tag, no name match needed
    tagged = Contact(id="x1", name="Someone", emails=[EmailCandidate(address="Sanjay.Kumar+cv@iitrpr.ac.in", confidence=0.9)])
    assert store.blocked_reason(tagged, later) == "already_surfaced"
    # the same person, no email at all, abbreviated name, same organisation
    initial = Contact(id="x2", name="S. Kumar", org="IIT Ropar", org_domain="iitrpr.ac.in")
    assert store.blocked_reason(initial, later) == "already_surfaced"
    # a different person with the same surname at the same organisation is NOT blocked
    other = Contact(id="x3", name="Sunil Kumar", org="IIT Ropar", org_domain="iitrpr.ac.in")
    assert store.blocked_reason(other, later) is None
    # the same name at another organisation is a different person
    elsewhere = Contact(id="x4", name="Sanjay Kumar", org="IIT Delhi", org_domain="iitd.ac.in")
    assert store.blocked_reason(elsewhere, later) is None
    # an organisation's decision-maker counts as touched once the organisation was
    org = Contact(id="o1", kind="org", name="Voxa", org="Voxa", org_domain="voxa.ai",
                  decision_maker={"name": "Anita Rao", "title": "Head of Partnerships"})
    store.claim_batch("a", scope, [org])
    person_inside = Contact(id="o2", name="Anita Rao", org="Voxa", org_domain="voxa.ai")
    assert store.blocked_reason(person_inside, later) == "already_surfaced"
    assert store.blocked_reason(Contact(id="o3", kind="org", name="Voxa Inc", org_domain="www.voxa.ai"), later)


def test_claim_is_atomic_between_concurrent_batches(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    c1, c2, c3 = lab(1), lab(2), lab(3)
    kept_a, lost_a = store.claim_batch("a", touch.Scope("s", campaign_id="a"), [c1, c2])
    kept_b, lost_b = store.claim_batch("b", touch.Scope("s", campaign_id="b"), [c2.model_copy(), c3])
    assert [c.id for c in kept_a] == [c1.id, c2.id] and not lost_a
    assert [c.id for c in kept_b] == [c3.id] and [c.id for c in lost_b] == [c2.id]
    # claiming again from the same campaign (a resumed run) is idempotent
    again, none = store.claim_batch("a", touch.Scope("s", campaign_id="a"), [c1, c2])
    assert len(again) == 2 and not none


def test_external_lists_and_suppression_block_in_every_mode(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    assert store.import_external(["Bounced@Old.org", "badcorp.com", "Jane Doe", "# comment", "", "single"]) == 3
    for sc in (None, touch.Scope("s"), touch.Scope("s", "workspace"), touch.Scope("s", "cooldown", 1)):
        assert store.blocked_reason(Contact(id="1", name="X", emails=[EmailCandidate(address="bounced@old.org", confidence=1)]), sc)
        assert store.blocked_reason(Contact(id="2", kind="org", name="Bad", org_domain="www.badcorp.com"), sc)
        assert store.blocked_reason(Contact(id="3", name="Jane Doe", org="Anywhere"), sc) == "do_not_contact"
    assert store.blocked_reason(Contact(id="4", name="Jane Roe", org="Anywhere"), touch.Scope("s")) is None


def test_modes_against_the_ledger(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    c = lab(1)
    store.claim_batch("a", touch.Scope("s1", campaign_id="a"), [c])
    same = lambda m, **k: store.blocked_reason(c, touch.Scope("s1", m, campaign_id="b", **k))  # noqa: E731
    other = lambda m, **k: store.blocked_reason(c, touch.Scope("s2", m, campaign_id="b", **k))  # noqa: E731
    assert same("series") == "already_surfaced" and other("series") is None
    assert other("workspace") == "already_surfaced"
    assert same("cooldown", cooldown_days=30) == "already_surfaced"
    with store._Session.begin() as s:  # the surfacing is 40 days old
        for r in s.query(TouchRow).all():
            r.updated_at = datetime.utcnow() - timedelta(days=40)
    assert same("cooldown", cooldown_days=30) is None
    assert store.blocked_reason(c, touch.Scope("s1", "series", campaign_id="a")) is None  # a campaign never blocks itself


def test_existing_results_become_ledger_rows_once(tmp_path):
    url = f"sqlite:///{tmp_path}/t.db"
    store = Store(url)
    c = lab(1)
    store.upsert_contacts([c])
    store.save_results("old", [c], {})
    store.set_status("old", c.id, "sent")
    with store._Session.begin() as s:  # as if the database predates the ledger
        s.query(TouchRow).delete()
    fresh = Store(url)
    assert fresh.blocked_reason(c.model_copy(update={"id": "other"}), touch.Scope("brand-new-series")) == "already_contacted"


# ── batches through the whole pipeline ───────────────────────────────────────
def test_later_batches_never_repeat_anyone_and_do_not_pay_for_them(tmp_path):
    people = [lab(i) for i in range(10)]
    llm = CardLLM()
    deps, _ = make_deps(tmp_path, people, llm, max_calls=400)
    seen_ids, seen_addr = set(), set()
    for k, cid in enumerate(("camp1", "camp2", "camp3")):
        before = len(llm.judged)
        res = run_campaign(sp(cid, 3, series="" if k == 0 else "camp1", recycle="never"), deps)
        assert len(res.shortlist) == 3
        got = addresses(res)
        assert not got & seen_addr, "the same person was offered twice"  # the invariant
        assert not {c.id for c in res.shortlist} & seen_ids
        seen_addr |= got
        seen_ids |= {c.id for c in res.shortlist}
        # people from earlier batches are dropped before the judge: it is never asked about them
        assert not set(llm.judged[before:]) & (seen_ids - {c.id for c in res.shortlist})
        if k:
            assert res.funnel["dedup"].get("already_surfaced", 0) >= 3 * k
            assert res.funnel["supply"]["fresh"] == 3 and res.funnel["supply"]["recycled"] == 0
    # a fourth batch finds the last person only, and says why it is short
    res = run_campaign(sp("camp4", 3, series="camp1", recycle="never"), deps)
    assert len(res.shortlist) == 1 and not addresses(res) & seen_addr
    assert res.funnel["supply"]["exhausted"] and "already covered" in res.funnel["coverage"]["message"]


def test_per_organisation_cap_counts_earlier_batches(tmp_path):
    people = [person(i) for i in range(6)]  # everyone at one institute
    deps, _ = make_deps(tmp_path, people, max_calls=300)
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    assert len(first.shortlist) == 3  # BRAIN_MAX_PER_ORG is 3
    second = run_campaign(sp("camp2", 3, series="camp1", recycle="never"), deps)
    assert second.shortlist == [] and second.funnel["dedup"].get("already_surfaced", 0) >= 3


def test_rejected_never_returns_and_standby_waits(tmp_path):
    people = [lab(i) for i in range(7)]
    deps, _ = make_deps(tmp_path, people, max_calls=300)
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    a, b, c = first.shortlist
    deps.store.set_status("camp1", a.id, "rejected")
    deps.store.set_status("camp1", b.id, "standby")
    second = run_campaign(sp("camp2", 3, series="camp1", recycle="never"), deps)
    got = {x.id for x in second.shortlist}
    assert a.id not in got and b.id not in got and c.id not in got and len(got) == 3
    assert second.funnel["dedup"].get("rejected_earlier", 0) >= 1 and second.funnel["dedup"].get("on_standby", 0) >= 1
    # releasing the rejection puts the person back as undecided (not straight into a new batch)
    assert deps.store.revive(a.id, "camp1")["was"] == "rejected"
    assert deps.store.result_status("camp1", a.id) == "drafted"


def three_decided(tmp_path, name):
    """5 people; batch 1 takes 3: a is rejected, b put on standby, c left undecided. Two people (d, e) are still new."""
    llm = ScriptedLLM()
    deps, _ = make_deps(tmp_path / name, [lab(i) for i in range(5)], llm, max_calls=400)
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    a, b, c = first.shortlist
    deps.store.set_status("camp1", a.id, "rejected")
    deps.store.set_status("camp1", b.id, "standby")
    return deps, llm, (a, b, c)


def test_recycling_only_when_new_matches_are_few(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    # never / ask-without-an-answer: only the two new people, and the user is told how to get more
    deps, _, (a, b, c) = three_decided(tmp_path, "a")
    never = run_campaign(sp("camp2", 4, series="camp1", recycle="never"), deps)
    assert len(never.shortlist) == 2 and not {a.id, b.id, c.id} & {x.id for x in never.shortlist}
    deps, _, _ = three_decided(tmp_path, "b")
    ask = run_campaign(sp("camp2", 4, series="camp1", recycle="ask"), deps)
    assert len(ask.shortlist) == 2 and any("--recycle auto" in w for w in ask.warnings)
    assert ask.funnel["recycle_offer"]["accepted"] is False
    # enough new matches (2 of 2 wanted): nothing is recycled even with recycle=auto
    deps, _, _ = three_decided(tmp_path, "c")
    enough = run_campaign(sp("camp2", 2, series="camp1", recycle="auto"), deps)
    assert len(enough.shortlist) == 2 and enough.funnel["supply"]["recycled"] == 0 and "recycle_offer" not in enough.funnel


def test_recycled_people_are_moved_not_rewritten_and_rejected_never_return(tmp_path):
    (tmp_path / "x").mkdir()
    deps, llm, (a, b, c) = three_decided(tmp_path, "x")
    writes = sum(s.startswith("You write") for s in llm.seen)
    asked = {}
    res = run_campaign(sp("camp2", 4, series="camp1", recycle="ask"), deps,
                       confirm_recycle=lambda info: asked.update(info) or True)
    ids = {x.id for x in res.shortlist}
    assert len(ids) == 4 and {b.id, c.id} < ids and a.id not in ids  # standby + undecided back; the rejected one never
    assert asked == {"fresh": 2, "wanted": 4, "undecided": 1, "standby": 1, "offer": 2}
    assert res.funnel["supply"]["recycled"] == 2 and res.funnel["supply"]["fresh"] == 2
    assert sorted(x.id for x in res.shortlist if "previously_shown" in x.flags) == sorted([b.id, c.id])
    # moved, not copied: the old rows are marked, the new campaign owns them with their stored drafts
    assert deps.store.result_status("camp1", b.id) == "recycled" and deps.store.result_status("camp2", b.id) == "drafted"
    assert all(res.drafts[i].body for i in ids)
    assert sum(s.startswith("You write") for s in llm.seen) == writes + 2  # only the two NEW people were written
    # the ledger follows the move: the batch now holding them is camp2, and they are still undecided
    assert {r["campaign_id"] for r in deps.store.seen("camp1") if r["contact_id"] in (b.id, c.id)} == {"camp2"}


# ── series: more / ladder / until ────────────────────────────────────────────
def test_more_runs_the_ladder_then_says_the_pool_is_used_up(tmp_path):
    people = [lab(i) for i in range(4)]
    deps, _ = make_deps(tmp_path, people, max_calls=600)
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    out = more(deps, "camp1", 3, recycle="never", sync=False)
    assert out.fresh == 1 and out.exhausted and out.series_id == "camp1"
    assert "already covered" in out.stopped and "--widen" in out.stopped
    assert not {x.best_email.address for r in out.results for x in r.shortlist} & addresses(first)
    assert len(deps.store.series_campaigns("camp1")) >= 3  # the first batch, rung 0, then the lookalike rung


def test_more_offers_earlier_people_again_only_when_allowed(tmp_path):
    people = [lab(i) for i in range(4)]
    deps, _ = make_deps(tmp_path, people, max_calls=600)
    run_campaign(sp("camp1", 3, recycle="never"), deps)
    out = more(deps, "camp1", 3, recycle="auto", sync=False)
    assert out.fresh == 1 and out.recycled == 2 and out.delivered == 3
    emails = [x.best_email.address for r in out.results for x in r.shortlist]
    assert len(emails) == len(set(emails))  # nobody twice, even across the recycle campaign
    never = more(deps, "camp1", 3, recycle="never", sync=False)
    assert never.fresh == 0 and never.recycled == 0


def test_the_ladder_widens_only_as_far_as_allowed(tmp_path):
    deps, _ = make_deps(tmp_path, [lab(0)], max_calls=100)
    base = sp("camp1", 3)
    base.expanded_terms, base.regions = ["drug discovery", "protein folding", "chemistry", "x"], ["Punjab"]
    deps.store.save_campaign(base, "done", {})
    deps.store.ensure_series(base)
    r0, r2, r3 = (batch_spec(deps.store, "camp1", 5, r) for r in (0, 2, 3))
    assert r0.topics == base.topics and r0.regions == ["Punjab"]
    assert r2.topics == base.topics + ["drug discovery", "protein folding", "chemistry"] and r2.regions == ["Punjab"]
    assert r3.regions == [] and r3.countries == base.countries and r3.series_id == "camp1" and r3.id != "camp1"


def test_until_stops_at_the_pool_and_never_repeats(tmp_path):
    people = [lab(i) for i in range(7)]
    deps, _ = make_deps(tmp_path, people, max_calls=1200)
    run_campaign(sp("camp1", 3, recycle="never"), deps)
    batches = until(deps, "camp1", 100, batch_size=3, recycle="never", sync=False)
    got = [x.best_email.address for b in batches for r in b.results for x in r.shortlist]
    first = {c["email"] for c in deps.store.results("camp1")}
    assert len(got) == len(set(got)) == 4 and not set(got) & first  # 7 people in all: 3 + 4, once each
    assert batches[-1].exhausted


def test_until_respects_the_search_budget(tmp_path):
    deps, _ = make_deps(tmp_path, [lab(i) for i in range(9)], max_calls=1200)
    run_campaign(sp("camp1", 3, recycle="never"), deps)

    class Fake:  # a batch that always succeeds and "spends" 10 search calls
        def __call__(self, spec, deps, **k):
            res = run_campaign(spec, deps, **k)
            res.funnel["cost"]["search_calls"] = 10
            return res

    batches = until(deps, "camp1", 100, batch_size=2, budget_searches=15, recycle="never", sync=False, run=Fake())
    assert len(batches) == 2 and "search calls spent" in batches[-1].stopped


# ── review decisions reach the ledger; the send gate ─────────────────────────
def legacy_tables(store):
    with store._Session.begin() as s:
        s.execute(text("CREATE TABLE emails (id TEXT, professor_id TEXT, recipient_email TEXT, status TEXT, "
                       "session_id TEXT, sent_at TIMESTAMP)"))
    store._has_legacy_emails = True


def test_review_decisions_sync_into_the_ledger(tmp_path):
    from brain.bridge import sync_outcomes

    deps, _ = make_deps(tmp_path, [lab(i) for i in range(6)], max_calls=300)
    store = deps.store
    legacy_tables(store)
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    a, b, c = first.shortlist
    with store._Session.begin() as s:
        for c_, st in ((a, "rejected"), (b, "standby"), (c, "approved")):
            s.execute(text("INSERT INTO emails VALUES (:i, :p, :e, :s, 'camp1', NULL)"),
                      {"i": "e" + c_.id, "p": c_.id, "e": c_.best_email.address, "s": st})
    stats = sync_outcomes(store)
    assert stats["rejected"] == stats["standby"] == stats["approved"] == 1
    assert [store.result_status("camp1", x.id) for x in (a, b, c)] == ["rejected", "standby", "approved"]
    assert sync_outcomes(store).get("rejected", 0) == 0  # idempotent
    second = run_campaign(sp("camp2", 3, series="camp1", recycle="never"), deps)
    assert not {a.id, b.id, c.id} & {x.id for x in second.shortlist}
    # the user recalls the standby draft in the review: it is undecided again
    with store._Session.begin() as s:
        s.execute(text("UPDATE emails SET status = 'draft' WHERE professor_id = :p"), {"p": b.id})
    assert sync_outcomes(store)["recalled"] == 1 and store.result_status("camp1", b.id) == "drafted"


def test_send_gate(tmp_path, monkeypatch):
    from brain import bridge

    deps, _ = make_deps(tmp_path, [lab(i) for i in range(4)], max_calls=300)
    store = deps.store
    legacy_tables(store)
    monkeypatch.setattr(bridge, "_store", lambda: store)
    monkeypatch.setenv("BRAIN_MIN_SEND_DELAY", "0")
    res = run_campaign(sp("camp1", 3, recycle="never"), deps)
    a, b, c = res.shortlist
    addr = lambda x: x.best_email.address  # noqa: E731
    assert bridge.send_gate(addr(a), "camp1") is None  # a normal draft may go
    assert "no valid" in bridge.send_gate("nonsense")
    store.suppress(addr(b), "asked")
    assert "do-not-contact" in bridge.send_gate(addr(b), "camp1")
    store.record_outcome("camp1", c.id, "sent")
    assert "already been emailed" in bridge.send_gate(addr(c), "camp1")
    # a bounce problem pauses the whole series and the gate holds every later send
    monkeypatch.setenv("BRAIN_BOUNCE_MIN_SENT", "2")
    store.record_outcome("camp1", a.id, "sent")
    store.record_outcome("camp1", a.id, "bounced")
    assert store.get_series("camp1")["status"] == "paused"
    assert "paused" in bridge.send_gate("someone.new@elsewhere.org", "camp1")
    with pytest.raises(SeriesPaused):
        more(deps, "camp1", 3, sync=False)
    store.update_series("camp1", status="active", pause_reason="")
    assert bridge.send_gate("someone.new@elsewhere.org", "camp1") is None
    # the daily cap is counted from the database, so it survives a restart
    monkeypatch.setenv("BRAIN_DAILY_SEND_CAP", "2")
    with store._Session.begin() as s:
        s.execute(text("INSERT INTO emails VALUES ('1','p','x@y.z','sent','camp1', :t), ('2','p','w@y.z','sent','camp1', :t)"),
                  {"t": datetime.utcnow()})
    assert "Daily send limit" in bridge.send_gate("someone.new@elsewhere.org", "camp1")


def test_export_moves_a_recycled_draft_instead_of_copying_it(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")  # the legacy review stack is not in requirements-brain.txt
    from db import database as legacy

    calls = []
    monkeypatch.setattr(legacy, "init_db", lambda: None)
    monkeypatch.setattr(legacy, "upsert_session_state", lambda *a, **k: None)
    monkeypatch.setattr(legacy, "save_professor", lambda p, session_id=None: None)
    monkeypatch.setattr(legacy, "save_draft_email", lambda d, session_id=None: calls.append(("save", d.id)))
    monkeypatch.setattr(legacy, "move_email", lambda old, new, sid: calls.append(("move", old, new)) or True)
    monkeypatch.setattr(legacy, "get_email_status", lambda eid: "draft")
    from brain.bridge import export_to_review
    from brain.models import stable_id

    deps, _ = make_deps(tmp_path, [lab(i) for i in range(4)], max_calls=400)
    run_campaign(sp("camp1", 3, recycle="never"), deps)
    out = more(deps, "camp1", 3, recycle="auto", sync=False)
    recycle_spec = out.results[-1].spec
    assert export_to_review(deps.store, recycle_spec) == 2
    moves = [c for c in calls if c[0] == "move"]
    recycled_ids = [m["contact_id"] for m in deps.store.load_campaign(recycle_spec.id)["funnel"]["recycled"]]
    assert sorted(m[1] for m in moves) == sorted(stable_id("camp1", i) for i in recycled_ids)  # from the old batch ...
    assert sorted(m[2] for m in moves) == sorted(stable_id(recycle_spec.id, i) for i in recycled_ids)  # ... to this one
    assert {m[2] for m in moves} == {d[1] for d in calls if d[0] == "save"}  # the moved row is the one saved


def test_funnel_and_listing_report_who_was_covered(tmp_path):
    deps, _ = make_deps(tmp_path, [lab(i) for i in range(5)], max_calls=300)
    run_campaign(sp("camp1", 2, recycle="never"), deps)
    second = run_campaign(sp("camp2", 2, series="camp1", recycle="never"), deps)
    assert second.funnel["supply"]["requested"] == 2 and second.funnel["supply"]["covered_skipped"] >= 2
    rows = deps.store.seen("camp1")
    assert len(rows) == 4 and {r["state"] for r in rows} == {"drafted"}
    assert [s["id"] for s in deps.store.list_series()] == ["camp1"] and deps.store.list_series()[0]["batches"] == 2
    assert deps.store.series_campaigns("camp1") == ["camp1", "camp2"] and deps.store.series_id_of("camp2") == "camp1"


# ── the command line ─────────────────────────────────────────────────────────
def test_cli_commands_on_a_series(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from brain.cli import app

    url = f"sqlite:///{tmp_path}/cli.db"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("BRAIN_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("BRAIN_WORKSPACE", "default")
    from brain import bridge

    calls = []
    monkeypatch.setattr(bridge, "legacy_set_status", lambda *a: calls.append(a) or 1)  # never touch a real review database
    deps, _ = make_deps(tmp_path, [lab(i) for i in range(5)], max_calls=300)
    deps.store.engine.dispose()
    store = Store(url)
    deps.store = store
    first = run_campaign(sp("camp1", 3, recycle="never"), deps)
    a, b, c = first.shortlist
    run = CliRunner()
    out = run.invoke(app, ["seen", "camp1"]).output
    assert "3 drafted" in out and a.name in out
    assert run.invoke(app, ["standby", a.best_email.address]).exit_code == 0 and store.result_status("camp1", a.id) == "standby"
    assert run.invoke(app, ["reject", b.id, "--campaign", "camp1"]).exit_code == 0 and store.result_status("camp1", b.id) == "rejected"
    assert "1 standby" in run.invoke(app, ["seen", "camp1"]).output
    assert run.invoke(app, ["recall", a.id]).exit_code == 0 and store.result_status("camp1", a.id) == "drafted"
    assert run.invoke(app, ["release", b.id]).exit_code == 0 and store.result_status("camp1", b.id) == "drafted"
    assert [c[2] for c in calls] == ["standby", "rejected", "draft", "draft"]  # each decision was mirrored to the review"
    assert run.invoke(app, ["recall", c.id]).exit_code != 0  # c was never on standby: refuse, say so
    assert run.invoke(app, ["series"]).exit_code == 0 and "camp1" in run.invoke(app, ["series"]).output
    f = tmp_path / "dnc.txt"
    f.write_text("# already emailed\nasha-someone@old.org\nbadcorp.com\nJane Doe\n")
    assert "3 entries" in run.invoke(app, ["exclude", str(f)]).output
    store.update_series("camp1", status="paused", pause_reason="too many bounces")
    paused = run.invoke(app, ["more", "camp1", "-n", "2"])
    assert paused.exit_code == 1 and "too many bounces" in paused.output
    assert run.invoke(app, ["series-resume", "camp1"]).exit_code == 0 and store.get_series("camp1")["status"] == "active"
    assert run.invoke(app, ["more", "nope"]).exit_code == 1
    assert run.invoke(app, ["more", "camp1", "--widen", "everywhere"]).exit_code != 0


# ── regressions found in code review ─────────────────────────────────────────
def test_resumed_series_is_not_paused_again_by_its_old_bounces(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_BOUNCE_MIN_SENT", "2")
    deps, _ = make_deps(tmp_path, [lab(i) for i in range(4)], max_calls=300)
    store = deps.store
    a, b, _c = run_campaign(sp("camp1", 3, recycle="never"), deps).shortlist
    store.record_outcome("camp1", a.id, "sent")
    store.record_outcome("camp1", b.id, "sent")
    store.record_outcome("camp1", a.id, "bounced")
    assert store.get_series("camp1")["status"] == "paused"
    store.resume_series("camp1")
    assert store.get_series("camp1")["status"] == "active" and store.check_series_health("camp1") is None
    assert store.series_health("camp1")["sent"] == 0  # counted afresh from the resume
    more(deps, "camp1", 1, recycle="never", sync=False)  # no SeriesPaused


def test_cooldown_series_can_send_after_the_cooldown(tmp_path):
    deps, _ = make_deps(tmp_path, [lab(i) for i in range(4)], max_calls=300)
    store = deps.store
    first = run_campaign(sp("camp1", 2, no_repeat="cooldown", cooldown_days=30, recycle="never"), deps)
    old, recent = first.shortlist
    for c in (old, recent):
        store.record_outcome("camp1", c.id, "sent")
    with store._Session.begin() as s:  # one was emailed 40 days ago, the other yesterday
        for r in s.query(TouchRow).filter(TouchRow.contact_id == old.id):
            r.updated_at = datetime.utcnow() - timedelta(days=40)
    second = run_campaign(sp("camp2", 2, series="camp1", recycle="never"), deps)
    got = {c.id for c in second.shortlist}
    assert old.id in got and recent.id not in got  # the expired one is offered again, the recent one is not
    assert store.address_blocked(old.best_email.address, "camp2") is None  # ... and can actually be sent to
    assert store.address_blocked(recent.best_email.address, "camp2") == "already_contacted"
    assert store.address_blocked(old.best_email.address) == "already_contacted"  # outside the series: any send blocks


def test_sync_does_not_resurrect_a_recycled_person(tmp_path):
    from brain.bridge import sync_outcomes

    deps, _ = make_deps(tmp_path, [lab(i) for i in range(4)], max_calls=300)
    store = deps.store
    legacy_tables(store)
    a, b, c = run_campaign(sp("camp1", 3, recycle="never"), deps).shortlist
    store.set_status("camp1", b.id, "standby")
    with store._Session.begin() as s:  # the review still holds the old standby row (nothing was exported yet)
        s.execute(text("INSERT INTO emails VALUES ('e', :p, 'x@y.z', 'standby', 'camp1', NULL)"), {"p": b.id})
    out = more(deps, "camp1", 3, recycle="auto", sync=False)
    assert store.result_status("camp1", b.id) == "recycled"
    sync_outcomes(store)
    assert store.result_status("camp1", b.id) == "recycled"  # not flipped back to standby
    assert out.recycled == 2


def test_a_send_recorded_from_one_series_reaches_the_ledger_after_another_series_took_the_row(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    bob = lab(1)
    store.upsert_contacts([bob])
    store.claim_batch("a1", touch.Scope("A", campaign_id="a1"), [bob])
    store.save_results("a1", [bob], {})
    store.claim_batch("b1", touch.Scope("B", campaign_id="b1"), [bob.model_copy()])  # another brief surfaced him too
    store.record_outcome("a1", bob.id, "sent")
    assert {r["state"] for r in store.seen()} == {"sent"}
    assert store.blocked_reason(bob.model_copy(), touch.Scope("C", campaign_id="c1")) == "already_contacted"


def test_importing_a_do_not_contact_list_never_downgrades_a_send(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    carol = lab(1)
    store.upsert_contacts([carol])
    store.save_results("c1", [carol], {})
    store.record_outcome("c1", carol.id, "sent")
    store.import_external([carol.best_email.address, "lab1.ac.in"])
    with store._Session() as s:
        row = s.get(TouchRow, f"email:{carol.best_email.address}")
        assert row.state == "sent" and row.campaign_id == "c1"  # still linked: a later bounce still lands
    store.record_outcome("c1", carol.id, "bounced")
    assert store.seen()[0]["state"] == "bounced"


def test_a_recycle_candidate_decided_meanwhile_is_not_offered_or_rewritten(tmp_path, monkeypatch):
    (tmp_path / "x").mkdir()
    deps, llm, (a, b, c) = three_decided(tmp_path, "x")
    real = deps.store.recycle_into

    def flaky(new_campaign, scope, rows):  # c is decided by someone else just before the move
        deps.store.set_status("camp1", c.id, "rejected")
        return real(new_campaign, scope, rows)

    monkeypatch.setattr(deps.store, "recycle_into", flaky)
    res = run_campaign(sp("camp2", 4, series="camp1", recycle="auto"), deps)
    ids = {x.id for x in res.shortlist}
    assert c.id not in ids and b.id in ids and len(ids) == 3
    assert res.funnel["supply"]["recycled"] == 1 and res.funnel["supply"]["fresh"] == 2
    assert deps.store.result_status("camp2", c.id) is None  # no stray draft or result row for the skipped one

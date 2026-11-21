"""Production fixes (round 2): confirmation is enforced, the model cannot invent facts about the sender, corrections keep the
safety steps, a required criterion the judge skipped does not pass, follow-ups do not fake a reply, and "no" means no."""

import base64
import json

import pytest

from brain import draft as dr
from brain.intake import ScriptedUI, build, finalise, ground_data, is_grounded, negations_from_text, run_intake, source_text
from brain.judge import _rating, judge
from brain.models import (CampaignSpec, Contact, Draft, EmailCandidate, Evidence, FitCriterion, ProspectProfile,
                          SenderProfile)
from brain.pipeline import run_campaign
from brain.playbooks import get_playbook
from brain.replies import classify_reply, message_text, own_text, unsubscribe_headers
from brain.spec import spec_from_fields
from brain.store import Store


class FakeLLM:
    available = True
    calls = 0

    def __init__(self, answers):
        self.answers, self.seen = answers, []

    def json(self, *, system, user, purpose, tier="fast", **kw):
        self.calls += 1
        self.seen.append((purpose, user))
        a = self.answers[purpose]
        return a(user) if callable(a) else a

    def report(self):
        return {"paid_calls": self.calls, "est_cost_usd": 0.0}


def _data(**kw):
    d = {"playbook": "b2b_sales", "playbook_confidence": 0.9, "objective": "Pilot customers for our SDK",
         "understanding": "You want app studios to pilot your SDK.", "ask": "Would you try our SDK free for two months?",
         "target_kind": "org", "persona": {"description": "Mobile app studios", "roles": ["CTO"], "org_types": ["app studio"]},
         "topics": ["mobile app development"], "locations": [], "tone": "professional",
         "sender_facts": [], "assumptions": [], "questions": []}
    d.update(kw)
    return d


# ── 1. the model may not invent facts about the sender ───────────────────────

def test_a_claim_needs_the_users_own_words():
    said = "I built Voxa, an open source speech model for 11 Indian languages. We have 40 pilot users."
    assert is_grounded("Built Voxa, an open-source speech model for 11 Indian languages", said)
    assert is_grounded("40 pilot users", said)
    assert not is_grounded("Voxa has 5 years of experience at Google", said)  # number and company are not in the request
    assert not is_grounded("Won the Smart India Hackathon", said)
    assert not is_grounded("", said)


def test_ground_data_drops_invented_facts_and_offer_claims():
    data = {"sender_facts": ["I built Voxa for 11 Indian languages", "Raised 2 million dollars"],
            "offer": {"value_prop": "free SDK trial", "proof_points": ["40 pilot users", "used by 500 clinics"],
                      "terms": "free for two months", "audience": ""}}
    dropped = ground_data(data, "I built Voxa for 11 Indian languages. Offer a free SDK trial for two months, 40 pilot users")
    assert data["sender_facts"] == ["I built Voxa for 11 Indian languages"]
    assert data["offer"]["proof_points"] == ["40 pilot users"] and data["offer"]["value_prop"] == "free SDK trial"
    assert set(dropped) == {"Raised 2 million dollars", "used by 500 clinics"}


def test_an_invented_fact_never_reaches_the_emails_context():
    llm = FakeLLM({"intake": _data(sender_facts=["Built Voxa", "Ex-Google engineer with 8 years of experience"])})
    res = run_intake("find app studios to pilot Voxa, our speech SDK", llm, ScriptedUI([]),
                     SenderProfile(name="Riya", summary="Built Voxa, a speech model."), interactive=False)
    facts = [f.text for f in res.ctx.facts]
    assert "Built Voxa" in facts and not any("Google" in f for f in facts)
    assert any("Ex-Google" in a and "don't support" in a for a in res.ctx.assumptions)  # said on the card


# ── 1. confirmation is enforced ──────────────────────────────────────────────

def test_unattended_run_stops_when_something_essential_is_missing():
    llm = FakeLLM({"intake": _data(topics=[], ask="")})
    res = run_intake("do outreach", llm, ScriptedUI([]), SenderProfile(name="Riya"), interactive=False)
    assert not res.spec.confirmed and res.needs  # no targets, no ask, nothing about the sender


def test_unattended_run_is_marked_auto_and_an_interactive_one_user():
    llm = FakeLLM({"intake": _data()})
    sender = SenderProfile(name="Riya", summary="Builds speech models.")
    assert run_intake("find app studios", llm, ScriptedUI([]), sender, interactive=False).spec.confirmed_by == "auto"
    assert run_intake("find app studios", llm, ScriptedUI(["", "yes"]), SenderProfile(name="Riya", summary="x"),
                      interactive=True).spec.confirmed_by == "user"


def test_hand_typed_fields_count_as_confirmed():
    spec = spec_from_fields(topics=["gnn"], objective="PhD", sender=SenderProfile(name="A"))
    assert spec.confirmed and spec.confirmed_by == "fields"


def test_pipeline_refuses_an_unapproved_campaign(tmp_path):
    from brain.pipeline import Deps
    from types import SimpleNamespace

    store = Store(f"sqlite:///{tmp_path / 'c.db'}")
    deps = SimpleNamespace(store=store, settings=SimpleNamespace(max_search_calls=10, use_memory=False),
                           search=SimpleNamespace(calls=0, budget=0, by_kind={}), llm=SimpleNamespace(report=lambda: {}),
                           openalex=None, fetcher=SimpleNamespace(fetched=0, paid_calls={}))
    spec = CampaignSpec(id="x", pack="general", topics=["dental clinics"], target_count=3)
    res = run_campaign(spec, deps, require_confirmed=True)
    assert res.shortlist == [] and "never approved" in res.warnings[0]
    assert store.load_campaign("x")["status"] == "needs_input"


def test_the_last_correction_is_shown_before_it_is_approved():
    llm = FakeLLM({"intake": _data(), "intake_refine": _data()})
    sender = SenderProfile(name="Riya", summary="Builds speech models.")
    answers = ["", "change 1", "change 2", "yes"]  # max_edits=2 is spent by the two corrections; the third answer approves
    res = run_intake("find app studios", llm, ScriptedUI(answers), sender, max_edits=2)
    assert res.spec.confirmed and res.spec.confirmed_by == "user"
    with pytest.raises(KeyboardInterrupt):
        run_intake("find app studios", llm, ScriptedUI(["", "change 1", "change 2"]), sender, max_edits=2)  # default: quit


# ── 2. corrections keep the first pass's guards ──────────────────────────────

def test_a_correction_keeps_negations_size_limits_and_exclusions():
    first = _data(exclude_organizations=["Rivalco"])
    refined = _data(exclude_organizations=["Rivalco"], topics=["fintech app development"])
    llm = FakeLLM({"intake": first, "intake_refine": refined})
    sender = SenderProfile(name="Riya", summary="Builds speech models.")
    prompt = "find app studios with 50 to 500 employees, not agencies, excluding Rivalco"
    res = run_intake(prompt, llm, ScriptedUI(["", "only fintech", "yes"]), sender)
    assert res.spec.topics == ["fintech app development"]  # the correction applied...
    assert "agencies" in res.spec.profile.anti_patterns  # ...and the guards from the first pass survived it
    assert (res.spec.constraints.size_min, res.spec.constraints.size_max) == (50, 500)
    assert res.ctx.is_excluded(name="rivalco")


def test_dont_exclude_is_not_an_exclusion():
    assert negations_from_text("please don't exclude agencies, and not distributors") == ["distributors"]


# ── 3. a required criterion the judge skipped does not pass ──────────────────

def _judge_llm(criteria):
    return FakeLLM({"judge": lambda user: {"results": [{"id": "a", "score": 90, "reasons": ["ok"], "criteria": criteria,
                                                         "competitor_or_directory": False, "region_ok": True}]}})


def _candidate():
    return Contact(id="a", kind="org", name="ByteStudio", evidence=[Evidence(kind="page", text="Builds apps")],
                   emails=[EmailCandidate(address="hello@byte.in", confidence=0.7, method="inbox")])


def _spec_with_required():
    return CampaignSpec(id="s", pack="general", topics=["apps"], profile=ProspectProfile(
        criteria=[FitCriterion(name="has shipped an app", signal="topic", weight=2, must=True),
                  FitCriterion(name="hiring", signal="topic", weight=1)]))


def test_rating_matches_names_loosely():
    assert _rating("has shipped an app", {"Has shipped an app.": 80}) == 0.8
    assert _rating("has shipped an app", {"has shipped an app on google play": 80}) == 0.8
    assert _rating("has shipped an app", {"hiring": 80}) is None
    assert _rating("has shipped an app", {"has shipped an app": "n/a"}) is None


def test_a_required_criterion_the_model_did_not_rate_halves_the_score():
    rated, skipped = _candidate(), _candidate()
    judge([rated], _spec_with_required(), _judge_llm({"has shipped an app": 90, "hiring": 90}))
    judge([skipped], _spec_with_required(), _judge_llm({"hiring": 90}))  # the required one is missing
    assert not any(f.startswith("required_unrated") for f in rated.flags)
    assert any(f.startswith("required_unrated:has shipped an app") for f in skipped.flags)
    assert skipped.score < 0.5 * rated.score + 0.05

    none = _candidate()
    judge([none], _spec_with_required(), _judge_llm({}))  # no criteria at all used to pass silently
    assert any(f.startswith("required_unrated") for f in none.flags)


# ── 4. no fake reply in a subject; every email says how to stop ──────────────

def test_every_follow_up_subject_is_the_original_subject():
    c = Contact(id="c", name="Asha", org="ByteStudio", evidence=[Evidence(kind="page", text="Publishes QuickClinic")],
                emails=[EmailCandidate(address="asha@byte.in", confidence=0.9, method="page")])
    first = Draft(contact_id="c", to="asha@byte.in", subject="Re: Crash reports", body="Hi Asha,\n\nHello there.\n\nBest regards,\nR")
    llm = FakeLLM({"followup": {"body": "Saw QuickClinic hiring a Flutter developer; worth a look at crash reports?"}})
    spec = CampaignSpec(id="s", sender=SenderProfile(name="R"), word_target=60)
    d = dr.followup_email(c, spec, get_playbook("b2b_sales"), llm, first)
    assert d.subject == "Crash reports"


# ── 4. a reply of "no" means no ──────────────────────────────────────────────

@pytest.mark.parametrize("text,subject,expected", [
    ("no", "", "unsubscribed"),
    ("No.\n\nOn Tue, 3 Nov 2026, Riya wrote:\n> If this isn't relevant, just reply 'no'", "Re: Hello", "unsubscribed"),
    ("Please remove me from your list.", "", "unsubscribed"),
    ("Thanks but we're not interested.", "", "unsubscribed"),
    ("Don't email me again", "", "unsubscribed"),
    ("", "unsubscribe", "unsubscribed"),
    ("I am out of office until 5 Dec, back soon. For urgent matters call Sam.", "Automatic reply: Hello", "auto_reply"),
    ("This is not a good fit for us right now.", "", "negative"),
    ("Sounds interesting, can we talk Thursday?", "", ""),
    ("Who else is using this? No rush on your side.", "", ""),
    ("Interested! No problem sending the details.", "", ""),
])
def test_reply_classification(text, subject, expected):
    assert classify_reply(text, subject, "someone@x.com") == expected


def test_delivery_notices_are_not_replies():
    assert classify_reply("Delivery has failed", "Undeliverable", "mailer-daemon@googlemail.com") == "auto_reply"


def test_quoted_text_is_not_the_reply():
    body = "Sounds great, send details.\n\n> If this isn't relevant, just reply 'no' and I won't write again.\n"
    assert own_text(body) == "Sounds great, send details."
    assert classify_reply(body) == ""  # our own opt-out line, quoted back, must not read as a stop


def test_gmail_message_text_is_decoded():
    enc = base64.urlsafe_b64encode(b"No thanks").decode().rstrip("=")
    msg = {"snippet": "snip", "payload": {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/plain", "body": {"data": enc}}]}}
    assert message_text(msg) == "No thanks" and message_text({"snippet": "snip"}) == "snip"


def test_unsubscribe_header():
    assert unsubscribe_headers("me@x.com") == {"List-Unsubscribe": "<mailto:me@x.com?subject=unsubscribe>"}
    assert unsubscribe_headers("") == {}


def test_an_unsubscribe_suppresses_the_address_everywhere(tmp_path, monkeypatch):
    from brain import bridge

    store = Store(f"sqlite:///{tmp_path / 'r.db'}")
    monkeypatch.setattr(bridge, "_store", lambda: store)
    c = Contact(id="p1", name="Asha", org="ByteStudio", emails=[EmailCandidate(address="asha@byte.in", confidence=0.9)])
    spec = CampaignSpec(id="camp", pack="general", topics=["apps"])
    store.save_campaign(spec, "done", {})
    store.upsert_contacts([c])
    store.save_results("camp", [c], {})
    assert bridge.followup_gate("asha@byte.in") is None  # a follow-up to an ordinary recipient is allowed
    out = bridge.record_reply("asha@byte.in", "unsubscribed", "asked to stop")
    assert out["matched"] == 1
    assert "do-not-contact" in bridge.followup_gate("asha@byte.in")
    assert bridge.send_gate("asha@byte.in", "camp")  # and no other campaign may write to the address either
    other = bridge.record_reply("nobody@else.com", "unsubscribed")  # not one of ours: still honoured
    assert other["matched"] == 0 and bridge.followup_gate("nobody@else.com")

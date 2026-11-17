"""Intent understanding and email generation (docs/intent-and-email-plan.md steps 1-9). Scripted LLM, no network."""

import json

import pytest

from brain import draft as dr
from brain.intake import build, signals_text, unprovable_notes, validate
from brain.models import (Ask, CampaignSpec, Contact, Draft, EmailCandidate, Evidence, SenderProfile, TargetPersona)
from brain.playbooks import get_playbook
from brain.store import Store

SENDER = SenderProfile(name="Riya Shah", headline="Founder, Voxa", summary="Builds speech recognition for Indian languages.",
                       highlights=["Voxa: open-source ASR model for 11 Indian languages", "Project: Flutter app for clinics",
                                   "Award: Smart India Hackathon winner"], skills=["speech recognition", "pytorch"])


class FakeLLM:
    available = True

    def __init__(self, answers):
        self.answers, self.calls, self.seen = answers, 0, []

    def json(self, *, system, user, purpose, tier="fast", **kw):
        self.calls += 1
        self.seen.append((purpose, user))
        a = self.answers[purpose]
        return a(user) if callable(a) else a

    def report(self):
        return {"paid_calls": self.calls, "est_cost_usd": 0.0}


def _data(**kw):
    d = {"playbook": "b2b_sales", "playbook_confidence": 0.9, "objective": "Pilot customers for our SDK",
         "ask": "Would you try our SDK free for two months?", "target_kind": "org",
         "persona": {"description": "Mobile app studios", "roles": ["CTO"], "org_types": ["app studio"]},
         "topics": ["mobile app development"], "locations": ["near Indiranagar, BLR", "remote"],
         "ask_detail": {"type": "review", "link": "https://cal.com/riya", "availability": "Tue-Thu", "deadline": None,
                        "inbox_text": "could you point me to your engineering lead?"},
         "offer": {"value_prop": "40 KB crash reporting SDK", "proof_points": ["used by 3 studios"], "terms": "free pilot"},
         "constraints": {"size_min": 10, "size_max": 200, "stages": ["seed"], "must": ["builds Flutter apps"], "avoid": ["agency"]},
         "criteria": [{"name": "publishes apps", "signal": "app_publisher", "weight": 2},
                      {"name": "magic", "signal": "telepathy", "weight": 1}],
         "anti_patterns": ["freelancer marketplace"], "language": ""}
    d.update(kw)
    return d


# ── step 1-3: intake ─────────────────────────────────────────────────────────
def test_intake_knows_every_registered_kind_of_proof():
    text = signals_text()
    assert "app_publisher" in text and "google_play" in text and "grant_awarded" in text and "nsf_awards" in text


def test_build_fills_structured_ask_offer_constraints_and_locations():
    spec = build(_data(), SENDER, campaign_id="c1", target_count=None, locations=None, playbook=None)
    assert spec.ask_detail.type == "review" and spec.ask_detail.link == "https://cal.com/riya"
    assert spec.ask_detail.inbox_text.startswith("could you point")
    assert spec.offer.value_prop == "40 KB crash reporting SDK" and spec.offer.proof_points == ["used by 3 studios"]
    assert (spec.constraints.size_min, spec.constraints.size_max, spec.constraints.remote_ok) == (10, 200, True)
    assert spec.regions == ["Indiranagar", "Bangalore"]  # "near" and "BLR" normalised, "remote" is not a place
    assert [c.signal for c in spec.profile.criteria] == ["app_publisher", "topic"]  # unknown signal -> topic
    assert "agency" in spec.profile.anti_patterns and "freelancer marketplace" in spec.profile.anti_patterns
    assert build(_data(locations=["Germany"]), SENDER, campaign_id="c2", target_count=None, locations=None,
                 playbook=None).language == "German"
    assert build(_data(locations=["Germany"], language="English"), SENDER, campaign_id="c3", target_count=None,
                 locations=None, playbook=None).language == ""


def test_validation_catches_vague_asks_sender_topics_and_unsure_playbooks():
    qs = validate(_data(ask="regarding our product"), SENDER)
    assert any("has no clear action" in q["question"] for q in qs)
    qs = validate(_data(topics=["speech recognition"], persona={"description": "anyone"}), SENDER)
    assert any("look like your own work" in q["question"] for q in qs)
    qs = validate(_data(playbook_confidence=0.4, playbook_alternatives=[{"name": "integrations", "why": "x"}]), SENDER)
    assert any("closer to" in q["question"] for q in qs)
    assert validate(_data(), SENDER) == []


def test_unprovable_criteria_are_said_not_assumed():
    from brain.models import FitCriterion, ProspectProfile

    spec = CampaignSpec(id="u", profile=ProspectProfile(criteria=[FitCriterion(name="has an office cat", signal="csr_spend")]))
    assert unprovable_notes(spec)[0].startswith("No source proves 'has an office cat'")


def test_user_profile_prefills_and_learns(tmp_path):
    from brain.intake import ScriptedUI, run_intake

    store = Store(f"sqlite:///{tmp_path / 'u.db'}")
    store.save_user_profile({"signature": "Founder, Voxa\n+91 90000 00000", "exclusions": ["rivalco.com"], "tone": "friendly"})
    llm = FakeLLM({"intake": _data(tone=None)})
    sender = SenderProfile(name="Riya Shah", summary="Builds speech models.")
    res = run_intake("find app studios", llm, ScriptedUI([]), sender, interactive=False, store=store)
    assert res.spec.tone == "friendly" and res.spec.sender.signature.startswith("Founder, Voxa")
    assert res.ctx.is_excluded(domain="rivalco.com")
    run_intake("find app studios", llm, ScriptedUI(["yes"]), SenderProfile(name="Riya Shah", summary="x"), interactive=True,
               store=store)
    assert store.user_profile()["typical_asks"] == ["Would you try our SDK free for two months?"]


# ── step 4: judge hands over sender fact and why-now ─────────────────────────
def test_judge_returns_sender_ref_and_a_why_now_grounded_in_evidence():
    from brain.judge import judge

    c = Contact(id="a", kind="org", name="ByteStudio", evidence=[Evidence(kind="page", text="Hiring now: Flutter developer (Pune)")],
                emails=[EmailCandidate(address="hello@byte.in", confidence=0.7, method="inbox")])
    d = Contact(id="b", kind="org", name="Other", evidence=[Evidence(kind="page", text="Builds apps")],
                emails=[EmailCandidate(address="hi@other.in", confidence=0.7, method="inbox")])
    llm = FakeLLM({"judge": {"results": [
        {"id": "a", "score": 80, "reasons": ["fits"], "sender_ref": 1, "why_now": "Hiring now: Flutter developer (Pune)"},
        {"id": "b", "score": 70, "reasons": ["ok"], "sender_ref": 9, "why_now": "They just raised $5M"}]}})
    spec = CampaignSpec(id="j", objective="pilots", ask="try our SDK", sender=SENDER, topics=["apps"])
    judge([c, d], spec, llm)
    assert c.sender_ref == 1 and c.why_now.startswith("Hiring now")
    assert d.sender_ref is None and d.why_now is None  # out of range / not in the evidence
    assert "[1] Project: Flutter app for clinics" in llm.seen[0][1]


# ── steps 4-6: the writer ────────────────────────────────────────────────────
def _contact(**kw):
    base = dict(id="c", kind="person", name="Asha Rao", title="CTO", org="ByteStudio", org_domain="byte.in",
                evidence=[Evidence(kind="page", text="Publishes the app QuickClinic on Google Play (medical)")],
                emails=[EmailCandidate(address="asha@byte.in", confidence=0.85, method="mailto")])
    base.update(kw)
    return Contact(**base)


def _spec(**kw):
    base = dict(id="s", pack="b2b_sales", objective="Pilot customers", ask="Would you try our SDK for two months?",
                sender=SENDER, topics=["mobile apps"], word_target=120)
    base.update(kw)
    return CampaignSpec(**base)


def test_judge_choice_leads_the_sender_facts():
    c = _contact(sender_ref=2)
    assert dr.relevant_highlights(c, _spec())[0] == "Award: Smart India Hackathon winner"


def test_situations_are_computed_in_code():
    assert dr.situation(_contact()) == "person_own_address"
    assert dr.situation(_contact(emails=[EmailCandidate(address="asha@byte.in", confidence=0.5, method="pattern_prior")])) \
        == "person_guessed_address"
    assert dr.situation(Contact(kind="org", name="ByteStudio", decision_maker={"name": "Asha Rao", "title": "CTO"})) \
        == "decision_maker_via_inbox"
    assert dr.situation(Contact(kind="org", name="ByteStudio")) == "org_inbox"
    assert dr.situation(_contact(signals={"warm_path": 0.6})) == "warm_intro"


def test_subject_choice_prefers_clean_specific_lines():
    c = _contact()
    subs = ["RE: quick question!!", "A quick question about your apps", "Crash reports for QuickClinic"]
    assert dr.choose_subject(subs, c, _spec()) == "Crash reports for QuickClinic"
    assert dr.subject_problems("FREE OFFER inside!", c, _spec())


def test_quality_checks_catch_generic_spammy_and_overclaiming_emails():
    c = _contact()
    generic = "We build software for teams. Would you like to try it for two months?"
    assert any("not personalised" in p for p in dr.quality_problems(generic, c, _spec()))
    good = "QuickClinic caught my eye. Our SDK adds crash reports in 40 KB. Would you try it on QuickClinic for two months?"
    assert dr.quality_problems(good, c, _spec()) == []
    spammy = good + " Act now! Visit https://a.io and https://bit.ly/x!"
    probs = " | ".join(dr.quality_problems(spammy, c, _spec()))
    assert "links" in probs and "shortener" in probs and "exclamation" in probs and "spam-trigger" in probs
    warm = _contact(signals={"warm_path": 0.6}, evidence=c.evidence + [Evidence(kind="warm", text="You know Dev Iyer at ByteStudio")])
    assert any("introduction" in p for p in dr.quality_problems("Dev Iyer suggested I reach out about QuickClinic.", warm, _spec()))
    assert not any("introduction" in p for p in dr.quality_problems("We both know Dev Iyer. QuickClinic looks great; would you try it?", warm, _spec()))
    de = _spec(language="German")
    assert dr.quality_problems("Wir bauen Software. Möchten Sie es testen?", c, de) == []  # English-only checks skipped


def test_assembly_adds_signature_and_an_opt_out_for_commercial_mail():
    spec = _spec(sender=SENDER.model_copy(update={"signature": "Founder, Voxa", "postal_address": "12 MG Road, Bengaluru"}))
    full = dr.assemble("Body text here.", _contact(), spec, get_playbook("b2b_sales"))
    assert full.startswith("Hi Asha,") and "Riya Shah\nFounder, Voxa" in full
    assert full.endswith("reply 'no' and I won't write again.\n12 MG Road, Bengaluru")
    assert "--" not in dr.assemble("Body.", _contact(), spec, get_playbook("academic"))
    assert dr.body_of(full) == "Body text here."


def test_draft_uses_situation_why_now_and_structured_ask():
    c = _contact(why_now="Hiring now: Flutter developer", sender_ref=1, criteria={"publishes apps": 0.9})
    from brain.models import Offer

    spec = _spec(ask_detail=Ask(type="review", text="try our SDK", link="https://cal.com/riya", inbox_text="who handles QA?"),
                 offer=Offer(value_prop="crash reporting SDK that adds 40 KB to an app"))
    body = ("QuickClinic is a well-rated medical app on Google Play, and I noticed you are hiring Flutter developers now, "
            "which usually means new releases are coming.\n\nOur SDK adds crash reporting to an app in about 40 KB, so "
            "your team sees the exact line that failed without slowing the app down for patients.\n\n"
            "Would you try it on QuickClinic for two months? You can pick a time at https://cal.com/riya.")
    llm = FakeLLM({"draft": {"subjects": ["Hi!!!", "Crash reports for QuickClinic"], "body": body}})
    d = dr.draft_email(c, spec, get_playbook("b2b_sales"), llm)
    prompt = llm.seen[0][1]
    assert "WHY NOW: Hiring now: Flutter developer" in prompt and "Link to include: https://cal.com/riya" in prompt
    assert "SITUATION: Writing to the person's own address" in prompt and "STRONGEST FIT: publishes apps" in prompt
    assert d.subject == "Crash reports for QuickClinic" and d.ok, d.issues


# ── step 7: follow-ups ───────────────────────────────────────────────────────
def test_follow_up_brings_a_new_angle_and_never_repeats():
    c = _contact(evidence=[Evidence(kind="page", text="Publishes the app QuickClinic on Google Play (medical)"),
                           Evidence(kind="page", text="Hiring now: Flutter developer (Pune)")])
    first = Draft(contact_id="c", to="asha@byte.in", subject="Crash reports for QuickClinic",
                  body="Hi Asha,\n\nQuickClinic caught my eye. Would you try our SDK?\n\nBest regards,\nRiya Shah")
    llm = FakeLLM({"followup": {"body": "Saw you are hiring a Flutter developer in Pune; crash reports help new hires ramp up on QuickClinic. Worth a try?"}})
    d = dr.followup_email(c, _spec(), get_playbook("b2b_sales"), llm, first, n=1)
    assert d.subject == "Re: Crash reports for QuickClinic" and "Hiring now: Flutter developer (Pune)" in llm.seen[0][1]
    assert d.ok, d.issues
    llm2 = FakeLLM({"followup": {"body": "QuickClinic caught my eye. Would you try our SDK? It is small and fast for QuickClinic."}})
    assert "repeats the first email" in dr.followup_email(c, _spec(), get_playbook("b2b_sales"), llm2, first, n=2).issues


# ── step 8: evaluation tools ─────────────────────────────────────────────────
def test_intake_eval_scores_each_expectation():
    from brain.intakeeval import score_case

    spec = build(_data(), SENDER, campaign_id="e", target_count=None, locations=None, playbook=None)
    case = {"prompt": "find app studios near Indiranagar, BLR, 10-200 people, not agencies; our SDK is 40 KB, used by 3 studios",
            "expect": {"playbook": ["b2b_sales"], "target_kind": "org", "regions_any": ["indiranagar"], "size": [10, 200],
                       "avoid_any": ["agency"], "criteria_signal_any": ["app_publisher"], "remote_ok": True}}
    checks = score_case(case, spec, _data(), [])
    assert all(checks.values()), checks
    assert score_case({"prompt": "find app studios", "expect": {}}, spec, _data(), [])["no_invented_numbers"] is False


def test_draft_grades_and_summary():
    from brain.drafteval import grade, summarise

    c = _contact()
    d = Draft(contact_id="c", to="asha@byte.in", subject="s", body="Hi Asha,\n\nText.\n\nBest regards,\nRiya", issues=["too short (3 words)"])
    llm = FakeLLM({"draft_grade": {"results": [{"id": "c", "specificity": 4, "relevance": 5, "ask": 4, "tone": 5, "honesty": 5,
                                                "would_send": True, "critique": "Mention the app by name."}]}})
    g = grade([(c, d)], _spec(), llm)
    s = summarise({"c": d}, g)
    assert s["pass_rate"] == 0.0 and s["avg_relevance"] == 5 and s["would_send_rate"] == 1.0 and s["top_issues"][0][0] == "too short"

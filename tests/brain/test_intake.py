"""Intake and verification, with a scripted user and a scripted model."""

import json

import pytest

from brain.cache import Cache
from brain.intake import ScriptedUI, card, run_intake, validate
from brain.llm import LLM
from brain.models import SenderProfile
from brain.settings import Settings

FIRST = {
    "playbook": "sponsorship", "understanding": "You want brands to sponsor Advitiya, IIT Ropar's tech fest.",
    "objective": "Get sponsors for Advitiya 2027", "ask": "Sponsor the fest", "target_kind": "org",
    "persona": {"description": "Marketing teams at tech brands", "roles": ["Brand Manager"]},
    "topics": ["developer tools companies", "edtech companies"], "expanded_terms": ["coding platforms"],
    "locations": ["India"], "target_count": 15, "search_queries": ["edtech companies sponsoring college fests India"],
    "preferred_inboxes": ["partnerships", "marketing"], "sender_facts": ["Advitiya is IIT Ropar's technical fest"],
    "assumptions": ["Targets are in India", "15 sponsors is enough"],
    "questions": [{"question": "How many attendees?", "why": "brands buy reach", "default": "unknown"},
                  {"question": "Any brands to avoid?", "why": "exclusions", "default": "none"}],
}


class Model:
    def __init__(self):
        self.prompts = []

    def __call__(self, model, system, user, temperature, max_tokens):
        self.prompts.append((system[:30], user))
        if system.startswith("You configure"):
            return json.dumps(FIRST)
        if system.startswith("You update"):
            out = dict(FIRST, questions=[], assumptions=["15 sponsors is enough"],
                       sender_facts=FIRST["sender_facts"] + ["5000+ attendees from 120 colleges"],
                       lookalike_queries=["companies like Devfolio sponsoring hackathons"])
            if "only fintech" in user:
                out["topics"] = ["fintech companies"]
                out["understanding"] = "You want fintech brands to sponsor Advitiya."
            return json.dumps(out)
        if system.startswith("Extract a sender profile"):
            return json.dumps({"name": "", "highlights": ["Sponsorship deck: 3 tiers from INR 50,000"]})
        raise AssertionError(system[:50])


def llm_with(model):
    st = Settings()
    st.llm_api_key = "x"
    return LLM(st, Cache(), transport=model)


def test_full_interactive_verification(tmp_path):
    (tmp_path / "past_sponsors.csv").write_text("Company,Website\nDevfolio,devfolio.co\n")
    model = Model()
    ui = ScriptedUI([
        "5000+ attendees from 120 colleges",  # Q: how many attendees?
        "",                                   # Q: brands to avoid -> default
        str(tmp_path / "past_sponsors.csv"),  # uploads
        "",                                   # accept guessed role (examples)
        "only fintech brands please",        # verification: correction
        "yes",                                # verification: confirm
    ])
    res = run_intake("Find sponsors for our college fest", llm_with(model), ui, SenderProfile(name="Harsh"))

    assert res.spec.confirmed and res.spec.pack == "sponsorship" and res.spec.topics == ["fintech companies"]
    assert res.spec.countries == ["IN"] and res.spec.target_count == 15
    assert res.spec.lookalike_queries == ["companies like Devfolio sponsoring hackathons"]
    assert res.ctx.seed_names == ["Devfolio"] and res.ctx.uploads[0]["role"] == "examples"
    sources = {f.text: f.source for f in res.ctx.facts}
    assert sources["5000+ attendees from 120 colleges"] == "you said"
    assert sources["Advitiya is IIT Ropar's technical fest"] == "you said"
    # 1 understand + 1 refine (answers + upload) + 1 refine (correction)
    assert res.llm_calls == 3
    refine_inputs = [u for s, u in model.prompts if s.startswith("You update")]
    assert "How many attendees? → 5000+ attendees from 120 colleges" in refine_inputs[0]
    assert "Any brands to avoid? → (default) none" in refine_inputs[0] and "Devfolio" in refine_inputs[0]
    assert "only fintech brands please" in refine_inputs[1]
    shown = "\n".join(ui.shown)
    assert "Assumptions" in shown and "Do you have any older lists" in shown and "Facts the emails may use" in shown


def test_quit_cancels_and_noninteractive_is_one_call():
    model = Model()
    with pytest.raises(KeyboardInterrupt):
        run_intake("Find sponsors", llm_with(model), ScriptedUI(["", "", "", "quit"]),
                   SenderProfile(summary="We run a fest"))
    model2 = Model()
    llm = llm_with(model2)
    res = run_intake("Find sponsors", llm, ScriptedUI([]), SenderProfile(summary="We run a fest"), interactive=False)
    assert res.spec.confirmed and res.llm_calls == 1


def test_validation_catches_what_a_campaign_cannot_run_without():
    qs = validate({"topics": [], "ask": ""}, SenderProfile())
    assert len(qs) == 3
    assert validate(FIRST, SenderProfile()) == []


def test_missing_sender_facts_are_asked_for_and_recorded():
    bare = dict(FIRST, sender_facts=[])

    class M(Model):
        def __call__(self, model, system, user, temperature, max_tokens):
            if system.startswith("You configure"):
                return json.dumps(bare)
            return super().__call__(model, system, user, temperature, max_tokens)

    ui = ScriptedUI(["We run a 36-hour hackathon", "", "", "", "yes"])
    res = run_intake("Find sponsors", llm_with(M()), ui, SenderProfile())
    assert any("Tell me about yourself" in q for q in ui.asked)
    assert {f.text: f.source for f in res.ctx.facts}["We run a 36-hour hackathon"] == "your answer"

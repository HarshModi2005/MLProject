"""One honest score, and a shortlist that never pads (production-hardening, items 4 and 5)."""

import json
import random

import pytest

from brain import calibrate
from brain.judge import judge
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.pipeline import coverage, run_campaign
from tests.brain.test_pipeline import SENDER, ScriptedLLM, make_deps, person, spec


# ── judge: fit and reach apart, no lexical noise ─────────────────────────────
class Fixed:
    """An LLM transport giving every candidate the same judge score."""

    def __init__(self, score):
        self.score = score

    def __call__(self, model, system, user, temperature, max_tokens):
        cards = json.loads(user.split("Candidates:\n", 1)[1])
        return json.dumps({"results": [{"id": c["id"], "score": self.score, "reasons": ["r"]} for c in cards]})


def judged(contacts, score=80, **kw):
    from brain.cache import Cache
    from brain.llm import LLM
    from brain.settings import Settings

    st = Settings()
    llm = LLM(st, Cache(), transport=Fixed(score))
    judge(contacts, CampaignSpec(id="s", topics=["x"], sender=SENDER), llm, **kw)
    return contacts


def cand(name, prescore):
    return Contact(id=name, name=name, prescore=prescore, evidence=[Evidence(kind="page", text="t")],
                   emails=[EmailCandidate(address="a@x.com", confidence=0.9, method="mailto")])


def test_lexical_prescore_no_longer_moves_the_score():
    a, b = judged([cand("A One", 0.05), cand("B Two", 0.95)])
    assert a.score == b.score and a.fit == b.fit == 0.8  # same judge verdict, same score
    old_a, old_b = judged([cand("A One", 0.05), cand("B Two", 0.95)], prescore_weight=0.2)
    assert old_a.score < old_b.score  # the previous behaviour stays available, off by default


def test_fit_is_recorded_before_reachability_is_applied():
    c = cand("A One", 0.5)
    c.emails = [EmailCandidate(address="a@x.com", confidence=0.5, method="pattern_prior")]  # a guessed address
    judged([c])
    assert c.fit == 0.8 and c.score < c.fit  # a guess costs rank, but fit stays what the judge said


# ── calibration ──────────────────────────────────────────────────────────────
def synth(n, signal, seed=3):
    """Labelled rows. `signal` says which feature carries the truth; the judge's `fit` is noisy on purpose."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        truth = rnd.random() < 0.5
        crit = min(1, max(0, (0.75 if truth else 0.25) + rnd.uniform(-0.2, 0.2))) if signal == "criteria" else rnd.random()
        fit = rnd.random() if signal == "criteria" else rnd.random()  # uninformative
        f = {k: 0.0 for k in calibrate.FEATURES}
        f.update({"fit": fit, "criteria": crit, "prejudge": rnd.random(), "reach": 1.0, "tier_attributed": 1.0})
        rows.append({"features": f, "label": "good" if truth else "bad"})
    return rows


def test_too_few_labels_keeps_the_raw_score():
    out = calibrate.train(synth(20, "criteria"))
    assert out["model"] is None and "need at least" in out["reason"]


def test_a_model_that_does_not_beat_the_raw_score_is_not_used():
    out = calibrate.train(synth(120, "none"))  # labels unrelated to any feature
    assert out["model"] is None and "does not beat" in out["reason"]
    assert out["report"]["n"] == 120


def test_a_model_that_finds_real_signal_is_kept_and_applied():
    out = calibrate.train(synth(160, "criteria"))
    assert out["model"] is not None, out
    assert out["report"]["brier_model"] < out["report"]["brier_raw"] and out["report"]["auc_model"] > 0.9
    cal = calibrate.Calibrator(out["model"])
    strong, weak = cand("S One", 0.5), cand("W Two", 0.5)
    strong.criteria, weak.criteria = {"c": 0.95}, {"c": 0.05}
    strong.fit = weak.fit = 0.5
    assert cal.prob(strong) > 0.7 > 0.3 > cal.prob(weak)
    c = judged([cand("A One", 0.5)], calibrator=cal)[0]
    assert "calibrated" in c.flags and 0 < c.score < 1 and c.score != c.fit


def test_wrong_address_counts_as_not_good_for_calibration():
    rows = synth(120, "criteria")
    for r in rows:
        if r["label"] == "bad":
            r["label"] = "wrong_address"
    assert calibrate.train(rows)["model"] is not None  # the same signal, labelled the other way round


def test_labels_become_training_rows_only_with_recorded_signals(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"candidates": [
        {"label": "good", "name": "A", "features": {"fit": 0.9}}, {"label": "bad", "name": "B", "features": {"fit": 0.2}},
        {"label": "good", "name": "C"}, {"label": "", "name": "D", "features": {"fit": 0.5}}]}))
    rows = calibrate.rows_from_sheets(tmp_path)
    assert [r["name"] for r in rows] == ["A", "B"]


# ── the shortlist never pads ─────────────────────────────────────────────────
class Mixed(ScriptedLLM):
    """Judge scores: the first two candidates 85, everyone else 55 (a weak fit)."""

    def __call__(self, model, system, user, temperature, max_tokens):
        if system.startswith("You assess"):
            cards = json.loads(user.split("Candidates:\n", 1)[1])
            return json.dumps({"results": [{"id": c["id"], "score": 85 if c["name"] in ("AshaA Rani", "AshaB Rani") else 55,
                                            "reasons": ["r"], "hook": "x", "hook_ref": 0} for c in cards]})
        return super().__call__(model, system, user, temperature, max_tokens)


def test_weak_fits_are_left_out_instead_of_padding_the_list(tmp_path):
    people = [person(i) for i in range(6)]
    deps, _ = make_deps(tmp_path, people, llm_script=Mixed())
    result = run_campaign(spec(4), deps, upto="judge")
    names = sorted(c.name for c in result.shortlist)
    assert names == ["AshaA Rani", "AshaB Rani"]  # 4 were wanted; the 55s are below the 0.6 floor
    cov = result.funnel["coverage"]
    assert cov["wanted"] == 4 and cov["delivered"] == 2 and cov["short_by"] == 2 and cov["floor"] == 0.6
    assert any("Found 2 confident prospects of the 4 wanted" in w for w in result.warnings)
    assert all(c.score >= 0.6 for c in result.shortlist)


def test_full_list_means_no_shortfall_warning():
    c = [cand(n, 0.5) for n in ("A One", "B Two")]
    cov = coverage(2, c, 0.6)
    assert cov["short_by"] == 0 and cov["message"] == ""
    assert coverage(5, c, 0.6)["short_by"] == 3 and coverage(5, [], 0.6)["delivered"] == 0

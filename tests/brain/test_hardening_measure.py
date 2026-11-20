"""Measurement you can trust: tiers, intervals, held-out scenarios, repeats, labels (production-hardening, item 2)."""

import json

import brain.bench as bench
from brain.evalkit import evaluate_labels, wilson
from brain.models import CampaignSpec, Contact, EmailCandidate


def test_wilson_interval_is_honest_about_small_samples():
    lo, hi = wilson(3, 5)
    assert lo < 0.6 < hi and hi - lo > 0.4  # 3 of 5 tells you very little
    lo, hi = wilson(60, 100)
    assert hi - lo < 0.2
    assert wilson(0, 0) is None and wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_wrong_address_is_not_a_good_result():
    rows = [{"name": "Priya Nair", "email": "priya@x.in", "score": 0.9},
            {"name": "Rahul Sharma", "email": "rahul.gupta@x.in", "score": 0.8},
            {"name": "Anil Rao", "email": "anil@x.in", "score": 0.7}]
    ev = evaluate_labels(rows, {"good": ["priya@x.in"], "wrong_address": ["rahul.gupta@x.in"], "bad": ["Anil Rao"]}, ks=(3,))
    assert ev["precision@3"] == 0.333 and ev["wrong_address"] == 1 and ev["labeled@3"] == 3


def _result(short, k=5):
    from brain.pipeline import RunResult

    return RunResult(spec=CampaignSpec(id="s", target_count=k), shortlist=short, funnel={})


def _c(name, addr, method, score, deliverability=""):
    return Contact(id=name, name=name, org_domain="x.com", score=score,
                   emails=[EmailCandidate(address=addr, confidence=0.9, method=method, deliverability=deliverability)])


def test_metrics_separate_good_from_good_and_reachable():
    short = [_c("Anita Rao", "anita.rao@x.com", "mailto", 0.9, "deliverable"),  # verified
             _c("Ben Cole", "ben.cole@x.com", "mailto", 0.8),                    # attributed
             _c("Cara Dill", "cara.dill@x.com", "pattern_global", 0.8),          # a guess
             _c("Dev Shah", "dev.patel@x.com", "text", 0.8),                     # someone else's address
             _c("Eli Park", "eli.park@x.com", "mailto", 0.4)]                    # not good
    m = bench.metrics(_result(short), CampaignSpec(id="s", target_count=5))
    assert m["good@K"] == 4 and m["good_reachable@K"] == 2 and m["good_verified@K"] == 1
    assert m["address_tiers"] == {"verified": 1, "attributed": 2, "guess": 1, "unattributed": 1}


def test_metrics_report_label_precision_with_an_interval():
    short = [_c("Anita Rao", "anita.rao@x.com", "mailto", 0.9), _c("Ben Cole", "ben.cole@x.com", "mailto", 0.8),
             _c("Cara Dill", "cara.dill@x.com", "mailto", 0.7)]
    labels = {"good": ["Anita Rao"], "bad": ["Ben Cole"], "wrong_address": ["Cara Dill"]}
    m = bench.metrics(_result(short), CampaignSpec(id="s", target_count=3), labels)
    assert m["label_precision@K"] == 0.333 and m["label_n_k"] == 3 and m["label_wrong_address"] == 1
    assert len(m["label_precision@K_ci95"]) == 2


def test_summary_pools_across_scenarios_and_reports_the_holdout_separately():
    rows = [{"name": "a", "split": "dev", "target": 5, "good@K": 5, "good_reachable@K": 5, "label_n_k": 5, "label_good_k": 4},
            {"name": "b", "split": "holdout", "target": 5, "good@K": 1, "good_reachable@K": 0, "label_n_k": 5, "label_good_k": 1}]
    s = bench.summarize(rows)
    assert s["judge_precision"] == 0.6 and s["sendable_precision"] == 0.5 and s["labeled_slots"] == "10/10"
    assert s["by_split"]["dev"]["label_precision"] == 0.8 and s["by_split"]["holdout"]["label_precision"] == 0.2
    assert s["by_split"]["holdout"]["judge_precision_ci95"][0] < 0.2 < s["by_split"]["holdout"]["judge_precision_ci95"][1]


def test_scenarios_are_split_and_the_holdout_is_never_empty_or_mislabelled():
    allsc = bench.load_scenarios()
    dev, hold = bench.load_scenarios(split="dev"), bench.load_scenarios(split="holdout")
    stress = bench.load_scenarios(split="stress")  # complicated and large cases, run on demand (docs/ISSUES.md)
    assert len(allsc) == len(dev) + len(hold) + len(stress) and len(hold) >= 6 and len(dev) >= 10
    assert {s["name"] for s in dev}.isdisjoint({s["name"] for s in hold})
    from brain.playbooks import get_playbook

    for s in hold:
        assert get_playbook(s["playbook"]).name == s["playbook"] and s["prompt"] and "HELD OUT" in s["note"]
    # held-out scenarios must test playbooks the dev set does not, or they hold nothing out
    assert {s["playbook"] for s in hold}.isdisjoint({s["playbook"] for s in dev})


def test_repeat_runs_use_fresh_caches_and_report_spread(tmp_path, monkeypatch):
    seen = []

    def fake_run(sc, **kw):
        seen.append((sc["name"], kw["cache_tag"], str(kw["out_dir"])))
        good = [3, 5, 4][len(seen) - 1]
        return {"name": sc["name"], "split": "dev", "target": 5, "good@K": good, "reachable": 4, "search_calls": 1,
                "llm_paid_calls": 1, "llm_usd": 0.0}

    monkeypatch.setattr(bench, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(bench, "run_scenario", fake_run)
    rep = bench.run_bench(["academic_gnn"], label="t", repeat=3, progress=lambda m: None)
    assert [t for _, t, _ in seen] == [".rep1", ".rep2", ".rep3"]  # never the first run's cache
    assert len({o for _, _, o in seen}) == 3
    row = rep["rows"][0]
    assert row["repeat"]["good@K"] == [3, 5, 4] and row["repeat"]["spread"] == 2
    assert rep["summary"]["repeat_noise_good@K"] == 2
    try:
        bench.run_bench(["academic_gnn"], replay="replay", repeat=2)
        raise AssertionError("repeat + replay must be refused")
    except ValueError:
        pass


def test_label_status_and_interactive_labelling(tmp_path):
    sc = [{"name": "demo", "split": "dev"}]
    assert bench.label_status(30, tmp_path, sc)[0] == {"name": "demo", "split": "dev", "candidates": 0, "labeled": 0,
                                                       "good": 0, "bad": 0, "wrong_address": 0, "needed": 30, "sheet": False}
    sheet = {"candidates": [{"label": "", "name": "A One", "email": "a@x.com", "evidence": ["e1"]},
                            {"label": "", "name": "B Two", "email": "b@x.com"},
                            {"label": "", "name": "C Three", "email": "c@x.com"},
                            {"label": "", "name": "D Four", "email": "d@x.com"}]}
    (tmp_path / "demo.json").write_text(json.dumps(sheet))
    answers = iter(["x", "g", "w", "s", "q"])  # "x" is not an option and is asked again
    n = bench.label_interactive("demo", ask=lambda prompt: next(answers), say=lambda m: None, folder=tmp_path)
    assert n == 2
    saved = json.loads((tmp_path / "demo.json").read_text())
    assert [c["label"] for c in saved["candidates"]] == ["good", "wrong_address", "", ""]
    st = bench.label_status(30, tmp_path, sc)[0]
    assert st["labeled"] == 2 and st["good"] == 1 and st["wrong_address"] == 1 and st["needed"] == 28

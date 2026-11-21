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


# ── repeats use every run; failed runs are not results; scenario files are validated ─────────────────────────

def test_repeat_headline_is_the_mean_not_the_first_run(tmp_path, monkeypatch):
    runs = iter([{"good@K": 3, "reachable": 2, "target": 5, "good_reachable@K": 1},
                 {"good@K": 5, "reachable": 4, "target": 5, "good_reachable@K": 3},
                 {"good@K": 4, "reachable": 3, "target": 5, "good_reachable@K": 2}])
    monkeypatch.setattr(bench, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(bench, "run_scenario", lambda sc, **kw: {"name": sc["name"], "split": "dev", "search_calls": 1,
                                                               "llm_paid_calls": 1, "llm_usd": 0.0, **next(runs)})
    rep = bench.run_bench(["academic_gnn"], label="m", repeat=3, progress=lambda m: None)
    row = rep["rows"][0]
    assert row["good@K"] == 4 and row["reachable"] == 3 and row["good_reachable@K"] == 2 and row["target"] == 5
    assert rep["summary"]["judge_precision"] == 0.8  # 4 of 5, not the first run's 3 of 5


def test_combining_runs_leaves_out_broken_ones_and_says_so():
    good = {"name": "x", "target": 5, "good@K": 4}
    runs = [dict(good), {"name": "x", "target": 5, "good@K": 0, "invalid": "provider_failure"},
            {"name": "x", "error": "boom"}, {**good, "good@K": 2}]
    row = bench._combine_runs(runs)
    assert row["good@K"] == 3 and row["repeat"]["runs"] == 2 and row["repeat"]["of"] == 4
    assert row["repeat"]["left_out"] == ["provider_failure", "boom"]
    only_bad = bench._combine_runs([{"name": "x", "target": 5, "good@K": 0, "invalid": "provider_failure"}] * 2)
    assert only_bad["invalid"] == "provider_failure" and "repeat" in only_bad  # nothing fine to average: still invalid


def test_a_failed_provider_makes_an_empty_run_invalid_not_a_zero():
    from brain.pipeline import RunResult

    spec = CampaignSpec(id="s", target_count=5)
    dead = RunResult(spec=spec, shortlist=[], funnel={"provider_problem": "The search provider failed 9 of 9 searches."})
    m = bench.metrics(dead, spec)
    assert m["invalid"] == "provider_failure" and "failed 9 of 9" in m["provider_problem"]
    # the same warning on a run that still found people is degraded, not invalid
    part = RunResult(spec=spec, shortlist=[_c("Anita Rao", "anita.rao@x.com", "mailto", 0.9)],
                     funnel={"provider_problem": "GitHub failed 3 of 8 requests.", "unique_people": 4})
    m2 = bench.metrics(part, spec)
    assert "invalid" not in m2 and m2["provider_problem"]
    quiet = bench.metrics(_result([_c("Anita Rao", "anita.rao@x.com", "mailto", 0.9)]), spec)
    assert "invalid" not in quiet and "provider_problem" not in quiet


def test_summary_leaves_invalid_runs_out_and_reports_the_worst_case():
    rows = [{"name": "a", "split": "dev", "target": 10, "good@K": 8},
            {"name": "b", "split": "dev", "target": 10, "good@K": 0, "invalid": "provider_failure"},
            {"name": "c", "split": "dev", "target": 10, "error": "boom"},
            {"name": "d", "split": "dev", "target": 10, "good@K": 6, "provider_problem": "GitHub failed 3 of 8."}]
    s = bench.summarize(rows)
    assert s["judge_precision"] == 0.7 and s["invalid"] == 1 and s["invalid_names"] == ["b"] and s["failed"] == 1
    assert s["judge_precision_worst_case"] == 0.35 and s["degraded_names"] == ["d"]  # 14 of 40 slots
    assert "judge_precision_worst_case" not in bench.summarize(rows[:1] + rows[3:])


def test_compare_skips_invalid_runs_and_marks_changes_inside_the_noise():
    new = {"rows": [{"name": "a", "good@K": 5, "repeat": {"spread": 2}}, {"name": "b", "good@K": 0, "invalid": "provider_failure"},
                    {"name": "c", "good@K": 9}]}
    old = {"rows": [{"name": "a", "good@K": 4}, {"name": "b", "good@K": 3}, {"name": "c", "good@K": 3}]}
    a, b, c = bench.compare(new, old)
    assert a["good@K"] == (4, 5) and a["noise"] == 2 and a["within_noise"] is True
    assert "not comparable" in b["note"]
    assert "within_noise" not in c  # no repeat data on either side: nothing to call noise


def test_scenario_files_are_validated_when_loaded(tmp_path):
    import pytest

    def write(stem, **kw):
        (tmp_path / f"{stem}.json").write_text(json.dumps({"playbook": "academic", "prompt": "find people", **kw}))

    write("ok_one")
    write("ok_two", split="holdout", playbook_ok=["pr_media"])
    assert [s["name"] for s in bench.load_scenarios(folder=tmp_path, label_dir=tmp_path)] == ["ok_one", "ok_two"]
    write("typo_split", split="holdout ")
    write("typo_playbook", playbook="accademic")
    write("empty", prompt="")
    write("dupe", name="ok_one")
    # every file is checked even when only one scenario is asked for, and all problems come back together
    with pytest.raises(ValueError) as e:
        bench.load_scenarios(["ok_one"], folder=tmp_path, label_dir=tmp_path)
    msg = str(e.value)
    for needle in ("typo_split: split 'holdout '", "typo_playbook: playbook 'accademic'", "empty: has neither",
                   "name 'ok_one' is already used by"):
        assert needle in msg
    with pytest.raises(ValueError):
        bench.load_scenarios(split="holdot")


def test_every_shipped_scenario_passes_validation():
    assert len(bench.load_scenarios()) >= 40


def test_cluster_interval_is_wider_than_wilson_when_scenarios_differ():
    from brain.evalkit import cluster_ci, wilson

    # ten scenarios of 10 slots: five perfect, five empty. Pooled 50 of 100 looks tight to Wilson but is not:
    # the result depends entirely on which scenarios you happened to pick.
    parts = [(10, 10)] * 5 + [(0, 10)] * 5
    c = cluster_ci(parts)
    w = wilson(50, 100)
    assert c["method"].startswith("bootstrap over 10") and c["ci"][1] - c["ci"][0] > 2 * (w[1] - w[0])
    assert cluster_ci(parts) == c  # seeded: the same data gives the same interval
    steady = cluster_ci([(5, 10)] * 10)
    assert steady["ci"] == [0.5, 0.5]  # no scenario-to-scenario spread: nothing to widen


def test_cluster_interval_falls_back_to_wilson_with_few_scenarios_and_says_so():
    from brain.evalkit import cluster_ci, wilson

    c = cluster_ci([(4, 5), (1, 5)])
    assert c["ci"] == wilson(5, 10) and c["method"].startswith("wilson (only 2")
    assert cluster_ci([]) is None and cluster_ci([(0, 0)]) is None


def test_summary_reports_which_interval_it_used():
    rows = [{"name": str(i), "split": "dev", "target": 10, "good@K": g, "label_n_k": 10, "label_good_k": g}
            for i, g in enumerate([9, 1, 8, 2, 7, 3])]
    s = bench.summarize(rows)
    assert s["judge_precision_ci_method"] == "bootstrap over 6 scenarios" and s["label_precision_ci_method"].startswith("bootstrap")
    assert s["judge_precision_ci95"][0] < s["judge_precision"] < s["judge_precision_ci95"][1]

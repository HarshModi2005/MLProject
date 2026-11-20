"""
Measuring intent understanding (docs/intent-and-email-plan.md step 8): `brain eval-intake`.

Each case in evals/intake/cases.json is a request as users write it, with what a careful reader would conclude.
The intake runs non-interactively (one understand call, cached), and each expectation is scored deterministically:
playbook (any acceptable), target kind, countries, regions, topic words, criteria signals, ask type, exclusions,
avoid-patterns, size limits, remote, language, a needed clarifying question, and facts that must (not) appear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CASES = Path("evals/intake/cases.json")


def score_case(case: Dict[str, Any], spec, data: Dict[str, Any], questions: List[Dict]) -> Dict[str, bool]:
    e = case.get("expect") or {}
    low = lambda xs: [str(x).lower() for x in xs]  # noqa: E731
    checks: Dict[str, bool] = {}
    if "playbook" in e:
        checks["playbook"] = spec.pack in e["playbook"]
    if "target_kind" in e:
        checks["target_kind"] = spec.target_kind == e["target_kind"]
    if "countries" in e:
        checks["countries"] = all(c in spec.countries for c in e["countries"])
    if "regions_any" in e:
        regions = " ".join(low(spec.regions))
        checks["regions"] = any(r in regions for r in e["regions_any"])
    if "topic_any" in e:
        topics = " ".join(low(spec.topics + spec.expanded_terms))
        checks["topics"] = any(t in topics for t in e["topic_any"])
    if "topic_none" in e:
        topics = " ".join(low(spec.topics))
        checks["topics_describe_targets"] = not any(t in topics for t in e["topic_none"])
    if "criteria_signal_any" in e:
        from brain.fit import profile_for
        from brain.playbooks import get_playbook

        prof = profile_for(spec, get_playbook(spec.pack))  # what discovery uses: intake criteria + playbook defaults
        checks["criteria"] = any(c.signal in e["criteria_signal_any"] for c in prof.criteria)
    if "ask_type" in e:
        checks["ask_type"] = spec.ask_detail.type in e["ask_type"]
    if "exclude_any" in e:
        ex = " ".join(low(spec.exclude_organizations))
        checks["exclusions"] = all(x in ex for x in e["exclude_any"])
    if "avoid_any" in e:
        av = " ".join(low(spec.profile.anti_patterns + spec.constraints.avoid))
        checks["avoid"] = any(x in av for x in e["avoid_any"])
    if "size" in e:
        checks["size"] = (spec.constraints.size_min, spec.constraints.size_max) == tuple(e["size"])
    if "remote_ok" in e:
        checks["remote_ok"] = spec.constraints.remote_ok == e["remote_ok"]
    if "language" in e:
        checks["language"] = (spec.language or "").lower() == e["language"].lower()
    if e.get("needs_question"):
        checks["asks_question"] = bool(questions)
    facts = " ".join(str(f) for f in data.get("sender_facts") or []) + " " + spec.offer.value_prop + " " + \
        " ".join(spec.offer.proof_points)
    if "must_fact_any" in e:
        checks["facts_kept"] = any(f.lower() in facts.lower() for f in e["must_fact_any"])
    # Invented sender facts: numbers in the facts that the request did not contain.
    import re

    nums = _numbers(facts) - _numbers(case["prompt"])
    checks["no_invented_numbers"] = not nums
    return checks


def _numbers(text: str) -> set:
    """Numbers as values: '80k' == '80,000' == '80000'; trailing punctuation ignored."""
    import re

    out = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?\s?[kKmM]?", text or ""):
        t = raw.replace(",", "").replace(" ", "").lower()
        mult = 1000 if t.endswith("k") else 1_000_000 if t.endswith("m") else 1
        try:
            out.add(round(float(t.rstrip("km")) * mult, 2))
        except ValueError:
            continue
    return out


def run(llm, only: Optional[List[str]] = None, progress=print) -> Dict[str, Any]:
    from brain.intake import ScriptedUI, _merge_questions, understand, validate, build
    from brain.models import SenderProfile

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rows, totals = [], {}
    for case in cases:
        if only and case["id"] not in only:
            continue
        sender = SenderProfile(**(case.get("sender") or {"name": "Test User"}))
        data = understand(case["prompt"], sender, llm)
        if not data:
            rows.append({"id": case["id"], "error": "no answer"})
            continue
        questions = _merge_questions(data, validate(data, sender))
        spec = build(data, sender, campaign_id=f"ie_{case['id']}"[:32], target_count=None, locations=None, playbook=None)
        checks = score_case(case, spec, data, questions)
        for k, v in checks.items():
            t = totals.setdefault(k, [0, 0])
            t[0] += int(v)
            t[1] += 1
        failed = [k for k, v in checks.items() if not v]
        rows.append({"id": case["id"], "playbook": spec.pack, "confidence": spec.playbook_confidence,
                     "alternatives": spec.playbook_alternatives, "kind": spec.target_kind,
                     "where": spec.regions + spec.countries, "topics": spec.topics[:4],
                     "ask_type": spec.ask_detail.type, "language": spec.language, "failed": failed,
                     "questions": [q["question"] for q in questions][:3]})
        progress(f"[intake] {case['id']}: {'ok' if not failed else 'FAILED ' + ', '.join(failed)}")
    all_checks = sum(t[1] for t in totals.values())
    passed = sum(t[0] for t in totals.values())
    report = {"cases": len(rows), "check_accuracy": round(passed / all_checks, 3) if all_checks else None,
              "cases_fully_right": sum(1 for r in rows if not r.get("failed") and "error" not in r),
              "by_check": {k: f"{v[0]}/{v[1]}" for k, v in totals.items()}, "rows": rows, "llm": llm.report()}
    out = Path("evals/results/intake_eval.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(out)
    return report

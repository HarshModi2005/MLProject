"""
Measuring the brain: precision of the ranked list against human labels.

A labels file is JSON: {"good": ["name or email", ...], "bad": [...], "wrong_address": [...]}.
Label a campaign's results once; every later change to discovery or ranking can be scored against it.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from brain.textutil import name_tokens


def _key(value: str) -> str:
    return value.strip().lower() if "@" in value else " ".join(name_tokens(value))


def wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% Wilson interval for k successes in n trials. With ~25 labelled slots a precision of 0.6 is really
    "somewhere between 0.41 and 0.77": report the interval, not just the point."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def evaluate_labels(results: List[Dict[str, Any]], labels: Dict[str, Any], ks=(5, 10, 25)) -> Dict[str, Any]:
    """Precision of a ranked list against labels {"good": [...], "bad": [...], "wrong_address": [...]}.
    `wrong_address` = the right person but an address that is not theirs: not good (the email would go astray)."""
    good = {_key(v) for v in labels.get("good", [])}
    bad = {_key(v) for v in labels.get("bad", [])}
    wrong = {_key(v) for v in labels.get("wrong_address", [])}
    ranked = sorted(results, key=lambda r: r.get("score") or 0, reverse=True)

    def verdict(r: Dict[str, Any]) -> str:
        keys = {_key(r.get("name") or ""), _key(r.get("email") or "")}
        return ("wrong_address" if keys & wrong else "good" if keys & good else "bad" if keys & bad else "unlabeled")

    verdicts = [verdict(r) for r in ranked]
    out: Dict[str, Any] = {"results": len(ranked), "labeled": sum(v != "unlabeled" for v in verdicts),
                           "wrong_address": verdicts.count("wrong_address")}
    for k in ks:
        top = [v for v in verdicts[:k] if v != "unlabeled"]
        out[f"precision@{k}"] = round(top.count("good") / len(top), 3) if top else None
        out[f"labeled@{k}"], out[f"good@{k}_labeled"] = len(top), top.count("good")
        out[f"precision@{k}_ci95"] = wilson(top.count("good"), len(top))
    found_good = verdicts.count("good")
    out["recall_of_labeled_good"] = round(found_good / len(good), 3) if good else None
    return out


def evaluate(results: List[Dict[str, Any]], labels_path: str, ks=(5, 10, 25)) -> Dict[str, Any]:
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    return evaluate_labels(results, labels, ks)

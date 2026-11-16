"""
Measuring the brain: precision of the ranked list against human labels.

A labels file is JSON: {"good": ["name or email", ...], "bad": ["name or email", ...]}.
Label a campaign's results once; every later change to discovery or ranking can be scored against it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from brain.textutil import name_tokens


def _key(value: str) -> str:
    return value.strip().lower() if "@" in value else " ".join(name_tokens(value))


def evaluate(results: List[Dict[str, Any]], labels_path: str, ks=(5, 10, 25)) -> Dict[str, Any]:
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    good = {_key(v) for v in labels.get("good", [])}
    bad = {_key(v) for v in labels.get("bad", [])}
    ranked = sorted(results, key=lambda r: r.get("score") or 0, reverse=True)

    def verdict(r: Dict[str, Any]) -> str:
        keys = {_key(r.get("name") or ""), _key(r.get("email") or "")}
        return "good" if keys & good else "bad" if keys & bad else "unlabeled"

    verdicts = [verdict(r) for r in ranked]
    out: Dict[str, Any] = {"results": len(ranked), "labeled": sum(v != "unlabeled" for v in verdicts)}
    for k in ks:
        top = [v for v in verdicts[:k] if v != "unlabeled"]
        out[f"precision@{k}"] = round(top.count("good") / len(top), 3) if top else None
    found_good = verdicts.count("good")
    out["recall_of_labeled_good"] = round(found_good / len(good), 3) if good else None
    return out

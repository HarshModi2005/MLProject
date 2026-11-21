"""
Measuring the brain: precision of the ranked list against human labels.

A labels file is JSON: {"good": ["name or email", ...], "bad": [...], "wrong_address": [...]}.
Label a campaign's results once; every later change to discovery or ranking can be scored against it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from brain.textutil import name_tokens


def _key(value: str) -> str:
    """Identity used to match labels to results: an address as written, a name as its tokens, and "Name | Org" (a
    person with no address, see bench.label_ident) as both, so the same name at another organisation does not match."""
    if "@" in value:
        return value.strip().lower()
    if "|" in value:
        who, org = value.split("|", 1)
        return " ".join(name_tokens(who)) + " | " + " ".join(re.findall(r"[a-z0-9]+", org.lower()))
    return " ".join(name_tokens(value))


def label_conflicts(labels: Dict[str, Any]) -> List[str]:
    """Identities labelled with more than one verdict (good and also bad, ...): the evaluation would silently pick
    one by precedence, so they are reported."""
    seen: Dict[str, set] = {}
    for verdict in ("good", "bad", "wrong_address"):
        for v in labels.get(verdict, []) or []:
            seen.setdefault(_key(v), set()).add(verdict)
    return sorted(k for k, v in seen.items() if len(v) > 1)


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


MIN_CLUSTERS = 5


def cluster_ci(parts: List[tuple], draws: int = 4000, seed: int = 7, z: float = 1.96) -> Optional[Dict[str, Any]]:
    """95% interval for a pooled precision when the slots come in clusters (one cluster per scenario: its slots share a
    query, its sources and its playbook, so they are not independent). parts = [(good, slots), ...]. Resamples whole
    scenarios with replacement. With fewer than MIN_CLUSTERS scenarios a bootstrap is not trustworthy, so it falls back
    to the Wilson interval, which is too narrow for clustered data, and says so in "method"."""
    import random

    parts = [(int(g), int(n)) for g, n in parts if n]
    slots = sum(n for _, n in parts)
    if not slots:
        return None
    good = sum(g for g, _ in parts)
    if len(parts) < MIN_CLUSTERS:
        return {"ci": wilson(good, slots, z), "method": f"wilson (only {len(parts)} scenario(s); too narrow if they differ)"}
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        pick = [parts[rng.randrange(len(parts))] for _ in parts]
        n = sum(x for _, x in pick)
        stats.append(sum(g for g, _ in pick) / n)
    stats.sort()
    lo, hi = stats[int(draws * 0.025)], stats[min(draws - 1, int(draws * 0.975))]
    return {"ci": [round(lo, 3), round(hi, 3)], "method": f"bootstrap over {len(parts)} scenarios"}


def evaluate_labels(results: List[Dict[str, Any]], labels: Dict[str, Any], ks=(5, 10, 25)) -> Dict[str, Any]:
    """Precision of a ranked list against labels {"good": [...], "bad": [...], "wrong_address": [...]}.
    `wrong_address` = the right person but an address that is not theirs: not good (the email would go astray)."""
    good = {_key(v) for v in labels.get("good", [])}
    bad = {_key(v) for v in labels.get("bad", [])}
    wrong = {_key(v) for v in labels.get("wrong_address", [])}
    ranked = sorted(results, key=lambda r: r.get("score") or 0, reverse=True)
    conflicts = label_conflicts(labels)
    if conflicts:
        logging.getLogger("brain.evalkit").warning("labels with two different verdicts (wrong_address, then good, then "
                                                   "bad wins): %s", ", ".join(conflicts))

    def verdict(r: Dict[str, Any]) -> str:
        name = _key(r.get("name") or "")
        keys = {name, _key(r.get("email") or "")}
        if r.get("org") and r.get("kind", "person") == "person":
            keys.add(_key(f"{r['name']} | {r['org']}"))
        return ("wrong_address" if keys & wrong else "good" if keys & good else "bad" if keys & bad else "unlabeled")

    verdicts = [verdict(r) for r in ranked]
    out: Dict[str, Any] = {"conflicts": conflicts, "results": len(ranked), "labeled": sum(v != "unlabeled" for v in verdicts),
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

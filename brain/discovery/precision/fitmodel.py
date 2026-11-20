"""
Learned fit model: what "qualified" means for a playbook, learned from what the judge (and humans) decided.

The hand-written rule in brain/fit.py counts a candidate as qualified when any topic word appears in its
evidence. Discovery stops on that count, so a loose rule fills the quota with junk. Every campaign already
pays the judge (and the pre-judge) to score candidates; those verdicts, plus human labels when they exist,
are stored as training examples (Store.record_fit_examples) and a small logistic regression is fitted per
playbook. Once a playbook has enough examples, the explorer uses the model instead of the rule.

Features are campaign-independent so they transfer between campaigns of one playbook:
    sig:<signal>    best support among the profile's criteria with that signal (topic, hiring, ...)
    proof:<signal>  a source attached direct proof of it
    anti, off_type, kind_org, has_site, has_email, evidence, bio
Targets, best first: a human label; else judge score (>= 0.7 good, < 0.5 bad); else a pre-judge rejection (bad).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brain.models import Contact, ProspectProfile

MIN_EXAMPLES = 50
MIN_PER_CLASS = 10
GOOD_JUDGE, BAD_JUDGE = 0.7, 0.5


def features(c: Contact, prof: ProspectProfile) -> Dict[str, float]:
    from brain.discovery.signals import proof_signals
    from brain.fit import _norm, anti_pattern_hits, criterion_support

    text = _norm(c.match_text() + " " + (c.org or "") + " " + c.name)
    out: Dict[str, float] = {}
    for cr in prof.criteria:
        key = f"sig:{cr.signal}"
        out[key] = max(out.get(key, 0.0), criterion_support(cr, c, text))
    for s in proof_signals():
        if c.signals.get(s, 0) > 0:
            out[f"proof:{s}"] = 1.0
    out["anti"] = 1.0 if anti_pattern_hits(prof, c) else 0.0
    out["off_type"] = 1.0 if any(f.startswith("off_type:") for f in c.flags) else 0.0
    out["kind_org"] = 1.0 if c.kind == "org" else 0.0
    out["has_site"] = 1.0 if c.org_domain else 0.0
    out["has_email"] = 1.0 if c.emails else 0.0
    out["evidence"] = min(1.0, len(c.evidence) / 5)
    out["bio"] = 1.0 if c.bio else 0.0
    return {k: round(v, 4) for k, v in out.items()}


def target(ex: Dict[str, Any]) -> Optional[int]:
    label = (ex.get("label") or "").lower()
    if label in ("good", "bad"):
        return 1 if label == "good" else 0
    js = ex.get("judge_score")
    if js is not None:
        return 1 if js >= GOOD_JUDGE else 0 if js < BAD_JUDGE else None
    pj = ex.get("prejudge")
    if pj is not None and pj < 0.4:
        return 0
    return None


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def train(examples: Sequence[Dict[str, Any]], l2: float = 0.5, iters: int = 600, lr: float = 0.5
          ) -> Optional[Dict[str, Any]]:
    """Fit a logistic regression on examples [{"features": {...}, "label"/"judge_score"/"prejudge"}]. Returns the
    model as a dict (JSON-serialisable), or None when there is not enough data of both kinds."""
    import numpy as np

    rows = [(ex["features"], target(ex)) for ex in examples if ex.get("features")]
    rows = [(f, y) for f, y in rows if y is not None]
    pos = sum(y for _, y in rows)
    if len(rows) < MIN_EXAMPLES or pos < MIN_PER_CLASS or len(rows) - pos < MIN_PER_CLASS:
        return None
    names = sorted({k for f, _ in rows for k in f})
    X = np.array([[f.get(k, 0.0) for k in names] for f, _ in rows], dtype=float)
    y = np.array([y for _, y in rows], dtype=float)
    w = np.zeros(len(names))
    b = 0.0
    n = len(rows)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    probs = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
    threshold = _recall_threshold(probs, y, recall=0.85)
    pred = probs >= threshold
    tp = float((pred & (y == 1)).sum())
    return {"features": names, "weights": [round(float(v), 5) for v in w], "bias": round(float(b), 5),
            "threshold": round(float(threshold), 4), "n": n, "positives": int(pos), "trained_at": time.time(),
            "train_precision": round(tp / max(1.0, float(pred.sum())), 3), "train_recall": round(tp / max(1.0, pos), 3)}


def _recall_threshold(probs, y, recall: float) -> float:
    """The highest threshold that still keeps `recall` of the good examples: discovery should not throw away good
    prospects, it should stop counting junk."""
    good = sorted((float(p) for p, t in zip(probs, y) if t == 1), reverse=True)
    if not good:
        return 0.5
    k = max(0, min(len(good) - 1, int(math.ceil(recall * len(good))) - 1))
    return min(0.5, good[k])


class FitModel:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.names: List[str] = list(data["features"])
        self.w = dict(zip(self.names, data["weights"]))
        self.b = float(data["bias"])
        self.threshold = float(data["threshold"])

    def prob(self, c: Contact, prof: ProspectProfile) -> float:
        f = features(c, prof)
        return _sigmoid(self.b + sum(self.w.get(k, 0.0) * v for k, v in f.items()))

    def qualified(self, c: Contact, prof: ProspectProfile) -> bool:
        return self.prob(c, prof) >= self.threshold

    def top_weights(self, k: int = 8) -> List[Tuple[str, float]]:
        return sorted(self.w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


def examples_from(contacts: Sequence[Contact], prof: ProspectProfile) -> List[Dict[str, Any]]:
    """Training rows for the candidates the (pre-)judge scored in this campaign."""
    out = []
    for c in contacts:
        if c.score is None and c.prejudge is None:
            continue
        out.append({"contact_id": c.id, "name": c.name, "features": features(c, prof),
                    "judge_score": c.score, "prejudge": c.prejudge})
    return out

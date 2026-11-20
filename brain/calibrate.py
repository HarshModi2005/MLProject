"""
Calibration: turn the pipeline's signals into one score that means "a human would call this a good prospect".

The judge's 0-100 and the blend of constants around it are not probabilities: 0.7 "good" is a convention nobody
checked. With human labels (`brain bench-label`) we can check it. A small logistic regression over a handful of
campaign-independent signals is fitted on the labelled candidates, pooled across playbooks (a few hundred labels are
enough for ~9 features; per-playbook models would not be).

A calibrator is only used if it BEATS the raw score under cross-validation (lower Brier score and no worse AUC). If it
does not, or there are too few labels, the raw score stays: a model fitted to 40 labels must not replace a rule.
Fit and reachability are kept apart until the end: `fit` is how right they are, the tier says how safely we can reach them.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence

from brain.models import Contact

KEY = "__calibration__"  # store.save_fit_model key
MIN_LABELS, MIN_PER_CLASS = 60, 15
FEATURES = ["fit", "criteria", "prejudge", "reach", "proof", "tier_verified", "tier_attributed", "tier_guess", "kind_org"]


def features(c: Contact) -> Dict[str, float]:
    """The signals a ranking can legitimately use, all in [0, 1]. `fit` is the judge's view; the rest are independent
    evidence (criteria support, the cheap pre-judge, how reachable, direct proof, address trust)."""
    from brain.discovery.reach.tiers import contact_tier
    from brain.fit import SIGNAL_KEYS, reach_factor

    fit = c.fit if c.fit is not None else (c.score or 0.0)
    crit = [v for v in c.criteria.values() if v is not None]
    tier = contact_tier(c)
    return {
        "fit": round(float(fit), 4),
        "criteria": round(sum(crit) / len(crit), 4) if crit else round(float(fit), 4),
        "prejudge": round(float(c.prejudge), 4) if c.prejudge is not None else round(float(fit), 4),
        "reach": round(reach_factor(c) if c.emails else 0.0, 4),
        "proof": 1.0 if any(c.signals.get(s, 0) > 0 for s in SIGNAL_KEYS if s != "publications") else 0.0,
        "tier_verified": 1.0 if tier == "verified" else 0.0,
        "tier_attributed": 1.0 if tier == "attributed" else 0.0,
        "tier_guess": 1.0 if tier == "guess" else 0.0,
        "kind_org": 1.0 if c.kind == "org" else 0.0,
    }


def row(c: Contact) -> Dict[str, Any]:
    """What is stored about a judged candidate so a later human label can become a training example."""
    best = c.best_email
    return {"id": c.id, "name": c.name, "email": best.address if best else None, "features": features(c), "score": c.score}


def _sigmoid(z):
    import numpy as np

    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _fit(X, y, l2: float = 1.0, iters: int = 800, lr: float = 0.5):
    import numpy as np

    # Plain elementwise maths rather than `X @ w`: identical result, and no dependence on the platform BLAS (Apple's
    # Accelerate emits spurious divide-by-zero warnings for matmul on perfectly finite inputs).
    w, b, n = np.zeros(X.shape[1]), 0.0, len(y)
    for _ in range(iters):
        g = _sigmoid((X * w).sum(axis=1) + b) - y
        w -= lr * ((X * g[:, None]).sum(axis=0) / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def brier(p, y) -> float:
    return float(((p - y) ** 2).mean())


def auc(p, y) -> float:
    """Probability a random good example outranks a random bad one (ties count half)."""
    pos, neg = [v for v, t in zip(p, y) if t == 1], [v for v, t in zip(p, y) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _matrix(rows: Sequence[Dict[str, Any]]):
    import numpy as np

    X = np.array([[r["features"].get(k, 0.0) for k in FEATURES] for r in rows], dtype=float)
    y = np.array([1.0 if r["label"] == "good" else 0.0 for r in rows], dtype=float)
    return X, y


def cross_validate(rows: Sequence[Dict[str, Any]], folds: int = 5, l2: float = 1.0) -> Dict[str, Any]:
    """Out-of-fold Brier and AUC of the model against the raw score (`fit` x reach, as the pipeline would rank it)."""
    import numpy as np

    X, y = _matrix(rows)
    n = len(y)
    order = np.random.RandomState(7).permutation(n)
    pred = np.zeros(n)
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        w, b = _fit(X[train], y[train], l2)
        pred[test] = _sigmoid((X[test] * w).sum(axis=1) + b)
    raw = np.array([r["features"].get("fit", 0.0) * (0.85 + 0.15 * r["features"].get("reach", 1.0)) for r in rows])
    return {"n": n, "positives": int(y.sum()), "brier_model": round(brier(pred, y), 4), "brier_raw": round(brier(raw, y), 4),
            "auc_model": round(auc(pred, y), 4), "auc_raw": round(auc(raw, y), 4)}


def train(rows: Sequence[Dict[str, Any]], l2: float = 1.0) -> Dict[str, Any]:
    """Fit on labelled rows [{"features": {...}, "label": "good"|"bad"|"wrong_address"}]. Returns
    {"model": dict|None, "report": {...}, "reason": str}. `model` is set only when it beats the raw score out of fold."""
    rows = [r for r in rows if r.get("features") and r.get("label") in ("good", "bad", "wrong_address")]
    pos = sum(1 for r in rows if r["label"] == "good")
    if len(rows) < MIN_LABELS or pos < MIN_PER_CLASS or len(rows) - pos < MIN_PER_CLASS:
        return {"model": None, "report": {"n": len(rows), "positives": pos},
                "reason": f"need at least {MIN_LABELS} labelled candidates with {MIN_PER_CLASS} good and {MIN_PER_CLASS} not "
                          f"good (have {len(rows)}, {pos} good)"}
    rep = cross_validate(rows, l2=l2)
    if not (rep["brier_model"] < rep["brier_raw"] * 0.97 and rep["auc_model"] >= rep["auc_raw"] - 0.005):
        return {"model": None, "report": rep, "reason": "the fitted model does not beat the raw score out of fold; "
                                                         "keeping the raw score"}
    X, y = _matrix(rows)
    w, b = _fit(X, y, l2)
    model = {"features": FEATURES, "weights": [round(float(v), 5) for v in w], "bias": round(float(b), 5), "n": len(rows),
             "positives": pos, "trained_at": time.time(), "cv": rep}
    return {"model": model, "report": rep, "reason": "ok"}


class Calibrator:
    def __init__(self, model: Dict[str, Any]):
        self.names: List[str] = list(model["features"])
        self.w = dict(zip(self.names, model["weights"]))
        self.b = float(model["bias"])
        self.n = int(model.get("n", 0))

    def prob(self, c: Contact) -> float:
        f = features(c)
        z = self.b + sum(self.w.get(k, 0.0) * f.get(k, 0.0) for k in self.names)
        return float(1 / (1 + math.exp(-max(-30.0, min(30.0, z)))))


def load(store) -> Optional[Calibrator]:
    try:
        model = store.load_fit_model(KEY) if store is not None else None
        return Calibrator(model) if model else None
    except Exception:
        return None


def save(store, model: Dict[str, Any]) -> None:
    store.save_fit_model(KEY, model)


def rows_from_sheets(folder="evals/labels") -> List[Dict[str, Any]]:
    """Labelled training rows from the label sheets: every candidate with a human label and the signals recorded when it
    was judged (sheets exported before signals were recorded are skipped: re-export them from a new run)."""
    import json
    from pathlib import Path

    out: List[Dict[str, Any]] = []
    for path in sorted(Path(folder).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for c in data.get("candidates") or []:
            label = str(c.get("label") or "").strip().lower()
            if label in ("good", "bad", "wrong_address") and c.get("features"):
                out.append({"scenario": path.stem, "name": c.get("name"), "features": c["features"], "label": label})
    return out

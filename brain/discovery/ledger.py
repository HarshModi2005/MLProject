"""
Cost ledger: what each source spent and produced in one campaign, so sources can be compared and the
router / bandit can learn which ones pay off.

    with ledger.measure("podcasts", deps) as entry:
        contacts, stats = tool.fn(run)
        entry["found"] = len(contacts)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


class Ledger:
    def __init__(self) -> None:
        self.entries: Dict[str, Dict[str, float]] = {}

    @contextmanager
    def measure(self, name: str, deps) -> Iterator[Dict[str, Any]]:
        search, fetcher, llm = getattr(deps, "search", None), getattr(deps, "fetcher", None), getattr(deps, "llm", None)
        s0 = getattr(search, "calls", 0)
        f0 = getattr(fetcher, "fetched", 0)
        l0 = _llm_calls(llm)
        t0 = time.time()
        entry: Dict[str, Any] = {}
        try:
            yield entry
        finally:
            row = self.entries.setdefault(name, {"runs": 0, "search_calls": 0, "fetches": 0, "llm_calls": 0,
                                                 "seconds": 0.0})
            row["runs"] += 1
            row["search_calls"] += getattr(search, "calls", 0) - s0
            row["fetches"] += getattr(fetcher, "fetched", 0) - f0
            row["llm_calls"] += _llm_calls(llm) - l0
            row["seconds"] = round(row["seconds"] + time.time() - t0, 2)
            for k, v in entry.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = row.get(k, 0) + v

    def report(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self.entries.items()}


def _llm_calls(llm) -> int:
    if llm is None:
        return 0
    try:
        return int((llm.report() or {}).get("paid_calls", 0))
    except Exception:
        return 0

"""
Per-source usage rules, enforced in one place.

    robots        "respect": page reads go through brain.fetch.Fetcher, which honours robots.txt and a
                  per-host delay. "api": a documented API meant for programs (not robots-gated).
    rate_per_min  network calls per minute to this source; cached answers never wait.
    fields        only these mapped fields are kept from a record.
"""

from __future__ import annotations

import threading
import time
from typing import Dict

from brain.discovery.contracts import SourceSpec

_last: Dict[str, float] = {}
_lock = threading.Lock()


def throttle(spec: SourceSpec, cached: bool) -> None:
    """Wait if this source's rate limit requires it. Free for cached answers and in replay."""
    rate = spec.policy.rate_per_min
    if cached or not rate:
        return
    gap = 60.0 / rate
    with _lock:
        now = time.time()
        wait = _last.get(spec.name, 0.0) + gap - now
        _last[spec.name] = max(now, _last.get(spec.name, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


def allowed_field(spec: SourceSpec, field: str) -> bool:
    return spec.policy.fields is None or field in spec.policy.fields

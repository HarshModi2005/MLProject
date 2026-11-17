"""
Address verification, pluggable. Guessed addresses are checked before they can be shortlisted.

    MXVerifier       free: the domain must accept mail (dnspython); unknown when DNS cannot be checked
    HunterVerifier   paid plugin (HUNTER_API_KEY): Hunter's email-verifier says deliverable / undeliverable
A verifier answers True (deliverable), False (undeliverable) or None (unknown). Only False removes an address.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from brain.emailfinder import has_mx
from brain.models import Contact

GUESSED = ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior")


class MXVerifier:
    name = "mx"

    def check(self, address: str) -> Optional[bool]:
        return has_mx(address.split("@")[-1])


class HunterVerifier:
    """https://hunter.io/api-documentation#email-verifier. Each check costs a Hunter credit; capped per campaign."""
    name = "hunter"

    def __init__(self, fetcher, key: str, max_checks: int = 15):
        self.fetcher, self.key, self.max_checks, self.checks = fetcher, key, max_checks, 0

    def check(self, address: str) -> Optional[bool]:
        from brain.discovery.reach.providers import _count

        if self.checks >= self.max_checks or not _count(self.fetcher, "hunter_verify", self.max_checks):
            return None
        self.checks += 1
        data = self.fetcher.get_json("https://api.hunter.io/v2/email-verifier", params={"email": address},
                                     headers={"Authorization": f"Bearer {self.key}"})
        status = ((data or {}).get("data") or {}).get("status") if isinstance(data, dict) else None
        if status in ("valid",):
            return True
        if status in ("invalid", "disposable"):
            return False
        return None


def verifiers(fetcher=None) -> List:
    out: List = [MXVerifier()]
    key = os.getenv("HUNTER_API_KEY", "")
    if key and fetcher is not None:
        out.append(HunterVerifier(fetcher, key))
    return out


def verify_guesses(contacts: Iterable[Contact], fetcher=None, limit: int = 30) -> dict:
    """Drop guessed addresses any verifier calls undeliverable; mark the ones a paid verifier confirmed."""
    stats = {"checked": 0, "removed": 0, "confirmed": 0}
    vs = verifiers(fetcher)
    for c in list(contacts)[:limit]:
        keep = []
        for e in c.emails:
            if e.method not in GUESSED:
                keep.append(e)
                continue
            stats["checked"] += 1
            verdicts = [v.check(e.address) for v in vs]
            if any(v is False for v in verdicts):
                stats["removed"] += 1
                continue
            if any(v is True for v, ver in zip(verdicts, vs) if ver.name != "mx"):
                e.confidence = max(e.confidence, 0.75)
                stats["confirmed"] += 1
            keep.append(e)
        c.emails = keep
        if not any(e.method in GUESSED for e in c.emails):
            c.flags = [f for f in c.flags if f != "email_guessed"]
    return stats

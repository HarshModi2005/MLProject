"""
Annotate professor emails with source + confidence heuristics (directory vs scrape).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import config
from orchestrator.state import ProfessorProfile


def _url_signals(profile_url: str | None, lab_url: str | None) -> tuple[str, float]:
    """Return (source_hint, url_confidence_boost)."""
    for u in (profile_url, lab_url):
        if not u:
            continue
        low = u.lower()
        if "directory" in low or "/people/" in low or "/staff/" in low or "faculty" in low:
            return "directory_page", 0.9
        if any(x in low for x in (".edu", ".ac.uk", ".ac.in")):
            return "institutional_profile", 0.75
    return "unknown", 0.55


def _email_pattern_confidence(email: str) -> float:
    e = email.lower()
    if re.match(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$", e):
        if any(
            x in e
            for x in (
                "noreply",
                "admin@",
                "office@",
                "contact@",
                "info@",
                "webmaster",
            )
        ):
            return 0.35
        if re.match(r"^[a-z]+\.[a-z]+@", e):
            return 0.65
        return 0.72
    return 0.25


def enrich_professor_email_metadata(prof: ProfessorProfile) -> ProfessorProfile:
    if not config.ENABLE_EMAIL_DISCOVERY_METADATA:
        return prof
    if not prof.email or not str(prof.email).strip():
        return prof.model_copy(
            update={
                "email_source": "missing",
                "email_confidence": 0.0,
            }
        )

    src_hint, url_boost = _url_signals(prof.profile_url, prof.lab_url)
    pat = _email_pattern_confidence(prof.email)
    conf = min(0.97, 0.5 * pat + 0.5 * url_boost)

    return prof.model_copy(
        update={
            "email_source": src_hint,
            "email_confidence": round(conf, 3),
        }
    )


def enrich_professor_emails_batch(profiles: list[ProfessorProfile]) -> list[ProfessorProfile]:
    return [enrich_professor_email_metadata(p) for p in profiles]

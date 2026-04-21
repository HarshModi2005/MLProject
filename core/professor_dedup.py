"""
Merge duplicate ProfessorProfile rows from multiple crawl sources
(email, normalized profile URL, name+institution key).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from orchestrator.state import ProfessorProfile


def _norm_email(e: str | None) -> str:
    return (e or "").strip().lower()


def _norm_url(u: str | None) -> str:
    if not u:
        return ""
    p = urlparse(u.strip().lower())
    host = (p.netloc or "").removeprefix("www.")
    path = (p.path or "").rstrip("/")
    return f"{host}{path}"


def compute_identity_key(prof: ProfessorProfile) -> str:
    e = _norm_email(prof.email)
    if e:
        return f"email:{e}"
    u = _norm_url(prof.profile_url) or _norm_url(prof.lab_url)
    if u:
        return f"url:{u}"
    inst = re.sub(r"\s+", " ", (prof.institution or "").lower())[:80]
    name = re.sub(r"\s+", " ", (prof.name or "").lower())[:80]
    return f"name:{name}|{inst}"


def _merge_profiles(primary: ProfessorProfile, other: ProfessorProfile) -> ProfessorProfile:
    """Prefer non-empty fields; union keywords; union publications by title."""
    kw = list(dict.fromkeys((primary.research_keywords or []) + (other.research_keywords or [])))
    pubs_a = primary.recent_publications or []
    pubs_b = other.recent_publications or []
    seen = {p.title.lower() for p in pubs_a if p.title}
    pubs_m = list(pubs_a)
    for p in pubs_b:
        if p.title and p.title.lower() not in seen:
            seen.add(p.title.lower())
            pubs_m.append(p)

    email = primary.email or other.email
    bio = (primary.bio_snippet or "") if len(primary.bio_snippet or "") >= len(other.bio_snippet or "") else (
        other.bio_snippet or ""
    )
    dept = primary.department or other.department
    lab = primary.lab_url or other.lab_url
    purl = primary.profile_url or other.profile_url
    raw = (primary.raw_page_text or "") if len(primary.raw_page_text or "") >= len(other.raw_page_text or "") else (
        other.raw_page_text or ""
    )

    acc = primary.accepting_students
    if acc is None:
        acc = other.accepting_students

    return primary.model_copy(
        update={
            "research_keywords": kw[:40],
            "recent_publications": pubs_m[:25],
            "email": email,
            "bio_snippet": bio or None,
            "department": dept,
            "lab_url": lab,
            "profile_url": purl,
            "raw_page_text": raw or None,
            "accepting_students": acc,
            "institution": primary.institution or other.institution or "",
        }
    )


def dedupe_professor_profiles(profiles: list[ProfessorProfile]) -> list[ProfessorProfile]:
    if not profiles:
        return []
    buckets: dict[str, ProfessorProfile] = {}
    order: list[str] = []
    for p in profiles:
        key = p.identity_key or compute_identity_key(p)
        if key not in buckets:
            buckets[key] = p.model_copy(update={"identity_key": key})
            order.append(key)
        else:
            buckets[key] = _merge_profiles(buckets[key], p)
            buckets[key] = buckets[key].model_copy(update={"identity_key": key})
    return [buckets[k] for k in order]

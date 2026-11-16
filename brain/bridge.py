"""Hands finished drafts to the existing review UI by writing them into the legacy tables."""

from __future__ import annotations

from typing import List

from brain.models import CampaignSpec, stable_id
from brain.store import Store


def export_to_review(store: Store, spec: CampaignSpec, include_flagged: bool = False) -> int:
    """Copies drafts into `professors` / `emails` so `python main.py review` shows them. Returns count."""
    from db import database as legacy
    from orchestrator.state import DraftEmail, EmailStatus, ProfessorProfile, Publication, UserIntent

    legacy.init_db()
    s = spec.sender
    legacy.upsert_session_state(spec.id, UserIntent(
        user_name=s.name or None, user_profile_summary=s.summary or None, cv_path=s.attachment_path,
        research_domains=spec.topics, countries=spec.countries, timeline=spec.timeline, confirmed=True,
    ))
    n = 0
    for r in store.results(spec.id):
        if r["status"] not in ("drafted", "needs_review") or not r["body"] or not r["email"]:
            continue
        if r["status"] == "needs_review" and not include_flagged:
            continue
        c = r["contact"]
        pubs: List[Publication] = [Publication(title=e.text, year=e.year, url=e.url)
                                   for e in c.evidence if e.kind == "publication"][:5]
        legacy.save_professor(ProfessorProfile(
            id=c.id, name=c.name, email=r["email"], institution=c.org or "", profile_url=(c.urls or [None])[0],
            research_keywords=c.topics, recent_publications=pubs, bio_snippet=c.bio,
            relevance_score=r["score"] or 0.0,
        ), session_id=spec.id)
        notes = "; ".join(r["reasons"])
        if r["issues"]:
            notes += " | CHECK: " + "; ".join(r["issues"])
        legacy.save_draft_email(DraftEmail(
            id=stable_id(spec.id, c.id), professor_id=c.id, recipient_name=c.name, recipient_email=r["email"],
            subject=r["subject"] or "", body=r["body"], personalization_notes=notes, status=EmailStatus.DRAFT,
        ), session_id=spec.id)
        n += 1
    return n

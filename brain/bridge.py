"""Hands finished drafts to the existing review UI by writing them into the legacy tables."""

from __future__ import annotations

import logging
from typing import List

from brain.models import CampaignSpec, stable_id
from brain.store import Store

log = logging.getLogger("brain.bridge")


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
        why = store.blocked_reason(c)  # discovery checked this too, but the do-not-contact list may have grown since
        if why:
            log.warning("not exporting %s: %s", c.name, why)
            continue
        pubs: List[Publication] = [Publication(title=e.text, year=e.year, url=e.url)
                                   for e in c.evidence if e.kind == "publication"][:5]
        legacy.save_professor(ProfessorProfile(
            id=c.id, name=c.name, email=r["email"], institution=c.org or "", profile_url=(c.urls or [None])[0],
            research_keywords=c.topics, recent_publications=pubs, bio_snippet=c.bio,
            relevance_score=r["score"] or 0.0,
        ), session_id=spec.id)
        notes = "; ".join(r["reasons"])
        from brain.discovery.reach.tiers import contact_tier

        notes += f" | ADDRESS: {contact_tier(c)}"  # verified | attributed | inbox | guess: how far to trust the recipient
        if r["issues"]:
            notes += " | CHECK: " + "; ".join(r["issues"])
        legacy.save_draft_email(DraftEmail(
            id=stable_id(spec.id, c.id), professor_id=c.id, recipient_name=c.name, recipient_email=r["email"],
            subject=r["subject"] or "", body=r["body"], personalization_notes=notes, status=EmailStatus.DRAFT,
        ), session_id=spec.id)
        n += 1
    return n


def sync_outcomes(store: Store) -> dict:
    """Pull what happened to sent emails out of the legacy review/send tables (status sent / replied / bounced, kept up
    to date by the Gmail sender and the reply monitor) into the brain's outcome log, where it teaches the engine:
    bounces suppress addresses and demote bad email conventions, replies label good prospects. Safe to run repeatedly."""
    from sqlalchemy import text

    stats = {"sent": 0, "replied": 0, "bounced": 0, "unmatched": 0}
    if not getattr(store, "_has_legacy_emails", False):
        return stats
    with store._Session() as s:
        rows = s.execute(text("SELECT lower(recipient_email), status FROM emails WHERE status IN "
                              "('sent','replied','followed_up','bounced') AND recipient_email IS NOT NULL")).fetchall()
    kinds = {"sent": "sent", "followed_up": "sent", "replied": "replied", "bounced": "bounced"}
    for address, status in rows:
        matches = store.results_by_email(address)
        if not matches:
            stats["unmatched"] += 1
            continue
        for campaign_id, contact_id in matches:
            kind = kinds[status]
            if kind != "sent" and not store.has_outcome(campaign_id, contact_id, "sent"):
                store.record_outcome(campaign_id, contact_id, "sent")  # a reply or bounce implies it was sent
                stats["sent"] += 1
            if not store.has_outcome(campaign_id, contact_id, kind):
                store.record_outcome(campaign_id, contact_id, kind)
                stats[kind] += 1
    return stats

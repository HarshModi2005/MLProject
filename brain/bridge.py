"""Hands finished drafts to the existing review UI by writing them into the legacy tables, and brings what the user
decided there (approved, rejected, standby, sent, replied, bounced) back into the brain."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

from brain.models import CampaignSpec, stable_id
from brain.store import Store

log = logging.getLogger("brain.bridge")


def export_to_review(store: Store, spec: CampaignSpec, include_flagged: bool = False) -> int:
    """Copies drafts into `professors` / `emails` so `python main.py review` shows them. Returns count.

    A person offered again from an earlier batch (recycled) keeps their one review row: it moves here, it is not copied.
    A decision made in the review (approved, rejected, standby, sent) is never reset by exporting again."""
    from db import database as legacy
    from orchestrator.state import DraftEmail, EmailStatus, ProfessorProfile, Publication, UserIntent

    legacy.init_db()
    s = spec.sender
    legacy.upsert_session_state(spec.id, UserIntent(
        user_name=s.name or None, user_profile_summary=s.summary or None, cv_path=s.attachment_path,
        research_domains=spec.topics, countries=spec.countries, timeline=spec.timeline, confirmed=True,
    ))
    moved = {m["contact_id"]: m["from"] for m in ((store.load_campaign(spec.id) or {}).get("funnel", {}).get("recycled") or [])}
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
        email_id = stable_id(spec.id, c.id)
        if c.id in moved:
            _try(legacy.move_email, stable_id(moved[c.id], c.id), email_id, spec.id)
        if _try(legacy.get_email_status, email_id) not in (None, "draft"):
            continue  # the user already decided on this one in the review
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
        if "previously_shown" in c.flags or c.id in moved:
            notes += " | PREVIOUSLY SHOWN: offered again because few new matches were found"
        if r["issues"]:
            notes += " | CHECK: " + "; ".join(r["issues"])
        legacy.save_draft_email(DraftEmail(
            id=email_id, professor_id=c.id, recipient_name=c.name, recipient_email=r["email"],
            subject=r["subject"] or "", body=r["body"], personalization_notes=notes, status=EmailStatus.DRAFT,
        ), session_id=spec.id)
        n += 1
    return n


def legacy_set_status(campaign_id: str, contact_id: str, status: str) -> int:
    """Mirror a decision onto the review UI's table. A brain-only install has no review stack (it needs LangGraph): then
    there is nothing to mirror and this returns 0 rather than failing."""
    try:
        from db import database as legacy

        return legacy.set_email_status_by_professor(campaign_id, contact_id, status)
    except Exception as e:
        log.debug("review table not updated: %s", e)
        return 0


def _try(fn, *args):
    """A legacy-table lookup that must never break an export (the review tables may not exist yet): None on any error."""
    try:
        return fn(*args)
    except Exception as e:
        log.debug("legacy lookup failed: %s", e)
        return None


_LEGACY_TO_RESULT = {"approved": "approved", "rejected": "rejected", "standby": "standby"}


def sync_outcomes(store: Store) -> dict:
    """Pull what happened in the review/send tables into the brain. Sent / replied / bounced (kept up to date by the Gmail
    sender and the reply monitor) go into the outcome log, where they teach the engine: bounces suppress addresses and
    demote bad email conventions, replies label good prospects. Approved / rejected / standby decisions go into the touch
    ledger, so a person the user rejected is never offered again and one on standby waits. A draft the user recalled
    from standby goes back to undecided. Safe to run repeatedly."""
    from sqlalchemy import text

    stats = {"sent": 0, "replied": 0, "bounced": 0, "unmatched": 0}  # decision counts appear only when non-zero
    if not getattr(store, "_has_legacy_emails", False):
        return stats
    with store._Session() as s:
        rows = s.execute(text("SELECT lower(recipient_email), status FROM emails WHERE status IN "
                              "('sent','replied','followed_up','bounced') AND recipient_email IS NOT NULL")).fetchall()
        try:
            decided = s.execute(text("SELECT session_id, professor_id, status FROM emails WHERE session_id IS NOT NULL "
                                     "AND status IN ('approved','rejected','standby','draft')")).fetchall()
        except Exception:  # a review table without session / professor columns: nothing to learn from it
            decided = []
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
            if kind == "replied" and (store.has_outcome(campaign_id, contact_id, "unsubscribed")
                                      or store.has_outcome(campaign_id, contact_id, "negative")):
                continue  # the reply monitor already read it: a "no" must not become a good-prospect label
            if not store.has_outcome(campaign_id, contact_id, kind):
                store.record_outcome(campaign_id, contact_id, kind)
                stats[kind] += 1
    for campaign_id, contact_id, status in decided:
        current = store.result_status(campaign_id, contact_id)
        if current is None:
            continue
        if status == "draft":
            if current == "standby":  # recalled in the review
                if store.revive(contact_id, campaign_id):
                    stats["recalled"] = stats.get("recalled", 0) + 1
            continue
        want = _LEGACY_TO_RESULT[status]
        # Only results still in play take a review decision; `recycled` (moved to a later batch), sent, replied, bounced
        # and rejected ones keep their status, or a stale review row would bring a moved person back.
        if current != want and current in ("shortlisted", "drafted", "needs_review", "standby", "approved"):
            store.set_status(campaign_id, contact_id, want)
            stats[want] = stats.get(want, 0) + 1
    return stats


# ── used by the review UI (and anything else that sends) ─────────────────────

def _store() -> Store:
    from brain.settings import Settings

    st = Settings()
    return Store(st.database_url, workspace=st.workspace)


def send_gate(address: str, campaign_id: str = "") -> Optional[str]:
    """None when this email may be sent now, else the reason in plain words. Checked right before every send, as the last
    line of defence behind discovery: suppressed or dead addresses, anyone already emailed, a series paused for bounces,
    and the daily cap / minimum delay (counted from the database, so they survive restarts)."""
    try:
        store = _store()
        why = store.address_blocked(address, campaign_id)
        if why:
            return explain(why)
        cap = int(os.getenv("BRAIN_DAILY_SEND_CAP", "100"))
        sent = store.sent_since(24)
        if sent >= cap:
            return f"Daily send limit reached: {sent} emails in the last 24 hours (limit {cap}, BRAIN_DAILY_SEND_CAP)."
        gap = float(os.getenv("BRAIN_MIN_SEND_DELAY", "30"))
        last = store.last_sent_at()
        if last is not None:
            wait = gap - (datetime.utcnow() - last).total_seconds()
            if wait > 0:
                return f"Sending too fast: wait {wait:.0f}s before the next email (BRAIN_MIN_SEND_DELAY)."
    except Exception as e:  # a broken gate must stop the send, not wave it through
        log.warning("send gate failed: %s", e)
        return f"Could not check the do-not-contact list ({e}); not sending."
    return None


def followup_gate(address: str) -> Optional[str]:
    """None when a follow-up may go to this address. Only the hard stops apply (an unsubscribe, a bounce, the
    do-not-contact list): the person has of course been emailed already. Fails closed."""
    try:
        if _store().is_suppressed(address):
            return explain("suppressed")
    except Exception as e:
        log.warning("follow-up gate failed: %s", e)
        return f"Could not check the do-not-contact list ({e}); not sending."
    return None


def record_reply(address: str, kind: str, detail: str = "") -> dict:
    """A reply that asks to stop (kind 'unsubscribed') suppresses the address for every campaign; a polite decline
    ('negative') is recorded so the prospect is never labelled good. Never raises: the reply monitor must keep going."""
    out = {"matched": 0}
    try:
        store = _store()
        if kind == "unsubscribed":
            store.suppress(address, "unsubscribed")
        for campaign_id, contact_id in store.results_by_email(address.lower()):
            if not store.has_outcome(campaign_id, contact_id, "sent"):
                store.record_outcome(campaign_id, contact_id, "sent")
            store.record_outcome(campaign_id, contact_id, kind, detail=detail)
            out["matched"] += 1
    except Exception as e:
        log.warning("could not record reply from %s: %s", address, e)
        out["error"] = str(e)
    return out


def explain(reason: str) -> str:
    return {
        "suppressed": "This address or its domain is on the do-not-contact list.",
        "already_contacted": "This person has already been emailed.",
        "do_not_contact": "This address is on the do-not-contact list (bounced, unsubscribed or imported).",
        "no_address": "There is no valid recipient address.",
    }.get(reason, reason.replace("series_paused: ", "This outreach is paused: "))


def record_decision(campaign_id: Optional[str], contact_id: Optional[str], decision: str) -> None:
    """Tell the brain what the user did in the review: approved | rejected | standby | recalled | sent. Never raises: the
    review must keep working if the brain is unreachable (the next sync catches up)."""
    if not campaign_id or not contact_id:
        return
    try:
        store = _store()
        if decision == "sent":
            store.record_outcome(campaign_id, contact_id, "sent")
        elif decision == "recalled":
            store.revive(contact_id, campaign_id)
        else:
            store.set_status(campaign_id, contact_id, decision)
    except Exception as e:
        log.warning("could not record %s for %s: %s", decision, contact_id, e)

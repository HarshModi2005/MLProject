"""
db/database.py

SQLAlchemy database setup.

Provides:
  - Engine creation
  - Session factory (scoped)
  - Table creation (create_all)
  - CRUD helper functions for Professor, Email, and Session models
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import sessionmaker

import config
from db.models import Base, Email, Professor, Session
from orchestrator.state import (
    DraftEmail,
    ProfessorProfile,
    Publication,
    EmailStatus,
    UserIntent,
    ExecutionPlan,
)
from rich.console import Console

console = Console()

# ── Engine & Session factory ──────────────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            connect_args={"check_same_thread": False}   # needed for SQLite
            if "sqlite" in config.DATABASE_URL else {},
            echo=False,
        )
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionFactory


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    console.print("[dim]✓ Database initialized.[/dim]")


@contextmanager
def db_session():
    """Context manager for database sessions with automatic rollback."""
    factory = get_session_factory()
    session: DBSession = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Professor CRUD ────────────────────────────────────────────────────────────

def save_professor(profile: ProfessorProfile, session_id: Optional[str] = None) -> None:
    """Insert or update a professor profile in the database."""
    with db_session() as db:
        existing = db.get(Professor, profile.id)
        if existing:
            # Update score if re-crawled
            existing.relevance_score = profile.relevance_score
            existing.crawled_at = datetime.utcnow()
        else:
            row = Professor(
                id=profile.id,
                name=profile.name,
                email=profile.email,
                institution=profile.institution,
                department=profile.department,
                lab_url=profile.lab_url,
                profile_url=profile.profile_url,
                research_keywords=json.dumps(profile.research_keywords),
                recent_publications=json.dumps(
                    [p.model_dump() for p in profile.recent_publications]
                ),
                accepting_students=profile.accepting_students,
                bio_snippet=profile.bio_snippet,
                raw_page_text=profile.raw_page_text,
                relevance_score=profile.relevance_score,
                session_id=session_id,
            )
            db.add(row)


def save_professors_batch(
    profiles: list[ProfessorProfile],
    session_id: Optional[str] = None,
) -> None:
    """Bulk save professors."""
    for p in profiles:
        save_professor(p, session_id)
    console.print(f"[green]✓ Saved {len(profiles)} professors to database.[/green]")


def get_professor(professor_id: str) -> Optional[Professor]:
    """Fetch a professor row by ID."""
    with db_session() as db:
        return db.get(Professor, professor_id)


def email_already_sent(professor_email: str) -> bool:
    """
    Duplicate detection: check if we've already sent an email
    to this address in any previous session.
    """
    with db_session() as db:
        existing = (
            db.query(Email)
            .filter(
                Email.recipient_email == professor_email,
                Email.status.in_(["sent", "followed_up", "replied"]),
            )
            .first()
        )
        return existing is not None


# ── Email CRUD ────────────────────────────────────────────────────────────────

def save_draft_email(draft: DraftEmail, session_id: Optional[str] = None) -> None:
    """Save a draft email to the database."""
    with db_session() as db:
        existing = db.get(Email, draft.id)
        if existing:
            existing.status = draft.status.value
            existing.sent_at = draft.sent_at
            existing.reply_received_at = draft.reply_received_at
            existing.follow_up_count = draft.follow_up_count
        else:
            row = Email(
                id=draft.id,
                professor_id=draft.professor_id,
                recipient_name=draft.recipient_name,
                recipient_email=draft.recipient_email,
                subject=draft.subject,
                body=draft.body,
                personalization_notes=draft.personalization_notes,
                status=draft.status.value,
                sent_at=draft.sent_at,
                reply_received_at=draft.reply_received_at,
                follow_up_count=draft.follow_up_count,
                session_id=session_id,
            )
            db.add(row)


def mark_email_sent(email_id: str) -> None:
    """Update an email's status to 'sent' and record send timestamp."""
    with db_session() as db:
        row = db.get(Email, email_id)
        if row:
            row.status = "sent"
            row.sent_at = datetime.utcnow()


def mark_email_replied(email_id: str) -> None:
    """Mark an email as having received a reply."""
    with db_session() as db:
        row = db.get(Email, email_id)
        if row:
            row.status = "replied"
            row.reply_received_at = datetime.utcnow()


def get_pending_followups(followup_days: int = 7) -> list[Email]:
    """
    Return emails that were sent > followup_days ago and haven't been replied to.
    """
    from sqlalchemy import and_
    cutoff = datetime.utcnow()
    with db_session() as db:
        rows = (
            db.query(Email)
            .filter(
                and_(
                    Email.status == "sent",
                    Email.sent_at != None,
                    Email.follow_up_count < 2,
                )
            )
            .all()
        )
        # Filter in Python (simpler than SQL date math for SQLite)
        pending = []
        for row in rows:
            if row.sent_at:
                days_elapsed = (datetime.utcnow() - row.sent_at).days
                if days_elapsed >= followup_days:
                    pending.append(row)
        return pending


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(session_id: str, user_name: Optional[str] = None) -> None:
    """Create a new session record."""
    with db_session() as db:
        row = Session(id=session_id, user_name=user_name)
        db.add(row)


def update_session(
    session_id: str,
    intent_json: Optional[str] = None,
    plan_json: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Update session state."""
    with db_session() as db:
        row = db.get(Session, session_id)
        if row:
            if intent_json is not None:
                row.intent_json = intent_json
            if plan_json is not None:
                row.plan_json = plan_json
            if mode is not None:
                row.mode = mode


def upsert_session_state(
    session_id: str,
    intent: UserIntent,
    plan: Optional[ExecutionPlan] = None,
) -> None:
    """
    Persist intent (and optional plan) so the Streamlit review UI can attach CVs
    and read outreach settings when sending.
    """
    intent_js = intent.model_dump_json()
    plan_js = plan.model_dump_json() if plan else None
    with db_session() as db:
        row = db.get(Session, session_id)
        if row is None:
            row = Session(
                id=session_id,
                user_name=intent.user_name,
                intent_json=intent_js,
                plan_json=plan_js,
            )
            db.add(row)
        else:
            row.intent_json = intent_js
            if plan_js is not None:
                row.plan_json = plan_js
            if intent.user_name:
                row.user_name = intent.user_name


def get_session_intent_plan(session_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (intent_json, plan_json) for a session, or (None, None) if missing."""
    if not session_id:
        return None, None
    with db_session() as db:
        row = db.get(Session, session_id)
        if not row:
            return None, None
        return row.intent_json, row.plan_json

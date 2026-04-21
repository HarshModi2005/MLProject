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

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import sessionmaker

import config
from db.models import Base, Email, Professor, Session, SessionRevision
from orchestrator.state import (
    DraftEmail,
    EmailStatus,
    ExecutionPlan,
    ProfessorProfile,
    Publication,
    UserIntent,
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


def _sqlite_try_add_column(engine, table: str, column: str, ddl_suffix: str) -> None:
    if "sqlite" not in config.DATABASE_URL:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {column} {ddl_suffix}'))
            conn.commit()
    except Exception as e:
        msg = str(e).lower()
        if "duplicate column" in msg or "already exists" in msg:
            return
        raise


def migrate_schema() -> None:
    """Add columns introduced after first deploy (SQLite-friendly)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    if "sqlite" not in config.DATABASE_URL:
        return
    _sqlite_try_add_column(engine, "sessions", "pipeline_step", "VARCHAR(40) DEFAULT 'intent'")
    _sqlite_try_add_column(engine, "sessions", "state_version", "INTEGER DEFAULT 1")
    _sqlite_try_add_column(engine, "professors", "relevance_breakdown_json", "TEXT")
    _sqlite_try_add_column(engine, "professors", "identity_key", "VARCHAR(120)")
    _sqlite_try_add_column(engine, "professors", "orcid", "VARCHAR(40)")
    _sqlite_try_add_column(engine, "professors", "email_source", "VARCHAR(80)")
    _sqlite_try_add_column(engine, "professors", "email_confidence", "FLOAT")
    _sqlite_try_add_column(engine, "emails", "variants_json", "TEXT")


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    migrate_schema()
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

def _professor_extras(profile: ProfessorProfile) -> dict:
    bd = None
    if profile.relevance_breakdown is not None:
        bd = profile.relevance_breakdown.model_dump_json()
    return {
        "relevance_breakdown_json": bd,
        "identity_key": profile.identity_key,
        "orcid": profile.orcid,
        "email_source": profile.email_source,
        "email_confidence": profile.email_confidence,
    }


def save_professor(profile: ProfessorProfile, session_id: Optional[str] = None) -> None:
    """Insert or update a professor profile in the database."""
    extras = _professor_extras(profile)
    with db_session() as db:
        existing = db.get(Professor, profile.id)
        if existing:
            existing.relevance_score = profile.relevance_score
            existing.crawled_at = datetime.utcnow()
            existing.email = profile.email
            existing.name = profile.name
            existing.institution = profile.institution
            existing.department = profile.department
            existing.lab_url = profile.lab_url
            existing.profile_url = profile.profile_url
            existing.research_keywords = json.dumps(profile.research_keywords)
            existing.recent_publications = json.dumps(
                [p.model_dump() for p in profile.recent_publications]
            )
            existing.accepting_students = profile.accepting_students
            existing.bio_snippet = profile.bio_snippet
            existing.raw_page_text = profile.raw_page_text
            existing.session_id = session_id or existing.session_id
            existing.relevance_breakdown_json = extras["relevance_breakdown_json"]
            existing.identity_key = extras["identity_key"]
            existing.orcid = extras["orcid"]
            existing.email_source = extras["email_source"]
            existing.email_confidence = extras["email_confidence"]
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
                relevance_breakdown_json=extras["relevance_breakdown_json"],
                identity_key=extras["identity_key"],
                orcid=extras["orcid"],
                email_source=extras["email_source"],
                email_confidence=extras["email_confidence"],
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

def _draft_variants_json(draft: DraftEmail) -> Optional[str]:
    if not draft.variants:
        return None
    return json.dumps(
        {
            "selected_variant_index": draft.selected_variant_index,
            "variants": [v.model_dump() for v in draft.variants],
        },
        default=str,
    )


def save_draft_email(draft: DraftEmail, session_id: Optional[str] = None) -> None:
    """Save a draft email to the database."""
    vjson = _draft_variants_json(draft)
    with db_session() as db:
        existing = db.get(Email, draft.id)
        if existing:
            existing.status = draft.status.value
            existing.sent_at = draft.sent_at
            existing.reply_received_at = draft.reply_received_at
            existing.follow_up_count = draft.follow_up_count
            if vjson:
                existing.variants_json = vjson
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
                variants_json=vjson,
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
        row = Session(
            id=session_id,
            user_name=user_name,
            pipeline_step="intent",
            state_version=1,
        )
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


def ensure_session_row(session_id: str, user_name: Optional[str] = None) -> None:
    with db_session() as db:
        if db.get(Session, session_id) is None:
            db.add(
                Session(
                    id=session_id,
                    user_name=user_name,
                    pipeline_step="intent",
                    state_version=1,
                )
            )


def append_session_revision(
    session_id: str,
    step: str,
    payload: Optional[dict] = None,
) -> None:
    """Bump session version, set pipeline_step, append revision row."""
    with db_session() as db:
        row = db.get(Session, session_id)
        if row is None:
            row = Session(id=session_id, pipeline_step=step, state_version=1)
            db.add(row)
            db.flush()
        nv = (row.state_version or 0) + 1
        row.state_version = nv
        row.pipeline_step = step
        row.updated_at = datetime.utcnow()
        db.add(
            SessionRevision(
                session_id=session_id,
                version=nv,
                step=step,
                payload_json=json.dumps(payload or {}, default=str),
            )
        )


def list_resumable_sessions(limit: int = 8) -> list[tuple[str, str, Optional[datetime]]]:
    """Sessions that have intent saved and are not marked done. Returns detached tuples."""
    with db_session() as db:
        q = (
            db.query(Session)
            .filter(Session.intent_json.isnot(None))
            .filter(Session.pipeline_step != "done")
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        rows = q.all()
        return [(r.id, r.pipeline_step or "", r.updated_at) for r in rows]


def load_session_dicts(session_id: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return (intent dict, plan dict) from DB for a session id."""
    ij, pj = get_session_intent_plan(session_id)
    intent_d = json.loads(ij) if ij else None
    plan_d = json.loads(pj) if pj else None
    return intent_d, plan_d

"""
db/models.py

SQLAlchemy ORM models for all persistent entities:
  - Professor: crawled and scored professor profiles
  - Email: draft and sent email tracking
  - Session: user session + intent + plan state
  - SessionRevision: append-only audit trail for resume / multi-machine sync
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Professor(Base):
    """Stores a scraped and scored professor profile."""

    __tablename__ = "professors"

    id = Column(String(36), primary_key=True)                  # UUID
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=True, index=True)
    institution = Column(String(300), nullable=True)
    department = Column(String(200), nullable=True)
    lab_url = Column(String(500), nullable=True)
    profile_url = Column(String(500), nullable=True)
    research_keywords = Column(Text, nullable=True)            # JSON array string
    recent_publications = Column(Text, nullable=True)          # JSON array string
    accepting_students = Column(Boolean, nullable=True)
    bio_snippet = Column(Text, nullable=True)
    raw_page_text = Column(Text, nullable=True)
    relevance_score = Column(Float, default=0.0)
    crawled_at = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=True)
    relevance_breakdown_json = Column(Text, nullable=True)
    identity_key = Column(String(120), nullable=True, index=True)
    orcid = Column(String(40), nullable=True)
    email_source = Column(String(80), nullable=True)
    email_confidence = Column(Float, nullable=True)

    # Relationships
    emails = relationship("Email", back_populates="professor")

    def __repr__(self) -> str:
        return f"<Professor {self.name} @ {self.institution} (score={self.relevance_score:.3f})>"


class Email(Base):
    """Tracks all outreach emails (draft, sent, replied, followed-up)."""

    __tablename__ = "emails"

    id = Column(String(36), primary_key=True)                  # UUID
    professor_id = Column(String(36), ForeignKey("professors.id"), nullable=False)
    recipient_name = Column(String(200), nullable=False)
    recipient_email = Column(String(200), nullable=False, index=True)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    personalization_notes = Column(Text, nullable=True)
    status = Column(String(50), default="draft", index=True)   # EmailStatus value
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)
    reply_received_at = Column(DateTime, nullable=True)
    follow_up_count = Column(Integer, default=0)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=True)
    variants_json = Column(Text, nullable=True)

    # Relationships
    professor = relationship("Professor", back_populates="emails")

    def __repr__(self) -> str:
        return f"<Email to={self.recipient_email} status={self.status}>"


class Session(Base):
    """
    Stores a complete user session:
      - Serialized UserIntent (JSON)
      - Serialized ExecutionPlan (JSON)
      - Current pipeline mode / step for reliable resume
    """

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    intent_json = Column(Text, nullable=True)                  # Serialized UserIntent
    plan_json = Column(Text, nullable=True)                    # Serialized ExecutionPlan
    mode = Column(String(50), default="clarification")         # AgentMode value
    user_name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    pipeline_step = Column(String(40), default="intent")
    state_version = Column(Integer, default=1)

    revisions = relationship("SessionRevision", back_populates="session")

    def __repr__(self) -> str:
        return f"<Session id={self.id[:8]} mode={self.mode} step={self.pipeline_step}>"


class SessionRevision(Base):
    """Append-only history of session checkpoints (intent/plan/crawl/score/email)."""

    __tablename__ = "session_revisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    step = Column(String(40), nullable=False)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="revisions")

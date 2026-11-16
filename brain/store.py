"""
Persistence for the brain (SQLAlchemy; SQLite locally, Postgres in production via DATABASE_URL).

Holds what must outlive a run: campaigns and their stage checkpoints (so a crashed run resumes
instead of paying again), every contact with a stable id, per-campaign results and drafts, and
the do-not-contact list that is enforced across all campaigns.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from brain.models import CampaignSpec, Contact, Draft
from brain.resolve import strong_keys
from brain.textutil import registrable_domain

CONTACTED = ("sent", "replied", "followed_up", "bounced")


class Base(DeclarativeBase):
    pass


class CampaignRow(Base):
    __tablename__ = "brain_campaigns"
    id = Column(String(32), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="running")
    spec_json = Column(Text)
    funnel_json = Column(Text, default="{}")


class StageRow(Base):
    __tablename__ = "brain_stages"
    campaign_id = Column(String(32), primary_key=True)
    stage = Column(String(32), primary_key=True)
    payload_json = Column(Text)
    completed_at = Column(DateTime, default=datetime.utcnow)


class ContactRow(Base):
    __tablename__ = "brain_contacts"
    id = Column(String(32), primary_key=True)
    name = Column(String(200), index=True)
    org = Column(String(300))
    email = Column(String(200), index=True)
    email_confidence = Column(Float, default=0.0)
    data_json = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContactKeyRow(Base):
    __tablename__ = "brain_contact_keys"
    key = Column(String(250), primary_key=True)
    contact_id = Column(String(32), index=True)


class ResultRow(Base):
    __tablename__ = "brain_results"
    campaign_id = Column(String(32), primary_key=True)
    contact_id = Column(String(32), primary_key=True)
    status = Column(String(32), default="shortlisted", index=True)  # shortlisted|drafted|needs_review|approved|sent|replied|bounced|rejected
    email = Column(String(200), index=True)
    prescore = Column(Float, default=0.0)
    score = Column(Float)
    reasons_json = Column(Text, default="[]")
    hook = Column(Text)
    subject = Column(Text)
    body = Column(Text)
    issues_json = Column(Text, default="[]")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SuppressionRow(Base):
    __tablename__ = "brain_suppression"
    value = Column(String(250), primary_key=True)  # an email address or a domain
    reason = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)


class Store:
    def __init__(self, database_url: str = "sqlite:///:memory:"):
        args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=args)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._has_legacy_emails = inspect(self.engine).has_table("emails")

    # ── campaigns and checkpoints ────────────────────────────────────────────
    def save_campaign(self, spec: CampaignSpec, status: str = "running", funnel: Optional[Dict[str, Any]] = None) -> None:
        with self._Session.begin() as s:
            row = s.get(CampaignRow, spec.id) or CampaignRow(id=spec.id)
            row.spec_json = spec.model_dump_json()
            row.status = status
            if funnel is not None:
                row.funnel_json = json.dumps(funnel)
            s.merge(row)

    def load_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        with self._Session() as s:
            row = s.get(CampaignRow, campaign_id)
            if not row:
                return None
            return {"spec": CampaignSpec.model_validate_json(row.spec_json), "status": row.status,
                    "funnel": json.loads(row.funnel_json or "{}"), "created_at": row.created_at}

    def list_campaigns(self) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = s.query(CampaignRow).order_by(CampaignRow.created_at.desc()).all()
            return [{"id": r.id, "status": r.status, "created_at": r.created_at,
                     "objective": json.loads(r.spec_json).get("objective", "")} for r in rows]

    def checkpoint(self, campaign_id: str, stage: str, contacts: List[Contact], extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"contacts": [c.model_dump() for c in contacts], "extra": extra or {}}
        with self._Session.begin() as s:
            s.merge(StageRow(campaign_id=campaign_id, stage=stage, payload_json=json.dumps(payload),
                             completed_at=datetime.utcnow()))

    def load_checkpoint(self, campaign_id: str, stage: str) -> Optional[Dict[str, Any]]:
        with self._Session() as s:
            row = s.get(StageRow, (campaign_id, stage))
            if not row:
                return None
            payload = json.loads(row.payload_json)
            return {"contacts": [Contact.model_validate(c) for c in payload["contacts"]], "extra": payload.get("extra", {})}

    def save_context(self, campaign_id: str, ctx) -> None:
        with self._Session.begin() as s:
            s.merge(StageRow(campaign_id=campaign_id, stage="__context__", payload_json=ctx.model_dump_json(),
                             completed_at=datetime.utcnow()))

    def load_context(self, campaign_id: str):
        from brain.context import CampaignContext

        with self._Session() as s:
            row = s.get(StageRow, (campaign_id, "__context__"))
            return CampaignContext.model_validate_json(row.payload_json) if row else CampaignContext()

    # ── contacts ─────────────────────────────────────────────────────────────
    def upsert_contacts(self, contacts: List[Contact]) -> None:
        """Save contacts, reusing the id of anyone already known under one of their identifiers."""
        with self._Session.begin() as s:
            for c in contacts:
                keys = strong_keys(c)
                known = s.query(ContactKeyRow).filter(ContactKeyRow.key.in_(keys)).first() if keys else None
                if known:
                    c.id = known.contact_id
                best = c.best_email
                s.merge(ContactRow(id=c.id, name=c.name, org=c.org, email=best.address if best else None,
                                   email_confidence=best.confidence if best else 0.0, data_json=c.model_dump_json()))
                for k in keys:
                    s.merge(ContactKeyRow(key=k[:250], contact_id=c.id))

    # ── do-not-contact ───────────────────────────────────────────────────────
    def suppress(self, value: str, reason: str = "manual") -> None:
        with self._Session.begin() as s:
            s.merge(SuppressionRow(value=value.strip().lower(), reason=reason))

    def blocked_reason(self, c: Contact) -> Optional[str]:
        """Why this contact must not be emailed, or None. Checked for every address we hold."""
        addresses = [e.address for e in c.emails]
        domains = {a.split("@")[1] for a in addresses} | ({c.org_domain} if c.org_domain else set())
        candidates = set(addresses) | domains | {registrable_domain(d) for d in domains}
        with self._Session() as s:
            if candidates and s.query(SuppressionRow).filter(SuppressionRow.value.in_(candidates)).first():
                return "suppressed"
            if addresses:
                if s.query(ResultRow).filter(ResultRow.email.in_(addresses), ResultRow.status.in_(CONTACTED)).first():
                    return "already_contacted"
                if self._has_legacy_emails:
                    stmt = text("SELECT 1 FROM emails WHERE status IN ('sent','replied','followed_up') "
                                "AND lower(recipient_email) = :e LIMIT 1")
                    for a in addresses:
                        if s.execute(stmt, {"e": a}).first():
                            return "already_contacted"
        return None

    # ── results ──────────────────────────────────────────────────────────────
    def save_results(self, campaign_id: str, contacts: List[Contact], drafts: Dict[str, Draft]) -> None:
        with self._Session.begin() as s:
            for c in contacts:
                d = drafts.get(c.id)
                best = c.best_email
                status = "shortlisted" if d is None else ("drafted" if d.ok else "needs_review")
                existing = s.get(ResultRow, (campaign_id, c.id))
                if existing and existing.status in CONTACTED + ("approved", "rejected"):
                    continue  # never overwrite a human decision or a send
                s.merge(ResultRow(
                    campaign_id=campaign_id, contact_id=c.id, status=status, email=best.address if best else None,
                    prescore=c.prescore, score=c.score, reasons_json=json.dumps(c.reasons), hook=c.hook,
                    subject=d.subject if d else None, body=d.body if d else None,
                    issues_json=json.dumps(d.issues if d else []),
                ))

    def results(self, campaign_id: str) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = (s.query(ResultRow, ContactRow).join(ContactRow, ContactRow.id == ResultRow.contact_id)
                    .filter(ResultRow.campaign_id == campaign_id).order_by(ResultRow.score.desc()).all())
            return [{
                "contact_id": r.contact_id, "name": c.name, "org": c.org, "email": r.email, "status": r.status,
                "score": r.score, "prescore": r.prescore, "reasons": json.loads(r.reasons_json or "[]"),
                "hook": r.hook, "subject": r.subject, "body": r.body, "issues": json.loads(r.issues_json or "[]"),
                "contact": Contact.model_validate_json(c.data_json),
            } for r, c in rows]

    def set_status(self, campaign_id: str, contact_id: str, status: str) -> None:
        with self._Session.begin() as s:
            row = s.get(ResultRow, (campaign_id, contact_id))
            if row:
                row.status = status

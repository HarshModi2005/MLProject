"""
Persistence for the brain (SQLAlchemy; SQLite locally, Postgres in production via DATABASE_URL).

Holds what must outlive a run: campaigns and their stage checkpoints (so a crashed run resumes
instead of paying again), every contact with a stable id, per-campaign results and drafts, and
the do-not-contact list that is enforced across all campaigns.

It is also the brain's long-term memory:
  * prospects verified in earlier campaigns (with evidence and a timestamp) are offered to later
    campaigns before any search is paid for (`known_prospects`, the `memory` discovery tool);
  * which query templates found qualified prospects, per playbook (`learned_templates`);
  * human review decisions become evaluation labels (`labels_from_review`).
NOTE: memory is shared across all campaigns in ONE workspace. Give every user their own workspace
(Store(url, workspace=user_id), BRAIN_WORKSPACE or `brain --workspace`): each gets separate storage, so nothing
learned for one user can reach another (brain/workspace.py).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from brain import touch
from brain.models import CampaignSpec, Contact, Draft, EmailCandidate
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


class OutcomeRow(Base):
    """What actually happened to an email after it was sent: the only ground truth the engine ever gets."""
    __tablename__ = "brain_outcomes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(String(64), index=True)
    contact_id = Column(String(64), index=True)
    address = Column(String(250), index=True)
    kind = Column(String(16), index=True)  # sent | bounced | replied | positive | negative | unsubscribed | spam
    method = Column(String(24))  # how the address was found (mailto, pattern_global, hunter ...)
    tier = Column(String(16))  # verified | attributed | inbox | guess | unattributed, when it was sent
    detail = Column(Text, default="")
    at = Column(DateTime, default=datetime.utcnow)


OUTCOME_KINDS = ("sent", "bounced", "replied", "positive", "negative", "unsubscribed", "spam")


class TouchRow(Base):
    """The touch ledger (brain/touch.py): one row per identity key of everyone shown to the user or emailed, so
    "have we been here before" is one primary-key lookup however the person was found."""
    __tablename__ = "brain_touch"
    key = Column(String(250), primary_key=True)  # email:… | orcid:… | person:<name>|<org> | org:<domain> | name:<name>
    contact_id = Column(String(64), index=True)
    campaign_id = Column(String(64), index=True)
    series_id = Column(String(64), index=True)
    state = Column(String(16), index=True)  # see brain/touch.py
    kind = Column(String(8), default="person")  # person | org
    name = Column(String(300))
    org_key = Column(String(300), index=True)  # for the per-organisation cap and name matching
    updated_at = Column(DateTime, default=datetime.utcnow)


class SeriesRow(Base):
    """One brief run in many batches ("find more"). The first campaign's id is the series id."""
    __tablename__ = "brain_series"
    id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(12), default="active")  # active | paused
    pause_reason = Column(Text, default="")
    no_repeat = Column(String(12), default="series")
    cooldown_days = Column(Integer, default=90)
    recycle = Column(String(8), default="ask")
    widen = Column(String(8), default="")  # how far discovery may widen: "" | topics | geo
    rung = Column(Integer, default=0)  # how far down the widening ladder the series has gone (brain/series.py)
    health_since = Column(DateTime, nullable=True)  # bounce rate counts only outcomes after this (set by series-resume)


class SeriesMemberRow(Base):
    __tablename__ = "brain_series_members"
    campaign_id = Column(String(64), primary_key=True)
    series_id = Column(String(64), index=True)
    batch_no = Column(Integer, default=1)
    rung = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryStatRow(Base):
    __tablename__ = "brain_query_stats"
    playbook = Column(String(64), primary_key=True)
    template = Column(String(300), primary_key=True)  # query with {topic} / {place} placeholders
    runs = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    good = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FitExampleRow(Base):
    """A candidate the (pre-)judge or a human scored: training data for the learned fit model."""
    __tablename__ = "brain_fit_examples"
    campaign_id = Column(String(64), primary_key=True)
    contact_id = Column(String(64), primary_key=True)
    playbook = Column(String(64), index=True)
    name = Column(String(300))
    features_json = Column(Text)
    judge_score = Column(Float, nullable=True)
    prejudge = Column(Float, nullable=True)
    label = Column(String(16), nullable=True)  # good | bad, from a human
    created_at = Column(DateTime, default=datetime.utcnow)


class FitModelRow(Base):
    __tablename__ = "brain_fit_models"
    playbook = Column(String(64), primary_key=True)
    model_json = Column(Text)
    n = Column(Integer, default=0)
    trained_at = Column(DateTime, default=datetime.utcnow)


class SourceStatRow(Base):
    """What a source produced across finished campaigns of a playbook (see brain/discovery/precision/priors.py)."""
    __tablename__ = "brain_source_stats"
    playbook = Column(String(64), primary_key=True)
    source = Column(String(64), primary_key=True)
    campaigns = Column(Integer, default=0)
    unique = Column(Float, default=0)
    reachable = Column(Float, default=0)
    passed = Column(Float, default=0)
    shortlisted = Column(Float, default=0)
    search_calls = Column(Float, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GraphNodeRow(Base):
    """Entity graph node (brain/discovery/graph.py): org, person, event, outlet, paper, package, show."""
    __tablename__ = "graph_nodes"
    id = Column(String(300), primary_key=True)  # "<type>:<key>"
    type = Column(String(32), index=True)
    name = Column(String(300))
    contact_id = Column(String(64), nullable=True, index=True)
    data_json = Column(Text)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)


class GraphEdgeRow(Base):
    __tablename__ = "graph_edges"
    src = Column(String(300), primary_key=True)
    dst = Column(String(300), primary_key=True, index=True)
    type = Column(String(32), primary_key=True)
    source = Column(String(64))
    url = Column(Text, nullable=True)
    seen_at = Column(DateTime, default=datetime.utcnow)


class DomainPatternRow(Base):
    """An organisation's email convention, confirmed by addresses found in any campaign (discovery/reach/patterns.py)."""
    __tablename__ = "brain_domain_patterns"
    domain = Column(String(200), primary_key=True)
    pattern = Column(String(32), primary_key=True)
    support = Column(Integer, default=0)
    last_confirmed = Column(DateTime, default=datetime.utcnow)


class NetworkRow(Base):
    """Someone the user knows, imported from their own data (discovery/warm): LinkedIn export, contacts, vCard."""
    __tablename__ = "brain_network"
    id = Column(String(64), primary_key=True)
    name = Column(String(200), index=True)
    email = Column(String(200), index=True, nullable=True)
    company = Column(String(300), nullable=True)
    domain = Column(String(200), index=True, nullable=True)
    title = Column(String(300), nullable=True)
    source = Column(String(32))
    connected_on = Column(String(32), nullable=True)
    imported_at = Column(DateTime, default=datetime.utcnow)


class FollowupRow(Base):
    __tablename__ = "brain_followups"
    campaign_id = Column(String(64), primary_key=True)
    contact_id = Column(String(64), primary_key=True)
    n = Column(Integer, primary_key=True)
    subject = Column(Text)
    body = Column(Text)
    issues_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserProfileRow(Base):
    """What the system remembers about the user between campaigns (signature, tone, standing exclusions ...)."""
    __tablename__ = "brain_user_profile"
    id = Column(String(64), primary_key=True)
    data_json = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def query_template(query: str, topics: List[str], places: List[str]) -> Optional[str]:
    """'edtech sponsors Bangalore' -> '{topic} sponsors {place}'. None if the query has no topic in it."""
    out, has_topic = query, False
    for t in sorted([t for t in topics if t], key=len, reverse=True):
        new = re.sub(re.escape(t), "{topic}", out, flags=re.I)
        has_topic |= new != out
        out = new
    for pl in sorted([p for p in places if p], key=len, reverse=True):
        out = re.sub(re.escape(pl), "{place}", out, flags=re.I)
    if not has_topic or out.count("{topic}") > 1 or "site:" in out:
        return None
    return out[:300]


class Store:
    def __init__(self, database_url: str = "sqlite:///:memory:", workspace: str = "default"):
        """`workspace` isolates one user's data from another's (brain/workspace.py): its own file or schema."""
        from brain.workspace import clean, scope_url

        self.workspace = clean(workspace)
        database_url, schema = scope_url(database_url, self.workspace)
        args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=args)
        if schema:  # PostgreSQL: every table of this store lives in the workspace's own schema
            with self.engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            self.engine = self.engine.execution_options(schema_translate_map={None: schema})
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._has_legacy_emails = inspect(self.engine).has_table("emails")
        self._backfill_touch()

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

    # ── memory ───────────────────────────────────────────────────────────────
    def known_prospects(self, spec: CampaignSpec, prof, limit: int = 30, max_age_days: int = 180,
                        scan: int = 3000, is_blocked=None) -> List[Contact]:
        """Prospects verified (reachable) in earlier campaigns that fit this one. Free: no search, no LLM."""
        from brain.fit import quick_fit

        kind = spec.target_kind or None
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        out: List[tuple] = []
        with self._Session() as s:
            rows = (s.query(ContactRow).filter(ContactRow.updated_at >= cutoff, ContactRow.email.isnot(None))
                    .order_by(ContactRow.updated_at.desc()).limit(scan).all())
        from brain.discovery.index import decay

        now = datetime.utcnow()
        for r in rows:
            try:
                c = Contact.model_validate_json(r.data_json)
            except ValueError:
                continue
            if kind and c.kind != kind:
                continue
            decay(c, (now - (r.updated_at or now)).total_seconds() / 86400)  # expired proof is not offered as current
            if spec.countries and c.country and c.country not in spec.countries:
                continue
            fit = quick_fit(c, prof)
            if fit < 0.3:
                continue
            if is_blocked and is_blocked(c):
                continue  # already shown or contacted: must not use up the quota (checked last: it costs queries)
            c.score, c.reasons, c.hook, c.prescore, c.criteria = None, [], None, 0.0, {}
            c.flags = [f for f in c.flags if f not in ("unjudged",)] + ["from_memory"]
            c.sources = list(dict.fromkeys(c.sources + ["memory"]))
            out.append((fit, c))
        out.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in out[:limit]]

    def record_query_stats(self, playbook: str, spec: CampaignSpec, query_log) -> int:
        """Fold this campaign's query yields into per-playbook template statistics."""
        places = spec.regions + list(spec.countries)
        from brain.searchplan import _country_name

        places += [_country_name(c) for c in spec.countries]
        n = 0
        with self._Session.begin() as s:
            for q in query_log:
                tpl = query_template(q.query, spec.topics + spec.expanded_terms, places)
                if not tpl:
                    continue
                row = s.get(QueryStatRow, (playbook, tpl)) or QueryStatRow(playbook=playbook, template=tpl, runs=0,
                                                                          hits=0, good=0)
                row.runs = (row.runs or 0) + 1
                row.hits = (row.hits or 0) + int(q.new)
                row.good = (row.good or 0) + int(q.good if q.good >= 0 else q.new)
                s.merge(row)
                n += 1
        return n

    def learned_templates(self, playbook: str, limit: int = 5) -> List[str]:
        with self._Session() as s:
            rows = s.query(QueryStatRow).filter(QueryStatRow.playbook == playbook, QueryStatRow.good > 0).all()
        rows.sort(key=lambda r: (r.good / max(1, r.runs), r.good), reverse=True)
        return [r.template for r in rows[:limit]]

    # ── learned fit model ────────────────────────────────────────────────────
    def record_fit_examples(self, playbook: str, campaign_id: str, rows: List[Dict[str, Any]]) -> int:
        """Upsert training rows ({contact_id, name, features, judge_score, prejudge}); human labels are kept."""
        with self._Session.begin() as s:
            for r in rows:
                row = s.get(FitExampleRow, (campaign_id, r["contact_id"])) or FitExampleRow(
                    campaign_id=campaign_id, contact_id=r["contact_id"])
                row.playbook, row.name = playbook, (r.get("name") or "")[:300]
                row.features_json = json.dumps(r["features"])
                if r.get("judge_score") is not None:
                    row.judge_score = r["judge_score"]
                if r.get("prejudge") is not None:
                    row.prejudge = r["prejudge"]
                s.merge(row)
        return len(rows)

    def label_fit_examples(self, playbook: str, good: List[str], bad: List[str]) -> int:
        """Attach human labels (names or contact ids) to stored examples of a playbook."""
        from brain.textutil import name_tokens

        def key(v: str) -> str:
            return " ".join(name_tokens(v))

        g, b = {key(x) for x in good}, {key(x) for x in bad}
        n = 0
        with self._Session.begin() as s:
            for row in s.query(FitExampleRow).filter(FitExampleRow.playbook == playbook):
                k = key(row.name or "")
                label = "good" if (k in g or row.contact_id in good) else "bad" if (k in b or row.contact_id in bad) else None
                if label and row.label != label:
                    row.label = label
                    n += 1
        return n

    def fit_examples(self, playbook: str) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = s.query(FitExampleRow).filter(FitExampleRow.playbook == playbook).all()
        return [{"contact_id": r.contact_id, "name": r.name, "features": json.loads(r.features_json or "{}"),
                 "judge_score": r.judge_score, "prejudge": r.prejudge, "label": r.label} for r in rows]

    def save_fit_model(self, playbook: str, model: Dict[str, Any]) -> None:
        with self._Session.begin() as s:
            s.merge(FitModelRow(playbook=playbook, model_json=json.dumps(model), n=int(model.get("n", 0)),
                                trained_at=datetime.utcnow()))

    def load_fit_model(self, playbook: str) -> Optional[Dict[str, Any]]:
        with self._Session() as s:
            row = s.get(FitModelRow, playbook)
        return json.loads(row.model_json) if row and row.model_json else None

    # ── source priors ────────────────────────────────────────────────────────
    def record_source_stats(self, playbook: str, source_funnel: Dict[str, Dict[str, int]],
                            ledger: Dict[str, Dict[str, float]]) -> int:
        sources = set(ledger) | {k for st in source_funnel.values() for k in st}
        with self._Session.begin() as s:
            for src in sources:
                row = s.get(SourceStatRow, (playbook, src)) or SourceStatRow(
                    playbook=playbook, source=src, campaigns=0, unique=0, reachable=0, passed=0, shortlisted=0,
                    search_calls=0)
                row.campaigns = (row.campaigns or 0) + 1
                for attr, stage in (("unique", "unique"), ("reachable", "reachable"), ("passed", "passed_judge"),
                                    ("shortlisted", "shortlisted")):
                    setattr(row, attr, (getattr(row, attr) or 0) + (source_funnel.get(stage) or {}).get(src, 0))
                row.search_calls = (row.search_calls or 0) + float((ledger.get(src) or {}).get("search_calls", 0))
                s.merge(row)
        return len(sources)

    def source_priors(self, playbook: str) -> Dict[str, Dict[str, float]]:
        with self._Session() as s:
            rows = s.query(SourceStatRow).filter(SourceStatRow.playbook == playbook).all()
        return {r.source: {"campaigns": r.campaigns or 0, "unique": r.unique or 0, "reachable": r.reachable or 0,
                           "passed": r.passed or 0, "shortlisted": r.shortlisted or 0,
                           "search_calls": r.search_calls or 0} for r in rows}

    # ── entity graph ─────────────────────────────────────────────────────────
    def graph_write(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        now = datetime.utcnow()
        with self._Session.begin() as s:
            for n in nodes:
                row = s.get(GraphNodeRow, n["id"][:300]) or GraphNodeRow(id=n["id"][:300], first_seen=now)
                row.type, row.name, row.last_seen = n["type"], (n.get("name") or "")[:300], now
                if n.get("contact_id"):
                    row.contact_id = n["contact_id"]
                if n.get("data") is not None:
                    row.data_json = json.dumps(n["data"], default=str)
                s.merge(row)
            for e in edges:
                s.merge(GraphEdgeRow(src=e["src"][:300], dst=e["dst"][:300], type=e["type"], source=e.get("source"),
                                     url=e.get("url"), seen_at=now))

    def graph_edges(self, node_ids: List[str], types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Edges touching any of these nodes (either direction)."""
        if not node_ids:
            return []
        with self._Session() as s:
            q = s.query(GraphEdgeRow).filter((GraphEdgeRow.src.in_(node_ids)) | (GraphEdgeRow.dst.in_(node_ids)))
            if types:
                q = q.filter(GraphEdgeRow.type.in_(types))
            rows = q.limit(5000).all()
        return [{"src": r.src, "dst": r.dst, "type": r.type, "source": r.source, "url": r.url} for r in rows]

    def graph_nodes(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not node_ids:
            return {}
        with self._Session() as s:
            rows = s.query(GraphNodeRow).filter(GraphNodeRow.id.in_(node_ids)).all()
        return {r.id: {"id": r.id, "type": r.type, "name": r.name, "contact_id": r.contact_id,
                       "data": json.loads(r.data_json) if r.data_json else None} for r in rows}

    def recent_contacts(self, limit: int = 500) -> List[Contact]:
        """Remembered prospects, most recently seen first (for brain refresh)."""
        with self._Session() as s:
            rows = s.query(ContactRow).order_by(ContactRow.updated_at.desc()).limit(limit).all()
        out = []
        for r in rows:
            try:
                out.append(Contact.model_validate_json(r.data_json))
            except ValueError:
                continue
        return out

    def contact(self, contact_id: str) -> Optional[Contact]:
        with self._Session() as s:
            row = s.get(ContactRow, contact_id)
        try:
            return Contact.model_validate_json(row.data_json) if row and row.data_json else None
        except ValueError:
            return None

    # ── email conventions (reach) ────────────────────────────────────────────
    def record_patterns(self, votes: Dict[str, Dict[str, int]]) -> int:
        now = datetime.utcnow()
        with self._Session.begin() as s:
            for domain, counts in votes.items():
                for pattern, n in counts.items():
                    row = s.get(DomainPatternRow, (domain, pattern)) or DomainPatternRow(domain=domain, pattern=pattern, support=0)
                    row.support = (row.support or 0) + int(n)
                    row.last_confirmed = now
                    s.merge(row)
        return sum(len(v) for v in votes.values())

    def domain_pattern(self, domain: str) -> Optional[tuple]:
        """(pattern, support) of the best-supported convention for a domain, if any."""
        with self._Session() as s:
            rows = s.query(DomainPatternRow).filter(DomainPatternRow.domain == domain).all()
        if not rows:
            return None
        best = max(rows, key=lambda r: r.support or 0)
        total = sum(r.support or 0 for r in rows)
        if (best.support or 0) * 2 <= total:  # no clear convention
            return None
        return best.pattern, int(best.support or 0)

    # ── the user's own network (warm paths) ──────────────────────────────────
    def network_add(self, people: List[Dict[str, Any]]) -> int:
        from brain.models import stable_id

        n = 0
        with self._Session.begin() as s:
            for p in people:
                if not p.get("name") and not p.get("email"):
                    continue
                pid = stable_id("net", (p.get("email") or "").lower() or (p.get("name") or "").lower(), p.get("source") or "")
                s.merge(NetworkRow(id=pid, name=(p.get("name") or "")[:200], email=(p.get("email") or "").lower() or None,
                                   company=(p.get("company") or None), domain=(p.get("domain") or None),
                                   title=(p.get("title") or None), source=p.get("source") or "csv",
                                   connected_on=p.get("connected_on") or None))
                n += 1
        return n

    def network(self) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = s.query(NetworkRow).all()
        return [{"name": r.name, "email": r.email, "company": r.company, "domain": r.domain, "title": r.title,
                 "source": r.source, "connected_on": r.connected_on} for r in rows]

    def network_size(self) -> int:
        with self._Session() as s:
            return s.query(NetworkRow).count()

    def network_clear(self) -> int:
        with self._Session.begin() as s:
            return s.query(NetworkRow).delete()

    def save_followup(self, campaign_id: str, contact_id: str, n: int, draft: Draft) -> None:
        with self._Session.begin() as s:
            s.merge(FollowupRow(campaign_id=campaign_id, contact_id=contact_id, n=n, subject=draft.subject,
                                body=draft.body, issues_json=json.dumps(draft.issues)))

    def followups(self, campaign_id: str) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = s.query(FollowupRow).filter(FollowupRow.campaign_id == campaign_id).order_by(FollowupRow.contact_id, FollowupRow.n).all()
        return [{"contact_id": r.contact_id, "n": r.n, "subject": r.subject, "body": r.body,
                 "issues": json.loads(r.issues_json or "[]")} for r in rows]

    # ── the user ─────────────────────────────────────────────────────────────
    PROFILE_KEYS = ("name", "signature", "postal_address", "tone", "exclusions", "language", "typical_asks")

    def user_profile(self, user: str = "default") -> Dict[str, Any]:
        with self._Session() as s:
            row = s.get(UserProfileRow, user)
        return json.loads(row.data_json) if row and row.data_json else {}

    def save_user_profile(self, data: Dict[str, Any], user: str = "default") -> Dict[str, Any]:
        cur = self.user_profile(user)
        for k, v in data.items():
            if k in self.PROFILE_KEYS and v not in (None, "", []):
                if k in ("exclusions", "typical_asks"):
                    v = list(dict.fromkeys(list(cur.get(k) or []) + list(v)))[-30:]
                cur[k] = v
        with self._Session.begin() as s:
            s.merge(UserProfileRow(id=user, data_json=json.dumps(cur)))
        return cur

    # ── draft edits (the writer learns the user's style) ─────────────────────
    def record_edit(self, campaign_id: str, contact_id: str, body: str) -> None:
        """The user edited a draft before sending: keep the edited text as a style example."""
        prof = self.user_profile()
        edits = list(prof.get("_edited_emails") or [])
        edits.append(body[:3000])
        prof["_edited_emails"] = edits[-3:]
        with self._Session.begin() as s:
            s.merge(UserProfileRow(id="default", data_json=json.dumps(prof)))
            row = s.get(ResultRow, (campaign_id, contact_id))
            if row is not None:
                row.body = body
                row.issues_json = json.dumps([])
                row.status = "approved"

    def style_examples(self) -> List[str]:
        return list(self.user_profile().get("_edited_emails") or [])

    def labels_from_review(self, campaign_id: str) -> Dict[str, List[str]]:
        """Human decisions and real outcomes as evaluation labels: approved/sent/replied are good, rejected is bad, and a
        bounce is `wrong_address` (right person or not, the address was not theirs)."""
        good, bad, wrong = [], [], []
        for r in self.results(campaign_id):
            if r["status"] == "bounced":
                wrong.append(r["email"] or r["name"])
            elif r["status"] in ("approved", "sent", "replied", "followed_up"):
                good.append(r["email"] or r["name"])
            elif r["status"] == "rejected":
                bad.append(r["email"] or r["name"])
        return {"good": good, "bad": bad, "wrong_address": wrong}

    # ── real outcomes (the engine's only ground truth) ───────────────────────
    def record_outcome(self, campaign_id: str, who: str, kind: str, detail: str = "") -> Optional[Dict[str, Any]]:
        """Record what happened to the email sent to `who` (a contact id or an address) in this campaign, and learn:
          bounced       suppress the address; if it was a guess, demote the email convention that produced it
          unsubscribed  suppress the address for good;  spam: suppress it too
          replied / positive   a reply: mark the result and label the prospect good for the fit model
        Returns what was learned, or None when no such result exists."""
        from brain.discovery.reach.tiers import email_tier
        from brain.emailfinder import _patterns

        if kind not in OUTCOME_KINDS:
            raise ValueError(f"outcome must be one of {OUTCOME_KINDS}")
        who = (who or "").strip()
        with self._Session() as s:
            row = (s.query(ResultRow).filter(ResultRow.campaign_id == campaign_id)
                   .filter((ResultRow.contact_id == who) | (ResultRow.email == who.lower())).first())
            crow = s.get(ContactRow, row.contact_id) if row else None
            camp = s.get(CampaignRow, campaign_id)
        if row is None:
            return None
        address = (row.email or "").lower()
        contact = Contact.model_validate_json(crow.data_json) if crow and crow.data_json else None
        cand = next((e for e in contact.emails if e.address == address), None) if contact else None
        tier = email_tier(contact, cand) if contact and cand else ""
        learned: Dict[str, Any] = {"address": address, "kind": kind, "tier": tier, "method": cand.method if cand else ""}
        with self._Session.begin() as s:
            s.add(OutcomeRow(campaign_id=campaign_id, contact_id=row.contact_id, address=address, kind=kind,
                             method=learned["method"], tier=tier, detail=detail[:500]))
            res = s.get(ResultRow, (campaign_id, row.contact_id))
            if res is not None:
                if kind in ("sent", "bounced"):
                    res.status = kind
                elif kind in ("replied", "positive"):
                    res.status = "replied"
        state = {"sent": touch.SENT, "bounced": touch.BOUNCED, "replied": touch.REPLIED, "positive": touch.REPLIED,
                 "negative": touch.REPLIED, "unsubscribed": touch.UNSUBSCRIBED, "spam": touch.SPAM}[kind]
        self.set_touch(campaign_id, row.contact_id, state)  # the ledger learns it: nobody emailed is offered again
        if kind == "bounced":
            self.check_series_health(self.series_id_of(campaign_id))
        if kind in ("bounced", "unsubscribed", "spam") and address:
            self.suppress(address, kind)
            learned["suppressed"] = True
        if kind == "bounced" and contact and cand and "@" in address:
            local, domain = address.split("@", 1)
            for pname, value in _patterns(contact.name).items():
                if value == local:
                    learned["pattern_demoted"] = self._demote_pattern(domain, pname)
                    break
        if kind in ("replied", "positive") and contact and camp:
            spec = CampaignSpec.model_validate_json(camp.spec_json)
            self.label_fit_examples(spec.pack, good=[contact.name], bad=[])
            learned["labelled_good"] = True
        return learned

    def _demote_pattern(self, domain: str, pattern: str, by: int = 2) -> bool:
        """A convention that produced a bounce is less trustworthy: lower its support, forget it at zero."""
        with self._Session.begin() as s:
            row = s.get(DomainPatternRow, (domain, pattern))
            if row is None:
                return False
            row.support = (row.support or 0) - by
            if row.support <= 0:
                s.delete(row)
        return True

    def outcome_summary(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """Bounce and reply rates, overall and by how the address was found and how far it was trusted when sent. This
        is how "attributed" and "verified" get tested against reality."""
        with self._Session() as s:
            q = s.query(OutcomeRow)
            if campaign_id:
                q = q.filter(OutcomeRow.campaign_id == campaign_id)
            rows = q.all()
        sent = {(r.campaign_id, r.contact_id) for r in rows if r.kind == "sent"}

        def bucket(key) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                k = key(r) or "unknown"
                b = out.setdefault(k, {"sent": 0, "bounced": 0, "replied": 0, "unsubscribed": 0})
                if r.kind == "sent":
                    b["sent"] += 1
                elif r.kind in ("bounced", "unsubscribed"):
                    b[r.kind] += 1
                elif r.kind in ("replied", "positive"):
                    b["replied"] += 1
            for b in out.values():
                n = b["sent"] or 0
                b["bounce_rate"] = round(b["bounced"] / n, 3) if n else None
                b["reply_rate"] = round(b["replied"] / n, 3) if n else None
            return out

        total = bucket(lambda r: "all")["all"] if rows else {"sent": 0, "bounced": 0, "replied": 0}
        return {"sent": len(sent), "total": total, "by_tier": bucket(lambda r: r.tier), "by_method": bucket(lambda r: r.method)}

    def results_by_email(self, address: str) -> List[tuple]:
        """(campaign_id, contact_id) of every result addressed to this email."""
        with self._Session() as s:
            rows = s.query(ResultRow.campaign_id, ResultRow.contact_id).filter(ResultRow.email == address.lower()).all()
        return [(r[0], r[1]) for r in rows]

    def has_outcome(self, campaign_id: str, contact_id: str, kind: str) -> bool:
        with self._Session() as s:
            return s.query(OutcomeRow).filter(OutcomeRow.campaign_id == campaign_id, OutcomeRow.contact_id == contact_id,
                                              OutcomeRow.kind == kind).first() is not None

    # ── do-not-contact ───────────────────────────────────────────────────────
    def suppress(self, value: str, reason: str = "manual") -> None:
        with self._Session.begin() as s:
            s.merge(SuppressionRow(value=value.strip().lower(), reason=reason))

    def blocked_reason(self, c: Contact, scope: Optional[touch.Scope] = None) -> Optional[str]:
        """Why this contact must not be emailed (or shown again), or None. Checked for every address and identity key
        we hold. With a `scope` the touch ledger applies the campaign's no-repeat mode; without one, only hard facts
        block: suppression, anyone already emailed, dead addresses, imported do-not-contact entries."""
        addresses = [e.address for e in c.emails]
        domains = {a.split("@")[1] for a in addresses} | ({c.org_domain} if c.org_domain else set())
        candidates = set(addresses) | domains | {registrable_domain(d) for d in domains}
        with self._Session() as s:
            if candidates and s.query(SuppressionRow).filter(SuppressionRow.value.in_(candidates)).first():
                return "suppressed"
            if addresses and (scope is None or scope.mode != "cooldown"):
                if s.query(ResultRow).filter(ResultRow.email.in_(addresses), ResultRow.status.in_(CONTACTED)).first():
                    return "already_contacted"
                if self._has_legacy_emails:
                    stmt = text("SELECT 1 FROM emails WHERE status IN ('sent','replied','followed_up') "
                                "AND lower(recipient_email) = :e LIMIT 1")
                    for a in addresses:
                        if s.execute(stmt, {"e": a}).first():
                            return "already_contacted"
            return self._ledger_reason(s, c, scope)

    def _ledger_reason(self, s, c: Contact, scope: Optional[touch.Scope]) -> Optional[str]:
        """The strongest reason an earlier touch blocks `c`: any identity key matches, or (same organisation) a name
        that is plausibly the same person."""
        rows = s.query(TouchRow).filter(TouchRow.key.in_(touch.lookup_keys(c))).all()
        org = touch.org_key(c)
        if c.kind != "org" and org and not rows:
            rows = [r for r in s.query(TouchRow).filter(TouchRow.org_key == org, TouchRow.kind == "person").all()
                    if touch.loose_names_compatible(c.name, r.name or "")]
        found = []
        for r in rows:
            if scope is not None and scope.campaign_id and r.campaign_id == scope.campaign_id:
                continue
            why = touch.row_blocks(r.state, r.series_id, r.updated_at, scope)
            if why:
                found.append(why)
        for why in ("do_not_contact", "already_contacted", "rejected_earlier", "on_standby", "already_surfaced"):
            if why in found:
                return why
        return None

    def is_suppressed(self, address: str) -> bool:
        """Only the hard stop: this address or its domain asked not to be written to, bounced, or was imported as
        do-not-contact. Unlike `address_blocked` it ignores "already emailed", so it is the check for a follow-up."""
        addr = (address or "").strip().lower()
        if "@" not in addr:
            return False
        domain = addr.split("@", 1)[1]
        with self._Session() as s:
            return s.query(SuppressionRow).filter(SuppressionRow.value.in_({addr, domain, registrable_domain(domain)})).first() is not None

    def address_blocked(self, address: str, campaign_id: str = "") -> Optional[str]:
        """The send gate: may an email go to this address right now? (suppressed, already emailed, dead address, imported
        do-not-contact, or the series that holds this draft is paused because its bounce rate is too high)."""
        addr = (address or "").strip().lower()
        if not addr or "@" not in addr:
            return "no_address"
        c = Contact(name="x", emails=[EmailCandidate(address=addr, confidence=1.0)])
        c.kind = "org"  # only the address matters here: no name or organisation rules
        row = self.get_series(self.series_id_of(campaign_id)) if campaign_id else None
        # In cooldown mode an earlier send expires after N days, so the gate must use the series' scope, or a person offered
        # again after the cooldown could never be sent to. Other modes: any earlier send blocks (unscoped).
        scope = (touch.Scope(row["id"], "cooldown", row["cooldown_days"] or 90, campaign_id=campaign_id)
                 if row and row["no_repeat"] == "cooldown" else None)
        why = self.blocked_reason(c, scope=scope)
        if why:
            return why
        if row and row["status"] == "paused":
            return f"series_paused: {row['pause_reason']}"
        return None

    # ── series (one brief, many batches) ─────────────────────────────────────
    def series_id_of(self, campaign_id: str) -> str:
        with self._Session() as s:
            m = s.get(SeriesMemberRow, campaign_id)
            return m.series_id if m else campaign_id

    def ensure_series(self, spec: CampaignSpec, rung: int = 0) -> touch.Scope:
        """Register this campaign as a batch of its series (creating the series for a first batch) and return the scope
        that decides what counts as a repeat. The series' stored settings win over the spec's."""
        sid = spec.series_id or spec.id
        with self._Session.begin() as s:
            row = s.get(SeriesRow, sid)
            if row is None:
                if spec.no_repeat not in touch.MODES or spec.recycle not in touch.RECYCLE:
                    raise ValueError(f"no_repeat must be one of {touch.MODES} and recycle one of {touch.RECYCLE}")
                row = SeriesRow(id=sid, no_repeat=spec.no_repeat, cooldown_days=spec.cooldown_days, recycle=spec.recycle)
                s.add(row)
            if s.get(SeriesMemberRow, spec.id) is None:
                n = s.query(SeriesMemberRow).filter(SeriesMemberRow.series_id == sid).count()
                s.add(SeriesMemberRow(campaign_id=spec.id, series_id=sid, batch_no=n + 1, rung=rung))
            return touch.Scope(sid, row.no_repeat or "series", row.cooldown_days or 90, campaign_id=spec.id)

    def scope_for(self, spec: CampaignSpec) -> touch.Scope:
        sid = spec.series_id or spec.id
        with self._Session() as s:
            row = s.get(SeriesRow, sid)
            if row is not None:
                return touch.Scope(sid, row.no_repeat or "series", row.cooldown_days or 90, campaign_id=spec.id)
        return touch.Scope(sid, spec.no_repeat, spec.cooldown_days, campaign_id=spec.id)

    def get_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        with self._Session() as s:
            r = s.get(SeriesRow, series_id)
            if r is None:
                return None
            return {"id": r.id, "status": r.status or "active", "pause_reason": r.pause_reason or "",
                    "no_repeat": r.no_repeat, "cooldown_days": r.cooldown_days, "recycle": r.recycle,
                    "widen": r.widen or "", "rung": r.rung or 0, "created_at": r.created_at,
                    "health_since": r.health_since}

    def update_series(self, series_id: str, **fields: Any) -> None:
        allowed = {"status", "pause_reason", "no_repeat", "cooldown_days", "recycle", "widen", "rung", "health_since"}
        with self._Session.begin() as s:
            row = s.get(SeriesRow, series_id)
            if row is None:
                raise KeyError(f"no series {series_id}")
            for k, v in fields.items():
                if k not in allowed:
                    raise ValueError(f"cannot set {k}")
                setattr(row, k, v)

    def list_series(self) -> List[Dict[str, Any]]:
        with self._Session() as s:
            rows = s.query(SeriesRow).order_by(SeriesRow.created_at.desc()).all()
            out = []
            for r in rows:
                members = s.query(SeriesMemberRow).filter(SeriesMemberRow.series_id == r.id).count()
                first = s.get(CampaignRow, r.id)
                obj = json.loads(first.spec_json).get("objective", "") if first and first.spec_json else ""
                out.append({"id": r.id, "status": r.status or "active", "batches": members, "mode": r.no_repeat,
                            "rung": r.rung or 0, "objective": obj, "created_at": r.created_at})
            return out

    def series_campaigns(self, series_id: str) -> List[str]:
        with self._Session() as s:
            rows = (s.query(SeriesMemberRow).filter(SeriesMemberRow.series_id == series_id)
                    .order_by(SeriesMemberRow.batch_no).all())
            ids = [r.campaign_id for r in rows]
        return ids or ([series_id] if self.load_campaign(series_id) else [])

    # ── the touch ledger ─────────────────────────────────────────────────────
    def _upsert_touch(self, s, c: Contact, campaign_id: str, series_id: str, state: str, force: bool = False) -> None:
        kind = "org" if c.kind == "org" else "person"
        org = touch.org_key(c)
        for k in touch.keys_for(c):
            row = s.get(TouchRow, k)
            if row is None:
                s.add(TouchRow(key=k, contact_id=c.id, campaign_id=campaign_id, series_id=series_id, state=state,
                               kind=kind, name=c.name[:300], org_key=org, updated_at=datetime.utcnow()))
                continue
            if row.campaign_id != campaign_id and row.state in touch.CONTACTED | {touch.REJECTED, touch.EXTERNAL}:
                continue  # another batch's decision or send stands: never overwritten by a re-shortlisting
            same = row.campaign_id == campaign_id
            row.state = touch.advance(row.state, state) if same and not force else state
            row.contact_id, row.campaign_id, row.series_id = c.id, campaign_id, series_id
            row.kind, row.name, row.org_key, row.updated_at = kind, c.name[:300], org, datetime.utcnow()

    def claim_batch(self, campaign_id: str, scope: touch.Scope, contacts: List[Contact],
                    state: str = touch.SURFACED) -> tuple:
        """Take these contacts for this batch, atomically: (kept, lost). A contact is lost when an earlier touch blocks it
        in this scope, or when another batch running at the same time claimed it first. Idempotent for the same campaign
        (a resumed run claims the same people again)."""
        from sqlalchemy.exc import IntegrityError

        kept: List[Contact] = []
        lost: List[Contact] = []
        for c in contacts:  # one small transaction each: a lost race for one person must not undo the others
            try:
                with self._Session.begin() as s:
                    if self._ledger_reason(s, c, scope):
                        lost.append(c)
                        continue
                    self._upsert_touch(s, c, campaign_id, scope.series_id, state, force=True)
                kept.append(c)
            except IntegrityError:  # a concurrent batch inserted the same key between our check and our insert
                lost.append(c)
        return kept, lost

    def set_touch(self, campaign_id: str, contact_id: str, state: str, force: bool = False) -> int:
        """Move a contact of this campaign to a new state (never backwards, unless `force`). Returns rows changed."""
        n = 0
        with self._Session.begin() as s:
            # By contact, not by campaign: the ledger keeps one row per identity, and in another series the latest batch
            # to touch a person owns the row. A send or a bounce recorded from this campaign must still reach it.
            for r in s.query(TouchRow).filter(TouchRow.contact_id == contact_id).all():
                new = state if force else touch.advance(r.state, state)
                if new != r.state:
                    r.state, r.updated_at = new, datetime.utcnow()
                    n += 1
        return n

    def import_external(self, values: List[str], reason: str = "imported") -> int:
        """Do-not-contact / "already emailed" entries (addresses, domains or full names): never offered, in any mode."""
        n = 0
        with self._Session.begin() as s:
            for raw in values:
                v = (raw or "").strip().lower()
                if not v or v.startswith("#"):
                    continue
                if "@" in v:
                    key = f"email:{touch.norm_email(v)}"
                elif "." in v and " " not in v:
                    key = f"org:{registrable_domain(v)}"
                else:
                    toks = touch.name_tokens(v)
                    if len(toks) < 2:
                        continue
                    key = f"name:{' '.join(toks)}"
                existing = s.get(TouchRow, key[:250])
                if existing is None:
                    s.add(TouchRow(key=key[:250], contact_id="", campaign_id="", series_id="", state=touch.EXTERNAL,
                                   kind="person", name=v[:300], org_key="", updated_at=datetime.utcnow()))
                elif existing.state not in touch.CONTACTED | {touch.EXTERNAL}:
                    existing.state, existing.updated_at = touch.EXTERNAL, datetime.utcnow()  # keeps its campaign link
                n += 1
        return n

    def org_counts(self, series_id: str) -> Dict[str, int]:
        """How many people in each organisation this series has already put forward (counts across batches)."""
        out: Dict[str, set] = {}
        with self._Session() as s:
            for r in s.query(TouchRow).filter(TouchRow.series_id == series_id, TouchRow.org_key != "",
                                              TouchRow.state.in_(touch.COUNTS_FOR_ORG_CAP)).all():
                out.setdefault(r.org_key, set()).add(r.contact_id)
        return {k: len(v) for k, v in out.items()}

    def seen(self, series_id: Optional[str] = None, state: Optional[str] = None) -> List[Dict[str, Any]]:
        """Distinct contacts in the ledger (one per contact id), optionally for one series or state."""
        with self._Session() as s:
            q = s.query(TouchRow)
            if series_id:
                q = q.filter(TouchRow.series_id == series_id)
            if state:
                q = q.filter(TouchRow.state == state)
            rows = q.order_by(TouchRow.updated_at.desc()).all()
        out, done = [], set()
        for r in rows:
            ident = r.contact_id or r.key
            if ident in done:
                continue
            done.add(ident)
            out.append({"contact_id": r.contact_id, "name": r.name, "org": r.org_key, "state": r.state,
                        "campaign_id": r.campaign_id, "series_id": r.series_id, "updated_at": r.updated_at})
        return out

    def result_status(self, campaign_id: str, contact_id: str) -> Optional[str]:
        with self._Session() as s:
            row = s.get(ResultRow, (campaign_id, contact_id))
            return row.status if row else None

    def _result_for(self, s, who: str, campaign_id: Optional[str] = None):
        q = s.query(ResultRow).filter((ResultRow.contact_id == who) | (ResultRow.email == who.lower()))
        if campaign_id:
            q = q.filter(ResultRow.campaign_id == campaign_id)
        rows = [r for r in q.order_by(ResultRow.updated_at.desc()).all() if r.status != "recycled"]
        return rows[0] if rows else None

    def decide(self, who: str, status: str, campaign_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """`brain standby` / `brain reject`: record a decision on a result by contact id or address. Sent results cannot
        be changed. Returns {contact_id, campaign_id, was} or None."""
        with self._Session() as s:
            row = self._result_for(s, who, campaign_id)
            if row is None or row.status in CONTACTED:
                return None
            out = {"contact_id": row.contact_id, "campaign_id": row.campaign_id, "was": row.status}
        self.set_status(out["campaign_id"], out["contact_id"], status)
        return out

    def revive(self, who: str, campaign_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """`release` (a rejected prospect) and `recall` (one on standby): back to undecided, with the draft shown in the
        review again. `who` is a contact id or an address. Returns what was done, or None when nobody matches."""
        with self._Session.begin() as s:
            row = self._result_for(s, who, campaign_id)
            if row is None or row.status not in ("rejected", "standby"):
                return None
            was = row.status
            row.status = "drafted" if row.body and not json.loads(row.issues_json or "[]") else (
                "needs_review" if row.body else "shortlisted")
            cid, camp = row.contact_id, row.campaign_id
            for r in s.query(TouchRow).filter(TouchRow.campaign_id == camp, TouchRow.contact_id == cid).all():
                r.state, r.updated_at = (touch.DRAFTED if row.body else touch.SURFACED), datetime.utcnow()
        return {"contact_id": cid, "campaign_id": camp, "was": was}

    # ── recycling: undecided and standby people, offered again when new matches are few ──
    def soft_pool(self, scope: touch.Scope, exclude_campaign: Any = "") -> List[Dict[str, Any]]:
        """Results of this series that passed the fit floor but were never decided on (shortlisted / drafted /
        needs_review) or were put on standby, best score first. Rejected and sent people are never here."""
        with self._Session() as s:
            members = [m.campaign_id for m in s.query(SeriesMemberRow).filter(SeriesMemberRow.series_id == scope.series_id)]
            skip = {exclude_campaign} if isinstance(exclude_campaign, str) else set(exclude_campaign or ())
            members = [m for m in (members or [scope.series_id]) if m not in skip]
            rows = (s.query(ResultRow, ContactRow).join(ContactRow, ContactRow.id == ResultRow.contact_id)
                    .filter(ResultRow.campaign_id.in_(members),
                            ResultRow.status.in_(("shortlisted", "drafted", "needs_review", "standby")))
                    .order_by(ResultRow.score.desc()).all())
            out, seen_ids = [], set()
            for r, c in rows:
                if r.contact_id in seen_ids:
                    continue
                seen_ids.add(r.contact_id)
                out.append({"campaign_id": r.campaign_id, "contact_id": r.contact_id, "status": r.status, "email": r.email,
                            "score": r.score, "prescore": r.prescore, "reasons": json.loads(r.reasons_json or "[]"),
                            "hook": r.hook, "subject": r.subject, "body": r.body,
                            "issues": json.loads(r.issues_json or "[]"), "contact": Contact.model_validate_json(c.data_json)})
        return out

    def recycle_into(self, new_campaign: str, scope: touch.Scope, rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Move earlier results into this campaign: one result row, the stored score and draft kept (nothing is judged or
        written again), the old row marked `recycled` so it is not shown twice. Returns [{contact_id, from}]."""
        moved = []
        with self._Session.begin() as s:
            for r in rows:
                old = s.get(ResultRow, (r["campaign_id"], r["contact_id"]))
                if old is None or old.status not in ("shortlisted", "drafted", "needs_review", "standby"):
                    continue
                issues = json.loads(old.issues_json or "[]")
                status = ("needs_review" if issues else "drafted") if old.body else "shortlisted"
                s.merge(ResultRow(campaign_id=new_campaign, contact_id=old.contact_id, status=status, email=old.email,
                                  prescore=old.prescore, score=old.score, reasons_json=old.reasons_json, hook=old.hook,
                                  subject=old.subject, body=old.body, issues_json=old.issues_json))
                old.status = "recycled"
                for t in s.query(TouchRow).filter(TouchRow.campaign_id == r["campaign_id"],
                                                  TouchRow.contact_id == old.contact_id).all():
                    t.campaign_id, t.series_id, t.updated_at = new_campaign, scope.series_id, datetime.utcnow()
                    t.state = touch.DRAFTED if old.body else touch.SURFACED
                moved.append({"contact_id": old.contact_id, "from": r["campaign_id"]})
        return moved

    # ── series health: stop sending into a bounce problem ────────────────────
    def resume_series(self, series_id: str) -> None:
        """Lift a pause. The bounce history before now no longer counts, otherwise the series would pause again at once;
        if the problem is not fixed, new bounces pause it again."""
        self.update_series(series_id, status="active", pause_reason="", health_since=datetime.utcnow())

    def series_health(self, series_id: str) -> Dict[str, Any]:
        ids = self.series_campaigns(series_id)
        row = self.get_series(series_id)
        since = row.get("health_since") if row else None
        with self._Session() as s:
            q = s.query(OutcomeRow).filter(OutcomeRow.campaign_id.in_(ids)) if ids else None
            if q is not None and since is not None:
                q = q.filter(OutcomeRow.at >= since)
            rows = q.all() if q is not None else []
        sent = len({(r.campaign_id, r.contact_id) for r in rows if r.kind == "sent"})
        bounced = len({(r.campaign_id, r.contact_id) for r in rows if r.kind == "bounced"})
        return {"sent": sent, "bounced": bounced, "bounce_rate": round(bounced / sent, 3) if sent else None}

    def check_series_health(self, series_id: str) -> Optional[str]:
        """Pause the series when its bounce rate passes BRAIN_BOUNCE_PAUSE (default 5%) once BRAIN_BOUNCE_MIN_SENT (20)
        emails are out: a bounce problem hurts the sender's domain for every later email. Returns the reason if paused."""
        import os

        threshold = float(os.getenv("BRAIN_BOUNCE_PAUSE", "0.05"))
        min_sent = int(os.getenv("BRAIN_BOUNCE_MIN_SENT", "20"))
        h = self.series_health(series_id)
        if h["sent"] >= min_sent and h["bounce_rate"] is not None and h["bounce_rate"] > threshold:
            reason = (f"{h['bounced']} of {h['sent']} emails bounced ({h['bounce_rate']:.0%}, limit {threshold:.0%}). "
                      f"Check the address sources, then `brain series-resume {series_id}`.")
            if self.get_series(series_id):
                self.update_series(series_id, status="paused", pause_reason=reason)
            return reason
        return None

    def sent_since(self, hours: int = 24) -> int:
        """Emails sent from the review UI in the last `hours` (the daily cap survives restarts because it is counted from
        the database, not from memory)."""
        if not self._has_legacy_emails:
            return 0
        since = datetime.utcnow() - timedelta(hours=hours)
        with self._Session() as s:
            return int(s.execute(text("SELECT count(*) FROM emails WHERE sent_at IS NOT NULL AND sent_at >= :t"),
                                 {"t": since}).scalar() or 0)

    def last_sent_at(self) -> Optional[datetime]:
        if not self._has_legacy_emails:
            return None
        with self._Session() as s:
            v = s.execute(text("SELECT max(sent_at) FROM emails")).scalar()
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v)
            except ValueError:
                return None
        return v

    def _backfill_touch(self) -> None:
        """Results saved before the ledger existed become ledger rows once, so old shortlists and sends count."""
        status_state = {"shortlisted": touch.SURFACED, "drafted": touch.DRAFTED, "needs_review": touch.DRAFTED,
                        "approved": touch.APPROVED, "sent": touch.SENT, "followed_up": touch.SENT,
                        "replied": touch.REPLIED, "bounced": touch.BOUNCED, "rejected": touch.REJECTED}
        with self._Session.begin() as s:
            if s.query(TouchRow).first() is not None or s.query(ResultRow).first() is None:
                return
            rows = s.query(ResultRow, ContactRow).join(ContactRow, ContactRow.id == ResultRow.contact_id).all()
            for r, crow in rows:
                state = status_state.get(r.status)
                if not state or not crow.data_json:
                    continue
                try:
                    c = Contact.model_validate_json(crow.data_json)
                except ValueError:
                    continue
                c.id = r.contact_id
                sid = (s.get(SeriesMemberRow, r.campaign_id).series_id if s.get(SeriesMemberRow, r.campaign_id)
                       else r.campaign_id)
                self._upsert_touch(s, c, r.campaign_id, sid, state)

    # ── results ──────────────────────────────────────────────────────────────
    def save_results(self, campaign_id: str, contacts: List[Contact], drafts: Dict[str, Draft]) -> None:
        with self._Session.begin() as s:
            m = s.get(SeriesMemberRow, campaign_id)
            sid = m.series_id if m else campaign_id
            for c in contacts:
                d = drafts.get(c.id)
                best = c.best_email
                status = "shortlisted" if d is None else ("drafted" if d.ok else "needs_review")
                existing = s.get(ResultRow, (campaign_id, c.id))
                if existing and existing.status in CONTACTED + ("approved", "rejected", "standby"):
                    continue  # never overwrite a human decision or a send
                s.merge(ResultRow(
                    campaign_id=campaign_id, contact_id=c.id, status=status, email=best.address if best else None,
                    prescore=c.prescore, score=c.score, reasons_json=json.dumps(c.reasons), hook=c.hook,
                    subject=d.subject if d else None, body=d.body if d else None,
                    issues_json=json.dumps(d.issues if d else []),
                ))
                self._upsert_touch(s, c, campaign_id, sid, touch.SURFACED if d is None else touch.DRAFTED)

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

    _STATUS_STATE = {"shortlisted": touch.SURFACED, "drafted": touch.DRAFTED, "needs_review": touch.DRAFTED,
                     "approved": touch.APPROVED, "sent": touch.SENT, "followed_up": touch.SENT, "replied": touch.REPLIED,
                     "bounced": touch.BOUNCED, "rejected": touch.REJECTED, "standby": touch.STANDBY}

    def set_status(self, campaign_id: str, contact_id: str, status: str) -> None:
        with self._Session.begin() as s:
            row = s.get(ResultRow, (campaign_id, contact_id))
            if row:
                row.status = status
        state = self._STATUS_STATE.get(status)
        if row and state:
            self.set_touch(campaign_id, contact_id, state)

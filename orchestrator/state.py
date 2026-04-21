"""
orchestrator/state.py

Defines all shared Pydantic models and the central AgentState TypedDict
that flows through the LangGraph execution graph.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import config as _app_config

from pydantic import BaseModel, Field
from typing import TypedDict


# ── Enums ─────────────────────────────────────────────────────────────────────

class AgentMode(str, Enum):
    CLARIFICATION = "clarification"
    PLANNING       = "planning"
    CRAWLING       = "crawling"
    SCORING        = "scoring"
    EMAIL_DRAFT    = "email_draft"
    REVIEW         = "review"
    SENDING        = "sending"
    AUTONOMOUS     = "autonomous"
    DONE           = "done"


class EmailStatus(str, Enum):
    DRAFT      = "draft"
    APPROVED   = "approved"
    REJECTED   = "rejected"
    SENT       = "sent"
    REPLIED    = "replied"
    FOLLOWED_UP = "followed_up"


class OutreachGoal(str, Enum):
    RESEARCH_INTERNSHIP  = "research_internship"
    PHDINQUIRY           = "phd_inquiry"
    COLLABORATION        = "collaboration"
    INDUSTRY_OUTREACH    = "industry_outreach"
    POSTDOC              = "postdoc"
    OTHER                = "other"


# ── Core Data Models ──────────────────────────────────────────────────────────

class Publication(BaseModel):
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    abstract_snippet: Optional[str] = None


class PublicationMatch(BaseModel):
    title: str
    reason: str


class RelevanceBreakdown(BaseModel):
    keyword_overlap: float = 0.0
    embedding_sim: float = 0.0
    recency: float = 0.0
    geographic: float = 0.0
    accepting_students: float = 0.0
    weights_used: dict[str, float] = Field(default_factory=dict)
    matched_terms: list[str] = Field(default_factory=list)
    publication_matches: list[PublicationMatch] = Field(default_factory=list)


class ProfessorProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    email: Optional[str] = None
    institution: str
    department: Optional[str] = None
    lab_url: Optional[str] = None
    profile_url: Optional[str] = None
    research_keywords: list[str] = Field(default_factory=list)
    recent_publications: list[Publication] = Field(default_factory=list)
    accepting_students: Optional[bool] = None
    bio_snippet: Optional[str] = None
    raw_page_text: Optional[str] = None
    relevance_score: float = 0.0
    relevance_breakdown: Optional[RelevanceBreakdown] = None
    identity_key: Optional[str] = None
    orcid: Optional[str] = None
    email_source: Optional[str] = None
    email_confidence: Optional[float] = None
    crawled_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DraftVariant(BaseModel):
    label: str = "A"
    subject: str = ""
    body: str = ""


class DraftEmail(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    professor_id: str
    recipient_name: str
    recipient_email: str
    subject: str
    body: str
    personalization_notes: str = ""
    status: EmailStatus = EmailStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    reply_received_at: Optional[datetime] = None
    follow_up_count: int = 0
    variants: list[DraftVariant] = Field(default_factory=list)
    selected_variant_index: int = 0

    def apply_selected_variant(self) -> "DraftEmail":
        """Copy subject/body from the chosen variant (no-op if variants empty)."""
        if not self.variants:
            return self
        i = max(0, min(self.selected_variant_index, len(self.variants) - 1))
        v = self.variants[i]
        return self.model_copy(
            update={"subject": v.subject, "body": v.body, "selected_variant_index": i}
        )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── User Intent (Slot-filling model) ─────────────────────────────────────────

class UserIntent(BaseModel):
    """
    Structured representation of the user's goal, progressively filled
    through the clarification dialogue.
    """
    goal: OutreachGoal = OutreachGoal.RESEARCH_INTERNSHIP
    research_domains: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    target_institutions: list[str] = Field(default_factory=list)
    timeline: Optional[str] = None           # e.g. "Summer 2025"
    user_name: Optional[str] = None
    user_profile_summary: Optional[str] = None
    cv_path: Optional[str] = None
    # Structured signals from CV (used to personalize cold emails)
    cv_degree: Optional[str] = None
    cv_institution: Optional[str] = None
    cv_institution_tag: Optional[str] = None  # e.g. IIT, NIT, Russell Group
    cv_cgpa: Optional[str] = None
    cv_rank: Optional[str] = None
    cv_achievements: list[str] = Field(default_factory=list)
    cv_key_projects: list[str] = Field(default_factory=list)
    cv_publications_list: list[str] = Field(default_factory=list)
    cv_skills: list[str] = Field(default_factory=list)
    # Curated for research-internship outreach (LLM-selected from CV; not a resume dump)
    research_internship_highlights: list[str] = Field(default_factory=list)
    cv_research_interests: list[str] = Field(default_factory=list)
    cv_research_experience: list[str] = Field(default_factory=list)
    cv_work_research_adjacent: list[str] = Field(default_factory=list)
    email_tone: str = "professional"          # professional / friendly / concise
    follow_up_days: int = 7
    outreach_per_week: int = 10
    extra_notes: Optional[str] = None        # Any other user instructions
    confirmed: bool = False

    def missing_required_slots(self) -> list[str]:
        """Returns list of slot names that are still empty."""
        missing = []
        if not self.research_domains:
            missing.append("research_domains")
        if not self.countries:
            missing.append("countries")
        if not self.timeline:
            missing.append("timeline")
        if not self.user_name:
            missing.append("user_name")
        if not self.user_profile_summary and not self.cv_path:
            missing.append("user_profile")
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_required_slots()) == 0


# ── Execution Plan ────────────────────────────────────────────────────────────

class SearchStrategy(BaseModel):
    keywords: list[str]
    country_filters: list[str]
    institution_filters: list[str] = Field(default_factory=list)


class FilteringCriteria(BaseModel):
    publication_recency: bool = True
    active_lab_only: bool = True
    min_relevance_score: float = 0.4
    require_email: bool = True


class EmailStrategy(BaseModel):
    personalization_level: str = "high"   # low / medium / high
    reference_publications: bool = True
    tone: str = "professional"
    word_count_target: int = 220
    include_cv_attachment: bool = True
    generate_variants: bool = False


class SendingSchedule(BaseModel):
    emails_per_week: int = 10
    follow_up_days: int = 7
    max_follow_ups: int = 2


class ExecutionPlan(BaseModel):
    search_strategy: SearchStrategy
    filtering_criteria: FilteringCriteria
    email_strategy: EmailStrategy
    sending_schedule: SendingSchedule
    top_k_professors: int = Field(default_factory=lambda: _app_config.TOP_K_PROFESSORS)
    confirmed: bool = False


# ── LangGraph Agent State ─────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Central state object passed between LangGraph nodes.
    All agents read from and write to this dict.
    """
    # Conversation history (list of {role: str, content: str})
    messages: list[dict[str, str]]

    # Current pipeline mode
    mode: str                       # AgentMode value

    # Structured intent, progressively filled
    intent: Optional[dict]          # UserIntent.model_dump()

    # Generated execution plan
    plan: Optional[dict]            # ExecutionPlan.model_dump()

    # Crawled professor data
    professors: list[dict]          # list of ProfessorProfile.model_dump()

    # Shortlisted top-K professors after scoring
    shortlisted: list[dict]

    # Generated draft emails
    draft_emails: list[dict]        # list of DraftEmail.model_dump()

    # Human feedback / latest user message
    human_input: Optional[str]

    # Error info for graceful handling
    error: Optional[str]

    # Session metadata
    session_id: str
    created_at: str

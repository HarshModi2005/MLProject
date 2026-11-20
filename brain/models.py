"""Vertical-agnostic data model: who is sending, what they want, and who they could reach."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SenderProfile(BaseModel):
    """Everything the system may claim about the sender. Drafts may not go beyond this."""

    name: str = ""
    headline: str = ""  # e.g. "Third-year CS undergraduate at X"
    summary: str = ""
    highlights: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    attachment_path: Optional[str] = None
    signature: str = ""  # sign-off block below the name (title, company, phone), from the user's profile
    postal_address: str = ""  # for the opt-out footer of commercial emails, if the user gave one

    def facts_text(self) -> str:
        parts = [self.name, self.headline, self.summary] + self.highlights + self.skills + self.links
        return "\n".join(p for p in parts if p)


class TargetPersona(BaseModel):
    description: str = ""
    roles: List[str] = Field(default_factory=list)  # job titles that qualify
    org_types: List[str] = Field(default_factory=list)  # university, startup, ...
    min_seniority: str = "any"  # any | established


class FitCriterion(BaseModel):
    """One thing that makes a prospect good, and what would prove it.

    `signal` ties the criterion to the discovery channels that can produce that proof
    (see brain/tools.py `proves`): topic, past_sponsorship, hiring, portfolio_fit, recent_coverage,
    publications, audience, spoke_before, location, local_business, funding_focus, open_source.
    """

    name: str
    description: str = ""
    weight: float = 1.0
    signal: str = "topic"
    terms: List[str] = Field(default_factory=list)  # words whose presence in the evidence supports it
    must: bool = False  # a hard requirement: candidates with no support are dropped after judging


class ProspectProfile(BaseModel):
    """What a good prospect looks like for this campaign, confirmed by the user at intake."""

    criteria: List[FitCriterion] = Field(default_factory=list)
    timing_signals: List[FitCriterion] = Field(default_factory=list)  # reasons why *now* is a good moment
    anti_patterns: List[str] = Field(default_factory=list)  # phrases that mark a poor target
    decision_roles: List[str] = Field(default_factory=list)  # who to address inside an organisation, best first

    @property
    def empty(self) -> bool:
        return not self.criteria

    def signals(self) -> List[str]:
        return list(dict.fromkeys(c.signal for c in self.criteria + self.timing_signals if c.signal))


class Ask(BaseModel):
    """What each email asks for, structured so the writer can phrase it right for each situation."""

    type: str = "other"  # call | meeting | reply | apply | intro | review | other
    text: str = ""  # the ask in one sentence (same as CampaignSpec.ask)
    link: Optional[str] = None  # a booking / application / deck link the user gave
    availability: Optional[str] = None  # e.g. "Tue-Thu afternoons IST"
    deadline: Optional[str] = None
    inbox_text: str = ""  # the version for a shared inbox: "could you point me to whoever handles X?"


class Offer(BaseModel):
    """What the sender offers (B2B, sponsorship, fundraising...). Only what the user or their documents state."""

    value_prop: str = ""
    proof_points: List[str] = Field(default_factory=list)  # customers, numbers, results
    terms: str = ""  # price, equity, sponsorship tiers...
    audience: str = ""  # who the offer is for / the audience the sender brings


class Constraints(BaseModel):
    """Hard or soft limits on who to contact, beyond topics and geography."""

    size_min: Optional[int] = None  # employees
    size_max: Optional[int] = None
    stages: List[str] = Field(default_factory=list)  # pre-seed, seed, series a ... / student, mid-career ...
    seniority: str = ""  # any | senior | junior ...
    remote_ok: Optional[bool] = None
    radius_km: Optional[int] = None
    must: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)


class CampaignSpec(BaseModel):
    id: str
    pack: str = "academic"  # see brain/packs.py
    objective: str = ""  # what the sender wants overall
    ask: str = ""  # the concrete request made in the email
    persona: TargetPersona = Field(default_factory=TargetPersona)
    topics: List[str] = Field(default_factory=list)
    expanded_terms: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)  # ISO-3166 alpha-2, upper case; empty = worldwide
    regions: List[str] = Field(default_factory=list)  # free-text sub-regions, e.g. "Punjab"
    organizations: List[str] = Field(default_factory=list)
    exclude_organizations: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    preferred_inboxes: List[str] = Field(default_factory=list)  # e.g. partnerships, marketing (org campaigns)
    lookalike_queries: List[str] = Field(default_factory=list)  # searches for organisations like the user's examples
    event_queries: List[str] = Field(default_factory=list)  # searches for similar past events (sponsors / speakers)
    target_kind: str = ""  # person | org; empty = the playbook's default
    profile: ProspectProfile = Field(default_factory=ProspectProfile)
    confirmed: bool = False  # the user saw and approved this spec
    timeline: Optional[str] = None
    tone: str = "professional"
    target_count: int = 25
    word_target: int = 170
    sender: SenderProfile = Field(default_factory=SenderProfile)
    ask_detail: Ask = Field(default_factory=Ask)
    offer: Offer = Field(default_factory=Offer)
    constraints: Constraints = Field(default_factory=Constraints)
    language: str = ""  # language of the emails; "" = English
    playbook_confidence: Optional[float] = None
    playbook_alternatives: List[str] = Field(default_factory=list)
    # Bulk outreach (brain/touch.py, brain/series.py): batches of one brief share a series and never repeat a person.
    series_id: str = ""  # "" = this campaign is its own series
    no_repeat: str = "series"  # series | workspace | cooldown
    cooldown_days: int = 90  # cooldown mode: earlier touches expire after this many days
    recycle: str = "ask"  # ask | auto | never: offer earlier, undecided people again when new matches are very few

    def query_text(self) -> str:
        """Text the cheap pre-ranker matches candidates against."""
        return " ; ".join(self.topics + self.expanded_terms)


class Evidence(BaseModel):
    """A sourced fact about a contact. Drafts may only reference these."""

    kind: str  # publication | page | profile
    text: str
    url: Optional[str] = None
    year: Optional[int] = None
    detail: Optional[str] = None  # e.g. abstract snippet, venue


class EmailCandidate(BaseModel):
    address: str
    confidence: float = 0.0
    method: str = ""  # mailto | text | directory | orcid | pattern | provided
    source_url: Optional[str] = None
    # Did anything check that the mailbox exists? "" = not checked; else deliverable | undeliverable | catch_all |
    # unknown (brain/discovery/reach/verify.py). `verified_by` names the checker (smtp, hunter, mx).
    deliverability: str = ""
    verified_by: str = ""


class Contact(BaseModel):
    id: str = ""
    kind: str = "person"  # person | org (an organisation reached through its shared inbox)
    name: str
    title: Optional[str] = None
    org: Optional[str] = None
    org_domain: Optional[str] = None
    org_ror: Optional[str] = None
    country: Optional[str] = None  # ISO alpha-2
    location_text: Optional[str] = None  # city / region / raw affiliation
    emails: List[EmailCandidate] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    keys: List[str] = Field(default_factory=list)  # strong identifiers: orcid:, openalex:, email:
    topics: List[str] = Field(default_factory=list)
    bio: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    signals: Dict[str, float] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    # For an organisation reached through its shared inbox: the person in the right role, if one was found
    # ({"name", "title", "source"}), so the email can be addressed to them.
    decision_maker: Optional[Dict[str, str]] = None
    criteria: Dict[str, float] = Field(default_factory=dict)  # support per fit criterion (cheap, then judged)

    # filled by ranking
    prescore: float = 0.0
    prejudge: Optional[float] = None  # cheap LLM screen before enrichment (brain/discovery/precision/prejudge.py)
    prejudge_reason: Optional[str] = None
    score: Optional[float] = None  # the final ranking score: calibrated P(good) when a calibrator exists, else fit x reach
    fit: Optional[float] = None  # the judge's fit for this sender and ask (0-1), before reachability is considered
    reasons: List[str] = Field(default_factory=list)
    hook: Optional[str] = None
    why_now: Optional[str] = None  # a timing fact from the evidence (hiring, just raised, new article), set by the judge
    sender_ref: Optional[int] = None  # index of the sender fact most relevant to this recipient, chosen by the judge
    flags: List[str] = Field(default_factory=list)

    @property
    def best_email(self) -> Optional[EmailCandidate]:
        return max(self.emails, key=lambda e: e.confidence) if self.emails else None

    def add_email(self, cand: EmailCandidate) -> None:
        addr = cand.address.strip().lower()
        cand.address = addr
        for e in self.emails:
            if e.address == addr:
                if cand.confidence > e.confidence:
                    e.confidence, e.method, e.source_url = cand.confidence, cand.method, cand.source_url
                return
        self.emails.append(cand)

    def add_url(self, url: Optional[str]) -> None:
        if url and url not in self.urls:
            self.urls.append(url)

    def add_key(self, key: str) -> None:
        if key and key not in self.keys:
            self.keys.append(key)

    def match_text(self) -> str:
        """Text used for cheap relevance matching."""
        parts = [self.title or "", " ".join(self.topics), self.bio or ""]
        for ev in self.evidence[:12]:
            parts.append(ev.text)
            if ev.detail:
                parts.append(ev.detail)
        return " . ".join(p for p in parts if p)


class Draft(BaseModel):
    contact_id: str
    to: str
    subject: str
    body: str
    issues: List[str] = Field(default_factory=list)  # non-empty => needs human attention
    address_tier: str = ""  # verified | attributed | inbox | guess | unattributed (brain/discovery/reach/tiers.py)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def sendable(self) -> bool:
        """Safe to send without a person re-checking it: the draft passed every check AND the mailbox is verified."""
        return self.ok and self.address_tier == "verified"


def stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

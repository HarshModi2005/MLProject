"""
The source contract: a manifest (SourceSpec) describes a source as data; adapters read it; normalize turns
what they read (Records) into Contacts.

A manifest is JSON, built in under brain/discovery/sources/ or user-supplied in ./sources/ (BRAIN_SOURCES_DIR).
Minimal example (an API whose items map straight onto contacts):

    {
      "name": "example_api", "title": "...", "adapter": "json_api", "kinds": ["org"],
      "proves": ["hiring"], "yields_email": false, "cost": "free", "tier": 1,
      "request": {"url": "https://api.example.com/search", "params": {"q": "{term}"}},
      "query": {"terms": ["topics"], "max": 3},
      "items": "results",
      "map": {"name": "company.name", "website": "company.url", "bio": "description"},
      "signals": {"hiring": 1.0},
      "evidence": "Hiring: {title}",
      "policy": {"robots": "api", "rate_per_min": 30}
    }
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ADAPTER_NAMES = ("json_api", "rss", "list_page", "sitemap", "search", "python")
COSTS = ("free", "fetch", "search", "search+llm", "paid")


class Policy(BaseModel):
    """How a source may be used. Every manifest must carry one."""

    robots: str = "respect"  # respect: pages go through the robots-aware fetcher | api: a documented API
    rate_per_min: Optional[int] = None  # network calls per minute to this source (cached answers are free)
    fields: Optional[List[str]] = None  # if set, only these mapped fields are kept
    retention_days: Optional[int] = None  # how long the persistent index may keep what this source gave
    terms: str = ""  # a note on the source's terms of use

    @field_validator("robots")
    @classmethod
    def _robots(cls, v: str) -> str:
        if v not in ("respect", "api"):
            raise ValueError("policy.robots must be 'respect' or 'api'")
        return v


class Auth(BaseModel):
    env: str  # environment variable holding the key; the source is unavailable without it
    header: Optional[str] = None  # send the key in this header ...
    param: Optional[str] = None  # ... or as this query parameter


class SourceSpec(BaseModel):
    name: str
    title: str = ""
    description: str = ""
    adapter: str
    tier: int = 1  # 1 = wide and free (recall), 2 = narrow and paid (precision)
    cost: str = "free"
    kinds: List[str] = Field(default_factory=lambda: ["person", "org"])
    stage: str = "seed"  # seed | expand | both
    order: Optional[int] = None  # position among tools when several could prove the same signal
    regions: List[str] = Field(default_factory=lambda: ["*"])  # ISO country codes served, or "*"
    proves: List[str] = Field(default_factory=list)
    yields_email: bool = False
    enabled_by_default: bool = True
    policy: Policy
    auth: Optional[Auth] = None

    # What to ask for. Adapter-specific; see each adapter's docstring.
    request: Dict[str, Any] = Field(default_factory=dict)
    query: Dict[str, Any] = Field(default_factory=dict)
    search: Dict[str, Any] = Field(default_factory=dict)
    items: Optional[str] = None  # path to the list of items in a JSON answer
    dedupe: Optional[str] = None  # item path whose value identifies an item
    follow: Dict[str, Any] = Field(default_factory=dict)  # fetch a URL found in each item and read it
    select: Dict[str, Any] = Field(default_factory=dict)  # CSS selectors (list_page / sitemap)
    entry: Optional[str] = None  # python adapter: "module:function" returning (contacts, stats)

    # How to turn a record into a contact (generic). A source with `hooks` may override any step.
    hooks: Optional[str] = None  # module with optional extract / to_contact / key / merge functions
    map: Dict[str, str] = Field(default_factory=dict)
    require: List[str] = Field(default_factory=list)  # record paths that must be non-empty
    signals: Dict[str, float] = Field(default_factory=dict)
    evidence: Optional[str] = None  # template rendered against the record
    email_confidence: float = 0.8
    max_items: int = 50
    declare_signals: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, v: str) -> str:
        if v not in ADAPTER_NAMES:
            raise ValueError(f"unknown adapter {v!r}; one of {ADAPTER_NAMES}")
        return v

    @field_validator("cost")
    @classmethod
    def _cost(cls, v: str) -> str:
        if v not in COSTS:
            raise ValueError(f"unknown cost {v!r}; one of {COSTS}")
        return v

    @field_validator("stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        if v not in ("seed", "expand", "both"):
            raise ValueError("stage must be seed, expand or both")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "SourceSpec":
        from brain.discovery.signals import SIGNALS, known, register_signal

        for d in self.declare_signals:
            if d.get("name") and d["name"] not in SIGNALS:
                register_signal(str(d["name"]), str(d.get("description") or ""), bool(d.get("proof", True)),
                                bool(d.get("judge", True)), d.get("decay_days"))
        bad = [s for s in list(self.proves) + list(self.signals) if not known(s)]
        if bad:
            raise ValueError(f"unregistered signals {bad}: add them to brain/discovery/signals.py or declare_signals")
        if self.adapter == "python" and not self.entry:
            raise ValueError("a python source needs 'entry': 'module:function'")
        if any(k not in ("person", "org") for k in self.kinds):
            raise ValueError("kinds must be person and/or org")
        return self

    @property
    def needs_search(self) -> bool:
        if "needs_search" in self.query:
            return bool(self.query["needs_search"])
        return self.adapter == "search" or self.cost in ("search", "search+llm") or self.query.get("via") == "search"

    def available(self) -> bool:
        import os

        return self.auth is None or bool(os.getenv(self.auth.env))


class Record(BaseModel):
    """One thing a source returned, before it becomes a contact."""

    source: str
    data: Dict[str, Any] = Field(default_factory=dict)
    url: Optional[str] = None
    fetched_at: float = Field(default_factory=time.time)

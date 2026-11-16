"""
Campaign context: the agent's working memory, built so that every prompt it writes is grounded.

What goes in has a source. Facts about the sender come from the user's own words, their CV or an
uploaded file — never from a model's guess. Targets the user already knows, people to exclude, past
emails to imitate and the log of searches that did or did not pay off all live here, persisted with
the campaign, and are rendered into compact briefs for the judge and the writer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from brain.textutil import registrable_domain, squash


class Fact(BaseModel):
    text: str
    source: str  # "user", "cv:<file>", "upload:<file>", a URL ...


class QueryLog(BaseModel):
    query: str
    kind: str = "search"  # search | places | news
    hits: int = 0
    new: int = 0  # new candidates this query added


class CampaignContext(BaseModel):
    understanding: str = ""  # one-paragraph restatement the user confirmed
    target_brief: str = ""  # what a good target looks like, in the user's terms
    assumptions: List[str] = Field(default_factory=list)  # made by the planner and shown to the user
    facts: List[Fact] = Field(default_factory=list)  # about the sender / offer, each with a source
    seed_names: List[str] = Field(default_factory=list)  # good examples to find more like
    seed_domains: List[str] = Field(default_factory=list)
    exclude_emails: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=list)
    exclude_names: List[str] = Field(default_factory=list)
    style_examples: List[str] = Field(default_factory=list)  # past emails the user liked
    uploads: List[Dict[str, str]] = Field(default_factory=list)  # {"file", "role", "summary"}
    query_log: List[QueryLog] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)  # things learned during the run

    # ── building ─────────────────────────────────────────────────────────────
    def add_fact(self, text: str, source: str) -> None:
        text = squash(text, 400)
        if not text or not source:
            return  # an unsourced fact is not a fact
        if not any(_same_fact(f.text, text) for f in self.facts):
            self.facts.append(Fact(text=text, source=source))

    def exclude(self, value: str) -> None:
        v = (value or "").strip().lower()
        if not v:
            return
        if "@" in v:
            if v not in self.exclude_emails:
                self.exclude_emails.append(v)
        elif "." in v and " " not in v:
            d = registrable_domain(v)
            if d not in self.exclude_domains:
                self.exclude_domains.append(d)
        elif v not in self.exclude_names:
            self.exclude_names.append(v)

    def add_seed(self, name: str = "", domain: str = "") -> None:
        if name and name not in self.seed_names:
            self.seed_names.append(squash(name, 80))
        if domain:
            d = registrable_domain(domain)
            if d and d not in self.seed_domains:
                self.seed_domains.append(d)

    def log_query(self, query: str, kind: str, hits: int, new: int) -> None:
        self.query_log.append(QueryLog(query=query, kind=kind, hits=hits, new=new))

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    # ── using ────────────────────────────────────────────────────────────────
    def is_excluded(self, name: str = "", emails: Optional[List[str]] = None, domain: str = "") -> Optional[str]:
        if name and name.strip().lower() in self.exclude_names:
            return "excluded_by_user"
        for e in emails or []:
            if e.lower() in self.exclude_emails:
                return "excluded_by_user"
        if domain and registrable_domain(domain) in self.exclude_domains:
            return "excluded_by_user"
        return None

    def productive_queries(self, min_new: int = 2) -> List[QueryLog]:
        return sorted((q for q in self.query_log if q.new >= min_new), key=lambda q: q.new, reverse=True)

    def tried(self, query: str) -> bool:
        norm = _norm_query(query)
        return any(_norm_query(q.query) == norm for q in self.query_log)

    def sender_facts_text(self, limit: int = 2500) -> str:
        out, used = [], 0
        for f in self.facts:
            line = f"- {f.text}"
            if used + len(line) > limit:
                break
            out.append(line)
            used += len(line)
        return "\n".join(out)

    def brief(self, limit: int = 1200) -> str:
        """Compact, verified context for judge/writer prompts."""
        parts = []
        if self.understanding:
            parts.append(f"Campaign (confirmed by the user): {squash(self.understanding, 400)}")
        if self.target_brief:
            parts.append(f"A good target: {squash(self.target_brief, 300)}")
        if self.seed_names:
            parts.append("Examples the user considers good: " + ", ".join(self.seed_names[:8]))
        if self.notes:
            parts.append("Observed so far: " + "; ".join(self.notes[:4]))
        text = "\n".join(parts)
        return text[:limit]


def _same_fact(a: str, b: str) -> bool:
    """'Advitiya 2026 attracted 5000+ attendees from 120 colleges' ~ 'Managed an event with 5000+ attendees from
    120 colleges': same claim when the numbers match and the wording largely overlaps."""
    if a.lower() == b.lower():
        return True
    nums_a, nums_b = set(re.findall(r"\d[\d,.]*", a)), set(re.findall(r"\d[\d,.]*", b))
    if nums_a != nums_b:
        return False
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(a.lower(), b.lower()) >= 80
    except ImportError:  # pragma: no cover
        return False


def _norm_query(q: str) -> str:
    return " ".join(sorted(re.findall(r"[a-z0-9]+", q.lower())))

"""
Search planner: spend the Serper budget like money.

Each campaign gets a budget sized to its target. Queries run in rounds:
  1. the best planned queries (from the confirmed campaign, lookalikes of the user's examples first);
  2. only if more candidates are still needed: cheap variants of the queries that actually produced
     candidates (other topics, other places) — no LLM;
  3. only if still short: one LLM call that sees the query log (what worked, what returned nothing)
     and proposes a few new queries.
It stops the moment enough candidates exist, never repeats a query (cached or not), and logs every
query's yield into the campaign context so later runs start from what worked.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional

from brain.context import CampaignContext
from brain.llm import LLM, LLMError
from brain.models import CampaignSpec
from brain.playbooks import Playbook
from brain.textutil import squash

log = logging.getLogger("brain.searchplan")

REPLAN_SYSTEM = """You plan web searches for an outreach campaign. You see which queries found new candidates and which
found nothing. Propose new queries that are likely to surface MORE organisations/people of the kind that worked,
in the same geography. Avoid repeating or lightly rephrasing queries that found nothing. Never use site:linkedin.com.
Return JSON: {"queries": ["...", "..."]} with at most 5 queries."""


def search_budget(target_count: int, ceiling: int) -> int:
    """Total Serper calls for a campaign: enough to find ~target_count good contacts, never above the ceiling."""
    return max(8, min(ceiling, 10 + 3 * target_count))


class SearchPlanner:
    def __init__(self, spec: CampaignSpec, playbook: Playbook, ctx: CampaignContext, llm: Optional[LLM] = None,
                 need: int = 0):
        self.spec, self.pb, self.ctx, self.llm = spec, playbook, ctx, llm
        self.need = need or max(20, spec.target_count * 6)
        self.replanned = False

    # ── query sources ────────────────────────────────────────────────────────
    def initial(self, kind: str = "search") -> List[str]:
        spec, pb = self.spec, self.pb
        places = spec.regions or [_country_name(c) for c in spec.countries] or [""]
        out: List[str] = []
        if kind == "places":
            for topic in spec.topics[:4]:
                for place in places[:3]:
                    out.append(squash(f"{topic} in {place}"))
            return _dedupe(out)
        if kind == "news":
            for topic in spec.topics[:4]:
                out.append(squash(f"{topic} {places[0]}"))
            return _dedupe(out)
        out += [q for q in spec.lookalike_queries if q]
        out += [q for q in spec.search_queries if "site:linkedin" not in q.lower()]
        role = (spec.persona.roles or pb.default_roles or [""])[0]
        for topic in spec.topics[:3]:
            for tpl in pb.query_templates[:2]:
                out.append(squash(tpl.format(topic=topic, place=places[0], role=role)))
        return _dedupe(out)

    def variants(self, kind: str = "search") -> List[str]:
        """Cheap, LLM-free expansions of queries that produced candidates."""
        spec = self.spec
        topics = spec.topics + spec.expanded_terms
        places = spec.regions + [_country_name(c) for c in spec.countries]
        out: List[str] = []
        for q in self.ctx.productive_queries():
            if q.kind != kind:
                continue
            low = q.query.lower()
            used_topic = next((t for t in topics if t.lower() in low), None)
            used_place = next((p for p in places if p and p.lower() in low), None)
            for t in topics:
                if used_topic and t != used_topic:
                    out.append(squash(_replace(q.query, used_topic, t)))
            for p in places:
                if used_place and p != used_place:
                    out.append(squash(_replace(q.query, used_place, p)))
        return [q for q in _dedupe(out) if not self.ctx.tried(q)]

    def replan(self, kind: str = "search") -> List[str]:
        if self.replanned or self.llm is None or not self.llm.available or not self.ctx.query_log:
            return []
        self.replanned = True
        log_lines = "\n".join(f"- [{q.kind}] {q.query}  -> {q.new} new of {q.hits} results"
                              for q in self.ctx.query_log[-25:])
        user = (f"Campaign: {self.spec.objective}\nLooking for: {self.spec.persona.description}\n"
                f"Topics: {', '.join(self.spec.topics)}\nWhere: {', '.join(self.spec.regions + self.spec.countries) or 'anywhere'}\n"
                f"Query log:\n{log_lines}")
        try:
            data = self.llm.json(system=REPLAN_SYSTEM, user=user, purpose="replan_search", tier="fast", max_tokens=400)
        except LLMError as e:
            log.warning("replanning failed: %s", e)
            return []
        qs = [squash(str(q), 120) for q in (data.get("queries") or [])][:5] if isinstance(data, dict) else []
        return [q for q in qs if q and "site:linkedin" not in q.lower() and not self.ctx.tried(q)]

    # ── execution ────────────────────────────────────────────────────────────
    def run(self, execute: Callable[[str], tuple], have: Callable[[], int], budget_left: Callable[[], int],
            kind: str = "search", first_round_max: int = 8) -> None:
        """execute(query) -> (hits, new_candidates). Stops when have() >= need or the budget is gone."""
        rounds: List[Iterable[str]] = [self.initial(kind)[:first_round_max], None, None]  # None = compute lazily
        for i in range(3):
            if have() >= self.need or budget_left() <= 0:
                return
            queries = rounds[i] if rounds[i] is not None else (self.variants(kind) if i == 1 else self.replan(kind))
            for q in queries:
                if have() >= self.need or budget_left() <= 0:
                    return
                if self.ctx.tried(q):
                    continue
                hits, new = execute(q)
                self.ctx.log_query(q, kind, hits, new)


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for q in items:
        k = " ".join(sorted(q.lower().split()))
        if q and k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _replace(text: str, old: str, new: str) -> str:
    import re

    return re.sub(re.escape(old), new, text, flags=re.I)


def _country_name(code: str) -> str:
    from brain.textutil import COUNTRY_CODES

    for name, c in COUNTRY_CODES.items():
        if c == code and len(name) > 3:
            return name.title() if name not in ("usa", "uae") else name.upper()
    return code

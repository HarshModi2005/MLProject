"""
Discovery tools, registered by name. A playbook lists which ones to use; the pipeline runs them.

Adding a capability means writing one function `(ToolRun) -> (contacts, stats)` and registering it.
Each tool declares:
    cost     what it spends (free | search | search+llm | fetch), so budget allocation is visible;
    proves   which fit signals its results carry proof of (see brain/fit.py) — the explorer adds a tool
             beyond the playbook's list when the campaign's criteria need that proof;
    kinds    which target kinds it produces (person / org);
    stage    "seed" (runs from the campaign description), "expand" (needs known-good prospects: the
             user's examples or the best finds so far) or "both".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from brain.context import CampaignContext
from brain.models import CampaignSpec, Contact, Evidence
from brain.playbooks import Playbook
from brain.textutil import clean_name, country_from_domain, looks_like_person_name, registrable_domain, squash

log = logging.getLogger("brain.tools")


@dataclass
class ToolRun:
    spec: CampaignSpec
    playbook: Playbook
    ctx: CampaignContext
    deps: "object"  # brain.pipeline.Deps (kept loose to avoid an import cycle)
    planner: "object"  # brain.searchplan.SearchPlanner
    budget: Optional[int] = None  # search calls this run may spend (None = whatever is left)
    winners: List[Contact] = field(default_factory=list)  # best prospects so far, for expansion tools


@dataclass
class Tool:
    name: str
    description: str
    cost: str  # free | search | search+llm | fetch
    fn: Callable[[ToolRun], Tuple[List[Contact], Dict]]
    needs_search: bool = False
    proves: Tuple[str, ...] = ()
    kinds: Tuple[str, ...] = ("person", "org")
    stage: str = "seed"  # seed | expand | both
    order: int = 1000  # position when several tools could prove the same signal (lower = preferred)
    spec: Optional[object] = None  # the manifest (brain.discovery.contracts.SourceSpec) for manifest sources


TOOLS: Dict[str, Tool] = {}

# Historical registration order, kept so tool choice stays the same now that some tools are manifests
# (brain/discovery/sources/*.json carry the matching "order").
_ORDER = ["openalex", "orgs", "places", "web", "github", "news", "events", "jobs", "portfolio", "podcasts",
          "lookalike", "coauthors", "memory"]


def register(name: str, description: str, cost: str, needs_search: bool = False, proves: Tuple[str, ...] = (),
             kinds: Tuple[str, ...] = ("person", "org"), stage: str = "seed"):
    def deco(fn):
        order = _ORDER.index(name) * 10 if name in _ORDER else 1000
        TOOLS[name] = Tool(name, description, cost, fn, needs_search, tuple(proves), tuple(kinds), stage, order)
        return fn
    return deco


def _left(run: "ToolRun", reserve: int = 2):
    """budget_left() for a tool: the global remaining budget, capped by this run's allowance."""
    search = run.deps.search
    start = search.calls

    def left() -> int:
        lim = search.remaining - reserve
        return min(lim, run.budget - (search.calls - start)) if run.budget is not None else lim
    return left


def _gl(spec: CampaignSpec) -> str:
    return spec.countries[0] if len(spec.countries) == 1 else ""


@register("openalex", "Researchers found through recent publications on the topics (OpenAlex).", "free",
          proves=("publications", "topic"), kinds=("person",))
def _openalex(run: ToolRun):
    source = run.deps.openalex
    found = source.search(run.spec)
    return found, {"people": len(found), "failed_terms": list(getattr(source, "failed_terms", []))}


@register("orgs", "Organisations via web search: their own sites, plus names read off list and event pages.",
          "search+llm", True, proves=("topic", "audience", "funding_focus"), kinds=("org",))
def _orgs(run: ToolRun):
    from brain.orgs import discover_orgs

    return discover_orgs(run.spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm, run.deps.settings,
                         planner=run.planner, budget=run.budget)


@register("places", "Local businesses on Google Maps, with website and address (Serper places).", "search", True,
          proves=("local_business", "location"), kinds=("org",))
def _places(run: ToolRun):
    from brain.orgs import discover_places

    return discover_places(run.spec, run.playbook, run.deps.search, run.planner, gl=_gl(run.spec), budget=run.budget)


@register("web", "People listed on staff, team or faculty pages found by web search.", "search", True,
          proves=("topic",), kinds=("person",))
def _web(run: ToolRun):
    from brain.discover import discover_web

    budget = max(0, min(8, run.deps.search.remaining - 4, run.budget if run.budget is not None else 8))
    return discover_web(run.spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm,
                        run.deps.settings, max_queries=budget)


@register("github", "Developers via GitHub user search (skills, location); public profile emails only.", "free",
          proves=("open_source",), kinds=("person",))
def _github(run: ToolRun):
    from brain.sources.github import GitHubSource

    src = GitHubSource(run.deps.settings, run.deps.cache)
    found = src.search(run.spec)
    return found, {"people": len(found), "requests": src.requests}


@register("events", "Sponsors (or speakers) of similar past events, read from the events' own pages.", "search", True,
          proves=("past_sponsorship", "spoke_before"))
def _events(run: ToolRun):
    from brain.channels import discover_events

    return discover_events(run)


@register("portfolio", "Investors whose own portfolio pages match the topics.", "search+llm", True,
          proves=("portfolio_fit",), kinds=("org",))
def _portfolio(run: ToolRun):
    from brain.channels import discover_portfolio

    return discover_portfolio(run)


@register("lookalike", "Organisations similar to the user's examples or to the best finds so far.", "search+llm", True,
          proves=("topic",), kinds=("org",), stage="expand")
def _lookalike(run: ToolRun):
    from brain.channels import discover_lookalike

    return discover_lookalike(run)


@register("coauthors", "Researchers who publish with the best finds so far (OpenAlex co-author graph).", "free",
          proves=("publications",), kinds=("person",), stage="expand")
def _coauthors(run: ToolRun):
    from brain.channels import discover_coauthors

    return discover_coauthors(run)


@register("memory", "Prospects already found and verified in earlier campaigns.", "free", stage="seed")
def _memory(run: ToolRun):
    from brain.channels import discover_memory

    return discover_memory(run)


def authors_from_article(html: str, url: str) -> List[Tuple[str, Optional[str]]]:
    """Bylines from an article page: JSON-LD author, <meta name=author>, rel=author links, 'By Name'."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    found: List[Tuple[str, Optional[str]]] = []

    def push(name: str, link: Optional[str] = None):
        name = clean_name(squash(name, 80))
        if looks_like_person_name(name) and name.lower() not in {n.lower() for n, _ in found}:
            found.append((name, link))

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            authors = item.get("author")
            for a in authors if isinstance(authors, list) else [authors]:
                if isinstance(a, dict) and a.get("@type", "Person") == "Person" and isinstance(a.get("name"), str):
                    push(a["name"], a.get("url") if isinstance(a.get("url"), str) else None)
                elif isinstance(a, str):
                    push(a)
    for attrs in ({"name": "author"}, {"property": "article:author"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content") and not tag["content"].startswith("http"):
            for part in re.split(r",| and ", tag["content"]):
                push(part)
    for a in soup.find_all("a", rel=lambda v: v and "author" in v):
        push(a.get_text(" "), a.get("href"))
    if not found:
        m = re.search(r"\bBy\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){1,3})", soup.get_text(" ")[:3000])
        if m:
            push(m.group(1))
    return found[:3]


# news, jobs and podcasts are manifest sources now: brain/discovery/sources/*.json

def describe_tools() -> str:
    return "\n".join(f"- {t.name} [{t.cost}; proves {', '.join(t.proves) or '-'}; {t.stage}]: {t.description}"
                     for t in TOOLS.values())


# Manifest sources (built-in and ./sources) join the registry last, then everything is ordered by `order`.
from brain.discovery import registry as _registry  # noqa: E402

_registry.install(TOOLS)

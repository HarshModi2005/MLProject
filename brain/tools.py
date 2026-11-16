"""
Discovery tools, registered by name. A playbook lists which ones to use; the pipeline runs them.

Adding a capability means writing one function `(ToolRun) -> (contacts, stats)` and registering it.
Each tool declares what it costs so the planner and the user can see where budget goes.
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


@dataclass
class Tool:
    name: str
    description: str
    cost: str  # free | search | search+llm | fetch
    fn: Callable[[ToolRun], Tuple[List[Contact], Dict]]
    needs_search: bool = False


TOOLS: Dict[str, Tool] = {}


def register(name: str, description: str, cost: str, needs_search: bool = False):
    def deco(fn):
        TOOLS[name] = Tool(name, description, cost, fn, needs_search)
        return fn
    return deco


def _gl(spec: CampaignSpec) -> str:
    return spec.countries[0] if len(spec.countries) == 1 else ""


@register("openalex", "Researchers found through recent publications on the topics (OpenAlex).", "free")
def _openalex(run: ToolRun):
    source = run.deps.openalex
    found = source.search(run.spec)
    return found, {"people": len(found), "failed_terms": list(getattr(source, "failed_terms", []))}


@register("orgs", "Organisations via web search: their own sites, plus names read off list pages.", "search+llm", True)
def _orgs(run: ToolRun):
    from brain.orgs import discover_orgs

    return discover_orgs(run.spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm, run.deps.settings,
                         planner=run.planner)


@register("places", "Local businesses on Google Maps, with website and address (Serper places).", "search", True)
def _places(run: ToolRun):
    from brain.orgs import discover_places

    return discover_places(run.spec, run.playbook, run.deps.search, run.planner, gl=_gl(run.spec))


@register("web", "People listed on staff, team or faculty pages found by web search.", "search", True)
def _web(run: ToolRun):
    from brain.discover import discover_web

    budget = max(0, min(8, run.deps.search.remaining - 4))
    return discover_web(run.spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm,
                        run.deps.settings, max_queries=budget)


@register("github", "Developers via GitHub user search (skills, location); public profile emails only.", "free")
def _github(run: ToolRun):
    from brain.sources.github import GitHubSource

    src = GitHubSource(run.deps.settings, run.deps.cache)
    found = src.search(run.spec)
    return found, {"people": len(found), "requests": src.requests}


@register("news", "Journalists via recent articles on the topic (Serper news), read from article bylines.",
          "search", True)
def _news(run: ToolRun):
    search, fetcher = run.deps.search, run.deps.fetcher
    people: Dict[str, Contact] = {}
    stats = {"queries": 0, "articles": 0, "bylines": 0}

    def execute(query: str):
        stats["queries"] += 1
        articles = search.news(query, gl=_gl(run.spec))
        stats["articles"] += len(articles)
        new = 0
        for art, page in zip(articles, fetcher.get_many([a.url for a in articles[:10]])):
            if not page.ok:
                continue
            for name, author_url in authors_from_article(page.html, page.url):
                key = name.lower()
                dom = registrable_domain(page.url)
                if key in people:
                    people[key].evidence.append(Evidence(kind="publication", text=art.title, url=art.url))
                    continue
                c = Contact(kind="person", name=name, title="Journalist", org=art.source or None, org_domain=dom,
                            country=country_from_domain(dom), sources=["news"])
                c.add_url(author_url)
                c.evidence.append(Evidence(kind="publication", text=squash(art.title, 200), url=art.url,
                                           detail=squash(f"{art.source} {art.date}: {art.snippet}", 300)))
                people[key] = c
                new += 1
                stats["bylines"] += 1
        return len(articles), new

    run.planner.run(execute, have=lambda: len(people), budget_left=lambda: search.remaining - 4, kind="news",
                    first_round_max=5)
    return list(people.values()), stats


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


def describe_tools() -> str:
    return "\n".join(f"- {t.name} [{t.cost}]: {t.description}" for t in TOOLS.values())

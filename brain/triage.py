"""
Search-hit triage: decide what each result IS before spending fetches or LLM calls on it.

    own_site   the target organisation's own website        -> becomes an organisation candidate
    list       a "top 20 X" page or a directory/aggregator  -> read for organisation names
    event      an event page (fest, hackathon, conference)  -> read for sponsors / speakers
    article    a news story or blog post                    -> never a target; may name targets
    irrelevant anything else                                -> dropped

Heuristics settle most hits for free. Only the hits they cannot place ("unsure": deep pages on
unknown sites) go to the LLM, in ONE batched fast call per query, and any failure falls back to
the old behaviour (treat the site as an organisation), so triage can only remove waste.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from brain.models import CampaignSpec
from brain.sources.websearch import Hit
from brain.textutil import registrable_domain, squash

log = logging.getLogger("brain.triage")

KINDS = ("own_site", "list", "event", "article", "irrelevant")

_LIST_TITLE = re.compile(
    r"\b(top|best|list|leading|directory|ranking|\d{1,3}\+?)\b.*\b(compan(y|ies)|startups?|firms|funds|investors|"
    r"angels|vcs|clinics|hospitals|brands|agencies|sponsors|organi[sz]ations|businesses|platforms|tools|"
    r"foundations|podcasts|manufacturers|suppliers|ngos)\b", re.I)
# Event names often fuse words ("HackFest", "Techfest", "HackMIT"), so "fest" and "hack" match inside words.
_EVENT = re.compile(r"(hackathon|\w*fest\b|festival|summit|conference|conclave|meetup|\bexpo\b|symposium|\bhack\w*|"
                    r"bootcamp|olympiad|championship)", re.I)
_EVENT_PATH = re.compile(r"/(sponsors?|partners?|speakers?|judges|events?)(/|$|\.)", re.I)
_ARTICLE_PATH = re.compile(r"/(20\d\d/\d{1,2}|news|blog|blogs|article|articles|story|stories|posts?|press-release|"
                           r"opinion|magazine)(/|$)", re.I)
NEWS_DOMAINS = ("indiatimes.com", "timesofindia", "hindustantimes", "economictimes", "livemint", "thehindu", "ndtv", "indianexpress",
                "moneycontrol", "business-standard", "techcrunch", "forbes", "bloomberg", "reuters", "theverge",
                "wired", "businessinsider", "cnbc", "bbc.", "nytimes", "theguardian", "venturebeat", "zdnet",
                "techradar", "analyticsindiamag", "medianama", "thenextweb", "fortune.com", "fastcompany")

TRIAGE_SYSTEM = """You classify web search results for an outreach campaign. For each result decide what the page is:
 "own_site"   the website of ONE organisation that could itself be a TARGET (any page of its own site)
 "list"       a page listing several organisations (rankings, directories, "top N", portfolios of others)
 "event"      a page about an event (fest, hackathon, conference) that may list sponsors or speakers
 "article"    a news story, blog post or press release
 "irrelevant" anything else, or an organisation that clearly cannot be a TARGET
Return JSON: {"results": [{"i": <index>, "kind": "<one of the five>"}]}. Include every index once."""


def _depth(url: str) -> int:
    return len([p for p in urlsplit(url).path.split("/") if p])


def heuristic_kind(hit: Hit, is_aggregator) -> str:
    """One of KINDS, or 'unsure'."""
    url, title, snippet = hit.url, hit.title or "", hit.snippet or ""
    dom = registrable_domain(url)
    path = urlsplit(url).path.lower()
    if is_aggregator(dom) or _LIST_TITLE.search(title):
        return "list"
    host = urlsplit(url).netloc.lower()
    if any(n in host for n in NEWS_DOMAINS) or _ARTICLE_PATH.search(path) or path.count("-") >= 6:
        return "article"
    if _EVENT_PATH.search(path) and (_EVENT.search(title + " " + snippet) or re.search(r"sponsor|speaker", title, re.I)):
        return "event"
    if _EVENT.search(title) and re.search(r"20\d\d|sponsor|partner", title + " " + snippet, re.I):
        return "event"
    if _depth(url) <= 1:
        return "own_site"
    return "unsure"


def triage(hits: List[Hit], spec: CampaignSpec, is_aggregator, llm=None, stats: Optional[Dict] = None
           ) -> List[Tuple[Hit, str]]:
    """Label each hit. `llm` (optional) settles the unsure ones in one batched call."""
    labels = [heuristic_kind(h, is_aggregator) for h in hits]
    unsure = [i for i, k in enumerate(labels) if k == "unsure"]
    if stats is not None:
        for k in labels:
            stats[f"triage_{k}"] = stats.get(f"triage_{k}", 0) + 1
    if unsure and llm is not None and getattr(llm, "available", False):
        target = (f"{spec.persona.description or ''} | kinds: {', '.join(spec.persona.org_types[:4])} | "
                  f"topics: {', '.join(spec.topics[:5])}")
        items = [{"i": i, "url": hits[i].url, "title": squash(hits[i].title, 120), "snippet": squash(hits[i].snippet, 200)}
                 for i in unsure]
        try:
            from brain.safety import guard

            data = llm.json(system=guard(TRIAGE_SYSTEM), user=f"TARGET: {target}\n\nRESULTS:\n{json.dumps(items)}",
                            purpose="triage", tier="fast", max_tokens=40 * len(items) + 80)
            for r in (data.get("results") or []) if isinstance(data, dict) else []:
                i, kind = r.get("i"), r.get("kind")
                if isinstance(i, int) and i in unsure and kind in KINDS:
                    labels[i] = kind
        except Exception as e:  # triage is an optimisation: never let it sink discovery
            log.warning("LLM triage failed, keeping heuristic labels: %s", e)
    return [(h, "own_site" if k == "unsure" else k) for h, k in zip(hits, labels)]

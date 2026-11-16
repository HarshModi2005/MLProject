"""
Organisation site crawl: find staff directories on an organisation's own website and read
names + emails off them. No search API, no LLM — only page fetches, tightly budgeted.

Used during enrichment for organisations where top candidates still lack an email: the people
found here are matched back to those candidates by name.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Tuple

from brain.fetch import Fetcher
from brain.models import Contact
from brain.pages import classify, directory_contacts, parse
from brain.textutil import registrable_domain

log = logging.getLogger("brain.sitecrawl")

# Links worth following, strongest first. Matched against the URL path and the anchor text.
_FOLLOW = [
    (3, re.compile(r"faculty|people|staff|directory|our[-_ ]?team|\bteam\b|members|researchers|leadership", re.I)),
    (2, re.compile(r"department|school of|schools|academics?|centres?|centers?|divisions?|about", re.I)),
]
_AVOID = re.compile(r"news|event|admission|alumni|tender|career|recruit|login|gallery|library|student|"
                    r"placement|download|notice|calendar|\.pdf|facult(y|ies)-of-|facultyposition", re.I)


def _link_score(url: str, text: str, hint_terms: List[str]) -> int:
    target = f"{url} {text}"
    if _AVOID.search(target):
        return 0
    score = max((w for w, rx in _FOLLOW if rx.search(target)), default=0)
    if score and any(t in target.lower() for t in hint_terms):
        score += 2  # e.g. the computer science department when the campaign is about ML
    return score


def crawl_org(home_url: str, fetcher: Fetcher, hint_terms: List[str], max_pages: int = 12) -> List[Contact]:
    """Best-first crawl of one organisation's site for directory pages. Returns the people listed."""
    domain = registrable_domain(home_url)
    frontier: List[Tuple[int, int, str]] = [(10, 0, home_url)]  # (score, depth, url)
    seen: Set[str] = {home_url}
    people: Dict[str, Contact] = {}
    fetched = 0
    while frontier and fetched < max_pages:
        frontier.sort(key=lambda x: (-x[0], x[1]))
        _, depth, url = frontier.pop(0)
        page = fetcher.get(url)
        fetched += 1
        if not page.ok:
            # Many sites only answer on the www. host.
            if depth == 0 and "//www." not in url:
                www = url.replace("://", "://www.", 1)
                seen.add(www)
                frontier.append((10, 0, www))
            continue
        parsed = parse(page.html, page.url)
        if classify(parsed) == "directory":
            for c in directory_contacts(parsed):
                key = c.name.lower()
                if key in people:
                    for e in c.emails:
                        people[key].add_email(e)
                else:
                    people[key] = c
        if depth >= 3:
            continue
        for href, text in parsed.nav_links:
            if href in seen or registrable_domain(href) != domain:
                continue
            score = _link_score(href, text, hint_terms)
            if text == "redirect":
                score = 9
            if score:
                seen.add(href)
                frontier.append((score, depth + 1, href))
    log.info("site crawl %s: %d pages, %d people", domain, fetched, len(people))
    return list(people.values())

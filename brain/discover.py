"""
Web discovery: search -> rank the hits -> fetch only the best pages -> read people off them.

The legacy crawler scraped the first 50 URLs in arrival order. Here hits are ranked on their title
and snippet first, so the fetch budget goes to the pages most likely to list the right people.
The LLM is used only as a fallback for promising pages the heuristics could not read.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from brain.emailfinder import attach_emails, deobfuscate
from brain.fetch import Fetcher
from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact, Evidence
from brain.packs import Pack
from brain.pages import DIRECTORY_PATH_HINTS, contacts_from_page, page_excerpt
from brain.rank import bm25_scores
from brain.settings import Settings
from brain.sources.websearch import Hit, WebSearch
from brain.spec import fallback_queries
from brain.textutil import country_from_domain, looks_like_person_name, clean_name, registrable_domain, squash

log = logging.getLogger("brain.discover")

EXTRACT_SYSTEM = """List the people described on this web page who could be contacted professionally.
Return JSON: {"people": [{"name": "<full name exactly as written>", "title": "<role or null>",
"email": "<address exactly as it appears, or null>", "about": "<one sentence on what they work on, or null>"}]}
Only include people whose name literally appears in the text. Never guess an email. At most 25 people.
If the page lists no individual people, return {"people": []}."""


def rank_hits(hits: List[Hit], spec: CampaignSpec, pack: Pack) -> List[Tuple[float, Hit]]:
    if not hits:
        return []
    roles = " ".join(spec.persona.roles or pack.default_roles)
    query = spec.query_text() + " ; " + roles + " ; " + " ".join(spec.regions + spec.organizations)
    sims = bm25_scores(query, [f"{h.title} . {h.snippet}" for h in hits])
    scored = []
    for h, s in zip(hits, sims):
        url = h.url.lower()
        bonus = 0.25 if any(x in url for x in DIRECTORY_PATH_HINTS) else 0.0
        if spec.exclude_domains and registrable_domain(h.url) in spec.exclude_domains:
            continue
        scored.append((s + bonus, h))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def llm_extract(html: str, url: str, llm: LLM) -> List[Contact]:
    """Fallback extractor. Everything the model returns is verified against the page text."""
    text = page_excerpt(html, url, limit=6000)
    if len(text) < 300:
        return []
    from brain.safety import clean_text, guard

    text = clean_text(text)
    data = llm.json(system=guard(EXTRACT_SYSTEM), user=f"URL: {url}\n\nPAGE TEXT:\n{text}", purpose="extract",
                    tier="fast", max_tokens=1400)
    haystack = deobfuscate(text).lower()
    domain = registrable_domain(url)
    out: List[Contact] = []
    for p in (data.get("people") or [])[:25]:
        if not isinstance(p, dict):
            continue
        name = clean_name(str(p.get("name") or ""))
        if not looks_like_person_name(name) or name.lower() not in haystack:
            continue  # the model made the name up or it is not a person
        c = Contact(name=name, title=squash(str(p.get("title") or ""), 80) or None, org_domain=domain,
                    country=country_from_domain(domain), sources=["web"], flags=["llm_extracted"])
        c.add_url(url)
        email = str(p.get("email") or "").strip().lower()
        if email and email in haystack:
            attach_emails(c, [email], "llm", url)
        about = squash(str(p.get("about") or ""), 300)
        if about:
            c.bio = about
            c.evidence.append(Evidence(kind="page", text=about, url=url))
        out.append(c)
    return out


def discover_web(spec: CampaignSpec, pack: Pack, search: WebSearch, fetcher: Fetcher, llm: Optional[LLM],
                 settings: Settings, max_queries: int = 10, max_pages: int = 40) -> Tuple[List[Contact], Dict[str, int]]:
    stats = {"queries": 0, "hits": 0, "pages_fetched": 0, "profile_pages": 0, "directory_pages": 0,
             "other_pages": 0, "llm_extracted_pages": 0}
    if not search.available:
        return [], stats
    queries = (spec.search_queries or fallback_queries(spec))[:max_queries]
    hits: Dict[str, Hit] = {}
    for q in queries:
        stats["queries"] += 1
        for h in search.search(q):
            hits.setdefault(h.url, h)
    stats["hits"] = len(hits)

    ranked = rank_hits(list(hits.values()), spec, pack)[:max_pages]
    pages = fetcher.get_many([h.url for _, h in ranked])
    contacts: List[Contact] = []
    unread: List[Tuple[float, str, str]] = []
    for (score, hit), page in zip(ranked, pages):
        if not page.ok:
            continue
        stats["pages_fetched"] += 1
        kind, found = contacts_from_page(page.html, page.url)
        stats[f"{kind}_pages"] += 1
        for c in found:
            if not c.evidence and hit.title:
                # Listing pages carry no per-person text; keep the page context so ranking has something to match.
                c.evidence.append(Evidence(kind="page", text=f"Listed on: {squash(hit.title, 150)}",
                                           url=page.url, detail=squash(hit.snippet, 250) or None))
        contacts.extend(found)
        if not found:
            unread.append((score, page.url, page.html))

    # LLM fallback only for the most promising pages that yielded nothing, within a fixed budget.
    if llm is not None and llm.available:
        for score, url, html in sorted(unread, key=lambda x: x[0], reverse=True)[: settings.max_extract_calls]:
            if score < 0.35:
                break
            try:
                found = llm_extract(html, url, llm)
            except LLMError as e:
                log.warning("LLM extraction failed for %s: %s", url, e)
                break
            if found:
                stats["llm_extracted_pages"] += 1
                contacts.extend(found)
    return contacts, stats

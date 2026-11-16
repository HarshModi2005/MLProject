"""
list_page: known pages that list many candidates (directories, member lists, exhibitor lists, sponsor walls).

    "request": {"urls": ["https://example.org/members?page={page}"], "pages": [1, 2, 3]},
               # or "urls": ["https://example.org/search?q={term|quote}"] with "query": {"terms": [...]}
    "select":  {"item": ".member-card", "fields": {"name": "h3", "website": "a.site@href",
                                                    "email": "a[href^='mailto:']@href", "bio": "p"}},

A source with hooks may implement extract(spec, run, hit, page) -> [dict] instead of selectors. Pages
are read through the robots-aware fetcher.
"""

from __future__ import annotations

from typing import Any, Dict, List

from brain.discovery.adapters.common import extract_from_page, finish
from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.normalize import campaign_terms, campaign_vars, render


def page_urls(spec: SourceSpec, run) -> List[str]:
    req = spec.request
    base = campaign_vars(run.spec, spec.query, run.playbook)
    terms = campaign_terms(run.spec, spec.query) if "{term" in str(req.get("urls")) else [""]
    pages = req.get("pages") or [None]
    out = []
    for tpl in req.get("urls") or []:
        for term in terms:
            for p in pages:
                out.append(render(tpl, dict(base, term=term, page=p if p is not None else "")))
    return list(dict.fromkeys(u for u in out if u.startswith("http")))[: int(req.get("max_pages") or 20)]


def collect(spec: SourceSpec, run):
    stats: Dict[str, Any] = {"pages": 0}
    urls = page_urls(spec, run)
    records: List[Record] = []
    for page in run.deps.fetcher.get_many(urls):
        if not page.ok:
            continue
        stats["pages"] += 1
        records += [Record(source=spec.name, url=page.url, data=row) for row in extract_from_page(spec, run, page)]
    return finish(spec, run, records, stats)

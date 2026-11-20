"""
sitemap: read a site's sitemap.xml, keep the pages whose path matches, and read them.

    "request": {"domains": ["example.org"],     # optional; otherwise the winners' / seeds' domains (expand)
                "path": "/(team|speakers|people)/", "max_pages": 15, "max_domains": 5},
    "select":  {...}                             # or hooks.extract

Sitemap indexes are followed one level (at most 3 child sitemaps per domain).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from brain.discovery.adapters.common import extract_from_page, finish
from brain.discovery.contracts import Record, SourceSpec


def sitemap_locs(xml: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml or "", re.I)]


def domain_urls(fetcher, domain: str, path_rx: str, limit: int) -> List[str]:
    root = fetcher.get(f"https://{domain}/sitemap.xml")
    if not root.ok:
        return []
    locs = sitemap_locs(root.html)
    children = [u for u in locs if u.lower().endswith(".xml")][:3]
    pages = [u for u in locs if not u.lower().endswith(".xml")]
    for child in fetcher.get_many(children):
        if child.ok:
            pages += [u for u in sitemap_locs(child.html) if not u.lower().endswith(".xml")]
    rx = re.compile(path_rx, re.I) if path_rx else None
    return [u for u in dict.fromkeys(pages) if rx is None or rx.search(u)][:limit]


def collect(spec: SourceSpec, run):
    req = spec.request
    domains = list(req.get("domains") or [])
    if not domains:
        domains = [c.org_domain for c in (run.winners or []) if c.org_domain]
    domains = list(dict.fromkeys(domains))[: int(req.get("max_domains") or 5)]
    stats: Dict[str, Any] = {"domains": len(domains), "pages": 0}
    records: List[Record] = []
    for dom in domains:
        urls = domain_urls(run.deps.fetcher, dom, req.get("path") or "", int(req.get("max_pages") or 15))
        for page in run.deps.fetcher.get_many(urls):
            if not page.ok:
                continue
            stats["pages"] += 1
            records += [Record(source=spec.name, url=page.url, data=dict(row, domain=dom))
                        for row in extract_from_page(spec, run, page)]
    return finish(spec, run, records, stats)

"""Pieces shared by the adapters: auth, following URLs, reading pages with CSS selectors."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from brain.cache import Cache
from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.normalize import Collector, get_path, hook
from brain.discovery.policy import throttle
from brain.textutil import squash


def auth_parts(spec: SourceSpec) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(headers, params) carrying the source's API key, if it needs one."""
    if spec.auth is None:
        return {}, {}
    key = os.getenv(spec.auth.env, "")
    headers = {spec.auth.header: key} if spec.auth.header and key else {}
    params = {spec.auth.param: key} if spec.auth.param and key else {}
    return headers, params


def get_json(spec: SourceSpec, run, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    fetcher = run.deps.fetcher
    headers, auth_params = auth_parts(spec)
    all_params = {**(params or {}), **auth_params} or None
    cached = getattr(run.deps, "cache", None) is not None and \
        run.deps.cache.get("api", Cache.key(url, all_params or {})) is not None
    throttle(spec, cached=cached or bool(getattr(run.deps.settings, "replay", "")), owner=run.deps.fetcher)
    if spec.cost == "paid" and not cached and hasattr(fetcher, "paid_calls"):
        fetcher.paid_calls[spec.name] = fetcher.paid_calls.get(spec.name, 0) + 1
    return fetcher.get_json(url, params=all_params, headers=headers or None)


def post_json(spec: SourceSpec, run, url: str, body: Dict[str, Any]) -> Any:
    fetcher = run.deps.fetcher
    headers, _ = auth_parts(spec)
    cached = getattr(run.deps, "cache", None) is not None and run.deps.cache.get("api_post", Cache.key(url, body)) is not None
    throttle(spec, cached=cached or bool(getattr(run.deps.settings, "replay", "")), owner=fetcher)
    if spec.cost == "paid" and not cached and hasattr(fetcher, "paid_calls"):
        fetcher.paid_calls[spec.name] = fetcher.paid_calls.get(spec.name, 0) + 1
    return fetcher.post_json(url, body, headers=headers or None)


def get_text(spec: SourceSpec, run, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    _, auth_params = auth_parts(spec)
    all_params = {**(params or {}), **auth_params} or None
    cached = getattr(run.deps, "cache", None) is not None and \
        run.deps.cache.get("api_text", Cache.key(url, all_params or {})) is not None
    throttle(spec, cached=cached or bool(getattr(run.deps.settings, "replay", "")), owner=run.deps.fetcher)
    return run.deps.fetcher.get_api_text(url, params=all_params)


def select_one(node, selector: str, base_url: str) -> Optional[str]:
    """'css' -> text; 'css@attr' -> attribute (hrefs made absolute); '@attr' -> the node's own attribute."""
    css, _, attr = selector.partition("@")
    el = node.select_one(css) if css.strip() else node
    if el is None:
        return None
    if attr and attr != "text":
        val = el.get(attr)
        if isinstance(val, list):
            val = " ".join(val)
        if val and attr in ("href", "src"):
            val = urljoin(base_url, val)
        return val or None
    return squash(el.get_text(" ", strip=True)) or None


def select_records(spec: SourceSpec, html: str, url: str, extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Items on a page per spec.select: {"item": css, "fields": {name: selector}, "limit": n}."""
    from bs4 import BeautifulSoup

    sel = spec.select or {}
    fields: Dict[str, str] = sel.get("fields") or {}
    if not fields:
        return []
    soup = BeautifulSoup(html, "lxml")
    nodes = soup.select(sel["item"]) if sel.get("item") else [soup]
    out = []
    for node in nodes[: int(sel.get("limit") or spec.max_items)]:
        row = {k: select_one(node, v, url) for k, v in fields.items()}
        if any(row.values()):
            row["page_url"] = url
            row.update(extra or {})
            out.append(row)
    return out


def extract_from_page(spec: SourceSpec, run, page, hit: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """A source's hook decides what a page holds; otherwise its CSS selectors do."""
    fn = hook(spec, "extract")
    if fn:
        return list(fn(spec, run, hit or {}, page) or [])
    return select_records(spec, page.html, page.url, {"hit": hit} if hit else None)


def follow_items(spec: SourceSpec, run, items: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Record]:
    """spec.follow = {"field": path, "parse": "rss"|"html"|"json", "as": "page", "max": n}: read the URL each
    item names and pair it with the item."""
    from brain.discovery.adapters.rss import parse_feed

    f = spec.follow
    pairs = [(it, get_path(it, f["field"])) for it in items]
    pairs = [(it, u) for it, u in pairs if isinstance(u, str) and u.startswith("http")][: int(f.get("max") or 30)]
    if not pairs:
        return []
    fetcher = run.deps.fetcher
    as_ = f.get("as") or "page"
    out: List[Record] = []
    if f.get("parse") == "json":
        for it, u in pairs:
            data = get_json(spec, run, u)
            stats["followed"] = stats.get("followed", 0) + 1
            if data is not None:
                out.append(Record(source=spec.name, url=u, data={"item": it, as_: data}))
        return out
    for (it, u), page in zip(pairs, fetcher.get_many([u for _, u in pairs])):
        if not page.ok:
            continue
        stats["followed"] = stats.get("followed", 0) + 1
        if f.get("parse") == "rss":
            out.append(Record(source=spec.name, url=u, data={"item": it, as_: parse_feed(page.html)}))
        else:
            for row in extract_from_page(spec, run, page, it):
                out.append(Record(source=spec.name, url=page.url, data={"item": it, **row}))
    return out


def text_matches(pattern: Optional[str], *texts: str) -> bool:
    return not pattern or bool(re.search(pattern, " ".join(t or "" for t in texts), re.I))


def post_process(spec: SourceSpec, run, contacts: List, stats: Dict[str, Any]) -> List:
    """A source's hook `post(spec, run, contacts, stats) -> contacts` (e.g. reading PDFs for author emails)."""
    fn = hook(spec, "post")
    return list(fn(spec, run, contacts, stats)) if fn else contacts


def finish(spec: SourceSpec, run, records: List[Record], stats: Dict[str, Any]):
    col = Collector(spec, run)
    for r in records:
        col.add(r)
    stats["records"] = stats.get("records", 0) + len(records)
    stats["found"] = len(col)
    return post_process(spec, run, col.contacts(), stats), stats

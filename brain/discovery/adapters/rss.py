"""
rss: RSS 2.0 / Atom feeds.

    "request": {"url": "https://news.google.com/rss/search?q={term|quote}"},
    "query":   {"terms": ["topics"], "max": 3}

Each feed item becomes a record {"channel": {...}, "item": {...}}. `parse_feed` is also used by other
adapters' "follow": {"parse": "rss"} (e.g. a podcast directory whose results point at feeds).
"""

from __future__ import annotations

from typing import Any, Dict, List

from brain.discovery.adapters.common import finish
from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.normalize import campaign_terms, campaign_vars, render
from brain.textutil import squash


def parse_feed(xml: str, max_items: int = 20) -> Dict[str, Any]:
    """Channel fields (title, author, description, email, owner, link), "episodes" (the first 5 items as
    title/date, kept for podcast sources) and "items" (title, link, date, description, author)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml, "xml")
    ch = soup.find("channel")
    atom = None
    if ch is None:
        atom = soup.find("feed")
        if atom is None:
            return {}
        ch = atom

    def first(tag: str, node=None) -> str:
        el = (node or ch).find(tag)
        return squash(el.get_text(" ")) if el else ""

    owner = ch.find("itunes:owner") or ch.find("owner")
    email = owner_name = ""
    if owner is not None:
        e = owner.find("itunes:email") or owner.find("email")
        n = owner.find("itunes:name") or owner.find("name")
        email = squash(e.get_text()) if e else ""
        owner_name = squash(n.get_text()) if n else ""
    entries = ch.find_all("entry" if atom is not None else "item")
    episodes = []
    for it in entries[:5]:
        t = it.find("title")
        d = it.find("updated" if atom is not None else "pubDate")
        episodes.append({"title": squash(t.get_text(" ") if t else "", 160), "date": squash(d.get_text() if d else "")})
    items = []
    for it in entries[:max_items]:
        link = it.find("link")
        href = (link.get("href") or link.get_text()) if link is not None else ""
        items.append({"title": first("title", it), "link": squash(href),
                      "date": first("updated", it) if atom is not None else first("pubDate", it),
                      "description": squash(first("description", it) or first("summary", it), 400),
                      "author": first("dc:creator", it) or first("author", it), "source": first("source", it)})
    link_el = ch.find("link")
    return {"title": first("title"), "author": first("itunes:author") or first("author"),
            "description": squash(first("description") or first("subtitle"), 400), "email": email.lower(),
            "owner": owner_name, "link": squash((link_el.get("href") or link_el.get_text()) if link_el else ""),
            "episodes": episodes, "items": items}


def collect(spec: SourceSpec, run):
    stats: Dict[str, Any] = {"feeds": 0}
    base = campaign_vars(run.spec, spec.query, run.playbook)
    url_t = spec.request["url"]
    terms = campaign_terms(run.spec, spec.query) if "{term" in url_t else [""]
    urls = list(dict.fromkeys(render(url_t, dict(base, term=t)) for t in terms))
    records: List[Record] = []
    for url, page in zip(urls, run.deps.fetcher.get_many(urls)):
        if not page.ok:
            continue
        feed = parse_feed(page.html, spec.max_items)
        if not feed:
            continue
        stats["feeds"] += 1
        channel = {k: v for k, v in feed.items() if k != "items"}
        records += [Record(source=spec.name, url=it.get("link") or url, data={"channel": channel, "item": it, "feed_url": url})
                    for it in feed["items"]]
    return finish(spec, run, records, stats)

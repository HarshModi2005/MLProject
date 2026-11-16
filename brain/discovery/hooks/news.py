"""News: recent articles on the topic -> the journalists in their bylines."""

from __future__ import annotations

from typing import Dict, List, Optional

from brain.models import Contact, Evidence
from brain.textutil import country_from_domain, registrable_domain, squash


def extract(spec, run, hit: Dict, page) -> List[Dict]:
    from brain.tools import authors_from_article

    return [{"name": name, "author_url": author_url, "page_url": page.url, "article": hit}
            for name, author_url in authors_from_article(page.html, page.url)]


def to_contact(spec, record, run) -> Optional[Contact]:
    d = record.data
    art = d.get("article") or {}
    dom = registrable_domain(d["page_url"])
    c = Contact(kind="person", name=d["name"], title="Journalist", org=art.get("source") or None, org_domain=dom,
                country=country_from_domain(dom), sources=[spec.name])
    c.add_url(d.get("author_url"))
    c.evidence.append(Evidence(kind="publication", text=squash(art.get("title") or "", 200), url=art.get("url"),
                               detail=squash(f"{art.get('source')} {art.get('date')}: {art.get('snippet')}", 300)))
    c.signals["recent_coverage"] = 1.0  # a byline on a recent article on the topic
    return c


def key(c: Contact) -> str:
    return c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    art = record.data.get("article") or {}
    existing.evidence.append(Evidence(kind="publication", text=art.get("title") or "", url=art.get("url")))

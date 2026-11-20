"""Podcasts: a directory result plus its RSS feed -> the host (a person) or the show (an organisation),
reached through the email the feed itself publishes."""

from __future__ import annotations

from typing import Optional

from brain.models import Contact, EmailCandidate, Evidence
from brain.textutil import clean_name, looks_like_person_name, squash


def _recent(date_text: str, days: int = 90) -> bool:
    from brain.channels import _recent as recent

    return recent(date_text, days)


def to_contact(spec, record, run) -> Optional[Contact]:
    r, info, feed = record.data.get("item") or {}, record.data.get("feed") or {}, record.url
    email = str(info.get("email") or "")
    if not info or "@" not in email:
        return None
    host = clean_name(str(info.get("owner") or info.get("author") or r.get("artistName") or ""))
    title = str(info.get("title") or r.get("collectionName") or "")
    person = looks_like_person_name(host)
    # No org_domain: the feed's email domain is often a host or mail provider (spotify.com, gmail.com), and
    # reading that site would replace the show's identity with the platform's.
    c = Contact(kind="person" if person else "org", name=host if person else title, title="Podcast host",
                org=title, sources=[spec.name], flags=["channel_verified"])
    c.add_url(r.get("collectionViewUrl") or feed)
    c.add_email(EmailCandidate(address=email, confidence=0.85, method="feed", source_url=feed))
    c.evidence.append(Evidence(kind="profile", text=squash(f"Hosts the podcast {title}: {info.get('description') or ''}", 300),
                               url=r.get("collectionViewUrl")))
    eps = info.get("episodes") or []
    for ep in eps[:3]:
        c.evidence.append(Evidence(kind="publication", text=f"Episode: {ep['title']}", detail=ep["date"] or None))
    if eps and _recent(eps[0]["date"]):
        c.signals["recent_coverage"] = 1.0
    if r.get("trackCount"):
        c.signals["audience"] = min(1.0, float(r["trackCount"]) / 200)
        c.signals["episodes"] = float(r["trackCount"])
    c.topics = [g for g in r.get("genres") or [] if g != "Podcasts"][:4]
    return c


def key(c: Contact) -> str:
    # Every feed is its own show: two shows by one host stay two candidates, as before.
    return (c.emails[0].source_url if c.emails and c.emails[0].source_url else c.name).lower()


def merge(existing: Contact, new: Contact, record) -> None:
    return None

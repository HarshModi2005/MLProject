"""Google Play: app developers on the topic, from the app page's structured data and its developer email."""

from __future__ import annotations

import html
import json
import re
from typing import Dict, List, Optional

from brain.discovery.normalize import add_published_email
from brain.models import Contact, Evidence
from brain.textutil import registrable_domain, squash

_LD = re.compile(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S)
_MAILTO = re.compile(r'mailto:([^"?\s<>]+@[^"?\s<>]+)')
# The developer link ("/store/apps/dev?id=..." or "/store/apps/developer?id=...") carries the developer's name;
# the structured data's author is sometimes the app's tagline instead.
_DEV_LINK = re.compile(r'href="/store/apps/dev(?:eloper)?\?id=[^"]*"[^>]*>\s*(?:<[^>]+>\s*)*([^<]{2,100}?)\s*<')


def extract(spec, run, hit: Dict, page) -> List[Dict]:
    app = None
    for block in _LD.findall(page.html or ""):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("@type") == "SoftwareApplication":
            app = data
            break
    if not app:
        return []
    author = app.get("author") or {}
    rating = app.get("aggregateRating") or {}
    emails = [e for e in _MAILTO.findall(page.html) if not e.lower().endswith(("google.com", "android.com"))]
    dev = _DEV_LINK.search(page.html or "")
    return [{"developer": (dev.group(1) if dev else None) or author.get("name"), "website": author.get("url"),
             "app": app.get("name"),
             "description": app.get("description"), "category": app.get("applicationCategory"),
             "rating": rating.get("ratingValue"), "ratings": rating.get("ratingCount"),
             "emails": list(dict.fromkeys(emails)), "url": page.url}]


def to_contact(spec, record, run) -> Optional[Contact]:
    d = record.data
    dev = squash(html.unescape(d.get("developer") or ""), 100)
    site = d.get("website") or ""
    if not dev:
        return None
    if ("," in dev or len(dev.split()) > 6) and site.startswith("http") and "google.com" not in site:
        # Keyword-stuffed developer names ("Doctor Appointment, Consultation, Meds, Tests&more"): use the website's name.
        dev = registrable_domain(site).split(".")[0].replace("-", " ").title()
    c = Contact(kind="org", name=dev, org=dev, sources=[spec.name])
    if site.startswith("http") and "google.com" not in site:
        c.org_domain = registrable_domain(site)
        c.add_url(site)
    c.add_url(d.get("url"))
    rating = f", rated {float(d['rating']):.1f} from {int(float(d['ratings'])):,} ratings" if d.get("rating") and d.get("ratings") else ""
    c.evidence.append(Evidence(kind="page", url=d.get("url"),
                               text=squash(f"Publishes the app {d.get('app')} on Google Play ({(d.get('category') or '').lower()}{rating})", 220),
                               detail=squash(d.get("description") or "", 250) or None))
    for e in d.get("emails") or []:  # the first address the page publishes that someone reads (not support@)
        if add_published_email(c, e, 0.8, d.get("url")):
            break
    c.signals["app_publisher"] = 1.0
    if d.get("ratings"):
        c.signals["reviews"] = float(d["ratings"])
    return c


def key(c: Contact) -> str:
    return c.org_domain or c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    seen = {e.text for e in existing.evidence}
    existing.evidence += [e for e in new.evidence if e.text not in seen]
    for e in new.emails:
        existing.add_email(e)
    existing.signals["reviews"] = existing.signals.get("reviews", 0) + new.signals.get("reviews", 0)

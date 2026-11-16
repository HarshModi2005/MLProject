"""
Records -> Contacts, plus the small template language manifests use.

Templates: "{path}" reads a dotted path from the record (or the campaign variables when querying);
"{path|lower}", "{path|upper}", "{path|quote}" apply a filter. A missing path renders as "".

Generic field map (manifest "map"), each value a record path or "=literal":
    name, kind, title, org, website, email, url, bio, location, country, topics, key
A source whose records need judgement (is this a person or a show?) ships hooks instead (see hooks/).
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.policy import allowed_field
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.textutil import looks_like_person_name, registrable_domain, squash

_TPL = re.compile(r"\{([^{}|]+)(?:\|(\w+))?\}")


def get_path(data: Any, path: str) -> Any:
    """'a.b.0.c' into nested dicts/lists; None when any step is missing."""
    cur = data
    for part in [p for p in (path or "").split(".") if p]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
        if cur is None:
            return None
    return cur


def render(template: Any, values: Dict[str, Any]) -> Any:
    """Render a template string; non-strings (e.g. an int API parameter) pass through unchanged."""
    if not isinstance(template, str):
        return template

    def sub(m: "re.Match[str]") -> str:
        v = get_path(values, m.group(1).strip())
        s = "" if v is None else str(v)
        f = m.group(2)
        if f == "lower":
            return s.lower()
        if f == "upper":
            return s.upper()
        if f == "quote":
            return quote(s)
        return s

    return _TPL.sub(sub, template)


# ── campaign variables for queries ───────────────────────────────────────────

def campaign_terms(spec: CampaignSpec, q: Dict[str, Any]) -> List[str]:
    """The search terms a manifest asks for: q["terms"] lists "topics", "expanded_terms:2", ... in order."""
    out: List[str] = []
    for sel in q.get("terms") or ["topics"]:
        field, _, n = str(sel).partition(":")
        vals = list(getattr(spec, field, []) or [])
        out += vals[: int(n)] if n.isdigit() else vals
    out = list(dict.fromkeys(t for t in out if t))
    return out[: int(q.get("max") or len(out) or 1)]


def campaign_vars(spec: CampaignSpec, q: Dict[str, Any], playbook=None) -> Dict[str, Any]:
    from brain.searchplan import _country_name

    country = spec.countries[0] if len(spec.countries) == 1 else str(q.get("default_country") or "")
    place = (spec.regions or [_country_name(c) for c in spec.countries] or [""])[0]
    roles = spec.persona.roles or (list(getattr(playbook, "default_roles", []) or []))
    return {"country": country, "place": place, "role": (roles or [""])[0], "objective": spec.objective}


# ── hooks ────────────────────────────────────────────────────────────────────

_HOOKS: Dict[str, Any] = {}


def hook(spec: SourceSpec, name: str) -> Optional[Callable]:
    if not spec.hooks:
        return None
    if spec.hooks not in _HOOKS:
        _HOOKS[spec.hooks] = importlib.import_module(spec.hooks)
    return getattr(_HOOKS[spec.hooks], name, None)


# ── record -> contact ────────────────────────────────────────────────────────

def _field(spec: SourceSpec, data: Dict[str, Any], name: str) -> Any:
    path = spec.map.get(name)
    if not path or not allowed_field(spec, name):
        return None
    if path.startswith("="):
        return path[1:]
    return get_path(data, path)


def generic_contact(spec: SourceSpec, record: Record) -> Optional[Contact]:
    data = record.data
    if any(not get_path(data, p) for p in spec.require):
        return None
    name = squash(str(_field(spec, data, "name") or ""), 120)
    if not name:
        return None
    kind = _field(spec, data, "kind")
    if kind not in ("person", "org"):
        kind = spec.kinds[0] if len(spec.kinds) == 1 else ("person" if looks_like_person_name(name) else "org")
    c = Contact(kind=kind, name=name, sources=[spec.name])
    title = _field(spec, data, "title")
    org = _field(spec, data, "org")
    c.title = squash(str(title), 120) if title else None
    c.org = squash(str(org), 120) if org else (name if kind == "org" else None)
    site = _field(spec, data, "website")
    if site and str(site).startswith("http"):
        c.org_domain = registrable_domain(str(site))
        c.add_url(str(site))
    url = _field(spec, data, "url")
    if url and str(url).startswith("http"):
        c.add_url(str(url))
    email = str(_field(spec, data, "email") or "").replace("mailto:", "").split("?")[0].strip().lower()
    if "@" in email:
        c.add_email(EmailCandidate(address=email, confidence=spec.email_confidence, method="published",
                                   source_url=record.url or (str(url) if url else None)))
    bio = _field(spec, data, "bio")
    if bio:
        c.bio = squash(str(bio), 600)
    loc = _field(spec, data, "location")
    if loc:
        c.location_text = squash(str(loc), 120)
    country = _field(spec, data, "country")
    if country and len(str(country)) == 2:
        c.country = str(country).upper()
    topics = _field(spec, data, "topics")
    if topics:
        c.topics = [squash(str(t), 60) for t in (topics if isinstance(topics, list) else str(topics).split(","))
                    if str(t).strip()][:8]
    key = _field(spec, data, "key")
    if key:
        c.add_key(f"{spec.name}:{key}")
    if spec.evidence:
        text = squash(render(spec.evidence, data), 300)
        if text:
            c.evidence.append(Evidence(kind="page", text=text, url=record.url or (str(url) if url else None)))
    for sig, v in spec.signals.items():
        c.signals[sig] = float(v)
    return c


def to_contact(spec: SourceSpec, record: Record, run=None) -> Optional[Contact]:
    fn = hook(spec, "to_contact")
    return fn(spec, record, run) if fn else generic_contact(spec, record)


def contact_key(spec: SourceSpec, c: Contact) -> str:
    fn = hook(spec, "key")
    if fn:
        return fn(c)
    if c.kind == "org" and c.org_domain:
        return c.org_domain
    return c.name.lower()


def merge(spec: SourceSpec, into: Contact, new: Contact, record: Optional[Record] = None) -> None:
    """A repeat sighting of a candidate this source already returned."""
    fn = hook(spec, "merge")
    if fn:
        fn(into, new, record)
        return
    seen = {e.text for e in into.evidence}
    into.evidence += [e for e in new.evidence if e.text not in seen]
    for e in new.emails:
        into.add_email(e)
    for u in new.urls:
        into.add_url(u)
    for k, v in new.signals.items():
        into.signals[k] = max(into.signals.get(k, 0.0), v)


class Collector:
    """Ordered, de-duplicated contacts for one source run; counts what is new per query."""

    def __init__(self, spec: SourceSpec, run=None):
        self.spec, self.run = spec, run
        self.by_key: Dict[str, Contact] = {}

    def add(self, record: Record) -> bool:
        c = to_contact(self.spec, record, self.run)
        if c is None:
            return False
        k = contact_key(self.spec, c)
        if k in self.by_key:
            merge(self.spec, self.by_key[k], c, record)
            return False
        self.by_key[k] = c
        return True

    def contacts(self) -> List[Contact]:
        return list(self.by_key.values())

    def __len__(self) -> int:
        return len(self.by_key)

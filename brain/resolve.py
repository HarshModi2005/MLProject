"""
Entity resolution: collapse the same person found through different sources into one Contact.

Strong identifiers (ORCID, OpenAlex id, email) merge unconditionally. Otherwise two records merge
only when the names are compatible AND the organisations agree — a deliberately conservative rule,
because merging two different people sends an email with someone else's work in it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from brain.models import Contact, stable_id
from brain.emailfinder import name_match
from brain.textutil import name_tokens, registrable_domain

try:
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b)
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return 100 * SequenceMatcher(None, a, b).ratio()


def names_compatible(a: str, b: str) -> bool:
    """'S. Kumar' ~ 'Sanjay Kumar' ~ 'Sanjay K. Kumar'; 'Sanjay Kumar' !~ 'Sunil Kumar'."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb or ta[-1] != tb[-1]:
        return False
    fa, fb = ta[0], tb[0]
    if len(ta) == 1 or len(tb) == 1:
        return False  # a lone surname is not enough
    if len(fa) == 1 or len(fb) == 1:
        return fa[0] == fb[0]
    if fa != fb:
        return False
    # Middle names/initials must not contradict each other.
    ma, mb = ta[1:-1], tb[1:-1]
    for x, y in zip(ma, mb):
        if x[0] != y[0]:
            return False
    return True


def orgs_compatible(a: Contact, b: Contact) -> bool:
    if a.org_ror and b.org_ror:
        return a.org_ror == b.org_ror
    if a.org_domain and b.org_domain:
        return registrable_domain(a.org_domain) == registrable_domain(b.org_domain)
    if a.org and b.org:
        return _ratio(a.org.lower(), b.org.lower()) >= 88
    return False  # unknown organisation on either side: do not guess


def strong_keys(c: Contact) -> List[str]:
    """Identifiers that prove two records are the same person. An email only counts when it carries the
    person's name or came from a source that ties it to them (ORCID, the user's own list); otherwise one
    mis-attributed address would chain unrelated people into a single record."""
    keys = list(c.keys)
    if c.kind == "org" and c.org_domain:
        keys.append(f"domain:{registrable_domain(c.org_domain)}")  # one website, one organisation
    for e in c.emails:
        if e.method == "pattern" or e.confidence < 0.8:
            continue
        if e.method in ("orcid", "provided") or name_match(c.name, e.address) == 1.0:
            keys.append(f"email:{e.address}")
    return keys


def merge_into(a: Contact, b: Contact) -> Contact:
    """Fold b into a, keeping the richer value for every field."""
    if len(b.name) > len(a.name):
        a.name = b.name
    for attr in ("title", "org", "org_domain", "org_ror", "country", "location_text"):
        if not getattr(a, attr) and getattr(b, attr):
            setattr(a, attr, getattr(b, attr))
    if b.bio and len(b.bio) > len(a.bio or ""):
        a.bio = b.bio
    for e in b.emails:
        a.add_email(e)
    for u in b.urls:
        a.add_url(u)
    for k in b.keys:
        a.add_key(k)
    seen = {ev.text.lower() for ev in a.evidence}
    for ev in b.evidence:
        if ev.text.lower() not in seen:
            a.evidence.append(ev)
            seen.add(ev.text.lower())
    a.topics = list(dict.fromkeys(a.topics + b.topics))[:12]
    for k, v in b.signals.items():
        a.signals[k] = max(a.signals.get(k, 0.0), v)
    a.sources = list(dict.fromkeys(a.sources + b.sources))
    # The pre-judge screen: keep the best one; the merged record is rejected only if that screen rejected it.
    best = a
    if b.prejudge is not None and (a.prejudge is None or b.prejudge > a.prejudge):
        a.prejudge, a.prejudge_reason, best = b.prejudge, b.prejudge_reason, b
    best_rejected = "prejudge_reject" in best.flags
    a.flags = list(dict.fromkeys(a.flags + b.flags))
    if a.prejudge is not None and not best_rejected:
        a.flags = [f for f in a.flags if f != "prejudge_reject"]
    return a


def assign_id(c: Contact) -> str:
    """Deterministic id from the strongest identifier, so the same person keeps their id across runs."""
    for prefix in ("orcid:", "openalex:"):
        for k in c.keys:
            if k.startswith(prefix):
                return stable_id(k)
    emails = [k for k in strong_keys(c) if k.startswith("email:")]
    if emails:
        return stable_id(sorted(emails)[0])
    return stable_id("name:" + " ".join(name_tokens(c.name)), c.org_ror or c.org_domain or (c.org or "").lower())


def resolve(contacts: Iterable[Contact]) -> List[Contact]:
    contacts = list(contacts)
    parent = list(range(len(contacts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_key: Dict[str, int] = {}
    for i, c in enumerate(contacts):
        for k in strong_keys(c):
            j = by_key.get(k)
            if j is None:
                by_key[k] = i
            elif k.startswith(("orcid:", "openalex:", "domain:")) or names_compatible(contacts[j].name, c.name):
                union(i, j)  # a shared email alone never merges two different names

    # Fuzzy pass, blocked by surname so it stays near-linear.
    blocks: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(contacts):
        tokens = name_tokens(c.name)
        if tokens:
            blocks[tokens[-1]].append(i)
    for members in blocks.values():
        if len(members) < 2 or len(members) > 400:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if find(i) == find(j):
                    continue
                if names_compatible(contacts[i].name, contacts[j].name) and orgs_compatible(contacts[i], contacts[j]):
                    union(i, j)

    groups: Dict[int, Contact] = {}
    for i, c in enumerate(contacts):
        root = find(i)
        if root in groups:
            merge_into(groups[root], c)
        else:
            groups[root] = c
    out = list(groups.values())
    for c in out:
        c.id = assign_id(c)
    return out

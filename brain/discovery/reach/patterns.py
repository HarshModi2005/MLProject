"""
Email conventions shared across campaigns, and clearly-flagged guesses.

v2 could only guess an address when 2-3 people at the same organisation had confirmed addresses in the SAME
campaign, which almost never happens for startups and newsrooms (the press scenario found the right
journalists and no addresses). Here every confirmed address from every campaign votes for its domain's
convention (brain_domain_patterns), so the second campaign that meets an outlet already knows it.

    pattern_global  the domain's convention confirmed by earlier addresses     confidence 0.55-0.62
    pattern_prior   a common convention, for playbooks that accept guesses     confidence 0.50
                    (Playbook.accept_guessed): first@ for companies, first.last@ for media
Both require the domain to accept mail (MX), and both are marked so drafts and reviewers know.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from brain.emailfinder import _patterns, has_mx, name_match
from brain.models import Contact, EmailCandidate
from brain.textutil import FREE_MAIL, registrable_domain

CONFIRMED = ("mailto", "text", "orcid", "directory", "pdf", "published", "feed", "provided", "llm")
GUESSED = ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior")


def votes(contacts: Iterable[Contact]) -> Dict[str, Dict[str, int]]:
    """{domain: {pattern: n}} from addresses that were found (not guessed) and carry the person's name."""
    out: Dict[str, Dict[str, int]] = {}
    for c in contacts:
        if c.kind != "person":
            continue
        for e in c.emails:
            if e.method not in CONFIRMED or e.confidence < 0.6 or name_match(c.name, e.address) < 1.0:
                continue
            local, domain = e.address.split("@")
            if registrable_domain(domain) in FREE_MAIL:
                continue
            for pname, value in _patterns(c.name).items():
                if value == local:
                    out.setdefault(domain, {})[pname] = out.setdefault(domain, {}).get(pname, 0) + 1
                    break
    return out


def learn(store, contacts: Iterable[Contact]) -> int:
    if store is None or not hasattr(store, "record_patterns"):
        return 0
    v = votes(contacts)
    try:
        return store.record_patterns(v) if v else 0
    except Exception:
        return 0


def _prior(domain: str, c: Contact) -> Optional[str]:
    from brain.discovery.precision.orgtype import classify

    cls, _ = classify(Contact(kind="org", name=c.org or domain, org=c.org, org_domain=domain))
    if cls in ("academic", "government", "publisher", "directory"):
        return None  # large institutions: conventions vary too much to guess
    return "first.last" if cls == "media" or "journalist" in (c.title or "").lower() else "first"


def guess(c: Contact, store=None, accept_prior: bool = False, domain: Optional[str] = None) -> Optional[EmailCandidate]:
    """An address for a person with a known organisation website and none found, or None."""
    domain = domain or (registrable_domain(c.org_domain) if c.org_domain else None)
    if c.kind != "person" or not domain or domain in FREE_MAIL:
        return None
    known = store.domain_pattern(domain) if store is not None and hasattr(store, "domain_pattern") else None
    names = _patterns(c.name)
    if known and known[1] >= 2 and names.get(known[0]):
        if has_mx(domain) is False:
            return None
        return EmailCandidate(address=f"{names[known[0]]}@{domain}", confidence=0.62 if known[1] >= 5 else 0.55,
                              method="pattern_global")
    if accept_prior:
        pname = _prior(domain, c)
        if pname and names.get(pname) and has_mx(domain) is True:
            return EmailCandidate(address=f"{names[pname]}@{domain}", confidence=0.5, method="pattern_prior")
    return None


def fill(contacts: Iterable[Contact], store=None, accept_prior: bool = False) -> int:
    """Add a guess to every person still without an address. Returns how many were added."""
    n = 0
    for c in contacts:
        if c.emails:
            continue
        g = guess(c, store, accept_prior)
        if g:
            c.add_email(g)
            c.flags.append("email_guessed")
            n += 1
    return n


def threshold(settings, playbook) -> float:
    """The confidence an address needs to count as reachable: lower for playbooks that accept flagged guesses."""
    base = float(getattr(settings, "min_email_confidence", 0.6))
    return min(base, 0.5) if getattr(playbook, "accept_guessed", False) else base

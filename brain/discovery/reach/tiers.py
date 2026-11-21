"""
How far an address can be trusted, as one label the rest of the engine (and a human reviewer) can act on.

    verified      a checker confirmed the mailbox exists AND the address is tied to this person
    attributed    found with evidence it is this person's (their own page, ORCID, their name in the address), but
                  nobody checked that the mailbox exists
    inbox         a shared inbox of the organisation (partnerships@, info@)
    guess         built from an email convention; the person's name is in it but nothing found it anywhere
    unattributed  an address with nothing tying it to this person (a namesake, a previous employer, someone else's)
    none          no address

Only `verified` (and a verified shared inbox) is safe to send without a person looking at the address first.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from brain.emailfinder import name_match
from brain.models import Contact, EmailCandidate
from brain.textutil import FREE_MAIL

GUESSED = ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior")
INBOX_METHODS = ("inbox", "site")
TIED_BY_SOURCE = ("provided", "orcid", "hunter")  # the source itself links the address to this person

TIERS = ("verified", "attributed", "inbox", "guess", "unattributed", "none")


def identity_ok(c: Contact, e: EmailCandidate) -> bool:
    """Is there evidence this address belongs to this person (not just to someone at the same place)?"""
    if e.method in TIED_BY_SOURCE or c.kind == "org":
        return True  # an organisation's address is the organisation's; its domain was matched when it was found
    return name_match(c.name, e.address) >= 1.0


def email_tier(c: Contact, e: Optional[EmailCandidate]) -> str:
    if e is None:
        return "none"
    if e.deliverability in ("undeliverable",):
        return "unattributed"  # callers normally drop these; never trusted
    if e.method in INBOX_METHODS:
        return "verified" if e.deliverability == "deliverable" else "inbox"
    if e.method in GUESSED:
        return "verified" if e.deliverability == "deliverable" and name_match(c.name, e.address) >= 1.0 else "guess"
    if not identity_ok(c, e):
        return "unattributed"
    if e.deliverability == "deliverable":
        return "verified"
    if c.kind == "person" and e.method != "orcid" and e.address.rsplit("@", 1)[-1].lower() in FREE_MAIL:
        return "guess"  # a personal gmail/yahoo for a professional ask: it may be theirs, nothing says it is read for this
    return "attributed"


def contact_tier(c: Contact) -> str:
    return email_tier(c, c.best_email)


def is_sendable(c: Contact) -> bool:
    """Safe to send without a human re-checking the address."""
    return contact_tier(c) == "verified"


def counts(contacts: Iterable[Contact]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in contacts:
        t = contact_tier(c)
        out[t] = out.get(t, 0) + 1
    return out

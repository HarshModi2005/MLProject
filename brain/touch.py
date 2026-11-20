"""
Touch ledger logic: who has the engine already put in front of the user (or emailed), and may they be offered again?

Pure functions, no database. The tables and queries live in `brain/store.py`; this module decides

  * which identity keys a contact has (any one matching an earlier contact means "the same person"),
  * whether an earlier touch blocks a candidate under the campaign's no-repeat mode,
  * whether two names at one organisation are plausibly one person (deliberately looser than entity resolution:
    wrongly skipping a different person costs one prospect, emailing the same person twice costs trust).

States of a touch
    surfaced -> drafted -> approved -> sent -> replied | bounced | unsubscribed | spam    (progress, never goes back)
    rejected   the user said no. Final: never offered again except by an explicit `release`
    standby    the user said "not now". Parked: skipped by normal batches, offered again only through recycling
    external   imported do-not-contact / "already emailed" list

Hard states never come back; soft states (surfaced, drafted, standby) are a fallback pool (`recycle`).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from brain.models import Contact
from brain.textutil import name_tokens, registrable_domain

SURFACED, DRAFTED, APPROVED, SENT, REPLIED = "surfaced", "drafted", "approved", "sent", "replied"
BOUNCED, UNSUBSCRIBED, SPAM = "bounced", "unsubscribed", "spam"
REJECTED, STANDBY, EXTERNAL = "rejected", "standby", "external"

STATES = (SURFACED, DRAFTED, APPROVED, SENT, REPLIED, BOUNCED, UNSUBSCRIBED, SPAM, REJECTED, STANDBY, EXTERNAL)
RANK = {SURFACED: 1, DRAFTED: 2, APPROVED: 3, SENT: 4, REPLIED: 5, BOUNCED: 6, UNSUBSCRIBED: 7, SPAM: 8}
NEVER_AGAIN = {BOUNCED, UNSUBSCRIBED, SPAM, EXTERNAL}  # whatever the mode
CONTACTED = {SENT, REPLIED, BOUNCED, UNSUBSCRIBED, SPAM}  # an email left, or the address is dead
SOFT = {SURFACED, DRAFTED, STANDBY}  # shown but undecided / parked: the recycle pool
COUNTS_FOR_ORG_CAP = {SURFACED, DRAFTED, APPROVED, SENT, REPLIED, STANDBY}  # not rejected: that prospect is gone

MODES = ("series", "workspace", "cooldown")
RECYCLE = ("ask", "auto", "never")

# Why a candidate was dropped; the funnel reports these.
REASONS = ("do_not_contact", "already_contacted", "already_surfaced", "rejected_earlier", "on_standby")


class Scope:
    """What "again" means for one campaign: its series, the no-repeat mode and (for cooldown) the number of days."""

    def __init__(self, series_id: str = "", mode: str = "series", cooldown_days: int = 90, campaign_id: str = ""):
        if mode not in MODES:
            raise ValueError(f"no-repeat mode must be one of {MODES}, not {mode!r}")
        self.series_id, self.mode, self.cooldown_days = series_id, mode, max(1, int(cooldown_days))
        self.campaign_id = campaign_id  # touches made by this very campaign never block it (resume, re-runs)

    def __repr__(self) -> str:
        return f"Scope({self.series_id!r}, {self.mode!r}, {self.cooldown_days})"


# ── identity ─────────────────────────────────────────────────────────────────

def norm_email(address: str) -> str:
    """lowercase; no +tag; Gmail ignores dots and treats googlemail.com as gmail.com."""
    a = (address or "").strip().lower()
    if "@" not in a:
        return ""
    local, domain = a.rsplit("@", 1)
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}" if local else ""


def org_key(c: Contact) -> str:
    """The organisation a contact belongs to, for the per-org cap: ROR id, else registrable domain, else the name."""
    if c.org_ror:
        return c.org_ror.lower()
    if c.org_domain:
        return registrable_domain(c.org_domain)
    return re.sub(r"\s+", " ", (c.org or "").strip().lower())


def person_key(name: str, org: str) -> str:
    toks = name_tokens(name)
    return f"person:{' '.join(toks)}|{org}" if len(toks) >= 2 and org else ""


def keys_for(c: Contact) -> List[str]:
    """Every identifier under which this contact may be known. Matching ANY of them means the same contact."""
    out: List[str] = []
    org = org_key(c)
    for e in c.emails:
        k = norm_email(e.address)
        if k:
            out.append(f"email:{k}")
    for k in c.keys:  # orcid:, openalex:, email:, github: ...
        k = k.strip().lower()
        if k.startswith("email:"):
            k = "email:" + norm_email(k[6:])
        if k and k != "email:":
            out.append(k)
    if c.kind == "org":
        if c.org_domain:
            out.append(f"org:{registrable_domain(c.org_domain)}")
        elif org:
            out.append(f"org:{org}")
        dm = (c.decision_maker or {}).get("name")
        if dm and person_key(dm, org):  # the person inside the organisation counts as touched too
            out.append(person_key(dm, org))
    else:
        pk = person_key(c.name, org)
        if pk:
            out.append(pk)
    return list(dict.fromkeys(k[:250] for k in out))


def lookup_keys(c: Contact) -> List[str]:
    """Keys to search the ledger with: the contact's own, the organisation-level row (an inbox contacted there counts
    for the people inside it) and the bare-name key external do-not-contact lists use."""
    out = keys_for(c)
    org = org_key(c)
    if c.kind != "org":
        if c.org_domain:
            out.append(f"org:{registrable_domain(c.org_domain)}")
        toks = name_tokens(c.name)
        if len(toks) >= 2:
            out.append(f"name:{' '.join(toks)}")
    return list(dict.fromkeys(out))


def loose_names_compatible(a: str, b: str) -> bool:
    """'S. Kumar' ~ 'Sanjay Kumar' ~ 'Sanjay K. Kumar' ~ 'Sanjay Rao Kumar'. Same surname and the first names agree
    (or one is an initial). Middle names are ignored: here a false match only skips one prospect."""
    ta, tb = name_tokens(a), name_tokens(b)
    if len(ta) < 2 or len(tb) < 2 or ta[-1] != tb[-1]:
        return False
    fa, fb = ta[0], tb[0]
    if len(fa) == 1 or len(fb) == 1:
        return fa[0] == fb[0]
    return fa == fb


# ── policy ───────────────────────────────────────────────────────────────────

def row_blocks(state: str, row_series: str, updated: Optional[datetime], scope: Optional[Scope],
               now: Optional[datetime] = None) -> Optional[str]:
    """The reason an earlier touch blocks a candidate in this scope, or None.

    scope=None is the unscoped check (plain `brain run`, the send gate): only hard facts block, i.e. anyone already
    emailed or whose address is dead, plus imported do-not-contact entries."""
    if state in NEVER_AGAIN:
        return "do_not_contact"
    if scope is None:
        return "already_contacted" if state in CONTACTED else None
    now = now or datetime.utcnow()
    if scope.mode == "cooldown":
        if state == REJECTED:
            return "rejected_earlier"  # a "no" does not expire
        if state == REPLIED:
            return "already_contacted"  # they answered: follow up through the follow-up engine, not a new batch
        age = now - (updated or now)
        if age >= timedelta(days=scope.cooldown_days):
            return None
        return {SENT: "already_contacted", STANDBY: "on_standby"}.get(state, "already_surfaced")
    if state in (SENT, REPLIED):
        return "already_contacted"  # emailed once: never again, in any series
    if scope.mode != "workspace" and row_series != scope.series_id:
        return None  # an earlier, different brief: not this series' business
    if state == REJECTED:
        return "rejected_earlier"
    if state == STANDBY:
        return "on_standby"
    return "already_surfaced"


def advance(current: Optional[str], new: str) -> str:
    """The state after a transition. Progress never goes backwards; rejected/standby are explicit and cannot undo a send."""
    if current is None:
        return new
    if new in (REJECTED, STANDBY):
        return current if current in CONTACTED or current == EXTERNAL else new
    if new == EXTERNAL:
        return new
    if current == EXTERNAL:
        return current
    if new in CONTACTED:
        return new if RANK.get(new, 0) >= RANK.get(current, 0) or current in (REJECTED, STANDBY) else current
    if current in (REJECTED, STANDBY):
        return current  # only release/recall change these
    return new if RANK.get(new, 0) >= RANK.get(current, 0) else current


def summarize(rows: List[Dict[str, str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["state"]] = out.get(r["state"], 0) + 1
    return out

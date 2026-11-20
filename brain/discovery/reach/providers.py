"""
Paid data providers, used only when their key is set (each call is capped per campaign and counted in
funnel["cost"]["paid_calls"]):

    Hunter domain search   HUNTER_API_KEY      personal work addresses at an organisation, with role and confidence
    BuiltWith              BUILTWITH_API_KEY   technologies an organisation's site uses (proof for `uses_tech`)
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from brain.models import Contact, EmailCandidate, Evidence, ProspectProfile
from brain.textutil import squash

_lock = threading.Lock()


def _count(fetcher, name: str, cap: int) -> bool:
    """Reserve one paid call for `name`; False once the per-campaign cap is reached."""
    with _lock:
        calls = getattr(fetcher, "paid_calls", None)
        if calls is None:
            return True
        if calls.get(name, 0) >= cap:
            return False
        calls[name] = calls.get(name, 0) + 1
        return True


def hunter_available() -> bool:
    return bool(os.getenv("HUNTER_API_KEY", ""))


def _hunter_cap(name: str, default: int = 15) -> int:
    try:
        return int(os.getenv("BRAIN_HUNTER_MAX_" + name.upper(), str(default)))
    except ValueError:
        return default


_SENIOR = ("founder", "owner", "ceo", "chief", "president", "director", "head", "vp", "vice president", "partner",
           "manager", "lead")


def _role_rank(position: str, roles: List[str]) -> int:
    """2: the title is one of the roles we want; 1: a senior person (founder, director, head...); 0: anyone else."""
    pos = (position or "").lower()
    if not pos:
        return 0
    if any(r.lower() in pos or pos in r.lower() for r in roles if r):
        return 2
    return 1 if any(w in pos for w in _SENIOR) else 0


def _hunter_candidate(e: dict, org: Contact, rank: int, source: str) -> Contact:
    name = f"{e.get('first_name') or ''} {e.get('last_name') or ''}".strip()
    c = Contact(kind="person", name=name, title=e.get("position") or None, org=org.name, org_domain=org.org_domain,
                country=org.country, location_text=org.location_text, urls=list(org.urls), evidence=list(org.evidence),
                bio=org.bio, sources=org.sources + ["hunter"], signals=dict(org.signals), flags=list(org.flags))
    if rank < 2:
        c.flags.append("role_approx")  # a senior person, but not in the role asked for: the reviewer should know
    status = ((e.get("verification") or {}).get("status") or "").lower()
    conf = e.get("confidence", e.get("score")) or 60
    cand = EmailCandidate(address=(e.get("value") or e.get("email") or "").lower(), confidence=min(0.85, conf / 100),
                          method="hunter", source_url=source)
    if status == "valid":  # Hunter checked the mailbox itself
        cand.deliverability, cand.verified_by = "deliverable", "hunter"
        cand.confidence = max(cand.confidence, 0.8)
    elif status in ("invalid", "disposable"):
        cand.deliverability, cand.verified_by = "undeliverable", "hunter"
    c.add_email(cand)
    return c


def hunter_person(fetcher, org: Contact, roles: List[str], cap: Optional[int] = None) -> Optional[Contact]:
    """The best personal address Hunter knows at the organisation's domain, as a contact. A person in the wanted role
    comes first, then a senior person (flagged `role_approx`); a junior employee is not who a cold ask should reach."""
    key = os.getenv("HUNTER_API_KEY", "")
    cap = cap if cap is not None else _hunter_cap("domain", 15)
    if not key or not org.org_domain or not _count(fetcher, "hunter_domain", cap):
        return None
    data = fetcher.get_json("https://api.hunter.io/v2/domain-search", params={"domain": org.org_domain, "limit": 10},
                            headers={"Authorization": f"Bearer {key}"})
    emails = (((data or {}).get("data") or {}).get("emails") or []) if isinstance(data, dict) else []
    people = [e for e in emails if e.get("type") == "personal" and e.get("value") and e.get("first_name")
              and ((e.get("verification") or {}).get("status") or "").lower() not in ("invalid", "disposable")]
    ranked = sorted(((_role_rank(e.get("position") or "", roles), e.get("confidence") or 0, e) for e in people),
                    key=lambda t: (t[0], t[1]), reverse=True)
    if not ranked or ranked[0][0] == 0:
        return None
    rank, _, e = ranked[0]
    return _hunter_candidate(e, org, rank, f"https://hunter.io/search/{org.org_domain}")


def hunter_find(fetcher, name: str, title: Optional[str], org: Contact, cap: Optional[int] = None) -> Optional[EmailCandidate]:
    """Hunter's email-finder: the address of a NAMED person at the organisation's domain (e.g. a decision-maker the
    web search identified but whose address nothing published). Replaces guessing from a convention."""
    key = os.getenv("HUNTER_API_KEY", "")
    cap = cap if cap is not None else _hunter_cap("find", 15)
    parts = name.split()
    if not key or not org.org_domain or len(parts) < 2 or not _count(fetcher, "hunter_find", cap):
        return None
    data = fetcher.get_json("https://api.hunter.io/v2/email-finder",
                            params={"domain": org.org_domain, "first_name": parts[0], "last_name": parts[-1]},
                            headers={"Authorization": f"Bearer {key}"})
    d = (data or {}).get("data") if isinstance(data, dict) else None
    if not d or not d.get("email") or (d.get("score") or 0) < 50:
        return None
    status = ((d.get("verification") or {}).get("status") or "").lower()
    if status in ("invalid", "disposable"):
        return None
    cand = EmailCandidate(address=d["email"].lower(), confidence=min(0.85, (d.get("score") or 60) / 100), method="hunter",
                          source_url=f"https://hunter.io/search/{org.org_domain}")
    if status == "valid":
        cand.deliverability, cand.verified_by = "deliverable", "hunter"
        cand.confidence = max(cand.confidence, 0.8)
    return cand


def builtwith_tech(fetcher, domain: str, cap: int = 20) -> List[str]:
    key = os.getenv("BUILTWITH_API_KEY", "")
    if not key or not domain or not _count(fetcher, "builtwith", cap):
        return []
    data = fetcher.get_json("https://api.builtwith.com/free1/api.json", params={"KEY": key, "LOOKUP": domain})
    names: List[str] = []
    for g in (data or {}).get("groups", []) if isinstance(data, dict) else []:
        for cat in g.get("categories") or []:
            names.append(str(cat.get("name") or ""))
        names.append(str(g.get("name") or ""))
    return [n for n in dict.fromkeys(names) if n]


def tag_tech(orgs: List[Contact], prof: ProspectProfile, fetcher, limit: int = 20) -> int:
    """For campaigns whose criteria need `uses_tech`: look the top organisations up and tag the proof."""
    crits = [c for c in prof.criteria if c.signal == "uses_tech"]
    if not crits or not os.getenv("BUILTWITH_API_KEY"):
        return 0
    terms = [t.lower() for c in crits for t in c.terms]
    n = 0
    for o in [o for o in orgs if o.kind == "org" and o.org_domain][:limit]:
        techs = builtwith_tech(fetcher, o.org_domain)
        hits = [t for t in techs if any(term in t.lower() for term in terms)]
        if hits:
            o.signals["uses_tech"] = 1.0
            o.evidence.append(Evidence(kind="page", text=squash("Uses " + ", ".join(hits[:4]) + " (BuiltWith)", 200),
                                       url=f"https://builtwith.com/{o.org_domain}"))
            n += 1
    return n

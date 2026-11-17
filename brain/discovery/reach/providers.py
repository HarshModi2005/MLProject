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


def hunter_person(fetcher, org: Contact, roles: List[str], cap: int = 8) -> Optional[Contact]:
    """The best personal address Hunter knows at the organisation's domain (role match first), as a contact."""
    key = os.getenv("HUNTER_API_KEY", "")
    if not key or not org.org_domain or not _count(fetcher, "hunter_domain", cap):
        return None
    data = fetcher.get_json("https://api.hunter.io/v2/domain-search", params={"domain": org.org_domain, "limit": 10},
                            headers={"Authorization": f"Bearer {key}"})
    emails = (((data or {}).get("data") or {}).get("emails") or []) if isinstance(data, dict) else []
    people = [e for e in emails if e.get("type") == "personal" and e.get("value") and e.get("first_name")]
    if not people:
        return None
    low = [r.lower() for r in roles]
    people.sort(key=lambda e: (any(r in (e.get("position") or "").lower() for r in low), e.get("confidence") or 0), reverse=True)
    e = people[0]
    name = f"{e['first_name']} {e.get('last_name') or ''}".strip()
    c = Contact(kind="person", name=name, title=e.get("position") or None, org=org.name, org_domain=org.org_domain,
                country=org.country, location_text=org.location_text, urls=list(org.urls), evidence=list(org.evidence),
                bio=org.bio, sources=org.sources + ["hunter"], signals=dict(org.signals), flags=list(org.flags))
    c.add_email(EmailCandidate(address=e["value"].lower(), confidence=min(0.85, (e.get("confidence") or 60) / 100),
                               method="hunter", source_url=f"https://hunter.io/search/{org.org_domain}"))
    return c


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

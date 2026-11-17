"""Mark candidates the user already knows, or can reach through someone they know."""

from __future__ import annotations

import re
from typing import Dict, List

from brain.models import Contact, Evidence
from brain.textutil import registrable_domain

_LEGAL = re.compile(r"\b(private|pvt|ltd|limited|inc|llc|llp|gmbh|corp(oration)?|co|company|technologies|technology|"
                    r"solutions|labs?|the)\b\.?", re.I)


def norm_company(name: str) -> str:
    return " ".join(_LEGAL.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())).split())


class Network:
    def __init__(self, people: List[Dict]):
        self.people = people
        self.by_email = {p["email"]: p for p in people if p.get("email")}
        self.by_domain: Dict[str, List[Dict]] = {}
        self.by_company: Dict[str, List[Dict]] = {}
        for p in people:
            if p.get("domain"):
                self.by_domain.setdefault(p["domain"], []).append(p)
            if p.get("company") and norm_company(p["company"]):
                self.by_company.setdefault(norm_company(p["company"]), []).append(p)

    def __len__(self) -> int:
        return len(self.people)


def _since(p: Dict) -> str:
    src = {"linkedin": "LinkedIn", "google_contacts": "your contacts", "vcard": "your contacts"}.get(p.get("source"), "your list")
    return f" ({src}{', since ' + p['connected_on'] if p.get('connected_on') else ''})"


def annotate(contacts: List[Contact], net: Network) -> Dict[str, int]:
    """Set signals["warm_path"] (1.0 known personally, 0.6 an introduction is possible) and add the evidence."""
    from brain.resolve import names_compatible

    stats = {"direct": 0, "introduction": 0}
    if not net or not len(net):
        return stats
    for c in contacts:
        direct = None
        if c.kind == "person":
            direct = next((net.by_email[e.address] for e in c.emails if e.address in net.by_email), None)
            if direct is None:
                pool = net.by_domain.get(registrable_domain(c.org_domain), []) if c.org_domain else []
                pool = pool + net.by_company.get(norm_company(c.org or ""), [])
                direct = next((p for p in pool if p.get("name") and names_compatible(p["name"], c.name)), None)
        if direct is not None:
            c.signals["warm_path"] = 1.0
            c.evidence.insert(0, Evidence(kind="warm", text=f"You are already connected to {c.name}{_since(direct)}"))
            stats["direct"] += 1
            continue
        mates = net.by_domain.get(registrable_domain(c.org_domain), []) if c.org_domain else []
        if not mates and (c.org or (c.name if c.kind == "org" else "")):
            mates = net.by_company.get(norm_company(c.org or c.name), [])
        mates = [p for p in mates if p.get("name") and p["name"].lower() != c.name.lower()]
        if mates:
            p = mates[0]
            c.signals["warm_path"] = max(c.signals.get("warm_path", 0.0), 0.6)
            who = p["name"] + (f", {p['title']}" if p.get("title") else "")
            c.evidence.insert(0, Evidence(kind="warm", text=f"You know {who} at {c.org or c.name}{_since(p)}: "
                                                              f"they could introduce you"))
            stats["introduction"] += 1
    return stats

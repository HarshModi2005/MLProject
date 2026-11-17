"""NSF awards (api.nsf.gov): researchers funded to work on the topic, with the address the award lists for
the principal investigator."""

from __future__ import annotations

import re
from typing import Optional

from brain.discovery.normalize import add_published_email
from brain.models import Contact, Evidence
from brain.textutil import squash


def _title_case(s: str) -> str:
    return " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in (s or "").split())


def to_contact(spec, record, run) -> Optional[Contact]:
    a = record.data
    first, last = (a.get("piFirstName") or "").strip(), (a.get("piLastName") or "").strip()
    if not first or not last:
        return None
    year = int(a["startDate"][-4:]) if re.search(r"\d{4}$", a.get("startDate") or "") else None
    c = Contact(kind="person", name=_title_case(f"{first} {last}"), title="Principal Investigator",
                org=_title_case(a.get("awardeeName") or "") or None, country="US", sources=[spec.name],
                location_text=", ".join(x for x in (_title_case(a.get("awardeeCity") or ""), a.get("awardeeStateCode")) if x) or None)
    url = f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={a.get('id')}" if a.get("id") else None
    c.add_url(url)
    amount = a.get("estimatedTotalAmt") or a.get("fundsObligatedAmt")
    c.evidence.append(Evidence(kind="publication", url=url, year=year,
                               text=squash(f"NSF award: {a.get('title') or ''}", 200),
                               detail=squash((f"${int(float(amount)):,} · " if amount else "") + (a.get("abstractText") or ""), 300)))
    add_published_email(c, a.get("piEmail") or "", 0.85, url)
    c.signals["grant_awarded"] = 1.0
    if year:
        c.signals["latest_year"] = float(year)
    if a.get("id"):
        c.add_key(f"nsf:{a['id']}")
    return c


def key(c: Contact) -> str:
    return c.best_email.address if c.best_email else c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    seen = {e.text for e in existing.evidence}
    existing.evidence += [e for e in new.evidence if e.text not in seen]
    existing.signals["latest_year"] = max(existing.signals.get("latest_year", 0), new.signals.get("latest_year", 0))

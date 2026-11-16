"""Bring-your-own list: a CSV of people to enrich, rank and draft for, skipping discovery."""

from __future__ import annotations

import csv
from typing import List

from brain.emailfinder import attach_emails
from brain.models import Contact, Evidence
from brain.textutil import FREE_MAIL, registrable_domain, to_country_code

_ALIASES = {
    "name": ("name", "full name", "full_name", "contact"),
    "email": ("email", "e-mail", "email address"),
    "org": ("org", "organization", "organisation", "company", "institution", "university"),
    "title": ("title", "role", "position", "job title"),
    "url": ("url", "website", "homepage", "profile", "link"),
    "country": ("country",),
    "notes": ("notes", "about", "bio", "description"),
}


def load_csv(path: str) -> List[Contact]:
    out: List[Contact] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        col = {field: next((headers[a] for a in aliases if a in headers), None) for field, aliases in _ALIASES.items()}
        if not col["name"]:
            raise ValueError(f"{path}: needs a 'name' column (found: {list(headers)})")
        for row in reader:
            def val(field: str) -> str:
                return (row.get(col[field]) or "").strip() if col[field] else ""

            if not val("name"):
                continue
            c = Contact(name=val("name"), org=val("org") or None, title=val("title") or None,
                        bio=val("notes") or None, sources=["csv"])
            if val("notes"):
                c.evidence.append(Evidence(kind="profile", text=val("notes")[:400]))
            if val("country"):
                c.country = to_country_code(val("country"))
            if val("url"):
                c.add_url(val("url"))
                c.org_domain = registrable_domain(val("url"))
            if val("email"):
                attach_emails(c, [val("email").lower()], "provided", None)
                mail_domain = registrable_domain(val("email").split("@")[1])
                if mail_domain not in FREE_MAIL:
                    c.org_domain = c.org_domain or mail_domain
            out.append(c)
    return out

"""Readers for the user's own contact exports. Each returns [{name, email, company, domain, title, source, connected_on}]."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Optional

from brain.textutil import FREE_MAIL, registrable_domain


def _domain(email: str) -> Optional[str]:
    if "@" not in (email or ""):
        return None
    d = registrable_domain(email.split("@")[1].lower())
    return None if d in FREE_MAIL else d


def _row(name: str, email: str = "", company: str = "", title: str = "", source: str = "csv", connected_on: str = "") -> Dict:
    email = (email or "").strip().lower()
    return {"name": " ".join((name or "").split()), "email": email or None, "company": (company or "").strip() or None,
            "domain": _domain(email), "title": (title or "").strip() or None, "source": source,
            "connected_on": (connected_on or "").strip() or None}


def _csv_rows(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    # LinkedIn exports start with a few "Notes:" lines before the header.
    start = next((i for i, l in enumerate(lines) if re.match(r'﻿?"?(First Name|Name|Given Name)"?\s*,', l, re.I)), 0)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def _get(row: Dict[str, str], *names: str) -> str:
    low = {k.strip().lower().lstrip("﻿"): v for k, v in row.items() if k}
    return next((low[n.lower()] for n in names if low.get(n.lower())), "")


def read_csv(text: str) -> List[Dict]:
    rows = _csv_rows(text)
    if not rows:
        return []
    header = {k.strip().lower().lstrip("﻿") for k in rows[0] if k}
    if {"first name", "last name", "connected on"} <= header:
        source = "linkedin"
    elif "e-mail 1 - value" in header or "organization name" in header or "organization 1 - name" in header:
        source = "google_contacts"
    else:
        source = "csv"
    out = []
    for r in rows:
        name = _get(r, "name", "full name") or " ".join(x for x in (_get(r, "first name", "given name"),
                                                                   _get(r, "last name", "family name")) if x)
        out.append(_row(name, _get(r, "email address", "e-mail 1 - value", "email", "e-mail"),
                        _get(r, "company", "organization name", "organization 1 - name", "organisation"),
                        _get(r, "position", "organization title", "organization 1 - title", "title", "job title"),
                        source, _get(r, "connected on")))
    return [p for p in out if p["name"] or p["email"]]


def read_vcard(text: str) -> List[Dict]:
    out = []
    for card in re.split(r"(?im)^BEGIN:VCARD\s*$", text):
        if "END:VCARD" not in card.upper():
            continue
        def field(tag: str) -> str:
            m = re.search(rf"(?im)^{tag}(?:;[^:]*)?:(.*)$", card)
            return m.group(1).strip() if m else ""
        out.append(_row(field("FN"), field("EMAIL"), field("ORG").split(";")[0], field("TITLE"), "vcard"))
    return [p for p in out if p["name"] or p["email"]]


def read_file(path: str) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return read_vcard(text) if path.lower().endswith((".vcf", ".vcard")) or "BEGIN:VCARD" in text[:200] else read_csv(text)

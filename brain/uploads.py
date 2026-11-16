"""
Older lists the user already has: load anything, work out what it is, and put it to use.

Supported: CSV/TSV, Excel, JSON, plain text (one item per line), PDF and Word documents.

Each upload gets one role:
    targets       people or organisations to contact (enrich, rank, draft)
    exclude       already contacted / do-not-contact (never email them)
    examples      good examples, e.g. last year's sponsors or current customers (find more like them)
    style         emails the user sent before (match their voice)
    about_sender  CV, deck, brochure, press release (facts the emails may use)

The role is guessed from the file name, columns and content without an LLM; the intake shows the
guess and the user can correct it.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from brain.context import CampaignContext
from brain.emailfinder import EMAIL_RE, is_role_address
from brain.models import Contact, EmailCandidate, Evidence
from brain.textutil import FREE_MAIL, registrable_domain, squash, to_country_code

ROLES = ("targets", "exclude", "examples", "style", "about_sender")

_NAME_COLS = ("name", "full name", "full_name", "contact", "contact name", "person")
_ORG_COLS = ("company", "organization", "organisation", "org", "institution", "university", "brand", "business",
             "sponsor", "firm", "fund", "account")
_EMAIL_COLS = ("email", "e-mail", "email address", "mail")
_URL_COLS = ("website", "url", "domain", "homepage", "site", "link", "profile")
_TITLE_COLS = ("title", "role", "position", "designation", "job title")
_STATUS_COLS = ("status", "contacted", "sent", "replied", "outcome", "date contacted", "last contacted", "response")
_NOTES_COLS = ("notes", "about", "bio", "description", "comment", "comments")


@dataclass
class Upload:
    path: str
    kind: str  # table | text
    rows: List[Dict[str, str]] = field(default_factory=list)
    text: str = ""
    role: str = ""
    reason: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    def summary(self) -> str:
        what = f"{len(self.rows)} rows" if self.kind == "table" else f"{len(self.text.split())} words of text"
        cols = f" (columns: {', '.join(list(self.rows[0])[:6])})" if self.rows else ""
        return f"{self.name}: {what}{cols} → {self.role} — {self.reason}"


# ── loading ──────────────────────────────────────────────────────────────────

def load(path: str) -> Upload:
    p = Path(path.strip().strip("'\"")).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    ext = p.suffix.lower()
    if ext in (".csv", ".tsv"):
        with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except csv.Error:
                dialect = csv.excel_tab if ext == ".tsv" else csv.excel
            rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f, dialect=dialect)]
        return Upload(str(p), "table", rows=rows)
    if ext in (".xlsx", ".xlsm", ".xls"):
        import pandas as pd

        frame = pd.read_excel(p, dtype=str).fillna("")
        return Upload(str(p), "table", rows=[{str(k).strip(): str(v).strip() for k, v in r.items()}
                                             for r in frame.to_dict("records")])
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        if data and all(isinstance(x, dict) for x in data):
            return Upload(str(p), "table", rows=[{str(k): str(v) for k, v in x.items()} for x in data])
        return Upload(str(p), "text", text="\n".join(str(x) for x in data))
    if ext == ".pdf":
        import fitz

        with fitz.open(p) as doc:
            return Upload(str(p), "text", text="\n".join(page.get_text() for page in doc))
    if ext == ".docx":
        import docx

        return Upload(str(p), "text", text="\n".join(par.text for par in docx.Document(str(p)).paragraphs))
    text = p.read_text(encoding="utf-8", errors="replace")
    return Upload(str(p), "text", text=text)


# ── classifying ──────────────────────────────────────────────────────────────

def _cols(u: Upload) -> List[str]:
    return [c.lower() for c in (u.rows[0].keys() if u.rows else [])]


def classify(u: Upload) -> Upload:
    """Guess the upload's role. Cheap and explainable; the user confirms or corrects it."""
    name = u.name.lower()
    cols = _cols(u)

    def has(options) -> bool:
        return any(any(o == c or o in c for o in options) for c in cols)

    if re.search(r"do.?not.?contact|dnc|unsubscrib|blacklist|blocklist|suppress|opt.?out|bounced", name):
        u.role, u.reason = "exclude", "file name says do-not-contact"
    elif re.search(r"\b(cv|resume|résumé|deck|brochure|pitch|press.?release|about|profile|portfolio)\b", name) \
            and u.kind == "text":
        u.role, u.reason = "about_sender", "document about you or your offer"
    elif u.kind == "text" and re.search(r"^\s*(dear|hi|hello)\b.*\n", u.text, re.I | re.M) and \
            re.search(r"(regards|thanks|thank you|sincerely|cheers|best)\s*,?\s*\n", u.text, re.I):
        u.role, u.reason = "style", "looks like emails you wrote"
    elif u.kind == "table" and has(_STATUS_COLS) or re.search(r"contacted|sent|outreach.?log|emailed|history", name):
        u.role, u.reason = "exclude", "has a contacted/status column — treating as already contacted"
    elif re.search(r"past|previous|last.?year|existing|current|client|customer|sponsor|partner|portfolio|alumni", name):
        u.role, u.reason = "examples", "past or current relationships — will find similar ones"
    elif u.kind == "table" and (has(_EMAIL_COLS) or has(_NAME_COLS) or has(_ORG_COLS)):
        u.role, u.reason = "targets", "a list of people/organisations to contact"
    elif u.kind == "text" and len(EMAIL_RE.findall(u.text)) >= 3:
        u.role, u.reason = "targets", "contains email addresses"
    elif u.kind == "text" and len(u.text.split()) > 150:
        u.role, u.reason = "about_sender", "longer document — using it as background about you"
    else:
        u.role, u.reason = "examples", "a list of names — will find similar ones"
    return u


# ── applying ─────────────────────────────────────────────────────────────────

def _pick(row: Dict[str, str], options) -> str:
    for k, v in row.items():
        kl = k.lower().strip()
        if v and any(o == kl or o in kl for o in options):
            return v.strip()
    return ""


def rows_to_contacts(rows: List[Dict[str, str]], source: str) -> List[Contact]:
    out: List[Contact] = []
    for row in rows:
        name, org = _pick(row, _NAME_COLS), _pick(row, _ORG_COLS)
        email = (_pick(row, _EMAIL_COLS) or "").lower()
        if not email:
            found = EMAIL_RE.findall(" ".join(row.values()))
            email = found[0].lower() if found else ""
        url = _pick(row, _URL_COLS)
        if not (name or org or email):
            continue
        kind = "person" if name else "org"
        c = Contact(kind=kind, name=name or org or email.split("@")[0], org=org or None,
                    title=_pick(row, _TITLE_COLS) or None, sources=[source])
        if url:
            url = url if url.startswith("http") else "https://" + url
            c.add_url(url)
            c.org_domain = registrable_domain(url)
        if email and EMAIL_RE.fullmatch(email):
            c.add_email(EmailCandidate(address=email, confidence=0.95, method="provided"))
            dom = registrable_domain(email.split("@")[1])
            if dom not in FREE_MAIL:
                c.org_domain = c.org_domain or dom
            if is_role_address(email) and kind == "org":
                c.emails[-1].method = "inbox"
        country = _pick(row, ("country",))
        if country:
            c.country = to_country_code(country)
        notes = _pick(row, _NOTES_COLS)
        if notes:
            c.bio = squash(notes, 400)
            c.evidence.append(Evidence(kind="profile", text=squash(notes, 300)))
        out.append(c)
    return out


def apply(u: Upload, ctx: CampaignContext) -> List[Contact]:
    """Fold an upload into the campaign context. Returns contacts to add to discovery (targets only)."""
    ctx.uploads.append({"file": u.name, "role": u.role, "summary": u.summary()})
    source = f"upload:{u.name}"
    if u.role == "targets":
        if u.kind == "table":
            return rows_to_contacts(u.rows, source)
        return rows_to_contacts([{"email": e} for e in dict.fromkeys(EMAIL_RE.findall(u.text))], source)
    if u.role == "exclude":
        items = []
        for row in u.rows:
            items += [_pick(row, _EMAIL_COLS), _pick(row, _URL_COLS), _pick(row, _NAME_COLS) or _pick(row, _ORG_COLS)]
        if u.kind == "text":
            items += [line.strip() for line in u.text.splitlines()]
        for v in items:
            ctx.exclude(v)
        return []
    if u.role == "examples":
        if u.kind == "table":
            for row in u.rows:
                ctx.add_seed(_pick(row, _ORG_COLS) or _pick(row, _NAME_COLS), _pick(row, _URL_COLS))
        else:
            for line in u.text.splitlines():
                line = line.strip(" -•\t")
                if 1 < len(line) < 80:
                    ctx.add_seed(line, line if re.fullmatch(r"[\w.-]+\.[a-z]{2,}", line) else "")
        return []
    if u.role == "style":
        for block in re.split(r"\n\s*-{3,}\s*\n|\n{3,}", u.text):
            if len(block.split()) > 30:
                ctx.style_examples.append(squash(block, 1200))
        ctx.style_examples = ctx.style_examples[:3]
        return []
    return []  # about_sender is turned into facts by the intake (it needs the LLM)


def sender_text(uploads: List[Upload]) -> str:
    return "\n\n".join(u.text for u in uploads if u.role == "about_sender" and u.text)[:12000]

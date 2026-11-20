"""arXiv: recent papers on the topic (export.arxiv.org API, Atom) -> their authors, kept only when the paper's
PDF prints an address that clearly belongs to them. A small but precise source of reachable researchers."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from brain.models import Contact, EmailCandidate, Evidence
from brain.textutil import clean_name, country_from_domain, looks_like_person_name, squash


def parse(spec, text: str) -> List[Dict]:
    """One item per (paper, author)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "xml")
    out = []
    for e in soup.find_all("entry"):
        abs_url = squash(e.find("id").get_text()) if e.find("id") else ""
        pdf = next((l.get("href") for l in e.find_all("link") if l.get("title") == "pdf"), None)
        if not pdf and "/abs/" in abs_url:
            pdf = abs_url.replace("/abs/", "/pdf/")
        title = squash(e.find("title").get_text(" ") if e.find("title") else "", 200)
        published = squash(e.find("published").get_text() if e.find("published") else "")
        summary = squash(e.find("summary").get_text(" ") if e.find("summary") else "", 300)
        authors = []
        for a in e.find_all("author"):
            n = a.find("name")
            aff = a.find("arxiv:affiliation") or a.find("affiliation")
            if n:
                authors.append((clean_name(n.get_text()), squash(aff.get_text()) if aff else ""))
        names = [n for n, _ in authors]
        for name, aff in authors:
            out.append({"author": name, "affiliation": aff, "title": title, "summary": summary,
                        "published": published, "abs": abs_url, "pdf": pdf, "authors": names})
    return out


def to_contact(spec, record, run) -> Optional[Contact]:
    d = record.data
    name = d.get("author") or ""
    if not looks_like_person_name(name):
        return None
    year = int(d["published"][:4]) if re.match(r"\d{4}", d.get("published") or "") else None
    c = Contact(kind="person", name=name, title="Researcher", org=d.get("affiliation") or None, sources=[spec.name])
    c.add_url(d.get("abs"))
    c.evidence.append(Evidence(kind="publication", text=d.get("title") or "", url=d.get("abs"), year=year,
                               detail=d.get("summary") or None))
    c.signals["publications"] = 0.8
    c.signals["matched_works"] = 1.0
    if year:
        c.signals["latest_year"] = float(year)
    c.flags.append("pdf:" + (d.get("pdf") or ""))  # read in `post`, removed there
    return c


def key(c: Contact) -> str:
    return c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    if new.evidence and new.evidence[0].text not in {e.text for e in existing.evidence}:
        existing.evidence += new.evidence
        existing.signals["matched_works"] = existing.signals.get("matched_works", 1.0) + 1
        existing.signals["latest_year"] = max(existing.signals.get("latest_year", 0), new.signals.get("latest_year", 0))
    existing.flags += [f for f in new.flags if f.startswith("pdf:") and f not in existing.flags]


def post(spec, run, contacts: List[Contact], stats: Dict) -> List[Contact]:
    """Read up to request.max_pdfs PDFs (most authors among our candidates first); keep authors whose own
    address the paper prints."""
    from brain.discovery.reach.pdfemails import author_emails

    by_pdf: Dict[str, List[Contact]] = {}
    for c in contacts:
        for f in c.flags:
            if f.startswith("pdf:") and len(f) > 4:
                by_pdf.setdefault(f[4:], []).append(c)
    limit = int(spec.request.get("max_pdfs") or 4)
    chosen = sorted(by_pdf.items(), key=lambda kv: len(kv[1]), reverse=True)[:limit]
    for pdf, authors in chosen:
        found = author_emails(run.deps.fetcher, pdf, [a.name for a in authors])
        stats["pdfs_read"] = stats.get("pdfs_read", 0) + 1
        for a in authors:
            addr = found.get(a.name)
            if addr and not a.emails:
                a.add_email(EmailCandidate(address=addr, confidence=0.8, method="pdf", source_url=pdf))
                dom = addr.split("@")[1]
                a.org_domain = a.org_domain or dom
                a.country = a.country or country_from_domain(dom)
    out = []
    for c in contacts:
        c.flags = [f for f in c.flags if not f.startswith("pdf:")]
        if c.emails:
            out.append(c)
    stats["with_email"] = len(out)
    return out

"""
Author emails from papers. Most papers print their authors' addresses on the first page, often in a
compressed form ("{alice, bob}@uni.edu"). Each address is matched to an author by name; an address that
matches nobody, or more than one author, is left alone.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from brain.emailfinder import extract_emails, is_role_address, name_match

# "{alice, bob.smith}@uni.edu" / "(alice|bob)@uni.edu" -> alice@uni.edu, bob.smith@uni.edu
_GROUPED = re.compile(r"[{(\[]\s*([A-Za-z0-9._\-]+(?:\s*[,|;]\s*[A-Za-z0-9._\-]+)+)\s*[})\]]\s*@\s*([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


def expand_grouped(text: str) -> List[str]:
    out = []
    for locals_, domain in _GROUPED.findall(text or ""):
        for loc in re.split(r"\s*[,|;]\s*", locals_):
            if loc:
                out.append(f"{loc}@{domain}".lower())
    return out


def emails_in(text: str) -> List[str]:
    found = expand_grouped(text) + extract_emails(text or "")
    return [e for e in dict.fromkeys(found) if not is_role_address(e)]


def match_authors(text: str, authors: List[str]) -> Dict[str, str]:
    """{author name: address} for addresses that clearly carry exactly one author's name."""
    out: Dict[str, str] = {}
    for addr in emails_in(text):
        full = [a for a in authors if name_match(a, addr) == 1.0]
        if len(full) == 1 and full[0] not in out:
            out[full[0]] = addr
    return out


def author_emails(fetcher, pdf_url: str, authors: List[str], max_pages: int = 2) -> Dict[str, str]:
    text: Optional[str] = fetcher.get_pdf_text(pdf_url, max_pages=max_pages) if pdf_url else None
    return match_authors(text or "", authors)

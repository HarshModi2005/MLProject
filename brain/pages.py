"""
Page understanding without an LLM.

A fetched page is one of:
  profile    one person's page             -> one Contact
  directory  a list of people              -> many Contacts (+ links to their profiles)
  other      anything else                 -> nothing (optionally handed to the LLM extractor)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup, Tag

from brain.emailfinder import attach_emails, extract_emails, name_match
from brain.models import Contact, Evidence
from brain.textutil import (
    clean_name, country_from_domain, host_of, looks_like_person_name, registrable_domain, squash,
)

DIRECTORY_PATH_HINTS = ("people", "faculty", "staff", "team", "directory", "members", "researchers",
                        "our-team", "leadership", "about-us", "employees")
TITLE_WORDS = re.compile(
    r"\b((?:assistant|associate|adjunct|visiting|distinguished|full|emeritus)?\s*professor|lecturer|"
    r"reader|research (?:scientist|fellow|engineer)|scientist|principal investigator|postdoc\w*|"
    r"(?:co-?)?founder|ceo|cto|coo|cfo|chief [a-z]+ officer|vp of [a-z ]+|vice president|"
    r"head of [a-z ]+|director(?: of [a-z ]+)?|(?:engineering|product|hiring|talent) manager|"
    r"recruiter|partner|principal|editor|journalist|reporter)\b",
    re.I,
)
INTEREST_RE = re.compile(
    r"(?:research\s+(?:interests?|areas?|topics?|focus)|areas?\s+of\s+(?:interest|research|expertise)|"
    r"interests?|expertise|speciali[sz]ation)\s*[:\-–]?\s*([^\n]{15,300})",
    re.I,
)


@dataclass
class ParsedPage:
    url: str
    title: str = ""
    h1: str = ""
    text: str = ""  # main text, navigation and footer removed
    links: List[Tuple[str, str]] = field(default_factory=list)  # (absolute url, anchor text)
    nav_links: List[Tuple[str, str]] = field(default_factory=list)  # every link incl. menus/footers
    mailtos: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)  # all addresses in main text + mailtos
    persons_ld: List[dict] = field(default_factory=list)  # schema.org Person objects
    soup: Optional[BeautifulSoup] = None


def parse(html: str, url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    page = ParsedPage(url=url, soup=soup)

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                types = item.get("@type")
                types = types if isinstance(types, list) else [types]
                if "Person" in types:
                    page.persons_ld.append(item)
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
            elif isinstance(item, list):
                stack.extend(item)

    # Site-wide links (menus, footers) matter for crawling but not for reading the page, so they are
    # recorded before navigation is stripped out of the content.
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        absolute = urldefrag(urljoin(url, href))[0]
        if absolute.startswith("http"):
            page.nav_links.append((absolute, squash(a.get_text(" "), 120)))
    refresh = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
    if refresh and "url=" in (refresh.get("content") or "").lower():
        target = re.split("url=", refresh["content"], flags=re.I)[1].strip(" '\"")
        page.nav_links.append((urljoin(url, target), "redirect"))

    for el in soup.find_all(["script", "style", "noscript", "nav", "footer", "form", "svg", "iframe"]):
        el.decompose()

    page.title = squash(soup.title.get_text(" ") if soup.title else "", 200)
    h1 = soup.find("h1")
    page.h1 = squash(h1.get_text(" ") if h1 else "", 120)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith("mailto:"):
            for addr in extract_emails(href[7:].split("?")[0]):
                if addr not in page.mailtos:
                    page.mailtos.append(addr)
            continue
        if href.startswith(("javascript:", "tel:", "#")):
            continue
        absolute = urldefrag(urljoin(url, href))[0]
        if absolute.startswith("http"):
            page.links.append((absolute, squash(a.get_text(" "), 120)))

    page.text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n")).strip()
    page.emails = list(dict.fromkeys(page.mailtos + extract_emails(page.text)))
    return page


_DOC_EXT = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".jpg", ".png", ".zip")


def _person_links(page: ParsedPage) -> List[Tuple[str, str]]:
    seen: Dict[str, str] = {}
    for href, text in page.links:
        if href.lower().split("?")[0].endswith(_DOC_EXT):
            continue  # a link to a document is never a person's profile
        if looks_like_person_name(text) and clean_name(text).lower() not in seen:
            seen[clean_name(text).lower()] = href
    return [(name, href) for name, href in seen.items()]


def page_person_name(page: ParsedPage) -> Optional[str]:
    for person in page.persons_ld:
        name = person.get("name")
        if isinstance(name, str) and looks_like_person_name(name):
            return clean_name(name)
    if looks_like_person_name(page.h1):
        return clean_name(page.h1)
    # "Sanjay Kumar | Department of CSE | IIT Ropar"
    for part in re.split(r"\s*[|·–—:]\s*|\s+-\s+", page.title):
        if looks_like_person_name(part):
            return clean_name(part)
    return None


def classify(page: ParsedPage) -> str:
    people = len(_person_links(page))
    emails = len(page.emails)
    path = page.url.lower()
    hinted = any(h in path for h in DIRECTORY_PATH_HINTS)
    name = page_person_name(page)
    if name and emails <= 4 and people < 12:
        return "profile"
    if people >= 8 or emails >= 6 or (hinted and (people >= 4 or emails >= 3)):
        return "directory"
    if name:
        return "profile"
    return "other"


def _org_from_page(page: ParsedPage, person: Optional[str] = None) -> Optional[str]:
    if page.soup:
        meta = page.soup.find("meta", attrs={"property": "og:site_name"})
        if meta and meta.get("content"):
            return squash(meta["content"], 120)
    parts = [p for p in re.split(r"\s*[|·–—]\s*|\s+-\s+", page.title) if p]
    for part in reversed(parts):
        if len(part) > 3 and clean_name(part).lower() != (person or "").lower():
            return squash(part, 120)
    return None


def _first_title(text: str) -> Optional[str]:
    m = TITLE_WORDS.search(text[:1500])
    return squash(m.group(0), 80).title() if m else None


def profile_contact(page: ParsedPage) -> Optional[Contact]:
    """Build a Contact from a single-person page. Returns None if no person name is found."""
    name = page_person_name(page)
    if not name:
        return None
    domain = registrable_domain(page.url)
    contact = Contact(name=name, org=_org_from_page(page, name), org_domain=domain,
                      country=country_from_domain(domain), sources=["web"])
    contact.add_url(page.url)
    for person in page.persons_ld:
        if isinstance(person.get("jobTitle"), str):
            contact.title = squash(person["jobTitle"], 80)
        works_for = person.get("worksFor") or person.get("affiliation")
        if isinstance(works_for, dict) and isinstance(works_for.get("name"), str):
            contact.org = squash(works_for["name"], 120)
        if isinstance(person.get("email"), str):
            attach_emails(contact, extract_emails(person["email"].replace("mailto:", "")), "mailto", page.url, True)
    contact.title = contact.title or _first_title(page.text)
    attach_emails(contact, page.mailtos, "mailto", page.url, on_own_page=True)
    attach_emails(contact, [e for e in page.emails if e not in page.mailtos], "text", page.url, on_own_page=True)

    m = INTEREST_RE.search(page.text)
    if m:
        topics = [squash(t, 60) for t in re.split(r"[,;•·|]|\band\b", m.group(1)) if 2 < len(t.strip()) < 60]
        contact.topics = topics[:10]
    paragraphs = [squash(p) for p in page.text.split("\n") if len(p.strip()) > 120]
    if paragraphs:
        contact.bio = paragraphs[0][:600]
        contact.evidence.append(Evidence(kind="page", text=paragraphs[0][:400], url=page.url))
    return contact


def _nearest_name(tag: Tag) -> Optional[str]:
    """Walk up from an email link to the enclosing card/row and find the person it belongs to."""
    node: Optional[Tag] = tag
    for _ in range(5):
        node = node.parent if node is not None else None
        if node is None or node.name in ("body", "html"):
            return None
        candidates = []
        for el in node.find_all(["a", "h2", "h3", "h4", "h5", "strong", "b", "td", "span", "p", "div"], limit=60):
            text = squash(el.get_text(" "), 80)
            if looks_like_person_name(text):
                candidates.append(clean_name(text))
        distinct = list(dict.fromkeys(candidates))
        if len(distinct) == 1:
            return distinct[0]
        if len(distinct) > 1:
            return None  # block already spans several people
    return None


def directory_contacts(page: ParsedPage) -> List[Contact]:
    """Extract people from a listing page: names, profile links and any emails shown beside them."""
    domain = registrable_domain(page.url)
    org = _org_from_page(page)
    country = country_from_domain(domain)
    out: Dict[str, Contact] = {}

    def get(name: str) -> Contact:
        key = name.lower()
        if key not in out:
            out[key] = Contact(name=name, org=org, org_domain=domain, country=country, sources=["web"],
                               flags=["from_directory"])
        return out[key]

    for name_key, href in _person_links(page):
        # Recover original casing from the link text
        original = next((clean_name(t) for h, t in page.links if clean_name(t).lower() == name_key), name_key.title())
        c = get(original)
        if registrable_domain(href) == domain:
            c.add_url(href)

    if page.soup:
        for a in page.soup.find_all("a", href=True):
            if not a["href"].lower().startswith("mailto:"):
                continue
            addrs = extract_emails(a["href"][7:].split("?")[0])
            name = _nearest_name(a)
            if name and addrs:
                attach_emails(get(name), addrs, "directory", page.url)

    # Addresses printed as text: assign to the unique person whose name they match.
    people = list(out.values())
    for addr in page.emails:
        if any(addr == e.address for c in people for e in c.emails):
            continue
        matches = [c for c in people if name_match(c.name, addr) == 1.0]
        if len(matches) == 1:
            attach_emails(matches[0], [addr], "directory", page.url)

    for c in people:
        if not c.urls:
            c.add_url(page.url)
    return people


def contacts_from_page(html: str, url: str) -> Tuple[str, List[Contact]]:
    page = parse(html, url)
    kind = classify(page)
    if kind == "profile":
        c = profile_contact(page)
        return kind, [c] if c else []
    if kind == "directory":
        return kind, directory_contacts(page)
    return kind, []


def page_excerpt(html: str, url: str, limit: int = 3500) -> str:
    """Compact text for the (rare) LLM extraction fallback."""
    return parse(html, url).text[:limit]

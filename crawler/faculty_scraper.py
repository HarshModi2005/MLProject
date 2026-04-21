"""
crawler/faculty_scraper.py

Web scraper for university faculty pages and lab websites.

Strategies:
  1. Fetch page HTML with requests + BeautifulSoup
  2. Extract name, email, research areas, bio from common page patterns
  3. Handle .edu/faculty pages, personal lab sites, and Google Scholar profiles
  4. Email extraction with regex (university patterns as fallback)
"""

from __future__ import annotations

import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

# ── Request Headers & Session ─────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[403, 429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# ── Email Patterns ────────────────────────────────────────────────────────────
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[a-zA-Z]{2,7}\b"
)

# ── Common faculty page CSS selectors ────────────────────────────────────────
NAME_SELECTORS = [
    "h1.faculty-name", "h1.person-name", "h1.professor-name",
    "div.faculty-name", "span.faculty-name",
    "h1", "h2.bio-title",
]
BIO_SELECTORS = [
    "div.faculty-bio", "div.bio-text", "div.profile-bio",
    "section.about", "div.research-interests", "p.bio",
    "div.field-items", "div.views-field-body",
]
EMAIL_SELECTORS = [
    "a[href^='mailto:']", "span.email", "div.email",
    "td.views-field-field-email", ".contact-email",
]
RESEARCH_SELECTORS = [
    "div.research-interests", "div.field-name-field-research-interests",
    "ul.interests", "section.research", "div.research-areas",
]


# ── Core Scraper ──────────────────────────────────────────────────────────────

def fetch_page(url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a parsed BeautifulSoup object using a robust session."""
    if "scholar.google.com" in url:
        console.print(f"  [dim]Skipping direct scrape of Google Scholar (avoids 403, no emails anyway).[/dim]")
        return None
        
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout, verify=False) # verify=False ignores some broken uni certs
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.exceptions.RequestException as e:
        console.print(f"  [red]Fetch error ({url[:60]}...): {e}[/red]")
        return None


def extract_email_from_page(soup: BeautifulSoup, page_text: str, domain: str = "") -> Optional[str]:
    """
    Extract an email address from a page.
    Priority: mailto links > text regex > university domain construction
    """
    # 1. Check mailto links
    for tag in soup.select("a[href^='mailto:']"):
        href = tag.get("href", "")
        match = EMAIL_REGEX.search(href)
        if match:
            return match.group()

    # 2. Check dedicated email elements
    for sel in EMAIL_SELECTORS[1:]:
        el = soup.select_one(sel)
        if el:
            match = EMAIL_REGEX.search(el.get_text())
            if match:
                return match.group()

    # 3. Scan full page text (academic emails only — filter .edu, .ac.*, .edu.*)
    emails = EMAIL_REGEX.findall(page_text)
    academic_emails = [
        e for e in emails
        if any(e.endswith(d) for d in [".edu", ".ac.uk", ".ac.in", ".edu.au", ".ac.de"])
        or (domain and domain in e)
    ]
    if academic_emails:
        return academic_emails[0]

    return None


def extract_name_from_page(soup: BeautifulSoup) -> Optional[str]:
    """Extract professor name using common selectors and heuristics."""
    for sel in NAME_SELECTORS:
        el = soup.select_one(sel)
        if el:
            name = el.get_text(strip=True)
            if 3 < len(name) < 60:  # Sanity check
                # Remove common prefixes
                name = re.sub(r"^(Prof\.?|Professor|Dr\.?|Associate Professor)\s+", "", name, flags=re.IGNORECASE)
                return name.strip()

    # Fallback: check <title> tag
    title = soup.title
    if title:
        text = title.get_text(strip=True)
        # Often "Name | Department | University"
        parts = re.split(r"[|·\-–]", text)
        if parts and 3 < len(parts[0].strip()) < 50:
            return parts[0].strip()

    return None


def extract_research_keywords(soup: BeautifulSoup, text: str) -> list[str]:
    """Extract research areas/keywords from the page."""
    # Try dedicated research sections
    for sel in RESEARCH_SELECTORS:
        el = soup.select_one(sel)
        if el:
            raw = el.get_text(separator=", ", strip=True)
            # Split on commas, newlines, semicolons
            keywords = [
                k.strip()
                for k in re.split(r"[,;\n·•]", raw)
                if 2 < len(k.strip()) < 80
            ]
            if keywords:
                return keywords[:15]

    # Fallback: look for common academic keyword patterns in text
    patterns = [
        r"Research (?:interests|areas?|topics?)[\s:]+([^\n.]{20,200})",
        r"(?:interests?|focuses?)[\s:]+([^\n.]{20,200})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            keywords = [k.strip() for k in re.split(r"[,;]", raw) if k.strip()]
            if keywords:
                return keywords[:10]

    return []


def extract_bio_snippet(soup: BeautifulSoup) -> Optional[str]:
    """Extract a short bio/research description."""
    for sel in BIO_SELECTORS:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 50:
                return text[:500]  # First 500 chars

    # Fallback: first substantial paragraph
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 100:
            return text[:500]

    return None


def extract_accepting_students(text: str) -> Optional[bool]:
    """Heuristically detect if the professor is accepting students."""
    text_lower = text.lower()

    accepting_signals = [
        "looking for students", "recruiting students", "accepting applications",
        "open positions", "phd openings", "taking students", "seeking graduate",
        "join my lab", "join our lab", "apply to", "we are hiring",
    ]
    rejecting_signals = [
        "not accepting", "no openings", "not looking for", "currently full",
        "not taking on new students",
    ]

    if any(sig in text_lower for sig in accepting_signals):
        return True
    if any(sig in text_lower for sig in rejecting_signals):
        return False
    return None


def scrape_faculty_page(url: str) -> dict:
    """
    Main scraping function for a faculty profile URL.
    Returns a dict ready to be converted into a ProfessorProfile.
    """
    result = {
        "profile_url": url,
        "name": None,
        "email": None,
        "institution": None,
        "department": None,
        "research_keywords": [],
        "bio_snippet": None,
        "accepting_students": None,
        "raw_page_text": None,
    }

    soup = fetch_page(url)
    if not soup:
        return result

    # Try to remove noisy elements
    for el in soup.find_all(["script", "style", "nav", "footer"]):
        el.decompose()

    page_text = soup.get_text(separator=" ", strip=True)
    result["raw_page_text"] = page_text[:5000]  # Cap for storage

    # Extract domain for email guessing
    domain = urlparse(url).netloc.replace("www.", "")

    result["name"] = extract_name_from_page(soup)
    result["email"] = extract_email_from_page(soup, page_text, domain)
    result["research_keywords"] = extract_research_keywords(soup, page_text)
    result["bio_snippet"] = extract_bio_snippet(soup)
    result["accepting_students"] = extract_accepting_students(page_text)

    # Try to extract institution from domain or page meta
    meta_org = soup.find("meta", attrs={"name": "organization"})
    if meta_org:
        result["institution"] = meta_org.get("content", "")
    else:
        # Guess from domain: "cs.mit.edu" → "MIT"
        result["institution"] = domain.upper().split(".")[-2] if "." in domain else domain

    return result


def scrape_batch(urls: list[str], delay: float = 1.0) -> list[dict]:
    """
    Scrape a list of faculty page URLs with polite rate limiting.
    Returns raw dicts for each scraped page.
    """
    results = []
    for i, url in enumerate(urls, 1):
        console.print(f"  [dim]({i}/{len(urls)})[/dim] Scraping: {url[:70]}...")
        result = scrape_faculty_page(url)
        name = result.get("name") or "(name not found)"
        email = result.get("email") or "(no email)"
        console.print(f"    → {name} | {email}")
        results.append(result)
        time.sleep(delay)  # Polite delay

    return results

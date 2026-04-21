"""
crawler/scholar_scraper.py

Google Scholar and Semantic Scholar scraping utilities.
Provides author/paper discovery beyond the REST API.
"""

from __future__ import annotations

import re
import time
from typing import Optional
from urllib.parse import urlencode, quote_plus

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
GOOGLE_SCHOLAR_BASE = "https://scholar.google.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def search_authors_semantic_scholar(
    query: str,
    limit: int = 10,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """
    Search for authors on Semantic Scholar by keyword query.
    Returns a list of raw author dicts.
    """
    if fields is None:
        fields = [
            "authorId", "name", "affiliations",
            "homepage", "paperCount", "hIndex",
            "papers.title", "papers.year", "papers.externalIds",
        ]

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": ",".join(fields),
    }
    url = f"{SEMANTIC_SCHOLAR_API}/author/search"

    try:
        resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as exc:
        console.print(f"[red]Semantic Scholar author search failed: {exc}[/red]")
        return []


def get_author_papers(author_id: str, limit: int = 5) -> list[dict]:
    """
    Fetch recent papers for a Semantic Scholar author ID.
    """
    params = {
        "fields": "title,year,venue,externalIds,abstract",
        "limit": min(limit, 100),
        "sort": "year:desc",
    }
    url = f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers"
    try:
        resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as exc:
        console.print(f"[red]Error fetching papers for {author_id}: {exc}[/red]")
        return []


def enrich_with_semantic_scholar(name: str, institution: str = "") -> dict:
    """
    Given a professor name (and optionally institution), search Semantic Scholar
    and return a dict of enrichment data (publications, hIndex, homepage).
    """
    query = f"{name} {institution}".strip()
    authors = search_authors_semantic_scholar(query, limit=3)
    if not authors:
        return {}

    # Pick the best match (first result)
    author = authors[0]
    author_id = author.get("authorId", "")

    papers = []
    if author_id:
        raw_papers = get_author_papers(author_id, limit=5)
        for p in raw_papers:
            papers.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "url": _paper_url(p),
            })

    return {
        "semantic_scholar_id": author_id,
        "h_index": author.get("hIndex"),
        "paper_count": author.get("paperCount"),
        "homepage": author.get("homepage"),
        "papers": papers,
    }


# ── Google Scholar ────────────────────────────────────────────────────────────

def search_google_scholar(
    query: str,
    num_results: int = 10,
    delay: float = 2.5,
) -> list[dict]:
    """
    Scrape Google Scholar search results for a given query.
    Returns list of dicts with keys: title, url, snippet, authors.

    NOTE: Google Scholar blocks automated scrapers aggressively.
    This is a best-effort implementation — use sparingly and with delays.
    Use Serper/Tavily APIs for production-grade search instead.
    """
    params = {"q": query, "hl": "en", "num": num_results}
    url = f"{GOOGLE_SCHOLAR_BASE}/scholar?{urlencode(params)}"

    results = []
    try:
        time.sleep(delay)  # Politely delay before each Scholar request
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)

        if resp.status_code == 429:
            console.print("[yellow]⚠ Google Scholar rate limited. Skipping.[/yellow]")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select(".gs_ri"):
            title_tag = item.select_one(".gs_rt a")
            snippet_tag = item.select_one(".gs_rs")
            author_tag = item.select_one(".gs_a")

            results.append({
                "title": title_tag.get_text(strip=True) if title_tag else "",
                "url": title_tag["href"] if title_tag and title_tag.has_attr("href") else "",
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                "authors": author_tag.get_text(strip=True) if author_tag else "",
            })

    except Exception as exc:
        console.print(f"[red]Google Scholar scrape failed: {exc}[/red]")

    return results


def get_scholar_profile_url(name: str) -> Optional[str]:
    """
    Attempt to find a Google Scholar profile URL for a given professor name.
    Returns the profile URL if found, else None.
    """
    query = f"author:{quote_plus(name)} site:scholar.google.com"
    results = search_google_scholar(query, num_results=3)
    for r in results:
        url = r.get("url", "")
        if "scholar.google.com/citations" in url:
            return url
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _paper_url(paper_dict: dict) -> Optional[str]:
    """Extract the best available URL from a Semantic Scholar paper dict."""
    ext_ids = paper_dict.get("externalIds", {})
    if ext_ids.get("DOI"):
        return f"https://doi.org/{ext_ids['DOI']}"
    if ext_ids.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext_ids['ArXiv']}"
    if ext_ids.get("CorpusId"):
        return f"https://www.semanticscholar.org/paper/{ext_ids['CorpusId']}"
    return None

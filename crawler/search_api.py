"""
crawler/search_api.py

Search API integrations for discovering professor profile URLs.

Supports:
  - Tavily (research-focused, recommended)
  - Serper (Google search wrapper, fallback)
  - Semantic Scholar public API (free, for paper/author search)
  - Direct university domain search construction

Returns lists of URLs and metadata for the faculty scraper to process.
"""

from __future__ import annotations

import time
from typing import Optional
import requests

import config
from rich.console import Console

console = Console()


# ── Tavily Search ─────────────────────────────────────────────────────────────

def search_tavily(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[dict]:
    """
    Search using Tavily API (best for academic content).
    Returns list of {url, title, content, score} dicts.
    """
    if not config.TAVILY_API_KEY:
        console.print("[yellow]⚠ Tavily API key not set. Skipping Tavily search.[/yellow]")
        return []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": [],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        console.print(f"[red]Tavily search error: {e}[/red]")
        return []


# ── Serper Search ─────────────────────────────────────────────────────────────

def search_serper(
    query: str,
    num_results: int = 10,
    site_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search using Serper API (Google Search wrapper).
    Returns list of {link, title, snippet} dicts.
    """
    if not config.SERPER_API_KEY:
        console.print("[yellow]⚠ Serper API key not set. Skipping Serper search.[/yellow]")
        return []

    full_query = f"site:{site_filter} {query}" if site_filter else query

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": config.SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": full_query, "num": num_results},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("organic", [])
    except Exception as e:
        console.print(f"[red]Serper search error: {e}[/red]")
        return []


# ── Semantic Scholar Author Search ────────────────────────────────────────────

def search_semantic_scholar_authors(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Search Semantic Scholar for researchers matching a query.
    Returns list of structured author dicts with affiliation, h-index, etc.
    Free API, no key required.
    """
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "name,affiliations,paperCount,citationCount,hIndex,papers.title,papers.year,papers.venue",
    }
    try:
        time.sleep(config.SEMANTIC_SCHOLAR_SLEEP_SEC)
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        console.print(f"[red]Semantic Scholar search error: {e}[/red]")
        return []


def get_semantic_scholar_papers(author_id: str, limit: int = 5) -> list[dict]:
    """
    Fetch the most recent papers for a Semantic Scholar author ID.
    """
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
    params = {
        "limit": limit,
        "fields": "title,year,venue,abstract,externalIds,authors",
        "sort": "year:desc",
    }
    try:
        time.sleep(config.SEMANTIC_SCHOLAR_SLEEP_SEC)
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        console.print(f"[red]Error fetching papers: {e}[/red]")
        return []

def get_semantic_scholar_author(author_id: str) -> Optional[dict]:
    """Fetch a full author profile by ID."""
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    params = {
        "fields": "name,affiliations,paperCount,citationCount,hIndex,papers.title,papers.year,papers.venue",
    }
    try:
        time.sleep(config.SEMANTIC_SCHOLAR_SLEEP_SEC)
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None

# ── Query Construction Helpers ────────────────────────────────────────────────

# ── Country-specific domain and institution mappings ─────────────────────────
COUNTRY_DOMAINS: dict[str, list[str]] = {
    "India": [".ac.in", "iitb", "iitd", "iitm", "iisc", "iitk", "iiit", "bits", "iit"],
    "USA": [".edu", "mit.edu", "stanford.edu", "cmu.edu"],
    "United Kingdom": [".ac.uk", "oxford.ac.uk", "cam.ac.uk"],
    "Germany": [".de", "tum.de", "kit.edu"],
    "Canada": [".ca", "utoronto.ca", "ubc.ca"],
    "Australia": [".edu.au", "anu.edu.au"],
}

COUNTRY_INSTITUTIONS: dict[str, list[str]] = {
    "India": [
        "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
        "IIT Hyderabad", "IIT Roorkee", "IIT Gandhinagar", "IISc Bangalore",
        "IIIT Hyderabad", "IIIT Delhi", "TIFR", "CMI Chennai", "ISI Kolkata",
    ],
    "USA": ["MIT", "Stanford", "CMU", "Berkeley", "Harvard"],
}


def get_top_institutions_for_region(region: str) -> list[str]:
    """Use LLM to dynamically discover top computer science universities in a region."""
    try:
        llm = config.get_groq_llm()
        prompt = (
            f"List the 5 best universities or engineering/research institutions in {region}. "
            "Return ONLY a comma-separated list of their short names or acronyms (e.g., 'IIT Ropar, Thapar University, PEC Chandigarh'). "
            "Do not include any other text, reasoning, or bullet points."
        )
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Clean up any potential markdown or prefixes
        if ":" in text and len(text.split(":")[0]) < 20:
            text = text.split(":", 1)[1]
        insts = [i.strip() for i in text.replace("\n", ",").split(",") if len(i.strip()) > 3]
        return insts[:5]
    except Exception as e:
        console.print(f"[yellow]Failed to fetch institutions for {region}: {e}[/yellow]")
        return []

def build_professor_search_queries(
    keywords: list[str],
    countries: list[str],
    goal: str = "research internship",
) -> list[str]:
    """
    Build a diverse set of search queries to maximize professor discovery.
    Dynamically finds top colleges if a specific region (like 'Punjab, India') is provided.
    """
    queries = []
    
    for location in countries:
        if location.lower() == "worldwide":
            continue
            
        # Consider it a specific subregion if it contains a comma or isn't exactly a base country
        is_base_country = any(location.lower() == c.lower() for c in COUNTRY_DOMAINS.keys())
        
        custom_institutions = []
        if not is_base_country:
            console.print(f"  [dim]Determining top institutions in {location}...[/dim]")
            custom_institutions = get_top_institutions_for_region(location)
            if custom_institutions:
                console.print(f"  [dim]Found regional institutions: {', '.join(custom_institutions)}[/dim]")
                
        for keyword in keywords[:5]:
            # Broad search for the location
            queries.append(f"{keyword} professor {location} research internship")
            
            # Domain specific searches if it's India
            if "india" in location.lower():
                queries.append(f"site:.ac.in {keyword} faculty professor {location}")

            if custom_institutions:
                # Add queries for the dynamically found regional colleges
                for inst in custom_institutions:
                    queries.append(f"{keyword} professor {inst} research")
            else:
                # Fallback to hardcoded top institutions if available for the base country
                base_country = next((c for c in COUNTRY_INSTITUTIONS if c.lower() in location.lower()), None)
                if base_country:
                    for inst in COUNTRY_INSTITUTIONS[base_country][:4]:
                        queries.append(f"{keyword} professor {inst} research")

    # Always add a few general lab-focused queries
    for keyword in keywords[:3]:
        country_str = " ".join(countries[:2]) if countries else ""
        queries.append(f"{keyword} research group lab {country_str} PhD students")

    return list(dict.fromkeys(queries))  # De-duplicate


def search_professors(
    keywords: list[str],
    countries: list[str],
    max_per_query: int = 8,
) -> list[dict]:
    """
    Master search function: runs multiple queries via available APIs
    and returns a combined, deduplicated list of candidate URLs.
    """
    queries = build_professor_search_queries(keywords, countries)
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    cap = max(1, config.MAX_SEARCH_QUERIES)
    limited = queries[:cap]
    console.print(f"\n[bold cyan]🔍 Running {len(limited)} search queries...[/bold cyan]")

    for i, query in enumerate(limited, 1):
        console.print(f"  [dim]({i}/{len(limited)}) {query}[/dim]")

        # Try Tavily first, fall back to Serper
        results = search_tavily(query, max_results=max_per_query)
        if not results:
            serper_results = search_serper(query, num_results=max_per_query)
            results = [
                {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                for r in serper_results
            ]

        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

        time.sleep(config.SEARCH_QUERY_SLEEP_SEC)

    console.print(f"  [green]✓ Found {len(all_results)} unique candidate URLs[/green]")
    return all_results


def search_semantic_scholar_batch(
    keywords: list[str],
    limit_per_keyword: int = 10,
) -> list[dict]:
    """
    Search Semantic Scholar for authors matching each keyword.
    Returns raw author objects from the Semantic Scholar API.
    """
    all_authors: list[dict] = []
    seen_ids: set[str] = set()

    console.print(f"\n[bold cyan]🎓 Searching Semantic Scholar...[/bold cyan]")

    # Caller (CrawlerAgent) already truncates keyword list via SEMANTIC_SCHOLAR_KEYWORD_COUNT.
    for kw_i, keyword in enumerate(keywords, 1):
        console.print(f"  [dim]Keyword ({kw_i}/{len(keywords)}): {keyword}[/dim]")
        authors = search_semantic_scholar_authors(keyword, limit=limit_per_keyword)
        for author in authors:
            author_id = author.get("authorId", "")
            if author_id and author_id not in seen_ids:
                seen_ids.add(author_id)
                all_authors.append(author)

    console.print(f"  [green]✓ Found {len(all_authors)} unique authors[/green]")
    return all_authors

"""
agents/crawler_agent.py

The Web Crawling & Data Extraction Agent.

Orchestrates the full crawling pipeline:
  1. Build search queries from an ExecutionPlan
  2. Search via Tavily / Serper / Semantic Scholar
  3. Scrape faculty profile pages
  4. Extract structured ProfessorProfile objects
  5. Filter and return clean profiles for scoring

Acts as both a standalone CLI module and a LangGraph node.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import config as app_config
from orchestrator.state import (
    ExecutionPlan,
    UserIntent,
    ProfessorProfile,
    AgentState,
)
from crawler.search_api import search_professors, search_semantic_scholar_batch
from crawler.faculty_scraper import scrape_batch
from crawler.profile_extractor import ProfileExtractor

console = Console()


# ── URL Filtering ─────────────────────────────────────────────────────────────

SKIP_DOMAINS = {
    "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com",
    "instagram.com", "youtube.com", "reddit.com", "quora.com",
    "glassdoor.com", "indeed.com", "researchgate.net",
}

ACADEMIC_SIGNALS = [
    ".edu", ".ac.uk", ".ac.in", ".edu.au", ".ac.de",
    "faculty", "professor", "staff", "/people/", "~", "cs.", "ee.",
    "scholar.google", "semanticscholar", "dblp.org",
]


def is_likely_faculty_url(url: str) -> bool:
    """Heuristically determine if a URL is likely a faculty profile page."""
    url_lower = url.lower()
    # Skip social/non-academic
    if any(skip in url_lower for skip in SKIP_DOMAINS):
        return False
    # Prefer academic domain signals
    return any(sig in url_lower for sig in ACADEMIC_SIGNALS)


def filter_candidate_urls(results: list[dict]) -> list[str]:
    """Filter search results to only likely faculty profile URLs."""
    urls = []
    for r in results:
        url = r.get("url", "")
        if url and is_likely_faculty_url(url):
            urls.append(url)
    return list(dict.fromkeys(urls))  # Deduplicate


# ── Crawler Agent ─────────────────────────────────────────────────────────────

class CrawlerAgent:
    """
    Orchestrates the full web crawling and extraction pipeline.
    """

    def __init__(self):
        self.extractor = ProfileExtractor()

    def run(
        self,
        plan: ExecutionPlan,
        max_urls: int | None = None,
    ) -> list[ProfessorProfile]:
        """
        Full pipeline: search → scrape → extract.
        Returns a list of raw ProfessorProfile objects (before scoring).
        """
        cap = max_urls if max_urls is not None else app_config.MAX_CRAWL_URLS

        console.print(Panel(
            "[bold cyan]🕷 Web Crawling Mode[/bold cyan]\n"
            "[dim]Searching for professors matching your criteria...[/dim]",
            border_style="cyan",
        ))

        keywords = plan.search_strategy.keywords
        countries = plan.search_strategy.country_filters

        # ── Step 1: Web Search ────────────────────────────────────────────────
        console.print("\n[bold]Step 1/3:[/bold] Running web searches...")
        search_results = search_professors(
            keywords=keywords,
            countries=countries,
            max_per_query=app_config.SEARCH_MAX_RESULTS_PER_QUERY,
        )

        candidate_urls = filter_candidate_urls(search_results)
        candidate_urls = candidate_urls[:cap]

        console.print(
            f"  [green]✓[/green] {len(candidate_urls)} faculty URLs identified for scraping"
        )

        # ── Step 2: Semantic Scholar ──────────────────────────────────────────
        console.print("\n[bold]Step 2/3:[/bold] Querying Semantic Scholar...")
        kw_limit = max(1, app_config.SEMANTIC_SCHOLAR_KEYWORD_COUNT)
        semantic_authors = search_semantic_scholar_batch(
            keywords=keywords[:kw_limit],
            limit_per_keyword=app_config.SEMANTIC_SCHOLAR_AUTHORS_PER_KEYWORD,
        )

        if app_config.CRAWL_SNOWBALL:
            try:
                from crawler.search_api import get_semantic_scholar_papers, get_semantic_scholar_author
                new_authors = []
                seen_author_ids = {a.get("authorId") for a in semantic_authors if a.get("authorId")}

                console.print("  [dim]Snowball Expansion: Fetching active co-authors...[/dim]")
                for author in semantic_authors[:3]:
                    aid = author.get("authorId")
                    if not aid:
                        continue

                    coauthors_count = 0
                    papers = get_semantic_scholar_papers(aid, limit=3)
                    for p in papers:
                        for co in p.get("authors", []):
                            co_id = co.get("authorId")
                            if co_id and co_id not in seen_author_ids:
                                seen_author_ids.add(co_id)
                                full_co = get_semantic_scholar_author(co_id)
                                if full_co and full_co.get("name"):
                                    new_authors.append(full_co)
                                    coauthors_count += 1
                            if coauthors_count >= 2:
                                break
                        if coauthors_count >= 2:
                            break

                if new_authors:
                    console.print(
                        f"  [green]✓ Snowball expansion discovered {len(new_authors)} relevant co-authors[/green]"
                    )
                    semantic_authors.extend(new_authors)
            except Exception as e:
                console.print(f"  [yellow]⚠ Snowball expansion failed: {e}[/yellow]")

        # ── Step 3: Scrape & Extract ──────────────────────────────────────────
        console.print(f"\n[bold]Step 3/3:[/bold] Scraping {len(candidate_urls)} pages...")

        scraped_pages = []
        if candidate_urls:
            scraped_pages = scrape_batch(
                candidate_urls, delay=app_config.SCRAPE_BATCH_DELAY
            )

        # ── Profile Extraction ────────────────────────────────────────────────
        profiles = self.extractor.batch_extract(
            scraped_pages=scraped_pages,
            semantic_authors=semantic_authors,
            use_llm=True,
        )

        # ── Apply Basic Filters ───────────────────────────────────────────────
        if plan.filtering_criteria.require_email:
            before = len(profiles)
            profiles = [p for p in profiles if p.email]
            console.print(
                f"\n  [dim]Email filter: kept {len(profiles)}/{before} professors with known emails[/dim]"
            )

        console.print(
            f"\n[bold green]✓ Crawl complete. {len(profiles)} professor profiles ready for scoring.[/bold green]"
        )

        return profiles

    def run_cli(self, plan: ExecutionPlan, intent: UserIntent) -> list[ProfessorProfile]:
        """CLI wrapper for the crawling pipeline."""
        profiles = self.run(plan)
        return profiles


# ── LangGraph Node ────────────────────────────────────────────────────────────

def crawl_node(state: AgentState) -> AgentState:
    """LangGraph node: plan → raw professor profiles."""
    plan = ExecutionPlan.model_validate(state["plan"])
    agent = CrawlerAgent()
    profiles = agent.run(plan)

    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": (
            f"✓ Crawling complete. Found {len(profiles)} professor profiles. "
            "Now ranking by relevance..."
        ),
    })

    return {
        **state,
        "professors": [p.model_dump() for p in profiles],
        "messages": messages,
        "mode": "scoring",
    }

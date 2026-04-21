"""
crawler/profile_extractor.py

LLM-powered profile extraction.

Takes raw page text from scraped faculty pages and uses Gemini to extract
structured ProfessorProfile fields that couldn't be parsed with regex/CSS selectors.

Also handles Semantic Scholar author objects → ProfessorProfile conversion.
"""

from __future__ import annotations

import json
import re
from typing import Optional
from uuid import uuid4

import config
from orchestrator.state import ProfessorProfile, Publication
from rich.console import Console

console = Console()

# ── LLM Extraction Prompt ─────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are helping extract structured information about a university professor from their webpage text.

PAGE URL: {url}
PAGE TEXT (truncated):
\"\"\"
{page_text}
\"\"\"

Extract the following fields as a JSON object. Use null if you cannot find the value.

{{
  "name": "<Full name of professor>",
  "email": "<email address or null>",
  "institution": "<University name>",
  "department": "<Department name>",
  "lab_url": "<URL of their lab/group page or null>",
  "research_keywords": ["<keyword1>", "<keyword2>", ...],
  "recent_publications": [
    {{"title": "...", "year": 2023, "venue": "...", "url": null}},
    ...
  ],
  "bio_snippet": "<2-3 sentence bio summary>",
  "accepting_students": <true | false | null>
}}

Rules:
- Extract up to 5 most recent publications.
- Research keywords: only **technical research topics** from the page body (e.g. "game theory", "formal verification").
  NEVER use website section titles or nav labels (e.g. "Highlights", "Research Profile", "Qualifications", "Academics").
- For accepting_students, only return true/false if clearly stated; otherwise null.
- Return ONLY the JSON, no other text.
"""


class ProfileExtractor:
    """
    Extracts structured ProfessorProfile from raw scraped data using Gemini.
    Falls back to heuristic data when LLM call fails.
    """

    def __init__(self):
        self.llm = config.get_groq_llm()

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            console.print(f"  [red]LLM extraction error: {e}[/red]")
            return ""

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {}

    def extract_from_scraped(self, scraped: dict) -> Optional[ProfessorProfile]:
        """
        Given raw scraped dict (from faculty_scraper), produce a ProfessorProfile.
        Uses LLM to fill gaps left by the heuristic scraper.
        """
        url = scraped.get("profile_url", "")
        raw_text = scraped.get("raw_page_text", "")

        # Only call LLM if we have meaningful page text
        if raw_text and len(raw_text) > 200:
            prompt = EXTRACTION_PROMPT.format(
                url=url,
                page_text=raw_text[:3500],  # Token budget
            )
            raw = self._call_llm(prompt)
            data = self._parse_json(raw)
        else:
            data = {}

        # Merge: prefer LLM data, fall back to heuristic scraper data
        name = data.get("name") or scraped.get("name")
        if not name:
            return None  # Can't create profile without a name

        email = data.get("email") or scraped.get("email")
        institution = data.get("institution") or scraped.get("institution", "Unknown")
        department = data.get("department") or scraped.get("department")
        lab_url = data.get("lab_url") or scraped.get("lab_url")
        bio_snippet = data.get("bio_snippet") or scraped.get("bio_snippet")
        accepting = data.get("accepting_students")
        if accepting is None:
            accepting = scraped.get("accepting_students")

        # Merge research keywords (LLM + heuristic, deduplicated)
        llm_keywords = data.get("research_keywords", []) or []
        heuristic_keywords = scraped.get("research_keywords", []) or []
        merged_keywords = list(dict.fromkeys(llm_keywords + heuristic_keywords))[:15]

        # Parse publications
        publications = []
        for pub_dict in (data.get("recent_publications") or [])[:5]:
            if isinstance(pub_dict, dict) and pub_dict.get("title"):
                publications.append(Publication(
                    title=pub_dict.get("title", ""),
                    year=pub_dict.get("year"),
                    venue=pub_dict.get("venue"),
                    url=pub_dict.get("url"),
                ))

        return ProfessorProfile(
            id=str(uuid4()),
            name=name,
            email=email,
            institution=institution,
            department=department,
            lab_url=lab_url,
            profile_url=url,
            research_keywords=merged_keywords,
            recent_publications=publications,
            accepting_students=accepting,
            bio_snippet=bio_snippet,
            raw_page_text=raw_text[:2000] if raw_text else None,
        )

    def extract_from_semantic_scholar(self, author: dict) -> Optional[ProfessorProfile]:
        """
        Convert a Semantic Scholar author API response to a ProfessorProfile.
        No LLM needed — Semantic Scholar provides structured data directly.
        """
        name = author.get("name")
        if not name:
            return None

        affiliations = author.get("affiliations", [])
        institution = affiliations[0] if affiliations else "Unknown"

        # Extract papers
        papers_raw = author.get("papers", [])
        publications = []
        for paper in (papers_raw or [])[:5]:
            if paper.get("title"):
                publications.append(Publication(
                    title=paper["title"],
                    year=paper.get("year"),
                    venue=paper.get("venue"),
                ))

        # Build keyword list from paper titles (heuristic)
        all_titles = " ".join(p.title for p in publications)
        research_keywords = self._extract_keywords_from_text(all_titles)

        return ProfessorProfile(
            id=str(uuid4()),
            name=name,
            email=None,  # Semantic Scholar doesn't provide emails
            institution=institution,
            department=None,
            profile_url=f"https://www.semanticscholar.org/author/{author.get('authorId', '')}",
            research_keywords=research_keywords,
            recent_publications=publications,
            accepting_students=None,
        )

    def _extract_keywords_from_text(self, text: str) -> list[str]:
        """Simple keyword extraction from text using common ML/AI terms."""
        ml_terms = [
            "machine learning", "deep learning", "neural network", "computer vision",
            "natural language processing", "NLP", "reinforcement learning",
            "federated learning", "graph neural", "transformer", "BERT", "GPT",
            "robotics", "autonomous", "optimization", "generative", "diffusion",
            "attention mechanism", "large language model", "knowledge graph",
            "semantic", "multimodal", "speech recognition", "image segmentation",
        ]
        found = [term for term in ml_terms if term.lower() in text.lower()]
        return found[:8]

    def batch_extract(
        self,
        scraped_pages: list[dict],
        semantic_authors: list[dict] | None = None,
        use_llm: bool = True,
    ) -> list[ProfessorProfile]:
        """
        Extract profiles from all scraped pages + Semantic Scholar authors.
        Returns a cleaned, non-null list of ProfessorProfile objects.
        """
        profiles: list[ProfessorProfile] = []

        console.print(f"\n[bold cyan]🧠 Extracting professor profiles...[/bold cyan]")

        # From scraped pages
        for i, scraped in enumerate(scraped_pages, 1):
            url = scraped.get("profile_url", "")[:60]
            console.print(f"  [dim]({i}/{len(scraped_pages)})[/dim] Extracting from {url}...")
            profile = self.extract_from_scraped(scraped)
            if profile:
                profiles.append(profile)
                console.print(f"    [green]✓[/green] {profile.name} | {profile.institution}")
            else:
                console.print(f"    [yellow]✗ Skipped (no name found)[/yellow]")

        # From Semantic Scholar
        if semantic_authors:
            console.print(f"\n  Processing {len(semantic_authors)} Semantic Scholar authors...")
            for author in semantic_authors:
                profile = self.extract_from_semantic_scholar(author)
                if profile:
                    profiles.append(profile)

        # Deduplicate by name + institution
        seen = set()
        deduped = []
        for p in profiles:
            key = (p.name.lower().strip(), p.institution.lower().strip())
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        console.print(
            f"\n[bold green]✓ Extracted {len(deduped)} unique professor profiles[/bold green]"
            f" ({len(profiles) - len(deduped)} duplicates removed)"
        )
        return deduped

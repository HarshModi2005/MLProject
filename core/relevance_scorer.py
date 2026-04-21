"""
core/relevance_scorer.py

Multi-factor relevance scoring engine.

Scores each ProfessorProfile against the user's UserIntent using:
  1. Keyword overlap score (lexical)
  2. Semantic embedding similarity (contextual)
  3. Publication recency score
  4. Geographic match score
  5. Student acceptance signal score

Final score = weighted combination of all factors ∈ [0.0, 1.0]
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table

from core.student_context import build_matching_profile_text
from orchestrator.state import ProfessorProfile, UserIntent, ExecutionPlan

console = Console()


# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "keyword_overlap":    0.30,
    "embedding_sim":      0.20,
    "recency":            0.15,
    "geographic":         0.25,
    "accepting_students": 0.10,
}


# ── Sub-scorer: Keyword Overlap ───────────────────────────────────────────────

def keyword_overlap_score(
    professor: ProfessorProfile,
    user_domains: list[str],
    plan_keywords: list[str],
) -> float:
    """
    Fraction of user interest terms found in professor's research keywords.
    Returns 0.0 – 1.0.
    """
    if not user_domains and not plan_keywords:
        return 0.5  # Neutral if no data

    all_user_terms = set(
        term.lower()
        for term in (user_domains + plan_keywords)
        for term in term.split()
        if len(term) > 3
    )

    prof_text = " ".join(professor.research_keywords).lower()
    if professor.bio_snippet:
        prof_text += " " + professor.bio_snippet.lower()
    prof_words = set(prof_text.split())

    if not all_user_terms:
        return 0.0

    matches = all_user_terms.intersection(prof_words)
    return min(len(matches) / len(all_user_terms), 1.0)


def llm_similarity_score(
    professor: ProfessorProfile,
    intent: UserIntent,
) -> float:
    """
    Zero-shot LLM scorer: student ↔ lab fit for a **research internship**.
    Uses curated CV signals + stated domains (not a raw resume dump).
    """
    try:
        import config
        llm = config.get_groq_llm()

        prof_text = f"Institution: {professor.institution or 'Unknown'}\n"
        prof_text += f"Keywords: {', '.join(professor.research_keywords)}\n"
        if professor.recent_publications:
            prof_text += "Publications: " + ", ".join(p.title for p in professor.recent_publications) + "\n"
        if professor.bio_snippet:
            prof_text += f"Bio: {professor.bio_snippet}\n"

        student_block = build_matching_profile_text(intent)

        prompt = f"""
You are an expert academic evaluator scoring fit for an undergraduate/masters **research internship**.

Rate how well this student's demonstrated research-relevant work and interests match this professor's lab
(0.0 = poor fit, 1.0 = strong fit). Consider methods, topics, and depth — not generic school prestige alone.

STUDENT (curated signals + summary):
{student_block}

PROFESSOR:
{prof_text}

Reply ONLY with a float between 0.0 and 1.0. No other text.
"""
        response = llm.invoke(prompt)
        text = response.content.strip()
        
        import re
        match = re.search(r"^0\.\d+|^1\.0|^0", text)
        if match:
            return float(match.group())
        return 0.5
    except Exception as e:
        console.print(f"  [yellow]⚠ LLM Scorer error: {e}. Using neutral score.[/yellow]")
        return 0.5


# ── Sub-scorer: Publication Recency ──────────────────────────────────────────

def recency_score(professor: ProfessorProfile) -> float:
    """
    Score based on how recent the professor's publications are.
    Most recent paper in last 1 year → 1.0; 5+ years → 0.0.
    """
    if not professor.recent_publications:
        return 0.3  # Unknown recency — slight penalty

    years = [
        p.year for p in professor.recent_publications
        if p.year and 2000 <= p.year <= 2030
    ]
    if not years:
        return 0.3

    latest_year = max(years)
    current_year = datetime.utcnow().year
    age = current_year - latest_year

    # Sigmoid decay: 0 years old → 1.0, 5 years old → ~0.2
    return max(0.0, min(1.0, math.exp(-0.4 * age)))


# ── Sub-scorer: Geographic Match ──────────────────────────────────────────────

COUNTRY_ALIASES: dict[str, list[str]] = {
    "India": [
        "india", "iit ", "iisc", "iiit", "tifr", "bits", ".ac.in",
        "indian institute", "iit bombay", "iit delhi", "iit madras",
        "iit kanpur", "iit kharagpur", "iit hyderabad", "iit roorkee",
        "iit gandhinagar", "iisc bangalore", "iiit hyderabad", "iiit delhi",
        "isi kolkata", "cmi chennai", "national institute of technology",
    ],
    "USA": ["united states", "usa", "u.s.", "us"],
    "United Kingdom": ["uk", "united kingdom", "england", "britain"],
    "Germany": ["germany", "deutschland"],
    "Canada": ["canada"],
    "Australia": ["australia"],
    "Singapore": ["singapore"],
    "Switzerland": ["switzerland", "eth", "epfl"],
    "France": ["france"],
    "Netherlands": ["netherlands", "holland"],
}


def geographic_score(
    professor: ProfessorProfile,
    target_countries: list[str],
) -> float:
    """
    1.0 if professor's institution matches a target country/region.
    0.0 if mismatch (strict filtering when a country/region is specified).
    0.5 if no preference specified ('Worldwide').
    """
    if not target_countries or "Worldwide" in target_countries:
        return 0.5  # No preference

    institution = (professor.institution or "").lower()
    profile_url = (professor.profile_url or "").lower()
    bio = (professor.bio_snippet or "").lower()
    text = institution + " " + profile_url + " " + bio

    for location in target_countries:
        loc_lower = location.lower()
        # Extract base country to pull aliases (e.g., "india" from "punjab, india")
        country_key = next((k for k in COUNTRY_ALIASES.keys() if k.lower() in loc_lower), None)
        
        # Build search terms for this specific location
        search_terms = set()
        
        # 1. Any specific region parts (e.g. "punjab")
        parts = [p.strip() for p in loc_lower.split(",")]
        for part in parts:
            if len(part) > 2 and not any(part == k.lower() for k in COUNTRY_ALIASES.keys()):
                search_terms.add(part)
                
        # 2. General country aliases if no specific subregion was provided,
        # OR if we want to also accept general country matches (optional).
        # We enforce that if a specific region IS requested, it MUST match the region OR a generic country alias. 
        # Actually, if they asked for Punjab, we should ideally find "Punjab" or just "India" but give Punjab a boost.
        # But for strict filtering, we will let it pass if ANY part or alias matches.
        if country_key:
            search_terms.update(COUNTRY_ALIASES[country_key])
        else:
            search_terms.add(loc_lower)

        # Check if any term matches
        if any(term in text for term in search_terms):
            # If they specified a subregion (e.g. punjab), boost if the subregion matches exactly
            if any(p in text for p in parts if p not in COUNTRY_ALIASES.get(country_key, [])):
                return 1.0  # Perfect subregion match
            return 0.8  # Valid country match, but maybe didn't explicitly mention the subregion

    # Hard fail: user specified a country/region but professor doesn't match
    return 0.0


# ── Sub-scorer: Student Acceptance ───────────────────────────────────────────

def acceptance_score(professor: ProfessorProfile) -> float:
    """
    1.0 if actively looking for students,
    0.5 if unknown,
    0.0 if not accepting.
    """
    if professor.accepting_students is True:
        return 1.0
    if professor.accepting_students is False:
        return 0.0
    return 0.5


# ── Master Scorer ─────────────────────────────────────────────────────────────

class RelevanceScorer:
    """
    Computes a composite relevance score for each professor
    and selects the top-K candidates.
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings

    def score(
        self,
        professor: ProfessorProfile,
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> float:
        """Compute and set the relevance score on a professor profile."""
        user_domains = intent.research_domains
        plan_keywords = plan.search_strategy.keywords
        user_profile = intent.user_profile_summary or ""

        kw = keyword_overlap_score(professor, user_domains, plan_keywords)
        rec = recency_score(professor)
        geo = geographic_score(professor, intent.countries)
        acc = acceptance_score(professor)

        if self.use_embeddings and (
            user_profile or intent.research_internship_highlights or intent.cv_key_projects
        ):
            emb = llm_similarity_score(professor, intent)
        else:
            emb = kw  # Proxy

        score = (
            WEIGHTS["keyword_overlap"]    * kw +
            WEIGHTS["embedding_sim"]      * emb +
            WEIGHTS["recency"]            * rec +
            WEIGHTS["geographic"]         * geo +
            WEIGHTS["accepting_students"] * acc
        )

        professor.relevance_score = round(score, 4)
        return professor.relevance_score

    def score_and_rank(
        self,
        professors: list[ProfessorProfile],
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> list[ProfessorProfile]:
        """Score all professors and return sorted by relevance (descending)."""
        console.print(f"\n[bold cyan]📊 Scoring {len(professors)} professors...[/bold cyan]")

        for i, prof in enumerate(professors, 1):
            self.score(prof, intent, plan)

        ranked = sorted(professors, key=lambda p: p.relevance_score, reverse=True)
        console.print(f"[green]✓ Scoring complete.[/green]")
        return ranked

    def shortlist(
        self,
        professors: list[ProfessorProfile],
        intent: UserIntent,
        plan: ExecutionPlan,
        top_k: Optional[int] = None,
    ) -> list[ProfessorProfile]:
        """
        Score, rank, filter by min_score, and return top-K professors.
        Pre-filters by geography when a target country is specified.
        """
        # ── Geographic pre-filter ───────────────────────────────────────────────
        target_countries = intent.countries
        if target_countries and "Worldwide" not in target_countries:
            before = len(professors)
            professors = [
                p for p in professors
                if geographic_score(p, target_countries) > 0.0
            ]
            dropped = before - len(professors)
            if dropped:
                console.print(
                    f"  [dim]Geographic filter: dropped {dropped} professors "
                    f"not in {', '.join(target_countries)}[/dim]"
                )

        ranked = self.score_and_rank(professors, intent, plan)

        min_score = plan.filtering_criteria.min_relevance_score
        filtered = [p for p in ranked if p.relevance_score >= min_score]

        k = top_k or plan.top_k_professors
        shortlisted = filtered[:k]

        self._display_shortlist(shortlisted)
        return shortlisted

    def _display_shortlist(self, professors: list[ProfessorProfile]) -> None:
        """Display top professors in a formatted table."""
        table = Table(
            title=f"🏆 Top {len(professors)} Shortlisted Professors",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Name", style="bold")
        table.add_column("Institution")
        table.add_column("Domains")
        table.add_column("Score", style="bold green")
        table.add_column("Email", style="dim")
        table.add_column("Accepting?", style="dim")

        for i, prof in enumerate(professors, 1):
            domains = ", ".join(prof.research_keywords[:3])
            accepting = (
                "✓" if prof.accepting_students is True
                else "✗" if prof.accepting_students is False
                else "?"
            )
            table.add_row(
                str(i),
                prof.name,
                prof.institution[:25],
                domains[:35],
                f"{prof.relevance_score:.3f}",
                prof.email[:25] if prof.email else "(none)",
                accepting,
            )

        console.print(table)


# ── LangGraph Node ────────────────────────────────────────────────────────────

def scoring_node(state: dict) -> dict:
    """LangGraph node: score and rank crawled professors."""
    prof_dicts = state.get("professors", [])
    if not prof_dicts:
        return state

    professors = [ProfessorProfile.model_validate(p) for p in prof_dicts]
    intent_data = state.get("intent")
    plan_data = state.get("plan")
    
    if not intent_data or not plan_data:
        return state

    intent = UserIntent.model_validate(intent_data)
    plan = ExecutionPlan.model_validate(plan_data)

    scorer = RelevanceScorer()
    shortlisted = scorer.shortlist(professors, intent, plan)

    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": f"Scoring complete. {len(shortlisted)} professors met criteria and were shortlisted."
    })

    return {
        **state,
        "shortlisted": [p.model_dump() for p in shortlisted],
        "messages": messages,
        "mode": "email_draft"
    }

"""
core/relevance_scorer.py

Multi-factor relevance scoring engine.

Scores each ProfessorProfile against the user's UserIntent using:
  1. Keyword overlap score (lexical)
  2. Semantic / LLM similarity (contextual)
  3. Publication recency score
  4. Geographic match score
  5. Student acceptance signal score

Final score = weighted combination of all factors ∈ [0.0, 1.0]
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table

import config
from core.observability import log_llm_call
from core.student_context import build_matching_profile_text
from orchestrator.state import (
    ExecutionPlan,
    ProfessorProfile,
    PublicationMatch,
    RelevanceBreakdown,
    UserIntent,
)

console = Console()


# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "keyword_overlap": 0.30,
    "embedding_sim": 0.20,
    "recency": 0.15,
    "geographic": 0.25,
    "accepting_students": 0.10,
}


# ── Sub-scorer: Keyword Overlap ───────────────────────────────────────────────


def keyword_overlap_detail(
    professor: ProfessorProfile,
    user_domains: list[str],
    plan_keywords: list[str],
) -> tuple[float, list[str]]:
    """
    Fraction of user interest terms found in professor text; returns matched terms.
    """
    if not user_domains and not plan_keywords:
        return 0.5, []

    all_user_terms: set[str] = set()
    for phrase in user_domains + plan_keywords:
        for w in phrase.split():
            wl = w.lower().strip()
            if len(wl) > 3:
                all_user_terms.add(wl)

    prof_text = " ".join(professor.research_keywords).lower()
    if professor.bio_snippet:
        prof_text += " " + professor.bio_snippet.lower()
    prof_words = set(prof_text.split())

    if not all_user_terms:
        return 0.0, []

    matches = sorted(all_user_terms.intersection(prof_words))
    score = min(len(matches) / len(all_user_terms), 1.0)
    return score, matches


def keyword_overlap_score(
    professor: ProfessorProfile,
    user_domains: list[str],
    plan_keywords: list[str],
) -> float:
    s, _ = keyword_overlap_detail(professor, user_domains, plan_keywords)
    return s


def publication_title_matches(
    professor: ProfessorProfile,
    user_domains: list[str],
    plan_keywords: list[str],
    top_n: int = 3,
) -> list[PublicationMatch]:
    term_set: set[str] = set()
    for phrase in user_domains + plan_keywords:
        for w in phrase.split():
            wl = w.lower().strip()
            if len(wl) > 3:
                term_set.add(wl)
    if not term_set:
        return []
    hits: list[PublicationMatch] = []
    for pub in professor.recent_publications or []:
        tl = (pub.title or "").lower()
        hit_terms = [t for t in term_set if t in tl]
        if hit_terms:
            hits.append(
                PublicationMatch(
                    title=pub.title,
                    reason="title overlap: " + ", ".join(hit_terms[:5]),
                )
            )
        if len(hits) >= top_n:
            break
    return hits[:top_n]


def llm_similarity_score(
    professor: ProfessorProfile,
    intent: UserIntent,
) -> float:
    """
    Zero-shot LLM scorer: student ↔ lab fit for a **research internship**.
    """
    try:
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
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        dt = time.perf_counter() - t0
        text = (response.content or "").strip()
        log_llm_call(
            label="relevance_llm_fit",
            prompt_chars=len(prompt),
            response_chars=len(text),
            latency_sec=dt,
            model=config.GROQ_MODEL,
        )

        match = re.search(r"^0\.\d+|^1\.0|^0", text)
        if match:
            return float(match.group())
        return 0.5
    except Exception as e:
        console.print(f"  [yellow]⚠ LLM Scorer error: {e}. Using neutral score.[/yellow]")
        return 0.5


# ── Sub-scorer: Publication Recency ──────────────────────────────────────────


def recency_score(professor: ProfessorProfile) -> float:
    if not professor.recent_publications:
        return 0.3

    years = [
        p.year for p in professor.recent_publications
        if p.year and 2000 <= p.year <= 2030
    ]
    if not years:
        return 0.3

    latest_year = max(years)
    current_year = datetime.utcnow().year
    age = current_year - latest_year
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
    if not target_countries or "Worldwide" in target_countries:
        return 0.5

    institution = (professor.institution or "").lower()
    profile_url = (professor.profile_url or "").lower()
    bio = (professor.bio_snippet or "").lower()
    text = institution + " " + profile_url + " " + bio

    for location in target_countries:
        loc_lower = location.lower()
        country_key = next((k for k in COUNTRY_ALIASES.keys() if k.lower() in loc_lower), None)

        search_terms: set[str] = set()
        parts = [p.strip() for p in loc_lower.split(",")]
        for part in parts:
            if len(part) > 2 and not any(part == k.lower() for k in COUNTRY_ALIASES.keys()):
                search_terms.add(part)

        if country_key:
            search_terms.update(COUNTRY_ALIASES[country_key])
        else:
            search_terms.add(loc_lower)

        if any(term in text for term in search_terms):
            if any(p in text for p in parts if p not in COUNTRY_ALIASES.get(country_key, [])):
                return 1.0
            return 0.8

    return 0.0


def acceptance_score(professor: ProfessorProfile) -> float:
    if professor.accepting_students is True:
        return 1.0
    if professor.accepting_students is False:
        return 0.0
    return 0.5


# ── Shortlist result ──────────────────────────────────────────────────────────


@dataclass
class ShortlistDiagnostics:
    n_candidates_in: int
    n_after_geo_filter: int
    min_relevance_threshold: float
    top_k_cap: int
    max_score: float
    n_above_threshold: int
    top_preview: list[tuple[str, str, float]]


@dataclass
class ShortlistResult:
    shortlisted: list[ProfessorProfile]
    diagnostics: ShortlistDiagnostics


def print_shortlist_guidance(
    diagnostics: ShortlistDiagnostics,
    intent: UserIntent,
    plan: ExecutionPlan,
) -> None:
    """Actionable suggestions when nobody passes the threshold."""
    console.print("\n[bold yellow]Shortlist is empty — diagnostics[/bold yellow]")
    t = Table(show_header=False, box=None)
    t.add_row("Candidates in (before geo)", str(diagnostics.n_candidates_in))
    t.add_row("After geographic filter", str(diagnostics.n_after_geo_filter))
    t.add_row("Min relevance threshold", f"{diagnostics.min_relevance_threshold:.2f}")
    t.add_row("Top-K cap", str(diagnostics.top_k_cap))
    t.add_row("Highest score seen", f"{diagnostics.max_score:.3f}")
    t.add_row("Count ≥ threshold", str(diagnostics.n_above_threshold))
    console.print(t)

    if diagnostics.top_preview:
        console.print("[dim]Top scores (may be below threshold):[/dim]")
        for name, inst, sc in diagnostics.top_preview:
            console.print(f"  • [bold]{name}[/bold] ({inst[:40]}) — {sc:.3f}")

    console.print("\n[bold cyan]Suggested next steps[/bold cyan]")
    sugg: list[str] = []
    if diagnostics.max_score < plan.filtering_criteria.min_relevance_score:
        sugg.append(
            f"Lower `filtering_criteria.min_relevance_score` in the plan "
            f"(currently {plan.filtering_criteria.min_relevance_score:.2f}; max observed {diagnostics.max_score:.3f})."
        )
    if intent.countries and "Worldwide" not in intent.countries:
        sugg.append("Broaden **countries** to include `Worldwide` if geography is filtering too hard.")
    if plan.search_strategy.keywords and len(plan.search_strategy.keywords) < 4:
        sugg.append("Add more **search keywords** in the execution plan.")
    sugg.append(
        f"Raise **top_k_professors** (currently {plan.top_k_professors}) or "
        f"env `TOP_K_PROFESSORS` / `MAX_CRAWL_URLS` for more candidates."
    )
    if plan.filtering_criteria.require_email:
        sugg.append("Temporarily set `require_email` to false in filtering criteria to score profiles without email.")
    for s in sugg:
        console.print(f"  – {s}")


# ── Master Scorer ─────────────────────────────────────────────────────────────


class RelevanceScorer:
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings

    def score(
        self,
        professor: ProfessorProfile,
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> float:
        user_domains = intent.research_domains
        plan_keywords = plan.search_strategy.keywords
        user_profile = intent.user_profile_summary or ""

        kw, matched_terms = keyword_overlap_detail(professor, user_domains, plan_keywords)
        rec = recency_score(professor)
        geo = geographic_score(professor, intent.countries)
        acc = acceptance_score(professor)
        pub_matches = publication_title_matches(professor, user_domains, plan_keywords)

        if self.use_embeddings and (
            user_profile or intent.research_internship_highlights or intent.cv_key_projects
        ):
            emb = llm_similarity_score(professor, intent)
        else:
            emb = kw

        total = (
            WEIGHTS["keyword_overlap"] * kw
            + WEIGHTS["embedding_sim"] * emb
            + WEIGHTS["recency"] * rec
            + WEIGHTS["geographic"] * geo
            + WEIGHTS["accepting_students"] * acc
        )

        professor.relevance_breakdown = RelevanceBreakdown(
            keyword_overlap=round(kw, 4),
            embedding_sim=round(emb, 4),
            recency=round(rec, 4),
            geographic=round(geo, 4),
            accepting_students=round(acc, 4),
            weights_used=dict(WEIGHTS),
            matched_terms=matched_terms[:20],
            publication_matches=pub_matches,
        )
        professor.relevance_score = round(total, 4)
        return professor.relevance_score

    def score_and_rank(
        self,
        professors: list[ProfessorProfile],
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> list[ProfessorProfile]:
        console.print(f"\n[bold cyan]📊 Scoring {len(professors)} professors...[/bold cyan]")

        for prof in professors:
            self.score(prof, intent, plan)

        ranked = sorted(professors, key=lambda p: p.relevance_score, reverse=True)
        console.print("[green]✓ Scoring complete.[/green]")
        return ranked

    def shortlist(
        self,
        professors: list[ProfessorProfile],
        intent: UserIntent,
        plan: ExecutionPlan,
        top_k: Optional[int] = None,
    ) -> ShortlistResult:
        n_in = len(professors)
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

        n_geo = len(professors)
        ranked = self.score_and_rank(professors, intent, plan)

        min_score = plan.filtering_criteria.min_relevance_score
        filtered = [p for p in ranked if p.relevance_score >= min_score]
        k = top_k or plan.top_k_professors
        shortlisted = filtered[:k]

        max_score = ranked[0].relevance_score if ranked else 0.0
        n_above = sum(1 for p in ranked if p.relevance_score >= min_score)
        preview = [(p.name, p.institution or "", p.relevance_score) for p in ranked[:5]]

        diag = ShortlistDiagnostics(
            n_candidates_in=n_in,
            n_after_geo_filter=n_geo,
            min_relevance_threshold=min_score,
            top_k_cap=k,
            max_score=max_score,
            n_above_threshold=n_above,
            top_preview=preview,
        )

        if shortlisted:
            self._display_shortlist(shortlisted)
        elif ranked:
            console.print(
                "[yellow]No professors met min_relevance_score; showing top 5 for transparency.[/yellow]"
            )
            self._display_shortlist(ranked[:5])

        return ShortlistResult(shortlisted=shortlisted, diagnostics=diag)

    def _display_shortlist(self, professors: list[ProfessorProfile]) -> None:
        table = Table(
            title=f"🏆 Professors ({len(professors)})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Name", style="bold")
        table.add_column("Institution")
        table.add_column("Score", style="bold green")
        table.add_column("kw|emb|geo", style="dim")
        table.add_column("Email", style="dim")

        for i, prof in enumerate(professors, 1):
            bd = prof.relevance_breakdown
            triple = (
                f"{bd.keyword_overlap:.2f}|{bd.embedding_sim:.2f}|{bd.geographic:.2f}"
                if bd
                else "—"
            )
            table.add_row(
                str(i),
                prof.name[:28],
                (prof.institution or "")[:28],
                f"{prof.relevance_score:.3f}",
                triple,
                (prof.email or "(none)")[:26],
            )

        console.print(table)
        for prof in professors[:10]:
            bd = prof.relevance_breakdown
            if not bd:
                continue
            terms = ", ".join(bd.matched_terms[:8]) if bd.matched_terms else "—"
            pubs = "; ".join(f"{m.title[:60]} ({m.reason})" for m in bd.publication_matches[:2])
            console.print(
                f"  [dim]↳ {prof.name[:30]}: terms [{terms}]"
                + (f" | pubs: {pubs}" if pubs else "")
                + "[/dim]"
            )


def scoring_node(state: dict) -> dict:
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
    result = scorer.shortlist(professors, intent, plan)
    shortlisted = result.shortlisted

    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": (
            f"Scoring complete. {len(shortlisted)} professors met criteria and were shortlisted."
        ),
    })

    return {
        **state,
        "shortlisted": [p.model_dump() for p in shortlisted],
        "messages": messages,
        "mode": "email_draft",
    }

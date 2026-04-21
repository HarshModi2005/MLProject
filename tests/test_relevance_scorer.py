"""Golden-style checks for relevance scoring (LLM mocked)."""

from __future__ import annotations

from unittest import mock

import core.relevance_scorer as rs
from orchestrator.state import (
    EmailStrategy,
    ExecutionPlan,
    FilteringCriteria,
    ProfessorProfile,
    Publication,
    SearchStrategy,
    SendingSchedule,
    UserIntent,
)


def _minimal_plan(min_score: float = 0.2) -> ExecutionPlan:
    return ExecutionPlan(
        search_strategy=SearchStrategy(
            keywords=["machine", "learning"],
            country_filters=["USA"],
        ),
        filtering_criteria=FilteringCriteria(min_relevance_score=min_score),
        email_strategy=EmailStrategy(),
        sending_schedule=SendingSchedule(),
        top_k_professors=5,
    )


def _minimal_intent() -> UserIntent:
    return UserIntent(
        research_domains=["machine learning"],
        countries=["Worldwide"],
        user_name="Test",
        user_profile_summary="Student in ML.",
    )


def test_keyword_overlap_detail_collects_terms():
    prof = ProfessorProfile(
        name="Dr. X",
        institution="Test University",
        research_keywords=["deep learning", "vision"],
        bio_snippet="We work on machine learning applications.",
    )
    score, terms = rs.keyword_overlap_detail(
        prof,
        user_domains=["machine learning"],
        plan_keywords=["vision"],
    )
    assert score > 0
    assert "machine" in terms or "learning" in terms or "vision" in terms


@mock.patch.object(rs, "llm_similarity_score", return_value=0.55)
def test_score_attaches_breakdown(_mock_llm):
    intent = _minimal_intent()
    plan = _minimal_plan(min_score=0.01)
    prof = ProfessorProfile(
        name="Dr. Y",
        institution="MIT",
        email="y@mit.edu",
        research_keywords=["machine learning", "nlp"],
        recent_publications=[Publication(title="Machine learning for NLP", year=2024)],
        accepting_students=True,
    )
    scorer = rs.RelevanceScorer()
    scorer.score(prof, intent, plan)
    assert prof.relevance_score > 0
    assert prof.relevance_breakdown is not None
    assert prof.relevance_breakdown.keyword_overlap >= 0
    assert "keyword_overlap" in prof.relevance_breakdown.weights_used


@mock.patch.object(rs, "llm_similarity_score", return_value=0.5)
def test_shortlist_returns_result_with_diagnostics(_mock_llm):
    intent = _minimal_intent()
    plan = _minimal_plan(min_score=0.99)
    prof = ProfessorProfile(
        name="Dr. Z",
        institution="Stanford",
        email="z@stanford.edu",
        research_keywords=["biology"],
    )
    scorer = rs.RelevanceScorer()
    result = scorer.shortlist([prof], intent, plan)
    assert hasattr(result, "shortlisted")
    assert hasattr(result, "diagnostics")
    assert result.diagnostics.n_candidates_in == 1
    assert result.diagnostics.max_score >= 0

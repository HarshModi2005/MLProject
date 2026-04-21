"""
core/__init__.py

Core utilities: embeddings, vector store, relevance scoring, CV parsing.

`RelevanceScorer` / `scoring_node` are loaded lazily to avoid import cycles
(orchestrator → agents → core.student_context ↔ orchestrator.state).
"""

from __future__ import annotations

from typing import Any

from core.cv_parser import CVParser
from core.embeddings import cosine_similarity, embed_single, embed_texts
from core.vector_store import VectorStore

__all__ = [
    "embed_texts",
    "embed_single",
    "cosine_similarity",
    "VectorStore",
    "RelevanceScorer",
    "scoring_node",
    "CVParser",
]


def __getattr__(name: str) -> Any:
    if name == "RelevanceScorer":
        from core.relevance_scorer import RelevanceScorer

        return RelevanceScorer
    if name == "scoring_node":
        from core.relevance_scorer import scoring_node

        return scoring_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

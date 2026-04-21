"""
core/embeddings.py

Sentence embedding utilities using sentence-transformers.

Provides:
  - A singleton model loader (loads once, reused)
  - cosine_similarity helper
  - Graceful fallback when sentence-transformers is not installed

Model: all-MiniLM-L6-v2 (fast, accurate, 384-dim embeddings)
"""

from __future__ import annotations

from typing import Any, Optional
from rich.console import Console

console = Console()

# Singleton model instance
_model: Any = None
_model_name: str = "all-MiniLM-L6-v2"


def get_embeddings_model(model_name: str = _model_name):
    """
    Load and return the sentence-transformers model.
    First call downloads/loads the model; subsequent calls return cached instance.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            console.print(f"[dim]Loading embedding model ({model_name})...[/dim]")
            _model = SentenceTransformer(model_name)
            console.print("[green]✓ Embedding model loaded.[/green]")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns list of float vectors.
    Falls back to empty lists if model unavailable.
    """
    try:
        model = get_embeddings_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return [list(e) for e in embeddings]
    except ImportError:
        console.print("[yellow]⚠ sentence-transformers not available. Embeddings skipped.[/yellow]")
        return [[] for _ in texts]
    except Exception as e:
        console.print(f"[red]Embedding error: {e}[/red]")
        return [[] for _ in texts]


def embed_single(text: str) -> list[float]:
    """Embed a single string. Returns float list."""
    result = embed_texts([text])
    return result[0] if result else []


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Pure-Python cosine similarity between two vectors.
    Falls back when torch is not available.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    try:
        import torch
        import torch.nn.functional as F
        a = torch.tensor(vec_a).unsqueeze(0)
        b = torch.tensor(vec_b).unsqueeze(0)
        return float(F.cosine_similarity(a, b).item())
    except ImportError:
        # Pure Python fallback
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(x ** 2 for x in vec_a) ** 0.5
        norm_b = sum(x ** 2 for x in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def batch_similarities(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
) -> list[float]:
    """
    Compute cosine similarity between one query and many candidates.
    Returns list of float scores.
    """
    return [cosine_similarity(query_embedding, c) for c in candidate_embeddings]

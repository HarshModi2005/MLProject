"""
core/vector_store.py

ChromaDB vector store interface.

Provides:
  - Storing professor profile embeddings
  - Semantic search/retrieval by user query
  - Persistence between runs (local ChromaDB)

Used when re-crawling to avoid re-embedding already processed profiles.
"""

from __future__ import annotations

import json
from typing import Optional

from rich.console import Console

import config
from core.embeddings import embed_single, embed_texts

console = Console()

COLLECTION_NAME = "professor_profiles"


class VectorStore:
    """
    ChromaDB-backed vector store for professor profiles.
    Handles persistence, upsert, and semantic search.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or config.CHROMADB_PATH
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazily initialize ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_path)
            except ImportError:
                raise ImportError("chromadb is not installed. Run: pip install chromadb")
        return self._client

    def _get_collection(self):
        """Get or create the professors collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert_professor(self, professor_id: str, text: str, metadata: dict) -> None:
        """
        Embed and store a professor profile in the vector store.
        Upsert: updates if ID already exists.
        """
        try:
            collection = self._get_collection()
            embedding = embed_single(text)
            if not embedding:
                return  # Skip if embedding failed

            collection.upsert(
                ids=[professor_id],
                embeddings=[embedding],
                documents=[text[:500]],  # Store snippet as document
                metadatas=[{
                    k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                    for k, v in metadata.items()
                }],
            )
        except Exception as e:
            console.print(f"[yellow]⚠ VectorStore upsert error: {e}[/yellow]")

    def upsert_batch(self, professors_data: list[dict]) -> None:
        """
        Batch upsert multiple profiles. More efficient than one-by-one.
        professors_data: list of {id, text, metadata} dicts.
        """
        if not professors_data:
            return

        try:
            collection = self._get_collection()
            ids = [d["id"] for d in professors_data]
            texts = [d["text"] for d in professors_data]
            embeddings = embed_texts(texts)

            valid = [
                (id_, emb, text[:500], data["metadata"])
                for id_, emb, text, data in zip(ids, embeddings, texts, professors_data)
                if emb
            ]
            if not valid:
                return

            collection.upsert(
                ids=[v[0] for v in valid],
                embeddings=[v[1] for v in valid],
                documents=[v[2] for v in valid],
                metadatas=[
                    {
                        k: (json.dumps(val) if isinstance(val, (list, dict)) else str(val))
                        for k, val in v[3].items()
                    }
                    for v in valid
                ],
            )
            console.print(f"[green]✓ Stored {len(valid)} embeddings in vector store.[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ VectorStore batch upsert error: {e}[/yellow]")

    def search(self, query: str, n_results: int = 20) -> list[dict]:
        """
        Semantic search: find top-N professors most similar to query.
        Returns list of metadata dicts.
        """
        try:
            collection = self._get_collection()
            query_embedding = embed_single(query)
            if not query_embedding:
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                include=["metadatas", "distances", "documents"],
            )
            return results.get("metadatas", [[]])[0]
        except Exception as e:
            console.print(f"[yellow]⚠ VectorStore search error: {e}[/yellow]")
            return []

    def get_all_ids(self) -> list[str]:
        """Return all stored professor IDs (for duplicate detection)."""
        try:
            collection = self._get_collection()
            result = collection.get(include=[])
            return result.get("ids", [])
        except Exception:
            return []

    def count(self) -> int:
        """Number of stored professor profiles."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def professor_exists(self, professor_id: str) -> bool:
        """Check if a professor ID is already in the store."""
        return professor_id in self.get_all_ids()


# ── Module-level singleton ────────────────────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the global VectorStore singleton."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

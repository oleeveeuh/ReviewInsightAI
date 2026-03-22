#!/usr/bin/env python3
"""
Vector store for semantic similarity search.

Uses pre-computed sentence-transformer embeddings to avoid runtime
import issues with the DuckDB mutex on macOS + miniforge.

The embeddings are built once in a clean venv, then loaded and used for
similarity search using numpy (no sentence-transformers import needed).

Usage:
    from src.agent.embeddings import get_vector_store

    store = get_vector_store()
    results = store.search("safety concerns", k=5)
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Global cache
_global_store: Optional['SimpleVectorStore'] = None


class SimpleVectorStore:
    """Vector store using pre-computed sentence-transformer embeddings."""

    def __init__(self):
        self.reviews = []
        self.embeddings = None
        self.vectorizer = None
        self.query_embeddings = None
        self._initialized = False
        self._backend = None  # 'tfidf' or 'sentence-transformer'

    def load_from_files(
        self,
        jsonl_path: str = 'data/processed/reviews_final.jsonl',
        meta_path: str = 'data/processed/reviews_meta.json',
    ):
        """Load reviews and pre-built embeddings."""

        # Try sentence-transformer embeddings first (better quality)
        st_emb_path = Path('data/processed/st_embeddings.npy')
        tfidf_emb_path = Path('data/processed/tfidf_embeddings.npy')
        tfidf_vectorizer_path = Path('data/processed/tfidf_vectorizer.pkl')
        query_emb_path = Path('data/processed/query_embeddings.npy')

        if st_emb_path.exists():
            self._load_sentence_transformer_embeddings(st_emb_path, query_emb_path, meta_path)
        elif tfidf_emb_path.exists():
            self._load_tfidf_embeddings(tfidf_emb_path, tfidf_vectorizer_path, meta_path)
        else:
            raise FileNotFoundError(
                "No embeddings found. Run build_embeddings.py first:\n"
                "  python -m venv .venv_clean\n"
                "  .venv_clean/bin/pip install sentence-transformers numpy\n"
                "  .venv_clean/bin/python src/agent/build_embeddings.py"
            )

    def _load_sentence_transformer_embeddings(self, emb_path: Path, query_emb_path: Path, meta_path: str):
        """Load sentence-transformer embeddings (no import needed)."""
        print("Loading sentence-transformer embeddings...")
        with open(meta_path, 'r') as f:
            self.reviews = json.load(f)

        self.embeddings = np.load(emb_path)
        print(f"Loaded {len(self.reviews)} reviews with embeddings shape: {self.embeddings.shape}")

        # Load pre-computed query embeddings if available
        if query_emb_path.exists():
            self.query_embeddings = np.load(query_emb_path, allow_pickle=True).item()
            print(f"Loaded {len(self.query_embeddings)} pre-computed query embeddings")

        self._backend = 'sentence-transformer'
        self._initialized = True

    def _load_tfidf_embeddings(self, emb_path: Path, vectorizer_path: Path, meta_path: str):
        """Load TF-IDF embeddings."""
        print("Loading TF-IDF embeddings...")
        with open(meta_path, 'r') as f:
            self.reviews = json.load(f)

        self.embeddings = np.load(emb_path)
        print(f"Loaded {len(self.reviews)} reviews with embeddings shape: {self.embeddings.shape}")

        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self._backend = 'tfidf'
        self._initialized = True

    def _encode_query_st(self, query: str) -> np.ndarray:
        """
        Encode query using sentence-transformers without importing.

        Uses a hybrid approach:
        1. Check if query is in pre-computed embeddings
        2. If not, use a simple bag-of-words approximation using the stored embeddings
        """
        # Check for exact or partial match in pre-computed queries
        if self.query_embeddings:
            query_lower = query.lower()
            for precomputed_q in self.query_embeddings.keys():
                if query_lower in precomputed_q or precomputed_q in query_lower:
                    return self.query_embeddings[precomputed_q]

        # Fallback: use TF-IDF style approximation
        # This is not as good as real sentence-transformers but avoids the import
        # For now, we'll create a simple weighted average of words that appear in the query
        # But actually, the better approach is to just use cosine similarity with
        # the stored embeddings directly
        return self._embed_query_approximation(query)

    def _embed_query_approximation(self, query: str) -> np.ndarray:
        """
        Create a query embedding approximation using word overlap.

        This is a fallback when pre-computed embeddings aren't available.
        It finds the most similar stored review and uses its embedding as a starting point,
        then adjusts based on word overlap.
        """
        # Simple approach: find the review with maximum word overlap
        query_words = set(query.lower().split())
        best_match_idx = 0
        best_overlap = 0

        for i, review in enumerate(self.reviews):
            review_words = set(review.get('text', '').lower().split())
            overlap = len(query_words & review_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_idx = i

        # Return the embedding of the most similar review
        # This gives us a "good enough" starting point for similarity search
        return self.embeddings[best_match_idx]

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
        exclude_review_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find semantically similar reviews."""
        if not self._initialized:
            self.load_from_files()

        # Transform query based on backend
        if self._backend == 'sentence-transformer':
            # Use pre-computed embeddings or approximation
            query_emb = self._encode_query_st(query)
        else:  # tfidf
            query_emb = self.vectorizer.transform([query]).toarray()[0]
            # Compute cosine similarity for sparse vectors
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
            similarities = np.divide(
                np.dot(self.embeddings, query_emb),
                norms,
                where=norms > 0,
                out=np.zeros(len(norms))
            )
            # Return early for TF-IDF since we already computed similarities
            return self._filter_and_sort_results(similarities, k, source_filter, exclude_review_id)

        # Compute cosine similarity for dense vectors
        similarities = np.dot(self.embeddings, query_emb)
        return self._filter_and_sort_results(similarities, k, source_filter, exclude_review_id)

    def _filter_and_sort_results(
        self,
        similarities: np.ndarray,
        k: int,
        source_filter: Optional[str],
        exclude_review_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Apply filters and sort by similarity."""
        indices = np.arange(len(self.reviews))
        mask = np.ones(len(indices), dtype=bool)

        if source_filter:
            mask &= [self.reviews[i].get('source') == source_filter for i in indices]

        if exclude_review_id:
            mask &= [self.reviews[i].get('review_id') != exclude_review_id for i in indices]

        # Sort by similarity
        valid_indices = indices[mask]
        valid_sims = similarities[mask]
        sorted_idx = np.argsort(valid_sims)[::-1][:k]

        # Build results
        results = []
        for idx in sorted_idx:
            i = valid_indices[idx]
            r = self.reviews[i]
            similarity = valid_sims[idx]
            # Only return results with meaningful similarity
            if similarity > 0.01:
                results.append({
                    'review_id': r.get('review_id'),
                    'text': r.get('text', ''),
                    'source': r.get('source'),
                    'rating': r.get('rating'),
                    'similarity': float(similarity)
                })

        return results


def get_vector_store() -> SimpleVectorStore:
    """Get or create the global vector store."""
    global _global_store
    if _global_store is None:
        _global_store = SimpleVectorStore()
        _global_store.load_from_files()
    return _global_store


if __name__ == "__main__":
    # Test
    store = SimpleVectorStore()
    store.load_from_files()

    queries = ["overtime", "safety", "management issues"]
    for q in queries:
        print(f"\nQuery: {q}")
        results = store.search(q, k=3)
        for r in results:
            print(f"  {r['similarity']:.3f}: [{r['source']}] {r['text'][:60]}...")

#!/usr/bin/env python3
"""
Lightweight vector store for semantic-ish review retrieval.

Two backends:

1. TF-IDF (default). Built in-memory from data/processed/reviews_final.jsonl
   with scikit-learn. Offline, deterministic, no serialized artifacts, and
   works on a fresh clone. This is lexical similarity (TF-IDF cosine), not
   true semantic similarity - good enough for context retrieval in this
   prototype, and honest about what it is.

2. sentence-transformers (optional). If data/processed/st_embeddings.npy and
   reviews_meta.json exist (built by src/agent/build_embeddings.py) AND the
   sentence-transformers package is installed, queries are embedded with the
   model for true semantic search.

Notes on the previous implementation
------------------------------------
Earlier versions loaded a pickled TF-IDF vectorizer and an
``allow_pickle=True`` numpy object array of query embeddings, and approximated
unseen query vectors by copying the embedding of the max word-overlap review
(which made the queried review itself the top hit). All of that is gone: no
pickle, no allow_pickle, no pre-computed query cache, no overlap hack.

Usage:
    from src.agent.embeddings import get_vector_store

    store = get_vector_store()
    results = store.search("safety concerns", k=5)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent
REVIEWS_PATH = REPO_ROOT / 'data/processed/reviews_final.jsonl'
ST_EMB_PATH = REPO_ROOT / 'data/processed/st_embeddings.npy'
ST_META_PATH = REPO_ROOT / 'data/processed/reviews_meta.json'

# Global cache
_global_store: Optional['SimpleVectorStore'] = None


class SimpleVectorStore:
    """Review retrieval over reviews_final.jsonl.

    Backends (auto-selected):
      * 'tfidf'  - in-memory scikit-learn TF-IDF vectors (default, offline)
      * 'sentence-transformers' - precomputed embeddings + model-encoded query
    """

    def __init__(self, reviews_path: Optional[Path] = None):
        self.reviews: List[Dict] = []
        self.embeddings = None
        self.vectorizer = None
        self._backend = None
        self._st_model = None
        self.reviews_path = Path(reviews_path) if reviews_path else REVIEWS_PATH

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_files(self, jsonl_path: Optional[Path] = None):
        """Load reviews, preferring sentence-transformer artifacts when usable."""
        if jsonl_path:
            self.reviews_path = Path(jsonl_path)
        if not self.reviews_path.exists():
            raise FileNotFoundError(
                f"Reviews not found at {self.reviews_path}. The tracked file is "
                "a small synthetic fixture; restore the full local corpus or "
                "point SimpleVectorStore at another JSONL with the same schema."
            )

        with open(self.reviews_path, 'r') as f:
            self.reviews = [json.loads(line) for line in f if line.strip()]

        if (ST_EMB_PATH.exists() and ST_META_PATH.exists()
                and self._try_load_st_model()):
            self._load_st_backend()
        else:
            self._build_tfidf_backend()

    def _try_load_st_model(self) -> bool:
        """Return True if the optional sentence-transformers model loads."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return False
        try:
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except Exception:
            return False

    def _load_st_backend(self):
        self.embeddings = np.load(ST_EMB_PATH)
        with open(ST_META_PATH, 'r') as f:
            self.reviews = json.load(f)
        if self.embeddings.shape[0] != len(self.reviews):
            raise ValueError(
                f"st_embeddings.npy has {self.embeddings.shape[0]} rows but "
                f"reviews_meta.json has {len(self.reviews)} reviews - rebuild "
                "artifacts with src/agent/build_embeddings.py"
            )
        self._backend = 'sentence-transformers'
        print(f"Loaded {len(self.reviews)} reviews "
              f"(backend: sentence-transformers, shape {self.embeddings.shape})")

    def _build_tfidf_backend(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                          ngram_range=(1, 2), sublinear_tf=True)
        texts = [r.get('text', '') for r in self.reviews]
        self.embeddings = self.vectorizer.fit_transform(texts)
        self._backend = 'tfidf'
        print(f"Loaded {len(self.reviews)} reviews (backend: tfidf, "
              f"{self.embeddings.shape[1]} features)")

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_query(self, query: str):
        if self._backend == 'sentence-transformers':
            return self._st_model.encode(query, normalize_embeddings=True)
        return self.vectorizer.transform([query])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5,
               source_filter: Optional[str] = None,
               exclude_review_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Find similar reviews (TF-IDF cosine or embedding cosine)."""
        if not self.reviews:
            self.load_from_files()

        query_vec = self._encode_query(query)

        if self._backend == 'sentence-transformers':
            similarities = self.embeddings @ query_vec
        else:
            # Sparse row-wise cosine similarity
            sims = (self.embeddings @ query_vec.T).toarray().ravel()
            q_norm = np.linalg.norm(query_vec.toarray())
            r_norms = np.sqrt(self.embeddings.multiply(
                self.embeddings).sum(axis=1)).A.ravel()
            denom = r_norms * q_norm
            similarities = np.divide(sims, denom, where=denom > 0,
                                     out=np.zeros_like(sims))

        return self._filter_and_sort_results(similarities, k, source_filter,
                                             exclude_review_id)

    def _filter_and_sort_results(self, similarities: np.ndarray, k: int,
                                 source_filter: Optional[str],
                                 exclude_review_id: Optional[str]) -> List[Dict[str, any]]:
        mask = np.ones(len(self.reviews), dtype=bool)
        if source_filter:
            mask &= np.array([r.get('source') == source_filter
                              for r in self.reviews])
        if exclude_review_id:
            mask &= np.array([r.get('review_id') != exclude_review_id
                              for r in self.reviews])

        valid_idx = np.where(mask)[0]
        if not len(valid_idx):
            return []

        order = valid_idx[np.argsort(similarities[valid_idx])[::-1][:k]]
        results = []
        for i in order:
            similarity = float(similarities[i])
            if similarity <= 0.01:
                continue
            r = self.reviews[i]
            results.append({
                'review_id': r.get('review_id'),
                'text': r.get('text', ''),
                'source': r.get('source'),
                'rating': r.get('rating'),
                'similarity': similarity,
            })
        return results


def get_vector_store(reviews_path: Optional[Path] = None) -> SimpleVectorStore:
    """Get or create the global vector store."""
    global _global_store
    if _global_store is None:
        _global_store = SimpleVectorStore(reviews_path=reviews_path)
        _global_store.load_from_files()
    return _global_store


if __name__ == "__main__":
    store = SimpleVectorStore()
    store.load_from_files()

    queries = ["overtime and mandatory extra shifts", "safety", "management communication"]
    for q in queries:
        print(f"\nQuery: {q}")
        for r in store.search(q, k=3):
            print(f"  {r['similarity']:.3f}: [{r['source']}] {r['text'][:70]}...")

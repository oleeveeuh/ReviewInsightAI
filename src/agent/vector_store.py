#!/usr/bin/env python3
"""
Vector store for semantic similarity search.

Uses sentence-transformers to embed reviews and DuckDB for vector storage.
Enables finding semantically similar reviews for context in LLM analysis.

Usage:
    from src.agent.vector_store import VectorStore

    store = VectorStore()
    store.build_index()  # Build embeddings from database
    results = store.search("management issues with overtime", k=5)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore:
    """Vector database for semantic review search using sentence-transformers."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality (384 dims)
    EMBEDDING_DIM = 384

    def __init__(self, db_path: str = "data/database/reviews.duckdb"):
        """
        Initialize vector store.

        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = Path(db_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            print(f"  Loaded embedding model: {self.EMBEDDING_MODEL}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embedding vectors.

        Args:
            texts: List of input texts

        Returns:
            Matrix of embedding vectors (n x dim)
        """
        self._load_model()
        return self.model.encode(texts, normalize_embeddings=True)

    def get_all_reviews(self) -> List[Dict[str, Any]]:
        """Load all reviews from database."""
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        result = conn.execute("""
            SELECT review_id, text, source, rating
            FROM reviews
            ORDER BY review_id
        """).fetchall()
        conn.close()

        return [
            {"review_id": row[0], "text": row[1], "source": row[2], "rating": row[3]}
            for row in result
        ]

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build embeddings for all reviews and store in database.

        Args:
            force_rebuild: Rebuild even if embeddings exist

        Returns:
            Number of reviews embedded
        """
        import duckdb

        # Check if embeddings already exist
        if not force_rebuild:
            conn = duckdb.connect(str(self.db_path))
            count = conn.execute("""
                SELECT COUNT(*) FROM reviews
                WHERE embedding IS NOT NULL
            """).fetchone()[0]
            conn.close()

            if count > 0:
                print(f"  Embeddings already exist for {count} reviews")
                return count

        # Load all reviews
        reviews = self.get_all_reviews()
        if not reviews:
            print("  No reviews found in database")
            return 0

        print(f"  Building embeddings for {len(reviews)} reviews...")
        texts = [r["text"] for r in reviews]

        # Batch encode
        embeddings = self.embed_batch(texts)

        # Store back to database
        conn = duckdb.connect(str(self.db_path))

        # Add embedding column if not exists
        conn.execute("""
            ALTER TABLE reviews
            ADD COLUMN IF NOT EXISTS embedding FLOAT[384]
        """)

        # Update each review with its embedding
        for i, (review, emb) in enumerate(zip(reviews, embeddings)):
            conn.execute("""
                UPDATE reviews
                SET embedding = ?
                WHERE review_id = ?
            """, [emb.tolist(), review["review_id"]])

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{len(reviews)}")

        # Create vector index for faster search
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS reviews_embedding_idx ON reviews USING hnsw(embedding)")
            print("  Created HNSW vector index")
        except Exception as e:
            print(f"  Note: Could not create vector index (DuckDB extension may not be loaded): {e}")

        conn.close()

        print(f"  ✅ Built embeddings for {len(reviews)} reviews")
        return len(reviews)

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
        exclude_review_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar reviews.

        Args:
            query: Search query text
            k: Number of results to return
            source_filter: Optional source filter (e.g., "reddit", "glassdoor")
            exclude_review_id: Exclude this review from results (useful during analysis)

        Returns:
            List of similar reviews with similarity scores
        """
        import duckdb

        # Embed query
        query_emb = self.embed_text(query)

        conn = duckdb.connect(str(self.db_path))

        # Build SQL query
        sql = """
            SELECT
                review_id, text, source, rating,
                array_cosine_similarity(embedding, ?) as similarity
            FROM reviews
            WHERE embedding IS NOT NULL
        """

        params = [query_emb.tolist()]

        # Add filters
        if source_filter:
            sql += " AND source = ?"
            params.append(source_filter)

        if exclude_review_id:
            sql += " AND review_id != ?"
            params.append(exclude_review_id)

        # Order by similarity and limit
        sql += f" ORDER BY similarity DESC LIMIT {k}"

        result = conn.execute(sql, params).fetchall()
        conn.close()

        return [
            {
                "review_id": row[0],
                "text": row[1][:500],  # Truncate for context
                "source": row[2],
                "rating": row[3],
                "similarity": float(row[4])
            }
            for row in result
        ]

    def get_similar_by_review_id(
        self,
        review_id: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find reviews similar to a specific review (by ID).

        Args:
            review_id: ID of the reference review
            k: Number of results to return

        Returns:
            List of similar reviews
        """
        import duckdb

        conn = duckdb.connect(str(self.db_path))

        # Get the reference review's embedding
        emb_result = conn.execute("""
            SELECT embedding
            FROM reviews
            WHERE review_id = ? AND embedding IS NOT NULL
        """, [review_id]).fetchone()

        if not emb_result:
            conn.close()
            return []

        query_emb = emb_result[0]

        # Find similar reviews
        result = conn.execute("""
            SELECT
                r.review_id, r.text, r.source, r.rating,
                array_cosine_similarity(r.embedding, CAST(? AS FLOAT[384])) as similarity
            FROM reviews r
            WHERE r.embedding IS NOT NULL AND r.review_id != ?
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_emb, review_id, k]).fetchall()

        conn.close()

        return [
            {
                "review_id": row[0],
                "text": row[1][:500],
                "source": row[2],
                "rating": row[3],
                "similarity": float(row[4])
            }
            for row in result
        ]


# Convenience function for quick search
def find_similar_reviews(
    query: str,
    k: int = 5,
    db_path: str = "data/database/reviews.duckdb"
) -> List[Dict[str, Any]]:
    """
    Quick search for similar reviews.

    Args:
        query: Search query
        k: Number of results
        db_path: Path to database

    Returns:
        List of similar reviews
    """
    store = VectorStore(db_path)
    return store.search(query, k=k)


# CLI for testing
if __name__ == "__main__":
    import sys

    store = VectorStore()

    # Build index if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        store.build_index(force_rebuild=True)
        sys.exit(0)

    # Ensure embeddings exist
    reviews = store.get_all_reviews()
    conn = __import__("duckdb").connect(str(store.db_path))
    has_embeddings = conn.execute("""
        SELECT COUNT(*) FROM reviews WHERE embedding IS NOT NULL
    """).fetchone()[0] > 0
    conn.close()

    if not has_embeddings:
        print("No embeddings found. Building...")
        store.build_index()

    # Interactive search
    print("\n" + "=" * 60)
    print("SEMANTIC REVIEW SEARCH")
    print("=" * 60)
    print("Enter queries to find similar reviews (Ctrl+C to exit)\n")

    try:
        while True:
            query = input("Search query: ").strip()
            if not query:
                continue

            results = store.search(query, k=5)

            print(f"\nFound {len(results)} similar reviews:")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [{r['source']}] {r['review_id']}")
                print(f"   Similarity: {r['similarity']:.3f}")
                print(f"   Text: {r['text'][:200]}...")

    except KeyboardInterrupt:
        print("\n\nExiting.")

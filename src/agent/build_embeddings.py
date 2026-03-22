#!/usr/bin/env python3
"""
Build sentence-transformer embeddings for all reviews.

IMPORTANT: Run this script in a terminal (NOT from VSCode) to avoid mutex conflicts:
    bash run_from_terminal.sh

Or disable VSCode Python extensions temporarily while running.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    # Paths
    jsonl_path = Path(__file__).parent.parent.parent / 'data/processed/reviews_final.jsonl'
    emb_path = Path(__file__).parent.parent.parent / 'data/processed/st_embeddings.npy'
    meta_path = Path(__file__).parent.parent.parent / 'data/processed/reviews_meta.json'

    print("=" * 60)
    print("BUILDING SENTENCE-TRANSFORMER EMBEDDINGS")
    print("=" * 60)

    # Load reviews
    print("\nLoading reviews...")
    with open(jsonl_path, 'r') as f:
        reviews = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(reviews)} reviews")

    # Load model
    print("\nLoading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model: {model}")

    # Create embeddings
    print("\nCreating embeddings...")
    texts = [r.get('text', '') for r in reviews]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"Created embeddings shape: {embeddings.shape}")

    # Save to disk
    print(f"\nSaving embeddings to {emb_path}")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)

    # Save metadata
    print(f"Saving metadata to {meta_path}")
    metadata = [
        {'review_id': r.get('review_id'), 'text': r.get('text', '')[:500], 'source': r.get('source'), 'rating': r.get('rating')}
        for r in reviews
    ]
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    # Test search
    print("\n" + "=" * 60)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 60)

    test_queries = [
        "safety concerns at the warehouse",
        "overtime and burnout",
        "management issues",
        "low pay and benefits"
    ]

    for query in test_queries:
        query_emb = model.encode(query, normalize_embeddings=True)
        similarities = np.dot(embeddings, query_emb)
        top_k = np.argsort(similarities)[::-1][:2]

        print(f"\nQuery: '{query}'")
        for idx in top_k:
            print(f"  {similarities[idx]:.3f}: [{metadata[idx]['source']}] {metadata[idx]['text'][:70]}...")

    print("\n" + "=" * 60)
    print("Done! Run from your terminal to avoid VSCode conflicts.")
    print("=" * 60)


if __name__ == "__main__":
    main()

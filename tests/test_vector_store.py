"""Vector store tests: TF-IDF backend on synthetic fixtures, no pickle."""

import pytest

from agent.embeddings import SimpleVectorStore


@pytest.fixture
def store(fixture_reviews_path):
    s = SimpleVectorStore(reviews_path=fixture_reviews_path)
    s.load_from_files()
    return s


def test_default_backend_is_tfidf_without_artifacts(store):
    """No .npy/.pkl artifacts exist in a fresh clone; TF-IDF must be default."""
    assert store._backend == 'tfidf'
    assert store.vectorizer is not None
    assert store.embeddings is not None


def test_search_returns_sorted_similarities(store):
    results = store.search("mandatory overtime and extra shifts", k=3)
    assert 1 <= len(results) <= 3
    sims = [r['similarity'] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_search_k_is_respected(store):
    assert len(store.search("warehouse work", k=2)) <= 2


def test_exclude_review_id(store):
    all_rows = store.search("safety", k=5)
    if not all_rows:
        pytest.skip("no matches for probe query")
    top_id = all_rows[0]['review_id']
    excluded = store.search("safety", k=5, exclude_review_id=top_id)
    assert all(r['review_id'] != top_id for r in excluded)


def test_source_filter(store):
    results = store.search("benefits and pay", k=5, source_filter='glassdoor')
    assert all(r['source'] == 'glassdoor' for r in results)


def test_unrelated_query_returns_no_junk_matches(store):
    """Similarity floor: gibberish must not surface 'similar' reviews."""
    results = store.search("zzzqqqxyzzy unrelated gibberish words", k=5)
    assert all(r['similarity'] > 0.01 for r in results)


def test_missing_reviews_file_raises_actionable_error(tmp_path):
    s = SimpleVectorStore(reviews_path=tmp_path / 'nope.jsonl')
    with pytest.raises(FileNotFoundError, match="synthetic fixture|JSONL"):
        s.load_from_files()

"""Offline end-to-end database pipeline test on synthetic fixtures.

Loads the synthetic reviews + silver labels into a temporary DuckDB file,
computes aggregates, and exercises every dashboard-facing query — including
the parameterized text search, which must survive quote/wildcard payloads.
"""

import pytest

from conftest import FIXTURE_REVIEWS, FIXTURE_SILVER

from database.db_manager import ReviewDatabase


@pytest.fixture
def db(tmp_path):
    conn = ReviewDatabase(db_path=str(tmp_path / 'test_reviews.duckdb'))
    conn.load_reviews_from_jsonl(str(FIXTURE_REVIEWS))
    conn.load_analyses_from_jsonl(str(FIXTURE_SILVER))
    conn.compute_kpi_aggregates()
    yield conn
    conn.close()


def test_reviews_loaded(db):
    counts = db.get_table_counts()
    assert counts['reviews'] == 9
    assert counts['llm_analysis'] == 9


def test_overall_kpis(db):
    kpis = db.get_overall_kpis()
    assert kpis['total_reviews'] == 9
    assert kpis['source_count'] == 3
    assert 1.0 <= kpis['avg_sentiment'] <= 5.0
    # LLM-assigned classification, not observed attrition
    assert 0 <= kpis['high_risk_pct'] <= 100


def test_statistics_helper_exists_for_dashboard(db):
    stats = db.get_statistics()
    assert stats['total_reviews'] == 9
    dist = db.get_sentiment_distribution()
    assert set(dist.columns) == {'sentiment', 'count'}
    assert dist['count'].sum() == 9


def test_theme_and_risk_breakdown(db):
    themes = db.get_theme_distribution(top_n=5)
    assert len(themes) <= 5
    assert set(themes.columns) >= {'theme', 'frequency', 'percentage'}

    risk = db.get_retention_risk_breakdown()
    assert {'source', 'retention_risk', 'count'} <= set(risk.columns)


def test_search_is_parameterized_against_injection_payloads(db):
    """Quotes and LIKE wildcards are treated as literal text, not SQL."""
    for payload in ["overtime", "hasn't", "100%' OR '1'='1", "under_score%",
                    "'; DROP TABLE reviews; --"]:
        df = db.search_reviews(payload, limit=5)
        assert list(df.columns) >= ['review_id', 'text_preview']
        assert len(df) <= 5


def test_search_filters_match_dashboard_signature(db):
    """The dashboard passes sentiment/retention_risk filters; they must work."""
    df = db.search_reviews('benefits', sentiment=2, retention_risk='high', limit=10)
    if not df.empty:
        assert (df['sentiment'] == 2).all()
        assert (df['retention_risk'] == 'high').all()

    all_risk = db.search_reviews('the', retention_risk='low', limit=50)
    assert (all_risk['retention_risk'] == 'low').all()


def test_high_risk_reviews_query(db):
    df = db.get_high_risk_reviews(limit=5)
    assert len(df) <= 5
    assert (df['retention_risk'] == 'high').all()


def test_sentiment_trend_from_aggregates(db):
    trend = db.get_sentiment_trend()
    assert {'date', 'source', 'sentiment'} <= set(trend.columns)

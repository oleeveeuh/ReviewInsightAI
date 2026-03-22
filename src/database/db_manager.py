#!/usr/bin/env python3
"""
DuckDB database management layer for ReviewInsight AI.

Provides fast SQL analytics on review data with precomputed KPI aggregates.

Tables:
- reviews: Raw review data
- llm_analysis: Structured LLM outputs (sentiment, themes, risk)
- kpi_aggregates: Precomputed metrics for dashboard

Usage:
    from src.database.db_manager import ReviewDatabase

    db = ReviewDatabase()
    db.load_reviews_from_jsonl('data/processed/reviews_final.jsonl')
    db.load_analyses_from_jsonl('data/labeled/silver_standard.jsonl')
    db.compute_kpi_aggregates()

    # Query for dashboard
    kpis = db.get_overall_kpis()
    sentiment_trend = db.get_sentiment_trend()
"""

import duckdb
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ReviewDatabase:
    """DuckDB database for review analytics"""

    def __init__(self, db_path='data/database/reviews.duckdb'):
        """
        Initialize database connection

        Args:
            db_path: Path to DuckDB file (created if doesn't exist)
        """
        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to DuckDB (creates file if doesn't exist)
        self.conn = duckdb.connect(str(db_path))

        # Create tables
        self.setup_schema()

        print(f"✅ Database initialized: {db_path}")

    def setup_schema(self):
        """Create tables if they don't exist"""

        # Table 1: Raw reviews
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id VARCHAR PRIMARY KEY,
                text VARCHAR NOT NULL,
                rating DOUBLE,
                date DATE,
                year INTEGER,
                quarter VARCHAR,
                location VARCHAR,
                job_title VARCHAR,
                source VARCHAR,
                review_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: LLM analysis results
        # Note: review_id is the natural key - analysis_id is just surrogate
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_analysis (
                review_id VARCHAR PRIMARY KEY,
                analysis_id INTEGER,
                sentiment INTEGER,
                themes VARCHAR[],
                retention_risk VARCHAR,
                confidence DOUBLE,
                reasoning VARCHAR,
                is_anomalous BOOLEAN,
                anomaly_reason VARCHAR,
                model_version VARCHAR,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create sequence for analysis_id
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analysis_id_seq START 1
        """)

        # Table 3: KPI aggregates (precomputed for dashboard)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kpi_aggregates (
                agg_id INTEGER PRIMARY KEY,
                agg_date DATE,
                agg_period VARCHAR,
                source VARCHAR,
                review_count INTEGER,
                avg_sentiment DOUBLE,
                avg_confidence DOUBLE,
                low_risk_count INTEGER,
                medium_risk_count INTEGER,
                high_risk_count INTEGER,
                anomaly_count INTEGER,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agg_date, agg_period, source)
            )
        """)

        # Create sequence for agg_id
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS agg_id_seq START 1
        """)

        print("  Tables created: reviews, llm_analysis, kpi_aggregates")

    def load_reviews_from_jsonl(self, jsonl_path='data/processed/reviews_final.jsonl'):
        """
        Bulk load reviews from JSONL file

        Args:
            jsonl_path: Path to reviews JSONL file
        """
        if not Path(jsonl_path).exists():
            print(f"⚠️  File not found: {jsonl_path}")
            return 0

        # Load JSONL
        reviews = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))

        if not reviews:
            print(f"⚠️  No reviews found in {jsonl_path}")
            return 0

        # Insert into database
        inserted = 0
        for r in reviews:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO reviews
                    (review_id, text, rating, date, year, quarter,
                     location, job_title, source, review_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get('review_id'),
                    r.get('text', ''),
                    r.get('rating'),
                    r.get('date'),
                    r.get('year'),
                    r.get('quarter'),
                    r.get('location'),
                    r.get('job_title'),
                    r.get('source', 'unknown'),
                    len(r.get('text', ''))
                ))
                inserted += 1
            except Exception as e:
                print(f"  Error inserting {r.get('review_id', 'unknown')}: {e}")

        print(f"  Loaded {inserted} reviews into database")
        return inserted

    def load_analyses_from_jsonl(self, jsonl_path='data/labeled/silver_standard.jsonl'):
        """
        Load LLM analysis results from JSONL

        Args:
            jsonl_path: Path to silver standard labels
        """
        if not Path(jsonl_path).exists():
            print(f"⚠️  File not found: {jsonl_path}")
            return 0

        # Load JSONL
        analyses = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    analyses.append(json.loads(line))

        if not analyses:
            print(f"⚠️  No analyses found in {jsonl_path}")
            return 0

        # Insert into database
        inserted = 0
        for a in analyses:
            try:
                labels = a.get('auto_labels', {})

                self.conn.execute("""
                    INSERT OR REPLACE INTO llm_analysis
                    (review_id, analysis_id, sentiment, themes, retention_risk,
                     confidence, reasoning, is_anomalous, model_version)
                    VALUES (?, nextval('analysis_id_seq'), ?, ?, ?, ?, ?, ?, ?)
                """, (
                    a.get('review_id'),
                    labels.get('sentiment'),
                    labels.get('themes', []),
                    labels.get('retention_risk'),
                    labels.get('confidence', 0.8),
                    labels.get('reasoning', ''),
                    False,  # Will update with anomaly detection later
                    'gpt-4o'
                ))
                inserted += 1
            except Exception as e:
                print(f"  Error inserting analysis for {a.get('review_id', 'unknown')}: {e}")

        print(f"  Loaded {inserted} analyses into database")
        return inserted

    def load_memory_analyses(self, jsonl_path='data/memory/analysis_log.jsonl'):
        """
        Load analyses from agent memory log

        Args:
            jsonl_path: Path to agent memory log
        """
        if not Path(jsonl_path).exists():
            print(f"⚠️  File not found: {jsonl_path}")
            return 0

        # Load JSONL
        analyses = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    analyses.append(json.loads(line))

        if not analyses:
            print(f"⚠️  No analyses found in {jsonl_path}")
            return 0

        # Insert into database
        inserted = 0
        for a in analyses:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO llm_analysis
                    (review_id, analysis_id, sentiment, themes, retention_risk,
                     is_anomalous, model_version)
                    VALUES (?, nextval('analysis_id_seq'), ?, ?, ?, ?, ?)
                """, (
                    a.get('review_id'),
                    a.get('sentiment'),
                    a.get('themes', []),
                    a.get('retention_risk'),
                    a.get('is_anomalous', False),
                    'agent_memory'
                ))
                inserted += 1
            except Exception as e:
                print(f"  Error inserting memory analysis for {a.get('review_id', 'unknown')}: {e}")

        print(f"  Loaded {inserted} memory analyses into database")
        return inserted

    def compute_kpi_aggregates(self):
        """
        Compute and store KPI aggregates for fast dashboard queries

        Aggregates by:
        - Date + Source (daily metrics)
        - Source only (overall metrics)
        """

        # Clear existing aggregates
        self.conn.execute("DELETE FROM kpi_aggregates")

        # Daily aggregates by source
        self.conn.execute("""
            INSERT INTO kpi_aggregates
            (agg_id, agg_date, agg_period, source, review_count,
             avg_sentiment, avg_confidence,
             low_risk_count, medium_risk_count, high_risk_count, anomaly_count)
            SELECT
                nextval('agg_id_seq'),
                r.date as agg_date,
                'daily' as agg_period,
                r.source,
                COUNT(*) as review_count,
                AVG(a.sentiment) as avg_sentiment,
                AVG(a.confidence) as avg_confidence,
                SUM(CASE WHEN a.retention_risk = 'low' THEN 1 ELSE 0 END) as low_risk_count,
                SUM(CASE WHEN a.retention_risk = 'medium' THEN 1 ELSE 0 END) as medium_risk_count,
                SUM(CASE WHEN a.retention_risk = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN a.is_anomalous THEN 1 ELSE 0 END) as anomaly_count
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
            WHERE r.date IS NOT NULL
            GROUP BY r.date, r.source
        """)

        # Overall aggregates by source
        self.conn.execute("""
            INSERT INTO kpi_aggregates
            (agg_id, agg_date, agg_period, source, review_count,
             avg_sentiment, avg_confidence,
             low_risk_count, medium_risk_count, high_risk_count, anomaly_count)
            SELECT
                nextval('agg_id_seq'),
                NULL as agg_date,
                'overall' as agg_period,
                r.source,
                COUNT(*) as review_count,
                AVG(a.sentiment) as avg_sentiment,
                AVG(a.confidence) as avg_confidence,
                SUM(CASE WHEN a.retention_risk = 'low' THEN 1 ELSE 0 END) as low_risk_count,
                SUM(CASE WHEN a.retention_risk = 'medium' THEN 1 ELSE 0 END) as medium_risk_count,
                SUM(CASE WHEN a.retention_risk = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN a.is_anomalous THEN 1 ELSE 0 END) as anomaly_count
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
            GROUP BY r.source
        """)

        agg_count = self.conn.execute("SELECT COUNT(*) FROM kpi_aggregates").fetchone()[0]
        print(f"  Computed {agg_count} KPI aggregates")

    # === QUERY METHODS FOR DASHBOARD ===

    def get_overall_kpis(self) -> Dict[str, Any]:
        """Get high-level KPIs for dashboard header"""

        result = self.conn.execute("""
            SELECT
                COUNT(DISTINCT r.review_id) as total_reviews,
                COUNT(DISTINCT r.source) as source_count,
                AVG(a.sentiment) as avg_sentiment,
                100.0 * SUM(CASE WHEN a.retention_risk = 'high' THEN 1 ELSE 0 END) / COUNT(*) as high_risk_pct,
                100.0 * SUM(CASE WHEN a.is_anomalous THEN 1 ELSE 0 END) / COUNT(*) as anomaly_rate
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
        """).fetchone()

        return {
            'total_reviews': result[0] or 0,
            'source_count': result[1] or 0,
            'avg_sentiment': round(result[2], 2) if result[2] is not None else 0.0,
            'high_risk_pct': round(result[3], 1) if result[3] is not None else 0.0,
            'anomaly_rate': round(result[4], 1) if result[4] is not None else 0.0
        }

    def get_sentiment_trend(self) -> pd.DataFrame:
        """Get sentiment trend over time (returns pandas DataFrame)"""

        return self.conn.execute("""
            SELECT
                agg_date as date,
                source,
                avg_sentiment as sentiment
            FROM kpi_aggregates
            WHERE agg_period = 'daily'
            ORDER BY agg_date, source
        """).df()

    def get_theme_distribution(self, top_n: int = 10) -> pd.DataFrame:
        """Get theme frequency distribution"""

        # DuckDB uses UNNEST differently - need to use from unnest()
        return self.conn.execute(f"""
            SELECT
                unnested.theme as theme,
                COUNT(*) as frequency,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER () as percentage
            FROM llm_analysis,
                 UNNEST(themes) as unnested(theme)
            GROUP BY unnested.theme
            ORDER BY frequency DESC
            LIMIT {top_n}
        """).df()

    def get_retention_risk_breakdown(self) -> pd.DataFrame:
        """Get retention risk distribution by source"""

        return self.conn.execute("""
            SELECT
                r.source,
                a.retention_risk,
                COUNT(*) as count
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
            GROUP BY r.source, a.retention_risk
            ORDER BY r.source, a.retention_risk
        """).df()

    def get_high_risk_reviews(self, limit: int = 10) -> pd.DataFrame:
        """Get most recent high-risk reviews for investigation"""

        return self.conn.execute(f"""
            SELECT
                r.review_id,
                SUBSTR(r.text, 1, 200) || '...' as text_preview,
                r.date,
                r.source,
                a.sentiment,
                a.themes,
                a.reasoning,
                a.retention_risk
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
            WHERE a.retention_risk = 'high'
            ORDER BY r.date DESC NULLS LAST
            LIMIT {limit}
        """).df()

    def search_reviews(self, search_term: str, limit: int = 20) -> pd.DataFrame:
        """Simple text search in reviews"""

        # Escape single quotes in search term
        search_escaped = search_term.replace("'", "''")

        return self.conn.execute(f"""
            SELECT
                r.review_id,
                SUBSTR(r.text, 1, 200) || '...' as text_preview,
                r.date,
                r.source,
                r.rating,
                a.sentiment,
                a.themes,
                a.retention_risk
            FROM reviews r
            JOIN llm_analysis a ON r.review_id = a.review_id
            WHERE LOWER(r.text) LIKE LOWER('%{search_escaped}%')
            ORDER BY r.date DESC NULLS LAST
            LIMIT {limit}
        """).df()

    def get_table_counts(self) -> Dict[str, int]:
        """Get count of records in each table"""

        return {
            'reviews': self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0],
            'llm_analysis': self.conn.execute("SELECT COUNT(*) FROM llm_analysis").fetchone()[0],
            'kpi_aggregates': self.conn.execute("SELECT COUNT(*) FROM kpi_aggregates").fetchone()[0]
        }

    def close(self):
        """Close database connection"""
        self.conn.close()
        print("Database connection closed")


# Example usage and testing
if __name__ == '__main__':
    # Initialize database
    db = ReviewDatabase()

    # Try to load data (files may not exist yet)
    print("\nLoading data...")
    db.load_reviews_from_jsonl('data/processed/reviews_final.jsonl')
    db.load_analyses_from_jsonl('data/labeled/silver_standard.jsonl')

    # Load from memory if available
    db.load_memory_analyses('data/memory/analysis_log.jsonl')

    # Show table counts
    print(f"\nTable counts:")
    counts = db.get_table_counts()
    for table, count in counts.items():
        print(f"  {table}: {count} records")

    # Compute KPIs if we have data
    if counts['llm_analysis'] > 0:
        print("\nComputing KPIs...")
        db.compute_kpi_aggregates()

        # Test queries
        print("\n" + "="*60)
        print("TESTING QUERIES")
        print("="*60)

        kpis = db.get_overall_kpis()
        print(f"\nOverall KPIs:")
        print(f"  Total reviews: {kpis['total_reviews']}")
        print(f"  Sources: {kpis['source_count']}")
        print(f"  Avg sentiment: {kpis['avg_sentiment']}/5")
        print(f"  High risk: {kpis['high_risk_pct']}%")
        print(f"  Anomaly rate: {kpis['anomaly_rate']}%")

        print(f"\nTheme distribution:")
        themes = db.get_theme_distribution(top_n=5)
        print(themes.to_string(index=False))

        if counts['kpi_aggregates'] > 0:
            print(f"\nSentiment trend (first 5 rows):")
            trend = db.get_sentiment_trend()
            print(trend.head().to_string(index=False))

    db.close()

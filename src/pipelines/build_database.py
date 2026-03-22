#!/usr/bin/env python3
"""
LLM to SQL pipeline for ReviewInsight AI.

Batch processes reviews through the agent and stores structured results in DuckDB.

Usage:
    python src/pipelines/build_database.py

This pipeline:
1. Loads reviews from JSONL
2. Checks which reviews are already analyzed (resume capability)
3. Runs agent analysis on new reviews
4. Stores structured outputs in SQL
5. Computes KPI aggregates
6. Prints summary statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.database.db_manager import ReviewDatabase
from src.agent.memory import MemoryAwareAgent


class DatabaseBuilder:
    """Build complete SQL database from reviews"""

    def __init__(self, db_path='data/database/reviews.duckdb'):
        """Initialize database and agent"""
        self.db = ReviewDatabase(db_path)
        self.agent = MemoryAwareAgent()
        print("✅ Database and Agent initialized")

    def get_unanalyzed_reviews(
        self,
        reviews_path: str = 'data/processed/reviews_final.jsonl'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load reviews and filter to those not yet analyzed

        Args:
            reviews_path: Path to reviews JSONL

        Returns:
            Tuple of (all_reviews, unanalyzed_reviews)
        """

        if not Path(reviews_path).exists():
            print(f"⚠️  Reviews file not found: {reviews_path}")
            return [], []

        # Load all reviews
        reviews = []
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))

        print(f"Loaded {len(reviews)} total reviews")

        # Get already analyzed review IDs from database
        try:
            analyzed_df = self.db.conn.execute("""
                SELECT review_id FROM llm_analysis
            """).df()

            analyzed_ids = set(analyzed_df['review_id'].tolist()) if not analyzed_df.empty else set()
        except Exception:
            analyzed_ids = set()

        print(f"Already analyzed: {len(analyzed_ids)} reviews")

        # Filter to unanalyzed
        unanalyzed = [r for r in reviews if r.get('review_id') not in analyzed_ids]

        print(f"Need to analyze: {len(unanalyzed)} reviews")

        return reviews, unanalyzed

    def analyze_and_store(self, review: dict) -> Dict:
        """
        Analyze single review with agent and store in database

        Args:
            review: Review dict

        Returns:
            Analysis results
        """

        # Run agent analysis
        result = self.agent.analyze(
            review_text=review.get('text', ''),
            review_id=review.get('review_id')
        )

        # Extract analysis
        analysis = result.get('final_analysis', {})

        # Store in database (use review_id as primary key for upsert)
        try:
            self.db.conn.execute("""
                INSERT OR REPLACE INTO llm_analysis
                (review_id, analysis_id, sentiment, themes, retention_risk,
                 confidence, reasoning, is_anomalous, anomaly_reason, model_version)
                VALUES (?, nextval('analysis_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review.get('review_id'),
                analysis.get('sentiment'),
                analysis.get('themes', []),
                analysis.get('retention_risk'),
                0.85,  # Default confidence
                analysis.get('reasoning', ''),
                analysis.get('is_anomalous', False),
                analysis.get('anomaly_reason', ''),
                'gpt-4o-mini'
            ))
        except Exception as e:
            print(f"  ⚠️ Error storing {review.get('review_id')}: {e}")

        return result

    def analyze_batch(
        self,
        reviews: List[Dict],
        show_progress: bool = True
    ) -> Tuple[int, int]:
        """
        Analyze a batch of reviews

        Args:
            reviews: List of review dicts
            show_progress: Show progress bar

        Returns:
            Tuple of (success_count, error_count)
        """

        if not reviews:
            return 0, 0

        success_count = 0
        error_count = 0

        iterator = tqdm(reviews, desc="Analyzing") if show_progress and HAS_TQDM else reviews

        for review in iterator:
            try:
                self.analyze_and_store(review)
                success_count += 1
            except Exception as e:
                print(f"\n  ⚠️ Error analyzing {review.get('review_id')}: {e}")
                error_count += 1

        return success_count, error_count

    def build_complete_database(
        self,
        reviews_path: str = 'data/processed/reviews_final.jsonl'
    ):
        """
        Complete pipeline: Load → Analyze → Store → Aggregate

        This is the main entry point.
        """

        print("\n" + "="*60)
        print("BUILDING REVIEWINSIGHT DATABASE")
        print("="*60)

        # Step 1: Load reviews into database
        print("\n📥 STEP 1: Loading reviews into database...")
        loaded = self.db.load_reviews_from_jsonl(reviews_path)

        if loaded == 0:
            print("  ⚠️  No reviews loaded. Check file path.")
            return

        # Step 2: Get reviews that need analysis
        print("\n🔍 STEP 2: Identifying reviews to analyze...")
        all_reviews, unanalyzed_reviews = self.get_unanalyzed_reviews(reviews_path)

        if not unanalyzed_reviews:
            print("  ✅ All reviews already analyzed!")
        else:
            # Step 3: Analyze with agent
            print(f"\n🤖 STEP 3: Analyzing {len(unanalyzed_reviews)} reviews with agent...")
            print("  (This may take several minutes...)\n")

            success_count, error_count = self.analyze_batch(
                unanalyzed_reviews,
                show_progress=True
            )

            print(f"\n  ✅ Successfully analyzed: {success_count}")
            print(f"  ❌ Errors: {error_count}")

        # Step 4: Compute KPI aggregates
        print("\n📊 STEP 4: Computing KPI aggregates...")
        self.db.compute_kpi_aggregates()

        # Step 5: Print summary statistics
        print("\n" + "="*60)
        print("DATABASE BUILD COMPLETE")
        print("="*60)

        self.print_summary()

        print(f"\n💾 Database saved to: data/database/reviews.duckdb")
        print(f"✅ Ready for dashboard!")

    def print_summary(self):
        """Print database statistics"""

        # Overall KPIs
        kpis = self.db.get_overall_kpis()

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total reviews: {kpis['total_reviews']}")
        print(f"  Data sources: {kpis['source_count']}")
        print(f"  Avg sentiment: {kpis['avg_sentiment']}/5")
        print(f"  High risk: {kpis['high_risk_pct']}%")
        print(f"  Anomaly rate: {kpis['anomaly_rate']}%")

        # Theme distribution
        print(f"\n🏷️  TOP 5 THEMES:")
        try:
            themes_df = self.db.get_theme_distribution(top_n=5)
            for _, row in themes_df.iterrows():
                print(f"  {row['theme']}: {row['frequency']} ({row['percentage']:.1f}%)")
        except Exception:
            print("  (No theme data available)")

        # Retention risk
        print(f"\n⚠️  RETENTION RISK BREAKDOWN:")
        try:
            risk_df = self.db.get_retention_risk_breakdown()
            for source in risk_df['source'].unique():
                source_data = risk_df[risk_df['source'] == source]
                print(f"  {source}:")
                for _, row in source_data.iterrows():
                    print(f"    {row['retention_risk']}: {row['count']}")
        except Exception:
            print("  (No risk data available)")


def main():
    """Main execution"""
    builder = DatabaseBuilder()
    builder.build_complete_database()


if __name__ == '__main__':
    main()

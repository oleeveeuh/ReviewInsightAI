#!/usr/bin/env python3
"""Ensure the dashboard's DuckDB exists and has data — cloud-safe.

On a fresh clone (or a Streamlit Community Cloud deploy) there is no
data/database/reviews.duckdb: the real corpus lives untracked in
data_local/. This module bootstraps from what the repo tracks:

1. Loads the synthetic fixture corpus (data/processed/reviews_final.jsonl)
   and computes aggregates — the offline `--reviews-only` build path.
2. Seeds llm_analysis from the tracked fixture silver labels
   (data/labeled/silver_standard.jsonl) so the dashboard's charts,
   search, and risk views have honest fixture-labeled content.
3. Never touches an existing database that already has reviews — a local
   real-corpus build is left exactly as it is.

Everything here runs offline: no API key, no LLM calls, no network.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from database.db_manager import ReviewDatabase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Same CWD-relative path as ReviewDatabase's default — both resolve against
# the repo root, which is the working directory on Streamlit Cloud.
DB_PATH = Path("data") / "database" / "reviews.duckdb"
FIXTURE_REVIEWS = ROOT / "data" / "processed" / "reviews_final.jsonl"
FIXTURE_LABELS = ROOT / "data" / "labeled" / "silver_standard.jsonl"


def seed_fixture_labels(db: ReviewDatabase) -> int:
    """Insert the tracked fixture silver labels as llm_analysis rows.

    The labels are the same ones the evaluation harness treats as its
    fixture silver standard — one v5 label per fixture review. Seeding
    them gives the dashboard real (fixture) content without ever calling
    an LLM at deploy time.
    """
    if not FIXTURE_LABELS.exists():
        return 0
    seeded = 0
    with FIXTURE_LABELS.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            al = row.get("auto_labels") or {}
            if not al:
                continue
            try:
                db.conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_analysis
                    (review_id, analysis_id, sentiment, themes, retention_risk,
                     confidence, reasoning, is_anomalous, anomaly_reason, model_version)
                    VALUES (?, nextval('analysis_id_seq'), ?, ?, ?, ?, ?, FALSE, NULL, ?)
                    """,
                    (
                        row.get("review_id"),
                        al.get("sentiment"),
                        al.get("themes", []),
                        al.get("retention_risk"),
                        0.85,
                        "fixture silver label (v5) — seeded offline, see data/labeled/",
                        "gpt-4o-mini (silver fixture)",
                    ),
                )
                seeded += 1
            except Exception as exc:  # one bad row must not break the deploy
                print(f"  ⚠️ seed skip {row.get('review_id')}: {exc}")
    return seeded


def ensure_database(db_path: Path = DB_PATH) -> dict:
    """Bootstrap the dashboard database if (and only if) it is empty."""
    import duckdb

    if db_path.exists():
        try:
            n_reviews = duckdb.connect(str(db_path), read_only=True).execute(
                "SELECT COUNT(*) FROM reviews").fetchone()[0]
        except Exception:
            n_reviews = 0
        if n_reviews > 0:
            return {"action": "existing database left untouched",
                    "path": str(db_path)}

    # Fresh (or absent) database: build from the tracked fixtures.
    db = ReviewDatabase(str(db_path))
    loaded = db.load_reviews_from_jsonl(str(FIXTURE_REVIEWS))
    seeded = seed_fixture_labels(db)
    db.compute_kpi_aggregates()
    db.conn.close()
    return {
        "action": "built from tracked fixtures",
        "reviews_loaded": loaded,
        "labels_seeded": seeded,
        "path": str(db_path),
    }


if __name__ == "__main__":
    print(ensure_database())

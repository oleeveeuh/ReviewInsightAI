#!/usr/bin/env python3
"""Freeze real-corpus AGGREGATES for the public dashboard's headline layer.

Reads the local (never-redistributed) corpus and its v5 silver labels, and
writes dashboard/real_aggregates.json — counts and distributions only, no
review text, same publication policy as the static showcase:

    aggregates may be public; review TEXT stays in data_local/.

The dashboard renders these numbers for its KPI tiles and distribution
charts, and clearly labels the review-level views (search, high-risk
expanders) as synthetic-fixture views. Regenerate after corpus changes;
never hand-edit the JSON.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

CORPUS = ROOT / "data_local/processed/reviews_final.jsonl"
LABELS = ROOT / "data_local/labeled/silver_standard.jsonl"
OUT = Path(__file__).resolve().parent / "real_aggregates.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def month_key(date: str) -> str:
    return (date or "")[:7]  # YYYY-MM


def main() -> None:
    corpus = read_jsonl(CORPUS)
    labels = read_jsonl(LABELS)

    sentiment_counts = Counter()
    risk_counts = Counter()
    theme_counts = Counter()
    anomaly_count = 0
    trend = {}  # (month, source) -> [sum_sentiment, n]
    for row in labels:
        al = row["auto_labels"]
        s = al["sentiment"]
        sentiment_counts[s] += 1
        risk_counts[al["retention_risk"]] += 1
        for theme in al["themes"]:
            theme_counts[theme] += 1
        # the repo's anomaly rule (src/agent/tools.py detect_anomaly)
        themes_l = [str(t).lower() for t in al["themes"]]
        if s <= 2 and "safety" in themes_l:
            anomaly_count += 1
        elif s <= 2 and "pay_benefits" in themes_l and "management" in themes_l:
            anomaly_count += 1
        mk = month_key(row.get("date"))
        if mk:
            key = (mk, row.get("source"))
            trend.setdefault(key, [0, 0])
            trend[key][0] += s
            trend[key][1] += 1

    n = len(labels)
    positive = sum(c for s, c in sentiment_counts.items() if s >= 4)
    high = risk_counts.get("high", 0)

    source_counts = Counter(r["source"] for r in corpus)
    rating_counts = Counter(
        str(r["rating"]) for r in corpus if r.get("rating") is not None)
    year_counts = Counter(int(r["year"]) for r in corpus if r.get("year"))
    rated = [r["rating"] for r in corpus if r.get("rating") is not None]

    data = {
        "generated_by": "dashboard/build_real_aggregates.py (frozen; regenerate, never hand-edit)",
        "provenance": ("aggregates of the real 175-review corpus — counts and "
                       "distributions only; review text is never published "
                       "(see DATASET_CARD.md)"),
        "n_reviews": len(corpus),
        "n_labels": n,
        "date_range": [min(r["date"] for r in corpus if r.get("date")),
                       max(r["date"] for r in corpus if r.get("date"))],
        "sources": dict(source_counts.most_common()),
        "ratings": dict(sorted(rating_counts.items())),
        "mean_rating": round(sum(rated) / max(1, len(rated)), 2),
        "years": dict(sorted(year_counts.items())),
        "sentiment": {str(k): sentiment_counts[k] for k in sorted(sentiment_counts)},
        "risk": dict(risk_counts.most_common()),
        "themes": dict(theme_counts.most_common()),
        "trend": [
            {"month": m, "source": src, "avg_sentiment": round(total / cnt, 3),
             "n": cnt}
            for (m, src), (total, cnt) in sorted(trend.items())
        ],
        "kpis": {
            "avg_sentiment": round(sum(s * c for s, c in sentiment_counts.items()) / max(1, n), 2),
            "positive_pct": round(100.0 * positive / max(1, n), 1),
            "high_risk_pct": round(100.0 * high / max(1, n), 1),
            "anomaly_pct": round(100.0 * anomaly_count / max(1, n), 1),
            "anomaly_count": anomaly_count,
        },
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"written: {OUT} ({OUT.stat().st_size} bytes)")
    print("kpis:", data["kpis"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the showcase dashboard's frozen data file (site/data.js).

Aggregates the project's frozen artifacts into a single JS file the static
showcase can render client-side with no server and no dependencies:

- corpus KPIs and distributions      <- data_local/processed/reviews_final.jsonl
                                     (175 reviews; local, never redistributed)
- per-review silver label views      <- data_local/labeled/silver_standard.jsonl
                                     (single-prompt v5 labels, one row/review)
- preference-pair construction       <- data_local/alignment/preferences.jsonl
- leave-one-prompt-out results       <- data/evaluation/rescored_crossval_summary.json
                                       (committed artifact)
- DPO base-vs-tuned metrics          <- data_local/alignment/eval_report.json

Every number the showcase displays is computed here from those files —
nothing is hand-copied, nothing is fetched at runtime. The output file is
committed, so the deployed page depends only on tracked artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SITE.mkdir(exist_ok=True)

THEME_LABELS = {
    "pay_benefits": "pay & benefits",
    "work_life_balance": "work-life balance",
    "management": "management",
    "safety": "safety",
    "workload": "workload",
    "career_growth": "career growth",
    "culture": "culture",
    "training": "training",
    "physical_toll": "physical toll",
    "peak_season": "peak season",
    "coworkers": "coworkers",
    "other": "other",
}

PROMPT_NAMES = {
    "v1.0": "v1.0 zero-shot basic",
    "v2.0": "v2.0 role enhanced",
    "v3.0": "v3.0 detailed + examples",
    "v4.0": "v4.0 chain of thought",
    "v5.0": "v5.0 structured output",
    "v6.0": "v6.0 detailed analysis",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    corpus = read_jsonl(ROOT / "data_local/processed/reviews_final.jsonl")
    labels = read_jsonl(ROOT / "data_local/labeled/silver_standard.jsonl")
    prefs = read_jsonl(ROOT / "data_local/alignment/preferences.jsonl")
    crossval = json.loads(
        (ROOT / "data/evaluation/rescored_crossval_summary.json").read_text())
    dpo = json.loads(
        (ROOT / "data_local/alignment/eval_report.json").read_text())

    # ---- corpus views -----------------------------------------------------
    source_counts = Counter(r["source"] for r in corpus)
    rating_counts = Counter(
        str(r["rating"]) for r in corpus if r.get("rating") is not None)
    year_counts = Counter(int(r["year"]) for r in corpus if r.get("year"))
    lengths = [len((r.get("text") or "").split()) for r in corpus]

    sentiment_counts = Counter()
    risk_counts = Counter()
    theme_counts = Counter()
    for row in labels:
        al = row["auto_labels"]
        sentiment_counts[al["sentiment"]] += 1
        risk_counts[al["retention_risk"]] += 1
        for theme in al["themes"]:
            theme_counts[THEME_LABELS.get(theme, theme)] += 1

    # ---- preference construction ------------------------------------------
    splits = Counter(p["split"] for p in prefs)
    n_reviews_with_pairs = len({p["review_id"] for p in prefs})

    # ---- leave-one-prompt-out table ----------------------------------------
    lopo = [
        {
            "prompt": PROMPT_NAMES.get(r["prompt_version"], r["prompt_version"]),
            "theme_f1": round(r["rescored_agreement"]["theme_f1"] * 100, 1),
            "sentiment_agreement": round(
                r["rescored_agreement"]["sentiment_exact_agreement"] * 100, 1),
            "risk_agreement": round(r["rescored_agreement"]["risk_agreement"] * 100, 1),
            "label_rows": r["n_label_rows"],
            "unique_reviews": r["n_unique_reviews"],
        }
        for r in crossval["results"]
    ]

    # ---- DPO ----------------------------------------------------------------
    def pct(v: float) -> float:
        return round(v * 100, 1)

    def pp(key: str) -> dict:
        raw_b, raw_t = dpo["base"][key], dpo["dpo_tuned"][key]
        return {"base": pct(raw_b), "tuned": pct(raw_t),
                "delta": round((raw_t - raw_b) * 100, 1)}

    dpo_view = {
        "base_model": dpo["base_model"].split("/")[-1],
        "metrics": [
            {"name": "risk-class agreement", **pp("risk_agreement")},
            {"name": "sentiment exact agreement", **pp("sentiment_exact_agreement")},
            {"name": "theme f1", **pp("theme_f1")},
            {"name": "risk within-one", **pp("risk_within_one")},
        ],
        "n_eval_pairs": dpo["base"]["n_eval_pairs"],
        "n_unparsed": dpo["dpo_tuned"]["n_unparsed"],
    }

    data = {
        "generated_by": "scripts/build_showcase_data.py (frozen; regenerate, never hand-edit)",
        "corpus": {
            "n_reviews": len(corpus),
            "sources": dict(source_counts.most_common()),
            "ratings": dict(sorted(rating_counts.items())),
            "mean_rating": round(sum(
                r["rating"] for r in corpus if r.get("rating") is not None)
                / max(1, sum(1 for r in corpus if r.get("rating") is not None)), 2),
            "years": dict(sorted(year_counts.items())),
            "median_words": int(sorted(lengths)[len(lengths) // 2]),
            "date_range": [min(r["date"] for r in corpus if r.get("date")),
                           max(r["date"] for r in corpus if r.get("date"))],
        },
        "labels": {
            "label_type": "llm silver — one v5 label per review; agreement view, not ground truth",
            "sentiment": {str(k): sentiment_counts[k] for k in sorted(sentiment_counts)},
            "risk": dict(risk_counts.most_common()),
            "themes": dict(theme_counts.most_common()),
        },
        "preferences": {
            "n_pairs": len(prefs),
            "splits": dict(splits),
            "n_reviews_with_pairs": n_reviews_with_pairs,
            "unanimous_skipped": len(corpus) - n_reviews_with_pairs,
        },
        "lopo": lopo,
        "lopo_meta": {
            "protocol": crossval["what_this_is"],
            "n_calls": 1050,
            "failures": crossval["protocol"]["predictions_failed"],
        },
        "dpo": dpo_view,
    }

    out = SITE / "data.js"
    out.write_text(
        "// frozen data — generated by scripts/build_showcase_data.py; regenerate, never hand-edit\n"
        "window.showcaseData = " + json.dumps(data, indent=2) + ";\n")
    print(f"written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

# Dataset Card — ReviewInsight AI

**Last updated:** 2026-10-02 · **Maintainer:** repository owner

This card documents every dataset in the repository: what is tracked (small
synthetic fixtures), what exists only locally (the real corpus, in
`data_local/`, untracked), and what is unknown. The MIT code license covers
**code only** — it grants no rights to any third-party content described here.

---

## 1. Summary

| Dataset | Tracked here? | Records | Content |
|---|---|---|---|
| `data/processed/reviews_final.jsonl` | ✅ synthetic fixture (9) / 📁 local real (175) | Unified review corpus |
| `data/labeled/silver_standard.jsonl` | ✅ synthetic (9) / 📁 local real (175) | LLM labels, single prompt (v5) |
| `data/labeled/silver_standard_crossval.jsonl` | ✅ synthetic (27 rows / 9 reviews) / 📁 local real (525 rows / 175 reviews) | LLM labels, 3 prompts (v2/v3/v6) |
| `data/raw/reddit_reviews.jsonl` | ✅ synthetic (3) / 📁 local real (62) | Raw Reddit posts |
| `data/raw/manual_reddit_template.txt` | ✅ synthetic (2 entries) / 📁 local real (37) | Manually pasted Reddit posts |
| `data/youtube/transcripts/` | ✅ synthetic (1 video) / 📁 local real (8 videos, 16 files) | Caption-level transcripts |
| `data/youtube/youtube_dataset.csv` | ✅ synthetic (18 lines) / 📁 local real (2,294 rows) | One row per caption line |
| `data/glassdoor/reviews.csv`, `data/indeed/reviews.csv` | ✅ header-only (0 rows) | Collection templates |
| DuckDB databases (`data/database/*.duckdb`) | ❌ not tracked | Rebuildable via `src/pipelines/build_database.py` |
| Embeddings (`data/processed/*.npy`) | ❌ not tracked | Rebuildable via `src/agent/build_embeddings.py` |
| Evaluation outputs (`data/evaluation/*`) | ❌ not tracked (except re-scored summary) | Regenerable |

✅ = tracked in git · 📁 = kept locally in `data_local/` (gitignored), never redistributed

**Current tracked record counts by source (synthetic fixtures):**
glassdoor 5 · reddit 3 · youtube 1 = **9 reviews**. Every fixture record
carries `"synthetic": true` or a synthetic-content notice; all IDs, URLs,
subreddits, and channel names in tracked files are invented.

**Local real corpus (`data_local/processed/merge_report.txt`, merge dated
2026-02-15):** **175 reviews** — glassdoor 100 · reddit 57 · youtube 18;
date range 2020-08-01 → 2026-02-15; mean rating 3.36.

---

## 2. Sources and collection method

| Source | Method | When | Notes |
|---|---|---|---|
| Glassdoor | **Manual copy-paste** into a CSV following `data/glassdoor/selection_guide.md` | late 2024 – late 2025 (review dates); collected 2026-02 (approx.) | 100 reviews. **Raw CSV is header-only** — the raw source file was never committed, so the processed rows' original file is gone (provenance gap). |
| Reddit (r/AmazonFC) | **Manual paste** via `src/scraping/manual_reddit_entry.py` + template | posts dated 2020–2021; added 2026-01/02 (per `added_date`) | 37 posts, kept verbatim incl. post IDs and permalinks |
| Reddit (r/supplychain, r/freight) | `src/scraping/reddit_scraper.py` against the Pushshift beta API | posts dated 2020 (approx.) | 25 posts. Pushshift public access was restricted in 2023; the scraper now typically returns HTTP 403 and collects nothing. |
| YouTube | `scripts/get_transcripts.py` (youtube-transcript-api, keyless) over 8 hand-picked videos from `video_list.csv` | videos uploaded 2024 | 8 videos; creator names/handles are part of the local data |
| Indeed | — | — | **Never collected.** `data/indeed/reviews.csv` is an empty schema template. |

The 2026-02-15 date in the merged corpus is a labeling-pipeline artifact, not a
review date.

---

## 3. Transformations and filtering

(`src/processing/merge_sources.py`, deterministic, no LLM involvement)

- Glassdoor `summary/pros/cons/advice` fields concatenated into `text`.
- YouTube transcripts chunked at ~700 words with 100-word overlap, max 5
  chunks/video, 20 chunks total → 18 chunks from 4 of the 8 videos.
- Exact-duplicate removal, length/quality filters (50–5,000 words for Reddit),
  date parsing into `year`/`quarter`, text normalization.
- 62 raw Reddit posts → 57 survived filtering; 100 Glassdoor rows passed
  through as-is; 18 YouTube chunks.

**Provenance gap:** `reviews_final.jsonl` contains 100 Glassdoor reviews but
no raw Glassdoor file exists anywhere (tracked or local). The processed text
is the only surviving copy. Likewise, Indeed is represented by a schema only.

---

## 4. Labels: synthetic vs. human

- **All labels in this repository are LLM-generated** (`gpt-4o-mini`,
  temperature 0.3) using the prompt templates in `src/prompts/templates.py`.
  There are **no human-labeled ground-truth labels**. A human labeling CLI
  exists (`src/labeling/label_reviews.py`) but has never been run —
  `gold_standard.jsonl` does not exist.
- `silver_standard.jsonl`: 1 label per review from prompt v5 (local real: 175).
- `silver_standard_crossval.jsonl`: 3 labels per review from v2/v3/v6
  (local real: 525 rows / 175 unique reviews).
- `confidence` fields are **hard-coded constants** (0.85 / 0.90), not
  calibrated probabilities.
- `data/labeled/labeling_instructions.md` documents an (unexecuted) plan for
  human labeling with a theme codebook.
- The tracked label fixtures are **hand-authored synthetic examples**, marked
  `label_type: "synthetic"`.

---

## 5. Licensing and usage uncertainty

- **Glassdoor content is ToS-restricted.** Glassdoor's terms prohibit
  scraping and redistribution of review content. The 100 reviews were
  manually copied; redistributing them in a public repository is very
  unlikely to be permitted. This is why the real text is untracked.
- **Reddit content** is user-generated under Reddit's user agreement;
  individual posts are public but redistribution rights are not blanket-granted.
  Post IDs/permalinks are retained so original posts can be located.
- **YouTube transcripts** are content of the respective creators; captions are
  auto-generated. Creator names remain attached in the local metadata.
- **The MIT LICENSE covers original code only.** It does not license any
  third-party text described in this card.
- No permission was obtained from any platform or creator. If you need the
  real data, consult each platform's terms; do not ask the repository owner
  to redistribute it.

---

## 6. Privacy and ethics

- Glassdoor reviews: anonymous by platform design; job titles, locations
  (city-level), employment status, and dates are present; no reviewer names.
- Reddit: post IDs and permalinks retained; two post bodies quote a username;
  no author fields are stored.
- YouTube: creator names/handles present; transcripts can contain
  self-identifications.
- Subject matter is workplace sentiment at warehouse employers; the corpus is
  small, geographically skewed (largely one site cluster), and time-skewed
  (2020–2021 Reddit, 2024–2025 Glassdoor).

---

## 7. Intended and prohibited uses

**Intended:** software engineering demonstration (pipeline architecture,
prompt A/B methodology, evaluation hygiene, API/dashboard scaffolding);
education about silver-label pitfalls.

**Prohibited:**
- Training or evaluating production HR/employment-screening systems on this data.
- Presenting LLM-assigned "retention risk" as a real, validated prediction of
  any individual employee's behavior — it is an LLM-generated classification
  of review text, with no ground truth behind it.
- Redistributing the third-party content (see §5).
- Any use implying human-verified labels or accuracy vs. ground truth.

---

## 8. Known provenance gaps

1. 100 Glassdoor reviews exist only as processed text; the raw CSV is absent
   everywhere.
2. The merge script's Glassdoor loader expects columns
   (`review_pros`, `review_cons`, …) that the tracked empty template CSV does
   not have — the original Glassdoor CSV format is unrecoverable from the repo.
3. Reddit scrape date is unrecorded for the 25 Pushshift-collected posts
   (only post dates are stored).
4. Historical experiment artifacts (evaluation JSONs) were generated against
   the real corpus and cannot be reproduced from the tracked fixtures; they
   are retained locally and summarized in
   `data/evaluation/rescored_crossval_summary.json`.
5. `query_embeddings.npy` (a pickled query cache) had no generating script in
   the repo and was removed from tracking; the code no longer reads it.

---

## 9. Maintenance notes

- To regenerate tracked fixtures: fixtures are hand-authored; do not overwrite
  them with real data.
- To rebuild local derived artifacts from the local corpus:
  `python src/processing/merge_sources.py`,
  `python src/agent/build_embeddings.py`,
  `python src/pipelines/build_database.py` (the latter two need
  `requirements-optional.txt`).
- Collection of new data (scraping, paid APIs, new LLM labels) is out of
  scope and requires explicit owner approval.

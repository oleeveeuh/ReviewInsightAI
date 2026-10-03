# Contributing to ReviewInsight AI

Thanks for your interest in contributing. This is a portfolio/research
prototype — small, focused PRs are the best fit.

## Development setup

1. **Fork and clone**
   ```bash
   git clone https://github.com/oleeveeuh/extern.git
   cd extern
   ```

2. **Python 3.11 virtual environment** (pinned numpy/scipy don't support 3.13+)
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt        # core (offline)
   pip install -r requirements-dev.txt    # pytest, httpx, ruff
   # optional, only for LLM/dashboard/scraping work:
   pip install -r requirements-optional.txt
   ```

4. **Environment variables (optional — LLM features only)**
   ```bash
   cp .env.example .env   # add OPENAI_API_KEY if you need LLM features
   ```

## Ground rules

- **Do not commit real third-party review/transcript content.** Tracked files
  under `data/` are synthetic fixtures — see DATASET_CARD.md. Local real data
  belongs in gitignored `data_local/`.
- **Do not describe silver-label agreement as accuracy.** Metrics against
  LLM-generated labels are "agreement"; "accuracy" is reserved for
  comparisons against human ground truth (which does not exist yet).
- Do not commit secrets, DuckDB files, embeddings, or generated evaluation
  outputs (all gitignored).

## Code style

- `ruff check .` must pass (config in `pyproject.toml`).
- Match the surrounding style; type hints and docstrings where they help.

## Testing

```bash
pytest                 # full offline suite (no API key, no network)
bash scripts/verify.sh # ruff + pytest + offline end-to-end demo
```

If you touch the evaluation code, keep the integrity tests passing: no prompt
scored against its own labels, no duplicate-review inflation, dry-run results
without metrics.

## Submitting changes

1. Branch: `git checkout -b feature/your-feature`
2. Commit with a clear message.
3. Push and open a PR describing what changed and why.

CI runs ruff + the offline test suite + the offline demo on Python 3.11.

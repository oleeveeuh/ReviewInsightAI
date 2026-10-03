"""Shared test configuration: paths, sys.path, and environment hygiene.

The whole suite runs offline: no OpenAI package, no API key, no network.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Make both import styles work: `src.agent...` (repo root) and
# `agent...` / `prompts` / `evaluation` (src/ on the path).
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

FIXTURE_REVIEWS = REPO_ROOT / 'data' / 'processed' / 'reviews_final.jsonl'
FIXTURE_SILVER = REPO_ROOT / 'data' / 'labeled' / 'silver_standard.jsonl'
FIXTURE_CROSSVAL = REPO_ROOT / 'data' / 'labeled' / 'silver_standard_crossval.jsonl'


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    """Guarantee tests never see an API key, so no code path can call out."""
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)


@pytest.fixture
def fixture_reviews_path():
    assert FIXTURE_REVIEWS.exists(), "synthetic fixture reviews missing"
    return FIXTURE_REVIEWS


@pytest.fixture
def crossval_evaluator():
    """PromptEvaluator over the synthetic cross-prompt label fixture."""
    from evaluation.evaluate_prompts import PromptEvaluator
    return PromptEvaluator(gold_standard_path=FIXTURE_CROSSVAL)

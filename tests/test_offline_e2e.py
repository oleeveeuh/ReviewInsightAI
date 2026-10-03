"""End-to-end offline workflow: cross-prompt dry-run evaluation on fixtures.

Runs the full evaluate_crossval pipeline in dry-run mode (no API key, no
network) and verifies the output artifact carries honest metadata and no
self-labeled scoring.
"""

import json


import evaluate_crossval as crossval_module


def test_crossval_dry_run_end_to_end(tmp_path, capsys):
    output = tmp_path / 'crossval_dry_run.json'

    crossval_module.evaluate_crossval(
        labels_path='data/labeled/silver_standard_crossval.jsonl',
        prompts_to_test=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        dry_run=True,
        output_path=str(output),
    )

    assert output.exists()
    payload = json.loads(output.read_text())

    meta = payload['run_metadata']
    assert meta['dry_run'] is True
    assert meta['label_type'] == 'synthetic'
    assert meta['generator_prompts'] == ['v2', 'v3', 'v6']
    assert 'no performance meaning' in meta['note']

    results = payload['results']
    assert len(results) == 6
    by_prompt = {r['prompt_version']: r for r in results}

    # Generator prompts must have had their own rows excluded
    assert by_prompt['v2.0']['n_label_rows'] == 18
    assert by_prompt['v2.0']['n_unique_reviews'] == 9
    assert by_prompt['v6.0']['n_label_rows'] == 18
    # Non-generators use all 27 rows
    assert by_prompt['v5.0']['n_label_rows'] == 27
    assert by_prompt['v1.0']['n_label_rows'] == 27

    # Dry-run results must never carry computed metrics
    for r in results:
        assert r['metrics'] is None
        assert r['dry_run'] is True
        assert r['exclude_own_labels'] is True
        assert r['split_method'] == 'unique_review_id'

    # Every unique review is represented
    assert {r['n_unique_reviews'] for r in results} == {9}


def test_live_agreement_metrics_on_fixture(tmp_path, monkeypatch):
    """With metrics enabled (but an offline fake client), agreement computes.

    We monkeypatch the evaluator's LLM entry point to return a fixed valid
    response, exercising the real metrics path without any network access.
    """
    from evaluation.evaluate_prompts import PromptEvaluator
    from prompts.templates import ALL_PROMPTS

    ev = PromptEvaluator(gold_standard_path='data/labeled/silver_standard_crossval.jsonl')

    def fake_call_llm(self, system_prompt, user_prompt, model='gpt-4o-mini',
                      max_retries=3):
        return {'content': json.dumps({'sentiment': 3,
                                       'themes': ['overtime', 'pay_benefits'],
                                       'retention_risk': 'medium'}),
                'tokens_in': 100, 'tokens_out': 30}

    monkeypatch.setattr(PromptEvaluator, 'call_llm', fake_call_llm)

    result = ev.evaluate_prompt(ALL_PROMPTS['v1'], dataset='validation',
                                exclude_own_labels=True, dry_run=False)
    assert result['metrics'] is not None
    m = result['metrics']
    assert 0.0 <= m['theme_f1'] <= 1.0
    assert 0.0 <= m['sentiment_exact_agreement'] <= 1.0
    assert 0.0 <= m['risk_agreement'] <= 1.0
    assert result['n_label_rows'] == 27  # v1 generated no reference labels
    assert result['n_unique_reviews'] == 9
    assert result['label_source']['type'] == 'synthetic'

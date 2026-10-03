"""Evaluation protocol tests.

These tests fail if label leakage (a prompt scored against its own labels)
or duplicate-review inflation is ever reintroduced.
"""

import json


from conftest import FIXTURE_CROSSVAL

from evaluation.evaluate_prompts import PromptEvaluator, _norm_prompt
from prompts.templates import ALL_PROMPTS


def test_splits_operate_on_unique_review_ids(crossval_evaluator):
    """Validation/test splits split the unique-review list, not raw rows."""
    ids = crossval_evaluator.split_review_ids('validation')
    # Fixture: 9 unique reviews (30 label rows); all fit in validation.
    assert len(ids) == 9
    assert len(set(ids)) == len(ids)

    # Multiple label rows per review must all land in the same split.
    rows = crossval_evaluator.select_rows(ids)
    assert len(rows) == 27
    assert {r['review_id'] for r in rows} == set(ids)


def test_no_review_crosses_split_boundary(tmp_path):
    """A review with many labels can never appear in both splits."""
    labels_path = tmp_path / 'many.jsonl'
    with open(labels_path, 'w') as f:
        for i in range(60):  # 60 unique reviews, 3 label rows each
            for pv in ('v2', 'v3', 'v6'):
                f.write(json.dumps({
                    'review_id': f'r_{i:03d}', 'text': f'text {i}',
                    'auto_labels': {'sentiment': 3, 'themes': ['other'],
                                    'retention_risk': 'low'},
                    'prompt_version': pv, 'model': 'synthetic-fixture',
                    'label_type': 'synthetic',
                }) + '\n')

    ev = PromptEvaluator(gold_standard_path=labels_path)
    assert len(ev.review_ids) == 60
    val = set(ev.split_review_ids('validation'))
    test = set(ev.split_review_ids('test'))
    assert len(val) == 50 and len(test) == 10
    assert val.isdisjoint(test), "a review appears in both splits"
    assert val | test == set(ev.review_ids)


def test_leave_one_prompt_out_excludes_self_labeled_rows(crossval_evaluator):
    """The prompt under evaluation must not see labels it generated."""
    for key in ('v2', 'v3', 'v6'):
        prompt = ALL_PROMPTS[key]
        crossval_evaluator.evaluate_prompt(
            prompt, dataset='validation', k_shot=0,
            exclude_own_labels=True, dry_run=True)

        # Recompute exactly what evaluate_prompt scored and assert exclusion.
        rows = crossval_evaluator.select_rows(
            crossval_evaluator.split_review_ids('validation'),
            exclude_prompt_version=prompt.version)
        assert {_norm_prompt(r['prompt_version']) for r in rows} == \
            {'v2', 'v3', 'v6'} - {key}, f"{key}: its own labels must be excluded"
        # 27 total rows -> 18 after excluding one generator's third
        assert len(rows) == 18


def test_including_own_labels_changes_metrics(crossval_evaluator):
    """Sanity: exclusion is active, not a no-op flag (fixture has v6 deviate)."""
    prompt = ALL_PROMPTS['v6']
    excluded = crossval_evaluator.evaluate_prompt(
        prompt, dataset='validation', exclude_own_labels=True, dry_run=True)
    included = crossval_evaluator.evaluate_prompt(
        prompt, dataset='validation', exclude_own_labels=False, dry_run=True)

    assert excluded['n_label_rows'] == 18
    assert included['n_label_rows'] == 27
    assert excluded['metrics'] is None and included['metrics'] is None  # dry-run
    # Non-dry-run metric difference is covered by scripts/rescore_crossval.py
    # output: v6 theme F1 drops from 85.4% to 81.8% once self labels are removed.


def test_counts_report_rows_and_unique_reviews(crossval_evaluator):
    result = crossval_evaluator.evaluate_prompt(
        ALL_PROMPTS['v2'], dataset='validation', exclude_own_labels=True,
        dry_run=True)
    assert result['n_label_rows'] == 18
    assert result['n_unique_reviews'] == 9  # duplicates never inflate this
    assert result['split_method'] == 'unique_review_id'


def test_dry_run_predictions_are_not_metrics(crossval_evaluator):
    """Dry-run predictions derive from reference labels -> metrics must be None."""
    result = crossval_evaluator.evaluate_prompt(
        ALL_PROMPTS['v5'], dataset='validation', dry_run=True)
    assert result['dry_run'] is True
    assert result['metrics'] is None
    assert result['predictions']  # plumbing check produced mock predictions


def test_results_carry_provenance_metadata(crossval_evaluator):
    result = crossval_evaluator.evaluate_prompt(
        ALL_PROMPTS['v3'], dataset='validation', exclude_own_labels=True,
        dry_run=True)
    src = result['label_source']
    assert src['type'] == 'synthetic'
    assert src['generator_prompts'] == ['v2', 'v3', 'v6']
    assert result['timestamp_utc']
    assert result['model']
    assert result['prompt_version'] == 'v3.0'
    assert result['exclude_own_labels'] is True


def test_rescored_summary_exists_and_is_honest():
    """The committed re-scored file must declare its limits and counts."""
    path = FIXTURE_CROSSVAL.parent.parent / 'evaluation' / 'rescored_crossval_summary.json'
    data = json.loads(path.read_text())
    assert 'NOT accuracy against human ground truth' in data['what_this_is']
    assert data['provenance']['label_type'] == 'llm_silver'
    for r in data['results']:
        assert r['n_unique_reviews'] == 17
        assert r['n_label_rows'] <= r['rows_before_exclusion']
    # Self-labeled rows were actually excluded for the generator prompts
    by_prompt = {r['prompt_version']: r for r in data['results']}
    assert by_prompt['v2.0']['rows_excluded_as_self_labeled'] == 17
    assert by_prompt['v6.0']['rows_excluded_as_self_labeled'] == 16
    assert by_prompt['v1.0']['rows_excluded_as_self_labeled'] == 0

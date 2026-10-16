"""Tests for silver-preference pair construction (offline, core deps only)."""

import json

from src.alignment.build_preferences import build_pairs


def _row(rid, prompt, sentiment, themes, risk, text='review text'):
    return {'review_id': rid, 'prompt_version': prompt, 'text': text,
            'model': 'synthetic-fixture',
            'auto_labels': {'sentiment': sentiment, 'themes': themes,
                            'retention_risk': risk}}


def test_unanimous_labels_yield_no_pair():
    rows = [_row('r1', pv, 3, ['overtime'], 'low') for pv in ('v2', 'v3', 'v6')]
    pairs, stats = build_pairs(rows)
    assert pairs == []
    assert stats['unanimous_reviews_skipped'] == 1


def test_disagreement_yields_consensus_ordered_pair():
    rows = [
        _row('r1', 'v2', 2, ['overtime', 'management'], 'high'),
        _row('r1', 'v3', 2, ['overtime', 'management'], 'high'),
        _row('r1', 'v6', 5, ['safety'], 'low'),          # outlier
    ]
    pairs, stats = build_pairs(rows)
    assert stats['pairs_built'] == 1
    pair = pairs[0]
    assert pair['chosen_from'] == 'v2' or pair['chosen_from'] == 'v3'
    assert pair['rejected_from'] == 'v6'
    assert pair['chosen_consensus'] > pair['rejected_consensus']
    assert json.loads(pair['chosen']) == {'sentiment': 2,
                                          'themes': ['overtime', 'management'],
                                          'retention_risk': 'high'}
    assert pair['label_type'] == 'silver_preference'


def test_splits_partition_by_review_id():
    rows = []
    for i in range(10):  # 10 disagreeing reviews
        rows += [
            _row(f'r{i:02d}', 'v2', 2, ['overtime'], 'high'),
            _row(f'r{i:02d}', 'v3', 2, ['overtime'], 'high'),
            _row(f'r{i:02d}', 'v6', 4, ['safety'], 'low'),
        ]
    pairs, stats = build_pairs(rows, eval_fraction=0.2)
    train_ids = {p['review_id'] for p in pairs if p['split'] == 'train'}
    eval_ids = {p['review_id'] for p in pairs if p['split'] == 'eval'}
    assert train_ids.isdisjoint(eval_ids), 'review crossed the split boundary'
    assert stats['train_pairs'] + stats['eval_pairs'] == 10
    assert stats['eval_pairs'] == 2


def test_provenance_is_silver_not_human():
    rows = [_row('r1', 'v2', 2, ['overtime'], 'high'),
            _row('r1', 'v6', 4, ['safety'], 'low')]
    pairs, stats = build_pairs(rows)
    assert all(p['label_type'] == 'silver_preference' for p in pairs)
    assert 'no human preference data' in stats['provenance']


def test_training_scripts_have_actionable_dependency_guard():
    """Core env has no torch; the guard must explain how to set one up."""
    from src.alignment import evaluate_model, train_dpo

    for module in (train_dpo, evaluate_model):
        try:
            module.check_dependencies()
            raised = None
        except RuntimeError as e:
            raised = str(e)
        if raised is not None:  # torch absent in core env
            assert 'requirements-training.txt' in raised
            assert '.venv-rl' in raised


def test_review_truncation_is_token_exact_and_flagged():
    """Efficient-profile truncation: token-level, identity below the cap."""
    from src.alignment.train_dpo import truncate_review_text

    # fake tokenizer: 1 token per word
    def encode(t):
        return t.split()

    def decode(ids):
        return ' '.join(ids)

    short = 'a b c'
    out, changed = truncate_review_text(short, encode, decode, max_tokens=5)
    assert out == short and changed is False

    long = ' '.join(f'w{i}' for i in range(10))
    out, changed = truncate_review_text(long, encode, decode, max_tokens=4)
    assert changed is True
    assert out == 'w0 w1 w2 w3'  # exactly 4 tokens, order preserved

"""Metric computation and LLM-output parsing tests."""

import pytest

from evaluation.evaluate_prompts import PromptEvaluator


def _row(sentiment, themes, risk):
    return {'auto_labels': {'sentiment': sentiment, 'themes': themes,
                            'retention_risk': risk}}


def _pred(sentiment, themes, risk):
    return {'sentiment': sentiment, 'themes': themes, 'retention_risk': risk}


class TestCalculateMetrics:
    def test_perfect_agreement(self):
        rows = [_row(3, ['overtime', 'pay_benefits'], 'medium'),
                _row(1, ['safety'], 'high')]
        preds = [_pred(3, ['pay_benefits', 'overtime'], 'medium'),
                 _pred(1, ['safety'], 'high')]
        m = PromptEvaluator.calculate_metrics(rows, preds)
        assert m['sentiment_exact_agreement'] == 1.0
        assert m['risk_agreement'] == 1.0
        assert m['theme_f1'] == 1.0
        assert m['sentiment_mae'] == 0.0

    def test_agreement_keys_not_accuracy_keys(self):
        """Silver-label metrics must be named 'agreement', never 'accuracy'."""
        m = PromptEvaluator.calculate_metrics([_row(3, ['other'], 'low')],
                                              [_pred(3, ['other'], 'low')])
        assert 'sentiment_exact_agreement' in m
        assert 'risk_agreement' in m
        assert not any('accuracy' in k for k in m), \
            "accuracy naming is reserved for human ground truth comparisons"

    def test_imperfect_agreement_values(self):
        rows = [_row(4, ['overtime'], 'low'),
                _row(2, ['safety', 'management'], 'high'),
                _row(3, ['culture'], 'medium')]
        preds = [_pred(4, ['overtime'], 'low'),        # exact
                 _pred(1, ['safety'], 'medium'),       # sentiment off by 1, risk off by 1
                 _pred(5, ['culture', 'pay_benefits'], 'high')]  # off by 2
        m = PromptEvaluator.calculate_metrics(rows, preds)
        assert m['sentiment_exact_agreement'] == pytest.approx(1 / 3)
        assert m['sentiment_mae'] == pytest.approx((0 + 1 + 2) / 3)
        assert m['risk_agreement'] == pytest.approx(1 / 3)
        # risk gaps: low|low = 0, high|medium = 1, medium|high = 1 -> all within one
        assert m['risk_within_one'] == 1.0
        # themes: tp = 1 + 1 + 1 = 3; fp = 0 + 0 + 1 = 1; fn = 0 + 1 + 0 = 1
        assert m['theme_precision'] == pytest.approx(3 / 4)
        assert m['theme_recall'] == pytest.approx(3 / 4)
        assert m['theme_f1'] == pytest.approx(0.75)

    def test_no_valid_predictions(self):
        m = PromptEvaluator.calculate_metrics([_row(3, ['other'], 'low')], [None])
        assert m == {'error': 'No valid predictions'}


class TestParseLLMResponse:
    def test_valid_json(self):
        out = PromptEvaluator.parse_llm_response(
            '{"sentiment": 4, "themes": ["Work Life Balance"], '
            '"retention_risk": "LOW"}')
        assert out['sentiment'] == 4
        assert out['themes'] == ['work_life_balance']
        assert out['retention_risk'] == 'low'

    def test_markdown_wrapped_json(self):
        out = PromptEvaluator.parse_llm_response(
            '```json\n{"sentiment": 2, "themes": ["safety"], '
            '"retention_risk": "high"}\n```')
        assert out['sentiment'] == 2

    def test_missing_required_key_returns_none(self):
        assert PromptEvaluator.parse_llm_response(
            '{"sentiment": 3, "themes": []}') is None

    def test_invalid_json_returns_none(self):
        assert PromptEvaluator.parse_llm_response('not json at all') is None

    def test_non_numeric_sentiment_returns_none(self):
        assert PromptEvaluator.parse_llm_response(
            '{"sentiment": "high", "themes": [], "retention_risk": "low"}') is None

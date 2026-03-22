#!/usr/bin/env python3
"""
Evaluate prompt templates against labeled data.
Measures accuracy of sentiment, themes, and retention risk predictions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.templates import ALL_PROMPTS, get_prompt


# Mock LLM responses for testing (replace with actual API calls)
MOCK_RESPONSES = {
    "positive": '{"sentiment": 4, "themes": ["pay_benefits", "culture"], "retention_risk": "low"}',
    "neutral": '{"sentiment": 3, "themes": ["workload", "other"], "retention_risk": "low"}',
    "negative": '{"sentiment": 2, "themes": ["overtime", "management"], "retention_risk": "medium"}',
}


class PromptEvaluator:
    """Evaluate prompt performance against gold standard labels."""

    def __init__(self, labeled_path=None, predictions_path=None):
        """
        Args:
            labeled_path: Path to gold standard labeled data
            predictions_path: Path to LLM predictions (or generate mock)
        """
        self.labeled_path = labeled_path or Path(__file__).parent.parent.parent / "data" / "labeled" / "gold_standard.jsonl"
        self.predictions_path = predictions_path

        self.labeled_data = self.load_labeled()
        self.predictions = self.load_predictions()

    def load_labeled(self) -> List[Dict]:
        """Load gold standard labels."""
        labeled = []
        if not self.labeled_path.exists():
            print(f"⚠️  No labeled data found at {self.labeled_path}")
            print("   Run the labeling tool first: python src/labeling/label_reviews.py")
            return []

        with open(self.labeled_path, 'r') as f:
            for line in f:
                if line.strip():
                    labeled.append(json.loads(line))

        print(f"Loaded {len(labeled)} labeled reviews")
        return labeled

    def load_predictions(self) -> Dict[str, Dict]:
        """Load LLM predictions."""
        predictions = {}

        if self.predictions_path and Path(self.predictions_path).exists():
            with open(self.predictions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        predictions[data['review_id']] = data.get('prediction', {})
        else:
            print("⚠️  No predictions file. Using mock data for demonstration.")

        return predictions

    def parse_prediction(self, prediction_text: str) -> Dict:
        """Parse LLM response into structured format."""
        try:
            # Try direct JSON parse
            return json.loads(prediction_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[^}]+\}', prediction_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {"sentiment": None, "themes": [], "retention_risk": None}

    def evaluate_sentiment(self, predicted: int, actual: int) -> Dict:
        """Evaluate sentiment prediction accuracy."""
        results = {
            "exact_match": predicted == actual,
            "off_by_one": abs(predicted - actual) <= 1,
            "direction_match": (predicted >= 3 and actual >= 3) or (predicted < 3 and actual < 3)
        }
        return results

    def evaluate_themes(self, predicted: List[str], actual: List[str]) -> Dict:
        """Evaluate theme prediction using Jaccard similarity."""
        pred_set = set(predicted or [])
        actual_set = set(actual or [])

        if not pred_set and not actual_set:
            return {"jaccard": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

        intersection = pred_set & actual_set
        union = pred_set | actual_set

        jaccard = len(intersection) / len(union) if union else 0

        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(actual_set) if actual_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "intersection": list(intersection),
            "missed": list(actual_set - pred_set),
            "extra": list(pred_set - actual_set)
        }

    def evaluate_retention_risk(self, predicted: str, actual: str) -> Dict:
        """Evaluate retention risk prediction."""
        # Exact match
        exact = predicted == actual

        # Directional match (low < medium < high)
        risk_order = {"low": 1, "medium": 2, "high": 3}
        pred_val = risk_order.get(predicted, 0)
        actual_val = risk_order.get(actual, 0)

        within_one = abs(pred_val - actual_val) <= 1

        return {
            "exact_match": exact,
            "within_one": within_one,
            "predicted": predicted,
            "actual": actual
        }

    def evaluate_all(self) -> pd.DataFrame:
        """Evaluate all predictions against labels."""
        results = []

        for labeled in self.labeled_data:
            review_id = labeled['review_id']

            # Get prediction (use mock if not available)
            if review_id in self.predictions:
                pred = self.predictions[review_id]
            else:
                # Use mock based on actual sentiment for demo
                sentiment = labeled.get('labels', {}).get('sentiment', 3)
                if sentiment >= 4:
                    mock = MOCK_RESPONSES["positive"]
                elif sentiment <= 2:
                    mock = MOCK_RESPONSES["negative"]
                else:
                    mock = MOCK_RESPONSES["neutral"]
                pred = self.parse_prediction(mock)

            actual = labeled.get('labels', {})

            # Evaluate each component
            sentiment_pred = pred.get('sentiment')
            sentiment_actual = actual.get('sentiment')
            themes_pred = pred.get('themes', [])
            themes_actual = actual.get('themes', [])
            risk_pred = pred.get('retention_risk')
            risk_actual = actual.get('retention_risk')

            result = {
                'review_id': review_id,
                'source': labeled.get('source'),
                'sentiment_pred': sentiment_pred,
                'sentiment_actual': sentiment_actual,
                'themes_pred': themes_pred,
                'themes_actual': themes_actual,
                'risk_pred': risk_pred,
                'risk_actual': risk_actual,
            }

            # Add metrics
            if sentiment_pred and sentiment_actual:
                sentiment_eval = self.evaluate_sentiment(sentiment_pred, sentiment_actual)
                result.update({f'sent_{k}': v for k, v in sentiment_eval.items()})

            theme_eval = self.evaluate_themes(themes_pred, themes_actual)
            result.update({f'theme_{k}': v for k, v in theme_eval.items()})

            if risk_pred and risk_actual:
                risk_eval = self.evaluate_retention_risk(risk_pred, risk_actual)
                result.update({f'risk_{k}': v for k, v in risk_eval.items()})

            results.append(result)

        return pd.DataFrame(results)

    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        if len(df) == 0:
            print("No data to evaluate.")
            return

        print(f"\nEvaluated {len(df)} reviews\n")

        # Sentiment accuracy
        if 'sent_exact_match' in df.columns:
            sent_acc = df['sent_exact_match'].mean() * 100
            sent_off_by_one = df['sent_off_by_one'].mean() * 100
            print(f"Sentiment Accuracy:")
            print(f"  Exact match:    {sent_acc:.1f}%")
            print(f"  Within ±1:      {sent_off_by_one:.1f}%")

            # Confusion matrix
            print(f"\n  Predicted vs Actual:")
            for actual in sorted(df['sentiment_actual'].dropna().unique()):
                actual_preds = df[df['sentiment_actual'] == actual]
                preds = actual_preds['sentiment_pred'].value_counts()
                pred_str = ", ".join([f"{int(k)}:{v}" for k, v in preds.items()])
                print(f"    Actual {int(actual)}: {pred_str}")

        # Theme F1
        if 'theme_f1' in df.columns:
            avg_f1 = df['theme_f1'].mean() * 100
            avg_precision = df['theme_precision'].mean() * 100
            avg_recall = df['theme_recall'].mean() * 100
            print(f"\nTheme Detection:")
            print(f"  Precision: {avg_precision:.1f}%")
            print(f"  Recall:    {avg_recall:.1f}%")
            print(f"  F1 Score:  {avg_f1:.1f}%")

            # Most confused themes
            all_missed = []
            for missed_list in df['theme_missed'].dropna():
                all_missed.extend(missed_list)
            if all_missed:
                missed_counts = Counter(all_missed)
                print(f"\n  Most Missed Themes:")
                for theme, count in missed_counts.most_common(5):
                    print(f"    {theme}: {count}")

        # Retention risk
        if 'risk_exact_match' in df.columns:
            risk_acc = df['risk_exact_match'].mean() * 100
            risk_within = df['risk_within_one'].mean() * 100
            print(f"\nRetention Risk:")
            print(f"  Exact match:  {risk_acc:.1f}%")
            print(f"  Within one:  {risk_within:.1f}%")

        print("\n" + "=" * 60)


def main():
    """Run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate prompt performance")
    parser.add_argument('--labeled', '-l', type=str,
                       help='Path to labeled gold standard data')
    parser.add_argument('--predictions', '-p', type=str,
                       help='Path to LLM predictions')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path for results CSV')

    args = parser.parse_args()

    evaluator = PromptEvaluator(
        labeled_path=args.labeled,
        predictions_path=args.predictions
    )

    df = evaluator.evaluate_all()
    evaluator.print_summary(df)

    if args.output and len(df) > 0:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

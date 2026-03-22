#!/usr/bin/env python3
"""
Evaluate prompt templates against gold standard labels using OpenAI API.
Compares different prompt versions and k-shot settings.
"""

import json
import pandas as pd
import numpy as np
import os
import time
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Run: pip install openai python-dotenv")

from prompts.templates import ALL_PROMPTS


class PromptEvaluator:
    """Evaluate prompt performance against gold standard labels."""

    def __init__(self, gold_standard_path=None, api_key=None):
        """
        Args:
            gold_standard_path: Path to labeled gold standard data
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        # Set up paths
        if gold_standard_path is None:
            gold_standard_path = Path(__file__).parent.parent.parent / "data" / "labeled" / "gold_standard.jsonl"

        self.gold_standard_path = Path(gold_standard_path)

        # Load API key
        if HAS_OPENAI:
            load_dotenv()
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                self.client = None
                print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        else:
            self.client = None

        # Load gold standard data
        self.gold_data = self.load_gold_standard()

        # Split into validation and test sets
        self.val_data = self.gold_data[:50]
        self.test_data = self.gold_data[50:] if len(self.gold_data) > 50 else []

        print(f"Loaded {len(self.gold_data)} gold standard samples")
        print(f"Validation set: {len(self.val_data)}")
        print(f"Test set: {len(self.test_data)}")

    def load_gold_standard(self) -> List[Dict]:
        """Load gold standard labels."""
        if not self.gold_standard_path.exists():
            print(f"⚠️  No labeled data found at {self.gold_standard_path}")
            print("   Run the labeling tool first: python src/labeling/label_reviews.py")
            return []

        data = []
        with open(self.gold_standard_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return data

    def call_llm(self, system_prompt: str, user_prompt: str,
                 model: str = "gpt-4o-mini", max_retries: int = 3) -> Optional[Dict]:
        """Call OpenAI API with retry logic."""
        if not self.client:
            return None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                return {
                    'content': response.choices[0].message.content,
                    'tokens_in': response.usage.prompt_tokens,
                    'tokens_out': response.usage.completion_tokens
                }

            except Exception as e:
                print(f"  API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

        return None

    def parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM JSON response."""
        try:
            # Remove markdown code blocks
            response_text = response_text.replace('```json', '').replace('```', '').strip()

            # Parse JSON
            result = json.loads(response_text)

            # Validate structure
            if 'sentiment' not in result or 'themes' not in result or 'retention_risk' not in result:
                return None

            # Normalize
            result['sentiment'] = int(result['sentiment'])
            result['themes'] = [str(t).lower().replace(' ', '_').replace('-', '_')
                               for t in result['themes']]
            result['retention_risk'] = str(result['retention_risk']).lower()

            return result

        except Exception as e:
            return None

    def evaluate_prompt(self, prompt_template, dataset: str = 'validation',
                       k_shot: int = 0, model: str = "gpt-4o-mini",
                       max_samples: Optional[int] = None,
                       dry_run: bool = False) -> Dict:
        """
        Evaluate a prompt template on dataset.

        Args:
            prompt_template: PromptTemplate object
            dataset: 'validation' or 'test'
            k_shot: Number of few-shot examples
            model: OpenAI model name
            max_samples: Limit evaluation to N samples
            dry_run: If True, simulate predictions without API calls

        Returns:
            Dict with metrics and predictions
        """
        data = self.val_data if dataset == 'validation' else self.test_data
        if max_samples:
            data = data[:max_samples]

        print(f"\n{'='*60}")
        print(f"Evaluating: {prompt_template.name} (k={k_shot})")
        print(f"Model: {model} | Dataset: {dataset} ({len(data)} samples)")
        if dry_run:
            print(f"🔄 DRY RUN MODE - No API calls")
        print(f"{'='*60}")

        predictions = []
        total_tokens_in = 0
        total_tokens_out = 0
        failed = 0

        for i, sample in enumerate(data):
            print(f"Progress: {i+1}/{len(data)}", end='\r')

            # Render prompt
            system_prompt, user_prompt = prompt_template.render(
                sample['text'],
                k_shot=k_shot
            )

            if dry_run or not self.client:
                # Mock prediction based on gold label (for testing)
                gold = sample.get('labels', {})
                mock_sentiment = max(1, min(5, gold.get('sentiment', 3) +
                                         np.random.randint(-1, 2)))
                predictions.append({
                    'sentiment': mock_sentiment,
                    'themes': gold.get('themes', ['other'])[:2],
                    'retention_risk': gold.get('retention_risk', 'low')
                })
                time.sleep(0.01)  # Simulate API delay
                continue

            # Call LLM
            response = self.call_llm(system_prompt, user_prompt, model=model)

            if response is None:
                failed += 1
                predictions.append(None)
                continue

            total_tokens_in += response['tokens_in']
            total_tokens_out += response['tokens_out']

            # Parse response
            parsed = self.parse_llm_response(response['content'])

            if parsed is None:
                failed += 1
                predictions.append(None)
            else:
                predictions.append(parsed)

            # Rate limiting
            time.sleep(0.2)

        print(f"\nCompleted! Failed: {failed}/{len(data)}")

        # Calculate metrics
        metrics = self.calculate_metrics(
            data, predictions,
            total_tokens_in, total_tokens_out,
            model, len(data) - failed
        )

        # Store full results
        results = {
            'prompt_version': prompt_template.version,
            'prompt_name': prompt_template.name,
            'k_shot': k_shot,
            'model': model,
            'dataset': dataset,
            'n_samples': len(data),
            'n_failed': failed,
            'n_valid': len(data) - failed,
            'metrics': metrics,
            'predictions': predictions
        }

        self.print_metrics(results)

        return results

    def calculate_metrics(self, gold_data: List[Dict], predictions: List[Optional[Dict]],
                         tokens_in: int, tokens_out: int, model: str,
                         n_valid: int) -> Dict:
        """Calculate evaluation metrics."""

        # Filter out failed predictions
        valid_pairs = [(g, p) for g, p in zip(gold_data, predictions) if p is not None]

        if not valid_pairs:
            return {'error': 'No valid predictions'}

        # Handle both 'labels' and 'auto_labels' keys (for silver standard compatibility)
        def get_labels(gold_item):
            return gold_item.get('labels') or gold_item.get('auto_labels', {})

        gold_labels = [get_labels(g) for g, _ in valid_pairs]
        pred_labels = [p for _, p in valid_pairs]

        metrics = {}

        # 1. Sentiment metrics
        gold_sentiment = [g['sentiment'] for g in gold_labels]
        pred_sentiment = [p['sentiment'] for p in pred_labels]

        # MAE and RMSE
        errors = [(g - p) for g, p in zip(gold_sentiment, pred_sentiment)]
        metrics['sentiment_mae'] = sum(abs(e) for e in errors) / len(errors)
        metrics['sentiment_rmse'] = np.sqrt(np.mean([e**2 for e in errors]))

        # Exact match accuracy
        metrics['sentiment_exact_accuracy'] = sum(g == p for g, p in
                                                  zip(gold_sentiment, pred_sentiment)) / len(gold_sentiment)

        # Directional accuracy (positive vs negative)
        gold_pos = [g >= 3 for g in gold_sentiment]
        pred_pos = [p >= 3 for p in pred_sentiment]
        metrics['sentiment_direction_accuracy'] = sum(g == p for g, p in
                                                      zip(gold_pos, pred_pos)) / len(gold_pos)

        # 2. Theme metrics (multi-label)
        all_themes = set()
        for g in gold_labels:
            all_themes.update(g.get('themes', []))
        for p in pred_labels:
            all_themes.update(p.get('themes', []))

        gold_theme_vectors = []
        pred_theme_vectors = []

        for g, p in zip(gold_labels, pred_labels):
            gold_vec = [1 if theme in g.get('themes', []) else 0 for theme in all_themes]
            pred_vec = [1 if theme in p.get('themes', []) else 0 for theme in all_themes]
            gold_theme_vectors.append(gold_vec)
            pred_theme_vectors.append(pred_vec)

        # Flatten for micro metrics
        gold_flat = [item for sublist in gold_theme_vectors for item in sublist]
        pred_flat = [item for sublist in pred_theme_vectors for item in sublist]

        if gold_flat and pred_flat:
            # True positives, false positives, false negatives
            tp = sum(1 for g, p in zip(gold_flat, pred_flat) if g == 1 and p == 1)
            fp = sum(1 for g, p in zip(gold_flat, pred_flat) if g == 0 and p == 1)
            fn = sum(1 for g, p in zip(gold_flat, pred_flat) if g == 1 and p == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['theme_precision'] = precision
            metrics['theme_recall'] = recall
            metrics['theme_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Exact match
        exact_matches = sum(1 for g, p in zip(gold_labels, pred_labels)
                           if set(g.get('themes', [])) == set(p.get('themes', [])))
        metrics['theme_exact_match'] = exact_matches / len(gold_labels)

        # 3. Retention risk metrics
        gold_risk = [g.get('retention_risk', 'low') for g in gold_labels]
        pred_risk = [p.get('retention_risk', 'low') for p in pred_labels]

        risk_map = {'low': 0, 'medium': 1, 'high': 2}
        gold_risk_num = [risk_map.get(r, 1) for r in gold_risk]
        pred_risk_num = [risk_map.get(r, 1) for r in pred_risk]

        metrics['risk_accuracy'] = sum(g == p for g, p in zip(gold_risk_num, pred_risk_num)) / len(gold_risk_num)

        # Within one
        within_one = sum(1 for g, p in zip(gold_risk_num, pred_risk_num) if abs(g - p) <= 1)
        metrics['risk_within_one'] = within_one / len(gold_risk_num)

        # 4. Cost metrics
        # Pricing (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
        if model == "gpt-4o-mini":
            cost_per_1k_in = 0.150 / 1000
            cost_per_1k_out = 0.600 / 1000
        elif model == "gpt-4o":
            cost_per_1k_in = 2.50 / 1000
            cost_per_1k_out = 10.00 / 1000
        else:
            cost_per_1k_in = 0.150 / 1000
            cost_per_1k_out = 0.600 / 1000

        total_cost = (tokens_in * cost_per_1k_in / 1000) + (tokens_out * cost_per_1k_out / 1000)
        cost_per_sample = total_cost / n_valid if n_valid > 0 else 0

        metrics['total_tokens_in'] = tokens_in
        metrics['total_tokens_out'] = tokens_out
        metrics['total_cost'] = total_cost
        metrics['cost_per_sample'] = cost_per_sample
        metrics['cost_per_1k_samples'] = cost_per_sample * 1000

        return metrics

    def print_metrics(self, results: Dict):
        """Pretty print metrics."""
        m = results.get('metrics', {})

        if 'error' in m:
            print(f"\n❌ Error: {m['error']}")
            return

        print(f"\n{'='*60}")
        print(f"RESULTS: {results['prompt_name']} (k={results['k_shot']})")
        print(f"{'='*60}")

        print(f"\n📊 SENTIMENT:")
        print(f"  MAE: {m.get('sentiment_mae', 0):.3f}")
        print(f"  RMSE: {m.get('sentiment_rmse', 0):.3f}")
        print(f"  Exact Accuracy: {m.get('sentiment_exact_accuracy', 0):.1%}")
        print(f"  Direction Accuracy: {m.get('sentiment_direction_accuracy', 0):.1%}")

        print(f"\n🏷️  THEMES:")
        print(f"  Precision: {m.get('theme_precision', 0):.1%}")
        print(f"  Recall: {m.get('theme_recall', 0):.1%}")
        print(f"  F1: {m.get('theme_f1', 0):.1%}")
        print(f"  Exact Match: {m.get('theme_exact_match', 0):.1%}")

        print(f"\n⚠️  RETENTION RISK:")
        print(f"  Accuracy: {m.get('risk_accuracy', 0):.1%}")
        print(f"  Within One: {m.get('risk_within_one', 0):.1%}")

        print(f"\n💰 COST:")
        print(f"  Total: ${m.get('total_cost', 0):.4f}")
        print(f"  Per sample: ${m.get('cost_per_sample', 0):.4f}")
        print(f"  Tokens: {m.get('total_tokens_in', 0):,} in / {m.get('total_tokens_out', 0):,} out")


def run_comparison(evaluator: PromptEvaluator, prompts_to_test: List[str] = None,
                   k_shots_to_test: List[int] = [0, 3], dry_run: bool = False) -> List[Dict]:
    """Run comparison across multiple prompts and k-shot settings."""

    if prompts_to_test is None:
        prompts_to_test = ['v1', 'v2', 'v3', 'v5']

    all_results = []

    for prompt_key in prompts_to_test:
        if prompt_key not in ALL_PROMPTS:
            print(f"⚠️  Unknown prompt: {prompt_key}")
            continue

        prompt = ALL_PROMPTS[prompt_key]

        for k in k_shots_to_test:
            # Skip k>0 for prompts without few-shot examples
            if k > 0 and not prompt.few_shot_examples:
                continue

            results = evaluator.evaluate_prompt(
                prompt,
                dataset='validation',
                k_shot=k,
                dry_run=dry_run
            )

            all_results.append(results)

    return all_results


def save_results(all_results: List[Dict], output_path: str = None):
    """Save evaluation results."""
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "prompt_comparison.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save summary without predictions
    summary_results = []
    for r in all_results:
        summary = {k: v for k, v in r.items() if k != 'predictions'}
        summary_results.append(summary)

    with open(output_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Save detailed with predictions
    detailed_path = str(output_path).replace('.json', '_detailed.json')
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"💾 Detailed results saved to: {detailed_path}")

    # Create comparison table
    create_comparison_table(summary_results, output_path.parent / "comparison.csv")


def create_comparison_table(results: List[Dict], output_path: Path):
    """Create CSV comparison table."""
    rows = []
    for r in results:
        m = r.get('metrics', {})
        rows.append({
            'Prompt': r['prompt_name'],
            'K-shot': r['k_shot'],
            'Valid': r['n_valid'],
            'Sentiment_MAE': m.get('sentiment_mae', ''),
            'Sentiment_Acc': f"{m.get('sentiment_exact_accuracy', 0):.1%}",
            'Theme_F1': f"{m.get('theme_f1', 0):.1%}",
            'Risk_Acc': f"{m.get('risk_accuracy', 0):.1%}",
            'Cost_per_1k': f"${m.get('cost_per_1k_samples', 0):.2f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"💾 Comparison table saved to: {output_path}")


def main():
    """Run prompt evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate prompt performance")
    parser.add_argument('--gold', '-g', type=str,
                       help='Path to gold standard labels')
    parser.add_argument('--prompts', '-p', type=str, nargs='+',
                       default=['v1', 'v2', 'v3', 'v5'],
                       help='Prompt versions to test')
    parser.add_argument('--k-shots', '-k', type=int, nargs='+', default=[0, 3],
                       help='K-shot values to test')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Run without API calls (for testing)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path for results')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = PromptEvaluator(gold_standard_path=args.gold)

    if len(evaluator.gold_data) == 0:
        print("❌ No gold standard data available")
        return

    # Run comparison
    print("\n🚀 Starting prompt evaluation...")
    if args.dry_run:
        print("🔄 DRY RUN MODE - No API calls")

    results = run_comparison(
        evaluator,
        prompts_to_test=args.prompts,
        k_shots_to_test=args.k_shots,
        dry_run=args.dry_run
    )

    # Save results
    save_results(results, args.output)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

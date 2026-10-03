#!/usr/bin/env python3
"""
Evaluate prompt templates against reference labels using the OpenAI API.

Important terminology
---------------------
All labels shipped in this repository are LLM-generated ("silver" labels); none
are human-verified. Metrics computed against them are **agreement** metrics
(agreement with a reference labeler), NOT accuracy against ground truth, and
NOT unbiased estimates of real-world performance. The word "accuracy" in this
codebase is reserved for a future evaluation against human-labeled data.

Evaluation protocol
-------------------
- Splits operate on UNIQUE review IDs. Multiple label rows for the same review
  (e.g. labels from several generator prompts) never straddle a split boundary.
- By default, label rows produced by the prompt under evaluation are excluded
  (leave-one-prompt-out), so a prompt is never scored against its own output.
- Every result records the prompt version, model, timestamp, split method,
  label source, and whether labels are synthetic or LLM-generated.
- Dry-run "predictions" are derived from the reference labels themselves and
  are explicitly flagged as non-performance data (metrics are set to None).

Usage:
    python -m src.evaluation.evaluate_prompts --dry-run
"""

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add src/ to path so `prompts` and `evaluation` import as top-level packages
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from prompts.templates import ALL_PROMPTS

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "labeled" / "silver_standard.jsonl"

# Number of unique reviews in the validation split
VALIDATION_REVIEWS = 50


def _norm_prompt(version: str) -> str:
    """Normalize a prompt version to its major key ('v2.0' -> 'v2')."""
    return str(version).split('.')[0].strip().lower()


class PromptEvaluator:
    """Evaluate prompt agreement against reference (silver) labels.

    All reference labels are treated as LLM-generated unless a row explicitly
    declares ``label_type == 'human'``.
    """

    def __init__(self, gold_standard_path=None, api_key=None):
        self.gold_standard_path = Path(gold_standard_path or DEFAULT_LABELS_PATH)
        self.client = None

        if HAS_OPENAI:
            load_dotenv()
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)

        self.label_rows = self.load_label_rows()
        self.review_ids = self._unique_review_ids()
        self.labels_by_review = self._group_by_review()
        self.generator_prompts = sorted({
            _norm_prompt(r.get('prompt_version', ''))
            for r in self.label_rows if r.get('prompt_version')
        })
        self.label_models = sorted({str(r.get('model', 'unknown')) for r in self.label_rows})
        self.label_type = self._detect_label_type()

        print(f"Loaded {len(self.label_rows)} label rows "
              f"covering {len(self.review_ids)} unique reviews")
        print(f"Label source: {self.label_type} "
              f"(models: {', '.join(self.label_models)}; generator prompts: "
              f"{', '.join(self.generator_prompts) or 'n/a'})")

    # ------------------------------------------------------------------
    # Loading and splitting
    # ------------------------------------------------------------------

    def load_label_rows(self) -> List[Dict]:
        """Load reference label rows from a JSONL file."""
        if not self.gold_standard_path.exists():
            print(f"No label file found at {self.gold_standard_path}")
            print("Run the labeling pipeline first, or use the synthetic "
                  "fixtures in data/labeled/.")
            return []

        rows = []
        with open(self.gold_standard_path, 'r') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _unique_review_ids(self) -> List[str]:
        """Unique review IDs in first-seen order (split unit)."""
        seen, ordered = set(), []
        for row in self.label_rows:
            rid = row.get('review_id')
            if rid and rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        return ordered

    def _group_by_review(self) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        for row in self.label_rows:
            groups.setdefault(row.get('review_id'), []).append(row)
        return groups

    def _detect_label_type(self) -> str:
        for row in self.label_rows:
            lt = row.get('label_type')
            if lt:
                return str(lt)
        return 'llm_silver'

    def split_review_ids(self, dataset: str = 'validation',
                         max_reviews: Optional[int] = None) -> List[str]:
        """Split by UNIQUE review ID so repeated labels cannot cross splits."""
        if dataset == 'validation':
            ids = self.review_ids[:VALIDATION_REVIEWS]
        elif dataset == 'test':
            ids = self.review_ids[VALIDATION_REVIEWS:]
        elif dataset == 'all':
            ids = list(self.review_ids)
        else:
            raise ValueError(f"Unknown dataset split: {dataset!r}")
        if max_reviews:
            ids = ids[:max_reviews]
        return ids

    def select_rows(self, review_ids: List[str],
                    exclude_prompt_version: Optional[str] = None) -> List[Dict]:
        """Collect label rows for the given reviews.

        Rows whose generating prompt matches ``exclude_prompt_version`` are
        dropped (leave-one-prompt-out), so the prompt under evaluation is never
        scored against labels it produced itself.
        """
        excluded_key = _norm_prompt(exclude_prompt_version) if exclude_prompt_version else None
        rows = []
        for rid in review_ids:
            for row in self.labels_by_review.get(rid, []):
                if excluded_key and _norm_prompt(row.get('prompt_version', '')) == excluded_key:
                    continue
                rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # LLM calls and parsing
    # ------------------------------------------------------------------

    def call_llm(self, system_prompt: str, user_prompt: str,
                 model: str = "gpt-4o-mini", max_retries: int = 3) -> Optional[Dict]:
        """Call OpenAI API with retry logic."""
        if not self.client:
            raise RuntimeError(
                "No OpenAI client available. Install the optional LLM "
                "dependencies (pip install -r requirements-optional.txt) and "
                "set OPENAI_API_KEY in your environment."
            )

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
                print(f"  API error (attempt {attempt + 1}): {type(e).__name__}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    @staticmethod
    def parse_llm_response(response_text: str) -> Optional[Dict]:
        """Parse an LLM JSON response into a normalized label dict."""
        try:
            text = response_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(text)

            if not all(k in result for k in ('sentiment', 'themes', 'retention_risk')):
                return None

            result['sentiment'] = int(result['sentiment'])
            result['themes'] = [str(t).lower().replace(' ', '_').replace('-', '_')
                                for t in result['themes']]
            result['retention_risk'] = str(result['retention_risk']).lower()
            return result
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_prompt(self, prompt_template, dataset: str = 'validation',
                        k_shot: int = 0, model: str = "gpt-4o-mini",
                        max_reviews: Optional[int] = None,
                        exclude_own_labels: bool = True,
                        dry_run: bool = False) -> Dict:
        """Evaluate a prompt template against reference labels.

        Args:
            prompt_template: PromptTemplate object
            dataset: 'validation', 'test', or 'all' (split by unique review ID)
            k_shot: Number of few-shot examples appended to the user message
            model: OpenAI model name
            max_reviews: Limit evaluation to N unique reviews
            exclude_own_labels: Drop label rows generated by this prompt
                (leave-one-prompt-out; strongly recommended)
            dry_run: If True, simulate predictions from the reference labels.
                Metrics are NOT computed (predictions are label-derived and
                carry no performance meaning).

        Returns:
            Result dict with metrics, counts, and full run metadata.
        """
        review_ids = self.split_review_ids(dataset, max_reviews=max_reviews)
        data = self.select_rows(review_ids, exclude_prompt_version=prompt_template.version
                                if exclude_own_labels else None)

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {prompt_template.name} (k={k_shot})")
        print(f"Model: {model} | Split: {dataset} "
              f"({len(review_ids)} unique reviews -> {len(data)} label rows)")
        if exclude_own_labels:
            print(f"Leave-one-prompt-out: rows labeled by "
                  f"{prompt_template.version} are excluded")
        if dry_run:
            print("DRY RUN - predictions are derived from the reference "
                  "labels; metrics will not be computed")
        print(f"{'=' * 60}")

        predictions = []
        total_tokens_in = total_tokens_out = failed = 0

        for i, sample in enumerate(data):
            print(f"Progress: {i + 1}/{len(data)}", end='\r')
            system_prompt, user_prompt = prompt_template.render(sample['text'], k_shot=k_shot)

            if dry_run or not self.client:
                # Plumbing check only: jitter the reference label.
                ref = sample.get('labels') or sample.get('auto_labels') or {}
                mock = {
                    'sentiment': max(1, min(5, int(ref.get('sentiment', 3)) +
                                            int(np.random.randint(-1, 2)))),
                    'themes': list(ref.get('themes', ['other']))[:2],
                    'retention_risk': ref.get('retention_risk', 'low'),
                }
                predictions.append(mock)
                continue

            response = self.call_llm(system_prompt, user_prompt, model=model)
            if response is None:
                failed += 1
                predictions.append(None)
                continue

            total_tokens_in += response['tokens_in']
            total_tokens_out += response['tokens_out']

            parsed = self.parse_llm_response(response['content'])
            if parsed is None:
                failed += 1
                predictions.append(None)
            else:
                predictions.append(parsed)
            time.sleep(0.2)

        print(f"\nCompleted. Failed: {failed}/{len(data)}")

        reviewed_ids = list(dict.fromkeys(s['review_id'] for s in data))

        results = {
            'prompt_version': prompt_template.version,
            'prompt_name': prompt_template.name,
            'k_shot': k_shot,
            'model': model,
            'dataset': dataset,
            'split_method': 'unique_review_id',
            'exclude_own_labels': exclude_own_labels,
            'n_label_rows': len(data),
            'n_unique_reviews': len(reviewed_ids),
            'n_failed': failed,
            'n_valid': len(data) - failed,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'label_source': {
                'path': str(self.gold_standard_path),
                'type': self.label_type,
                'models': self.label_models,
                'generator_prompts': self.generator_prompts,
            },
            # Metrics are None in dry-run mode: label-derived predictions say
            # nothing about real performance.
            'metrics': None if dry_run else self.calculate_metrics(
                data, predictions, total_tokens_in, total_tokens_out, model),
            'dry_run': dry_run,
            'predictions': predictions,
        }

        if not dry_run:
            self.print_metrics(results)
        return results

    # ------------------------------------------------------------------
    # Metrics (agreement with the reference labeler)
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_labels(row: Dict) -> Dict:
        return row.get('labels') or row.get('auto_labels') or {}

    @staticmethod
    def calculate_metrics(rows: List[Dict], predictions: List[Optional[Dict]],
                          tokens_in: int = 0, tokens_out: int = 0,
                          model: str = "gpt-4o-mini") -> Dict:
        """Compute agreement metrics between predictions and reference labels.

        Usable standalone (no instance state). Names use ``*_agreement``
        because the reference labels are LLM- or synthetic-generated, not
        human ground truth.
        """
        valid = [(r, p) for r, p in zip(rows, predictions) if p is not None]
        if not valid:
            return {'error': 'No valid predictions'}

        refs = [PromptEvaluator._ref_labels(r) for r, _ in valid]
        preds = [p for _, p in valid]
        metrics: Dict = {}

        # 1. Sentiment agreement
        ref_s = [int(g['sentiment']) for g in refs]
        pred_s = [int(p['sentiment']) for p in preds]
        errors = [g - p for g, p in zip(ref_s, pred_s)]
        metrics['sentiment_mae'] = sum(abs(e) for e in errors) / len(errors)
        metrics['sentiment_rmse'] = float(np.sqrt(np.mean([e ** 2 for e in errors])))
        metrics['sentiment_exact_agreement'] = sum(
            g == p for g, p in zip(ref_s, pred_s)) / len(ref_s)
        ref_pos = [g >= 3 for g in ref_s]
        pred_pos = [p >= 3 for p in pred_s]
        metrics['sentiment_direction_agreement'] = sum(
            g == p for g, p in zip(ref_pos, pred_pos)) / len(ref_pos)

        # 2. Theme agreement (multi-label, micro-averaged)
        all_themes = set()
        for g in refs:
            all_themes.update(g.get('themes', []))
        for p in preds:
            all_themes.update(p.get('themes', []))

        tp = fp = fn = 0
        for g, p in zip(refs, preds):
            g_set, p_set = set(g.get('themes', [])), set(p.get('themes', []))
            tp += len(g_set & p_set)
            fp += len(p_set - g_set)
            fn += len(g_set - p_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics['theme_precision'] = precision
        metrics['theme_recall'] = recall
        metrics['theme_f1'] = 2 * precision * recall / (precision + recall) \
            if (precision + recall) else 0.0
        metrics['theme_exact_match'] = sum(
            set(g.get('themes', [])) == set(p.get('themes', []))
            for g, p in zip(refs, preds)) / len(refs)

        # 3. Retention-risk agreement
        risk_map = {'low': 0, 'medium': 1, 'high': 2}
        ref_r = [risk_map.get(g.get('retention_risk', 'low'), 1) for g in refs]
        pred_r = [risk_map.get(p.get('retention_risk', 'low'), 1) for p in preds]
        metrics['risk_agreement'] = sum(
            g == p for g, p in zip(ref_r, pred_r)) / len(ref_r)
        metrics['risk_within_one'] = sum(
            abs(g - p) <= 1 for g, p in zip(ref_r, pred_r)) / len(ref_r)

        # 4. Cost
        pricing = {
            'gpt-4o-mini': (0.150 / 1_000_000, 0.600 / 1_000_000),
            'gpt-4o': (2.50 / 1_000_000, 10.00 / 1_000_000),
        }
        pin, pout = pricing.get(model, pricing['gpt-4o-mini'])
        total_cost = tokens_in * pin + tokens_out * pout
        n_valid = len(valid)
        metrics['total_tokens_in'] = tokens_in
        metrics['total_tokens_out'] = tokens_out
        metrics['total_cost'] = total_cost
        metrics['cost_per_sample'] = total_cost / n_valid if n_valid else 0.0
        metrics['cost_per_1k_samples'] = metrics['cost_per_sample'] * 1000

        return metrics

    def print_metrics(self, results: Dict):
        """Pretty print agreement metrics."""
        m = results.get('metrics') or {}
        if 'error' in m:
            print(f"\nError: {m['error']}")
            return

        print(f"\n{'=' * 60}")
        print(f"RESULTS: {results['prompt_name']} (k={results['k_shot']})")
        print(f"Label type: {results['label_source']['type']} "
              f"(agreement, not accuracy vs. ground truth)")
        print(f"Rows: {results['n_label_rows']} label rows | "
              f"{results['n_unique_reviews']} unique reviews")
        print(f"{'=' * 60}")

        print("\nSENTIMENT:")
        print(f"  MAE: {m.get('sentiment_mae', 0):.3f}")
        print(f"  RMSE: {m.get('sentiment_rmse', 0):.3f}")
        print(f"  Exact agreement: {m.get('sentiment_exact_agreement', 0):.1%}")
        print(f"  Direction agreement: {m.get('sentiment_direction_agreement', 0):.1%}")

        print("\nTHEMES:")
        print(f"  Precision: {m.get('theme_precision', 0):.1%}")
        print(f"  Recall: {m.get('theme_recall', 0):.1%}")
        print(f"  F1: {m.get('theme_f1', 0):.1%}")
        print(f"  Exact match: {m.get('theme_exact_match', 0):.1%}")

        print("\nRETENTION RISK (LLM-assigned review-risk class):")
        print(f"  Agreement: {m.get('risk_agreement', 0):.1%}")
        print(f"  Within one: {m.get('risk_within_one', 0):.1%}")

        print("\nCOST:")
        print(f"  Total: ${m.get('total_cost', 0):.4f}")
        print(f"  Per sample: ${m.get('cost_per_sample', 0):.4f}")
        print(f"  Tokens: {m.get('total_tokens_in', 0):,} in / "
              f"{m.get('total_tokens_out', 0):,} out")


def run_comparison(evaluator: PromptEvaluator, prompts_to_test: List[str] = None,
                   k_shots_to_test: List[int] = (0,), dry_run: bool = False) -> List[Dict]:
    """Run a comparison across prompts and k-shot settings.

    Note: k_shot > 0 appends the built-in few-shot examples to the user
    message; every prompt template ships the same example set, so k_shot is a
    runtime setting independent of the prompt's name (a prompt called
    "Zero-shot" is still zero-shot only when k_shot=0).
    """
    if prompts_to_test is None:
        prompts_to_test = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    all_results = []
    for prompt_key in prompts_to_test:
        if prompt_key not in ALL_PROMPTS:
            print(f"Unknown prompt: {prompt_key}")
            continue
        prompt = ALL_PROMPTS[prompt_key]
        for k in k_shots_to_test:
            all_results.append(evaluator.evaluate_prompt(
                prompt, dataset='validation', k_shot=k, dry_run=dry_run))
    return all_results


def save_results(all_results: List[Dict], output_path: str = None):
    """Save evaluation results (summary + detailed) and a CSV comparison."""
    if output_path is None:
        output_path = REPO_ROOT / "data" / "evaluation" / "prompt_comparison.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_results = [{k: v for k, v in r.items() if k != 'predictions'}
                       for r in all_results]
    with open(output_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    detailed_path = str(output_path).replace('.json', '_detailed.json')
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")

    create_comparison_table(summary_results, output_path.parent / "comparison.csv")


def create_comparison_table(results: List[Dict], output_path: Path):
    """Create CSV comparison table."""
    rows = []
    for r in results:
        m = r.get('metrics') or {}
        rows.append({
            'Prompt': r['prompt_name'],
            'Version': r['prompt_version'],
            'K-shot': r['k_shot'],
            'LabelRows': r['n_label_rows'],
            'UniqueReviews': r['n_unique_reviews'],
            'LabelType': r['label_source']['type'],
            'SplitMethod': r['split_method'],
            'OwnLabelsExcluded': r['exclude_own_labels'],
            'Sentiment_MAE': m.get('sentiment_mae', ''),
            'Sentiment_Agreement': f"{m.get('sentiment_exact_agreement', 0):.1%}",
            'Theme_F1': f"{m.get('theme_f1', 0):.1%}",
            'Risk_Agreement': f"{m.get('risk_agreement', 0):.1%}",
            'Cost_per_1k': f"${m.get('cost_per_1k_samples', 0):.2f}",
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Comparison table saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate prompt agreement against silver labels")
    parser.add_argument('--gold', '-g', type=str,
                        help='Path to reference labels (JSONL)')
    parser.add_argument('--prompts', '-p', type=str, nargs='+',
                        default=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
    parser.add_argument('--k-shots', '-k', type=int, nargs='+', default=[0])
    parser.add_argument('--max-reviews', '-n', type=int, default=None,
                        help='Limit to N unique reviews')
    parser.add_argument('--include-own-labels', action='store_true',
                        help='Disable leave-one-prompt-out exclusion (not '
                             'recommended; results will be optimistic)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Plumbing check without API calls (no metrics)')
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    evaluator = PromptEvaluator(gold_standard_path=args.gold)
    if not evaluator.label_rows:
        print("No reference labels available.")
        return

    print("\nStarting prompt evaluation...")
    if args.dry_run:
        print("DRY RUN MODE - no API calls, no metrics")

    results = run_comparison(
        evaluator,
        prompts_to_test=args.prompts,
        k_shots_to_test=args.k_shots,
        dry_run=args.dry_run,
    )
    save_results(results, args.output)
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

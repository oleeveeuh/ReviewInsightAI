#!/usr/bin/env python3
"""
Re-score SAVED cross-validation predictions under a corrected protocol.

This script performs no API calls. It takes:
  1. a saved evaluation file whose per-sample predictions align row-by-row
     with a reference-label JSONL (e.g. data/evaluation/experiment_crossval.json
     evaluated against the first 50 rows of silver_standard_crossval.jsonl),
  2. the reference-label JSONL,

and recomputes agreement metrics with leave-one-prompt-out exclusion: rows
whose generating prompt equals the prompt under evaluation are dropped.

The committed output (data/evaluation/rescored_crossval_summary.json) is a
re-analysis of historical predictions generated in February 2026 with
gpt-4o-mini; it is NOT a fresh run and NOT accuracy vs. ground truth.

Usage:
    python scripts/rescore_crossval.py \
        --predictions data_local/evaluation/experiment_crossval.json \
        --labels data_local/labeled/silver_standard_crossval.jsonl
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from evaluation.evaluate_prompts import PromptEvaluator, _norm_prompt

# Defaults point at the recoverable local copies of the historical artifacts
DEFAULT_PREDICTIONS = REPO_ROOT / 'data_local/evaluation/experiment_crossval.json'
DEFAULT_LABELS = REPO_ROOT / 'data_local/labeled/silver_standard_crossval.jsonl'
DEFAULT_OUTPUT = REPO_ROOT / 'data/evaluation/rescored_crossval_summary.json'


def rescore(predictions_path: Path, labels_path: Path, output_path: Path):
    with open(predictions_path) as f:
        saved = json.load(f)

    label_rows = []
    with open(labels_path) as f:
        for line in f:
            if line.strip():
                label_rows.append(json.loads(line))

    label_models = sorted({str(r.get('model', 'unknown')) for r in label_rows})
    generator_prompts = sorted({_norm_prompt(r.get('prompt_version', ''))
                                for r in label_rows if r.get('prompt_version')})
    label_type = next((r.get('label_type') for r in label_rows if r.get('label_type')),
                      'llm_silver')

    print(f"Saved evaluation file : {predictions_path} ({len(saved)} entries)")
    print(f"Reference labels      : {labels_path} ({len(label_rows)} rows)")
    print(f"Label type            : {label_type}; models: {label_models}; "
          f"generator prompts: {generator_prompts}")

    reanalyzed = []
    for entry in saved:
        prompt_key = _norm_prompt(entry.get('prompt_version', ''))
        preds = entry.get('predictions')
        if not preds:
            print(f"  {prompt_key}: no saved predictions, skipping")
            continue

        # Predictions align 1:1 (in order) with the first len(preds) label rows.
        pairs = list(zip(label_rows[:len(preds)], preds))
        kept = [(r, p) for r, p in pairs
                if _norm_prompt(r.get('prompt_version', '')) != prompt_key]
        excluded = len(pairs) - len(kept)

        kept_rows = [r for r, _ in kept]
        kept_preds = [p for _, p in kept]
        unique_reviews = len({r['review_id'] for r in kept_rows})

        old_m = entry.get('metrics') or {}
        new_m = PromptEvaluator.calculate_metrics(kept_rows, kept_preds)

        reanalyzed.append({
            'prompt_version': entry.get('prompt_version'),
            'prompt_name': entry.get('prompt_name'),
            'k_shot': entry.get('k_shot'),
            'model': entry.get('model'),
            'rows_before_exclusion': len(pairs),
            'rows_excluded_as_self_labeled': excluded,
            'n_label_rows': len(kept_rows),
            'n_unique_reviews': unique_reviews,
            'original_metrics_uncorrected': {
                'theme_f1': old_m.get('theme_f1'),
                'sentiment_exact_agreement': old_m.get('sentiment_exact_accuracy'),
                'risk_agreement': old_m.get('risk_accuracy'),
            },
            'rescored_agreement': {
                'theme_f1': new_m.get('theme_f1'),
                'theme_precision': new_m.get('theme_precision'),
                'theme_recall': new_m.get('theme_recall'),
                'sentiment_exact_agreement': new_m.get('sentiment_exact_agreement'),
                'sentiment_mae': new_m.get('sentiment_mae'),
                'risk_agreement': new_m.get('risk_agreement'),
            },
        })

        print(f"  {prompt_key}: kept {len(kept_rows)} rows / {unique_reviews} reviews "
              f"(excluded {excluded} self-labeled) "
              f"theme_f1 {old_m.get('theme_f1', 0):.3f} -> {new_m.get('theme_f1', 0):.3f}")

    payload = {
        'what_this_is': (
            'Offline re-analysis of previously saved predictions under a '
            'leave-one-prompt-out protocol. No new model calls were made. '
            'Reference labels are LLM-generated (silver); metrics are '
            'agreement with that labeler, NOT accuracy against human ground '
            'truth, and NOT unbiased estimates of real-world performance.'),
        'source_artifacts': {
            'predictions_file': str(predictions_path),
            'labels_file': str(labels_path),
        },
        'provenance': {
            'predictions_generated': '2026-02 (per source file timestamps)',
            'prediction_model': 'gpt-4o-mini',
            'label_model': label_models,
            'generator_prompts': generator_prompts,
            'label_type': label_type,
            'original_protocol_known_issues': [
                'Original run scored every prompt against ALL label rows, '
                'including rows the prompt generated itself.',
                'Original run used the first 50 label rows only '
                '(~17 unique reviews), not the full 175-review corpus.',
                'All runs were zero-shot (k_shot=0); published k-shot '
                'annotations were not from these artifacts.',
            ],
        },
        'rescored_at_utc': datetime.now(timezone.utc).isoformat(),
        'split_method': 'row-level re-scoring; splits in the live evaluator '
                        'operate on unique review IDs',
        'results': reanalyzed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved re-scored summary to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Re-score saved predictions with leave-one-prompt-out exclusion")
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.predictions.exists() or not args.labels.exists():
        parser.error(
            f"Input artifacts not found.\n  predictions: {args.predictions}\n"
            f"  labels: {args.labels}\nThe historical files are not tracked in "
            f"git; they must exist locally (see data_local/).")
    rescore(args.predictions, args.labels, args.output)

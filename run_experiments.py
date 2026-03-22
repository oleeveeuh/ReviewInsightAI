#!/usr/bin/env python3
"""
Comprehensive A/B testing experiment runner for prompt engineering.

Generates silver standard labels then runs 30+ experiments across:
- 6 prompt versions (v1-v6)
- Multiple k-shot values (0, 1, 2, 3, 5)
- Multiple models (optional)
- Multiple temperatures (optional)

Usage:
    # Step 1: Generate silver standard labels
    python run_experiments.py --generate-silver --prompt v5 --samples 175

    # Step 2: Run full experiment grid
    python run_experiments.py --run-all

    # Step 3: Run specific experiments
    python run_experiments.py --prompts v1 v2 v3 v5 --k-shots 0 3 --model gpt-4o-mini
"""

import json
import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from prompts.templates import ALL_PROMPTS
from evaluation.evaluate_prompts import PromptEvaluator, save_results


def generate_silver_standard(
    output_path: str = 'data/labeled/silver_standard.jsonl',
    prompt_version: str = 'v5',
    model: str = 'gpt-4o-mini',
    max_samples: int = None,
    api_key: str = None
):
    """
    Generate silver standard labels using a specific prompt.

    Args:
        output_path: Where to save silver standard labels
        prompt_version: Which prompt to use (default: v5)
        model: OpenAI model to use
        max_samples: Max reviews to label (None = all)
        api_key: OpenAI API key
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    import time

    load_dotenv()
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)
    prompt = ALL_PROMPTS[prompt_version]

    # Load reviews
    reviews_path = Path('data/processed/reviews_final.jsonl')
    if not reviews_path.exists():
        print(f"❌ Reviews not found at {reviews_path}")
        return

    reviews = []
    with open(reviews_path, 'r') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    if max_samples:
        reviews = reviews[:max_samples]

    print(f"📝 Generating silver standard with {prompt_version}")
    print(f"   Model: {model}")
    print(f"   Reviews: {len(reviews)}")
    print(f"   Output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    labeled = 0
    failed = 0

    with open(output_path, 'w') as out:
        for i, review in enumerate(reviews):
            print(f"  Progress: {i+1}/{len(reviews)}", end='\r')

            system_prompt, user_prompt = prompt.render(review['text'], k_shot=0)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                content = response.choices[0].message.content
                # Parse JSON response
                content = content.replace('```json', '').replace('```', '').strip()
                labels = json.loads(content)

                # Normalize
                labels['sentiment'] = int(labels.get('sentiment', 3))
                labels['themes'] = [str(t).lower().replace(' ', '_').replace('-', '_')
                                   for t in labels.get('themes', [])]
                labels['retention_risk'] = str(labels.get('retention_risk', 'low')).lower()
                labels['confidence'] = 0.85  # Fixed for silver standard

                output = {
                    'review_id': review.get('review_id', f"review_{i}"),
                    'text': review.get('text', ''),
                    'source': review.get('source', 'unknown'),
                    'date': review.get('date'),
                    'auto_labels': labels,
                    'model': model,
                    'prompt_version': prompt_version
                }

                out.write(json.dumps(output) + '\n')
                labeled += 1

            except Exception as e:
                failed += 1
                print(f"\n  ❌ Error on review {i}: {e}")

            time.sleep(0.2)  # Rate limiting

    print(f"\n✅ Generated {labeled} labels ({failed} failed)")
    print(f"   Saved to: {output_path}")


def run_experiment_grid(
    prompts: List[str] = None,
    k_shots: List[int] = None,
    models: List[str] = None,
    gold_path: str = None,
    output_dir: str = None
) -> List[Dict]:
    """
    Run a grid of experiments.

    Args:
        prompts: Prompt versions to test (default: all)
        k_shots: K-shot values to test (default: [0, 1, 2, 3, 5])
        models: Models to test (default: ['gpt-4o-mini'])
        gold_path: Path to gold standard labels
        output_dir: Where to save results

    Returns:
        List of experiment results
    """
    if prompts is None:
        prompts = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    if k_shots is None:
        k_shots = [0, 1, 2, 3, 5]
    if models is None:
        models = ['gpt-4o-mini']

    # Calculate total experiments
    total_experiments = len(prompts) * len(k_shots) * len(models)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT GRID")
    print(f"{'='*70}")
    print(f"Prompts: {', '.join(prompts)} ({len(prompts)})")
    print(f"K-shots: {k_shots} ({len(k_shots)})")
    print(f"Models: {models} ({len(models)})")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}\n")

    # Initialize evaluator
    evaluator = PromptEvaluator(gold_standard_path=gold_path)

    if len(evaluator.gold_data) == 0:
        print("❌ No gold standard data available")
        print("   Run: python run_experiments.py --generate-silver")
        return []

    results = []

    for model in models:
        for prompt_key in prompts:
            if prompt_key not in ALL_PROMPTS:
                print(f"⚠️  Unknown prompt: {prompt_key}")
                continue

            prompt = ALL_PROMPTS[prompt_key]

            for k in k_shots:
                # Skip k>0 for prompts without few-shot examples
                if k > 0 and not prompt.few_shot_examples:
                    print(f"⊘ Skipping {prompt_key} k={k} (no few-shot examples)")
                    continue

                experiment_name = f"{prompt_key}_k{k}_{model}"

                print(f"\n🧪 Running: {experiment_name}")

                try:
                    result = evaluator.evaluate_prompt(
                        prompt,
                        dataset='validation',
                        k_shot=k,
                        model=model,
                        dry_run=False  # Set to True for testing without API
                    )
                    results.append(result)

                except Exception as e:
                    print(f"  ❌ Error: {e}")

    # Save results
    if output_dir is None:
        output_dir = Path('data/evaluation')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_results(results, output_dir / f"experiment_grid_{timestamp}.json")

    return results


def print_experiment_summary(results: List[Dict]):
    """Print summary of experiment results."""
    if not results:
        print("No results to summarize")
        return

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")

    # Sort by F1 score
    sorted_results = sorted(
        results,
        key=lambda r: r.get('metrics', {}).get('theme_f1', 0),
        reverse=True
    )

    print("Top 10 by Theme F1:")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:10]):
        m = r.get('metrics', {})
        print(f"{i+1:2}. {r['prompt_name']:20} k={r['k_shot']} | "
              f"F1: {m.get('theme_f1', 0):.3f} | "
              f"Acc: {m.get('sentiment_exact_accuracy', 0):.3f} | "
              f"Cost: ${m.get('cost_per_1k_samples', 0):.2f}/1k")

    print("\nBest by each metric:")
    print("-" * 70)

    metrics_to_check = [
        ('sentiment_mae', False, 'Lowest MAE'),
        ('sentiment_exact_accuracy', True, 'Best Sentiment Acc'),
        ('theme_f1', True, 'Best Theme F1'),
        ('risk_accuracy', True, 'Best Risk Acc'),
        ('cost_per_1k_samples', False, 'Lowest Cost')
    ]

    for metric_key, higher_is_better, name in metrics_to_check:
        best = max(results, key=lambda r: r.get('metrics', {}).get(metric_key, 0))
        m = best.get('metrics', {})
        val = m.get(metric_key, 0)
        print(f"{name:20}: {best['prompt_name']} k={best['k_shot']} ({val:.3f})")


def main():
    import time

    parser = argparse.ArgumentParser(
        description="Run A/B testing experiments on prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate silver standard labels first
  python run_experiments.py --generate-silver --prompt v5 --samples 175

  # Run full experiment grid (30+ experiments)
  python run_experiments.py --run-all

  # Run specific experiments
  python run_experiments.py --prompts v1 v2 v3 --k-shots 0 3

  # Dry run (no API calls)
  python run_experiments.py --run-all --dry-run
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--generate-silver', action='store_true',
                           help='Generate silver standard labels')
    mode_group.add_argument('--run-all', action='store_true',
                           help='Run full experiment grid')
    mode_group.add_argument('--run-specific', action='store_true',
                           help='Run specific experiments')

    # Silver generation options
    parser.add_argument('--silver-output', type=str,
                       default='data/labeled/silver_standard.jsonl',
                       help='Output path for silver standard')
    parser.add_argument('--prompt', type=str, default='v5',
                       choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
                       help='Prompt version for silver generation')

    # Experiment options
    parser.add_argument('--prompts', type=str, nargs='+',
                       choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
                       help='Prompt versions to test')
    parser.add_argument('--k-shots', type=int, nargs='+',
                       help='K-shot values to test')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models to test (e.g., gpt-4o-mini gpt-4o)')
    parser.add_argument('--gold', type=str,
                       help='Path to gold standard labels')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--samples', type=int,
                       help='Max samples to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without API calls (for testing)')

    args = parser.parse_args()

    # Execute based on mode
    if args.generate_silver:
        generate_silver_standard(
            output_path=args.silver_output,
            prompt_version=args.prompt,
            max_samples=args.samples
        )

    elif args.run_all or args.run_specific:
        # Default grid for --run-all
        if args.run_all:
            if args.prompts is None:
                args.prompts = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
            if args.k_shots is None:
                args.k_shots = [0, 1, 2, 3, 5]
            if args.models is None:
                args.models = ['gpt-4o-mini']

        # For --run-specific, require explicit args
        elif args.run_specific:
            if args.prompts is None or args.k_shots is None:
                parser.error("--run-specific requires --prompts and --k-shots")

        results = run_experiment_grid(
            prompts=args.prompts,
            k_shots=args.k_shots,
            models=args.models,
            gold_path=args.gold,
            output_dir=args.output_dir
        )

        if results:
            print_experiment_summary(results)


if __name__ == '__main__':
    main()

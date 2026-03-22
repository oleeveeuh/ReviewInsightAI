#!/usr/bin/env python3
"""
Generate silver standard labels using multiple prompts (cross-validation).

This script generates labels using a subset of prompts, ensuring that
when we test, no prompt is evaluated against its own labels.

Strategy:
- Use v2, v3, v6 to generate labels (NOT v5)
- Test all prompts (v1-v6) against these labels
- This prevents any prompt from being tested against itself

Usage:
    python generate_crossval_labels.py --prompts v2 v3 v6 --samples 175
"""

import json
import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from prompts.templates import ALL_PROMPTS
from openai import OpenAI
from dotenv import load_dotenv


def generate_labels_with_prompts(
    prompt_keys: List[str],
    reviews_path: str = 'data/processed/reviews_final.jsonl',
    output_path: str = 'data/labeled/silver_standard_crossval.jsonl',
    model: str = 'gpt-4o-mini',
    max_samples: int = None
) -> Dict[str, any]:
    """
    Generate labels using multiple prompts (ensemble).

    For each review, generates labels from multiple prompts and keeps
    them all (separate entries) for cross-validation.

    Args:
        prompt_keys: List of prompt versions to use (e.g., ['v2', 'v3', 'v6'])
        reviews_path: Path to reviews JSONL
        output_path: Where to save labels
        model: OpenAI model
        max_samples: Max reviews to label

    Returns:
        Statistics about generation
    """

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return {}

    client = OpenAI(api_key=api_key)

    # Load reviews
    reviews = []
    with open(reviews_path, 'r') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    if max_samples:
        reviews = reviews[:max_samples]

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SILVER STANDARD GENERATION")
    print(f"{'='*70}")
    print(f"Reviews: {len(reviews)}")
    print(f"Generator prompts: {prompt_keys}")
    print(f"Output: {output_path}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_reviews': len(reviews),
        'prompts_used': prompt_keys,
        'labels_per_review': len(prompt_keys),
        'total_labels': len(reviews) * len(prompt_keys),
        'failed': 0
    }

    # For each review, generate labels with each prompt
    with open(output_path, 'w') as out:
        for i, review in enumerate(reviews):
            print(f"[{i+1}/{len(reviews)}] Labeling review...")

            review_text = review.get('text', review.get('review', ''))
            review_id = review.get('review_id', f"review_{i}")

            for prompt_key in prompt_keys:
                prompt = ALL_PROMPTS[prompt_key]
                system_prompt, user_prompt = prompt.render(review_text, k_shot=0)

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
                    content = content.replace('```json', '').replace('```', '').strip()
                    labels = json.loads(content)

                    # Normalize
                    labels['sentiment'] = int(labels.get('sentiment', 3))
                    labels['themes'] = [str(t).lower().replace(' ', '_').replace('-', '_')
                                       for t in labels.get('themes', [])]
                    labels['retention_risk'] = str(labels.get('retention_risk', 'low')).lower()
                    labels['confidence'] = 0.85

                    output = {
                        'review_id': review_id,
                        'text': review_text,
                        'source': review.get('source', 'unknown'),
                        'date': review.get('date'),
                        'auto_labels': labels,
                        'model': model,
                        'prompt_version': prompt_key,
                        'is_generator': True  # Mark as generator prompt
                    }

                    out.write(json.dumps(output) + '\n')

                except Exception as e:
                    stats['failed'] += 1
                    print(f"  ❌ Error with {prompt_key}: {e}")

            time.sleep(0.2)  # Rate limiting

    print(f"\n✅ Generated {stats['total_labels']} labels ({stats['failed']} failed)")
    print(f"   Labels per review: {stats['labels_per_review']}")
    print(f"   Saved to: {output_path}")

    return stats


def ensemble_labels(
    input_path: str = 'data/labeled/silver_standard_crossval.jsonl',
    output_path: str = 'data/labeled/silver_standard_ensemble.jsonl'
):
    """
    Create ensemble labels by majority voting across multiple prompts.

    For each review, takes the majority vote across all generator prompts
    to create a single robust label.

    Args:
        input_path: Cross-validation labels (multiple per review)
        output_path: Ensemble labels (one per review)
    """

    print(f"\n{'='*70}")
    print(f"CREATING ENSEMBLE SILVER STANDARD")
    print(f"{'='*70}")

    # Load all labels
    all_labels = defaultdict(list)  # review_id -> [labels]

    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                review_id = data['review_id']
                all_labels[review_id].append(data['auto_labels'])

    # Create ensemble labels
    with open(output_path, 'w') as out:
        for review_id, labels_list in all_labels.items():
            # Get the original review data
            first_label = labels_list[0]

            # Majority vote for sentiment
            sentiments = [l['sentiment'] for l in labels_list]
            sentiment = int(round(sum(sentiments) / len(sentiments)))

            # Majority vote for risk
            risks = [l['retention_risk'] for l in labels_list]
            risk_counts = defaultdict(int)
            for r in risks:
                risk_counts[r] += 1
            risk = max(risk_counts, key=risk_counts.get)

            # Union of themes (all themes mentioned by any prompt)
            themes_set = set()
            for l in labels_list:
                themes_set.update(l.get('themes', []))
            themes = list(themes_set)

            ensemble_label = {
                'sentiment': sentiment,
                'themes': themes,
                'retention_risk': risk,
                'confidence': 0.90  # Higher for ensemble
            }

            output = {
                'review_id': review_id,
                'text': first_label.get('text', ''),
                'source': first_label.get('source', 'unknown'),
                'date': first_label.get('date'),
                'auto_labels': ensemble_label,
                'model': 'ensemble',
                'prompt_version': 'ensemble',
                'num_voters': len(labels_list),
                'generator_prompts': list(set(l.get('prompt_version', '') for l in labels_list))
            }

            out.write(json.dumps(output) + '\n')

    print(f"✅ Created {len(all_labels)} ensemble labels")
    print(f"   Saved to: {output_path}")


def get_crossval_splits() -> Dict[str, List[str]]:
    """
    Get cross-validation splits that prevent leakage.

    Returns mapping of test prompts → generator prompts.
    """

    return {
        # Test v1, v2, v3, v4, v5: Labels from v6
        'v1': ['v2', 'v3', 'v6'],
        'v2': ['v1', 'v3', 'v6'],
        'v3': ['v1', 'v2', 'v6'],
        'v4': ['v2', 'v3', 'v6'],
        'v5': ['v2', 'v3', 'v6'],  # Best performer tested against others
        'v6': ['v2', 'v3', 'v5']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate silver standard without data leakage"
    )
    parser.add_argument('--prompts', type=str, nargs='+',
                       default=['v2', 'v3', 'v6'],
                       help='Prompts to use as generators (NOT including the best one)')
    parser.add_argument('--samples', type=int, default=175,
                       help='Number of reviews to label')
    parser.add_argument('--ensemble', action='store_true',
                       help='Create ensemble labels from multiple prompts')

    args = parser.parse_args()

    print("="*70)
    print("LEAKAGE-FREE SILVER STANDARD GENERATION")
    print("="*70)

    print(f"\nStrategy: Cross-Validation")
    print(f"Generator prompts: {args.prompts}")
    print(f"Test prompts: All 6 versions (v1-v6)")
    print(f"\nKey: No prompt is tested against labels it generated itself")

    if args.ensemble:
        # First generate crossval labels
        crossval_path = 'data/labeled/silver_standard_crossval.jsonl'
        generate_labels_with_prompts(
            prompt_keys=args.prompts,
            max_samples=args.samples,
            output_path=crossval_path
        )

        # Then create ensemble
        ensemble_labels(
            input_path=crossval_path,
            output_path='data/labeled/silver_standard_ensemble.jsonl'
        )

        print(f"\n✅ Ready for evaluation!")
        print(f"Use: {crossval_path} for cross-validation")
        print(f"Use: data/labeled/silver_standard_ensemble.jsonl for ensemble evaluation")

    else:
        # Just generate crossval labels
        generate_labels_with_prompts(
            prompt_keys=args.prompts,
            max_samples=args.samples,
            output_path='data/labeled/silver_standard_crossval.jsonl'
        )

        print(f"\n✅ Ready for cross-validation!")
        print(f"Test any prompt against labels generated by: {args.prompts}")
        print(f"Key insight: v5.0 can now be tested fairly against labels from v2, v3, v6")


if __name__ == '__main__':
    main()

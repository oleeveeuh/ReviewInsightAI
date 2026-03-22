#!/usr/bin/env python3
"""
Automated labeling system for ReviewInsight AI.

Uses GPT-4o to generate structured labels for employee reviews.
Saves incrementally to allow resuming if interrupted.

Usage:
    python src/labeling/auto_label.py --size 200 --model gpt-4o
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Configuration
LABELING_PROMPT = """You are an expert at analyzing employee sentiment and workplace reviews.

Your task is to analyze the following employee review about Amazon warehouse/fulfillment center work and extract structured labels.

Return ONLY a valid JSON object with these exact keys:
{
  "sentiment": integer from 1 to 5 (1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive),
  "themes": array of theme strings from this list: ["overtime", "pay_benefits", "management", "safety", "career_growth", "workload", "work_life_balance", "training", "culture", "other"],
  "retention_risk": "low", "medium", or "high",
  "confidence": float between 0.0 and 1.0,
  "reasoning": "brief 1-2 sentence explanation for your labels"
}

THEMES:
- overtime: discussions about working extra hours, mandatory overtime, peak season hours
- pay_benefits: wage discussions, bonuses, raises, 401k, insurance, time off
- management: supervisor behavior, communication issues, micromanagement, favoritism
- safety: workplace safety concerns, injuries, equipment, heat, physical demands
- career_growth: promotions, internal mobility, skill development, advancement opportunities
- workload: pace, pressure, quotas, rates, productivity demands
- work_life_balance: work-life harmony, scheduling issues, burnout, personal time
- training: onboarding, learning period, training quality, skill instruction
- culture: workplace environment, team dynamics, diversity, social aspects
- other: anything not fitting above categories

RETENTION RISK:
- low: employee speaks positively, likely to stay
- medium: mixed signals, some complaints but engaged
- high: negative tone, discussing leaving, strong complaints

IMPORTANT:
- Be objective and consistent
- Base sentiment on overall tone, not just isolated complaints
- Consider context (former vs current employee status if known)
- Use confidence to indicate how certain you are
- Return valid JSON only - no markdown, no explanation text"""

API_MODEL = "gpt-4o"
API_TEMPERATURE = 0.1
REQUEST_DELAY = 0.5  # seconds between API calls (rate limiting)
MAX_RETRIES = 3

# Cost estimation (GPT-4o pricing)
INPUT_PRICE_PER_1K = 0.15  # $0.15 per 1M input tokens
OUTPUT_PRICE_PER_1K = 0.60  # $0.60 per 1M output tokens


def estimate_tokens(text: str) -> tuple[int, int]:
    """Estimate input and output tokens for a review.

    Returns: (input_tokens, output_tokens)
    """
    # Input: ~0.75 tokens per word (conservative)
    input_tokens = int(len(text.split()) * 0.85)

    # Output: ~150 tokens for structured JSON response
    output_tokens = 150

    return input_tokens, output_tokens


def estimate_cost(reviews: List[Dict]) -> Dict[str, float]:
    """Estimate total API cost for labeling reviews."""

    total_input = 0
    total_output = 0

    for review in reviews:
        text = review.get('text', '')
        inp, out = estimate_tokens(text)
        total_input += inp
        total_output += out

    # Calculate pricing
    input_cost = (total_input / 1_000_000) * INPUT_PRICE_PER_1K
    output_cost = (total_output / 1_000_000) * OUTPUT_PRICE_PER_1K
    total = input_cost + output_cost

    return {
        'input_tokens': total_input,
        'output_tokens': total_output,
        'estimated_cost': total
    }


def generate_label(client: OpenAI, review_text: str, review_id: str) -> Optional[Dict]:
    """Generate a single label using GPT-4o."""

    prompt = f"""Review to analyze:
{review_text[:3000]}

{LABELING_PROMPT}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=API_TEMPERATURE,
                response_format={"type": "json_object"}
            )

            label_text = response.choices[0].message.content

            # Parse JSON response
            label = json.loads(label_text)

            # Validate structure
            required = ['sentiment', 'themes', 'retention_risk', 'confidence', 'reasoning']
            if not all(k in label for k in required):
                print(f"  Warning: Missing keys in {review_id}: {label.keys()}")
                # Add missing keys with defaults
                for k in required:
                    if k not in label:
                        label[k] = None

            # Normalize sentiment to int
            if isinstance(label.get('sentiment'), float):
                label['sentiment'] = int(label['sentiment'])

            # Normalize themes to list and lowercase
            if isinstance(label.get('themes'), str):
                label['themes'] = [t.lower().strip() for t in label['themes'].split(',')]
            elif not isinstance(label.get('themes'), list):
                label['themes'] = []

            # Normalize retention_risk
            label['retention_risk'] = label.get('retention_risk', 'unknown').lower()

            # Ensure confidence is float
            label['confidence'] = float(label.get('confidence', 0.5))

            return label

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  Error labeling {review_id}: {e}")
            return None

    return None


def load_reviews(input_path: Path, sample_size: Optional[int] = None) -> List[Dict]:
    """Load reviews from JSONL file, optionally sampling."""

    reviews = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    # If sample_size specified, stratified sampling by source
    if sample_size and sample_size < len(reviews):
        print(f"\nSampling {sample_size} reviews from {len(reviews)} total...")

        # Group by source
        by_source = {'glassdoor': [], 'reddit': [], 'youtube': []}
        for r in reviews:
            src = r.get('source', 'unknown')
            if src in by_source:
                by_source[src].append(r)

        # Sample from each source proportionally
        sampled = []
        for src, lst in by_source.items():
            if lst:
                n = max(1, int(len(lst) * sample_size / len(reviews)))
                sampled.extend(lst[:n])

        reviews = sampled
        print(f"  Sampled: glassdoor={len([r for r in sampled if r['source']=='glassdoor'])}, "
              f"reddit={len([r for r in sampled if r['source']=='reddit'])}, "
              f"youtube={len([r for r in sampled if r['source']=='youtube'])}")

    return reviews


def save_incremental(output_path: Path, labels: List[Dict], mode: str = 'a'):
    """Save labels incrementally."""

    with open(output_path, mode, encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')


def print_cost_estimate(cost_info: Dict):
    """Print cost estimate in a nice format."""

    print("\n" + "="*60)
    print("COST ESTIMATE")
    print("="*60)
    print(f"\n  Input tokens:   {cost_info['input_tokens']:,}")
    print(f"  Output tokens:  {cost_info['output_tokens']:,}")
    print(f"\n  Estimated cost:   ${cost_info['estimated_cost']:.2f}")
    print(f"  Pricing: ${INPUT_PRICE_PER_1K}/1M input + ${OUTPUT_PRICE_PER_1K}/1M output")
    print("="*60)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-label reviews using GPT-4o"
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=200,
        help='Number of reviews to label (default: 200)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=API_MODEL,
        help=f'OpenAI model to use (default: {API_MODEL})'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/processed/reviews_final.jsonl',
        help='Input reviews file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/labeled/silver_standard.jsonl',
        help='Output labels file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output file'
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        return 1

    # Load input reviews
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"\nLoading reviews from: {input_path}")
    reviews = load_reviews(input_path, sample_size=args.size)

    if not reviews:
        print("Error: No reviews to label.")
        return 1

    print(f"Loaded {len(reviews)} reviews for labeling")

    # Check for existing output (resume mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_labeled = set()
    if args.resume and output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    label = json.loads(line)
                    already_labeled.add(label['review_id'])
        print(f"Found {len(already_labeled)} existing labels (resume mode)")

    # Estimate cost before starting
    to_label = [r for r in reviews if r['review_id'] not in already_labeled]

    if not to_label:
        print("No new reviews to label.")
        return 0

    cost_info = estimate_cost(to_label)
    print_cost_estimate(cost_info)

    # Ask for confirmation
    print(f"\nReady to label {len(to_label)} reviews.")
    response = input("\nProceed? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        return 0

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Process reviews
    results = []
    failures = []

    print(f"\nProcessing with {API_MODEL}...")
    print("-" * 60)

    for review in tqdm(to_label, desc="Labeling"):
        label = generate_label(client, review['text'], review['review_id'])

        if label:
            label['review_id'] = review['review_id']
            results.append(label)
            save_incremental(output_path, [label], mode='a')
        else:
            failures.append(review['review_id'])

    print("-" * 60)
    print(f"\nCompleted!")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print(f"\nFailed review IDs:")
        for fid in failures[:10]:
            print(f"  - {fid}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    # Print statistics
    print("\n" + "="*60)
    print("LABEL STATISTICS")
    print("="*60)

    if results:
        # Sentiment distribution
        sentiments = [r['sentiment'] for r in results if r['sentiment'] is not None]
        if sentiments:
            print("\nSentiment distribution:")
            for s in range(1, 6):
                count = sentiments.count(s)
                pct = count / len(sentiments) * 100
                bar = '█' * int(pct / 5)
                print(f"  {s} ({count:3}) {pct:5.1f}% {bar}")

        # Theme frequency
        all_themes = []
        for r in results:
            if r.get('themes'):
                all_themes.extend(r['themes'])

        if all_themes:
            print("\nTop themes:")
            from collections import Counter
            theme_counts = Counter(all_themes)
            for theme, count in theme_counts.most_common(10):
                pct = count / len(all_themes) * 100
                print(f"  {theme}: {count} ({pct:5.1f}%)")

        # Retention risk
        risks = [r.get('retention_risk') for r in results if r.get('retention_risk')]
        if risks:
            print("\nRetention risk:")
            for risk in ['low', 'medium', 'high']:
                count = risks.count(risk)
                pct = count / len(risks) * 100
                print(f"  {risk}: {count} ({pct:5.1f}%)")

        # Average confidence
        confidences = [r.get('confidence') for r in results if r.get('confidence') is not None]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"\nAverage confidence: {avg_conf:.2f}")

    return 0


if __name__ == '__main__':
    exit(main())

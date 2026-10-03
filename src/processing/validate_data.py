#!/usr/bin/env python3
"""
Validate merged dataset quality.
Usage: python src/processing/validate_data.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_DATA_PATH = PROCESSED_DIR / 'reviews_final.jsonl'


def validate_merged_data(data_path=DEFAULT_DATA_PATH):
    """Run validation checks on merged dataset"""

    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        return False

    # Load data
    reviews = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    if not reviews:
        print("Error: No reviews found in dataset")
        return False

    print("=" * 50)
    print("DATA VALIDATION CHECKS")
    print("=" * 50)

    # Check 1: Required fields
    print("\n1. Required Fields Check:")
    required_fields = ['review_id', 'text', 'source']
    any_missing = False
    for field in required_fields:
        missing = sum(1 for r in reviews if not r.get(field))
        status = "❌" if missing > 0 else "✅"
        print(f"   {status} {field:15} {missing} missing")
        if missing > 0:
            any_missing = True

    # Check 2: Optional fields coverage
    print("\n2. Optional Fields Coverage:")
    optional_fields = ['rating', 'date', 'year', 'quarter', 'location', 'job_title']
    for field in optional_fields:
        present = sum(1 for r in reviews if r.get(field))
        pct = present / len(reviews) * 100
        print(f"   {field:15} {present:4}/{len(reviews):4} ({pct:5.1f}%)")

    # Check 3: Text quality
    print("\n3. Text Quality:")
    word_counts = [len(r['text'].split()) for r in reviews]
    too_short = sum(1 for r in reviews if len(r['text'].split()) < 20)
    too_long = sum(1 for r in reviews if len(r['text'].split()) > 5000)

    print(f"   Min words:      {min(word_counts)}")
    print(f"   Max words:      {max(word_counts)}")
    print(f"   Avg words:      {sum(word_counts)//len(word_counts)}")
    print(f"   Too short (<20): {too_short}")
    print(f"   Too long (>5000): {too_long}")

    # Check for potential issues
    empty_text = sum(1 for r in reviews if not r['text'].strip())
    if empty_text > 0:
        print(f"   ⚠️  Empty text fields: {empty_text}")

    # Check for non-English content (simple heuristic)
    non_ascii = sum(1 for r in reviews if sum(1 for c in r['text'] if ord(c) > 127) > len(r['text']) * 0.3)
    if non_ascii > 0:
        print(f"   ⚠️  High non-ASCII content: {non_ascii}")

    # Check 4: Date coverage
    print("\n4. Date Coverage:")
    with_dates = [r for r in reviews if r.get('date')]
    without_dates = len(reviews) - len(with_dates)
    pct = len(with_dates) / len(reviews) * 100
    print(f"   With dates:    {len(with_dates)}/{len(reviews)} ({pct:.1f}%)")
    print(f"   Without dates: {without_dates}")

    if with_dates:
        dates = pd.to_datetime([r['date'] for r in with_dates])
        print(f"   Date range:    {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")

        # Check for future dates
        future_dates = sum(1 for r in with_dates if pd.to_datetime(r['date']) > pd.Timestamp.now())
        if future_dates > 0:
            print(f"   ⚠️  Future dates: {future_dates}")

        # Check for very old dates
        old_dates = sum(1 for r in with_dates if pd.to_datetime(r['date']) < pd.Timestamp('2020-01-01'))
        if old_dates > 0:
            print(f"   ⚠️  Pre-2020 dates: {old_dates}")

    # Check 5: Source distribution
    print("\n5. Source Distribution:")
    sources = Counter(r['source'] for r in reviews)
    for source, count in sorted(sources.items()):
        pct = count / len(reviews) * 100
        print(f"   {source:12} {count:4} ({pct:5.1f}%)")

    # Check 6: Rating distribution (for sources that have ratings)
    print("\n6. Rating Distribution:")
    with_ratings = [r for r in reviews if r.get('rating') is not None]
    if with_ratings:
        rating_counts = Counter(r['rating'] for r in with_ratings)
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            pct = count / len(with_ratings) * 100
            bar = "█" * int(pct / 5)
            print(f"   {rating} star: {count:3} ({pct:5.1f}%) {bar}")
        avg = sum(r['rating'] for r in with_ratings) / len(with_ratings)
        print(f"   Average: {avg:.2f}")
    else:
        print("   No ratings found")

    # Check 7: Duplicate detection
    print("\n7. Duplicate Detection:")
    seen_texts = {}
    exact_duplicates = 0
    for r in reviews:
        text_norm = r['text'].lower().strip()
        if text_norm in seen_texts:
            exact_duplicates += 1
        else:
            seen_texts[text_norm] = r['review_id']

    if exact_duplicates > 0:
        print(f"   ⚠️  Exact duplicates: {exact_duplicates}")
    else:
        print("   ✅ No exact duplicates found")

    # Check for near-duplicates (similar first 100 chars)
    text_starts = {}
    near_duplicates = 0
    for r in reviews:
        start = r['text'][:100].lower()
        if start in text_starts:
            near_duplicates += 1
        else:
            text_starts[start] = r['review_id']

    print(f"   Near-duplicates: {near_duplicates}")

    # Check 8: Sample reviews
    print("\n8. Sample Reviews (first from each source):")
    for source in sources.keys():
        sample = next((r for r in reviews if r['source'] == source), None)
        if sample:
            print(f"\n   {source.upper()}:")
            preview = sample['text'][:100].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            print(f"   Rating:  {sample.get('rating', 'N/A')}")
            print(f"   Date:    {sample.get('date', 'N/A')}")
            print(f"   ID:      {sample['review_id']}")

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total reviews: {len(reviews)}")

    warnings = []
    if too_short > 0:
        warnings.append(f"{too_short} reviews too short")
    if without_dates > len(reviews) * 0.5:
        warnings.append("More than 50% missing dates")
    if exact_duplicates > 0:
        warnings.append(f"{exact_duplicates} duplicate reviews")
    if len(with_ratings) < len(reviews) * 0.5:
        warnings.append("Less than 50% have ratings")

    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("\n✅ All checks passed!")

    if any_missing:
        print("\nResult: FAILED - required fields are missing")
        return False
    return True


if __name__ == '__main__':
    validate_merged_data()

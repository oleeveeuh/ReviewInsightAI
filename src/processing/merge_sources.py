#!/usr/bin/env python3
"""
Merge all data sources into unified dataset.
Usage: python src/processing/merge_sources.py
"""

import pandas as pd
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
YOUTUBE_TRANSCRIPTS_DIR = DATA_DIR / "youtube" / "transcripts"
YOUTUBE_VIDEO_LIST = DATA_DIR / "youtube" / "video_list.csv"


def load_glassdoor(csv_path=None):
    """Load and standardize Glassdoor data"""
    if csv_path is None:
        # Try to find the glassdoor CSV
        possible_paths = [
            Path("glassdoor_cleaned (2).csv"),
            Path("glassdoor_cleaned.csv"),
            DATA_DIR.parent / "glassdoor_cleaned (2).csv",
            RAW_DIR / "glassdoor_reviews.csv",
        ]
        for p in possible_paths:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            print("Glassdoor CSV not found, skipping...")
            return []

    df = pd.read_csv(csv_path)

    reviews = []
    for idx, row in df.iterrows():
        # Combine pros and cons for full text
        pros = str(row.get('review_pros', '')).strip()
        cons = str(row.get('review_cons', '')).strip()
        summary = str(row.get('summary', '')).strip()
        advice = str(row.get('advice_to_management', '')).strip()

        # Build combined text
        text_parts = []
        if summary and summary.lower() not in ['nan', 'none']:
            text_parts.append(summary)
        if pros and pros.lower() not in ['nan', 'none']:
            text_parts.append(f"Pros: {pros}")
        if cons and cons.lower() not in ['nan', 'none']:
            text_parts.append(f"Cons: {cons}")
        if advice and advice.lower() not in ['nan', 'none']:
            text_parts.append(f"Advice: {advice}")

        text = " | ".join(text_parts)

        if not text or len(text.split()) < 10:
            continue

        # Parse date
        date_str = row.get('rating_date')
        try:
            if pd.notna(date_str):
                date_obj = pd.to_datetime(date_str)
                date = date_obj.strftime('%Y-%m-%d')
                year = date_obj.year
                quarter = f"Q{(date_obj.month - 1) // 3 + 1}"
            else:
                date = None
                year = None
                quarter = None
        except:
            date = None
            year = None
            quarter = None

        reviews.append({
            'review_id': f"glassdoor_{row.get('review_id', idx)}",
            'text': text,
            'rating': float(row.get('rating_overall')) if pd.notna(row.get('rating_overall')) else None,
            'date': date,
            'year': year,
            'quarter': quarter,
            'location': row.get('employee_location'),
            'job_title': row.get('employee_job_title'),
            'employee_type': row.get('employee_type'),
            'source': 'glassdoor',
            'review_length': len(text.split()),
            'metadata': {
                'culture_rating': row.get('rating_culture_values'),
                'work_life_rating': row.get('rating_work_life'),
                'comp_rating': row.get('rating_compensation_benefits'),
            }
        })

    print(f"Loaded {len(reviews)} Glassdoor reviews")
    return reviews


def load_reddit(jsonl_path=None):
    """Load Reddit data"""
    if jsonl_path is None:
        jsonl_path = RAW_DIR / "reddit_reviews.jsonl"

    if not Path(jsonl_path).exists():
        print(f"Reddit data not found at {jsonl_path}, skipping...")
        return []

    reviews = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    print(f"Loaded {len(reviews)} Reddit samples")
    return reviews


def load_youtube(chunk_words: int = 700, chunk_overlap: int = 100,
                 max_chunks_per_video: int = 5, max_total_chunks: int = 20):
    """Load YouTube transcripts and convert to review format.

    Args:
        chunk_words: Target words per chunk (default: 700)
        chunk_overlap: Word overlap between chunks (default: 100)
        max_chunks_per_video: Max chunks to create per video (default: 5)
        max_total_chunks: Total chunks across all videos (default: 20)
    """
    if not YOUTUBE_VIDEO_LIST.exists():
        print("YouTube video list not found, skipping...")
        return []

    if not YOUTUBE_TRANSCRIPTS_DIR.exists():
        print("YouTube transcripts directory not found, skipping...")
        return []

    # Read video metadata
    videos = []
    with open(YOUTUBE_VIDEO_LIST, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        videos = list(reader)

    reviews = []
    total_chunks_created = 0
    skipped_videos = 0

    for video in videos:
        video_id = video.get('video_id', '').strip()
        title = video.get('title', '').strip()
        channel = video.get('channel', '').strip()

        if not video_id:
            continue

        # Check for transcript
        json_path = YOUTUBE_TRANSCRIPTS_DIR / f"{video_id}.json"
        txt_path = YOUTUBE_TRANSCRIPTS_DIR / f"{video_id}.txt"

        if not json_path.exists():
            continue

        # Guardrail: Skip if we've hit total chunk limit
        if total_chunks_created >= max_total_chunks:
            skipped_videos += 1
            continue

        # Load transcript
        with open(json_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        # Build full text from transcript
        transcript_entries = transcript_data.get('transcript', [])
        text = " ".join([entry.get('text', '') for entry in transcript_entries])

        # Clean text
        text = re.sub(r'\[(?:Music|Applause|Laughter)\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 50:
            continue

        # Get upload date
        upload_date = video.get('upload_date', '').strip()
        try:
            if upload_date:
                date_obj = pd.to_datetime(upload_date)
                date = date_obj.strftime('%Y-%m-%d')
                year = date_obj.year
                quarter = f"Q{(date_obj.month - 1) // 3 + 1}"
            else:
                date = None
                year = None
                quarter = None
        except:
            date = None
            year = None
            quarter = None

        # Chunk long transcripts for LLM processing
        words = text.split()
        total_words = len(words)

        if total_words <= chunk_words:
            # Short video: use as single chunk
            chunks = [(0, total_words, text)]
        else:
            # Long video: split into overlapping chunks (with guardrails)
            chunks = []
            start = 0
            chunk_num = 0
            while start < total_words and chunk_num < max_chunks_per_video:
                end = min(start + chunk_words, total_words)
                chunk_text = " ".join(words[start:end])
                chunks.append((start, end - start, chunk_text, f"chunk_{chunk_num + 1}"))
                start = end - chunk_overlap  # Overlap for context
                chunk_num += 1

            # Guardrail: Warn if truncated
            if start < total_words:
                remaining = total_words - start
                print(f"  ⚠️  {video_id}: Truncated {remaining} words (hit max_chunks={max_chunks_per_video})")

        # Guardrail: Check total chunk limit before adding
        chunks_this_video = len(chunks)
        if total_chunks_created + chunks_this_video > max_total_chunks:
            chunks_this_video = max_total_chunks - total_chunks_created
            if chunks_this_video <= 0:
                skipped_videos += 1
                print(f"  ⚠️  {video_id}: Skipped (hit max_total_chunks={max_total_chunks})")
                continue
            chunks = chunks[:chunks_this_video]
            print(f"  ⚠️  {video_id}: Limited to {chunks_this_video} chunks (hit max_total_chunks)")

        # Create review for each chunk
        for chunk_start, chunk_length, chunk_text, chunk_id in chunks:
            reviews.append({
                'review_id': f"youtube_{video_id}_{chunk_id}" if len(chunks) > 1 else f"youtube_{video_id}",
                'text': chunk_text,
                'rating': None,  # YouTube doesn't have ratings
                'date': date,
                'year': year,
                'quarter': quarter,
                'location': None,
                'job_title': 'Warehouse Employee',
                'source': 'youtube',
                'review_length': chunk_length,
                'metadata': {
                    'video_id': video_id,
                    'title': title,
                    'channel': channel,
                    'employee_status': video.get('employee_status'),
                    'notes': video.get('notes'),
                    'chunk_id': chunk_id if len(chunks) > 1 else None,
                    'chunk_start': chunk_start if len(chunks) > 1 else None,
                    'total_video_words': total_words if len(chunks) > 1 else None,
                }
            })

        total_chunks_created += chunks_this_video

    print(f"Loaded {len(reviews)} YouTube transcript chunks from {len(videos) - skipped_videos} videos")
    if skipped_videos > 0:
        print(f"  ⚠️  Skipped {skipped_videos} videos (guardrail limits)")
    if reviews:
        chunked_count = sum(1 for r in reviews if r['metadata'].get('chunk_id'))
        print(f"  ({chunked_count} chunks from {len(set(r['metadata']['video_id'] for r in reviews))} videos)")
    return reviews


def remove_duplicates(reviews):
    """Remove exact duplicate texts"""
    seen_texts = set()
    unique_reviews = []
    duplicates = 0

    for review in reviews:
        text_normalized = review['text'].lower().strip()
        if text_normalized not in seen_texts:
            seen_texts.add(text_normalized)
            unique_reviews.append(review)
        else:
            duplicates += 1

    print(f"Removed {duplicates} exact duplicates")
    return unique_reviews


def quality_filter(reviews):
    """Remove low-quality reviews"""
    filtered = []
    removed = 0

    for review in reviews:
        text = review['text']

        # Skip if too short
        if len(text.split()) < 20:
            removed += 1
            continue

        # Skip if too long (likely spam/errors)
        if len(text.split()) > 10000:
            removed += 1
            continue

        # Skip if mostly non-alphanumeric
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
        if alnum_ratio < 0.5:
            removed += 1
            continue

        filtered.append(review)

    print(f"Removed {removed} low-quality reviews")
    return filtered


def analyze_dataset(reviews):
    """Generate analysis of the merged dataset"""
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)

    # Source breakdown
    source_counts = Counter(r['source'] for r in reviews)
    print("\nBy source:")
    for source, count in sorted(source_counts.items()):
        pct = count / len(reviews) * 100
        print(f"  {source:12} {count:4} ({pct:5.1f}%)")

    # Rating distribution (Glassdoor only)
    ratings = [r['rating'] for r in reviews if r.get('rating')]
    if ratings:
        print(f"\nRating distribution (n={len(ratings)}):")
        rating_counts = Counter(ratings)
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            pct = count / len(ratings) * 100
            bar = "█" * int(pct / 5)
            print(f"  {rating} star: {count:3} ({pct:5.1f}%) {bar}")

    # Date range
    dates = [r['date'] for r in reviews if r['date']]
    if dates:
        print(f"\nDate range: {min(dates)} to {max(dates)}")

        # Year distribution
        years = [r['year'] for r in reviews if r.get('year')]
        if years:
            year_counts = Counter(years)
            print("\nBy year:")
            for year in sorted(year_counts.keys()):
                count = year_counts[year]
                pct = count / len(reviews) * 100
                print(f"  {year}: {count:4} ({pct:5.1f}%)")

    # Review length stats
    lengths = [r['review_length'] for r in reviews]
    if lengths:
        print(f"\nReview length stats:")
        print(f"  Min:     {min(lengths):4} words")
        print(f"  Max:     {max(lengths):4} words")
        print(f"  Average: {sum(lengths)//len(lengths):4} words")
        print(f"  Median:  {sorted(lengths)[len(lengths)//2]:4} words")

    # Total word count
    total_words = sum(lengths)
    print(f"\nTotal words in dataset: {total_words:,}")


def merge_and_clean():
    """Main merge function"""
    print("=" * 50)
    print("MERGING DATA SOURCES")
    print("=" * 50 + "\n")

    # CSV already imported at module level

    # Load all sources
    glassdoor = load_glassdoor()
    reddit = load_reddit()
    youtube = load_youtube()

    # Combine
    all_reviews = glassdoor + reddit + youtube
    print(f"\nTotal before cleaning: {len(all_reviews)}")

    if len(all_reviews) == 0:
        print("No reviews found!")
        return []

    # Remove duplicates
    all_reviews = remove_duplicates(all_reviews)

    # Quality filter
    all_reviews = quality_filter(all_reviews)

    # Sort by date (oldest first, nulls last)
    all_reviews.sort(key=lambda x: (x['date'] is None, x['date'] or '9999'))

    # Analyze
    analyze_dataset(all_reviews)

    # Save JSONL
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / 'reviews_final.jsonl'

    with open(output_path, 'w', encoding='utf-8') as f:
        for review in all_reviews:
            f.write(json.dumps(review, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_path}")

    # Also save as CSV
    csv_path = PROCESSED_DIR / 'reviews_final.csv'
    df_data = []
    for r in all_reviews:
        df_data.append({
            'review_id': r['review_id'],
            'source': r['source'],
            'text': r['text'][:500] + '...' if len(r['text']) > 500 else r['text'],
            'rating': r.get('rating'),
            'date': r['date'],
            'year': r['year'],
            'quarter': r['quarter'],
            'location': r['location'],
            'job_title': r['job_title'],
            'review_length': r['review_length'],
        })

    pd.DataFrame(df_data).to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # Save merge report
    report_path = PROCESSED_DIR / 'merge_report.txt'
    with open(report_path, 'w') as f:
        f.write("DATA MERGE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Merge date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"SOURCES:\n")
        source_counts = Counter(r['source'] for r in all_reviews)
        for source, count in sorted(source_counts.items()):
            f.write(f"  - {source}: {count}\n")

        f.write(f"\nTOTAL REVIEWS: {len(all_reviews)}\n")

        dates = [r['date'] for r in all_reviews if r['date']]
        if dates:
            f.write(f"DATE RANGE: {min(dates)} to {max(dates)}\n")

        ratings = [r['rating'] for r in all_reviews if r.get('rating')]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            f.write(f"AVERAGE RATING: {avg_rating:.2f}\n")

    print(f"Report saved to: {report_path}")

    return all_reviews


if __name__ == '__main__':
    reviews = merge_and_clean()

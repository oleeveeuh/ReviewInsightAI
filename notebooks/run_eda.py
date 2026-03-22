#!/usr/bin/env python3
"""
Run Exploratory Data Analysis on merged dataset.
This is a standalone Python script version of the EDA notebook.
Usage: python notebooks/run_eda.py
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load data
    reviews = []
    with open(DATA_DIR / 'reviews_final.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    df = pd.DataFrame(reviews)

    print(f"\nTotal reviews: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Convert dates
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df_with_dates = df[df['date_parsed'].notna()].copy()

    # ==================== SOURCE DISTRIBUTION ====================
    print("\n" + "=" * 60)
    print("1. SOURCE DISTRIBUTION")
    print("=" * 60)

    source_counts = df['source'].value_counts()
    print(source_counts)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    source_counts.plot(kind='pie', ax=ax[0], autopct='%1.1f%%',
                       startangle=90, colors=colors[:len(source_counts)])
    ax[0].set_ylabel('')
    ax[0].set_title('Reviews by Source')

    source_counts.plot(kind='bar', ax=ax[1], color=colors[:len(source_counts)])
    ax[1].set_xlabel('Source')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Review Count by Source')
    ax[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'source_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: source_distribution.png")

    # ==================== TEMPORAL ANALYSIS ====================
    print("\n" + "=" * 60)
    print("2. TEMPORAL ANALYSIS")
    print("=" * 60)

    if len(df_with_dates) > 0:
        print(f"Date range: {df_with_dates['date_parsed'].min().strftime('%Y-%m-%d')} to "
              f"{df_with_dates['date_parsed'].max().strftime('%Y-%m-%d')}")

        df_with_dates['year_month'] = df_with_dates['date_parsed'].dt.to_period('M')
        monthly_counts = df_with_dates.groupby('year_month').size()

        plt.figure(figsize=(14, 5))
        monthly_counts.plot(kind='line', marker='o', linewidth=2, markersize=8)
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.title('Reviews Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'reviews_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: reviews_over_time.png")

        # Yearly distribution
        year_counts = df_with_dates['year'].value_counts().sort_index()
        print("\nReviews by year:")
        for year, count in year_counts.items():
            print(f"  {int(year)}: {count} ({count/len(df_with_dates)*100:.1f}%)")

    # ==================== RATING ANALYSIS ====================
    print("\n" + "=" * 60)
    print("3. RATING ANALYSIS (Glassdoor)")
    print("=" * 60)

    df_glassdoor = df[df['source'] == 'glassdoor'].copy()
    df_with_rating = df_glassdoor[df_glassdoor['rating'].notna()]

    if len(df_with_rating) > 0:
        print(f"Glassdoor reviews with ratings: {len(df_with_rating)}")
        print(f"Average rating: {df_with_rating['rating'].mean():.2f}")
        print(f"Median rating: {df_with_rating['rating'].median():.1f}")

        rating_dist = df_with_rating['rating'].value_counts().sort_index()
        print("\nRating distribution:")
        for rating, count in rating_dist.items():
            pct = count / len(df_with_rating) * 100
            bar = '█' * int(pct / 5)
            print(f"  {rating} star: {count:3} ({pct:5.1f}%) {bar}")

        plt.figure(figsize=(10, 5))
        colors_rating = ['#d62728', '#ff7f0e', '#ffcc00', '#2ca02c', '#1f77b4']
        rating_dist.plot(kind='bar', color=colors_rating)
        plt.xlabel('Rating (Stars)')
        plt.ylabel('Count')
        plt.title('Glassdoor Rating Distribution')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: rating_distribution.png")

    # ==================== TEXT ANALYSIS ====================
    print("\n" + "=" * 60)
    print("4. TEXT ANALYSIS")
    print("=" * 60)

    all_text = ' '.join(df['text'].tolist())
    print(f"Total word count: {len(all_text.split()):,}")

    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
                     'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they', 'it',
                     'from', 'by', 'as', 'not', 'just', 'like', 'get', 'got', 'can'])

    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter([w for w in words if w not in stopwords and len(w) > 3])

    print("\nTop 20 most common words:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        pct = count / sum(word_freq.values()) * 100
        print(f"  {i:2}. {word:15} {count:4} ({pct:4.1f}%)")

    # Theme analysis
    themes = {
        'Pay & Benefits': r'\b(pay|salary|wage|benefit|insurance|pto|vacation|raise|bonus|hourly|dollar)\b',
        'Management': r'\b(manager|management|supervisor|leadership|boss|lead|ceo)\b',
        'Safety': r'\b(safety|safe|unsafe|injury|dangerous|hazard|accident)\b',
        'Work-Life Balance': r'\b(balance|life|family|schedule|shift|overtime|ot)\b',
        'Physical Toll': r'\b(feet|hurt|pain|tired|exhausted|sore|standing|stand|body|physic)\b',
        'Peak Season': r'\b(peak|season|holiday|christmas|prime day|q4)\b',
        'Coworkers': r'\b(coworker|team|people|friend|help|together|culture)\b',
        'Workload': r'\b(rate|quota|packages|boxes|scanner|items|workload|fast|slow|target)\b'
    }

    print("\n=== THEME FREQUENCY ===")
    theme_results = []
    for theme, pattern in themes.items():
        count = sum(1 for text in df['text'] if re.search(pattern, text, re.IGNORECASE))
        pct = count / len(df) * 100
        theme_results.append((theme, count, pct))
        print(f"{theme:20} {count:3} reviews ({pct:5.1f}%)")

    # Visualize themes
    theme_df = pd.DataFrame(theme_results, columns=['Theme', 'Count', 'Percentage'])
    theme_df = theme_df.sort_values('Count', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(theme_df['Theme'], theme_df['Count'], color='steelblue')
    plt.xlabel('Number of Reviews Mentioning Theme')
    plt.title('Theme Frequency Across Reviews')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'theme_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: theme_frequency.png")

    # ==================== REVIEW LENGTH ====================
    print("\n" + "=" * 60)
    print("5. REVIEW LENGTH ANALYSIS")
    print("=" * 60)

    print(df['review_length'].describe())

    plt.figure(figsize=(12, 5))
    plt.hist(df['review_length'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Review Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Review Lengths')
    plt.axvline(df['review_length'].median(), color='red', linestyle='--',
               label=f"Median: {df['review_length'].median():.0f} words")
    plt.axvline(df['review_length'].mean(), color='orange', linestyle='--',
               label=f"Mean: {df['review_length'].mean():.0f} words")
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'review_length_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: review_length_dist.png")

    print("\nAverage length by source:")
    length_by_source = df.groupby('source')['review_length'].agg(['mean', 'median', 'count'])
    length_by_source = length_by_source.sort_values('mean', ascending=False)
    print(length_by_source.round(1))

    # ==================== KEY OBSERVATIONS ====================
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)

    observations = []

    observations.append(f"1. Dataset: {len(df)} total reviews from {df['source'].nunique()} sources")

    if len(df_with_dates) > 0:
        date_range = (df_with_dates['date_parsed'].max() - df_with_dates['date_parsed'].min()).days
        observations.append(f"2. Temporal coverage: {date_range} days ({len(df_with_dates)} reviews with dates)")

    if len(df_with_rating) > 0:
        avg_rating = df_with_rating['rating'].mean()
        observations.append(f"3. Average Glassdoor rating: {avg_rating:.2f}/5.0")

    top_themes = sorted(theme_results, key=lambda x: x[1], reverse=True)
    observations.append(f"4. Top mentioned theme: '{top_themes[0][0]}' ({top_themes[0][1]} reviews, {top_themes[0][2]:.1f}%)")

    observations.append(f"5. Average review length: {df['review_length'].mean():.0f} words")

    glassdoor_pct = len(df[df['source'] == 'glassdoor']) / len(df) * 100
    youtube_pct = len(df[df['source'] == 'youtube']) / len(df) * 100
    observations.append(f"6. Source mix: {glassdoor_pct:.1f}% Glassdoor, {youtube_pct:.1f}% YouTube")

    for obs in observations:
        print(obs)

    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print(f"All figures saved to: {REPORTS_DIR}")
    print("=" * 60)

    # Save summary to markdown
    summary_path = Path(__file__).parent.parent / "reports" / "eda_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# EDA Summary Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Dataset Overview\n\n")
        for obs in observations:
            f.write(f"- {obs}\n")
        f.write("\n## Theme Frequency\n\n")
        f.write("| Theme | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for theme, count, pct in theme_results:
            f.write(f"| {theme} | {count} | {pct:.1f}% |\n")
        f.write("\n## Generated Figures\n\n")
        for fig_file in REPORTS_DIR.glob("*.png"):
            f.write(f"- `{fig_file.name}`\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()

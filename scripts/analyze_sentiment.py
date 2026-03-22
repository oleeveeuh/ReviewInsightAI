#!/usr/bin/env python3
"""
Analyze sentiment in YouTube transcripts.
Usage: python scripts/analyze_sentiment.py
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRANSCRIPTS_DIR = DATA_DIR / "youtube" / "transcripts"
VIDEO_LIST = DATA_DIR / "youtube" / "video_list.csv"
OUTPUT_DIR = DATA_DIR / "youtube" / "analysis"

# Sentiment word lists - workplace focused
POSITIVE_WORDS = {
    # Good aspects of work
    "easy", "good", "great", "love", "like", "enjoy", "happy", "nice", "best",
    "amazing", "awesome", "excellent", "perfect", "fun", "exciting",
    # Amazon/benefits specific
    "benefits", "benefit", "bonus", "pay", "paid", "money", "raise", "promotion",
    "flexible", "schedule", "overtime", "time off", "break", "lunch",
    # Environment
    "clean", "safe", "comfortable", "modern", "new", "diverse",
    # Social
    "friendly", "helpful", "team", "support", "friends", "people",
    # Experience
    "learn", "learning", "opportunity", "growth", "advance",
    # Positive emotions
    "thank", "thanks", "appreciate", "recommend", "yes",
    # General positive
    "better", "improving", "smooth", "quick", "fast", "efficient",
}

NEGATIVE_WORDS = {
    # Bad aspects of work
    "hard", "difficult", "tired", "exhausting", "stress", "stressful", "hate",
    "bad", "worst", "terrible", "horrible", "awful", "suck", "sucks",
    # Physical toll
    "hurt", "pain", "feet", "back", "legs", "ache", "sore", "tired",
    "standing", "stand", "no chairs", "nowhere to sit",
    # Management
    "manager", "supervisor", "boss", "rude", "mean", "unfair", "favoritism",
    # Work conditions
    "hot", "cold", "dirty", "messy", "crowded", "slow", "boring",
    "quit", "quit right", "walk out", "leave", "leaving",
    # Schedule issues
    "long hours", "overtime", "mandatory", "forced", "short staff",
    # Negative emotions
    "frustrat", "annoying", "upset", "angry", "mad", "cry", "crying",
    "don't like", "can't stand", "not worth", "regret",
    # General negative
    "problem", "issue", "complaint", "wrong", "error", "fail", "failure",
}

def clean_text(text):
    """Clean transcript text for analysis."""
    # Remove [Music] and similar bracketed content
    text = re.sub(r'\[(?:Music|Applause|Laughter)\]', '', text)
    # Remove timestamps
    text = re.sub(r'\[\d+\.\d+\]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def analyze_sentiment(text):
    """
    Simple lexicon-based sentiment analysis.
    Returns sentiment score and classification.
    """
    words = text.split()
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    # Calculate normalized score (-1 to 1)
    total = positive_count + negative_count
    if total == 0:
        return 0, "neutral"

    score = (positive_count - negative_count) / total

    if score > 0.15:
        classification = "positive"
    elif score < -0.15:
        classification = "negative"
    else:
        classification = "neutral"

    return round(score, 3), classification

def extract_themes(text):
    """Extract common themes from transcript."""
    words = clean_text(text).split()

    themes = {
        "pay/benefits": 0,
        "schedule/hours": 0,
        "management": 0,
        "physical toll": 0,
        "workload": 0,
        "environment": 0,
        "coworkers": 0,
    }

    theme_keywords = {
        "pay/benefits": ["pay", "money", "wage", "salary", "benefit", "bonus", "raise", "dollar", "hourly"],
        "schedule/hours": ["schedule", "hours", "shift", "overtime", "weekend", "day off", "break", "lunch"],
        "management": ["manager", "supervisor", "boss", "lead", "management", "hr", "leadership"],
        "physical toll": ["feet", "hurt", "pain", "tired", "exhausted", "sore", "back", "legs", "stand", "standing"],
        "workload": ["fast", "slow", "rate", "quota", "packages", "boxes", "scanner", "items", "workload"],
        "environment": ["hot", "cold", "clean", "dirty", "safe", "safety", "warehouse", "facility"],
        "coworkers": ["coworker", "team", "people", "friend", "help", "together"],
    }

    for word in words:
        for theme, keywords in theme_keywords.items():
            if word in keywords:
                themes[theme] += 1

    return themes

def analyze_transcript(video_id, video_title):
    """Analyze a single transcript file."""
    txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    json_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    if not txt_path.exists():
        return None

    # Read transcript
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Get word count
    words = clean_text(text).split()
    word_count = len(words)

    # Analyze sentiment
    score, classification = analyze_sentiment(text)

    # Extract themes
    themes = extract_themes(text)

    # Find top themes
    top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "video_id": video_id,
        "title": video_title,
        "word_count": word_count,
        "sentiment_score": score,
        "sentiment": classification,
        "themes": dict(themes),
        "top_themes": top_themes,
    }

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read video list
    videos = []
    with open(VIDEO_LIST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        videos = list(reader)

    print(f"Analyzing {len(videos)} videos...\n")

    results = []
    classifications = []

    for video in videos:
        video_id = video["video_id"].strip()
        title = video["title"]

        if not video_id:
            continue

        result = analyze_transcript(video_id, title)
        if result:
            results.append(result)
            classifications.append(result["sentiment"])
            print(f"✓ {title[:40]:40} | {result['sentiment']:8} | score: {result['sentiment_score']:+.2f}")

    # Summary statistics
    if results:
        total_words = sum(r["word_count"] for r in results)
        avg_score = sum(r["sentiment_score"] for r in results) / len(results)
        sentiment_dist = Counter(classifications)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Videos analyzed:     {len(results)}")
        print(f"Total words:         {total_words:,}")
        print(f"Avg sentiment score: {avg_score:+.2f}")
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.most_common():
            pct = count / len(results) * 100
            print(f"  {sentiment:8} {count:2} ({pct:.0f}%)")

        # Aggregate themes
        all_themes = {}
        for r in results:
            for theme, count in r["themes"].items():
                all_themes[theme] = all_themes.get(theme, 0) + count

        print(f"\nTop Themes Across All Videos:")
        for theme, count in sorted(all_themes.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {theme:15} {count:3} mentions")

        # Save detailed results
        results_path = OUTPUT_DIR / "sentiment_analysis.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_videos": len(results),
                    "total_words": total_words,
                    "avg_sentiment_score": avg_score,
                    "sentiment_distribution": dict(sentiment_dist),
                    "top_themes": dict(sorted(all_themes.items(), key=lambda x: x[1], reverse=True)),
                },
                "videos": results,
            }, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

        # Save CSV summary
        csv_path = OUTPUT_DIR / "sentiment_summary.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_id", "title", "word_count", "sentiment_score",
                "sentiment", "top_theme_1", "top_theme_2", "top_theme_3"
            ])
            for r in results:
                top_themes_str = [f"{t}({c})" for t, c in r["top_themes"]]
                while len(top_themes_str) < 3:
                    top_themes_str.append("")
                writer.writerow([
                    r["video_id"], r["title"], r["word_count"],
                    r["sentiment_score"], r["sentiment"],
                    *top_themes_str
                ])
        print(f"CSV summary saved to: {csv_path}")

    return 0

if __name__ == "__main__":
    main()

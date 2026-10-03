#!/usr/bin/env python3
"""
CLI tool for manually labeling reviews to create a gold standard dataset.
Usage: python src/labeling/label_reviews.py

Features:
- Incremental saving - progress is preserved
- Resume anytime - tracks labeled review IDs
- Stratified sampling by source
- Progress tracking
"""

import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd

# Paths
DEFAULT_REVIEWS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "reviews_final.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "labeled" / "gold_standard.jsonl"


class ReviewLabeler:
    """Interactive CLI tool for labeling reviews."""

    def __init__(self, reviews_path=None, output_path=None, target_count=100):
        self.reviews_path = Path(reviews_path or DEFAULT_REVIEWS_PATH)
        self.output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
        self.target_count = target_count

        # Load reviews
        print(f"Loading reviews from {self.reviews_path}...")
        self.reviews = []
        with open(self.reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.reviews.append(json.loads(line))

        print(f"Loaded {len(self.reviews)} reviews")

        # Load existing labels if any
        self.labeled = self.load_existing_labels()
        self.labeled_ids = set(r['review_id'] for r in self.labeled)

        # Sample reviews for labeling
        self.to_label = self.sample_reviews()
        self.current_index = 0

    def load_existing_labels(self):
        """Load previously labeled reviews."""
        if self.output_path.exists():
            labeled = []
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        labeled.append(json.loads(line))
            print(f"Found {len(labeled)} previously labeled reviews")
            return labeled
        return []

    def sample_reviews(self):
        """Stratified sample of reviews to label."""
        # Get unlabeled reviews
        unlabeled = [r for r in self.reviews if r['review_id'] not in self.labeled_ids]

        # How many more do we need?
        remaining = self.target_count - len(self.labeled)

        if remaining <= 0:
            print(f"✅ Already have {len(self.labeled)} labeled reviews!")
            return []

        print(f"Need to label {remaining} more reviews")

        # Stratified sampling by source
        sample = []

        # Get proportion from each source
        df = pd.DataFrame(unlabeled)
        source_counts = df['source'].value_counts()

        for source in source_counts.index:
            source_reviews = [r for r in unlabeled if r['source'] == source]
            n_from_source = max(1, int(remaining * (source_counts[source] / len(unlabeled))))
            n_from_source = min(n_from_source, len(source_reviews))

            # Random sample from this source
            sampled = random.sample(source_reviews, n_from_source)
            sample.extend(sampled)

        # Shuffle and trim to exact number needed
        random.shuffle(sample)
        return sample[:remaining]

    def save_label(self, review, labels):
        """Save a labeled review."""
        labeled_review = {
            **review,
            'labels': labels,
            'labeled_by': 'human',
            'labeled_date': datetime.now().isoformat()
        }

        # Rewrite entire file (to handle skips/backs properly)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Add to labeled list
        # Check if this review is already in labeled
        already_labeled = False
        for i, r in enumerate(self.labeled):
            if r['review_id'] == review['review_id']:
                self.labeled[i] = labeled_review
                already_labeled = True
                break

        if not already_labeled:
            self.labeled.append(labeled_review)
            self.labeled_ids.add(review['review_id'])

        # Write all labeled reviews
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for r in self.labeled:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def display_review(self, review):
        """Display a review for labeling."""
        print("\n" + "=" * 80)
        print(f"  Review {len(self.labeled) + 1}/{self.target_count}")
        print("=" * 80)

        # Metadata
        meta_parts = []
        meta_parts.append(f"Source: {review['source']}")
        if review.get('date'):
            meta_parts.append(f"Date: {review['date']}")
        if review.get('rating'):
            meta_parts.append(f"Original Rating: {review['rating']}")
        if review.get('location'):
            meta_parts.append(f"Location: {review['location']}")

        print("  |  ".join(meta_parts))
        print("-" * 80)

        # Text (chunked for readability)
        text = review['text']
        if len(text) > 800:
            # Show first part, indicate truncation
            print(text[:800])
            print(f"\n... ({len(text) - 800} more characters)")
        else:
            print(text)

        print("-" * 80)

    def get_sentiment_label(self):
        """Get sentiment label from user."""
        print("\n1️⃣  OVERALL SENTIMENT")
        print("   " + "-" * 70)
        print("   Based on the ENTIRE review, how would you rate the sentiment?")
        print()
        print("   1 = Very Negative    (Strong criticism, wants to quit)")
        print("   2 = Negative         (Mostly complaints, disappointed)")
        print("   3 = Neutral          (Balanced pros/cons, factual)")
        print("   4 = Positive         (Mostly praise, satisfied)")
        print("   5 = Very Positive    (Enthusiastic, highly recommended)")
        print()

        while True:
            try:
                response = input("   Enter sentiment (1-5): ").strip()
                if response == '':
                    response = '3'  # default to neutral
                sentiment = int(response)
                if 1 <= sentiment <= 5:
                    return sentiment
                print("   ⚠️  Please enter a number 1-5")
            except ValueError:
                print("   ⚠️  Please enter a valid number")

    def get_theme_labels(self):
        """Get theme labels from user."""
        print("\n2️⃣  THEMES (Select ALL that apply)")
        print("   " + "-" * 70)
        print("   Enter letters separated by commas (e.g., 'a,c,g')")
        print("   Press Enter to skip (no themes)")
        print()
        print("   a) 📅 Overtime / Scheduling issues")
        print("   b) 💰 Pay / Benefits / Compensation")
        print("   c) 👔 Management / Leadership")
        print("   d) ⚠️  Safety / Working conditions")
        print("   e) 📈 Career Growth / Promotion opportunities")
        print("   f) 📦 Workload / Pace / Rate pressure")
        print("   g) ⚖️  Work-Life Balance / Shift hours")
        print("   h) 🎓 Training / Onboarding")
        print("   i) 👥 Coworkers / Culture")
        print("   j) 📝 Other (specify in notes)")
        print()

        theme_map = {
            'a': 'overtime',
            'b': 'pay_benefits',
            'c': 'management',
            'd': 'safety',
            'e': 'career_growth',
            'f': 'workload',
            'g': 'work_life_balance',
            'h': 'training',
            'i': 'culture',
            'j': 'other'
        }

        while True:
            response = input("   Enter themes: ").strip().lower()
            if response == '':
                return []  # No themes

            selected = [t.strip() for t in response.split(',')]
            invalid = [t for t in selected if t not in theme_map]

            if invalid:
                print(f"   ⚠️  Invalid selections: {', '.join(invalid)}")
                print("      Please use letters a-j")
                continue

            return [theme_map[t] for t in selected]

    def get_retention_risk(self):
        """Get retention risk label from user."""
        print("\n3️⃣  RETENTION RISK")
        print("   " + "-" * 70)
        print("   Based on this review, how likely is the employee to leave?")
        print()
        print("   a) 🟢 Low  - Satisfied or neutral, likely to stay")
        print("   b) 🟡 Medium - Has complaints but manageable")
        print("   c) 🔴 High  - Very dissatisfied, likely to quit or has quit")
        print()

        while True:
            response = input("   Enter risk level (a/b/c): ").strip().lower()
            if response == '':
                response = 'a'  # default to low

            risk_map = {'a': 'low', 'b': 'medium', 'c': 'high'}
            if response in risk_map:
                return risk_map[response]

            print("   ⚠️  Please enter a, b, or c")

    def get_notes(self):
        """Optional notes from labeler."""
        print("\n4️⃣  NOTES (optional)")
        print("   " + "-" * 70)
        notes = input("   Any additional notes? Press Enter to skip: ").strip()
        return notes

    def label_review(self, review):
        """Interactive labeling for one review."""
        self.display_review(review)

        labels = {}

        # Get each label
        labels['sentiment'] = self.get_sentiment_label()
        labels['themes'] = self.get_theme_labels()
        labels['retention_risk'] = self.get_retention_risk()
        notes = self.get_notes()
        if notes:
            labels['notes'] = notes

        return labels

    def show_progress(self):
        """Show current progress."""
        labeled_count = len(self.labeled)
        remaining = self.target_count - labeled_count
        pct = (labeled_count / self.target_count) * 100

        bar_length = 40
        filled = int(bar_length * labeled_count / self.target_count)
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f"\n   Progress: [{bar}] {pct:.1f}%")
        print(f"   Labeled: {labeled_count}/{self.target_count}")
        print(f"   Remaining: {remaining}")

    def print_summary(self):
        """Print summary statistics of labeled data."""
        if not self.labeled:
            return

        df = pd.DataFrame([r['labels'] for r in self.labeled])

        print("\n" + "=" * 60)
        print("  📊 LABELING SUMMARY")
        print("=" * 60)

        # Sentiment distribution
        print("\n  Sentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        for s, c in sentiment_counts.items():
            label = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos'][s - 1]
            bar = '█' * (c * 40 // len(self.labeled))
            print(f"    {s} ({label}): {c:2} {bar}")

        # Theme frequency
        print("\n  Theme Frequency:")
        all_themes = []
        for themes in df['themes']:
            all_themes.extend(themes)
        theme_counts = Counter(all_themes)
        for theme, count in theme_counts.most_common():
            pct = count / len(self.labeled) * 100
            print(f"    {theme:20} {count:2} ({pct:5.1f}%)")

        # Risk distribution
        print("\n  Retention Risk Distribution:")
        risk_counts = df['retention_risk'].value_counts()
        for risk in ['low', 'medium', 'high']:
            count = risk_counts.get(risk, 0)
            label = {'low': '🟢 Low', 'medium': '🟡 Medium', 'high': '🔴 High'}[risk]
            print(f"    {label}: {count}")

    def run(self):
        """Main labeling loop."""
        if not self.to_label:
            print("\n✅ All target reviews already labeled!")
            self.print_summary()
            return

        print("\n" + "=" * 60)
        print("  🏷️  REVIEW LABELING TOOL")
        print("=" * 60)
        print(f"\n  Target: {self.target_count} labels")
        print(f"  Already labeled: {len(self.labeled)}")
        print(f"  To label: {len(self.to_label)}")
        print("\n  Commands:")
        print("    [Enter] - Label this review")
        print("    [s]     - Skip this review")
        print("    [q]     - Quit and save progress")
        print("    [b]     - Go back to previous review")
        print("    [p]     - Show progress summary")
        print("    [?]     - Show help")
        print()

        i = 0
        while i < len(self.to_label) and len(self.labeled) < self.target_count:
            review = self.to_label[i]

            self.show_progress()

            # Get command
            cmd = input("\n  Command? ").strip().lower()

            if cmd == 'q':
                print("\n  💾 Saving progress and exiting...")
                break
            elif cmd == 's':
                print("  ⏭️  Skipping...")
                i += 1
                continue
            elif cmd == 'b' and i > 0:
                print("  ⏮️  Going back...")
                i -= 1
                continue
            elif cmd == 'p':
                self.print_summary()
                continue
            elif cmd == '?':
                print("\n  Labeling Guide:")
                print("    - Sentiment: Consider the ENTIRE review tone")
                print("    - Themes: Select ALL that apply (multi-select)")
                print("    - Retention Risk: Focus on severity of dissatisfaction")
                print("    - A 3-star rating can still be negative sentiment if tone is critical")
                continue
            elif cmd in ['h', 'help']:
                print("\n  Commands:")
                print("    [Enter] - Label this review")
                print("    [s]     - Skip this review")
                print("    [q]     - Quit and save progress")
                print("    [b]     - Go back to previous review")
                print("    [p]     - Show progress summary")
                continue

            # Label the review
            try:
                labels = self.label_review(review)
                self.save_label(review, labels)

                print(f"\n  ✅ Saved! ({len(self.labeled)}/{self.target_count})")
                i += 1
            except KeyboardInterrupt:
                print("\n\n  💾 Interrupted. Saving progress...")
                break
            except Exception as e:
                print(f"\n  ⚠️  Error: {e}")
                print("  Skipping this review...")
                i += 1

        print("\n\n  🎉 Session complete!")
        print(f"  Total labeled: {len(self.labeled)}/{self.target_count}")
        print(f"  Saved to: {self.output_path}")

        self.print_summary()


def main():
    """Entry point for the labeling tool."""
    import argparse

    parser = argparse.ArgumentParser(description="Label reviews for sentiment analysis")
    parser.add_argument('--target', '-t', type=int, default=100,
                       help='Target number of reviews to label (default: 100)')
    parser.add_argument('--reviews', '-r', type=str,
                       help='Path to reviews JSONL file')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to output labeled JSONL file')

    args = parser.parse_args()

    labeler = ReviewLabeler(
        reviews_path=args.reviews,
        output_path=args.output,
        target_count=args.target
    )

    try:
        labeler.run()
    except KeyboardInterrupt:
        print("\n\n  💾 Saving and exiting...")
        labeler.print_summary()


if __name__ == '__main__':
    main()

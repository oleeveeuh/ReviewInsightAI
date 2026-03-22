#!/usr/bin/env python3
"""
Label verification tool for ReviewInsight AI.

Interactive CLI to spot-check auto-generated labels against human judgment.
Ensures LLM labels are trustworthy before using for model training.

Usage:
    python src/labeling/verify_labels.py --sample 30
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from readchar import readchar, render_ansored
except ImportError:
    # Fallback if termcap not available
    readchar = None
    render_ansored = lambda text, **kwargs: text


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'     # Magenta
    OKBLUE = '\033[94m'    # Blue
    OKGREEN = '\033[92m'    # Green
    WARNING = '\033[93m'    # Yellow
    FAIL = '\033[91m'     # Red
    RESET = '\033[0m'      # Reset
    BOLD = '\033[1m'      # Bold


def truncate_text(text: str, max_len: int = 400, suffix: str = '...') -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


class LabelVerifier:
    """Interactive label verification tool."""

    THEMES = [
        "overtime", "pay_benefits", "management", "safety",
        "career_growth", "workload", "work_life_balance", "training",
        "culture", "other"
    ]

    RETENTION_RISKS = ["low", "medium", "high"]
    SENTIMENTS = [1, 2, 3, 4, 5]

    def __init__(self, silver_standard_path: Path):
        """Initialize verifier."""
        self.silver_path = silver_standard_path
        self.verification_path = silver_standard_path.parent / 'silver_standard_verified.jsonl'
        self.reviews = {}
        self.verified_count = 0
        self.agree_count = 0
        self.disagree_count = 0
        self.skipped_count = 0

        # Load silver standard labels
        self._load_silver_labels()

        # Load existing verifications if any
        self._load_verifications()

    def _load_silver_labels(self):
        """Load auto-generated labels."""
        if not self.silver_path.exists():
            print(f"{Colors.WARNING}Warning: Silver standard not found: {self.silver_path}")
            return

        with open(self.silver_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    label = json.loads(line)
                    self.reviews[label['review_id']] = label

        print(f"{Colors.OKBLUE}Loaded {len(self.reviews)} auto-generated labels")

    def _load_verifications(self):
        """Load existing human verifications."""
        if self.verification_path.exists():
            with open(self.verification_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # Update our tracking
                        if record.get('human_verified'):
                            review_id = record.get('review_id')
                            if review_id in self.reviews:
                                self.reviews[review_id]['human_verified'] = record['human_verified']
                                if record.get('auto_labels'):
                                    self.reviews[review_id]['auto_labels'] = record['auto_labels']

        # Count existing verifications
        existing = sum(1 for r in self.reviews.values() if r.get('human_verified'))
        if existing > 0:
            self.verified_count = existing
            print(f"{Colors.OKBLUE}Resuming with {existing} existing verifications")

    def _display_review(self, review: Dict, labels: Dict):
        """Display a review and its labels for verification."""

        print(f"\n{Colors.BOLD}{Colors.HEADER}═{'═'*60}{Colors.RESET}")

        # Review metadata
        review_id = review['review_id']
        source = review.get('source', 'unknown')
        date = review.get('date', 'unknown')
        rating = review.get('rating')
        if rating is None:
            rating_display = "N/A"
        else:
            rating_display = f"{rating}/5"

        print(f"{Colors.BOLD}Review ID:{Colors.RESET} {review_id}")
        print(f"{Colors.BOLD}Source:{Colors.RESET}     {source}")
        print(f"{Colors.BOLD}Date:{Colors.RESET}       {date}")
        print(f"{Colors.BOLD}Rating:{Colors.RESET}      {rating_display}")

        # Review text (truncated)
        text = review.get('text', '')
        display_text = truncate_text(text.replace('\n', ' '), 400)
        print(f"\n{Colors.BOLD}Text:{Colors.RESET}")
        print(f"  {display_text}")

        # Auto labels
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}AUTO LABELS:{Colors.RESET}")

        sentiment = labels.get('sentiment')
        if sentiment is not None:
            sentiment_display = f"{sentiment}/5"
            star = '★' * sentiment
            print(f"  Sentiment: {star} ({sentiment_display})")
        else:
            print(f"  Sentiment: {Colors.WARNING}N/A")

        themes = labels.get('themes', [])
        if themes:
            print(f"  Themes:   {', '.join(themes[:5])}")
            if len(themes) > 5:
                print(f"            {', '.join(themes[5:])}")
        else:
            print(f"  Themes:   {Colors.WARNING}none")

        risk = labels.get('retention_risk')
        if risk:
            color = Colors.OKGREEN if risk == 'low' else Colors.WARNING if risk == 'medium' else Colors.FAIL
            print(f"  Risk:     {color}{risk.upper()}{Colors.RESET}")

        confidence = labels.get('confidence')
        if confidence is not None:
            conf_display = f"{confidence:.2f}"
            print(f"  Confidence: {conf_display}")
        else:
            print(f"  Confidence: {Colors.WARNING}N/A")

        reasoning = labels.get('reasoning', '')
        if reasoning:
            display_reasoning = truncate_text(reasoning, 100)
            print(f"  Reasoning: {display_reasoning}")
        else:
            print(f"  Reasoning: {Colors.WARNING}none")

        print(f"\n{Colors.BOLD}{Colors.HEADER}═{'═'*60}{Colors.RESET}")

    def _prompt_verification(self, review_id: str) -> str:
        """Prompt user for verification decision."""

        print(f"\n{Colors.BOLD}Are these labels correct?{Colors.RESET}")
        print(f"  (y=Yes, n=No, s=Skip, q=Quit)")

        if readchar:
            return readchar(prompt_suffix="\n> ", render=True)
        else:
            # Fallback for systems without termcap
            return input("\n> ").strip()

    def _record_verification(self, review_id: str, labels: Dict, verified: bool):
        """Record a verification decision."""

        # Update auto labels with verification
        if review_id in self.reviews:
            self.reviews[review_id]['auto_labels'] = labels
            self.reviews[review_id]['human_verified'] = verified

        # Record verification entry
        verification_record = {
            'review_id': review_id,
            'auto_labels': labels,
            'human_verified': verified,
            'verification_date': datetime.now().isoformat()
        }

        # Save incrementally
        self.verification_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.verification_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(verification_record) + '\n')

        if verified:
            self.verified_count += 1
            self.agree_count += 1
            print(f"{Colors.OKGREEN}  Recorded: VERIFIED{Colors.RESET}")
        else:
            self.disagree_count += 1
            print(f"{Colors.WARNING}  Recorded: CHALLENGED{Colors.RESET}")

    def _calculate_agreement(self) -> float:
        """Calculate agreement rate."""

        total = self.agree_count + self.disagree_count
        if total == 0:
            return 0.0
        return (self.agree_count / total) * 100

    def verify_sample(self, n: int = 30) -> str:
        """Verify a random sample of n unverified reviews."""

        # Find unverified reviews
        unverified = [
            (rid, r) for rid, r in self.reviews.items()
            if not r.get('human_verified')
        ]

        if not unverified:
            print(f"{Colors.WARNING}No unverified reviews found!")
            return "No unverified reviews"

        # Sample n reviews
        sample_size = min(n, len(unverified))
        sample = random.sample(unverified, sample_size)

        print(f"\n{Colors.BOLD}{Colors.HEADER}═{'═'*60}{Colors.RESET}")
        print(f"{Colors.BOLD}VERIFICATION SAMPLE{Colors.RESET}")
        print(f"Reviewing {sample_size} randomly sampled reviews")
        print(f"Unverified pool: {len(unverified)} reviews")
        print()

        # Verify each review
        for i, (review_id, review) in enumerate(sample, 1):
            # Progress counter
            print(f"{Colors.BOLD}Review {i+1}/{sample_size}{Colors.RESET}", end='\r')

            # Get auto labels if available
            labels = review.get('auto_labels', {})
            if not labels:
                labels = {
                    'sentiment': None,
                    'themes': [],
                    'retention_risk': None,
                    'confidence': None,
                    'reasoning': None
                }

            # Display review and labels
            self._display_review(review, labels)

            # Prompt for verification
            choice = self._prompt_verification(review_id)

            # Handle choice
            if choice.lower() in ['y', 'yes']:
                self._record_verification(review_id, labels, verified=True)
                print(f"\n{Colors.OKGREEN}✓ Verified{Colors.RESET}\n")
                time.sleep(0.1)  # Brief pause

            elif choice.lower() in ['s', 'skip']:
                self.skipped_count += 1
                print(f"{Colors.WARNING}→ Skipped{Colors.RESET}\n")
                time.sleep(0.1)

            elif choice.lower() in ['q', 'quit']:
                print(f"\n{Colors.FAIL}✗ Quitting{Colors.RESET}\n")
                print(f"{Colors.BOLD}Verified so far:{Colors.RESET} {self.verified_count}")
                print(f"{Colors.BOLD}Agreement rate:{Colors.RESET} {self._calculate_agreement():.1f}%")
                return "quit"

            else:
                # Treat as challenge/disagreement
                self._record_verification(review_id, labels, verified=False)
                print(f"\n{Colors.WARNING}✗ Challenged{Colors.RESET}\n")
                print(f"Recording disagreement for review statistics\n")
                time.sleep(0.1)

        # Final summary
        print(f"\n{Colors.BOLD}{Colors.HEADER}═{'═'*60}{Colors.RESET}")
        print(f"{Colors.BOLD}VERIFICATION SUMMARY{Colors.RESET}")
        print()
        print(f"Total reviewed:     {sample_size}")
        print(f"Verified:           {self.agree_count} {Colors.OKGREEN}✓{Colors.RESET}")
        print(f"Challenged:         {self.disagree_count} {Colors.WARNING}✗{Colors.RESET}")
        print(f"Skipped:            {self.skipped_count} →")
        print()

        # Calculate agreement
        total_decisions = self.agree_count + self.disagree_count
        if total_decisions > 0:
            agreement_rate = (self.agree_count / total_decisions) * 100
            print(f"{Colors.BOLD}Agreement Rate:{Colors.RESET} {agreement_rate:.1f}%")

            # Pass/fail threshold
            if agreement_rate >= 85:
                print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ PASS{Colors.RESET}")
                print("Agreement is >=85%. Labels are trustworthy.")
                print("You can proceed with model training.")
            elif agreement_rate >= 70:
                print(f"\n{Colors.WARNING}⚠ CAUTION{Colors.RESET}")
                print("Agreement is 70-84%. Review labels before proceeding.")
                print("\nCommon issues to check:")
                print("  - Sentiment over/under-estimation")
                print("  - Missing or irrelevant themes")
                print("  - Reasoning doesn't match labels")
            else:
                print(f"\n{Colors.FAIL}✗ FAIL{Colors.RESET}")
                print("Agreement is <70%. Do NOT use these labels.")
                print("\nRecommended actions:")
                print("  - Review labeling prompt")
                print("  - Adjust temperature (lower for more consistency)")
                print("  - Consider human labeling some samples")
        else:
            print("\nNo decisions recorded.")

        return "complete"

    def verify_all(self) -> str:
        """Verify all unverified reviews."""

        unverified = [
            (rid, r) for rid, r in self.reviews.items()
            if not r.get('human_verified')
        ]

        return self.verify_sample(len(unverified))


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify auto-generated labels against human judgment"
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=30,
        help='Number of reviews to verify (default: 30)'
    )
    parser.add_argument(
        '--silver', '-i',
        type=str,
        default='data/labeled/silver_standard.jsonl',
        help='Path to silver standard labels'
    )

    args = parser.parse_args()

    # Initialize verifier
    verifier = LabelVerifier(Path(args.silver))

    print(f"{Colors.BOLD}{Colors.HEADER}ReviewInsight AI{Colors.RESET}")
    print(f"{Colors.BOLD}Label Verification Tool{Colors.RESET}")
    print()
    print(f"Silver standard: {args.silver}")
    print(f"Sample size:       {args.sample}")
    print()

    # Run verification
    result = verifier.verify_sample(args.sample)

    print(f"\n{Colors.BOLD}Thanks for using ReviewInsight!{Colors.RESET}")

    return 0


if __name__ == '__main__':
    exit(main())

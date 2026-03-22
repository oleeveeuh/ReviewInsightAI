#!/usr/bin/env python3
"""
Manual Reddit Entry Tool

Simple tool to manually add Reddit posts to the dataset.
Run this script and paste Reddit post content when prompted.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


def generate_review_id(post_id: str) -> str:
    """Generate a unique review ID from Reddit post ID."""
    return f"reddit_{post_id}"


def parse_quarter(date_str: str) -> str:
    """Parse quarter from date string."""
    try:
        date = datetime.fromisoformat(date_str.replace('T', ' ').split()[0])
        month = date.month
        return f"Q{(month - 1) // 3 + 1}"
    except:
        return "Q1"


def add_reddit_post(
    post_id: str,
    title: str,
    text: str,
    subreddit: str,
    date: str = None,
    url: str = None
) -> dict:
    """
    Create a formatted Reddit entry.

    Args:
        post_id: Reddit post ID (e.g., "1abc123" from URL)
        title: Post title
        text: Post body content
        subreddit: Subreddit name (without r/)
        date: Date in YYYY-MM-DD format (defaults to today)
        url: Full Reddit post URL

    Returns:
        Formatted review dict
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Extract year from date
    try:
        year = int(date.split('-')[0])
    except:
        year = datetime.now().year

    quarter = parse_quarter(date)

    # Combine title and text for full content
    if title:
        full_text = f"{title}\n\n{text}" if text else title
    else:
        full_text = text or ""

    # Generate URL if not provided
    if url is None:
        url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}/"

    review = {
        "review_id": generate_review_id(post_id),
        "text": full_text.strip(),
        "rating": None,
        "date": date,
        "year": year,
        "quarter": quarter,
        "location": None,
        "job_title": None,
        "source": "reddit",
        "review_length": len(full_text.split()),
        "metadata": {
            "subreddit": subreddit,
            "post_id": post_id,
            "url": url,
            "title": title,
            "manually_added": True,
            "added_date": datetime.now().isoformat()
        }
    }

    return review


def save_review(review: dict, output_path: Path = None) -> None:
    """Save a review to the JSONL file."""
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "reddit_reviews.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing reviews to check for duplicates
    existing_ids = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_ids.add(data.get('review_id'))
                    except:
                        pass

    # Check if already exists
    if review['review_id'] in existing_ids:
        print(f"⚠️  Post {review['review_id']} already exists in dataset!")
        return

    # Append new review
    with open(output_path, 'a') as f:
        f.write(json.dumps(review) + '\n')

    print(f"✅ Added: {review['review_id']} ({review['review_length']} words)")


def interactive_mode():
    """Interactive mode for manual entry."""
    print("=" * 60)
    print("MANUAL REDDIT ENTRY TOOL")
    print("=" * 60)
    print("\nPaste information from a Reddit post when prompted.")
    print("Press Ctrl+C to exit.\n")

    count = 0

    try:
        while True:
            print("\n" + "-" * 40)
            print(f"Entry #{count + 1}")

            url = input("Reddit URL (or just post ID): ").strip()

            # Extract post ID from URL if full URL provided
            if '/' in url:
                # URL format: https://reddit.com/r/subreddit/comments/abc123/title/
                parts = url.split('/')
                post_id = None
                for i, part in enumerate(parts):
                    if 'comments' in parts[i-1] if i > 0 else False:
                        post_id = part
                        break
                    if part == 'comments' and i + 1 < len(parts):
                        post_id = parts[i + 1]
                        break
                if not post_id:
                    # Try to find the ID after /comments/
                    if 'comments/' in url:
                        post_id = url.split('comments/')[1].split('/')[0]
                    else:
                        post_id = url.split('/')[-1].split('_')[0]

                # Extract subreddit from URL
                subreddit = 'unknown'
                if '/r/' in url:
                    subreddit = url.split('/r/')[1].split('/')[0]
            else:
                post_id = url
                subreddit = input("Subreddit (without r/): ").strip()

            title = input("Title: ").strip()
            text = input("\nPaste post text (Ctrl+D when done):\n")

            date_str = input("\nDate (YYYY-MM-DD, or press Enter for today): ").strip()
            if not date_str:
                date_str = datetime.now().strftime("%Y-%m-%d")

            # Create and save review
            review = add_reddit_post(
                post_id=post_id,
                title=title,
                text=text,
                subreddit=subreddit,
                date=date_str,
                url=url if '/' in url else None
            )

            save_review(review)
            count += 1

            print(f"\n✅ Saved! Total entries this session: {count}")

            another = input("\nAdd another? (y/n): ").strip().lower()
            if another != 'y':
                break

    except KeyboardInterrupt:
        print(f"\n\nExiting. Added {count} entries this session.")


def parse_template_file(template_path: Path) -> list:
    """Parse entries from a template file."""
    entries = []
    current_entry = {}

    with open(template_path, 'r') as f:
        content = f.read()

    # Split by ---ENTRY---
    blocks = content.split('---ENTRY---')

    for block in blocks:
        if '---END---' not in block:
            continue

        lines = block.strip().split('\n')
        entry = {}

        for line in lines:
            if ':' in line and not line.strip().startswith('#'):
                key, value = line.split(':', 1)
                entry[key.strip()] = value.strip()

        if entry.get('post_id') and entry.get('subreddit'):
            entries.append(entry)

    return entries


def batch_mode(template_path: Path = None):
    """Batch mode to import from template file."""
    if template_path is None:
        template_path = Path(__file__).parent.parent.parent / "data" / "raw" / "manual_reddit_template.txt"

    if not template_path.exists():
        print(f"❌ Template file not found: {template_path}")
        return

    entries = parse_template_file(template_path)

    if not entries:
        print("❌ No valid entries found in template file")
        return

    print(f"Found {len(entries)} entries in template file")
    print("=" * 60)

    count = 0
    for entry in entries:
        review = add_reddit_post(
            post_id=entry['post_id'],
            title=entry.get('title', ''),
            text=entry.get('text', ''),
            subreddit=entry['subreddit'],
            date=entry.get('date')
        )
        save_review(review)
        count += 1

    print("=" * 60)
    print(f"✅ Imported {count} entries")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print(__doc__)
            print("\nUsage:")
            print("  python manual_reddit_entry.py           # Interactive mode")
            print("  python manual_reddit_entry.py --batch   # Import from template file")
            print("  python manual_reddit_entry.py --help    # Show this help")
            print("\nFor programmatic use, import and call add_reddit_post()")
            sys.exit(0)
        elif sys.argv[1] == '--batch':
            batch_mode()
            sys.exit(0)

    interactive_mode()

#!/usr/bin/env python3
"""
Reddit scraper using the Pushshift beta API.

WARNING: Pushshift access has been restricted since 2023; this endpoint
typically returns HTTP 403 for anonymous clients today, in which case the
scraper collects nothing. data/raw/reddit_reviews.jsonl in the shipped
fixtures is synthetic. Use src/scraping/manual_reddit_entry.py for a
no-API alternative. Not affiliated with or endorsed by Reddit.

Usage:
    python src/scraping/reddit_scraper.py [--target 150] [--output path/to/output.jsonl]
"""

import requests
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlencode

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class RedditScraper:
    """
    Scrape Reddit posts from Amazon-related subreddits using Pushshift API.

    Pushshift Beta API: https://beta.pushshift.io/reddit/search
    No authentication required for basic usage.
    """

    # Pushshift Beta API endpoint
    SEARCH_URL = "https://beta.pushshift.io/reddit/search/submission/"

    # Target subreddits for Amazon warehouse content
    TARGET_SUBREDDITS = [
        "AmazonFC",        # Primary - most active
        "FASCAmazon",      # Secondary - Former Amazon
        "AmazonDSPDrivers", # Tertiary - DSP drivers
        "AmazonEmployees", # If accessible
        "amazonwarehouse", # Niche community
    ]

    # Search queries related to warehouse work
    SEARCH_QUERIES = [
        "fulfillment center",
        "warehouse work",
        "overtime",
        "peak season",
        "MET",           # Mandatory Extra Time
        "VTO",           # Voluntary Time Off
        "rate",          # Rate/quota
        "stand up",      # Standing all shift
        "shift",
        " FC ",          # Fulfillment Center
        "sorting",
        "picker",
        "packer",
        "tier 1",
        "ambassador",    # Problem solver role
        "PATH",          # Amazon training program
        "blue badge",    # Converted employee
    ]

    # Minimum post date (Jan 1, 2023)
    MIN_DATE = datetime(2023, 1, 1)

    # Quality filters
    MIN_WORD_COUNT = 50
    MAX_WORD_COUNT = 5000

    # API request settings
    RATE_LIMIT_DELAY = 1.5  # seconds between requests
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30

    def __init__(self, target_count: int = 150, output_path: Optional[str] = None):
        """
        Initialize the Reddit scraper.

        Args:
            target_count: Target number of quality posts to collect
            output_path: Where to save the output JSONL file
        """
        self.target_count = target_count
        self.seen_post_ids: Set[str] = set()
        self.collected_posts: List[Dict] = []
        self.stats = {
            "total_fetched": 0,
            "filtered_deleted": 0,
            "filtered_short": 0,
            "filtered_long": 0,
            "filtered_self_post": 0,
            "duplicates": 0,
            "quality_posts": 0,
        }

        # Set output path
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "reddit_reviews.jsonl"
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ReviewInsight/1.0 (Employee Sentiment Research)'
        })

    def search_subreddit(self, subreddit: str, query: str = "",
                         after_date: Optional[datetime] = None,
                         before_date: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict]:
        """
        Search a subreddit for posts matching criteria.

        Args:
            subreddit: Subreddit name (without r/)
            query: Search query string
            after_date: Fetch posts after this date
            before_date: Fetch posts before this date
            limit: Maximum posts per request

        Returns:
            List of post dictionaries from Pushshift API
        """
        # Build URL with parameters
        params = {
            "subreddit": subreddit,
            "size": str(limit),
            "sort": "desc",
            "sort_type": "created_utc",
            "fields": "id,title,selftext,created_utc,score,num_comments,subreddit,permalink,author,is_self,over_18"
        }

        if query:
            params["q"] = query

        if after_date:
            params["after"] = str(int(after_date.timestamp()))

        if before_date:
            params["before"] = str(int(before_date.timestamp()))

        url = f"{self.SEARCH_URL}?{urlencode(params, doseq=True)}"

        # Make request with retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                # Add delay between requests
                if attempt > 0:
                    time.sleep(2 ** attempt + random.uniform(1, 3))

                response = self.session.get(
                    url,
                    timeout=self.REQUEST_TIMEOUT,
                    headers={'Accept': 'application/json'}
                )

                if response.status_code == 403:
                    # Pushshift rejects anonymous access - make this loud,
                    # not silent, so an empty result is never mistaken for
                    # "no matching posts".
                    print("  ⚠️ Pushshift returned 403 (access restricted); "
                          "the scraper cannot collect data. Use "
                          "manual_reddit_entry.py instead.")
                    return []

                response.raise_for_status()

                data = response.json()

                # Handle different response formats
                if isinstance(data, dict):
                    if "error" in data:
                        if "rate limit" in str(data["error"]).lower():
                            time.sleep(5)
                            continue
                        return []
                    return data.get("data", [])
                elif isinstance(data, list):
                    return data

                return []

            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES - 1:
                    continue
                return []

            except requests.exceptions.RequestException:
                if attempt < self.MAX_RETRIES - 1:
                    continue
                return []

            except (json.JSONDecodeError, ValueError):
                return []

        return []

    def filter_quality_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Apply quality filters to posts.

        Filters:
        - Must be self-post (not link-only)
        - Minimum word count
        - Maximum word count
        - Exclude [deleted] and [removed]
        - Exclude posts with empty selftext

        Args:
            posts: List of raw posts from API

        Returns:
            List of filtered quality posts
        """
        quality_posts = []

        for post in posts:
            self.stats["total_fetched"] += 1

            # Check for deleted/removed
            title = post.get("title", "")
            selftext = post.get("selftext", "")

            if title in ["[deleted]", "[removed]"] or selftext in ["[deleted]", "[removed]", ""]:
                self.stats["filtered_deleted"] += 1
                continue

            # Must be self-post
            if not post.get("is_self", False):
                self.stats["filtered_self_post"] += 1
                continue

            # Combine title and selftext for word count
            full_text = f"{title} {selftext}"
            word_count = len(full_text.split())

            if word_count < self.MIN_WORD_COUNT:
                self.stats["filtered_short"] += 1
                continue

            if word_count > self.MAX_WORD_COUNT:
                self.stats["filtered_long"] += 1
                continue

            # Check for duplicates
            post_id = post.get("id")
            if post_id in self.seen_post_ids:
                self.stats["duplicates"] += 1
                continue

            quality_posts.append(post)
            self.seen_post_ids.add(post_id)
            self.stats["quality_posts"] += 1

        return quality_posts

    def convert_to_standard_format(self, post: Dict) -> Dict:
        """
        Convert Reddit post to standard data schema.

        Schema matches Glassdoor and YouTube sources for easy merging.

        Args:
            post: Raw post from Pushshift API

        Returns:
            Dictionary in standard format
        """
        post_id = post.get("id")
        title = post.get("title", "")
        selftext = post.get("selftext", "")

        # Combine title and body
        text = f"{title}\n\n{selftext}".strip()

        # Convert UTC timestamp to date
        created_utc = post.get("created_utc", 0)
        post_date = datetime.utcfromtimestamp(created_utc)
        date_str = post_date.strftime("%Y-%m-%d")
        year = post_date.year
        quarter = f"Q{(post_date.month - 1) // 3 + 1}"

        # Calculate review length
        word_count = len(text.split())

        return {
            "review_id": f"reddit_{post_id}",
            "text": text,
            "rating": None,
            "date": date_str,
            "year": year,
            "quarter": quarter,
            "location": None,
            "job_title": None,
            "source": "reddit",
            "review_length": word_count,
            "metadata": {
                "subreddit": post.get("subreddit"),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "post_id": post_id,
                "url": f"https://reddit.com{post.get('permalink', '')}",
                "author": post.get("author", "[unknown]"),
            }
        }

    def scrape_amazon_fc_content(self) -> List[Dict]:
        """
        Main scraping method - collect Amazon warehouse content from Reddit.

        Strategy:
        1. Search each target subreddit with each query
        2. Apply quality filters
        3. Deduplicate across searches
        4. Stop when target_count is reached

        Returns:
            List of reviews in standard format
        """
        print("=" * 60)
        print("REDDIT SCRAPER - Amazon Employee Sentiment")
        print("=" * 60)
        print(f"\nTarget: {self.target_count} quality posts")
        print(f"Subreddits: {', '.join(self.TARGET_SUBREDDITS)}")
        print(f"Date range: {self.MIN_DATE.strftime('%Y-%m-%d')} to present")
        print(f"Min words: {self.MIN_WORD_COUNT}")

        all_reviews = []

        # Create search configurations
        searches = []
        for subreddit in self.TARGET_SUBREDDITS:
            # Add unfiltered search (no query) for general posts
            searches.append((subreddit, ""))

            # Add filtered searches for specific topics
            for query in self.SEARCH_QUERIES[:5]:  # Limit queries per subreddit
                searches.append((subreddit, query))

        # Remove duplicates from searches
        searches = list(set(searches))

        if HAS_TQDM:
            search_pbar = tqdm(searches, desc="Searching subreddits")
        else:
            search_pbar = searches
            print(f"\nSearching {len(searches)} subreddit/query combinations...")

        for subreddit, query in search_pbar:
            # Check if we've reached target
            if len(all_reviews) >= self.target_count:
                break

            if HAS_TQDM:
                search_pbar.set_description_str(
                    f"r/{subreddit} q='{query[:20] if query else 'all'}' | "
                    f"Collected: {len(all_reviews)}/{self.target_count}"
                )

            # Search subreddit
            posts = self.search_subreddit(
                subreddit=subreddit,
                query=query,
                after_date=self.MIN_DATE,
                limit=100
            )

            if not posts:
                time.sleep(self.RATE_LIMIT_DELAY)
                continue

            # Filter quality posts
            quality_posts = self.filter_quality_posts(posts)

            # Convert to standard format
            for post in quality_posts:
                review = self.convert_to_standard_format(post)
                all_reviews.append(review)

                # Check if we've reached target
                if len(all_reviews) >= self.target_count:
                    break

            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)

        # Trim to exact target
        all_reviews = all_reviews[:self.target_count]
        self.collected_posts = all_reviews

        return all_reviews

    def save_reviews(self, filename: Optional[str] = None) -> Path:
        """
        Save collected reviews to JSONL file.

        Args:
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if filename:
            output_path = Path(filename)
        else:
            output_path = self.output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for review in self.collected_posts:
                f.write(json.dumps(review, ensure_ascii=False) + '\n')

        return output_path

    def print_statistics(self):
        """Print comprehensive collection statistics."""
        print("\n" + "=" * 60)
        print("SCRAPING STATISTICS")
        print("=" * 60)

        print("\n📊 COLLECTION SUMMARY:")
        print(f"  Target:        {self.target_count}")
        print(f"  Collected:     {len(self.collected_posts)}")
        print(f"  Success rate:  {len(self.collected_posts) / self.target_count * 100:.1f}%")

        print("\n📈 QUALITY FILTERS:")
        print(f"  Total fetched:      {self.stats['total_fetched']}")
        print(f"  Deleted/removed:    {self.stats['filtered_deleted']}")
        print(f"  Too short:          {self.stats['filtered_short']}")
        print(f"  Too long:           {self.stats['filtered_long']}")
        print(f"  Not self-post:      {self.stats['filtered_self_post']}")
        print(f"  Duplicates:         {self.stats['duplicates']}")
        print(f"  Quality posts:      {self.stats['quality_posts']}")

        if self.collected_posts:
            # Date range
            dates = [r['date'] for r in self.collected_posts if r['date']]
            if dates:
                print("\n📅 DATE RANGE:")
                print(f"  From: {min(dates)}")
                print(f"  To:   {max(dates)}")

            # Subreddit distribution
            subreddit_counts = {}
            for r in self.collected_posts:
                sub = r['metadata']['subreddit']
                subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

            print("\n📍 SUBREDDIT DISTRIBUTION:")
            for sub, count in sorted(subreddit_counts.items(), key=lambda x: -x[1]):
                pct = count / len(self.collected_posts) * 100
                print(f"  r/{sub:20} {count:4} ({pct:5.1f}%)")

            # Word count stats
            word_counts = [r['review_length'] for r in self.collected_posts]
            print("\n📝 REVIEW LENGTH:")
            print(f"  Min:     {min(word_counts)} words")
            print(f"  Max:     {max(word_counts)} words")
            print(f"  Average: {sum(word_counts) // len(word_counts)} words")
            print(f"  Median:  {sorted(word_counts)[len(word_counts)//2]} words")

            # Engagement stats
            scores = [r['metadata']['score'] for r in self.collected_posts]
            comments = [r['metadata']['num_comments'] for r in self.collected_posts]
            print("\n💬 ENGAGEMENT:")
            print(f"  Avg upvotes:     {sum(scores) / len(scores):.1f}")
            print(f"  Avg comments:    {sum(comments) / len(comments):.1f}")

        print(f"\n💾 Saved to: {self.output_path}")
        print("=" * 60)


def main():
    """Main execution block."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Reddit for Amazon warehouse employee discussions"
    )
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=150,
        help='Target number of quality posts (default: 150)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=50,
        help='Minimum word count (default: 50)'
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = RedditScraper(
        target_count=args.target,
        output_path=args.output
    )

    # Override min words if specified
    if args.min_words:
        scraper.MIN_WORD_COUNT = args.min_words

    # Scrape content
    print("\n🚀 Starting Reddit scrape...")
    print("Estimated time: 10-15 minutes\n")

    start_time = time.time()

    reviews = scraper.scrape_amazon_fc_content()

    elapsed = time.time() - start_time

    # Save results
    if reviews:
        scraper.save_reviews()
        scraper.print_statistics()

        print(f"\n⏱️  Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    else:
        print("\n⚠️  No posts collected. Check your internet connection.")


if __name__ == '__main__':
    main()

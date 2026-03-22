#!/usr/bin/env python3
"""
Batch process all reviews with LLM sentiment analysis.
Uses the best prompt configuration from evaluation.

Usage:
    python src/analysis/batch_process.py --prompt v3 --k 3 --model gpt-4o-mini
"""

import json
import pandas as pd
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Try to import OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Run: pip install openai python-dotenv")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.templates import ALL_PROMPTS


def estimate_cost(n_reviews: int, avg_tokens_in: int = 600,
                 avg_tokens_out: int = 100, model: str = 'gpt-4o-mini') -> float:
    """Estimate API cost."""
    if model == 'gpt-4o-mini':
        cost_in = 0.150 / 1_000_000
        cost_out = 0.600 / 1_000_000
    elif model == 'gpt-4o':
        cost_in = 2.50 / 1_000_000
        cost_out = 10.00 / 1_000_000
    else:
        cost_in = 0.150 / 1_000_000
        cost_out = 0.600 / 1_000_000

    total_cost = (n_reviews * avg_tokens_in * cost_in) + (n_reviews * avg_tokens_out * cost_out)
    return total_cost


class BatchProcessor:
    """Process reviews with LLM analysis."""

    def __init__(self, api_key: str = None):
        if HAS_OPENAI:
            load_dotenv()
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                self.client = None
                print("⚠️  No OpenAI API key found")
        else:
            self.client = None

    def call_llm(self, system_prompt: str, user_prompt: str,
                 model: str = 'gpt-4o-mini') -> Optional[Dict]:
        """Call OpenAI API."""
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return {
                'content': response.choices[0].message.content,
                'tokens_in': response.usage.prompt_tokens,
                'tokens_out': response.usage.completion_tokens
            }
        except Exception as e:
            print(f"API error: {e}")
            return None

    def parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM JSON response."""
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(response_text)

            # Normalize
            result['sentiment'] = int(result['sentiment'])
            result['themes'] = [str(t).lower().replace(' ', '_').replace('-', '_')
                               for t in result.get('themes', [])]
            result['retention_risk'] = str(result.get('retention_risk', 'low')).lower()

            return result
        except Exception as e:
            return None

    def load_reviews(self, input_path: str) -> List[Dict]:
        """Load reviews from JSONL file."""
        reviews = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
        return reviews

    def load_processed_ids(self, output_path: str) -> set:
        """Load IDs of already processed reviews."""
        processed_ids = set()
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        processed_ids.add(result['review_id'])
                    except:
                        pass
        return processed_ids

    def batch_process(self,
                     input_path: str = None,
                     output_path: str = None,
                     prompt_version: str = 'v3',
                     k_shot: int = 3,
                     model: str = 'gpt-4o-mini',
                     batch_size: int = 10,
                     checkpoint_every: int = 50,
                     dry_run: bool = False):
        """
        Process all reviews with LLM analysis.

        Args:
            input_path: Path to reviews JSONL
            output_path: Path to save results
            prompt_version: Prompt template version
            k_shot: Number of few-shot examples
            model: OpenAI model name
            batch_size: Reviews per batch for rate limiting
            checkpoint_every: Save checkpoint every N reviews
            dry_run: Simulate without API calls
        """

        # Default paths
        if input_path is None:
            input_path = Path(__file__).parent.parent.parent / "data" / "processed" / "reviews_final.jsonl"
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / "data" / "analysis" / "llm_results.jsonl"

        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load reviews
        print("=" * 60)
        print("BATCH LLM ANALYSIS")
        print("=" * 60)
        print(f"\nInput: {input_path}")
        print(f"Output: {output_path}")

        reviews = self.load_reviews(input_path)
        print(f"Loaded {len(reviews)} reviews")

        # Load processed IDs (resume capability)
        processed_ids = self.load_processed_ids(str(output_path))
        print(f"Already processed: {len(processed_ids)}")

        # Filter to unprocessed
        to_process = [r for r in reviews if r['review_id'] not in processed_ids]
        print(f"To process: {len(to_process)}")

        if not to_process:
            print("\n✅ All reviews already processed!")
            return

        # Get prompt template
        if prompt_version not in ALL_PROMPTS:
            available = ', '.join(ALL_PROMPTS.keys())
            raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {available}")

        prompt_template = ALL_PROMPTS[prompt_version]
        print(f"\nPrompt: {prompt_template.name} (v{prompt_version}, k={k_shot})")
        print(f"Model: {model}")

        # Estimate cost
        if not dry_run:
            est_cost = estimate_cost(len(to_process), model=model)
            print(f"\nEstimated cost: ${est_cost:.2f}")

            if len(to_process) > 20:
                print(f"\n⚠️  Processing {len(to_process)} reviews...")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled.")
                    return

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process
        total_tokens_in = 0
        total_tokens_out = 0
        failed = 0
        pbar = None

        if HAS_TQDM and not dry_run:
            pbar = tqdm(to_process, desc="Processing")
        else:
            pbar = to_process
            print(f"\nProcessing {len(to_process)} reviews...")

        for i, review in enumerate(pbar):
            # Render prompt
            system_prompt, user_prompt = prompt_template.render(review['text'], k_shot=k_shot)

            if dry_run or not self.client:
                # Mock response
                gold = review.get('labels', {})
                mock_sentiment = max(1, min(5, gold.get('sentiment', 3) + 0))
                analysis = {
                    'sentiment': mock_sentiment,
                    'themes': gold.get('themes', ['other'])[:2],
                    'retention_risk': gold.get('retention_risk', 'low')
                }
                tokens_in = 600
                tokens_out = 100
            else:
                # Call API
                response = self.call_llm(system_prompt, user_prompt, model=model)

                if response is None:
                    failed += 1
                    analysis = None
                    tokens_in = 0
                    tokens_out = 0
                else:
                    tokens_in = response['tokens_in']
                    tokens_out = response['tokens_out']
                    total_tokens_in += tokens_in
                    total_tokens_out += tokens_out

                    # Parse response
                    analysis = self.parse_response(response['content'])

                    if analysis is None:
                        failed += 1

            # Build result
            result_data = {
                'review_id': review['review_id'],
                'source': review.get('source'),
                'date': review.get('date'),
                'llm_sentiment': analysis.get('sentiment') if analysis else None,
                'llm_themes': analysis.get('themes') if analysis else None,
                'llm_retention_risk': analysis.get('retention_risk') if analysis else None,
                'processing_error': analysis is None,
                'model': model if not dry_run else 'dry_run',
                'prompt_version': prompt_version,
                'k_shot': k_shot,
                'processed_at': datetime.now().isoformat()
            }

            # Save incrementally (append mode)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')

            # Checkpoint
            if not HAS_TQDM and (i + 1) % checkpoint_every == 0:
                print(f"Checkpoint: {i+1}/{len(to_process)} processed")

            # Rate limiting
            if not dry_run and (i + 1) % batch_size == 0:
                time.sleep(1)  # Conservative rate limiting

        # Final stats
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {len(to_process)}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(len(to_process) - failed) / len(to_process) * 100:.1f}%")

        if not dry_run:
            actual_cost = (total_tokens_in * 0.150 / 1_000_000) + (total_tokens_out * 0.600 / 1_000_000)
            print(f"Total tokens: {total_tokens_in:,} in / {total_tokens_out:,} out")
            print(f"Actual cost: ${actual_cost:.2f}")
        else:
            actual_cost = 0
            print("Dry run mode - no API calls made")

        print(f"Saved to: {output_path}")

        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_path': str(input_path),
            'output_path': str(output_path),
            'total_reviews': len(reviews),
            'already_processed': len(processed_ids),
            'processed_this_run': len(to_process),
            'failed': failed,
            'model': model,
            'prompt_version': prompt_version,
            'k_shot': k_shot,
            'dry_run': dry_run,
            'total_tokens_in': total_tokens_in,
            'total_tokens_out': total_tokens_out,
            'actual_cost': actual_cost
        }

        report_path = str(output_path).replace('.jsonl', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {report_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch process reviews with LLM sentiment analysis'
    )
    parser.add_argument('--prompt', '-p', default='v3',
                       help='Prompt version (v1-v6), default: v3')
    parser.add_argument('--k', '-k', type=int, default=3,
                       help='Number of few-shot examples, default: 3')
    parser.add_argument('--model', '-m', default='gpt-4o-mini',
                       help='OpenAI model, default: gpt-4o-mini')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Simulate without API calls')
    parser.add_argument('--input', '-i', type=str,
                       help='Input reviews JSONL path')
    parser.add_argument('--output', '-o', type=str,
                       help='Output results JSONL path')

    args = parser.parse_args()

    processor = BatchProcessor()

    processor.batch_process(
        input_path=args.input,
        output_path=args.output,
        prompt_version=args.prompt,
        k_shot=args.k,
        model=args.model,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

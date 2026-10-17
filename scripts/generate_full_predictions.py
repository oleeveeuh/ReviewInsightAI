#!/usr/bin/env python3
"""
Generate one gpt-4o-mini prediction per (prompt, review) over the FULL corpus.

This is the remedy for the historical evaluation's 17-review ceiling: the
original run scored only the first 50 label rows (= first 17 reviews, all
Reddit). This script produces predictions for all 175 reviews x 6 prompts
(1,050 calls), which the scorer then evaluates leave-one-prompt-out.

Protocol matches the historical run: gpt-4o-mini, zero-shot (k_shot=0),
temperature 0.3, max_tokens 500.

Checkpointed and resumable: completed predictions are appended to the output
and skipped on re-run.

Usage (requires OPENAI_API_KEY in .env and requirements-optional.txt):
    .venv/bin/python scripts/generate_full_predictions.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

REVIEWS_PATH = REPO_ROOT / 'data_local/processed/reviews_final.jsonl'
OUT_PATH = REPO_ROOT / 'data_local/evaluation/full_predictions.jsonl'
MODEL = 'gpt-4o-mini'


def main():
    from dotenv import load_dotenv
    from openai import OpenAI
    from prompts.templates import ALL_PROMPTS
    from evaluation.evaluate_prompts import PromptEvaluator

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('sk-your-'):
        sys.exit('OPENAI_API_KEY not set. Copy .env.example to .env and add '
                 'your key, then re-run.')
    client = OpenAI(api_key=api_key)

    reviews = [json.loads(line) for line in REVIEWS_PATH.read_text().splitlines()
               if line.strip()]
    prompts = list(ALL_PROMPTS.items())

    # Resume support: skip already-completed (prompt, review) pairs
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row['prompt_version'], row['review_id']))
    n_total = len(prompts) * len(reviews)
    print(f'Predictions: {len(done)}/{n_total} already done, '
          f'{n_total - len(done)} to go (~${(n_total - len(done)) * 0.00015:.2f})')

    with open(OUT_PATH, 'a') as out:
        for prompt_key, prompt in prompts:
            for review in reviews:
                rid = review['review_id']
                if (prompt_key, rid) in done:
                    continue
                system_prompt, user_prompt = prompt.render(review['text'], k_shot=0)

                parsed, raw, error = None, None, None
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=MODEL,
                            messages=[{'role': 'system', 'content': system_prompt},
                                      {'role': 'user', 'content': user_prompt}],
                            temperature=0.3, max_tokens=500)
                        raw = resp.choices[0].message.content
                        parsed = PromptEvaluator.parse_llm_response(raw)
                        if parsed is None:
                            error = 'unparseable'
                        break
                    except Exception as e:
                        error = f'{type(e).__name__}'
                        time.sleep(2 ** attempt)
                else:
                    print(f'\nFAILED {prompt_key}/{rid} after 3 retries: {error}')

                out.write(json.dumps({
                    'review_id': rid,
                    'prompt_version': prompt_key,
                    'prediction': parsed,
                    'raw': raw,
                    'error': error,
                    'model': MODEL,
                    'k_shot': 0,
                    'temperature': 0.3,
                }) + '\n')
                out.flush()

                done_n = len(done) + 1
                if done_n % 25 == 0:
                    print(f'{done_n}/{n_total} predictions '
                          f'({datetime.now(timezone.utc).strftime("%H:%M:%S")})')
                time.sleep(0.2)

    print(f'Done: {n_total} predictions in {OUT_PATH}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Construct DPO preference pairs from cross-prompt silver labels.

For each review labeled by multiple generator prompts (v2, v3, v6), the
prompt whose labels agree most with the others becomes the CHOSEN response
and the most outlying one becomes REJECTED. Reviews where all prompts agree
yield no pair (nothing to prefer).

Provenance: these are SILVER preferences - constructed from LLM-generated
labels by consensus, not from human judgments. Every output row records
``label_type: "silver_preference"`` and the generator prompt ids, and the
per-field consensus scores are kept for transparency.

Splits by unique review ID (``split`` field, 80/20 by default) so the same
review never appears in both the DPO training set and the evaluation set.

Usage:
    python -m src.alignment.build_preferences \
        --labels data_local/labeled/silver_standard_crossval.jsonl \
        --out data_local/alignment/preferences.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LABELS = REPO_ROOT / 'data_local/labeled/silver_standard_crossval.jsonl'
DEFAULT_OUT = REPO_ROOT / 'data_local/alignment/preferences.jsonl'

THEMES_KEY = 'themes'


def _theme_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _sentiment_score(a: int, b: int) -> float:
    """1.0 exact, 0.5 within one step on the 1-5 scale, else 0.0."""
    diff = abs(a - b)
    return 1.0 if diff == 0 else 0.5 if diff == 1 else 0.0


def _consensus(candidate: Dict, others: List[Dict]) -> float:
    """Mean per-field agreement between one candidate and the rest."""
    scores = []
    for other in others:
        s = _sentiment_score(int(candidate['sentiment']), int(other['sentiment']))
        r = 1.0 if candidate['retention_risk'] == other['retention_risk'] else 0.0
        t = _theme_jaccard(candidate.get(THEMES_KEY, []), other.get(THEMES_KEY, []))
        scores.append(statistics.mean([s, r, t]))
    return statistics.mean(scores)


def _response_json(labels: Dict) -> str:
    """Render a label set as the JSON response string the model should emit."""
    return json.dumps({
        'sentiment': int(labels['sentiment']),
        'themes': list(labels.get(THEMES_KEY, [])),
        'retention_risk': str(labels['retention_risk']),
    })


def build_pairs(rows: List[Dict], eval_fraction: float = 0.2,
                min_prompts: int = 2) -> Tuple[List[Dict], Dict]:
    """Build one chosen/rejected pair per review with generator disagreement.

    Returns (pairs, stats). Pairs from every ``min_prompts``-th distinct
    generator prompts onward are considered; reviews with unanimous labels
    are skipped and counted.
    """
    by_review: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        labels = row.get('auto_labels') or row.get('labels')
        if not labels or not row.get('prompt_version') or not row.get('review_id'):
            continue
        by_review[row['review_id']].append({
            'prompt': row['prompt_version'],
            'labels': {
                'sentiment': int(labels['sentiment']),
                'themes': list(labels.get(THEMES_KEY, [])),
                'retention_risk': str(labels['retention_risk']),
            },
            'text': row.get('text', ''),
            'model': row.get('model'),
        })

    pairs: List[Dict] = []
    unanimous = skipped_insufficient = 0
    for review_id, candidates in by_review.items():
        if len(candidates) < min_prompts:
            skipped_insufficient += 1
            continue
        scored = [(c, _consensus(c['labels'], [o['labels'] for o in candidates
                                               if o is not c]))
                  for c in candidates]
        scored.sort(key=lambda cs: cs[1], reverse=True)
        best, worst = scored[0], scored[-1]
        if best[0]['labels'] == worst[0]['labels'] or best[1] <= worst[1]:
            unanimous += 1
            continue
        pairs.append({
            'review_id': review_id,
            'review_text': best[0]['text'],
            'chosen': _response_json(best[0]['labels']),
            'rejected': _response_json(worst[0]['labels']),
            'chosen_from': best[0]['prompt'],
            'rejected_from': worst[0]['prompt'],
            'chosen_consensus': round(best[1], 4),
            'rejected_consensus': round(worst[1], 4),
            'generator_models': sorted({c['model'] for c in candidates if c['model']}),
            'label_type': 'silver_preference',
        })

    # Deterministic 80/20 split by unique review ID (sort, then slice).
    pairs.sort(key=lambda p: p['review_id'])
    n_eval = round(len(pairs) * eval_fraction)
    for i, pair in enumerate(pairs):
        pair['split'] = 'eval' if i < n_eval else 'train'

    stats = {
        'reviews_with_labels': len(by_review),
        'pairs_built': len(pairs),
        'train_pairs': sum(1 for p in pairs if p['split'] == 'train'),
        'eval_pairs': sum(1 for p in pairs if p['split'] == 'eval'),
        'unanimous_reviews_skipped': unanimous,
        'insufficient_labels_skipped': skipped_insufficient,
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': 'silver_preference: consensus of LLM-generated labels '
                      '(no human preference data)',
    }
    return pairs, stats


def main():
    parser = argparse.ArgumentParser(
        description='Build DPO silver-preference pairs from crossval labels')
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--eval-fraction', type=float, default=0.2)
    args = parser.parse_args()

    if not args.labels.exists():
        parser.error(f'Labels not found: {args.labels}\n'
                     '(the real crossval file lives in gitignored data_local/;'
                     ' the tracked synthetic fixture also works for a smoke run)')

    rows = [json.loads(line) for line in args.labels.read_text().splitlines()
            if line.strip()]
    pairs, stats = build_pairs(rows, eval_fraction=args.eval_fraction)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(json.dumps(stats, indent=2))
    print(f'Pairs written to {args.out}')
    if stats['pairs_built'] < 30:
        print('WARNING: few disagreement pairs - DPO signal will be weak; '
              'consider reporting that limitation prominently.')


if __name__ == '__main__':
    main()

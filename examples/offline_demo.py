#!/usr/bin/env python3
"""
Offline end-to-end demo — runs the full pipeline on synthetic fixtures with
no API key and no network:

  1. Inter-prompt agreement across the synthetic silver labels
     (leave-one-prompt-out style: a prompt is never compared to itself).
  2. TF-IDF context retrieval over the synthetic review corpus.
  3. Agent memory + KL-divergence drift detection.

Run:  python examples/offline_demo.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

CROSSVAL = REPO_ROOT / 'data/labeled/silver_standard_crossval.jsonl'
REVIEWS = REPO_ROOT / 'data/processed/reviews_final.jsonl'


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def demo_inter_prompt_agreement():
    section("1. INTER-PROMPT AGREEMENT (synthetic silver labels)")
    by_review = defaultdict(dict)
    with open(CROSSVAL) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_review[row['review_id']][row['prompt_version']] = row['auto_labels']

    prompts = sorted({pv for labels in by_review.values() for pv in labels})
    print(f"Reviews: {len(by_review)} | label prompts: {prompts}")
    print("(leave-one-prompt-out: each pair excludes the reviewed prompt's own labels)\n")

    header = f"{'prompt':<8}" + "".join(f"{'vs ' + p:^18}" for p in prompts)
    print(header)
    for a in prompts:
        line = f"{a:<8}"
        for b in prompts:
            if a == b:
                line += f"{'—':^18}"
                continue
            sent = risk = n = 0
            for labels in by_review.values():
                if a not in labels or b not in labels:
                    continue
                n += 1
                sent += labels[a]['sentiment'] == labels[b]['sentiment']
                risk += labels[a]['retention_risk'] == labels[b]['retention_risk']
            line += f"{'s %d%% / r %d%%' % (100 * sent / n, 100 * risk / n):^18}"
        print(line)
    print("\nThese are agreement rates between two LLM-style label sets on 9 "
          "synthetic\nreviews — a demonstration of the metric machinery, not "
          "model accuracy.")


def demo_retrieval():
    section("2. TF-IDF CONTEXT RETRIEVAL (synthetic corpus)")
    from agent.embeddings import SimpleVectorStore

    store = SimpleVectorStore(reviews_path=REVIEWS)
    store.load_from_files()
    for query in ("mandatory overtime complaints", "safety reports ignored"):
        results = store.search(query, k=2)
        print(f"\nQuery: {query!r}")
        for r in results:
            print(f"  {r['similarity']:.3f} [{r['source']}] "
                  f"{r['text'][:60]}...")


def demo_memory_and_drift():
    section("3. AGENT MEMORY + DRIFT DETECTION (offline)")
    from agent.drift import DriftDetector
    from agent.memory import AgentMemory

    memory = AgentMemory(memory_dir=str(REPO_ROOT / 'data' / 'memory' / 'demo'))
    synthetic = [
        ('demo_1', {'sentiment': 2, 'themes': ['overtime', 'management'],
                    'retention_risk': 'high', 'is_anomalous': False}),
        ('demo_2', {'sentiment': 1, 'themes': ['safety', 'overtime'],
                    'retention_risk': 'high', 'is_anomalous': True}),
        ('demo_3', {'sentiment': 4, 'themes': ['pay_benefits', 'culture'],
                    'retention_risk': 'low', 'is_anomalous': False}),
        ('demo_4', {'sentiment': 3, 'themes': ['overtime', 'pay_benefits'],
                    'retention_risk': 'medium', 'is_anomalous': False}),
    ]
    for rid, analysis in synthetic:
        memory.record_analysis(rid, analysis)

    stats = memory.get_summary_stats()
    print(f"Analyses recorded: {stats['total_analyses']}")
    print(f"Theme distribution: "
          f"{dict(sorted(stats['theme_distribution'].items(), key=lambda kv: -kv[1]))}")
    print(f"Anomaly rate: {stats['anomaly_rate']:.0%}")

    baseline = {'overtime': 0.15, 'management': 0.28, 'safety': 0.12,
                'pay_benefits': 0.35, 'culture': 0.10}
    current = stats['theme_distribution']
    drift = DriftDetector(baseline).detect_drift(current, threshold=0.10)
    print(f"Drift vs. baseline: KL={drift['kl_divergence']:.3f} -> "
          f"{'DRIFTING' if drift['is_drifting'] else 'stable'}")
    print("(LLM-assigned classifications over synthetic demos — not observed "
          "employee outcomes.)")


if __name__ == '__main__':
    demo_inter_prompt_agreement()
    demo_retrieval()
    demo_memory_and_drift()
    print(f"\n{'=' * 64}\nOffline demo complete — no API key was used.\n{'=' * 64}")

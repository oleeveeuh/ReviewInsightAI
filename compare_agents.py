#!/usr/bin/env python3
"""
Compare custom ReAct agent vs LangChain agent implementation.

This script runs both agents on the same reviews and compares:
- Performance (latency)
- Output quality (metrics)
- Token usage
- Error rates

Usage:
    python compare_agents.py --reviews data/labeled/silver_standard.jsonl --samples 10
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent / 'src'))

def compare_agents():
    """Run comparison between custom and LangChain agents."""

    parser = argparse.ArgumentParser(description="Compare agent implementations")
    parser.add_argument('--reviews', type=str,
                       default='data/labeled/silver_standard.jsonl',
                       help='Path to reviews JSONL')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of reviews to test')
    parser.add_argument('--output', type=str,
                       default='data/evaluation/agent_comparison.json',
                       help='Output path for comparison results')

    args = parser.parse_args()

    # Load reviews
    reviews = []
    with open(args.reviews, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                reviews.append(data.get('text', data.get('review', '')))

    reviews = reviews[:args.samples]

    print(f"\n{'='*70}")
    print(f"AGENT COMPARISON: Custom ReAct vs LangChain")
    print(f"{'='*70}")
    print(f"Reviews: {len(reviews)}")
    print(f"Output: {args.output}\n")

    # Import agents
    try:
        from agent.controller import ReviewAgent
        custom_available = True
        print("✅ Custom agent available")
    except ImportError as e:
        custom_available = False
        print(f"❌ Custom agent not available: {e}")

    try:
        from agent.langchain_agent import LangChainAgent
        langchain_available = True
        print("✅ LangChain agent available")
    except ImportError as e:
        langchain_available = False
        print(f"❌ LangChain agent not available: {e}")

    if not custom_available and not langchain_available:
        print("\n❌ No agents available for comparison")
        return

    # Initialize agents
    agents = {}
    if custom_available:
        try:
            agents['custom'] = ReviewAgent()
            print("✅ Custom agent initialized")
        except Exception as e:
            print(f"⚠️  Custom agent initialization failed: {e}")

    if langchain_available:
        try:
            agents['langchain'] = LangChainAgent()
            print("✅ LangChain agent initialized")
        except Exception as e:
            print(f"⚠️  LangChain agent initialization failed: {e}")

    # Run comparison
    results = {
        'num_reviews': len(reviews),
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'agents': {}
    }

    for agent_name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"Testing {agent_name.upper()} agent")
        print(f"{'='*70}")

        agent_results = {
            'latencies': [],
            'token_usage': [],
            'errors': 0,
            'outputs': []
        }

        for i, review in enumerate(reviews):
            print(f"\n[{i+1}/{len(reviews)}] Analyzing...")

            try:
                start = time.time()

                result = agent.analyze(review)

                latency = time.time() - start
                agent_results['latencies'].append(latency)

                # Count tokens (rough estimate)
                input_tokens = len(review.split()) * 1.3  # Rough estimate
                if 'final_analysis' in result:
                    output_text = str(result['final_analysis'])
                else:
                    output_text = str(result)
                output_tokens = len(output_text.split()) * 1.3

                agent_results['token_usage'].append({
                    'input': input_tokens,
                    'output': output_tokens,
                    'total': input_tokens + output_tokens
                })

                agent_results['outputs'].append(result)
                print(f"  ✓ Latency: {latency:.2f}s")

            except Exception as e:
                agent_results['errors'] += 1
                print(f"  ✗ Error: {e}")

        # Calculate summary stats
        results['agents'][agent_name] = {
            'avg_latency': sum(agent_results['latencies']) / len(agent_results['latencies']),
            'total_tokens': sum(t['total'] for t in agent_results['token_usage']),
            'avg_tokens_per_review': sum(t['total'] for t in agent_results['token_usage']) / len(agent_results['token_usage']),
            'error_rate': agent_results['errors'] / len(reviews),
            'success_rate': (len(reviews) - agent_results['errors']) / len(reviews)
        }

        print(f"\nSummary:")
        print(f"  Avg Latency: {results['agents'][agent_name]['avg_latency']:.2f}s")
        print(f"  Avg Tokens: {results['agents'][agent_name]['avg_tokens_per_review']:.0f}")
        print(f"  Error Rate: {results['agents'][agent_name]['error_rate']:.1%}")
        print(f"  Success Rate: {results['agents'][agent_name]['success_rate']:.1%}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Print comparison table
    if len(results['agents']) > 1:
        print(f"\n{'='*70}")
        print(f"COMPARISON TABLE")
        print(f"{'='*70}\n")

        print(f"{'Metric':<25} {'Custom':<15} {'LangChain':<15}")
        print(f"{'-'*55}")

        for metric in ['avg_latency', 'avg_tokens_per_review', 'success_rate']:
            custom_val = results['agents'].get('custom', {}).get(metric, 0)
            langchain_val = results['agents'].get('langchain', {}).get(metric, 0)

            if metric == 'avg_latency':
                print(f"{'Avg Latency (s)':<25} {custom_val:<15.2f} {langchain_val:<15.2f}")
            elif metric == 'avg_tokens_per_review':
                print(f"{'Avg Tokens':<25} {custom_val:<15.0f} {langchain_val:<15.0f}")
            elif metric == 'success_rate':
                print(f"{'Success Rate':<25} {custom_val*100:<14.1f}% {langchain_val*100:<14.1f}%")


if __name__ == '__main__':
    compare_agents()

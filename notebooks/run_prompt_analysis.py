#!/usr/bin/env python3
"""
Analyze prompt engineering evaluation results.
Standalone script version of 02_prompt_engineering.ipynb
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("PROMPT ENGINEERING ANALYSIS")
    print("=" * 60)

    # Load results
    results_path = DATA_DIR / 'prompt_comparison.json'

    if not results_path.exists():
        print(f"\nResults file not found at {results_path}")
        print("Run the evaluation first: python src/evaluation/evaluate_prompts.py")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'Prompt': r['prompt_name'],
            'Version': r['prompt_version'],
            'K-Shot': r['k_shot'],
            'Model': r.get('model', 'gpt-4o-mini'),
            'Sentiment MAE': r['metrics'].get('sentiment_mae', 0),
            'Sentiment Acc': r['metrics'].get('sentiment_exact_accuracy', 0),
            'Theme F1': r['metrics'].get('theme_f1', 0),
            'Theme Precision': r['metrics'].get('theme_precision', 0),
            'Theme Recall': r['metrics'].get('theme_recall', 0),
            'Risk F1': r['metrics'].get('risk_f1', 0),
            'Risk Acc': r['metrics'].get('risk_accuracy', 0),
            'Cost per 1K': r['metrics'].get('cost_per_1k_samples', 0)
        }
        for r in results
    ])

    print(f"\nTotal configurations tested: {len(df)}")
    print("\n" + "-" * 60)
    print(df.to_string(index=False))

    # Best performers
    print("\n\n" + "=" * 60)
    print("BEST PERFORMERS")
    print("=" * 60)

    best_theme_idx = df['Theme F1'].idxmax()
    print(f"\nBest Theme F1: {df.loc[best_theme_idx, 'Prompt']} "
          f"(k={df.loc[best_theme_idx, 'K-Shot']}) = {df['Theme F1'].max():.1%}")

    best_mae_idx = df['Sentiment MAE'].idxmin()
    print(f"Best Sentiment MAE: {df.loc[best_mae_idx, 'Prompt']} "
          f"(k={df.loc[best_mae_idx, 'K-Shot']}) = {df['Sentiment MAE'].min():.3f}")

    best_risk_idx = df['Risk F1'].idxmax()
    print(f"Best Risk F1: {df.loc[best_risk_idx, 'Prompt']} "
          f"(k={df.loc[best_risk_idx, 'K-Shot']}) = {df['Risk F1'].max():.1%}")

    cheapest_idx = df['Cost per 1K'].idxmin()
    print(f"Lowest Cost: {df.loc[cheapest_idx, 'Prompt']} "
          f"= ${df['Cost per 1K'].min():.2f} per 1K samples")

    # Visualization 1: Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(df))
    labels = [f"{row['Prompt'][:12]}\n(k={int(row['K-Shot'])})"
              for _, row in df.iterrows()]

    # Theme F1
    ax = axes[0]
    ax.bar(x, df['Theme F1'] * 100, color='steelblue', alpha=0.7)
    ax.set_ylabel('Theme F1 (%)')
    ax.set_title('Theme Classification F1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=df['Theme F1'].max() * 100, color='r', linestyle='--', alpha=0.5, label='Best')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Sentiment MAE
    ax = axes[1]
    ax.bar(x, df['Sentiment MAE'], color='coral', alpha=0.7)
    ax.set_ylabel('Sentiment MAE (lower is better)')
    ax.set_title('Sentiment Error')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=df['Sentiment MAE'].min(), color='r', linestyle='--', alpha=0.5, label='Best')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Risk F1
    ax = axes[2]
    ax.bar(x, df['Risk F1'] * 100, color='mediumseagreen', alpha=0.7)
    ax.set_ylabel('Risk F1 (%)')
    ax.set_title('Retention Risk F1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=df['Risk F1'].max() * 100, color='r', linestyle='--', alpha=0.5, label='Best')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'prompt_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: prompt_performance.png")

    # Visualization 2: Cost vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(df['Cost per 1K'], df['Theme F1'] * 100,
                        s=200, alpha=0.6, c=df['K-Shot'], cmap='viridis',
                        edgecolors='black', linewidth=1)

    for idx, row in df.iterrows():
        ax.annotate(f"{row['Prompt'][:10]}\n(k={int(row['K-Shot'])})",
                    (row['Cost per 1K'], row['Theme F1'] * 100),
                    fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Cost per 1,000 Samples ($)')
    ax.set_ylabel('Theme F1 (%)')
    ax.set_title('Cost vs Performance Trade-off')
    plt.colorbar(scatter, label='K-Shot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'cost_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: cost_performance.png")

    # Visualization 3: Few-Shot Impact
    few_shot_comparison = []
    for prompt in df['Prompt'].unique():
        prompt_data = df[df['Prompt'] == prompt]
        zero_shot = prompt_data[prompt_data['K-Shot'] == 0]
        few_shot = prompt_data[prompt_data['K-Shot'] > 0]

        if len(zero_shot) > 0 and len(few_shot) > 0:
            few_shot_comparison.append({
                'Prompt': prompt,
                'Zero-Shot F1': zero_shot['Theme F1'].iloc[0],
                'Few-Shot F1': few_shot['Theme F1'].iloc[0],
                'Improvement': few_shot['Theme F1'].iloc[0] - zero_shot['Theme F1'].iloc[0]
            })

    if few_shot_comparison:
        df_fs = pd.DataFrame(few_shot_comparison)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df_fs))
        width = 0.35

        ax.bar(x - width/2, df_fs['Zero-Shot F1'] * 100, width,
               label='Zero-Shot', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, df_fs['Few-Shot F1'] * 100, width,
               label='Few-Shot (k=3)', alpha=0.8, color='coral')

        for i, row in df_fs.iterrows():
            improvement = row['Improvement'] * 100
            y_pos = max(row['Zero-Shot F1'], row['Few-Shot F1']) * 100 + 2
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'+{improvement:.1f}pp' if improvement > 0 else f'{improvement:.1f}pp',
                       xy=(i, y_pos), ha='center', fontsize=9,
                       color=color, fontweight='bold')

        ax.set_ylabel('Theme F1 (%)')
        ax.set_title('Impact of Few-Shot Learning')
        ax.set_xticks(x)
        ax.set_xticklabels(df_fs['Prompt'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'few_shot_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: few_shot_impact.png")

    # Statistical summary
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print("\nTheme F1 by K-Shot:")
    print(df.groupby('K-Shot')['Theme F1'].agg(['mean', 'std', 'min', 'max']))

    print("\nCost by K-Shot:")
    cost_summary = df.groupby('K-Shot')['Cost per 1K'].agg(['mean', 'std', 'min', 'max'])
    for k, row in cost_summary.iterrows():
        print(f"  k={int(k)}: mean=${row['mean']:.2f}, std=${row['std']:.2f}")

    # Final recommendation
    best_config = df.loc[df['Theme F1'].idxmax()]

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    print("\nOptimal Configuration:")
    print(f"  Prompt: {best_config['Prompt']}")
    print(f"  Version: {best_config['Version']}")
    print(f"  K-Shot: {int(best_config['K-Shot'])}")
    print(f"  Theme F1: {best_config['Theme F1']:.1%}")
    print(f"  Sentiment MAE: {best_config['Sentiment MAE']:.3f}")
    print(f"  Risk F1: {best_config['Risk F1']:.1%}")
    print(f"  Cost per 1K: ${best_config['Cost per 1K']:.2f}")

    # Compare to baseline
    baseline = df[(df['Version'] == 'v1.0') & (df['K-Shot'] == 0)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        improvement = (best_config['Theme F1'] - baseline['Theme F1']) * 100

        print("\nImprovement over baseline:")
        print(f"  Baseline (v1 zero-shot): {baseline['Theme F1']:.1%}")
        print(f"  Best config: {best_config['Theme F1']:.1%}")
        print(f"  Improvement: +{improvement:.1f} percentage points")

    # Save summary
    summary_path = Path(__file__).parent.parent / "reports" / "prompt_engineering_summary.md"
    with open(summary_path, 'w') as f:
        f.write(f"""# Prompt Engineering Results Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Recommended Configuration

- **Prompt**: {best_config['Prompt']}
- **Version**: {best_config['Version']}
- **K-Shot**: {int(best_config['K-Shot'])}

### Performance

| Metric | Value |
|--------|-------|
| Theme F1 | {best_config['Theme F1']:.1%} |
| Theme Precision | {best_config['Theme Precision']:.1%} |
| Theme Recall | {best_config['Theme Recall']:.1%} |
| Sentiment MAE | {best_config['Sentiment MAE']:.3f} |
| Sentiment Accuracy | {best_config['Sentiment Acc']:.1%} |
| Risk F1 | {best_config['Risk F1']:.1%} |

### Cost

- **Cost per 1,000 samples**: ${best_config['Cost per 1K']:.2f}

### Generated Visualizations

- `prompt_performance.png` - Side-by-side metric comparison
- `cost_performance.png` - Cost vs performance scatter plot
- `few_shot_impact.png` - Zero-shot vs few-shot comparison
""")

    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()

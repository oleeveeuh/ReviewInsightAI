# Prompt Engineering Analysis Guide

## Overview

This notebook/script analyzes results from prompt evaluation experiments to determine the best performing configuration.

## Files

| File | Description |
|------|-------------|
| [02_prompt_engineering.ipynb](notebooks/02_prompt_engineering.ipynb) | Jupyter notebook |
| [run_prompt_analysis.py](notebooks/run_prompt_analysis.py) | Standalone Python script |

## Usage

### Option 1: Jupyter Notebook
```bash
jupyter notebook notebooks/02_prompt_engineering.ipynb
```

### Option 2: Standalone Script
```bash
python notebooks/run_prompt_analysis.py
```

## Prerequisites

The analysis requires evaluation results from:
```bash
python src/evaluation/evaluate_prompts.py
```

This creates `data/evaluation/prompt_comparison.json`

## Output

### Visualizations

| File | Description |
|------|-------------|
| `prompt_performance.png` | Side-by-side comparison of all metrics |
| `cost_performance.png` | Cost vs F1 score scatter plot |
| `few_shot_impact.png` | Zero-shot vs few-shot comparison |

### Report

| File | Description |
|------|-------------|
| `prompt_engineering_summary.md` | Key findings and recommendation |

## What Gets Analyzed

1. **Best Performers** - Which prompt version scored highest
2. **Cost vs Performance** - Pareto frontier analysis
3. **Few-Shot Impact** - Does adding examples help?
4. **Statistical Summary** - Mean, std, min, max by K-Shot
5. **Final Recommendation** - Optimal configuration

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Sentiment MAE** | Mean Absolute Error (lower is better) |
| **Sentiment Acc** | Exact match accuracy |
| **Theme F1** | Harmonic mean of precision/recall |
| **Theme Precision** | Of predicted themes, how many are correct? |
| **Theme Recall** | Of actual themes, how many were found? |
| **Risk F1** | Retention risk classification F1 |
| **Cost per 1K** | API cost for 1,000 samples |

## Interpreting Results

### Good Performance Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| Theme F1 | > 60% | > 75% |
| Sentiment MAE | < 0.7 | < 0.5 |
| Risk F1 | > 60% | > 75% |

### Cost Considerations

| Model | Cost per 1K samples |
|-------|---------------------|
| gpt-4o-mini | $0.50 - $2.00 |
| gpt-4o | $8.00 - $30.00 |

## Typical Findings

From similar prompt engineering tasks:

1. **Few-shot helps** - Usually 5-15% F1 improvement
2. **Domain context matters** - V2 (role-enhanced) beats V1 (zero-shot)
3. **Diminishing returns** - k=3 is usually optimal, k>3 adds cost without much gain
4. **Cost-performance tradeoff** - Sometimes simpler prompts are "good enough"

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Label Data                                              │
│     python src/labeling/label_reviews.py --target 50       │
├─────────────────────────────────────────────────────────────┤
│  2. Evaluate Prompts                                        │
│     python src/evaluation/evaluate_prompts.py               │
├─────────────────────────────────────────────────────────────┤
│  3. Analyze Results                                        │
│     python notebooks/run_prompt_analysis.py                 │
├─────────────────────────────────────────────────────────────┤
│  4. Select Best Prompt                                     │
│     Use recommendation from analysis                        │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps After Analysis

1. **Select optimal prompt** based on your priorities:
   - Best accuracy? Use highest F1
   - Best cost? Use cheapest with acceptable accuracy
   - Balance? Use Pareto optimal

2. **Update production code** to use selected prompt version

3. **Monitor performance** on new data over time

4. **Iterate** - Modify prompts based on failure cases

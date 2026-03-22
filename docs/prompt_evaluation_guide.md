# Prompt Evaluation Guide

## Overview

The prompt evaluation system compares different prompt versions against labeled gold standard data.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `openai` - OpenAI API client
- `scikit-learn` - Evaluation metrics
- `python-dotenv` - Environment variable management

### 2. Set API Key

Create `.env` file in project root:

```bash
OPENAI_API_KEY=your_key_here
```

Or set environment variable:

```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Create Gold Standard Labels

```bash
python src/labeling/label_reviews.py --target 50
```

Target: 50-100 labeled reviews for evaluation.

## Usage

### Dry Run (Test without API calls)

```bash
python src/evaluation/evaluate_prompts.py --dry-run
```

### Evaluate Specific Prompts

```bash
# Test v1, v2, v3 with zero-shot
python src/evaluation/evaluate_prompts.py --prompts v1 v2 v3 --k-shots 0

# Test with 3-shot learning
python src/evaluation/evaluate_prompts.py --prompts v2 v3 --k-shots 3

# Full evaluation
python src/evaluation/evaluate_prompts.py --prompts v1 v2 v3 v5 --k-shots 0 3
```

### Custom Output Path

```bash
python src/evaluation/evaluate_prompts.py --output results/my_run.json
```

## Prompt Versions

| Version | Name | Description |
|---------|------|-------------|
| v1 | Zero-shot Basic | Minimal prompt, baseline |
| v2 | Role Enhanced | HR analyst role + warehouse knowledge |
| v3 | 3-Shot Learning | V2 + few-shot examples |
| v4 | Chain of Thought | Step-by-step reasoning |
| v5 | Structured Output | Explicit JSON schema |
| v6 | Detailed Analysis | Comprehensive with key insights |

## Metrics

### Sentiment
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **Exact Accuracy**: Exact match percentage
- **Direction Accuracy**: Positive vs Negative correctly classified

### Themes
- **Precision**: Of predicted themes, how many are correct?
- **Recall**: Of actual themes, how many were found?
- **F1**: Harmonic mean of precision and recall
- **Exact Match**: All themes match exactly

### Retention Risk
- **Accuracy**: Exact match (low/medium/high)
- **Within One**: Adjacent risk levels count as match

### Cost
- **Per sample**: Average API cost per review
- **Per 1K samples**: Projected cost for 1,000 reviews

## Output Files

| File | Description |
|------|-------------|
| `prompt_comparison.json` | Summary metrics |
| `prompt_comparison_detailed.json` | Full predictions |
| `comparison.csv` | Quick comparison table |

## Interpreting Results

### Good Performance

| Metric | Target |
|--------|--------|
| Sentiment MAE | < 0.5 |
| Sentiment Accuracy | > 70% |
| Theme F1 | > 60% |
| Risk Accuracy | > 60% |

### Cost Considerations

| Model | Input | Output | Est. cost/1000 reviews |
|-------|-------|--------|----------------------|
| gpt-4o-mini | $0.15/M | $0.60/M | ~$0.50-$2.00 |
| gpt-4o | $2.50/M | $10/M | ~$8-$30 |

## Workflow

1. **Label 50-100 reviews** using `src/labeling/label_reviews.py`
2. **Dry run evaluation** to validate pipeline
3. **Run full evaluation** with API calls
4. **Compare results** in `comparison.csv`
5. **Select best prompt** based on accuracy/cost tradeoff
6. **Iterate** - modify prompts in `src/prompts/templates.py`

## Troubleshooting

**"No gold standard data found"**
- Run the labeling tool first
- Check path: `data/labeled/gold_standard.jsonl`

**API errors**
- Verify `OPENAI_API_KEY` is set
- Check API quota/billing
- Use `--dry-run` to test without API

**Parse errors**
- LLM returned invalid JSON
- Check prompt template output format
- May need to add post-processing

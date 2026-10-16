# Alignment Extension: DPO on Silver Preferences

## What this is — and what it is not

This extension fine-tunes a small open-weights language model with **Direct
Preference Optimization (DPO)** so that its review-labeling outputs move
toward the consensus of the original LLM generator prompts.

| Claim | Status |
|---|---|
| "DPO training with LoRA on an open-weights model" | ✅ accurate |
| "Preference-based alignment / preference tuning" | ✅ accurate |
| "RLHF" | ❌ no reward model, no PPO — DPO skips both by design |
| "Trained on human preferences" | ❌ preferences are constructed from LLM label consensus |
| "Fine-tuned gpt-4o-mini" | ❌ impossible (closed weights); the base model is Qwen2.5, trained from scratch here |

Every preference row carries `label_type: "silver_preference"`, the source
prompt ids, and consensus scores. **`data_local/alignment/` holds all
training data and outputs — none of it is tracked in git.**

## Pipeline

```
silver_standard_crossval.jsonl   (175 reviews × v2/v3/v6 = 525 silver labels)
        │  build_preferences.py
        │  chosen = consensus-closest label set · rejected = outlier
        ▼
preferences.jsonl   (137 pairs; 38 unanimous reviews skipped;
        │            80/20 split by unique review ID → 110 train / 27 eval)
        │  train_dpo.py   (TRL DPOTrainer + PEFT LoRA, Qwen2.5-0.5B-Instruct)
        ▼
dpo_adapter/   (LoRA adapter only — base weights untouched)
        │  evaluate_model.py
        ▼
eval_report.json   (base vs tuned agreement on the 27 held-out pairs)
```

Design choices worth defending in an interview:

- **Consensus-as-preference**: where two generator prompts agree and the third
  disagrees, agreement is a cheap, replicable preference signal. It is *not*
  a substitute for human preference — it teaches the model to match the
  generator ensemble, bias included.
- **Splits by review ID**: the 27 eval pairs come from reviews never seen in
  training, same discipline as the prompt-evaluation harness.
- **LoRA adapter, not full weights**: 16 GB of unified memory trains a 0.5B
  model in fp32 comfortably; the adapter is a few MB and keeps the base model
  reusable.
- **Evaluation reuses the main harness**: theme F1, sentiment exact agreement,
  and risk agreement — the same `calculate_metrics` used everywhere else, with
  the same "agreement, not accuracy" caveat.

## Commands

```bash
# 1. Build preference pairs (works on real data in data_local/ or on the
#    tracked synthetic fixture for a smoke run)
.venv/bin/python -m src.alignment.build_preferences

# 2. Training environment (separate venv; ~2 GB of packages)
python3.11 -m venv .venv-rl
.venv-rl/bin/pip install -r requirements-training.txt -r requirements.txt

# 3. Smoke run (2 minutes) before the real run
.venv-rl/bin/python -m src.alignment.train_dpo --max-pairs 8 --epochs 1 \
    --out data_local/alignment/dpo_adapter_smoke

# 4. Full run — pick one profile:
#
#    Full fidelity (no truncation; ~40-60 min on a cool M4, several hours
#    if the machine is heat-soaked):
.venv-rl/bin/python -m src.alignment.train_dpo --epochs 2
#
#    Efficient profile (~25-35 min on a cool M4): truncates review text
#    to 300 tokens (the truncation rate is printed - REPORT it alongside
#    results) and doubles the batch size:
.venv-rl/bin/python -m src.alignment.train_dpo --epochs 2 \
    --max-review-tokens 300 --batch-size 2

# 5. Base-vs-tuned agreement on the held-out split
.venv-rl/bin/python -m src.alignment.evaluate_model
```

## Results (measured 2026-10-16)

Run: efficient profile (`--epochs 2 --max-review-tokens 300 --batch-size 2`,
16% of training pairs truncated; 26 steps, final DPO loss 0.669 from 0.693
initial). Evaluation: 27 review-disjoint held-out pairs, reference = consensus
(chosen) labels; 0 parse failures for either model.

| Metric | Base | DPO-tuned | Δ |
|---|---|---|---|
| Risk-class agreement | 40.7% | 74.1% | +33.3 pp (9/27 reviews) |
| Sentiment exact agreement | 40.7% | 40.7% | 0.0 |
| Theme F1 | 39.4% | 34.4% | −5.0 pp (within noise) |
| Risk within-one | 100% | 100% | 0.0 |

Interpretation: risk-class alignment improved markedly (the coarsest field,
and the one most exposed to the consensus-circularity caveat below); sentiment
behavior was unchanged; theme F1 moved within noise. With n=27 the confidence
interval on any proportion is roughly ±18 pp — report deltas with that
caveat. These are agreement shifts with an LLM consensus labeler, not
accuracy against ground truth.

## Known limitations

- **Silver preferences**: the tuned model converges toward the LLM ensemble's
  biases, not toward human judgment. A human-labeled preference set is the
  obvious next step (`src/labeling/label_reviews.py` is the entry point).
- **Small eval split**: 27 held-out pairs; every percentage point is ~⅓ of a
  review. Report deltas, not point estimates.
- **Reference circularity risk**: tuning toward consensus and evaluating
  against consensus are related; the held-out split mitigates but cannot
  eliminate this. External (human) evaluation is the cure.
- **One seed, one run**: no variance estimates. Re-running with different
  seeds would be needed for any stronger claim.
- Training ran on Apple MPS in fp16 (fp32 LoRA adapters); results are not
  directly comparable to CUDA runs and no distributed/batched optimizations
  were used.
- If the efficient profile is used, the review-truncation rate it prints is
  part of the result and must be reported with it.
- Sustained GPU training on a 16 GB MacBook degrades step time roughly 10×
  as the machine heat-soaks (measured: 142 s/step cool vs ~16-35 min/step
  after hours of load, with heavy system swapping). Run cool and idle, or
  the epoch checkpoints (checkpoint-13, checkpoint-26) preserve progress.

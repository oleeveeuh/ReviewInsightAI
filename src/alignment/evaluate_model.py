#!/usr/bin/env python3
"""
Compare base vs DPO-tuned model agreement on the held-out eval pairs.

Reference labels are the consensus (chosen) labels of the held-out eval
split - i.e. agreement with the LLM consensus, NOT accuracy vs. human
ground truth. Metrics reuse the same functions as the prompt-evaluation
harness (theme F1, sentiment/risk agreement).

Usage:
    .venv-rl/bin/python -m src.alignment.evaluate_model \
        --pairs data_local/alignment/preferences.jsonl \
        --adapter data_local/alignment/dpo_adapter \
        --out data_local/alignment/eval_report.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

from src.alignment.train_dpo import DEFAULT_MODEL, check_dependencies  # noqa: E402


def load_eval_pairs(pairs_path: Path):
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines()
            if line.strip()]
    return [r for r in rows if r.get('split') == 'eval']


def generate_labels(model, tokenizer, rows, max_new_tokens=128):
    """Greedy-decode a label JSON for each eval pair; returns raw strings."""
    from prompts.templates import ALL_PROMPTS
    import torch

    template = ALL_PROMPTS['v2']
    device = next(model.parameters()).device
    outputs = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for row in rows:
            system, user = template.render(row['review_text'], k_shot=0)
            prompt = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': system},
                 {'role': 'user', 'content': user}],
                tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True)
            outputs.append(text)
    if was_training:
        model.train()
    return outputs


def score(rows, raw_outputs) -> dict:
    """Agreement of parsed outputs vs. consensus (chosen) reference labels."""
    from evaluation.evaluate_prompts import PromptEvaluator

    refs, preds = [], []
    unparsed = 0
    for row, raw in zip(rows, raw_outputs):
        parsed = PromptEvaluator.parse_llm_response(raw)
        if parsed is None:
            unparsed += 1
            continue
        refs.append({'auto_labels': json.loads(row['chosen'])})
        preds.append(parsed)

    if not preds:
        return {'error': 'no parseable outputs'}
    metrics = PromptEvaluator.calculate_metrics(refs, preds)
    metrics['n_eval_pairs'] = len(rows)
    metrics['n_unparsed'] = unparsed
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Base vs DPO adapter agreement on held-out eval pairs')
    parser.add_argument('--base-model', default=DEFAULT_MODEL)
    parser.add_argument('--pairs', type=Path,
                        default=REPO_ROOT / 'data_local/alignment/preferences.jsonl')
    parser.add_argument('--adapter', type=Path,
                        default=REPO_ROOT / 'data_local/alignment/dpo_adapter')
    parser.add_argument('--out', type=Path,
                        default=REPO_ROOT / 'data_local/alignment/eval_report.json')
    args = parser.parse_args()

    check_dependencies()
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_rows = load_eval_pairs(args.pairs)
    if not eval_rows:
        parser.error('No eval-split pairs found in the pairs file.')
    print(f'Eval pairs: {len(eval_rows)} (held out from training by review ID)')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print('\n== Base model ==')
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    base_metrics = score(eval_rows, generate_labels(base, tokenizer, eval_rows))
    print(json.dumps(base_metrics, indent=2))
    del base  # free memory before loading the adapter stack

    print('\n== DPO-tuned (base + adapter) ==')
    tuned = AutoPeftModelForCausalLM.from_pretrained(str(args.adapter),
                                                     torch_dtype=torch.float32)
    tuned_metrics = score(eval_rows, generate_labels(tuned, tokenizer, eval_rows))
    print(json.dumps(tuned_metrics, indent=2))

    report = {
        'what_this_is': 'Agreement of base vs LoRA-DPO-tuned model with the '
                        'LLM-consensus labels of held-out silver-preference '
                        'pairs. NOT accuracy vs. human ground truth; '
                        'preferences were constructed from LLM label '
                        'consensus (silver), not human judgments.',
        'base_model': args.base_model,
        'adapter': str(args.adapter),
        'pairs_file': str(args.pairs),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'base': base_metrics,
        'dpo_tuned': tuned_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'\nReport saved to {args.out}')


if __name__ == '__main__':
    main()

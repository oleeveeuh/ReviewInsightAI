#!/usr/bin/env python3
"""Story figures for the portfolio entry — reads frozen results JSONs only.

Three figures, fieldnotes theme, rendered at --scale 2 (~2200 px wide):

* contamination  — the same prompt measured twice: the feb 2026 grid's
  self-scored v5.0 k=0 (experiment_grid_final_20260223_011059.json, wine)
  vs the full-corpus leave-one-prompt-out re-measure
  (data/evaluation/rescored_crossval_summary.json, sage).
* leaderboard    — six prompts under leave-one-prompt-out scoring
  (rescored_crossval_summary.json): theme F1, sentiment agreement,
  risk agreement.
* dpo_shift      — base vs LoRA-DPO-tuned Qwen2.5-0.5B-Instruct on the 27
  review-disjoint held-out pairs
  (data_local/alignment/eval_report.json).

Writes PNGs only; never writes a results file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# fieldnotes theme (site palette): sage = positive/honest,
# wine/crimson = failure/negative, sumi = neutral baseline.
SURFACE = "#fbf9f4"
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#77776c"
GRID = "#d8d1c4"
EDGE = "#c9c2b2"
SAGE = "#89927a"
WINE = "#482728"
CRIMSON = "#a43b36"
SUMI = "#1E342E"

plt.rcParams["font.sans-serif"] = ["Source Sans 3", "Helvetica Neue", "Arial",
                                   "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["IBM Plex Mono", "Courier New",
                                  "DejaVu Sans Mono"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PROMPT_LABELS = {
    "v1.0": "v1.0 zero-shot basic",
    "v2.0": "v2.0 role enhanced",
    "v3.0": "v3.0 detailed + examples",
    "v4.0": "v4.0 chain of thought",
    "v5.0": "v5.0 structured output",
    "v6.0": "v6.0 detailed analysis",
}


def spaced(s: str) -> str:
    # matplotlib has no letter-spacing; mimic the site's mono eyebrows.
    return " ".join(s)


def load_crossval() -> dict:
    path = ROOT / "data" / "evaluation" / "rescored_crossval_summary.json"
    data = json.loads(path.read_text())
    return {r["prompt_version"]: r["rescored_agreement"] for r in data["results"]}


def load_grid_v5() -> dict:
    path = ROOT / "data" / "evaluation" / "experiment_grid_final_20260223_011059.json"
    runs = json.loads(path.read_text())
    for r in runs:
        if r["prompt_version"] == "v5.0" and r["k_shot"] == 0:
            return r["metrics"]
    raise SystemExit("v5.0 k_shot=0 run not found in frozen grid")


def load_eval_report() -> dict:
    path = ROOT / "data_local" / "alignment" / "eval_report.json"
    return json.loads(path.read_text())


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


# ---------------------------------------------------------------- contamination
def fig_contamination(out: Path) -> None:
    grid = load_grid_v5()
    cross = load_crossval()["v5.0"]
    metrics = [
        ("theme f1", grid["theme_f1"] * 100, cross["theme_f1"] * 100),
        ("sentiment exact agreement", grid["sentiment_exact_accuracy"] * 100,
         cross["sentiment_exact_agreement"] * 100),
        ("risk-class agreement", grid["risk_accuracy"] * 100,
         cross["risk_agreement"] * 100),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(metrics))[::-1].astype(float)
    bh = 0.30
    bars = ax.barh(y + bh / 2 + 0.02, [m[1] for m in metrics], height=bh,
                   color=WINE, zorder=3,
                   label="feb 2026 grid — v5 × v5's own labels (50-review slice)")
    ax.barh(y - bh / 2 - 0.02, [m[2] for m in metrics], height=bh,
            color=SAGE, zorder=3,
            label="oct 2026 — full corpus, leave-one-prompt-out (n = 175)")
    for yi, (_, g, c) in zip(y, metrics):
        ax.text(g + 1.2, yi + bh / 2 + 0.02, f"{g:.1f}%", va="center",
                fontsize=11, color=WINE, fontweight="bold")
        ax.text(c + 1.2, yi - bh / 2 - 0.02, f"{c:.1f}%", va="center",
                fontsize=11, color=INK2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.tick_params(length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(spaced("ONE PROMPT, MEASURED TWICE") + "  —  " + spaced("V5.0 AT K_SHOT = 0"),
                 fontsize=13, color=INK2, loc="left", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), fontsize=9.8,
              frameon=False)
    ax.text(0.0, -0.34, "the 30-run grid's reference labels came from v5 itself — "
            "unrescuable by re-scoring;\nthe re-measure excludes each prompt's "
            "own label rows",
            transform=ax.transAxes, fontsize=9, color=MUTED,
            fontfamily="IBM Plex Mono")
    save(fig, out)


# ------------------------------------------------------------------ leaderboard
def fig_leaderboard(out: Path) -> None:
    cross = load_crossval()
    panels = [
        ("theme f1", "theme_f1"),
        ("sentiment exact agreement", "sentiment_exact_agreement"),
        ("risk-class agreement", "risk_agreement"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), sharex=True)
    for ax, (title, key) in zip(axes, panels):
        items = sorted(cross.items(), key=lambda kv: kv[1][key])
        names = [PROMPT_LABELS[v] for v, _ in items]
        vals = [m[key] * 100 for _, m in items]
        worst = vals[0]
        colors = [SAGE if v > 60 else WINE for v in vals]
        bars = ax.barh(np.arange(len(vals)), vals, height=0.62, color=colors,
                       zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=10.5,
                    color=INK2 if colors[i] == SAGE else WINE,
                    fontweight="bold")
        ax.set_yticks(np.arange(len(items)))
        ax.set_yticklabels(names, fontsize=10.5)
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9.5)
        ax.tick_params(length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=8)
    fig.suptitle(spaced("SIX PROMPTS, ONE LABELER") + "  —  " +
                 spaced("LEAVE-ONE-PROMPT-OUT OVER 525 SILVER ROWS"),
                 x=0.012, y=0.99, ha="left", fontsize=13, color=INK2)
    fig.text(0.012, -0.04, "sage = usable agreement · wine = the near-zero "
             "theme f1 collapse — vocabulary mismatch:\nv1/v4 don't constrain "
             "themes to the fixed taxonomy · v2.0 and v3.0 are the same prompt "
             "at k_shot = 0\nthe 0.5 pp gap is sampling noise · 1,050 "
             "gpt-4o-mini calls, 0 parse failures",
             fontsize=9, color=MUTED, fontfamily="IBM Plex Mono")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, out)


# -------------------------------------------------------------------- dpo_shift
def fig_dpo_shift(out: Path) -> None:
    rep = load_eval_report()
    base, tuned = rep["base"], rep["dpo_tuned"]
    metrics = [
        ("risk-class\nagreement", base["risk_agreement"] * 100,
         tuned["risk_agreement"] * 100),
        ("sentiment exact\nagreement", base["sentiment_exact_agreement"] * 100,
         tuned["sentiment_exact_agreement"] * 100),
        ("theme f1", base["theme_f1"] * 100, tuned["theme_f1"] * 100),
    ]
    deltas = ["+33.3 pp", "0.0 pp", "−5.0 pp (within noise)"]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(metrics))
    bw = 0.32
    ax.bar(x - bw / 2 - 0.02, [m[1] for m in metrics], width=bw, color=SUMI,
           zorder=3, label="base Qwen2.5-0.5B-Instruct")
    ax.bar(x + bw / 2 + 0.02, [m[2] for m in metrics], width=bw, color=SAGE,
           zorder=3, label="LoRA-DPO tuned")
    for xi, (_, b, t), d in zip(x, metrics, deltas):
        ax.text(xi - bw / 2 - 0.02, b + 1.5, f"{b:.1f}%", ha="center",
                fontsize=11, color=INK2, fontweight="bold")
        ax.text(xi + bw / 2 + 0.02, t + 1.5, f"{t:.1f}%", ha="center",
                fontsize=11, color=INK2, fontweight="bold")
        dcol = SAGE if d.startswith("+") else (CRIMSON if d.startswith("−") else MUTED)
        ax.text(xi, max(b, t) + 9.5, d, ha="center", fontsize=11.5,
                color=dcol, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    ax.set_title(spaced("BASE VS TUNED") + "  —  " + spaced("27 REVIEW-DISJOINT HELD-OUT PAIRS"),
                 fontsize=13, color=INK2, loc="left", pad=12)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.text(0.0, -0.20, "agreement with an llm consensus labeler, not accuracy "
            "· n = 27 → every proportion\nroughly ±18 pp — report deltas, not "
            "point estimates · 0 parse failures for either model",
            transform=ax.transAxes, fontsize=9, color=MUTED,
            fontfamily="IBM Plex Mono")
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "figures" / "portfolio")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="render scale (2 = retina)")
    args = parser.parse_args()
    if args.scale != 1.0:
        plt.rcParams["figure.dpi"] = 150 * args.scale
    fig_contamination(args.out / "contamination.png")
    fig_leaderboard(args.out / "prompt_leaderboard.png")
    fig_dpo_shift(args.out / "dpo_shift.png")


if __name__ == "__main__":
    main()

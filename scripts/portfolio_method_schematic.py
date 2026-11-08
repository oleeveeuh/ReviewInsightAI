#!/usr/bin/env python3
"""Method schematic for the portfolio card: how ReviewInsight AI works.

One combined flow diagram (not two flat rows): three sources fan into the
merged corpus, and three consumer lanes fan out — the evaluation harness,
the alignment extension (fed by the labels, with the retired self-scoring
failure site marked wine-dashed), and the agent that reads the same corpus.
Counts, model names, and protocol values are real; no metrics are drawn.

Fieldnotes theme, Source Sans 3 / IBM Plex Mono, canvas padded to the
gallery art-box aspect at save time. Structure follows
~/retailPRED-satellite/scripts/method_schematic.py.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Source Sans 3", "Helvetica Neue", "Arial",
                                   "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["IBM Plex Mono", "Courier New",
                                  "DejaVu Sans Mono"]

SURFACE = "#e4e4d7"  # the gallery card-art light green
PAPER = "#fbf9f4"
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#6e6d66"
SAGE = "#89927a"
WINE = "#482728"
CRIMSON = "#a43b36"
SUMI = "#1E342E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
})


def chip(ax, x, y, w, h, text, fc, ec, tc="#fbf9f4", fs=9.0, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                facecolor=fc, edgecolor=ec, linewidth=1.3,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, fontweight=weight,
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, rad=0.0, color=INK2, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                                connectionstyle=f"arc3,rad={rad}",
                                linestyle=(0, (4, 3)) if dashed else "solid"),
                zorder=2)


def spaced(s: str) -> str:
    return " ".join(s)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" / "portfolio" / "method_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.42, 3.45))
    ax = fig.add_axes((0.005, 0.055, 0.99, 0.875))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(spaced("SENTIMENT BY AGENT, ALIGNMENT BY DPO"),
                 fontsize=11.5, color=INK2, loc="left", pad=7)

    # ---- sources (left column), fanning into the corpus -------------------
    chip(ax, 0.005, 0.730, 0.130, 0.120, "glassdoor\n100", PAPER, INK2, INK, 8.6)
    chip(ax, 0.005, 0.585, 0.130, 0.120, "reddit\n57", PAPER, INK2, INK, 8.6)
    chip(ax, 0.005, 0.440, 0.130, 0.120, "youtube\n18", PAPER, INK2, INK, 8.6)

    chip(ax, 0.175, 0.460, 0.150, 0.175, "merged corpus\n175 reviews", SUMI, SUMI, fs=9.2)
    arrow(ax, 0.135, 0.790, 0.175, 0.585, rad=-0.18)
    arrow(ax, 0.135, 0.645, 0.175, 0.5475, rad=-0.08)
    arrow(ax, 0.135, 0.500, 0.175, 0.510, rad=-0.10)

    # ---- lane 1: the evaluation harness (top row) -------------------------
    chip(ax, 0.175, 0.755, 0.155, 0.150, "6 prompts\nv1–v6", PAPER, INK2, INK, 8.8)
    chip(ax, 0.372, 0.755, 0.165, 0.150, "silver labels\n525 rows", SAGE, SAGE, fs=9.0)
    chip(ax, 0.580, 0.755, 0.190, 0.150, "leave-one-prompt-out\nscoring", CRIMSON, CRIMSON, fs=8.2)
    chip(ax, 0.815, 0.755, 0.180, 0.150, "the honest table\nsix prompts scored", PAPER, INK2, INK, 8.2)
    arrow(ax, 0.252, 0.635, 0.252, 0.755)          # corpus -> prompts
    arrow(ax, 0.330, 0.830, 0.372, 0.830)
    arrow(ax, 0.537, 0.830, 0.580, 0.830)
    arrow(ax, 0.770, 0.830, 0.815, 0.830)

    # the retired failure site — wine dashed, hangs under the gate
    ax.add_patch(FancyBboxPatch((0.585, 0.615), 0.410, 0.095,
                                boxstyle="round,pad=0.008",
                                facecolor="none", edgecolor=WINE, linewidth=1.2,
                                linestyle=(0, (4, 3)), zorder=3))
    ax.text(0.790, 0.6625, "✗  retired: v5 scored against its own labels",
            ha="center", va="center", fontsize=8.2, color=WINE,
            style="italic", zorder=4)
    arrow(ax, 0.675, 0.755, 0.675, 0.712, color=WINE, dashed=True)

    # ---- lane 2: the alignment extension (middle row), fed by the labels --
    chip(ax, 0.365, 0.280, 0.180, 0.155, "137 preference pairs\nconsensus chosen,\noutlier rejected",
         PAPER, INK2, INK, 7.9)
    chip(ax, 0.575, 0.335, 0.120, 0.100, "110 / 27\nby review id", PAPER, INK2, INK, 7.6)
    chip(ax, 0.725, 0.335, 0.130, 0.100, "LoRA-DPO\nQwen2.5-0.5B", WINE, WINE, fs=8.2)
    chip(ax, 0.885, 0.335, 0.115, 0.100, "base vs tuned\n27 held-out", SAGE, SAGE, fs=7.9)
    arrow(ax, 0.4545, 0.755, 0.455, 0.437)         # labels -> preference pairs
    arrow(ax, 0.545, 0.3575, 0.575, 0.385)
    arrow(ax, 0.695, 0.385, 0.725, 0.385)
    arrow(ax, 0.855, 0.385, 0.885, 0.385)

    # ---- lane 3: the agent (bottom row), fed by the same corpus -----------
    chip(ax, 0.175, 0.085, 0.325, 0.150, "plan-and-execute agent\nretrieve → sentiment → anomaly",
         PAPER, INK2, INK, 8.6)
    chip(ax, 0.550, 0.085, 0.300, 0.150, "memory log + kl-divergence\ndrift detection", PAPER, INK2, INK, 8.6)
    arrow(ax, 0.252, 0.460, 0.252, 0.237)          # corpus -> agent
    arrow(ax, 0.500, 0.160, 0.550, 0.160)

    # ---- footnotes ---------------------------------------------------------
    ax.text(0.005, -0.005, "splits by review id · silver preferences — not RLHF · agreement, never accuracy",
            fontsize=8.3, color=MUTED, ha="left", va="top",
            fontfamily="IBM Plex Mono")

    # ---- save: pad to the art-box aspect (copied from the reference) ------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.2)
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    target = 320.7 / 160
    wing = matplotlib.colors.to_rgba(SURFACE)
    if w < h * target:
        pad = int(round(h * target)) - w
        wings = np.full((h, 1, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 1), img,
                              wings.repeat(pad - pad // 2, 1)], axis=1)
    elif h < w / target:
        pad = int(round(w / target)) - h
        wings = np.full((1, w, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 0), img,
                              wings.repeat(pad - pad // 2, 0)], axis=0)
    plt.imsave(out, img)
    print(f"written: {out} ({img.shape[1]}x{img.shape[0]}, "
          f"aspect {img.shape[1] / img.shape[0]:.3f})")


if __name__ == "__main__":
    main()

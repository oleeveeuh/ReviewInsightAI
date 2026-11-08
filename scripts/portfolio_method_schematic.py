#!/usr/bin/env python3
"""Method schematic for the portfolio card: how ReviewInsight AI works.

Band 1: the silver-label scoring loop (three-source corpus -> six prompt
templates -> silver labels -> leave-one-prompt-out scoring), with the wine
dashed annotation marking the retired self-scoring failure site.
Band 2: the alignment extension (silver labels -> constructed preference
pairs -> review-ID split -> LoRA-DPO -> held-out base-vs-tuned check).

Illustrative diagram: counts, model names, and protocol values are real
(frozen artifacts named in the entry); no metrics are drawn. Fieldnotes
theme, Source Sans 3 / IBM Plex Mono, canvas padded to the gallery art-box
aspect at save time. Structure copied from
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# The site's loaded fonts: Source Sans 3 is the body voice, IBM Plex Mono
# the label/caption voice. Both are installed locally as static TTFs —
# matplotlib silently falls back to DejaVu Sans when a family is missing.
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


def chip(ax, x, y, w, h, text, fc, ec, tc="#fbf9f4", fs=10.5, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                facecolor=fc, edgecolor=ec, linewidth=1.4,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, fontweight=weight)


def arrow(ax, x0, y0, x1, y1, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.8,
                                connectionstyle=f"arc3,rad={rad}"))


def spaced(s: str) -> str:
    # matplotlib has no letter-spacing; mimic the site's mono eyebrows.
    return " ".join(s)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" / "portfolio" / "method_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.42, 3.30))  # padded to the art-box aspect at save
    gs = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.0), hspace=0.66,
                          left=0.03, right=0.97, top=0.905, bottom=0.03)

    # ---- band 1: the silver-label scoring loop ----------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.28, 1.0)
    ax.axis("off")
    ax.set_title(spaced("THE SILVER-LABEL LOOP") + "  —  " + spaced("SCORING"),
                 fontsize=14, color=INK2, loc="left", pad=8)

    chip(ax, 0.000, 0.34, 0.160, 0.56,
         "glassdoor 100\nreddit 57\nyoutube 18", PAPER, INK2, INK, fs=9.5)
    chip(ax, 0.200, 0.40, 0.145, 0.44, "merged corpus\n175 reviews", SUMI, SUMI, fs=10)
    chip(ax, 0.385, 0.40, 0.140, 0.44, "6 prompts\nv1–v6", PAPER, INK2, INK, fs=10)
    chip(ax, 0.565, 0.40, 0.155, 0.44,
         "silver labels\n525 rows", SAGE, SAGE, fs=10)
    chip(ax, 0.745, 0.34, 0.255, 0.56,
         "leave-one-prompt-out\nscoring — every prompt\nexcludes its own rows",
         CRIMSON, CRIMSON, fs=9.5)
    for x0, x1 in ((0.160, 0.200), (0.345, 0.385), (0.525, 0.565), (0.720, 0.745)):
        arrow(ax, x0, 0.62, x1, 0.62)

    # the retired failure site — wine dashed chip, house style honesty mark
    ax.add_patch(FancyBboxPatch((0.500, 0.02), 0.500, 0.20,
                                boxstyle="round,pad=0.008",
                                facecolor="none", edgecolor=WINE, linewidth=1.3,
                                linestyle=(0, (4, 3)), zorder=3))
    ax.text(0.750, 0.12, "✗  retired: v5 scored against v5's own labels",
            ha="center", va="center", fontsize=8.6, color=WINE,
            style="italic", zorder=4)
    ax.text(0.0, -0.24, "splits by review id · every artifact carries model, "
            "timestamp, label type",
            fontsize=9, color=MUTED, ha="left", va="top",
            fontfamily="IBM Plex Mono")

    # ---- band 2: the alignment extension ----------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.28, 1.0)
    ax2.axis("off")
    ax2.set_title(spaced("THE ALIGNMENT EXTENSION") + "  —  " + spaced("DPO ON SILVER PREFERENCES"),
                  fontsize=14, color=INK2, loc="left", pad=8)

    chip(ax2, 0.000, 0.36, 0.130, 0.42, "525 silver\nlabels", SUMI, SUMI, fs=10)
    chip(ax2, 0.165, 0.28, 0.210, 0.58,
         "137 preference pairs\nconsensus = chosen\noutlier = rejected", PAPER, INK2, INK, fs=9.3)
    chip(ax2, 0.410, 0.36, 0.130, 0.42, "110 / 27 split\nby review id", PAPER, INK2, INK, fs=9.5)
    chip(ax2, 0.575, 0.28, 0.190, 0.58,
         "LoRA-DPO\nQwen2.5-0.5B\nInstruct", WINE, WINE, fs=9.8)
    chip(ax2, 0.800, 0.32, 0.200, 0.50,
         "base vs tuned\n27 held-out pairs", SAGE, SAGE, fs=9.3)
    for x0, x1 in ((0.130, 0.165), (0.375, 0.410), (0.540, 0.575), (0.765, 0.800)):
        arrow(ax2, x0, 0.57, x1, 0.57)

    ax2.text(0.0, -0.24, "consensus-as-preference — not RLHF · trl 0.14 + peft "
             "on apple silicon",
             fontsize=9, color=MUTED, ha="left", va="top",
             fontfamily="IBM Plex Mono")

    # The gallery card box (.project-art, 320.7 x 160) shows previews with
    # object-fit: cover, so the tight render is padded to the box's exact
    # aspect — otherwise the centered crop amputates the titles. The wings
    # are the same #e4e4d7 as the card background, making the padding
    # invisible. (Save block copied from the retailpred reference script.)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.2)
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    target = 320.7 / 160
    wing = matplotlib.colors.to_rgba(SURFACE)
    if w < h * target:  # too tall: widen symmetrically
        pad = int(round(h * target)) - w
        wings = np.full((h, 1, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 1), img,
                              wings.repeat(pad - pad // 2, 1)], axis=1)
    elif h < w / target:  # too wide: pad top and bottom
        pad = int(round(w / target)) - h
        wings = np.full((1, w, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 0), img,
                              wings.repeat(pad - pad // 2, 0)], axis=0)
    plt.imsave(out, img)
    print(f"written: {out} ({img.shape[1]}x{img.shape[0]}, "
          f"aspect {img.shape[1] / img.shape[0]:.3f})")


if __name__ == "__main__":
    main()

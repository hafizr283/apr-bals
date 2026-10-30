#!/usr/bin/env python3
r"""
generate_figures3.py — companion to generate_figures.py/generate_figures2.py
(same house style), adding the two figures needed for the external-comparison
extension of the APR-BALS paper (Section IV-H onward):

    Fig 4.6  fig46_ccd_sweep    justapr vs cuMF_CCD training time vs K,
                                 four datasets (small multiples, log scale)
    Fig 4.7  fig47_bals_ladder  BALS-equivalent -> cuMF_ALS -> justapr,
                                 K=32/F=30, all four datasets (log-scale bars)

All numbers are transcribed from results/JUSTAPR_VS_CCD_SGD_20260721_192433.txt,
results/JUSTAPR_VS_CUMF_20260721_045838.txt (F=30 rows), and the freshly
re-measured BALS-equivalent run (results/BALS_EQUIV_VALIDATION_20260721_205101/,
K=32, lambda=0.048, BALS_REORDER=1 baseline vs +BALS_SYMTILE=1).

Run:  python3 generate_figures3.py            (from this directory)
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Style: identical conventions to generate_figures.py ──────────────────────
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#ffffff"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

K_VALUES = [16, 32, 48, 64, 96]
K_LABELS = [str(k) for k in K_VALUES]


def save(fig, basename):
    fig.savefig(os.path.join(OUT, basename + ".pdf"))
    fig.savefig(os.path.join(OUT, basename + ".png"), dpi=600)
    print(f"  wrote {basename}.pdf / .png")
    plt.close(fig)


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.6 — justapr vs cuMF_CCD training time across K, four datasets
# (results/JUSTAPR_VS_CCD_SGD_20260721_192433.txt, "train s" columns)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_6_ccd_sweep():
    data = {
        "ML-10M":  dict(apr=[0.078, 0.154, 0.236, 0.426, 0.758],
                        ccd=[0.348, 0.686, 1.020, 1.362, 2.046]),
        "ML-20M":  dict(apr=[0.173, 0.330, 0.509, 0.911, 1.725],
                        ccd=[0.736, 1.443, 2.182, 2.883, 4.310]),
        "ML-32M":  dict(apr=[0.313, 0.610, 0.967, 1.746, 3.628],
                        ccd=[1.284, 2.554, 3.802, 5.062, 7.603]),
        "Netflix": dict(apr=[0.870, 1.428, 2.071, 3.529, 5.937],
                        ccd=[3.362, 6.599, 9.981, 13.249, 19.796]),
    }

    fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.2), sharex=True)
    x = np.arange(len(K_VALUES))
    for ax, (name, d) in zip(axs.flat, data.items()):
        ax.plot(x, d["ccd"], color=ORANGE, lw=2.0, marker="s", ms=4,
                label="cuMF_CCD ($t{=}15$)")
        ax.plot(x, d["apr"], color=BLUE, lw=2.0, marker="o", ms=4,
                label="justapr")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(x, K_LABELS)
        despine(ax)

    for ax in axs[-1, :]:
        ax.set_xlabel("Latent dimension $K$")
    for ax in axs[:, 0]:
        ax.set_ylabel("Training time (s, log)")

    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig46_ccd_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.7 — precision/system ladder at matched rank (K=32, F=30): faithful
# scalar-FP32 BALS-equivalent -> cuMF_ALS (FP32+CG) -> justapr (tensor core).
# BALS-equivalent = best of {full-matrix, symtile} scalar kernel, both with
# BALS_REORDER=1, lambda=0.048 (results/BALS_EQUIV_VALIDATION_20260721_205101).
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_7_bals_ladder():
    datasets = ["ML-10M", "ML-20M", "ML-32M", "Netflix"]
    bals_equiv = [1.581, 3.798, 6.452, 15.618]   # symtile, best of the two
    cumf_als = [0.838, 1.849, 3.077, 9.967]        # F=30
    justapr = [0.154, 0.330, 0.610, 1.428]         # K=32, ship recipe

    x = np.arange(len(datasets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for off, vals, color, label in (
        (-w, bals_equiv, ORANGE, "BALS-equivalent (scalar FP32)"),
        (0.0, cumf_als, AQUA, "cuMF_ALS (FP32+CG)"),
        (+w, justapr, BLUE, "justapr (tensor core)"),
    ):
        bars = ax.bar(x + off, vals, w, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.0)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1.5, color=INK2)

    ax.set_yscale("log")
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Training time, $K{=}32$/$F{=}30$ (s, log)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=1,
              fontsize=8.5)
    despine(ax)
    save(fig, "fig47_bals_ladder")


if __name__ == "__main__":
    print("Generating additional TPDS figures into", OUT)
    fig_4_6_ccd_sweep()
    fig_4_7_bals_ladder()

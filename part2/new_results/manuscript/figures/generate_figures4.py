#!/usr/bin/env python3
r"""
generate_figures4.py — regenerates the four data-driven Section-IV figures of
the TPDS paper (fig41/fig43/fig44/fig45) from the 2026-07-21 weighted-lambda
re-measurement of the baseline-vs-APR sweep
(results/MAIN3060_WL_20260721_*/ — main_experiment.cu, lambda=0.048 weighted
ALS-WR, CUMF_INIT=1, FP32 solver both legs, GT netflix=1024 / ml=512, GPU1
RTX 3060). fig42 (3060-vs-T4) IS regenerated here, but as a 3-series chart:
the new weighted-lambda 3060 numbers, the old uniform-lambda 3060 numbers
paired with the T4 run (same protocol on both GPUs, so that pair is the
valid apples-to-apples architecture comparison), and the T4 numbers
themselves — see the protocol note in Sec IV-C.

Same house style as generate_figures.py / generate_figures3.py.

Run:  python3 generate_figures4.py            (from this directory)
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

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
# Fig 4.1 — wall-time speedup vs K, three datasets (new Table VI)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_1_speedup_vs_k():
    # netflix/ml20m/ml10m: results/MAIN3060_WL_20260721_225757/
    # ml32m: results/MAIN3060_WL_ML32M_20260722_061135/
    netflix = [4.07, 7.21, 10.14, 13.90, 14.84]
    ml20m = [4.08, 6.76, 8.83, 11.62, 11.16]
    ml10m = [3.93, 6.42, 8.16, 10.87, 10.45]
    ml32m = [4.43, 6.94, 9.07, 11.59, 10.82]

    x = np.arange(len(K_VALUES))
    w = 0.20
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for off, vals, color, label in (
        (-1.5 * w, netflix, BLUE, "Netflix"),
        (-0.5 * w, ml20m, AQUA, "ML-20M"),
        (+0.5 * w, ml10m, YELLOW, "ML-10M"),
        (+1.5 * w, ml32m, ORANGE, "ML-32M"),
    ):
        bars = ax.bar(x + off, vals, w, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.0)
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1.5, color=INK2)

    ax.set_xticks(x, K_LABELS)
    ax.set_xlabel("Latent dimension $K$")
    ax.set_ylabel("Wall-time speedup over FP32 baseline ($\\times$)")
    ax.set_ylim(0, 17.0)
    ax.legend(loc="upper left", ncols=4)
    despine(ax)
    save(fig, "fig41_speedup_vs_k")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.2 — Netflix wall speedup vs K: RTX 3060 vs Tesla T4, both measured
# under the SAME weighted-lambda protocol (3060: results/MAIN3060_WL_20260721_*;
# T4: results/t4_latest_res_22_7_2026/T4_MAIN_WL_20260721_210336, Colab).
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_2_t4_vs_3060():
    rtx3060 = [4.07, 7.21, 10.14, 13.90, 14.84]
    t4 = [2.75, 4.98, 6.27, 9.08, 10.78]

    x = np.arange(len(K_VALUES))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    b1 = ax.bar(x - w / 2, rtx3060, w, color=BLUE,
                label="RTX 3060 (Ampere)", edgecolor=SURFACE, linewidth=1.0)
    b2 = ax.bar(x + w / 2, t4, w, color=AQUA,
                label="Tesla T4 (Turing)", edgecolor=SURFACE, linewidth=1.0)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=1.5, color=INK2)

    ax.set_xticks(x, K_LABELS)
    ax.set_xlabel("Latent dimension $K$")
    ax.set_ylabel("Netflix wall-time speedup ($\\times$)")
    ax.set_ylim(0, 17.0)
    ax.legend(loc="upper left", fontsize=8.5)
    despine(ax)
    save(fig, "fig42_t4_vs_3060")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.3 — convergence trajectories, Netflix K=32 (weighted lambda, new Table)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_3_convergence():
    it = [5, 10, 15, 20]
    base_train = [0.766660, 0.755732, 0.753442, 0.752687]
    apr_train = [0.766632, 0.755697, 0.753405, 0.752649]
    base_test = [0.832330, 0.824836, 0.823564, 0.823308]
    apr_test = [0.832304, 0.824810, 0.823538, 0.823282]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.0, 2.9))
    for ax, base, apr, title in (
        (axl, base_train, apr_train, "Train RMSE"),
        (axr, base_test, apr_test, "Test RMSE"),
    ):
        ax.plot(it, base, color=ORANGE, lw=2.2, solid_capstyle="round",
                label="FP32 baseline")
        ax.plot(it, apr, color=BLUE, lw=1.6, ls=(0, (5, 3)),
                dash_capstyle="round", label="APR-BALS")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_xticks(it)
        despine(ax)
    axl.set_ylabel("RMSE")
    axl.legend(loc="upper right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig43_convergence")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.4 — sustained Gram-stage throughput, Netflix, log scale (new Table)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_4_throughput():
    base_user = [217, 261, 249, 187, 207]
    apr_user = [2798, 4158, 5495, 4425, 5919]
    apr_item = [1010, 3365, 5525, 6807, 11484]

    x = np.arange(len(K_VALUES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for off, vals, color, label, inside in (
        (-w, base_user, ORANGE, "Baseline (user side)", False),
        (0.0, apr_user, BLUE, "APR-BALS user side", True),
        (+w, apr_item, AQUA, "APR-BALS item side", True),
    ):
        bars = ax.bar(x + off, vals, w, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.0)
        txt = [f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}" for v in vals]
        if inside:
            ax.bar_label(bars, labels=txt, fontsize=7, padding=-11,
                         color="#ffffff")
        else:
            ax.bar_label(bars, labels=txt, fontsize=7, padding=1.5, color=INK2)

    ax.set_yscale("log")
    ax.set_ylim(100, 130000)
    ax.axhline(12700, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(x[0] - w - 0.12, 12700 * 1.3, "FP32 peak (12.7 TFLOP/s)",
            fontsize=8, color=INK2)
    ax.axhline(51000, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(x[0] - w - 0.12, 51000 * 1.3, "FP16 tensor-core peak ($\\sim$51 TFLOP/s)",
            fontsize=8, color=INK2)

    ax.set_xticks(x, K_LABELS)
    ax.set_xlabel("Latent dimension $K$")
    ax.set_ylabel("Gram-stage throughput (GFLOP/s, log)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3,
              fontsize=8.5)
    despine(ax)
    save(fig, "fig44_throughput")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4.5 — composition of one APR-BALS iteration per K, Netflix (new Table)
# RMSE ms/call amortized over the 5-iteration check cadence.
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_5_breakdown():
    user = np.array([8.8, 21.7, 35.9, 78.1, 129.4])
    item = np.array([24.2, 26.8, 35.7, 50.8, 66.7])
    solve = np.array([16.4, 29.8, 49.9, 89.4, 248.5])
    rmse = np.array([50.2, 107.3, 174.9, 242.7, 379.1]) / 5.0

    total = user + item + solve + rmse
    parts = [
        (user / total, BLUE, "User Gram (WMMA + scalar)"),
        (item / total, AQUA, "Item Gram (WMMA + scalar)"),
        (solve / total, ORANGE, "Cholesky solve"),
        (rmse / total, YELLOW, "RMSE check (amortized)"),
    ]

    x = np.arange(len(K_VALUES))
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    bottom = np.zeros(len(K_VALUES))
    for frac, color, label in parts:
        ax.bar(x, frac * 100, 0.56, bottom=bottom, color=color, label=label,
               edgecolor=SURFACE, linewidth=1.4)
        bottom += frac * 100

    cum = (user + item) / total * 100
    for xi, (c, s) in enumerate(zip(cum, solve / total * 100)):
        ax.text(xi, c + s / 2, f"{s:.0f}%", ha="center", va="center",
                fontsize=8.5, color="#ffffff")

    ax.set_xticks(x, K_LABELS)
    ax.set_xlabel("Latent dimension $K$")
    ax.set_ylabel("Share of APR-BALS iteration time (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2,
              fontsize=8.5)
    despine(ax)
    save(fig, "fig45_breakdown")


if __name__ == "__main__":
    print("Regenerating Section-IV figures into", OUT)
    fig_4_1_speedup_vs_k()
    fig_4_2_t4_vs_3060()
    fig_4_3_convergence()
    fig_4_4_throughput()
    fig_4_5_breakdown()

"""Shared style for every matplotlib figure in the TPDS paper.

IEEE Author Center: graphics 3.5 in (one column) or 7.16 in (two columns) wide,
type about 9-10 pt at printed size and never below 6 pt, one typeface and one set
of sizes across all figures, lines >= 0.5 pt, fonts embedded. Palatino is the body
face of the compsoc class, so every figure uses it. Series are told apart by colour
AND marker/line style (Okabe-Ito palette) for colour-vision deficiency.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "comparing_baseline", "results"))
COL, TWO_COL = 3.5, 7.16

# Okabe-Ito
BLUE, ORANGE, GREEN, VERMIL, PURPLE, SKY, YELLOW = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442")
GREY, INK, INK2 = "#7f7f7f", "#222222", "#555555"


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic", "mathtext.bf": "Palatino Linotype:bold",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def read(rel):
    with open(os.path.join(RES, rel), encoding="utf-8") as fh:
        return fh.read()


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=600)
    print(f"wrote {name}.pdf / .png")

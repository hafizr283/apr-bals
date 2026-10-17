"""Fig. 1 of the TPDS paper: ratings-per-entity long tail on Netflix, column width.

Same data path as fig_1_1_longtail() in thesis_latex/figures/generate_figures.py,
redrawn for one IEEE column (3.5 in) in the shared figstyle (9 pt labels, 8 pt ticks, Palatino).
Changes from the thesis version: column-width canvas, direct labels only
(the legend duplicated them), threshold labels right-aligned off the curves.

Requires the preprocessed Netflix binary (Linux box):
    python3 gen_fig11_longtail_col.py [path/to/netflix_ratings.bin]
Writes fig11_longtail.pdf / .png next to this script.
"""

import os
import struct
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
NETFLIX_BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(OUT, "..", "..", "figures", "datasets", "netflix_ratings.bin"))

THETA_G = 1024   # giant threshold (Section III-D; measured Netflix optimum)
THETA_D = 4      # dense threshold (Section III-D)

import figstyle as S   # shared paper style: Palatino, 9/8 pt, Okabe-Ito

INK, INK2 = S.INK, S.INK2
BLUE, ORANGE = S.BLUE, S.VERMIL
S.apply()


def read_degrees(path):
    def skip_vec(f, itemsize):
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n * itemsize, 1)

    def read_vec_i32(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return np.frombuffer(f.read(n * 4), dtype=np.int32)

    with open(path, "rb") as f:
        f.read(20)                       # version + dims + nnz counts
        for _ in range(6):               # train u/i/r, test u/i/r
            skip_vec(f, 4)
        user_off = read_vec_i32(f)
        skip_vec(f, 4)
        skip_vec(f, 4)
        item_off = read_vec_i32(f)
    return np.diff(user_off), np.diff(item_off)


def main():
    users, items = read_degrees(NETFLIX_BIN)
    # Printed so the caption and text can be checked against the plotted data.
    for name, d in (("users", users), ("items", items)):
        print(f"{name}: n={len(d)}  max={d.max()}  median={np.median(d):.0f}  "
              f">=theta_g: {(d >= THETA_G).sum()}  <theta_d: {(d < THETA_D).sum()}")

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for d, color, label, xy in ((users, BLUE, "users", (2.0e4, 60)),
                                (items, ORANGE, "items", (60, 3.0e4))):
        c = np.sort(d)[::-1].astype(float)
        c[c < 1] = np.nan
        ax.loglog(np.arange(1, len(c) + 1), c, color=color, lw=1.4)
        ax.text(*xy, label, color=color, fontsize=9)

    for y, label in ((THETA_G, r"$\theta_g = 1024$"), (THETA_D, r"$\theta_d = 4$")):
        ax.axhline(y, color=INK2, lw=0.6, ls=(0, (4, 3)))
        ax.text(1.2, y * 1.25, label, fontsize=8, color=INK2)

    ax.set_xlabel("Entity rank (by training-rating count)")
    ax.set_ylabel("Training ratings")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig11_longtail.pdf"))
    fig.savefig(os.path.join(OUT, "fig11_longtail.png"), dpi=600)
    print("wrote fig11_longtail.pdf / .png")


if __name__ == "__main__":
    main()

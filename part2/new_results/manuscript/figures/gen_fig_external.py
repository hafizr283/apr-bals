"""Fig. (external comparison): time per sweep versus rank, one panel per dataset.

Every value is parsed from the raw archive logs; nothing is typed by hand.
  APR-BALS and cuMF_als : JUSTAPR_VS_CUMF_20260721_045838.txt   (both momentum off,
                          RMSE every 5 sweeps, tol 1e-3; "| rank | code | iters | wall s |" rows)
  cuMF_CCD              : JUSTAPR_VS_CCD_SGD_20260721_192433.txt (compute time, 15 outer iterations)
  BALS-equivalent       : PAPER_MEASUREMENTS_20260803/bals_ladder_all_K.txt
                          (reorder-only and +symmetric tiling, wall / sweeps)
Time per sweep = wall / sweeps actually run. cuMF_als is plotted at its own rank F.

Before drawing, the script re-derives the ratios quoted in Section VI and stops if any
disagrees with the paper (to the printed precision).
Drawn at two-column width (7.16 in). Run: python gen_fig_external.py -> fig_external.pdf
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "comparing_baseline", "results"))
DS = ["ml10m", "ml20m", "ml32m", "netflix"]
NAME = {"ml10m": "ML-10M", "ml20m": "ML-20M", "ml32m": "ML-32M", "netflix": "Netflix"}
K = [16, 32, 48, 64, 96]
F = [20, 30, 50, 60, 100]


def read(rel):
    with open(os.path.join(RES, rel), encoding="utf-8") as fh:
        return fh.read()


# ---- APR-BALS and cuMF_als (per-dataset markdown tables) --------------------------
apr, cumf = {}, {}
txt = read("JUSTAPR_VS_CUMF_20260721_045838.txt")
for ds, block in re.findall(r"table \((\w+)\):(.*?)(?=\n\s*\n|\Z)", txt, re.S):
    for kind, r, it, wall in re.findall(
            r"\|\s*([KF])=(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", block):
        (apr if kind == "K" else cumf)[(ds, int(r))] = 1e3 * float(wall) / int(it)

# ---- cuMF_CCD (compute time) and APR-BALS with momentum, same run ------------------
ccd, apr_mom = {}, {}
txt = read("JUSTAPR_VS_CCD_SGD_20260721_192433.txt")
for ds, body in re.findall(r"── dataset: (\w+).*?\n(.*?)(?=\n─|\n=)", txt, re.S):
    for k, wall, it in re.findall(r"justapr K=(\d+)\s+test [\d.]+\s+wall ([\d.]+)s.*?\((\d+) it\)", body):
        apr_mom[(ds, int(k))] = 1e3 * float(wall) / int(it)
    for k, comp, it in re.findall(r"cuMF_CCD K=(\d+)\s+test [\d.]+\s+compute ([\d.]+)s.*?\((\d+) it", body):
        ccd[(ds, int(k))] = 1e3 * float(comp) / int(it)

# ---- BALS-equivalent ladder ---------------------------------------------------------
reo, sym = {}, {}
txt = read(os.path.join("PAPER_MEASUREMENTS_20260803", "bals_ladder_all_K.txt"))
for k, ds, w_r, w_s, it_r, it_s in re.findall(
        r"^\s*(\d+)\s+(ml10m|ml20m|ml32m|netflix)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+(\d+)/(\d+)", txt, re.M):
    reo[(ds, int(k))] = 1e3 * float(w_r) / int(it_r)
    sym[(ds, int(k))] = 1e3 * float(w_s) / int(it_s)

for d in (apr, cumf, ccd, apr_mom, reo, sym):
    assert len(d) == 20, f"expected 20 points, parsed {len(d)}"


# ---- cross-check against the numbers printed in Section VI ---------------------------
def rng(vals, nd=1):
    return round(min(vals), nd), round(max(vals), nd)


r_cumf = [cumf[(d, f)] / apr[(d, k)] for d in DS for k, f in zip(K, F)]
assert rng(r_cumf) == (3.0, 6.4), rng(r_cumf)
best = {key: min(reo[key], sym[key]) for key in reo}
assert rng([best[(d, 16)] / apr[(d, 16)] for d in DS]) == (4.1, 4.7)
assert rng([best[(d, 96)] / apr[(d, 96)] for d in DS]) == (13.1, 19.9)
symf = [reo[key] / sym[key] for key in reo]
assert rng(symf, 3) == (0.772, 1.209), rng(symf, 3)
assert sum(s < 1 for s in symf) == 13
ccd_avg = {d: sum(ccd[(d, k)] / apr_mom[(d, k)] for k in K) / 5 for d in DS}
assert [round(ccd_avg[d], 2) for d in DS] == [3.83, 3.72, 3.44, 4.08], ccd_avg
cumf_vs_bals32 = [best[(d, 32)] / cumf[(d, 30)] for d in DS]
assert rng(cumf_vs_bals32, 2) == (1.26, 1.67), rng(cumf_vs_bals32, 2)
print("all Section VI ratios reproduced from the raw logs")

# ---- plot ------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic", "mathtext.bf": "Palatino Linotype:bold",
    # IEEE Author Center: type ~9-10 pt at printed size and one typeface across all
    # figures; Palatino is the compsoc body face and matches the other figures
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
    "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Okabe-Ito, one colour and one marker per system; hollow marker = variant of the same system
SERIES = [
    ("APR-BALS",           apr,  K, dict(color="#0072B2", marker="o", ls="-",  lw=1.4, ms=3.6, zorder=5)),
    ("cuMF_als (at rank $F$)",  cumf, F, dict(color="#E69F00", marker="s", ls="--", lw=1.0, ms=3.2)),
    ("cuMF_CCD",         ccd,  K, dict(color="#009E73", marker="^", ls="--", lw=1.0, ms=3.4)),
    ("BALS-equivalent, reorder only",      reo,  K, dict(color="#D55E00", marker="D", ls=":",  lw=1.1, ms=3.0)),
    ("BALS-equivalent, + symmetric tiling", sym, K, dict(color="#D55E00", marker="D", ls=":",  lw=1.1, ms=3.0,
                                                    mfc="white", mew=0.8)),
]

fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
for i, (ax, ds) in enumerate(zip(axes, DS)):
    for label, data, xs, st in SERIES:
        ax.plot(xs, [data[(ds, x)] for x in xs], label=label, **st)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_locator(FixedLocator(K))
    ax.set_xlim(10, 102)
    ax.set_xlabel("Rank $K$")
    ax.set_title(f"({'abcd'[i]}) {NAME[ds]}", pad=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Time per sweep (ms)")

h, l = axes[0].get_legend_handles_labels()
# legend fills column-first; order so rows read: APR, cuMF_als, CCD / BALS, BALS+sym
h, l = [h[i] for i in (0, 3, 1, 4, 2)], [l[i] for i in (0, 3, 1, 4, 2)]
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.83))
fig.legend(h, l, loc="upper center", ncol=3, handlelength=2.6, columnspacing=1.6,
           bbox_to_anchor=(0.5, 1.005))
fig.savefig(os.path.join(HERE, "fig_external.pdf"))
fig.savefig(os.path.join(HERE, "fig_external.png"), dpi=600)
print("wrote fig_external.pdf / .png")

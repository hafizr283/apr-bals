"""Fig. (per-stage profile), Netflix on the RTX 3060, both paths of the two-path harness.

Source: comparing_baseline/results/MAIN3060_WL_20260721_225757/netflix_k<K>.txt,
"=== Profiling: ... ===" blocks (first = FP32 baseline, second = APR-BALS).
Per sweep: Gram = user + item LHS+RHS ms/iter (EXPOSED time: stage span minus the
overlapped solve's busy time, so its GFLOP/s is an upper bound); solve = user + item
Cholesky ms/iter; RMSE check = RMSE total / sweeps. The three add up to the logged
wall time within 1%, which the script asserts.
Also asserts every cell of the per-stage table this figure replaces.
Run: python gen_fig_stages.py -> fig_stages.pdf
"""
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import figstyle as S

K = [16, 32, 48, 64, 96]
PEAK_FP32, PEAK_TC = 12.7e3, 25.5e3          # GFLOP/s, Table 1 (RTX 3060)


def parse(k):
    txt = S.read(f"MAIN3060_WL_20260721_225757/netflix_k{k}.txt")
    blocks = re.findall(r"=== Profiling: (.*?)\((\d+) iters\) ===(.*?)Training Complete in ([\d.]+) seconds",
                        txt, re.S)
    assert len(blocks) == 2
    out = []
    for _, iters, body, wall in blocks:
        it = int(iters)
        g = lambda pat: re.search(pat, body)
        uL = g(r"User LHS\+RHS:\s*[\d.]+ ms total \|\s*([\d.]+) ms/iter \|\s*([\d.]+) GFlops")
        iL = g(r"Item LHS\+RHS:\s*[\d.]+ ms total \|\s*([\d.]+) ms/iter \|\s*([\d.]+) GFlops")
        uC = float(g(r"User Cholesky:\s*[\d.]+ ms total \|\s*([\d.]+) ms/iter").group(1))
        iC = float(g(r"Item Cholesky:\s*[\d.]+ ms total \|\s*([\d.]+) ms/iter").group(1))
        rm = float(g(r"RMSE:\s*([\d.]+) ms total").group(1))
        d = dict(u_ms=float(uL.group(1)), u_gf=float(uL.group(2)),
                 i_ms=float(iL.group(1)), i_gf=float(iL.group(2)),
                 solve=uC + iC, rmse=rm / it, wall=1e3 * float(wall) / it)
        d["gram"] = d["u_ms"] + d["i_ms"]
        parts = d["gram"] + d["solve"] + d["rmse"]
        assert abs(parts / d["wall"] - 1) < 0.01, (k, parts, d["wall"])
        out.append(d)
    return out  # [baseline, apr]


data = {k: parse(k) for k in K}

# ---- the table this figure replaces (sec/4_results.tex, tab:throughput) --------------
TABLE = {  # K: (base user ms, GF), (base item ms, GF), (apr user), (apr item), apr solve
    16: ((112.6, 217), (102.1, 240), (8.8, 2798), (24.2, 1010), 16.4),
    32: ((346.2, 261), (321.9, 280), (21.7, 4158), (26.8, 3365), 29.8),
    48: ((791.5, 249), (711.3, 277), (35.9, 5495), (35.7, 5525), 49.9),
    64: ((1851.1, 187), (1719.9, 201), (78.1, 4425), (50.8, 6807), 89.4),
    96: ((3705.0, 207), (3691.4, 207), (129.4, 5919), (66.7, 11484), 248.5),
}
for k, (bu, bi, au, ai, sol) in TABLE.items():
    b, a = data[k]
    for (ms, gf), d, side in ((bu, b, "u"), (bi, b, "i"), (au, a, "u"), (ai, a, "i")):
        assert round(d[side + "_ms"], 1) == ms and round(d[side + "_gf"]) == gf, (k, side, d)
    assert round(a["solve"], 1) == sol, (k, a["solve"])
gf_b = [d[s + "_gf"] for k in K for d in data[k][:1] for s in "ui"]
gf_a = [d[s + "_gf"] for k in K for d in data[k][1:] for s in "ui"]
assert (round(min(gf_b)), round(max(gf_b))) == (187, 280)
assert (round(min(gf_a) / 1e3, 1), round(max(gf_a) / 1e3, 1)) == (1.0, 11.5)
assert round(100 * max(gf_a) / PEAK_TC) == 45
print("per-stage table reproduced from the raw logs")
for k in K:
    b, a = data[k]
    print(f"K={k:3d} share  base gram {100*b['gram']/b['wall']:5.1f}%  "
          f"apr gram {100*a['gram']/a['wall']:5.1f}% solve {100*a['solve']/a['wall']:5.1f}% "
          f"rmse {100*a['rmse']/a['wall']:5.1f}%")

# ---- plot ---------------------------------------------------------------------------
S.apply()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.COL, 2.05),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) Gram throughput against the two peaks. One colour per line, each labelled at
# its own right end: APR-BALS users, APR-BALS items, FP32 baseline (both sides:
# 2 x Gram FLOPs over the summed exposed time; the two sides differ by < 15%).
def combined(d):   # total Gram FLOP/s of both sides = sum(rate x time) / sum(time)
    return (d["u_gf"] * d["u_ms"] + d["i_gf"] * d["i_ms"]) / (d["u_ms"] + d["i_ms"])


base = [combined(data[k][0]) for k in K]
assert all(187 <= b <= 280 for b in base), base
LINES = [("users", [data[k][1]["u_gf"] / 1e3 for k in K], S.BLUE, "o", "-"),
         ("items", [data[k][1]["i_gf"] / 1e3 for k in K], S.PURPLE, "s", "-"),
         ("FP32 baseline", [b / 1e3 for b in base], S.GREY, "D", "--")]
for lab, ys, col, mk, ls in LINES:
    ax1.plot(K, ys, color=col, ls=ls, lw=1.2, marker=mk, ms=3.2,
             label=f"APR-BALS, {lab}" if lab != "FP32 baseline" else lab)
# tensor-core label above its line, FP32 label below its line (the two are only 2x apart)
for y, lab, f, va in ((PEAK_TC / 1e3, "tensor-core peak", 1.08, "bottom"),
                      (PEAK_FP32 / 1e3, "FP32 peak", 0.90, "top")):
    ax1.axhline(y, color=S.INK2, lw=0.6, ls=":")
    ax1.text(11, y * f, lab, ha="left", va=va, fontsize=7.5, color=S.INK2,
             bbox=dict(fc="white", ec="none", pad=0.4))
ax1.set_yscale("log")
ax1.set_ylim(0.1, 60)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.xaxis.set_major_locator(FixedLocator(K))
ax1.set_xlim(10, 100)
ax1.set_xlabel("Rank $K$")
ax1.set_ylabel("TFLOP/s")
ax1.set_title("(a) Gram throughput", pad=3)
S.despine(ax1)

# (b) share of sweep time; under each K, bar B = FP32 baseline, bar A = APR-BALS
stages = (("gram", "Gram", S.GREEN, None), ("solve", "Solve", S.ORANGE, "////"),
          ("rmse", "RMSE check", S.GREY, None))
w, gap = 0.38, 0.05
pos = []
for j, k in enumerate(K):
    for path, dx in ((0, -(w + gap) / 2), (1, (w + gap) / 2)):
        d = data[k][path]
        bottom = 0.0
        for key, lab, col, hatch in stages:
            h = 100 * d[key] / d["wall"]
            ax2.bar(j + dx, h, w, bottom=bottom, color=col, hatch=hatch, edgecolor="white",
                    lw=0.4, label=lab if (j == 0 and path == 1) else None)
            bottom += h
        pos.append(j + dx)
ax2.set_xticks(pos, minor=True)
ax2.set_xticklabels(["B", "A"] * len(K), minor=True, fontsize=7, color=S.INK2)
ax2.set_xticks(range(len(K)))
ax2.set_xticklabels(K)
ax2.tick_params(axis="x", which="major", length=0, pad=11)
ax2.tick_params(axis="x", which="minor", length=0, pad=1.5)
ax2.set_xlabel("Rank $K$")
ax2.set_ylim(0, 100)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_ylabel("Share of sweep (%)")
ax2.grid(axis="x", visible=False)
ax2.set_title("(b) Time per stage", pad=3)
S.despine(ax2)

# one legend for both panels, filled column-first: row 1 = lines of (a), row 2 = stages of (b)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h = [x for pair in zip(h1, h2) for x in pair]
l = [x for pair in zip(l1, l2) for x in pair]
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.835))
fig.legend(h, l, loc="upper center", ncol=3, handlelength=1.9, columnspacing=1.0,
           handletextpad=0.4, labelspacing=0.3, fontsize=7.5, bbox_to_anchor=(0.5, 1.01))
S.save(fig, "fig_stages")

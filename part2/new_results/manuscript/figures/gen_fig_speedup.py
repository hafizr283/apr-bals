"""Fig. 4 of the TPDS paper: wall-time speedup vs K on the RTX 3060 and Tesla T4.

Values are the 'Wall-time speedup' lines of the per-run logs, checked 2026-09-29:
  RTX 3060: comparing_baseline/results/MAIN3060_WL_20260721_225757/<ds>_k<K>.txt
            and MAIN3060_WL_ML32M_20260722_061135/ml32m_k<K>.txt
            (5-run means in PAPER_MEASUREMENTS_20260803/variance_5x_summary.txt
             agree to within 0.5%)
  Tesla T4: comparing_baseline/results/t4_latest_res_22_7_2026/T4_MAIN_WL_20260721_210336/
            and T4_MAIN_WL_ML32M_20261002_081249/ml32m_k<K>.txt (ML-32M, Colab, 2026-10-02)
Drawn at one IEEE column (3.5 in) so that text prints at 8 pt.
Run: python gen_fig_speedup.py   -> fig_speedup.pdf
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
K = [16, 32, 48, 64, 96]

RTX3060 = {
    "Netflix": [4.07, 7.21, 10.14, 13.90, 14.84],
    "ML-32M":  [4.43, 6.94, 9.07, 11.59, 10.82],
    "ML-20M":  [4.08, 6.76, 8.83, 11.62, 11.16],
    "ML-10M":  [3.93, 6.42, 8.16, 10.87, 10.45],
}
T4 = {
    "Netflix": [2.75, 4.98, 6.27, 9.08, 10.78],
    "ML-32M":  [2.88, 4.86, 5.77, 7.34, 7.76],
    "ML-20M":  [2.85, 4.72, 5.50, 7.23, 8.08],
    "ML-10M":  [2.95, 4.68, 5.26, 6.88, 7.36],
}
STYLE = {  # colour-blind-safe (Okabe-Ito); marker also encodes identity
    "Netflix": ("#0072B2", "o"),
    "ML-32M":  ("#D55E00", "D"),
    "ML-20M":  ("#009E73", "s"),
    "ML-10M":  ("#CC79A7", "^"),
}

import figstyle as S   # shared paper style: Palatino, 9/8 pt
S.apply()

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.08), sharey=True)
for ax, data, title in ((axes[0], RTX3060, "(a) RTX 3060 (Ampere)"),
                        (axes[1], T4, "(b) Tesla T4 (Turing)")):
    for name, ys in data.items():
        c, m = STYLE[name]
        ax.plot(K, ys, color=c, marker=m, ms=3.2, lw=1.1, label=name)
    ax.set_title(title, pad=3)
    ax.set_xticks(K)
    ax.set_xlabel("Rank $K$")
    ax.set_ylim(0, 16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Wall-time speedup")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.88))
fig.legend(h, l, loc="upper center", ncol=4, handlelength=1.4,
           columnspacing=0.9, bbox_to_anchor=(0.53, 1.0))
fig.savefig(os.path.join(OUT, "fig_speedup.pdf"))
fig.savefig(os.path.join(OUT, "fig_speedup.png"), dpi=600)
print("wrote fig_speedup.pdf / .png")

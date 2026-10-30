"""Fig. (cumulative ablation), Netflix K=96, RTX 3060. Every time is parsed from its log.
  1 FP32 baseline                 MAIN3060_WL_20260721_225757/netflix_k96.txt (first "Training Complete")
  2 + tensor-core dispatch        same file (second "Training Complete")
  3 FP32 solver, re-measured      CHOLMP_VALIDATION_20260720/SUMMARY.txt ("APR wall 10.586 s", 07-12 record)
  4 + FP16-tile solver            same file (fp16-nr row, 9.438 s)
  (rows 1-4 are the two-path harness, main_experiment.cu; rows 5-6 the shipped binary, justapr.cu)
  -- shipped binary, row-4 config logs_justapr_vs_cumf_20260720_203958/netflix_apr_k96.txt (9.349 s,
                                  FP16 solver tiles, before the 07-21 overhead removal; text only)
  5 + fused lambda, fast RMSE,
      async giant copy            JUSTAPR_VS_CUMF_20260721_045838.txt (netflix K=96, momentum off)
  6 + momentum (beta = 0.3)       JUSTAPR_VS_CCD_SGD_20260721_192433.txt (netflix K=96, 15 sweeps)
Iteration-matched ratio = baseline time scaled to 15 of its 20 sweeps, over row 6.
Asserts the ratios printed in Section V before drawing. Run: python gen_fig_ablation.py
"""
import re

import matplotlib.pyplot as plt

import figstyle as S

txt = S.read("MAIN3060_WL_20260721_225757/netflix_k96.txt")
t_base, t_tc = map(float, re.findall(r"Training Complete in ([\d.]+) seconds", txt))
base_iters = int(re.search(r"=== Profiling: FP32 Baseline.*?\((\d+) iters\)", txt).group(1))
txt = S.read("CHOLMP_VALIDATION_20260720/SUMMARY.txt")
t_fp32sol = float(re.search(r"APR wall\s+([\d.]+) s, test", txt).group(1))
t_fp16sol = float(re.search(r"fp16-nr\s+[\d.]+ -[\d.]+%\s+([\d.]+) s", txt).group(1))
txt = S.read("JUSTAPR_VS_CUMF_20260721_045838.txt")
blk = re.search(r"table \(netflix\):(.*?)(?=\n\s*\n|\Z)", txt, re.S).group(1)
t_over = float(re.search(r"\|\s*K=96\s*\|[^|]*\|\s*\d+\s*\|\s*([\d.]+)", blk).group(1))
txt = S.read("logs_justapr_vs_cumf_20260720_203958/netflix_apr_k96.txt")
assert "FP16 tiles, NO refinement" in txt
t_ship4 = float(re.search(r"Training Complete in ([\d.]+) seconds", txt).group(1))
txt = S.read("JUSTAPR_VS_CCD_SGD_20260721_192433.txt")
body = re.search(r"── dataset: netflix.*?\n(.*?)(?=\n─|\n=)", txt, re.S).group(1)
m = re.search(r"justapr K=96\s+test [\d.]+\s+wall ([\d.]+)s.*?\((\d+) it\)", body)
t_mom, mom_iters = float(m.group(1)), int(m.group(2))

ROWS = [("FP32 baseline", t_base),                  # labels kept short; the text and
        ("+ tensor cores", t_tc),                      # caption name each step in full
        ("FP32 solver (ref.)", t_fp32sol),
        ("+ FP16-tile solver", t_fp16sol),
        ("+ overhead removal", t_over),
        ("+ momentum", t_mom)]
sp = [t_base / t for _, t in ROWS]
iter_matched = t_base * mom_iters / base_iters / t_mom
assert [round(x, 1) for x in sp] == [1.0, 14.8, 14.6, 16.4, 19.9, 26.0], sp
assert round(iter_matched, 1) == 19.5 and (base_iters, mom_iters) == (20, 15)
assert [round(t, 2) for _, t in ROWS] == [154.42, 10.41, 10.59, 9.44, 7.75, 5.94]
# Section V-C: overhead removal measured within the shipped binary (9.35 -> 7.75 s, 1.21x)
assert round(t_ship4, 2) == 9.35 and round(t_ship4 / t_over, 2) == 1.21
# Section V-E: 14.8 (dispatch) x 1.34 (solver + overheads, bar 2 -> bar 5) x 1.31 (momentum) = 26.0
f_d, f_so, f_m = t_base / t_tc, t_tc / t_over, t_over / t_mom
assert (round(f_d, 1), round(f_so, 2), round(f_m, 2)) == (14.8, 1.34, 1.31)
assert round(f_d * f_so * f_m, 1) == 26.0
# bar 3 vs bar 2: two harness builds, 1.7% apart
assert round(100 * (t_fp32sol / t_tc - 1), 1) == 1.7
print("ablation ratios reproduced:", [round(x, 2) for x in sp], "iter-matched", round(iter_matched, 2))

S.apply()
fig, ax = plt.subplots(figsize=(S.COL, 1.62))
ys = list(range(len(ROWS)))[::-1]
cols = [S.GREY, S.BLUE, S.SKY, S.SKY, S.SKY, S.BLUE]
for y, (lab, t), x, c in zip(ys, ROWS, sp, cols):
    ax.barh(y, x, 0.64, color=c, edgecolor="none")
    ax.text(x + 0.35, y, f"{x:.1f}×", va="center", fontsize=8, color=S.INK)
    if x > 5:   # wall time inside the bar
        ax.text(x - 0.35, y, f"{t:.2f} s", va="center", ha="right", fontsize=7.5,
                color="white" if c == S.BLUE else S.INK)
    else:
        ax.text(x + 3.6, y, f"({t:.2f} s)", va="center", fontsize=7.5, color=S.INK)
ax.set_yticks(ys)
ax.set_yticklabels([r for r, _ in ROWS])
ax.tick_params(axis="y", length=0)
# rows 1-4: two-path harness (main_experiment.cu); rows 5-6: shipped binary (justapr.cu)
ax.axhline(ys[3] - 0.5, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
ax.text(32.3, ys[3] - 0.5 + 0.08, "harness", ha="right", va="bottom", fontsize=7, color=S.INK2,
        style="italic")
ax.text(32.3, ys[3] - 0.5 - 0.1, "shipped", ha="right", va="top", fontsize=7, color=S.INK2,
        style="italic")
# iteration-matched advantage: hollow diamond on the momentum bar
ax.plot([iter_matched], [ys[-1]], marker="D", ms=4.2, mfc="white", mec=S.INK, mew=0.8, zorder=5)
ax.text(iter_matched, ys[-1] - 0.52, f"{iter_matched:.1f}×", ha="center", va="top",
        fontsize=7.5, color=S.INK)
ax.set_xlim(0, 32.5)
ax.set_ylim(-1.05, len(ROWS) - 0.5)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("Speedup over the FP32 baseline")
ax.grid(axis="y", visible=False)
S.despine(ax)
fig.tight_layout()
S.save(fig, "fig_ablation")

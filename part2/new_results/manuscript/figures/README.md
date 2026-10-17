# TPDS paper figures — source of truth

This directory is the ONLY one the paper (`../main.tex`) reads
(`\includegraphics{figures/...}`). The parent `thesis_latex/figures/`
directory contains SAME-NAMED files from the older thesis; they are NOT
interchangeable — never copy from there into here. Regenerate instead:

| figure | script | data source |
|---|---|---|
| fig11_longtail, fig31–fig36 | `../../figures/generate_figures2.py` (writes to `../../figures/`; copy the changed PDF here) | architecture diagrams; fig31 updated 2026-07-22 (weighted λ·n_u fused into solver load + momentum step) |
| fig41, fig42, fig43, fig44, fig45 | `generate_figures4.py` (this dir) | fig41: `results/MAIN3060_WL_20260721_225757/` + `results/MAIN3060_WL_ML32M_20260722_061135/`; fig42: same 3060 data + `results/t4_latest_res_22_7_2026/T4_MAIN_WL_20260721_210336/`; fig43/44/45: `MAIN3060_WL_20260721_225757` |
| fig46_ccd_sweep, fig47_bals_ladder | `generate_figures3.py` (this dir) | `results/JUSTAPR_VS_CCD_SGD_20260721_192433.txt`, `results/BALS_EQUIV_VALIDATION_20260721_205101/` |

(`results/` = `latent_factorr_experimen/comparing_baseline/results/`.)

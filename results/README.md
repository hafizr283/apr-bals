# Historical experiments

[Repository entry](../README.md) · [October evidence](../part2/new_results/README.md) · [Evidence guide](../docs/evidence.md#historical-evidence)

These files preserve earlier experiments and their original protocols. Their age does
not invalidate them. Cite each result with its data split, recipe, hardware and timer.
All raw files and existing paths are retained.

## Index and status

| Archive | Role | Status for paper use |
|---|---|---|
| [MAIN3060_WL_20260721_225757](MAIN3060_WL_20260721_225757/), [ML-32M companion](MAIN3060_WL_ML32M_20260722_061135/) | RTX 3060 shared-binary FP32/APR comparison | Retained historical control; original 80/20 protocol. |
| [PAPER_MEASUREMENTS_20260803](PAPER_MEASUREMENTS_20260803/) | Variance, precision controls, profiling, thresholds, BALS ladder | Retained within each study's scope; repetitions do not characterize October runs. |
| [t4_latest_res_22_7_2026](t4_latest_res_22_7_2026/) | Historical T4 comparisons, including later additions | Retained GPU evidence; folder name is not every run's date. |
| [PAPER_CHECKS_20260929](PAPER_CHECKS_20260929/) | Dataset, cold-entity and tier checks | Retained diagnostics with their own masks and inputs. |
| [BALS_EQUIV_VALIDATION_20260721_205101](BALS_EQUIV_VALIDATION_20260721_205101/) | Local BALS mapping comparison | Local reimplementation evidence, not upstream BALS results. |
| [CHOLMP_VALIDATION_20260720](CHOLMP_VALIDATION_20260720/) | Solver validation under earlier configurations | Retained diagnostic evidence, not universal solver safety. |
| July `JUSTAPR_VS_*`, `RMSE_COMPARISON_*` and `logs_*` files | Earlier external and recipe comparisons | Superseded for October headlines by [external time to target](../part2/new_results/external_time_to_target.csv); preserved for original protocols. |
| [August lambda sweep](PAPER_MEASUREMENTS_20260803/lambda_sweep_k96.txt) | Earlier test-selected regularization search | Historical only; not validation-only selection. |
| [August target-time summary](PAPER_MEASUREMENTS_20260803/time_to_target_rmse.txt) | Earlier coarse crossings | Historical checking interval; superseded for October external headlines. |

## Interpretation

Original test RMSE and October warm-validation RMSE are different measurements.
Per-sweep time, wall time to convergence and first validation-target time also differ.
Shared-binary, standalone APR, local BALS and external ports answer distinct questions.

The [evidence guide](../docs/evidence.md) contains the supported claim matrix and CPU
check scope. Older summaries remain historical narrative and may include stale ranges,
protocol assumptions or unfinished plans. Retention does not endorse every statement.

Historical variance is not assigned to October single runs. No results are removed
to improve a reported range or hide negative outcomes.

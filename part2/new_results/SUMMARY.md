# October review — measurement status

Authoritative inputs: `../whattodo/Review Accelerating ALS Matrix Factorization on GPUs via Mixed-Precision Tensor Cores.pdf` and `../whattodo/GPU_ALS_Matrix_Factorization (2).zip`. Manuscript extracted unchanged into `manuscript/`. The baseline zip was not used.

Primary performance measurements use GPU 1, RTX 3060, sm_86. After the user requested both GPUs, accumulation-reset diagnostics and validation accuracy pairs moved to GPU 0 (also RTX 3060, desktop active); their timings are not performance headline evidence. External runs on GPU 1 overlap those GPU 0 accuracy jobs, sharing the host. Original source and historical results are preserved. Experimental source is in `src/`, FP32-fallback source in `fallback_src/`, BF16 source in `bf16_src/`, and external instrumentation in `external_src/`. Timings exclude preprocessing and loading.

Existing PDFs are intermediate and were not refreshed after the user requested no more plotting. Use the current CSVs and generated LaTeX tables; see `figures/STATUS.md`.

## Required results and status

- Four-way FP32 atomic / FP32 same-dataflow / TF32 / FP16: 80/80 runs collected; `four_way.csv`, raw logs and commands in `four_way/`. Fixed 20 sweeps, FP32 solver, no momentum or giant tier, exact FP32 RMSE every sweep.
- Validation-only lambda selection: 4/4 datasets selected; eight candidates, K=96, fixed 30 sweeps. `validation_selection.csv`. New split is approximately 70/10/20, carving 1/8 of old train into validation with seed 20261014. Original test preserved. Historical exposure to this test cannot be undone by carving validation.
- Per-sweep momentum: 40/40 curves complete; `momentum_crossings.csv`. Fixed 30-sweep trajectories, beta 0 vs .3, exact RMSE. Both adjacent-sweep and five-sweep-window deltas are reported; these are different stopping rules.
- FP64 Gram checks: 3200 per-entity/path records; `gram_errors.csv`. Top 20 users and items per dataset/rank; actual CUDA kernels on the same trained FP32 factor snapshot, upper-triangle errors.
- BF16 Gram operands and BF16 solver with refinement: 8/8 runs; beta .3 and .5 at K=96 on four datasets. `bf16.csv`. Any zeroed solve makes a configuration unsafe.
- Common held-out target curves: 84/84 curves collected in `external/`; all methods use identical warm-validation rows. APR, cuMF-ALS, CCD, and BALS across five ranks; SGD rank128 compared at APR K96. Cold validation rows excluded consistently. SGD uses one continuous model with per-epoch GPU validation and a final saved-model CPU scoring check; epoch barriers differ from the original persistent-kernel schedule.
- Cumulative ablation: six runs completed in `cumulative/`; one source snapshot. See `CUMULATIVE_PROTOCOL.md` for the switches and combined transitions. Gains are measured, never forced to be monotone. Existing code has no switch for lambda fusion / asynchronous staging, so those two effects are not separately isolated.
- Nsight administrator counters: complete. Dense `compute_LHS_RHS_wmma<6>` records 44.82% long-scoreboard stalls, 1.05% LG throttle, 20.43% SM busy, and 20.4% tensor-pipe utilization. Giant `compute_LHS_RHS_wmma_giant<6,4>` records 19.56% long-scoreboard stalls, 1.23% LG throttle, 25.54% SM busy, and 24.5% tensor-pipe utilization. Valid files are `ncu/dense_stalls.csv`, `ncu/giant_stalls.csv`, `ncu/dense_explicit_stalls.csv`, and `ncu/giant_explicit_stalls.csv`. This supports a latency/scoreboard explanation for the dense kernel and shows that giant reduces that stall share; low LG throttle does not support an LG-throttle bottleneck, and neither kernel saturates the tensor pipe.
- Solver safety: the new `fallback_src/` snapshot retries each failed FP16 entity solve with the validated FP32 tiled solver and zeros only a checked FP32 failure. The unchanged `src/` snapshot remains the zero-on-failure control. All 39 new GPU-1 runs passed the counter and finiteness audit; scope remains bounded to the tested configurations.
- Optional padded ranks and Yahoo/Amazon: not implemented/not collected. Existing 60,000-entity batches already split datasets whose user counts exceed that size; a larger dataset would add scale coverage.

## Newly gathered numerical findings

### FP32 solver fallback

At Netflix K=96 on the new 70/10/20 split, the validation-selected stable configuration (lambda .04, beta .3, no refinement) invoked no fallbacks in three runs of 14,938,770 entity-solves each. Its median validation/test RMSE was .820529/.819465. At beta .5 and the documented failure-protocol lambda .048, both no-refinement and one-refinement fallback builds had exactly two FP16 failures per run (0.000013%); FP32 recovered all two, no retry failed, and median validation/test RMSE was .822030/.821171 and .822022/.821168. The all-FP32 reference was .822022/.821168. Matching zero-on-failure builds accumulated six and nine failures and ended at test RMSE .821787 and .822242.

On the original split where the earlier failure was observed, retry changes the subsequent trajectory: no-refinement fallback had 33 failures and recoveries per run, while one-refinement fallback had 26-27; every FP32 retry succeeded. Median test RMSE was .817964/.817956, versus .817956 for all-FP32. Zero-on-failure controls cascaded to 3,406-3,463 and 3,161-3,175 failures and had median test RMSE 1.203685 and 1.263372; no-refinement zeroing also varied widely across repetitions (1.097043-2.267940).

Median GPU-1 wall-time differences between fallback and matching zero-on-failure builds were -0.119% to -0.001% in the failure cases and -0.087% in the stable control. These millisecond-scale negative estimates are reported as no detectable overhead, not as speedups. All final factors and evaluated predictions were finite. See `FP32_FALLBACK_PROTOCOL.md`, `FP32_FALLBACK_TABLE.md`, and `fp32_fallback_audit.json`.

| Dataset | K | atomic / FP16 wall | same-dataflow / FP16 wall | same-dataflow / FP16 compute phase |
|---|---:|---:|---:|---:|
| ml10m | 16 | 3.412 | 1.749 | 1.988 |
| ml10m | 32 | 5.715 | 1.762 | 1.998 |
| ml10m | 48 | 7.858 | 1.803 | 2.039 |
| ml10m | 64 | 9.954 | 1.425 | 1.512 |
| ml10m | 96 | 9.856 | 2.520 | 2.760 |
| ml20m | 16 | 3.592 | 1.625 | 1.798 |
| ml20m | 32 | 6.211 | 1.724 | 1.926 |
| ml20m | 48 | 8.602 | 1.737 | 1.936 |
| ml20m | 64 | 10.782 | 1.401 | 1.480 |
| ml20m | 96 | 10.551 | 2.369 | 2.575 |
| ml32m | 16 | 3.967 | 1.529 | 1.661 |
| ml32m | 32 | 6.449 | 1.618 | 1.772 |
| ml32m | 48 | 8.805 | 1.617 | 1.766 |
| ml32m | 64 | 10.807 | 1.339 | 1.400 |
| ml32m | 96 | 10.315 | 2.082 | 2.221 |
| netflix | 16 | 2.933 | 1.386 | 1.810 |
| netflix | 32 | 4.508 | 1.470 | 2.193 |
| netflix | 48 | 6.098 | 1.473 | 2.248 |
| netflix | 64 | 8.621 | 1.309 | 1.670 |
| netflix | 96 | 9.921 | 2.283 | 3.418 |

The compute-phase column sums the legacy User/Item LHS+RHS event intervals. Because these join the overlapping solve streams, they include solver work and MUST NOT be labeled isolated Gram-kernel time. Runs are single measurements, not new variance estimates. The same-dataflow control uses FP32 staging and direct FMA, one warp/entity, 16-row chunks, eight accumulator entries per lane per upper tile, and no Gram atomics. At K<=48, FP16 also uses its existing prefetch path; FP32/TF32 staging is wider and lacks that prefetch. Treat this as a structural dataflow control, not an instruction-identical substitution.

Validation safety failures (retained in `validation_sweep.csv`): ml32m_ship_k96_lambda0.01: 1 zeroed solves, netflix_ship_k96_lambda0.01: 2 zeroed solves.

- Matched BF16/FP16 safety controls: 12/12 additional FP16 runs in `bf16_controls.csv`; FP16 with one refinement at beta .3/.5 and shipped no-refinement FP16 at beta .5. GPU0 accuracy/safety diagnostics, not cross-GPU timing comparisons.

- Selected-recipe test evaluations: 20/20; `heldout_test.csv`.
- New-split controlled FP32/FP16 accuracy pair: 40/40; `validation_accuracy_pair.csv`. Fixed 30 sweeps, FP32 solver, no momentum/giant tier, validation-selected lambda, test checked only at the end.
- Rounded-input FP32 controls: 20/20; `rounding_control.csv`.
- Chunk-reset tensor accumulation controls: 20/20; `accumulation_control.csv`. These GPU0 diagnostic timings are not headline performance evidence.

Momentum at the adjacent-sweep training-delta threshold improves measured crossing time by 1.10–1.36x across the 20 settings. The five-sweep-window results are a different stopping rule and remain separately reported.

BF16 result: beta .5 is stable for FP16 on MovieLens, but Netflix FP16 fails with/without refinement (3,142/3,492 zeroed entity-solves; test RMSE 1.440298/1.339762). BF16 with one refinement has zero guards and RMSE 0.817964. This supports the tested BF16 configuration for this failure case, not universal safety or uniformly better accuracy.

Resetting the tensor accumulator every 16-row chunk and combining chunks with FP32 CUDA-core adds returns test RMSE to within 0.000002 of the FP32 control at all 20 dataset/rank points. Rounded operands alone do not reproduce the original FP16 shift. Together with the FP64 Gram errors, this isolates persistent tensor accumulation as the source of the small trajectory/RMSE shift in these experiments; it does not mean FP16 constructs a more accurate Gram matrix. See `NUMERICAL_AUDIT.md` and `accumulation_errors.csv`.

Validation selections: ml10m: lambda=0.06, validation RMSE=0.787916, ml20m: lambda=0.048, validation RMSE=0.776661, ml32m: lambda=0.048, validation RMSE=0.766086, netflix: lambda=0.04, validation RMSE=0.820528.

- atomic_fp32: relative upper-triangle Frobenius error median 1.23e-07, maximum 3.41e-07.
- dense_fp16: relative upper-triangle Frobenius error median 0.000186, maximum 0.00205.
- sameflow_fp32: relative upper-triangle Frobenius error median 2.01e-06, maximum 9.26e-06.
- giant_fp16: relative upper-triangle Frobenius error median 4.55e-05, maximum 0.000507.

## External target times at the K96 comparison

Seconds to the preset warm-validation RMSE target. APR/CCD/BALS rank96, cuMF-ALS rank100, SGD rank128. Single runs; shared-host execution and evaluation overhead are documented in `EXTERNAL_PROTOCOL.md`.

| Dataset | Target | APR | ALS | CCD | BALS | SGD |
|---|---:|---:|---:|---:|---:|---:|
| ml10m | 0.8 | 0.1646 | 0.7918 | 0.5798 | 2.5914 | 0.1964 |
| ml20m | 0.79 | 0.3736 | 1.2867 | 1.2219 | 6.1281 | 0.3327 |
| ml32m | 0.78 | 0.7817 | 2.1517 | 2.7185 | 12.0291 | 0.5162 |
| netflix | 0.84 | 1.9308 | 6.1046 | 8.1537 | 30.2483 | 1.9177 |

APR is 2.75–4.81x faster than cuMF-ALS in this rank pairing at these targets. SGD reaches the target sooner on ML20M and ML32M; Netflix differs by less than 1% and should be treated as a single-run near-tie, not evidence of a reliable ordering. APR reaches the target sooner on ML10M. Final validation accuracy is also reported in `external_time_to_target.csv`; reaching an easier target first does not establish better final accuracy.


## Existing evidence (historical protocol, not new runs)

- `../results/PAPER_MEASUREMENTS_20260803/`: 100 variance runs; three-way precision ablation; Nsight utilization/occupancy/registers; full-rank BALS ladder; thresholds; old lambda sweep; coarse cuMF time-to-target. The new manuscript already incorporates much of this.
- `empty_entity_census.txt`: freshly reproduced cold-item counts ML-10M 30, ML-20M 909, ML-32M 4,182, Netflix 0; affected original test ratings 41, 1,006, 4,723, 0.
- Old test-selected lambda sweep must not be presented as validation selection. Old TTT has five-sweep resolution. Old precision ablation bundles dataflow and tensor instructions.

## Next / paper integration

Use the completed new controls and validation results to replace the relevant tables; do not mix old 80/20 and new 70/10/20 absolute RMSE. Rebuild the main two-path harness with the validated FP32 control before replacing its headline. Integrate the external comparison first, then the mechanism analysis. The current manuscript is preserved unchanged; requested editorial restructuring, template changes, author confirmation, and repository/Zenodo publication are separate remaining work. No new acceptance or speedup claim should be made from incomplete cells.

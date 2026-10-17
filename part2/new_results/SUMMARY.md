# October review — measurement status

Authoritative inputs: `../whattodo/Review Accelerating ALS Matrix Factorization on GPUs via Mixed-Precision Tensor Cores.pdf` and `../whattodo/GPU_ALS_Matrix_Factorization (2).zip`. Manuscript extracted unchanged into `manuscript/`. The baseline zip was not used.

Primary performance measurements use GPU 1, RTX 3060, sm_86. After the user requested both GPUs, accumulation-reset diagnostics and validation accuracy pairs moved to GPU 0 (also RTX 3060, desktop active); their timings are not performance headline evidence. External runs on GPU 1 overlap those GPU 0 accuracy jobs, sharing the host. Original source and historical results are preserved. Experimental source is in `src/`, BF16 source in `bf16_src/`, external instrumentation in `external_src/`. Timings exclude preprocessing and loading.

Existing PDFs are intermediate and were not refreshed after the user requested no more plotting. Use the current CSVs and generated LaTeX tables; see `figures/STATUS.md`.

## Required results and status

- Four-way FP32 atomic / FP32 same-dataflow / TF32 / FP16: 52/80 runs collected; `four_way.csv`, raw logs and commands in `four_way/`. Fixed 20 sweeps, FP32 solver, no momentum or giant tier, exact FP32 RMSE every sweep.
- Validation-only lambda selection: 2/4 datasets selected; eight candidates, K=96, fixed 30 sweeps. `validation_selection.csv`. New split is approximately 70/10/20, carving 1/8 of old train into validation with seed 20261014. Original test preserved. Historical exposure to this test cannot be undone by carving validation.
- Per-sweep momentum: 12/40 curves complete; `momentum_crossings.csv`. Fixed 30-sweep trajectories, beta 0 vs .3, exact RMSE. Both adjacent-sweep and five-sweep-window deltas are reported; these are different stopping rules.
- FP64 Gram checks: 2080 per-entity/path records; `gram_errors.csv`. Top 20 users and items per dataset/rank; actual CUDA kernels on the same trained FP32 factor snapshot, upper-triangle errors.
- BF16 Gram operands and BF16 solver with refinement: 3/8 runs; beta .3 and .5 at K=96 on four datasets. `bf16.csv`. Any zeroed solve makes a configuration unsafe.
- Common held-out target curves: 84/84 curves collected in `external/`; all methods use identical warm-validation rows. APR, cuMF-ALS, CCD, and BALS across five ranks; SGD rank128 compared at APR K96. Cold validation rows excluded consistently. SGD uses one continuous model with per-epoch GPU validation and a final saved-model CPU scoring check; epoch barriers differ from the original persistent-kernel schedule.
- Cumulative ablation: six runs completed in `cumulative/`; one source snapshot. See `CUMULATIVE_PROTOCOL.md` for the switches and combined transitions. Gains are measured, never forced to be monotone. Existing code has no switch for lambda fusion / asynchronous staging, so those two effects are not separately isolated.
- Nsight long-scoreboard / LG-throttle / giant execution-pipe counters: permission-blocked (`ncu/permission_check.txt`, ERR_NVGPUCTRPERM); administrator script pending; old utilization counters do not establish a bandwidth bottleneck.
- Solver safety: original FP16 guard zeroes failed solutions; it does NOT retry in FP32. A production FP32 retry remains required before claiming a general safe solver.
- Optional padded ranks and Yahoo/Amazon: not implemented/not collected. Existing 60,000-entity batches already split datasets whose user counts exceed that size; a larger dataset would add scale coverage.

## Newly gathered numerical findings

| Dataset | K | atomic / FP16 wall | same-dataflow / FP16 wall | same-dataflow / FP16 compute phase |
|---|---:|---:|---:|---:|
| ml10m | 16 | 3.412 | 1.749 | 1.988 |
| ml32m | 16 | 3.967 | 1.529 | 1.661 |
| netflix | 32 | 4.508 | 1.470 | 2.193 |
| netflix | 96 | 9.921 | 2.283 | 3.418 |

The compute-phase column sums the legacy User/Item LHS+RHS event intervals. Because these join the overlapping solve streams, they include solver work and MUST NOT be labeled isolated Gram-kernel time. Runs are single measurements, not new variance estimates. The same-dataflow control uses FP32 staging and direct FMA, one warp/entity, 16-row chunks, eight accumulator entries per lane per upper tile, and no Gram atomics. At K<=48, FP16 also uses its existing prefetch path; FP32/TF32 staging is wider and lacks that prefetch. Treat this as a structural dataflow control, not an instruction-identical substitution.

Validation safety failures (retained in `validation_sweep.csv`): ml32m_ship_k96_lambda0.01: 1 zeroed solves.

- Matched BF16/FP16 safety controls: 9/12 additional FP16 runs in `bf16_controls.csv`; FP16 with one refinement at beta .3/.5 and shipped no-refinement FP16 at beta .5. GPU0 accuracy/safety diagnostics, not cross-GPU timing comparisons.

- Selected-recipe test evaluations: 17/20; `heldout_test.csv`.
- New-split controlled FP32/FP16 accuracy pair: 27/40; `validation_accuracy_pair.csv`. Fixed 30 sweeps, FP32 solver, no momentum/giant tier, validation-selected lambda, test checked only at the end.
- Rounded-input FP32 controls: 4/20; `rounding_control.csv`.
- Chunk-reset tensor accumulation controls: 3/20; `accumulation_control.csv`. These GPU0 diagnostic timings are not headline performance evidence.

Validation selections: ml10m: lambda=0.06, validation RMSE=0.787916, netflix: lambda=0.04, validation RMSE=0.820528.

- atomic_fp32: relative upper-triangle Frobenius error median 1.17e-07, maximum 3.41e-07.
- dense_fp16: relative upper-triangle Frobenius error median 0.000195, maximum 0.00194.
- sameflow_fp32: relative upper-triangle Frobenius error median 1.94e-06, maximum 8.61e-06.
- giant_fp16: relative upper-triangle Frobenius error median 4.82e-05, maximum 0.000482.

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

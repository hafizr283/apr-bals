# FP16 solver fallback protocol

## Scope and source

The implementation snapshot is `fallback_src/`; the pre-change experimental source remains unchanged in `src/`. The active build is identified at startup as `FP32-FALLBACK-2026-10-15`. Builds use CUDA `sm_86`, weighted ALS-WR regularization, the fixed cuMF-scale initialization, rank 96, momentum enabled with the beta selected by each configuration, exact FP32 RMSE evaluation, and 30 fixed sweeps. No regularization, momentum, refinement count, stopping rule, split, or initialization was changed within a comparison.

The original FP16 tiled solver had two active failure exits in `cholesky_solve_tiled_mp`: (1) the stored-diagonal overflow check before factorization and (2) the post-triangular-solve non-finite check. Both wrote an all-zero entity vector. The experimental stale-factor path had a third zeroing exit after refinement. The fallback snapshot retains the right-hand side in global memory on failure, appends the batch-local entity index to a device list, and launches the existing FP32 tiled Cholesky solver on that list in the same solver stream. The FP32 retry checks every Cholesky pivot for a finite positive value and checks every output component for finiteness. It writes the FP32 result on success and writes zeros only if this checked FP32 retry fails. The stale path now enqueues its failed result for a fresh solve instead of immediately zeroing it; stale mode was not enabled in these measurements.

Six unsigned 64-bit counters are reset once per training run and printed as one machine-readable `FALLBACK_COUNTERS` line: total entity solves, successful FP16 solves, FP16 failures, FP32 attempts, FP32 recoveries, and FP32 failures. The audit enforces `total = fp16_success + fp16_failure`, `fp16_failure = fp32_attempt`, and `fp32_attempt = fp32_recovery + fp32_failure` for every fallback log. A separate `SOLVER_STREAM_MS` line reports the sum of per-batch intervals from LHS readiness through completion of the FP16 solve and possible retry. These intervals may overlap LHS/RHS work and therefore are solver-stream work, not an additive decomposition of end-to-end wall time.

## Configurations

Primary accuracy and timing runs use GPU 1 exclusively; all pre-run snapshots show GPU 1 at 0% utilization. Every cell has three serial repetitions.

- Stable new-split control: Netflix K=96, lambda 0.04 selected on validation, beta 0.3, shipped FP16 without refinement. The same configuration was run with fallback, legacy zeroing, and all-FP32 solving.
- Difficult new-split controls: Netflix K=96, beta 0.5, lambda 0.048 from the existing failure protocol, with either zero or one refinement step. The all-FP32 solver is the reference. Training uses the approximately 70/10/20 split; validation is scored from the dumped factors against the companion validation file and test RMSE uses the preserved original test rows.
- Original-split failure reproduction: Netflix K=96, beta 0.5, lambda 0.048, with zero or one refinement step. This exactly targets the protocol that previously produced thousands of zeroed solves. It is retained as a stress reproduction and is not mixed with new-split absolute RMSE claims.

The zero-on-failure binaries are compiled from the unchanged `src/` snapshot. The FP32 reference uses the same validated tiled FP32 solver from that snapshot. Every run dumps final factors after the timed region. The runner checks all factor elements for finiteness, scores all validation predictions for the new split (or all original test predictions for the legacy split), checks every prediction for finiteness, and then removes the temporary factor dumps. CPU rescoring of legacy test predictions agrees with logged GPU RMSE within 3.75e-7.

## Interpretation boundary

Across these tested configurations, every FP32 retry succeeded and all final factors, evaluated predictions, validation RMSE values, and test RMSE values were finite. This supports a bounded claim for Netflix K=96 under the listed lambdas, beta values, refinement choices, split definitions, GPU, and 30-sweep budget. It does not establish universal solver safety across ranks, datasets, regularization regimes, architectures, or arbitrary ill-conditioned systems. The FP32 validity gate remains necessary; if it fires, the counter records the unrecovered failure and only then is the solution zeroed.

Observed median wall-time differences versus matching zero-on-failure builds range from -0.119% to -0.001% in failure cases and -0.087% in the stable control. These negative point estimates are treated as no detectable overhead, not as speedups; the three-run ranges overlap or differ by only milliseconds. The fallback builds remain 5.5-14.6% faster than all-FP32 solving in the corresponding tested cells, depending on refinement and split.

## Reproduction and artifacts

Run `python3 new_results/run_fp32_fallback.py` for the new-split queue, then `python3 new_results/run_fp32_fallback_legacy.py` for the original-split stress queue. Both scripts rebuild their binaries and record exact commands and environments under `fallback_commands/`. Run `python3 new_results/analyze_fp32_fallback.py` to verify raw logs and regenerate the aggregate CSV and Markdown/LaTeX tables. The runners do not generate plots.

- Per-run primary data: `fp32_fallback_runs.csv`
- Original-split stress data: `fp32_fallback_legacy_runs.csv`
- Aggregate data: `fp32_fallback_summary.csv`
- Raw stdout: `fallback_logs/`
- Exact command records: `fallback_commands/`
- Machine audit and hashes: `fp32_fallback_audit.json`
- Paper table: `FP32_FALLBACK_TABLE.md` and `paper_tables/fp32_fallback.tex`

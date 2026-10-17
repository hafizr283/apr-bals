# Continue the paper-results task

User wants the experiments requested in `whattodo/review`, results suitable for the paper, and a summary of required/gathered/remaining work. User explicitly requests both GPUs when useful, continued work, and **NO MORE PLOTTING**. Do not spawn agents. Preserve original source/user edits; all new work lives in `new_results`.

Workspace: `/home/pc/Desktop/2007080/latent_factorr_experimen/comparing_baseline`.

## Live state

- External queue: **COMPLETE**. Runner `run_external.py`, PID26723 (check before restarting), log `logs/external_progress.txt`; completion marker `logs/external.done`. GPU1. All other queues finished. Never restart the full script blindly: it does not skip completed runs.
- External curves: **84/84**. Each APR/ALS/CCD/BALS curve needs30 points; SGD40. `external_verification.json` checks lengths and final SGD GPU evaluation against saved-model CPU scoring.
- GPU0 completed numerical/accuracy and matched BF16 controls. Primary performance runs use GPU1. Both GPUs were used concurrently at the user's request. GPU0 serves the desktop; its diagnostic timings are not performance evidence. Shared-host overlap is disclosed in EXTERNAL_PROTOCOL.md.
- Administrator Nsight counters: **COMPLETE**. Valid section profiles are `ncu/dense_stalls.csv` and `ncu/giant_stalls.csv`; explicit counters are `ncu/dense_explicit_stalls.csv` and `ncu/giant_explicit_stalls.csv`. Dense long-scoreboard/LG-throttle are 44.82%/1.05%; giant values are 19.56%/1.23%. Dense/giant tensor-pipe utilization is 20.4%/24.5%. The old `ncu/permission_check.txt` and `ncu/stalls.txt` are failed non-administrator attempts and are NOT counter evidence.

## Completed results

- FP32 solver fallback is implemented in `fallback_src/`. Failed FP16 solves retain their RHS, enter a device failure list, and are retried by the validated FP32 tiled solver; only a checked FP32 failure is zeroed. Machine-readable counters cover total solves, FP16 successes/failures, FP32 attempts/recoveries/failures. The unchanged `src/` is the zero-on-failure control.
- Fallback evaluation is complete: 24 new-split runs and 15 original-split stress runs, all serial on GPU1 with three repetitions per cell. Stable Netflix K96 lambda.04 beta.3 had zero failures. New-split beta.5 lambda.048 had 2/14,938,770 FP16 failures per fallback run, all recovered; original-split fallback had 33 no-IR and 26-27 IR1 failures, all recovered. No FP32 retry failed. Final factors and evaluated predictions were finite in every run.
- New-split fallback median validation/test RMSE is .822030/.821171 (no IR) and .822022/.821168 (IR1), versus all-FP32 .822022/.821168. Original-split fallback test RMSE is .817964/.817956, versus FP32 .817956. Zero-on-failure controls reproduce the failure cascade and severe accuracy loss on the original split.
- GPU1 median wall overhead versus the same zero-on-failure FP16 configuration is not detectable: point estimates are -0.119% to -0.001% in failure cells and -0.087% in the stable cell. Do not call the negative estimates speedups. Raw/aggregate CSVs, logs, commands, protocol, tables, and a passing 39-run audit are under the `fp32_fallback*`, `fallback_*`, and `FP32_FALLBACK*` paths.

- Four-way precision/dataflow comparison:80 runs,4 datasets x5 ranks x4 variants. `four_way.csv`. Fixed20 sweeps, FP32 solver, no momentum/giants. K96 control tuned to zero-spill outer loop254regs; SASS has FFMA and no HMMA. Pilot archived. FP16 wall advantage vs this FP32 control:1.31–2.52x.
- Momentum:40 curves, beta0/.3,30 sweeps. `momentum_crossings.csv` separates adjacent-sweep and five-sweep-window stopping.
- New ~70/10/20 validation split, seed20261014, original test unchanged, verified. Selected lambda ML10/20/32/Netflix=.06/.048/.048/.04. `validation_selection.csv`. Twenty shipped-recipe test runs complete; K96 test RMSE=.786347/.774979/.764361/.819464. Selected configurations have zero guards. At lambda.01, ML32/Netflix zeroed1/2 solves; retain failures.
- New-split controlled accuracy:40 runs/20 pairs in `validation_accuracy_deltas.csv`; FP16 test RMSE lower0.0015–0.0117%. FP32 solver, no momentum/giants,30 sweeps, test checked only at end.
- FP64 Gram checks:3200 records,800 snapshots/path; top20 users/items eachdataset/rank. Original32-row tile neighborhoods preserved for atomic grouping; factor order checked. `gram_errors.csv`, raw factors/inputs/outputs in `gram/`.
- Atomic FP32 median/max Gram error1.23e-7/3.41e-7; sameflowFP32 2.01e-6/9.26e-6; denseFP16 1.86e-4/2.05e-3; giantFP16 4.55e-5/5.07e-4. Reviewer atomic-error hypothesis **rejected**.
-20 rounded-FMA controls and20 chunk-reset tensor controls isolate the RMSE shift. Resetting MMA every16 rows then scalar-FP32 combining restores test RMSE to within2e-6 of FP32 at all20 points. Chunked median Gram error4.05e-6, trace bias-2.18e-8 versus dense persistent MMA bias-2.16e-4. `accumulation_errors.csv` has4000 records. Interpretation/manuscript wording in NUMERICAL_AUDIT.md. Do not assert a portable hardware rounding mode.
- BF16:8 runs plus12 matched FP16 controls. MovieLens beta.5 stable for both. Netflix FP16 beta.5 fails: IR1 zeroes3142, RMSE1.440298; noIR zeroes3492, RMSE1.339762. BF16 IR1 zero guards, RMSE.817964. `bf16_matched_comparison.csv`. Both Gram and solver types change; not isolated individually. No universal safety/accuracy claim.
- Cumulative same-source NetflixK96:6 runs, wallseconds atomic79.82144, sameflow18.41733, FP32solve7.92772, FP16solve7.32146, fastcheck4.60367, momentum3.77525. Mostly10sweeps, momentum8. Sameflow-to-next also enables giant tier. Lambda fusion/async staging not separately switched.

## Remaining paper work

1. Integrate the fallback table and bounded safety wording into the authoritative manuscript only when explicitly requested; the manuscript remains unchanged in this task.
2. Editorial restructuring/template/name/reference fixes, clean repository/Zenodo publication, optional padded ranks/larger-dataset coverage, and broader solver-safety coverage remain separate work.
3. Do not rerun the completed external, numerical, Nsight, or fallback queues. **Never run `plot_results.py`**; existing PDFs remain stale by explicit instruction.

## Boundaries and remaining work

- Legacy LHS+RHS events include overlapped solver work: CSV `gram_s` means compute phase, NOT isolated Gram time. New timings are single runs; historical variance cannot be attached.
- External common warm-validation rows; APR/CCD/BALS ranks16/32/48/64/96, ALS20/30/50/60/100, SGD128. Targets predeclared .80/.79/.78/.84. cuMF FP16 switches off, CG6,batches1/3. CCD separately validation-tuned. SGD epoch barriers added; saved-model scoring checks scale/permutation. Full configuration caveats in EXTERNAL_PROTOCOL.md.
- New split retains IDs; prefix-based threshold dispatch becomes approximate after validation removal. All entities still covered; dispatch_audit.json quantifies it. Fresh sorting would require a new run protocol.
- The original `src/` guard still zeroes failures because it is preserved as the historical control. The completed implementation is `fallback_src/`; its tested evidence supports a bounded safety claim only for the documented Netflix K96 configurations. Manuscript restructure/editorial/template, author confirmation, clean repo/Zenodo, optional padded ranks/larger dataset, external final-test coverage and timing repetitions remain separate work. Do not claim the whole review complete.
- Authoritative manuscript: `whattodo/GPU_ALS_Matrix_Factorization (2).zip`, extracted unchanged into `manuscript/`; tpds_latex is older. Review text: `review.txt`. Ignore duplicate baseline zip.
- Dataset directory `/home/pc/Desktop/2007080/`: ratings100.bin, ratings10.bin, ratings.bin, ratings32.bin, netflix_ratings.bin.
- Original user modifications exist outside new_results. Do not revert them. Source copies: src/, bf16_src/, chunked_src/, external_src/. Audit278 numerical-run logs: only four documented guard-failure configurations, no missing final/metadata.

#!/usr/bin/env python3
"""Compact resumable state; intentionally does not generate plots."""
import csv
from pathlib import Path
R=Path(__file__).resolve().parent
def count(name):
 p=R/name
 return len(list(csv.DictReader(p.open()))) if p.exists() else 0
complete=(R/'logs/external.done').exists()
s=f'''# Continue the paper-results task

User wants the experiments requested in `whattodo/review`, results suitable for the paper, and a summary of required/gathered/remaining work. User explicitly requests both GPUs when useful, continued work, and **NO MORE PLOTTING**. Do not spawn agents. Preserve original source/user edits; all new work lives in `new_results`.

Workspace: `/home/pc/Desktop/2007080/latent_factorr_experimen/comparing_baseline`.

## Live state

- External queue: **{'COMPLETE' if complete else 'RUNNING'}**. Runner `run_external.py`, PID26723 (check before restarting), log `logs/external_progress.txt`; completion marker `logs/external.done`. GPU1. All other queues finished. Never restart the full script blindly: it does not skip completed runs.
- External curves: **{count('external_time_to_target.csv')}/84**. Each APR/ALS/CCD/BALS curve needs30 points; SGD40. `external_verification.json` checks lengths and final SGD GPU evaluation against saved-model CPU scoring.
- GPU0 completed numerical/accuracy and matched BF16 controls. Primary performance runs use GPU1. Both GPUs were used concurrently at the user's request. GPU0 serves the desktop; its diagnostic timings are not performance evidence. Shared-host overlap is disclosed in EXTERNAL_PROTOCOL.md.
- Administrator Nsight counters still unconfirmed. User agreed to run `sudo bash /home/pc/Desktop/2007080/latent_factorr_experimen/comparing_baseline/new_results/run_ncu_stalls.sh`. It waits for external.done. Check `ncu/dense_stalls.csv` and `giant_stalls.csv` for actual metric rows. `ncu/permission_check.txt` confirms ERR_NVGPUCTRPERM; sudo password unavailable. `ncu/stalls.txt` profiled no kernels and is NOT counter evidence.

## Completed results

- Four-way precision/dataflow comparison:80 runs,4 datasets x5 ranks x4 variants. `four_way.csv`. Fixed20 sweeps, FP32 solver, no momentum/giants. K96 control tuned to zero-spill outer loop254regs; SASS has FFMA and no HMMA. Pilot archived. FP16 wall advantage vs this FP32 control:1.31–2.52x.
- Momentum:40 curves, beta0/.3,30 sweeps. `momentum_crossings.csv` separates adjacent-sweep and five-sweep-window stopping.
- New ~70/10/20 validation split, seed20261014, original test unchanged, verified. Selected lambda ML10/20/32/Netflix=.06/.048/.048/.04. `validation_selection.csv`. Twenty shipped-recipe test runs complete; K96 test RMSE=.786347/.774979/.764361/.819464. Selected configurations have zero guards. At lambda.01, ML32/Netflix zeroed1/2 solves; retain failures.
- New-split controlled accuracy:40 runs/20 pairs in `validation_accuracy_deltas.csv`; FP16 test RMSE lower0.0015–0.0117%. FP32 solver, no momentum/giants,30 sweeps, test checked only at end.
- FP64 Gram checks:3200 records,800 snapshots/path; top20 users/items eachdataset/rank. Original32-row tile neighborhoods preserved for atomic grouping; factor order checked. `gram_errors.csv`, raw factors/inputs/outputs in `gram/`.
- Atomic FP32 median/max Gram error1.23e-7/3.41e-7; sameflowFP32 2.01e-6/9.26e-6; denseFP16 1.86e-4/2.05e-3; giantFP16 4.55e-5/5.07e-4. Reviewer atomic-error hypothesis **rejected**.
-20 rounded-FMA controls and20 chunk-reset tensor controls isolate the RMSE shift. Resetting MMA every16 rows then scalar-FP32 combining restores test RMSE to within2e-6 of FP32 at all20 points. Chunked median Gram error4.05e-6, trace bias-2.18e-8 versus dense persistent MMA bias-2.16e-4. `accumulation_errors.csv` has4000 records. Interpretation/manuscript wording in NUMERICAL_AUDIT.md. Do not assert a portable hardware rounding mode.
- BF16:8 runs plus12 matched FP16 controls. MovieLens beta.5 stable for both. Netflix FP16 beta.5 fails: IR1 zeroes3142, RMSE1.440298; noIR zeroes3492, RMSE1.339762. BF16 IR1 zero guards, RMSE.817964. `bf16_matched_comparison.csv`. Both Gram and solver types change; not isolated individually. No universal safety/accuracy claim.
- Cumulative same-source NetflixK96:6 runs, wallseconds atomic79.82144, sameflow18.41733, FP32solve7.92772, FP16solve7.32146, fastcheck4.60367, momentum3.77525. Mostly10sweeps, momentum8. Sameflow-to-next also enables giant tier. Lambda fusion/async staging not separately switched.

## Finish next

1. Monitor external queue and resolve any failure without duplicating finished runs. When the marker says complete, there are no queued experiments left. Verify84 curves,30/40 points, all finite,4 SGD CPU checks.
2. Run `python3 new_results/summarize_external.py`, `verify_external.py`, `summarize.py`, `audit_artifacts.py`, `make_paper_tables.py`, and `write_handoff.py`. These do not plot. **Never run plot_results.py**. Existing PDFs are stale; figures/STATUS.md explains this.
3. Update SUMMARY.md generator/status and relevant protocol notes with final outcomes. External K96 table is `external_k96_comparison.csv`. Do not claim APR uniformly beats SGD: SGD wins target-time on ML20M and ML32M; APR wins ML10M (rank96 vs128). Netflix is a single-run near-tie (APR1.93082s, SGD1.91767s); no reliable ordering claimed.
4. Verify administrator counters if produced; otherwise mark permission-blocked. Do not stall forever waiting for unavailable sudo access.
5. Final reply: concise completed/findings/remaining summary, links to SUMMARY.md and this handoff. User asked approximate percentage: previously90–95% experiment collection, **not** entire review completion.

## Boundaries and remaining work

- Legacy LHS+RHS events include overlapped solver work: CSV `gram_s` means compute phase, NOT isolated Gram time. New timings are single runs; historical variance cannot be attached.
- External common warm-validation rows; APR/CCD/BALS ranks16/32/48/64/96, ALS20/30/50/60/100, SGD128. Targets predeclared .80/.79/.78/.84. cuMF FP16 switches off, CG6,batches1/3. CCD separately validation-tuned. SGD epoch barriers added; saved-model scoring checks scale/permutation. Full configuration caveats in EXTERNAL_PROTOCOL.md.
- New split retains IDs; prefix-based threshold dispatch becomes approximate after validation removal. All entities still covered; dispatch_audit.json quantifies it. Fresh sorting would require a new run protocol.
- FP16 guard **zeroes failed solutions**, no FP32 retry. Implementing retry remains required engineering. Manuscript restructure/editorial/template, author confirmation, clean repo/Zenodo, optional padded ranks/larger dataset, external final-test coverage and timing repetitions remain separate work. Do not claim the whole review complete.
- Authoritative manuscript: `whattodo/GPU_ALS_Matrix_Factorization (2).zip`, extracted unchanged into `manuscript/`; tpds_latex is older. Review text: `review.txt`. Ignore duplicate baseline zip.
- Dataset directory `/home/pc/Desktop/2007080/`: ratings100.bin, ratings10.bin, ratings.bin, ratings32.bin, netflix_ratings.bin.
- Original user modifications exist outside new_results. Do not revert them. Source copies: src/, bf16_src/, chunked_src/, external_src/. Audit278 numerical-run logs: only four documented guard-failure configurations, no missing final/metadata.
'''
(R/'HANDOFF.md').write_text(s)
print('Updated compact handoff')

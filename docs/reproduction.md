# Reproduction guide

[Entry page](../README.md) Â· [Evidence guide](evidence.md) Â· [October index](../part2/new_results/README.md)

This guide documents the commands and dependencies present in the archive. It does
not claim that training has been reproduced from a clean checkout. Existing runners
have execution gaps and may overwrite artifacts; they have not been repaired here.

## Inspect archived evidence on CPU

Use Python 3.9 or newer from the repository root. No extra package is required:

```bash
python scripts/verify_paper_evidence.py --check
python -m unittest discover -s tests -p 'test_evidence.py'
```

The first command is offline and read-only. Expected output is explicit warnings for
scope limitations followed by `PASS: archived evidence consistent; ...`. Unexpected missing
files, duplicate keys, numerical disagreement, broken references or documentation drift
exit nonzero. Tests use isolated temporary fixtures; they do not alter archived results.

The [manifest](evidence_manifest.json) defines sources, protocols, keys, coverage,
raw-log coverage, expected metrics, document fingerprints and pinned source lines.
Fingerprint comparisons normalize line endings. A successful check establishes
consistency within its [stated scope](evidence.md#claim-to-source-matrix), not GPU reproduction.

## Hardware and toolchains

| Campaign | Hardware | Recorded CUDA / driver | Evidence and qualification |
|---|---|---|---|
| July historical primary | RTX 3060, sm_86 | CUDA 12.0 / 580.159.03 | [Historical README snapshot](https://github.com/hafizr283/apr-bals/tree/e798aa242dc5026247426bd96ca6055be26b2354#2-requirements); campaign metadata, not a new hardware query. |
| August repeat-run study | RTX 3060, GPU1 on a two-device host | CUDA 12.9 / 580.173.02 | [Historical toolchain table](https://github.com/hafizr283/apr-bals/tree/e798aa242dc5026247426bd96ca6055be26b2354#2-requirements5-L69), [variance summary](../results/PAPER_MEASUREMENTS_20260803/variance_5x_summary.txt). |
| Historical T4 | Tesla T4, sm_75, Colab | Exact CUDA/driver pair not established by the curated sources | [T4 logs](../results/t4_latest_res_22_7_2026/); do not substitute the desktop toolchain. |
| October external | RTX 3060, sm_86, GPU1; another RTX 3060 on same host | CUDA 12.9, nvcc V12.9.41 / 580.178.04 | [environment.json](../part2/new_results/environment.json), [execution conditions](#external-evaluation-protocol). GPU0 diagnostics/desktop can share host resources. |

Training requires Linux, Bash/Python as applicable, a compatible NVIDIA driver and
CUDA toolkit, `nvcc`, and a host C++ compiler supported by that CUDA release. The
scripts use C++14 for CUDA builds. Python preparation/analysis scripts additionally
use NumPy and SciPy. Compiler version provenance is incomplete; do not treat a
generic installed compiler as the historical compiler without checking build logs.

TF32 controls require compatible hardware; the October sm_86 recipe should not be
copied unchanged to T4. Historical T4 coverage does not validate all October controls.
Disk and memory needs depend on datasets and generated factors; large binary files
are intentionally excluded from Git. Obtain datasets under their original terms.
The [artifact guide](artifacts.md) provides the existing ZIP link, exact size/checksum,
excluded-file inventory and recommendations for a durable journal deposit.

## Inputs and preparation

The [preprocessor](../preprocess.cpp) accepts a ratings CSV and output binary path:

```bash
g++ -O3 preprocess.cpp -o preprocess
./preprocess /path/to/ratings.csv /path/to/ratings.bin
```

This is the original split pipeline: frequency-based entity ordering, seeded per-user
train/test partitioning, then sparse training structures. It does not produce the
October external warm-validation inputs by itself. The binary layout stores a header
and length-prefixed training/test ratings plus CSR/CSC arrays; inspect the loader and
preprocessor together before substituting a format.

[convert_bin_for_ccd_sgd.py](../convert_bin_for_ccd_sgd.py) prepares external sparse/COO
formats from that binary. External methods must use the same ratings and ID mapping;
copying a binary filename into a runner does not establish that equivalence.

The archived October split command, from its originally intended parent directory, is:

```bash
python new_results/make_validation_split.py /path/to/ratings.bin
```

In the current repository that parent is `part2/`, but the script inserts that parent
into its import path while `count_empty_entities.py` lives at the repository root.
Its output also uses a working-directory-relative `new_results/splits` path. Resolve
those layout assumptions in a separate reproduction effort before running it.
The command records the intended interface, not a verified preparation recipe.

The script carves roughly one eighth of old training into validation, retaining at
least one training observation for each nonempty user. It emits validation/test `.bin`
files, validation indices `.npy`, and a manifest. Only available small metadata is
included here. The original test's historical exposure remains a limitation.

External evaluation also needs masked `warm_validation.bin`, CCD inputs, SGD train/test
inputs and mask manifests. The restored [prepare_external.py](../part2/new_results/prepare_external.py),
[verify_external.py](../part2/new_results/verify_external.py) and
[verify_splits.py](../part2/new_results/verify_splits.py) retain the original interfaces.
Preparation imports root-level helpers via an old parent-directory assumption; resolve
that layout before execution. Saved [mask manifests](../part2/new_results/external_inputs/)
record retained/removed ratings. These restored scripts have not been run here.

## Build recipes

These are the distinctions verified in the existing sources/runners; there is no new
unified build system. Use the appropriate campaign's source snapshot and flags.

| Target | Required recipe distinctions | Source of command |
|---|---|---|
| Historical standalone APR | `-O3 -arch=sm_86 -std=c++14 -DK_DIM=... -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1`; add `-DCHOL_MP=2` only at K96 | [run_justapr_vs_cumf.sh](../run_justapr_vs_cumf.sh) |
| Historical shared-binary control | `-DK_DIM=... -DCUMF_INIT=1`; weighted lambda defaults on, FP32 solvers in both legs, no momentum support | [run_rmse_comparison.sh](../run_rmse_comparison.sh), [main_experiment.cu](../main_experiment.cu) |
| Local standalone BALS | Scalar FP32; `-DBALS_REORDER=1` for reordered mapping; `-DBALS_SYMTILE=1` is a different optional mapping | [only_scalar_fp32.cu](../only_scalar_fp32.cu) |
| October shared base | `/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -std=c++14 -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -Xptxas=-v -DK_DIM=...` against October `src/` | [run_experiments.py](../part2/new_results/run_experiments.py) |
| October shipped APR | Base plus `-DMOMENTUM=1 -DFAST_RMSE=0`; `-DCHOL_MP=2` at K96 only | Same runner; prebuilt `bin/ship_k*` required by external runner |
| October four-way controls | FP16 default, TF32 `-DWMMA_TF32=1`, same-dataflow FP32 `-DWMMA_TF32=2`, atomic scalar `-DDENSE_NNZ_THRESH=2000000000 -DATOMIC_BASELINE=1`; giant disabled, exact RMSE, FP32 solver | Same runner; tuned K96 control additionally depends on follow-up source selection |
| October BALS | Base plus `-DBALS_REORDER=1 -DFAST_RMSE=0` against `src/bals.cu` | [run_external.py](../part2/new_results/run_external.py) |

Bare `nvcc justapr.cu` does not select the documented shipped recipe. Initialization,
weighted regularization, solver precision, momentum and checking frequency affect
the comparison. The October same-dataflow control is structurally similar to WMMA;
its staging width/prefetch behavior differs, so it is not an instruction-only substitution.

## Historical command map

After obtaining the inputs and toolchain, these existing interfaces illustrate the
original 80/20 experiments. They train models and write new output:

```bash
GPU=0 ARCH=sm_86 LAMBDA=0.048 bash run_rmse_comparison.sh /path/to/ratings.bin
GPU=0 ARCH=sm_86 LAMBDA=0.048 bash run_justapr_vs_cumf.sh /path/to/ratings.bin
```

The first writes `results/RMSE_COMPARISON_<timestamp>.txt` plus `logs_<timestamp>/`;
the second writes `JUSTAPR_VS_CUMF_<timestamp>.txt` and matching logs. Both default
to GPU1 unless overridden and include temporary compilation. Named datasets in the
second script resolve to machine-specific paths; explicit paths avoid only that issue.

Knobs include `GPU`, `ARCH`, `LAMBDA`, `K_VALUES`, `CUMF_F`, `THETA_BATCH`, `MAXIT`
and `VERBOSE`. The paired runner additionally has `RUN_MAIN`, `RUN_SCALAR` and `INIT`.
Keep APR/ALS rank lists aligned. Read the scripts for baseline executable dependencies.

[run_ccd_sgd_comparison.py](../run_ccd_sgd_comparison.py) and
[run_t4_everything.sh](../run_t4_everything.sh) are historical campaign scripts with
their own assumptions, dataset registry and output locations. They are not advertised
as a clean-machine reproduction package and have not been executed for this update.

## October runner map

Commands below are archived interfaces from repository root, listed for inspection.
They are not a recommended sequential launcher: prerequisites remain unresolved.

| Existing command | Inputs / behavior | Expected output | Known execution gaps |
|---|---|---|---|
| `python part2/new_results/run_experiments.py` | Original binaries, split binaries, October source; four-way, momentum, selection, held-out runs | Build logs, raw `.txt`, command `.json`, selection JSONs | Hardcoded `/home/pc/Desktop/2007080`, nvcc path and GPU1; existing `bin/` and `logs/` expected; writes into archive. |
| `python part2/new_results/run_followups.py` | Control binaries, original data, probes | Tuning/control/Gram/BF16/ladder files and completion marker | Waits for old `/proc/14817`; prebuilt dependencies; modifies `src/wmma_kernels.cuh`; overwrites some results. |
| `python part2/new_results/run_external.py` | Warm-validation inputs, selections, prebuilt `ship_k*`, ALS/CCD/SGD executables and CPU predictor | External raw logs, command JSON, CCD selection JSON, final prediction logs, `logs/external.done` | Waits indefinitely for `logs/followups.done`; old binary-relative and helper import paths; no validated clean build chain. |
| `python part2/new_results/summarize.py` | Available raw logs and selection records | CSV summaries | Rewrites evidence; protocol interpretation remains in the maintained guides. Not a checker. |
| `python part2/new_results/summarize_external.py` | External logs plus `external_targets.json` | External and K96 comparison CSVs | Target manifest restored; rewrites CSVs; a saved declaration alone does not establish independent preregistration. |

The restored [plot_results.py](../part2/new_results/plot_results.py) writes figures from
CSV summaries. Existing figures may be intermediate; no paper figures were regenerated.
Completion markers are synchronization artifacts, not proof that all planned runs exist.

External settings include APR beta 0.3, exact evaluation each sweep; ALS six CG steps,
X_BATCH=1 and THETA_BATCH=3; CCD `-T 1`, tiles a=b=100000 with separately selected lambda;
SGD rank128, lambda 0.05, alpha 0.08, beta 0.3, workers 256/384/512/512 by dataset.
Consult command metadata and the [external protocol](#external-evaluation-protocol)
before recreating a run. The recovered [target manifest](../part2/new_results/external_targets.json)
records targets and a prior-declaration note; the checker compares its targets with
the CSVs. That saved note does not independently establish when preregistration occurred.

## Baseline provenance

| Implementation | Provenance and local changes |
|---|---|
| cuMF_als | [Upstream](https://github.com/cuMF/cumf_als), vendored in [cumf_als-master](../cumf_als-master/). CUDA sparse-API port, test-grid correction, empty-entity guard, every-sweep evaluation and cumulative wall instrumentation. It is not an untouched upstream executable. |
| cuMF_CCD | [Upstream](https://github.com/cuMF/cumf_ccd), vendored in [cumf_ccd-master](../cumf_ccd-master/). CUDA compatibility changes, cumulative timing and double host error summation. The old `iter ... time` field is a different timer. |
| cuMF_SGD | [Upstream](https://github.com/cuMF/cumf_sgd), vendored in [cumf_sgd-master](../cumf_sgd-master/). Per-epoch launch/synchronization, GPU validation and fixed seeds with nondeterministic Hogwild scheduling. Evaluation instrumentation changes overhead. |
| Local BALS | [only_scalar_fp32.cu](../only_scalar_fp32.cu). October's [src/bals.cu](../part2/new_results/src/bals.cu) is restored from the archive; a separate [chunked-source copy](../part2/new_results/chunked_src/bals.cu) is a distinct variant. Local reimplementation; the external protocol specifies reorder-only mapping, weighted lambda, FP32 and no momentum. |

Existing third-party attribution and license files remain in their original locations.
Root and October source snapshots are different artifacts; use the saved source hashes
and per-run commands to establish provenance. Training-loop wall includes training
RMSE for APR/BALS/ALS but held-out evaluation only for CCD/SGD. That cost difference,
rank mismatch, warm mask and shared-host conditions must accompany timing comparisons.

## External evaluation protocol

The October evaluation uses approximately 70% training, 10% validation and 20%
preserved original test ratings. Every external method sees the same warm-validation
mask: both entities must occur in training. The mask changes evaluation only;
[input manifests](../part2/new_results/external_inputs/) record retained/removed ratings.
Test results and validation-target curves answer separate questions. Non-finite
predictions are not silently removed, and warm masking does not establish cold-start accuracy.

APR uses validation-selected lambda, beta 0.3, FP32 solves below K96 and FP16 tiled
solves at K96. Training and validation RMSE are evaluated in FP32 each sweep.
The 30-sweep curves support crossings computed afterward from validation targets.
Local BALS uses frequency reordering, APR's initialization scale and weighted lambda,
FP32 arithmetic, no momentum, reorder-only mapping and exact per-sweep evaluation.

The guarded cuMF_als port uses items-by-users orientation, CUDA sparse-API replacements,
buffers allocated outside timing, a test-grid tail correction, six CG steps,
X_BATCH=1 and THETA_BATCH=3. Its FP16 operand/CG switches are disabled. Empty-entity
factors are zeroed after each CG batch; error reduction uses double accumulation.

CCD selects lambda from {0.02, 0.04, 0.05, 0.058, 0.08, 0.12} by K96/sweep30
warm-validation RMSE and carries it across ranks. Settings are `-T 1`, tiles
a=b=100000 and 30 sweeps. Host error summation is double. Its legacy compute timer
is distinct from the cumulative training-loop wall timer used for target crossings.

SGD uses rank128, lambda 0.05, alpha 0.08, beta 0.3, 40 epochs and workers
256/384/512/512 for ML-10M/20M/32M/Netflix. Each epoch advances `current_iter` while
retaining factors and random state. Per-epoch kernel launches introduce synchronization
relative to the original persistent kernel. Validation uses the training ID permutation
and rating scale, with FP64 squared-error accumulation; final scores are checked against
the saved-model CPU predictor. Seed 20261014 does not eliminate Hogwild race nondeterminism.

APR/CCD/BALS use K=16,32,48,64,96. ALS uses F=20,30,50,60,100 for those comparisons;
SGD K128 is compared only with APR K96. All training-loop timers include each method's
evaluation; APR/BALS/ALS also compute training RMSE whereas CCD/SGD evaluate held-out
ratings only. Loading and preparation are excluded. These overheads and mismatched
ranks remain part of the comparison, which has no new timing repetition estimate.

External performance runs used GPU1 while independent accuracy/numerical diagnostics
ran on GPU0 on the same two-RTX-3060 host. GPU0 also served the desktop. Its diagnostic
times are not performance comparisons; shared CPU, memory and PCIe contention can
affect GPU1 timings. The split retains original IDs, so contiguous dispatch prefixes
can include entities below their degree thresholds. The disjoint paths still cover
entities once; [dispatch_audit.json](../part2/new_results/dispatch_audit.json) quantifies
the routing limitation. A fresh sort would require a separate measured configuration.

## FP32 fallback protocol

The [fallback source](../part2/new_results/fallback_src/) retries failed FP16 entity
solves with checked FP32 tiled Cholesky. The original [source](../part2/new_results/src/)
remains the zero-on-failure control. The fallback preserves the RHS, records batch-local
failed entity IDs, and launches the retry in the same solver stream. It checks finite
positive pivots and finite outputs; an unrecovered FP32 failure is counted before zeroing.

The recorded build identifies itself as `FP32-FALLBACK-2026-10-15`, targets `sm_86`,
and uses weighted ALS-WR, cuMF-scale initialization, K96, momentum, exact FP32 RMSE
and 30 fixed sweeps. Each comparison cell has three serial GPU1 repetitions.

| Configuration | Split and controls |
|---|---|
| Stable Netflix | New split; selected lambda 0.04, beta 0.3, no refinement; fallback, legacy zeroing and FP32 controls. |
| Difficult Netflix | New split; lambda 0.048, beta 0.5; zero/one refinement, with all-FP32 reference. |
| Original-split stress | Original 80/20; lambda 0.048, beta 0.5; zero/one refinement. Keep these absolute RMSEs separate from the new split. |

The counter identities are `total = fp16_success + fp16_failure`,
`fp16_failure = fp32_attempt`, and `fp32_attempt = fp32_recovery + fp32_failure`.
`SOLVER_STREAM_MS` covers LHS readiness through the solve/retry and can overlap Gram
work; it is not an additive decomposition of wall time. Factor dumps occur after timing.
The saved run audit checks factors/predictions for finiteness and CPU/GPU scoring agreement.
Evidence supports the documented configurations only, not universal solver safety.
Millisecond-scale negative overhead estimates are not established speedups.

The existing interfaces are `python part2/new_results/run_fp32_fallback.py`,
`python part2/new_results/run_fp32_fallback_legacy.py` and
`python part2/new_results/analyze_fp32_fallback.py`. Runners rebuild and write artifacts;
the analyzer checks logs and writes summary CSV, LaTeX and JSON. Their path/toolchain
requirements still apply; none was used to launch training during documentation cleanup.
See [primary runs](../part2/new_results/fp32_fallback_runs.csv),
[original-split runs](../part2/new_results/fp32_fallback_legacy_runs.csv),
[commands](../part2/new_results/fallback_commands/), [logs](../part2/new_results/fallback_logs/)
and [saved audit](../part2/new_results/fp32_fallback_audit.json).
The CPU paper checker inventories these fallback summaries; its general trajectory
replay coverage is stated separately in the evidence guide.

## Figure provenance

October figures are intermediate artifacts; in particular `four_way.pdf` predates
the final K96 control tuning. Use current CSVs and verify a figure against them before
publication. Cleaning documentation does not regenerate or validate the paper figures.

The archived manuscript reads its own `manuscript/figures/` directory. Same-named
figures from another thesis/paper tree are not interchangeable. Its
[generation scripts](../part2/new_results/manuscript/figures/) retain the original
paths: architecture/tier diagrams came from `generate_figures2.py`, primary result
figures from `generate_figures4.py`, and CCD/BALS figures from `generate_figures3.py`.
These mappings describe that historical manuscript, not a claim that its figures use
the final October results. Repoint inputs explicitly in a separate figure update.

## What remains for full reproduction

The four previously missing October source files and five helper/config files are
restored. The recovered `src/wmma_kernels.cuh` matches the final archived SHA-256
in `artifact_sha256.json`, rather than the older original/root header. Source presence
does not establish a clean build; follow-up runners patch files and depend on existing
binaries, working directories and completion markers.

Resolve paths and working directories, establish source/target provenance and
a clean build chain for each external port, obtain
the excluded inputs under their terms, and verify saved masks/splits and source hashes.
Then run campaigns into separate output locations with explicit hardware, seeds,
checking rules and repeats. Those repairs and GPU experiments are outside this phase.

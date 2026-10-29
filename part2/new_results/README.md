# October evidence index

[Repository entry](../../README.md) · [Evidence guide](../../docs/evidence.md) · [Reproduction guide](../../docs/reproduction.md) · [Historical experiments](../../results/README.md)

This directory contains October CSV summaries, available raw logs, command metadata,
protocol notes and source snapshots. Coverage is partial. The evidence guide separates
calculated results, raw-log replay and preserved diagnostics. No overall campaign
completion claim is made.

## Supported external results

Warm-validation time-to-target results give **2.6–5.2× against cuMF_als across twenty
documented rank pairings**, and **3.3–6.9× against cuMF_CCD across twenty matched-rank
comparisons**. These are single runs. The ALS K96/F100-only range is 2.75–4.81×;
it must not be labeled an all-rank range.

At APR K96, absolute final warm-validation RMSE difference is below **0.1% versus
ALS F100** and **0.2% versus CCD K96**, using each baseline's RMSE as denominator.
APR is slightly worse than CCD on ML-10M and better on the other datasets.
See [source rows, calculations and direction](../../docs/evidence.md#final-validation-accuracy).

[External CSV](external_time_to_target.csv) · [Protocol](EXTERNAL_PROTOCOL.md) · [Raw files](external/) · [Mask metadata](external_inputs/)

## Momentum and partial controls

Momentum has **six complete pairs**, twelve raw curves, and an adjacent-sweep
training-delta crossing advantage of **1.10–1.23×** for beta 0.3 over beta 0.
These are original-split diagnostics, not external validation-target crossings.
[Both stopping-rule calculations](../../docs/evidence.md#momentum) include source links.

Coverage below means recorded rows, not intended completions. The cumulative ladder
has three rows out of six planned stages; no full-ladder speedup is promoted.

<!-- evidence:coverage:start -->
| Evidence table | Available records | Planned records | Protocol | Verification scope |
|---|---:|---:|---|---|
| [accumulation_control.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/accumulation_control.csv?plain=1#L1) | 3 | 20 | N80 | Inventory and fingerprints |
| [accumulation_errors.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/accumulation_errors.csv?plain=1#L1) | 4,000 | Not a run total | N80 | Inventory and fingerprints |
| [bf16.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/bf16.csv?plain=1#L1) | 3 | 8 | N80 | Available raw final summaries |
| [bf16_controls.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/bf16_controls.csv?plain=1#L1) | 9 | 12 | N80 | Available raw final summaries |
| [bf16_matched_comparison.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/bf16_matched_comparison.csv?plain=1#L1) | 1 | Not a run total | N80 | Inventory and fingerprints |
| [cumulative.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/cumulative.csv?plain=1#L1) | 3 | 6 | C80 | Available raw final summaries |
| [external_time_to_target.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/external_time_to_target.csv?plain=1#L1) | 84 | 84 | E70 | CSV calculations; 59 raw trajectories |
| [four_way.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/four_way.csv?plain=1#L1) | 52 | 80 | N80 | Available raw final summaries |
| [gram_errors.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/gram_errors.csv?plain=1#L1) | 2,080 | Not a run total | N80 | Inventory and fingerprints |
| [heldout_test.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/heldout_test.csv?plain=1#L1) | 17 | 20 | O70 | Available raw final summaries |
| [momentum_crossings.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/momentum_crossings.csv?plain=1#L1) | 12 | 40 | M80 | 12 raw curves; six complete pairs |
| [rounding_control.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/rounding_control.csv?plain=1#L1) | 4 | 20 | N80 | Inventory and fingerprints |
| [validation_accuracy_deltas.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/validation_accuracy_deltas.csv?plain=1#L1) | 7 | 20 | O70 | Inventory and fingerprints |
| [validation_accuracy_pair.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/validation_accuracy_pair.csv?plain=1#L1) | 27 | 40 | O70 | Available raw final summaries |
| [validation_selection.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/validation_selection.csv?plain=1#L1) | 2 | 4 | O70 | Inventory and fingerprints |
| [validation_sweep.csv](https://github.com/hafizr283/apr-bals/blob/e798aa242dc5026247426bd96ca6055be26b2354/part2/new_results/validation_sweep.csv?plain=1#L1) | 20 | 32 | O70 | Available raw final summaries |
<!-- evidence:coverage:end -->

The external CSV has **84 records**, with **59 available raw trajectories**.
The [ledger of 25 gaps](../../docs/evidence.md#raw-coverage-ledger) identifies missing logs.
Recorded historical verification reports do not replace those files.

## Directory index

| Path | Contents |
|---|---|
| [src/](src/), [external_src/](external_src/) | October APR/control and instrumented external source snapshots. |
| [four_way/](four_way/), [control_tuning_runs/](control_tuning_runs/) | Available control runs and tuning records. |
| [validation/](validation/), [heldout_test/](heldout_test/) | Selection and final evaluation records. |
| [splits/](splits/), [external_inputs/](external_inputs/) | Small preparation metadata; binary inputs excluded. |
| [momentum/](momentum/) | Per-sweep diagnostic trajectories. |
| [gram/](gram/), [accumulation_control/](accumulation_control/) | Probe JSONs, commands, diagnostic messages and control runs. |
| [bf16/](bf16/), [bf16_controls/](bf16_controls/), [rounding_control/](rounding_control/) | Partial numerical-control evidence. |
| [cumulative/](cumulative/) | Available cumulative-stage measurements. |
| [fallback_src/](fallback_src/), [fallback_logs/](fallback_logs/) | Separate fallback code and diagnostics; no aggregate claim here. |
| [ncu/](ncu/), [logs/](logs/) | Profiling/build/execution records, including failures. |
| [environment.json](environment.json), [dispatch_audit.json](dispatch_audit.json) | Hardware, source and dispatch metadata. |

## Verification and execution boundaries

From the repository root:

```bash
python scripts/verify_paper_evidence.py --check
```

Expected: declared-gap warnings followed by `PASS: archived evidence consistent; ...`.
This CPU check does not train models or rewrite result files.
See the [claim matrix](../../docs/evidence.md#claim-to-source-matrix) for exact scope.

Runners and summarizers can overwrite results; some rely on old paths, prebuilt
binaries, completion markers or missing helpers. The
[reproduction guide](../../docs/reproduction.md#october-runner-map) documents those gaps.
`summarize.py` and `summarize_external.py` are not read-only checkers.

Earlier narrative summaries and figures remain context. Some describe planned or
broader experiments than the stored CSVs support. This index and the evidence guide
supersede their headline ranges and completion descriptions.

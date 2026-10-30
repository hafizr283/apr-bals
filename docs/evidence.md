# Evidence guide

[Entry page](../README.md) · [October index](../part2/new_results/README.md) · [Reproduction](reproduction.md) · [Historical index](../results/README.md)

This guide connects documentation claims to archived measurements. A checked CSV
value is not proof of a rerunnable experiment. Source links pin evidence to commit
`437e10e55c066e50c6077043313eec2fdaf46247`; GitHub highlights the linked line.
This commit contains restored source snapshots, complete CSVs and recovered raw logs.

## Protocols

| ID | Data, configuration, and interpretation | Protocol source |
|---|---|---|
| H80 | Original per-user approximately 80/20 train/test split, seed 42, frequency-reordered IDs. Historical paired harness: weighted lambda 0.048, cuMF-scale initialization, FP32 solvers in both legs, no momentum, training-delta stopping checked every five sweeps. | [Preprocessor](../preprocess.cpp), [runner](../run_rmse_comparison.sh), [variance summary](../results/PAPER_MEASUREMENTS_20260803/variance_5x_summary.txt) |
| O70 | Approximately 70/10/20: validation carved from old training, seed 20261014, preserving original test ratings. New validation selection is distinct from historical test-selected lambda. | [Split code](../part2/new_results/make_validation_split.py), [saved verification](../part2/new_results/splits/verification.json) |
| E70 | O70 training, shared warm-validation mask. Targets: ML-10M 0.80, ML-20M 0.79, ML-32M 0.78, Netflix 0.84. APR/ALS/CCD/BALS: 30 sweeps; SGD: 40 epochs. Timers include each method's evaluation. | [Protocol](reproduction.md#external-evaluation-protocol), [CSV](../part2/new_results/external_time_to_target.csv), [mask metadata](../part2/new_results/external_inputs/) |
| M80 | Original split; shipped solver policy; beta 0 versus 0.3, fixed 30 sweeps, exact evaluation every sweep. First adjacent absolute training-RMSE change below 0.001 defines crossing. | [Runner](../part2/new_results/run_experiments.py), [crossings](../part2/new_results/momentum_crossings.csv) |
| N80 | Original-split numerical controls. Four-way: 20 fixed sweeps, FP32 solver, no momentum or giant tier. BF16 and accumulation interventions have separate flags and budgets. | [Runner](../part2/new_results/run_experiments.py), [follow-ups](../part2/new_results/run_followups.py), [BF16 controls](../part2/new_results/run_bf16_controls.py) |
| C80 | Original-split ladder, Netflix K96, lambda 0.048; per-sweep delta stopping with cap 30. All six recorded stages are indexed here. | [Ladder protocol](#cumulative-stage-definitions), [CSV](../part2/new_results/cumulative.csv) |

APR/CCD/BALS ranks are 16, 32, 48, 64, 96. Corresponding ALS F values are
20, 30, 50, 60, 100. The CSV's ALS `K` denotes the APR comparison rank, not actual F.
SGD K128 is compared only with APR K96. Only ALS uses the mismatched K/F mapping; APR, CCD and BALS use matched ranks.

The original test was historically inspected for tuning. A later validation carve-out
cannot establish a previously untouched test. Warm-validation masking does not
establish cold-start accuracy; it changes evaluation, not training data.

## Claim-to-source matrix

“Recomputed” means CPU replay of archived evidence. “Inventory” means keys, coverage,
finite values and fingerprints are checked, without independent numerical rederivation.

| Claim / family | Protocol | Configuration | Source | Calculation | Available coverage | Verification status |
|---|---|---|---|---|---|---|
| ALS target-time range | E70 | APR K / ALS F pairs | [Rows](#external-time-to-target) | ALS seconds / APR seconds; extrema | 20 pairings | Recomputed; all raw trajectories available |
| CCD target-time range | E70 | Matched K | [Rows](#external-time-to-target) | CCD seconds / APR seconds; extrema | 20 comparisons | Recomputed; all raw trajectories available |
| Dataset mean speedups | E70 | Five ranks each | [Means](#dataset-means) | Arithmetic mean of ratios | Four datasets per baseline | Recomputed |
| Final RMSE bounds and direction | E70 | APR K96, ALS F100, CCD K96 | [RMSEs](#final-validation-accuracy) | 100 × (APR − baseline) / baseline | Four datasets per baseline | Recomputed; single runs |
| BALS / SGD comparisons | E70 | BALS matched K; SGD K128 vs APR K96 | [BALS](#external-time-to-target), [SGD](#sgd-context) | Baseline seconds / APR seconds | 20 BALS, four SGD | Recomputed; no significance claim |
| CCD lambda | E70 | K96, sweep30 | [Selections](#ccd-regularization) | Minimum candidate RMSE; tie smaller lambda | Four JSONs | Recomputed; compared with final CCD CSV RMSE |
| Momentum advantage | M80 | Beta 0 / 0.3 | [Crossings](#momentum) | Ratio of wall times at first delta crossing | Twenty pairs, forty curves | Both stopping rules reparsed |
| Four-way controls | N80 | Atomic, FP16, TF32, same-dataflow FP32 | [CSV](../part2/new_results/four_way.csv) | Final metrics and timing fields | See coverage | Available final raw summaries reparsed |
| APR selection / final tests | O70 | Selected recipe | [selection](../part2/new_results/validation_selection.csv), [sweep](../part2/new_results/validation_sweep.csv), [test](../part2/new_results/heldout_test.csv) | Selection inventory; sweep/test raw finals | See coverage | Recorded grid restored; separate from CCD selection |
| New-split accuracy pairs | O70 | FP16 / same-dataflow FP32 | [pairs](../part2/new_results/validation_accuracy_pair.csv), [deltas](../part2/new_results/validation_accuracy_deltas.csv) | Run finals; delta-table fingerprint | See coverage | Pair table raw replay; deltas inventory |
| Gram / accumulation diagnostics | N80 | Same-factor entity/path probes | [Gram](../part2/new_results/gram_errors.csv), [accumulation](../part2/new_results/accumulation_errors.csv) | Count entity/path keys, not runs | See coverage | Inventory; binary factors excluded |
| Rounding / chunk reset | N80 | Separate interventions | [rounding](../part2/new_results/rounding_control.csv), [chunk reset](../part2/new_results/accumulation_control.csv) | Fingerprints and keys | See coverage | Inventory; no all-rank accuracy bound promoted |
| BF16 safety controls | N80 | BF16 and FP16 controls | [BF16](../part2/new_results/bf16.csv), [controls](../part2/new_results/bf16_controls.csv), [matched](../part2/new_results/bf16_matched_comparison.csv) | Available run finals; matched inventory | See coverage | Recorded grid restored; no universal safety claim |
| Cumulative ladder | C80 | Changing optimization recipe | [CSV](../part2/new_results/cumulative.csv) | Available finals and sweep counts | Six of six stages | Raw finals reparsed; no total-ladder speedup asserted |
| Historical variability | H80 | Shared-binary control, RTX 3060 | [History](#historical-evidence) | Archived summary; raw completion | 100 logs, 20 configurations | Summary fingerprint; variance not transferred to E70 |
| Historical T4 example | H80 | Netflix K96 | [History](#historical-evidence) | Baseline / APR wall | One cited configuration | Ratio reparsed |
| Hardware / mask / dispatch | Campaign-specific | Saved metadata | [masks](../part2/new_results/external_inputs/), [environment](../part2/new_results/environment.json), [dispatch](../part2/new_results/dispatch_audit.json) | Metadata inspection and fingerprints | Available metadata | No preparation rerun |
| Fallback / profiling archives | Configuration-specific | Diagnostic source variants | [fallback](../part2/new_results/fallback_logs/), [profiling](../part2/new_results/ncu/) | No aggregate result promoted | No curated run denominator | Preserved context; outside numerical replay |

## External time to target

Speedup = `baseline.seconds_to_target / apr.seconds_to_target` for a paired dataset/rank.
Across twenty comparisons, ALS spans **2.640216550–5.192919658×** and CCD spans
**3.270742821–6.859722125×**, rounded to **2.6–5.2×** and **3.3–6.9×**.
Netflix ALS spans **2.640216550–3.453838314×** (2.6–3.5×).
The narrower ALS **2.752487776–4.811928815×** range describes only K96/F100.
These are single-run measurements including evaluation on the shared host.

<!-- evidence:external:start -->
| Dataset | APR K / ALS F | APR seconds | ALS seconds | ALS / APR | CCD seconds | CCD / APR | BALS seconds | BALS / APR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ML-10M | 16 / 20 | [0.035308674000000005](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L60) | [0.172658753](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L7) | 4.889981227 | [0.205167957](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L34) | 5.810695610 | [0.14615213](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L54) | 4.139269858 |
| ML-10M | 32 / 30 | [0.051487713](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L9) | [0.170530399](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L27) | 3.312060083 | [0.293085374](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L3) | 5.692336228 | [0.366434265](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L12) | 7.116926421 |
| ML-10M | 48 / 50 | [0.065774818](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L2) | [0.30370593](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L73) | 4.617358728 | [0.36031547](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L22) | 5.478015462 | [0.651230347](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L42) | 9.900906864 |
| ML-10M | 64 / 60 | [0.089796478](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L69) | [0.38600433500000003](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L41) | 4.298657849 | [0.480780899](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L53) | 5.354117552 | [1.2266804199999999](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L84) | 13.660674086 |
| ML-10M | 96 / 100 | [0.164550949](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L75) | [0.791807453](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L44) | 4.811928815 | [0.5797738530000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L80) | 3.523369853 | [2.5914406739999998](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L72) | 15.748561098 |
| ML-20M | 16 / 20 | [0.07550479900000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L71) | [0.392090355](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L64) | 5.192919658 | [0.487499101](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L32) | 6.456531339 | [0.34048998999999996](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L31) | 4.509514554 |
| ML-20M | 32 / 30 | [0.114374657](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L55) | [0.458805607](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L4) | 4.011427173 | [0.715350457](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L61) | 6.254448982 | [0.839857117](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L58) | 7.343035066 |
| ML-20M | 48 / 50 | [0.142126968](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L25) | [0.658603206](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L78) | 4.633907381 | [0.923777701](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L6) | 6.499665152 | [1.52288208](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L8) | 10.714941024 |
| ML-20M | 64 / 60 | [0.260481781](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L83) | [0.842021597](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L51) | 3.232554668 | [1.017631907](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L10) | 3.906729688 | [3.519918701](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L77) | 13.513109007 |
| ML-20M | 96 / 100 | [0.37359249899999997](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L5) | [1.2867164009999998](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L39) | 3.444170867 | [1.221924984](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L52) | 3.270742821 | [6.128088378999999](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L23) | 16.403135490 |
| ML-32M | 16 / 20 | [0.135244614](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L62) | [0.6460989070000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L40) | 4.777261644 | [0.927740471](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L28) | 6.859722125 | [0.7545122679999999](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L11) | 5.578871096 |
| ML-32M | 32 / 30 | [0.20378274500000002](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L48) | [0.755653958](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L43) | 3.708135142 | [1.081676293](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L17) | 5.307987646 | [1.32607373](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L57) | 6.507291528 |
| ML-32M | 48 / 50 | [0.26854641700000004](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L82) | [1.0893901659999998](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L18) | 4.056617765 | [1.336894712](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L45) | 4.978263076 | [2.920302979](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L20) | 10.874481260 |
| ML-32M | 64 / 60 | [0.369735413](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L19) | [1.3969423280000002](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L74) | 3.778221612 | [1.798533996](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L37) | 4.864381211 | [5.324620117](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L26) | 14.401163453 |
| ML-32M | 96 / 100 | [0.7817379760000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L66) | [2.151724223](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L63) | 2.752487776 | [2.718504512](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L30) | 3.477513688 | [12.029125](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L70) | 15.387668719 |
| Netflix | 16 / 20 | [0.556185913](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L16) | [1.9209762160000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L65) | 3.453838314 | [2.794697872](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L49) | 5.024754865 | [2.0757341310000004](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L14) | 3.732086848 |
| Netflix | 32 / 30 | [0.75070105](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L50) | [2.333357915](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L46) | 3.108238513 | [3.8179715420000004](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L47) | 5.085874786 | [3.674747559](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L15) | 4.895087810 |
| Netflix | 48 / 50 | [0.961992249](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L21) | [3.227218075](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L33) | 3.354723573 | [4.943782554](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L81) | 5.139108511 | [8.015517578](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L67) | 8.332205988 |
| Netflix | 64 / 60 | [1.549030273](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L68) | [4.089775364](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L24) | 2.640216550 | [5.472837535](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L85) | 3.533073324 | [14.495842773](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L76) | 9.358011283 |
| Netflix | 96 / 100 | [1.930820068](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L36) | [6.104561985](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L79) | 3.161642085 | [8.153676788](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L29) | 4.222908661 | [30.248320312](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L38) | 15.666048232 |
<!-- evidence:external:end -->

All external CSV rows now have corresponding raw trajectories. The
[raw coverage ledger](#raw-coverage-ledger) records the restored coverage.

## Dataset means

Unweighted arithmetic means of each dataset's five ratios; not ratios of summed
times, and not independent repetitions.

<!-- evidence:means:start -->
| Dataset | Mean ALS / APR | Mean CCD / APR | Mean BALS / APR |
|---|---:|---:|---:|
| ML-10M | 4.385997340 | 5.171706941 | 10.113267665 |
| ML-20M | 4.102995949 | 5.277623596 | 10.496747028 |
| ML-32M | 3.814544788 | 5.097573549 | 10.549895211 |
| Netflix | 3.143731807 | 4.601144029 | 8.396688032 |
<!-- evidence:means:end -->

## Final validation accuracy

Signed difference = `100 * (APR_RMSE - baseline_RMSE) / baseline_RMSE`.
Negative means APR has lower RMSE. Maximum absolute differences are
**0.093374299% vs ALS** and **0.183354362% vs CCD**, only at APR K96.
APR's ML-10M RMSE is **0.034920325% higher** than CCD's.
Netflix APR/CCD round to **0.8205/0.8216**; ML-32M to **0.7611/0.7625**.
No “below run-to-run noise” conclusion follows from these single runs.

<!-- evidence:accuracy:start -->
| Dataset | APR K=96 RMSE | ALS F=100 RMSE | APR vs ALS (%) | CCD K=96 RMSE | APR vs CCD (%) |
|---|---:|---:|---:|---:|---:|
| ML-10M | [0.787782](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L75) | [0.787056](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L44) | +0.092242483 | [0.787507](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L80) | +0.034920325 |
| ML-20M | [0.775026](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L5) | [0.774303](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L39) | +0.093374299 | [0.775411](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L52) | -0.049651088 |
| ML-32M | [0.76106](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L66) | [0.760744](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L63) | +0.041538284 | [0.762458](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L30) | -0.183354362 |
| Netflix | [0.820529](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L36) | [0.820238](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L79) | +0.035477508 | [0.821569](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L29) | -0.126587055 |
<!-- evidence:accuracy:end -->

## SGD context

An earlier target crossing does not imply lower final RMSE. SGD uses per-epoch
synchronization and evaluation; its timing differs from historical persistent-kernel
measurements. Near-one ratios do not establish a reliable ordering from one run.

<!-- evidence:sgd:start -->
| Dataset | APR K=96 seconds | SGD K=128 seconds | SGD / APR | SGD final validation RMSE |
|---|---:|---:|---:|---:|
| ML-10M | [0.164550949](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L75) | [0.19641745](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L13) | 1.193657352 | [0.784436868](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L13) |
| ML-20M | [0.37359249899999997](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L5) | [0.33266765600000003](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L59) | 0.890455930 | [0.774513458](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L59) |
| ML-32M | [0.7817379760000001](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L66) | [0.516228754](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L35) | 0.660360338 | [0.763636337](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L35) |
| Netflix | [1.930820068](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L36) | [1.917665704](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L56) | 0.993187162 | [0.825522986](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L56) |
<!-- evidence:sgd:end -->

## CCD regularization

These use CCD's separate regularization convention, not APR lambdas. The checker
recomputes the minimum among candidates in each saved selection record.

<!-- evidence:lambdas:start -->
| Dataset | Selected CCD lambda |
|---|---:|
| ML-10M | [0.058](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external/ml10m_ccd_selection.json?plain=1#L2) |
| ML-20M | [0.05](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external/ml20m_ccd_selection.json?plain=1#L2) |
| ML-32M | [0.05](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external/ml32m_ccd_selection.json?plain=1#L2) |
| Netflix | [0.04](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external/netflix_ccd_selection.json?plain=1#L2) |
<!-- evidence:lambdas:end -->

## Momentum

There are **twenty complete pairs out of twenty planned pairs**. Find the first sweep
where `abs(train_rmse[t] - train_rmse[t-1]) < 0.001`. The wall ratio at that crossing
is **1.095113075–1.359098303×**, rounded to **1.10–1.36×**.
The separate five-sweep-window rule compares `t` with `t-5`; it is not a five-sweep
sampling schedule. All forty raw curves are available and both rules are reparsed.

<!-- evidence:momentum:start -->
| Dataset | K | Beta 0 crossing sweep | Beta 0.3 crossing sweep | Wall ratio | Five-sweep window crossings (0 / 0.3) |
|---|---:|---:|---:|---:|---:|
| ML-10M | 16 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L2) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L3) | 1.215486528 | 21 / 14 |
| ML-10M | 32 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L4) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L5) | 1.241474463 | 22 / 15 |
| ML-10M | 48 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L6) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L7) | 1.258962223 | 23 / 15 |
| ML-10M | 64 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L8) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L9) | 1.359098303 | 23 / 15 |
| ML-10M | 96 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L10) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L11) | 1.233513007 | 23 / 15 |
| ML-20M | 16 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L12) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L13) | 1.199040765 | 21 / 14 |
| ML-20M | 32 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L14) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L15) | 1.200073553 | 23 / 15 |
| ML-20M | 48 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L16) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L17) | 1.200596390 | 23 / 15 |
| ML-20M | 64 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L18) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L19) | 1.221509926 | 22 / 15 |
| ML-20M | 96 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L20) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L21) | 1.221716530 | 23 / 14 |
| ML-32M | 16 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L22) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L23) | 1.095113075 | 22 / 14 |
| ML-32M | 32 | [12](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L24) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L25) | 1.199985363 | 23 / 15 |
| ML-32M | 48 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L26) | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L27) | 1.100703104 | 22 / 15 |
| ML-32M | 64 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L28) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L29) | 1.220028701 | 22 / 15 |
| ML-32M | 96 | [11](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L30) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L31) | 1.221953218 | 22 / 14 |
| Netflix | 16 | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L32) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L33) | 1.111450780 | 19 / 14 |
| Netflix | 32 | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L34) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L35) | 1.111101806 | 19 / 14 |
| Netflix | 48 | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L36) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L37) | 1.110809764 | 19 / 14 |
| Netflix | 64 | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L38) | [9](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L39) | 1.110535303 | 19 / 14 |
| Netflix | 96 | [10](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L40) | [8](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L41) | 1.249976429 | 19 / 13 |
<!-- evidence:momentum:end -->

## Coverage

Planned totals come from archived runner loops, not completion markers. Available
means stored CSV records. The manifest holds exact key sets; counts alone do not
establish a complete matched grid. No campaign completion percentage is asserted.

<!-- evidence:coverage:start -->
| Evidence table | Available records | Planned records | Protocol | Verification scope |
|---|---:|---:|---|---|
| [accumulation_control.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/accumulation_control.csv?plain=1#L1) | 20 | 20 | N80 | Inventory and fingerprints |
| [accumulation_errors.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/accumulation_errors.csv?plain=1#L1) | 4,000 | Not a run total | N80 | Inventory and fingerprints |
| [bf16.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/bf16.csv?plain=1#L1) | 8 | 8 | N80 | Available raw final summaries |
| [bf16_controls.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/bf16_controls.csv?plain=1#L1) | 12 | 12 | N80 | Available raw final summaries |
| [bf16_matched_comparison.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/bf16_matched_comparison.csv?plain=1#L1) | 8 | Not a run total | N80 | Inventory and fingerprints |
| [cumulative.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/cumulative.csv?plain=1#L1) | 6 | 6 | C80 | Available raw final summaries |
| [external_time_to_target.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_time_to_target.csv?plain=1#L1) | 84 | 84 | E70 | CSV calculations; 84 raw trajectories |
| [four_way.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/four_way.csv?plain=1#L1) | 80 | 80 | N80 | Available raw final summaries |
| [gram_errors.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/gram_errors.csv?plain=1#L1) | 3,200 | Not a run total | N80 | Inventory and fingerprints |
| [heldout_test.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/heldout_test.csv?plain=1#L1) | 20 | 20 | O70 | Available raw final summaries |
| [momentum_crossings.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/momentum_crossings.csv?plain=1#L1) | 40 | 40 | M80 | 40 raw curves; twenty complete pairs |
| [rounding_control.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/rounding_control.csv?plain=1#L1) | 20 | 20 | N80 | Inventory and fingerprints |
| [validation_accuracy_deltas.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/validation_accuracy_deltas.csv?plain=1#L1) | 20 | 20 | O70 | Inventory and fingerprints |
| [validation_accuracy_pair.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/validation_accuracy_pair.csv?plain=1#L1) | 40 | 40 | O70 | Available raw final summaries |
| [validation_selection.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/validation_selection.csv?plain=1#L1) | 4 | 4 | O70 | Inventory and fingerprints |
| [validation_sweep.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/validation_sweep.csv?plain=1#L1) | 32 | 32 | O70 | Available raw final summaries |
| [external_k96_comparison.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/external_k96_comparison.csv?plain=1#L1) | 20 | Not a run total | E70 | Inventory and fingerprints; no new aggregate claim |
| [fp32_fallback_runs.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/fp32_fallback_runs.csv?plain=1#L1) | 24 | Not a run total | F70 | Inventory and fingerprints; no new aggregate claim |
| [fp32_fallback_legacy_runs.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/fp32_fallback_legacy_runs.csv?plain=1#L1) | 15 | Not a run total | F80 | Inventory and fingerprints; no new aggregate claim |
| [fp32_fallback_summary.csv](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/part2/new_results/fp32_fallback_summary.csv?plain=1#L1) | 13 | Not a run total | FMIX | Inventory; exact archived undefined NaN cells declared |
<!-- evidence:coverage:end -->

There are 84 external CSV records and 84 available raw trajectories. The 25 logs
missing from the earlier copy are restored. Each curve's sweep sequence, first target
crossing and final validation RMSE are checked against its CSV record.

## Raw coverage ledger

The manifest-derived ledger records external raw coverage. Other run tables have
their available/missing final-summary paths under `run_raw` in the manifest.
Unexpected disappearance is an error; a declared gap is an explicit warning.

<!-- evidence:gaps:start -->
All 84 external raw trajectories are present and numerically checked against their CSV records. No external raw-log gaps remain.
<!-- evidence:gaps:end -->

## Historical evidence

Older results remain useful within their original protocol.
The [August variance summary](../results/PAPER_MEASUREMENTS_20260803/variance_5x_summary.txt)
reports 20 configurations × five repetitions = 100 runs, with maximum relative
wall-speedup standard deviation 0.44% and mean 0.14%. The checker verifies its
fingerprint and completion of all 100 raw logs. It does not re-estimate variance from
rounded console values. These are shared-binary controls, weighted lambda 0.048,
cuMF-scale initialization, FP32 solvers in both legs, original 80/20 data, RTX 3060.

Historical T4 Netflix K96: **196.508 / 18.236 seconds = 10.78×**, rounded.
The checker reparses both times from the
[source log](https://github.com/hafizr283/apr-bals/blob/437e10e55c066e50c6077043313eec2fdaf46247/results/t4_latest_res_22_7_2026/T4_MAIN_WL_20260721_210336/netflix_k96.txt?plain=1#L42-L60).
This uses FP32 solvers, no momentum, and the original stopping rule. It does not
measure the October external recipe on T4.

The [build recipes](reproduction.md#build-recipes) preserve checked historical
distinctions: weighted regularization and initialization matter, and standalone APR
can enable momentum and a rank-specific solver. The shared-binary harness tests a
different comparison. Legacy LHS/RHS events can overlap solver work, so they are
not isolated kernel times.

## Cumulative stage definitions

The six recorded configurations use the October source tree, Netflix K96, lambda
0.048, weighted regularization and fixed initialization. Adjacent training-RMSE
change below 0.001 defines convergence, with a 30-sweep cap. These are convergence
times rather than fixed-sweep timings; the [CSV](../part2/new_results/cumulative.csv)
records each stopping count and raw final summary.

| Stage | Configuration and interpretation |
|---|---|
| `ladder_atomic` | Atomic FP32 Gram, FP32 solver, exact evaluation, no momentum. |
| `ladder_sameflow` | Staged FP32 warp/entity FMA; no dense Gram atomics, giant tier disabled, FP32 solver. |
| `ladder_fp32` | Dense FP16 tensor path **and** giant tier enabled, FP32 solver. This transition combines changes. |
| `ladder_fp16solve` | Previous configuration with FP16 solver tiles and no refinement. |
| `ladder_fastcheck` | Previous configuration with mixed-precision rating evaluation; the convergence measurement changes. |
| `ladder_momentum` | Previous configuration with beta 0.3 momentum. |

Wall time includes each stage's evaluation cost. Lambda fusion and asynchronous
staging are not separately switched, and monotonic speedup was not a condition for
retaining results. Use four-way controls for the precision/dataflow comparison;
do not interpret every ladder transition as a single isolated optimization.

## Numerical interpretation

The [FP64 probes](../part2/new_results/gram_errors.csv) compare actual atomic FP32,
same-dataflow FP32, dense FP16 and giant FP16 kernels on identical trained factors.
They retain the selected entities' original 32-row tile neighborhoods so atomic job
grouping is preserved. Only unregularized upper triangles consumed by the solver
are compared. Independently trained factor matrices would confound this comparison.

The scalar kernel accumulates segments in registers and atomically flushes per-job
sums; it does not atomically add each rating separately. For a unique user-item
matrix, a conservative componentwise absolute bound for this hierarchy is
`[(1+gamma_YB)(1+gamma_8)(1+gamma_J)-1] * abs(F).T @ abs(F)`, where
`gamma_n = n*u/(1-n*u)`, `u = 2^-24`, and `J` counts contributing jobs.
Cancellation can make relative entrywise error large, so the probes report
upper-triangle Frobenius error and normalized maximum absolute error. The measured
atomic FP32 path has lower Gram error than the tensor paths; the accuracy shift is
not evidence that atomic FP32 builds less accurate Grams.

[Chunk-reset controls](../part2/new_results/accumulation_control.csv) zero each
16-row MMA fragment, then combine chunks with FP32 scalar additions. Rounded operands,
tile order, RHS, solver and budget are retained. Together with the
[rounded-FMA controls](../part2/new_results/rounding_control.csv), these separate
operand quantization from persistent tensor accumulation. The tested bias and RMSE
shift are properties of this implementation/hardware, not universal regularization
or a portable tensor rounding guarantee. The diagnostics remain inventory/fingerprint
checks in the CPU checker; their binary factor inputs are held externally.

Ampere TF32 and FP16 have ten stored fraction bits; TF32 has the wider FP32 exponent
range. FP32 has 23 stored fraction bits. See the
[NVIDIA TF32 description](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
and [PTX specification](https://docs.nvidia.com/cuda/archive/11.5.1/parallel-thread-execution/index.html).
An FP32 accumulator type alone does not guarantee scalar IEEE-FMA accumulation order.

K96 same-dataflow tuning selected a zero-spill outer-contraction control using timing,
not accuracy. [Tuning records](../part2/new_results/control_tuning_runs/),
[initial results](../part2/new_results/control_tuning_initial/) and
[SASS audit](../part2/new_results/sameflow_sass_audit.json) preserve the choice.
Staging width and prefetch differ across paths, so this remains a structural control.

[BF16 controls](../part2/new_results/bf16_matched_comparison.csv) change both Gram
operand and solver types; they do not isolate which change prevents a failure.
Retain zeroed-solve counts and failure cases. GPU0 accuracy diagnostics cannot be
compared with GPU1 timings as a performance experiment. The separate
[FP32 fallback protocol](reproduction.md#fp32-fallback-protocol) documents retries
and the tested limits of the safety claim. Original-split and new-split absolute
RMSEs remain distinct throughout these controls.

Legacy LHS/RHS events can join overlapping solve streams, so `gram_s` in the older
summaries means a compute phase, not isolated Gram-kernel time. Per-sweep cumulative
wall measurements are used for target crossings instead of mean-sweep multiplication.

## Historical implementation context

The [full-rank BALS ladder](../results/PAPER_MEASUREMENTS_20260803/bals_ladder_all_K.txt)
retains both beneficial and adverse symmetric-tiling results. A rank-specific gain
must not be generalized to the full grid. Register limits, padding and occupancy are
separate contributors; see [occupancy records](../results/PAPER_MEASUREMENTS_20260803/occupancy/)
and [precision controls](../results/PAPER_MEASUREMENTS_20260803/precision_isolation_3way/).

[Threshold sweeps](../results/PAPER_MEASUREMENTS_20260803/threshold_sweep/),
[host-cost studies](../results/PAPER_MEASUREMENTS_20260803/host_factor_per_rank/),
[kernel shares](../results/PAPER_MEASUREMENTS_20260803/nsys_kernel_shares.txt) and
[hardware counters](../results/PAPER_MEASUREMENTS_20260803/ncu/) retain their measured
conditions. Failed profiler attempts are preserved as diagnostics; use successful
counter captures for hardware conclusions. These historical studies do not change
the October masks, stopping rules or timing-variance limitations.

## Retention and maintenance

The entry page and October index link the maintained scientific guides. Experiment
protocols and interpretation are consolidated here and in the reproduction guide.
Figures and manuscript snapshots retain their original provenance and may predate
final measurements; use the CSVs and raw logs for numerical claims.

The small metadata/diagnostic evidence remains indexed. See the
[artifact guide](artifacts.md) for archive identity and
binary storage. Datasets, factors, models, executables and caches remain excluded
from Git. Diagnostic failure messages and 22 explicitly undefined fallback-summary
NaN cells are preserved; these are not finite performance results or success claims.

At [restoration commit `437e10e`](https://github.com/hafizr283/apr-bals/commit/437e10e55c066e50c6077043313eec2fdaf46247),
`src/wmma_kernels.cuh` matched the final [archived source inventory](../part2/new_results/artifact_sha256.json),
and differed from the older root header. Later comment/docstring cleanup preserves
executable code and source line counts; only unsupported-rank diagnostic references
also changed. Current edited source bytes therefore differ from historical hashes.
The archived hashes remain unchanged; the [evidence manifest](evidence_manifest.json)
fingerprints the maintained copies. No source variant was substituted. Fallback
and K96 summary tables are indexed as context, without new headline claims.

The checker is offline and read-only. It validates relative links, Markdown headings,
pinned line ranges and their curated text, table keys, raw comparisons and normalized
fingerprints. It does not test remote availability, execute CUDA, verify excluded
binary payloads, or make all historical narrative authoritative.

For intentional changes, review the protocol/source, update the manifest and calculated
blocks together, and run the checker and `python -m unittest discover -s tests -p 'test_evidence.py'`.
Do not update hashes merely to silence a mismatch. The manifest is a review baseline,
not independent proof of experiment reproduction.

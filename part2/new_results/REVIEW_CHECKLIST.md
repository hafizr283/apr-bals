# Review-to-evidence map

Read `SUMMARY.md` for measured values and current counts. Raw experiment logs have adjacent JSON command/environment records. Do not treat queued experiments as completed evidence.

| Review request | New evidence | Interpretation / remaining work |
|---|---|---|
| FP32 CUDA-core control with WMMA dataflow | `four_way.csv`, `four_way/`, `sameflow_sass_audit.json`, `control_tuning_runs/` | Four variants, four datasets, five ranks. Same warp/entity and atomic-free upper-tile dataflow; FP32 staging differs in width and prefetch from FP16. K96 loop tuning is complete; final control uses the zero-spill outer contraction loop. Integrate this control into the paper's primary isolation harness. |
| Validation-selected lambda | `validation_selection.csv`, `validation_sweep.csv`, `heldout_test.csv`, `validation_accuracy_pair.csv`, `splits/verification.json` | New approximately 70/10/20 split, old test unchanged. Select on validation only. Historical test exposure remains a limitation. Guard failures at low lambda are recorded. |
| FP64 reference for 20 heaviest users/items | `gram_errors.csv`, `gram/`, `NUMERICAL_AUDIT.md` | Actual four CUDA paths on identical factors. Assess the reviewer's atomic-error hypothesis from measurements, not the loose gamma-degree estimate. |
| Explain lower train/test RMSE | `rounding_control.csv`, `rounding_control/` plus FP64 checks | Rounded FP32 FMA control isolates operand quantization from tensor instructions and atomics. `accumulation_control.csv` and `accumulation_errors.csv` test resetting MMA accumulation every 16 rows. A small improvement on these datasets does not establish universal regularization or greater Gram accuracy. |
| Momentum every sweep at every rank | `momentum_crossings.csv`, `momentum/`, `figures/momentum_k96.pdf` | Both adjacent-sweep and five-sweep-window rules reported; do not mix them. Exact FP32 rating evaluation, fixed 30-sweep curves. |
| Common held-out time-to-target across systems | `external_time_to_target.csv`, `EXTERNAL_RESULTS.md`, `figures/external_curves_k96.pdf`, `EXTERNAL_PROTOCOL.md` | Common warm-validation rows; explicit rank pairing; each method's evaluation overhead included. SGD epoch barriers added for measurement. Unreached targets remain censored. |
| Same-source cumulative ablation | `cumulative.csv`, `cumulative/` | Same source tree, compile-time variants. Do not force monotonicity. Lambda fusion and asynchronous staging are not separately switched in this implementation. |
| BF16 operands and solver; beta .5 | `bf16.csv`, `bf16/`, `bf16_src/` | Retain negative results and zeroed-solve counts. Lower-precision fallback paths must be distinguished from BF16 dense/giant paths. |
| Long-scoreboard/LG-throttle; giant dominant pipe | `ncu/dense_stalls.csv`, `ncu/giant_stalls.csv` when produced | Administrator profiling required by driver policy; user agreed to launch `run_ncu_stalls.sh`. Counter filenames alone do not establish a successful profile. `ncu/permission_check.txt` confirms ERR_NVGPUCTRPERM. |
| cuMF implementation and configuration details | `EXTERNAL_PROTOCOL.md`, `external_src/cumf_als/` | FP16 switches off, six CG steps, batch 1/3; no new batching search. Includes CUDA-12 port and cold-entity guard. |
| FP32 fallback after FP16 solver failure | `artifact_audit.json`, `validation_sweep.csv` | Existing guard zeroes failed solutions. FP32 retry is not implemented; required engineering remains. |
| Optional K20/50/100 padding and larger dataset | No new result | Not implemented/collected. Current datasets already exercise batches beyond 60,000 entities. |
| Restructure, wording, template, author names, repository/DOI | Original manuscript preserved in `manuscript/`; generated tables in `paper_tables/` | Manuscript revision and publication remain separate work after measurements. Avoid carrying over old absolute RMSE or self-baseline headline claims. |

## Protocol changes from the supplied manuscript

- Add the missing FP32 structural control and separate dataflow effects from precision effects.
- Replace test-selected lambda with validation selection; report test once per selected configuration.
- Replace five-sweep timing resolution with cumulative time at every sweep.
- Use one common warm-validation set and fixed targets for the external comparison.
- Report compute-phase timers honestly: legacy LHS+RHS intervals include overlapping solver work.
- Correct the TF32 precision statement: TF32 and FP16 both have ten stored fraction bits; TF32 has wider exponent range.
- Retain numerical failures, censored targets, and non-monotone ablation outcomes.

New runs are single measurements. Historical variance estimates cannot be attached to these changed configurations. Repetitions are needed before sub-percent timing claims.

# FP32 fallback results

Median of three serial GPU-1 runs. Each run uses 30 fixed sweeps at Netflix K=96. Failure rates use 14,938,770 entity-solves per run.

| Protocol | Solver behavior | IR | FP16 failures | Failure rate | FP32 recovered | FP32 failed | Validation RMSE | Test RMSE | Wall (s) | Overhead vs zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| New split, stable | FP16 + FP32 fallback | 0 | 0 [0, 0] | 0.000000% | -- | 0 | 0.820529 | 0.819465 | 10.54967 | -0.087% |
| New split, beta=.5 | FP16 zero, no IR | 0 | 6 [6, 6] | 0.000040% | -- | 0 | 0.822635 | 0.821787 | 10.56239 | -- |
| New split, beta=.5 | FP16 fallback, no IR | 0 | 2 [2, 2] | 0.000013% | 2 (100.0%) | 0 | 0.822030 | 0.821171 | 10.55798 | -0.042% |
| New split, beta=.5 | FP16 zero, IR1 | 1 | 9 [9, 9] | 0.000060% | -- | 0 | 0.823091 | 0.822242 | 11.64633 | -- |
| New split, beta=.5 | FP16 fallback, IR1 | 1 | 2 [2, 2] | 0.000013% | 2 (100.0%) | 0 | 0.822022 | 0.821168 | 11.64616 | -0.001% |
| New split, beta=.5 | FP32 | -- | 0 [0, 0] | 0.000000% | -- | 0 | 0.822022 | 0.821168 | 12.35772 | -- |
| Original split, beta=.5 | FP16 zero, no IR | 0 | 3417 [3406, 3463] | 0.022873% | -- | 0 | -- | 1.203685 | 11.09559 | -- |
| Original split, beta=.5 | FP16 fallback, no IR | 0 | 33 [33, 33] | 0.000221% | 33 (100.0%) | 0 | -- | 0.817964 | 11.08234 | -0.119% |
| Original split, beta=.5 | FP16 zero, IR1 | 1 | 3163 [3161, 3175] | 0.021173% | -- | 0 | -- | 1.263372 | 12.17853 | -- |
| Original split, beta=.5 | FP16 fallback, IR1 | 1 | 27 [26, 27] | 0.000181% | 27 (100.0%) | 0 | -- | 0.817956 | 12.17629 | -0.018% |
| Original split, beta=.5 | FP32 | -- | 0 [0, 0] | 0.000000% | -- | 0 | -- | 0.817956 | 12.88565 | -- |

Negative overhead estimates mean the fallback build was marginally faster in that three-run sample. They are treated as no measurable overhead, not as a speedup.

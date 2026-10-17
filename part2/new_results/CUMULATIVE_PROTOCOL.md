# Same-source cumulative ablation

All six configurations use the isolated October source tree, Netflix, K96, lambda .048, weighted regularization and fixed initialization. Convergence is checked every sweep using adjacent training-RMSE change below .001, with a cap of30 sweeps. These are convergence runs, not fixed30-sweep timings. Table values are in `cumulative.csv`.

| Configuration | Changes and interpretation |
|---|---|
| ladder_atomic | FP32 atomic Gram baseline; FP32 solver; exact rating evaluation; no momentum. |
| ladder_sameflow | FP32 warp/entity staged FMA dataflow; no Gram atomics on dense path; giant tier disabled; FP32 solver. |
| ladder_fp32 | Dense FP16 tensor path **and** giant tier enabled; FP32 solver. This transition does not isolate tensor instructions alone. Use four_way.csv for the controlled instruction/dataflow comparison. |
| ladder_fp16solve | Previous configuration with FP16 solver tiles, no refinement. |
| ladder_fastcheck | Previous configuration with fast mixed-precision rating evaluation. This changes the convergence measurement implementation; tiny RMSE differences must not be presented as a solver improvement. |
| ladder_momentum | Previous configuration plus beta .3 momentum. |

The first five runs stop at10 sweeps; momentum stops at8. Wall times include each configuration's evaluation cost. The observed sequence is monotone, but monotonicity was not imposed. Lambda-diagonal fusion and asynchronous staging remain enabled where the source uses them and are not separately isolated. Source provenance and flags are in `run_followups.py` and build logs.

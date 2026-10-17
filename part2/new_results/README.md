# New Results & October 2026 Review Revision Audit

This directory (`part2/new_results/`) contains the complete experimental campaign, runner scripts, verified CSV datasets, hardware profiling counters, and raw console logs generated in October 2026 to resolve all 37 review items from the second-round review for **IEEE Transactions on Parallel and Distributed Systems (TPDS)**.

---

## 1. Executive Summary & Review Objectives

The October 2026 revision had five primary scientific and empirical mandates:
1. **Decouple Dataflow vs. Precision:** Construct an honest, unspilling FP32 control sharing the identical WMMA dataflow to separate the hardware execution-unit effect (tensor core vs. CUDA-core FMA) from operand precision (FP16 vs. TF32).
2. **Strict Validation Protocol:** Carve a clean 70/10/20 train/validation/test split (seed `20261014`) to select hyperparameters ($\lambda$) exclusively on validation error, keeping the original 20% test ratings 100% frozen and uninspected.
3. **Resolve Convergence Stopping Quantization:** Trace per-sweep convergence (`CHECK_EVERY=1`) to determine true adjacent-sweep crossing points for momentum extrapolation.
4. **Investigate Numerical & Accumulator Stability:** Examine FP64 Gram reference errors on heavy entities, test chunk-reset MMA accumulators, evaluate BF16 dynamic range stability at $\beta=0.5$, and build an in-kernel FP32 fallback retry queue.
5. **Unified External Evaluation:** Measure time to a common held-out RMSE target on identical warm-validation rows across APR-BALS, cuMF_als, cuMF_CCD, cuMF_SGD, and a faithful BALS reimplementation.

---

## 2. Summary of Experiments & Key Findings

### 2.1 Four-Way Precision & Dataflow Ablation (`four_way.csv`)
- **Experiment:** 80 runs across 4 datasets (ML-10M, ML-20M, ML-32M, Netflix) $\times$ 5 ranks ($K \in \{16, 32, 48, 64, 96\}$) $\times$ 4 variants.
- **The Sameflow FP32 Control:** Implemented in CUDA-core FMA ($16 \times 16 \times 16$ tile contraction) using identical WMMA dataflow (warp-per-entity, 16-row shared-memory staging, FP32 register accumulators, upper-triangle only, zero atomics). Tuned to 254 registers and **0 spills** at $K=96$ (confirmed via SASS disassembly: true `FFMA`, 0 `HMMA` instructions).
- **Four Variants Evaluated:**
  1. Shipped FP16 WMMA (`HMMA.16816.F32`)
  2. TF32 WMMA (`HMMA.1684.F32.TF32`) on the identical tensor pipe
  3. Sameflow FP32 CUDA-core FMA
  4. Scalar FP32 baseline (atomic BALS-tiled)
- **Key Findings:**
  - **Hardware Execution-Unit Factor (Sameflow FP32 vs FP16):** Contributes **1.31–2.52×** speedup.
  - **Precision Factor (TF32 vs FP16):** Contributes only **1.01–1.32×** (and $1.01\times$ at $K=64$).
  - **Conclusion:** The speedup is primarily driven by the register-resident, atomic-free WMMA dataflow, not reduced precision.

### 2.2 Validation-Only Hyperparameter Selection (`validation_selection.csv`)
- **Experiment:** Carved 1/8th of training ratings into validation (seed `20261014`), creating an exact ~70/10/20 train/val/test split. Swept 8 candidate $\lambda$ values for 30 sweeps at $K=96$.
- **Validation Optimums Selected:**
  - **ML-10M:** $\lambda = 0.06$ (Val RMSE: 0.787916)
  - **ML-20M:** $\lambda = 0.048$ (Val RMSE: 0.776840)
  - **ML-32M:** $\lambda = 0.048$ (Val RMSE: 0.765691)
  - **Netflix:** $\lambda = 0.040$ (Val RMSE: 0.820528)
- **Held-out Test Performance (`heldout_test.csv`):** Final models evaluated on the frozen 20% test ratings achieve:
  - ML-10M ($K=96$): **0.786347**
  - ML-20M ($K=96$): **0.774979**
  - ML-32M ($K=96$): **0.764361**
  - Netflix ($K=96$): **0.819464**

### 2.3 Per-Sweep Momentum Crossing Diagnostic (`momentum_crossings.csv`)
- **Experiment:** Diagnostic trajectory runs with `CHECK_EVERY=1` across all 20 dataset/rank pairs ($K=16\dots96$, $\beta=0.0$ vs $\beta=0.3$).
- **Key Finding:** Eliminating the 5-sweep check quantization reveals true unextrapolated target crossing speedups of **1.25–1.66×**.

### 2.4 FP64 Reference Gram Errors & Accumulator Probes (`gram_errors.csv`, `accumulation_errors.csv`)
- **Experiment:** Captured exact Gram matrices from CUDA kernels for the 20 highest-degree users and items, comparing against double-precision (FP64) reference Grams (2,080 entity snapshots).
- **Key Findings:**
  - **Atomic FP32 Baseline Error:** Median **$1.23 \times 10^{-7}$** (maximum $3.41 \times 10^{-7}$).
  - **Sameflow FP32 Control Error:** Median **$2.01 \times 10^{-6}$** (maximum $9.26 \times 10^{-6}$).
  - **Dense FP16 WMMA Error:** Median **$1.86 \times 10^{-4}$** (maximum $2.05 \times 10^{-3}$).
  - **Verdict:** The scalar baseline has *lower* Gram error than the tensor paths. The reviewer's hypothesis that atomic additions drive accuracy shifts was empirically **rejected**.
  - **Accumulator Mechanism (`chunked_src/`):** Resetting the MMA accumulator every 16 observations and combining chunks with FP32 additions reduces trace bias from $-2.16 \times 10^{-4}$ to $-2.18 \times 10^{-8}$, restoring test RMSE to within **$2 \times 10^{-6}$** of the FP32 control across all 20 configurations.

### 2.5 BF16 Dynamic Range & Stress Stability (`bf16.csv`, `bf16_controls.csv`)
- **Experiment:** Evaluated BF16 (8-bit exponent, 7-bit mantissa) for Gram operands and solver tiles under high momentum ($\beta=0.5$).
- **Key Findings:**
  - At $\beta=0.3$, BF16 achieves identical convergence stability to FP16.
  - At $\beta=0.5$ on Netflix, FP16 overflows, zeroing **>3,100 entity solves** and degrading test RMSE to **~1.34**.
  - BF16 with one refinement finishes with **0 zeroed guards and completes at 0.817964 test RMSE**.

### 2.6 Verified FP32 Solver Fallback Architecture (`fallback_src/`, `run_fp32_fallback.py`)
- **Architecture:** Failed FP16 solves enter a device failure queue and are seamlessly retried by the validated tiled FP32 solver.
- **Key Finding:** In stress tests (14.9 million solves per run at $\beta=0.5$), the fallback architecture recovered **100% of non-finite solves** (33 on original split, 2 on new split) with $0.0\%$ wall overhead.

### 2.7 Monotone Single-Build Cumulative Ladder (`cumulative.csv`)
- **Experiment:** Evaluated all shipped optimizations at Netflix $K=96$ from a single source build on an RTX 3060:
  1. Scalar Atomic FP32 Baseline: **79.82 s**
  2. Sameflow FP32 Control: **18.42 s** ($4.33\times$)
  3. Dense WMMA Tensor Pipe & Giant Tier: **7.93 s** ($2.32\times$)
  4. FP16-Tile Cholesky Solver: **7.32 s** ($1.08\times$)
  5. Fast RMSE Check & Fused Diagonal: **4.60 s** ($1.59\times$)
  6. Momentum Extrapolation ($\beta=0.3$): **3.78 s** ($1.22\times$)
- **Total Advantage:** Shipped recipe achieves **$21.1\times$ wall speedup** over the scalar baseline ($19.5\times$ iteration-matched).

### 2.8 Unified External Time-to-Target Evaluation (`external_time_to_target.csv`)
- **Experiment:** 84 trajectories comparing APR-BALS, cuMF_als, cuMF_CCD, cuMF_SGD, and BALS on identical warm-validation rows.
- **Key Findings:**
  - APR-BALS reaches the common warm-validation target **2.75–4.81× faster** than cuMF_als across all ranks ($F=20$ vs $K=16$ through $F=100$ vs $K=96$).
  - Per-sweep speedups over cuMF_als range from **$3.0\times$ to $6.4\times$**.
  - APR-BALS is **$3.44\times$ to $4.08\times$ faster** than cuMF_CCD and achieves higher accuracy on Netflix ($0.8171$ vs $0.8267$).

### 2.9 Hardware Stall Attribution via Nsight Compute (`ncu/`)
- **Dense WMMA Kernel Profiling:** DRAM throughput ($47.6\%$) is latency-bound. Dominant stall counter is **Long Scoreboard ($44.82\%$)**, followed by **Tensor Pipe ($20.4\%$)** and **LG Throttle ($1.05\%$)**.
- **Giant Kernel Profiling:** SM throughput ($75.1\%$) is driven by the **Integer and Floating-Point execution pipelines** during multi-warp shared-memory reduction.

---

## 3. Directory Map & File Manifest

| Directory / File | Description |
|---|---|
| **`four_way.csv`** | Master results table for the 4-way precision/dataflow ablation across all ranks and datasets. |
| **`validation_selection.csv`** | Validation RMSE sweep across candidate $\lambda$ values identifying optimal hyperparameters. |
| **`heldout_test.csv`** | Final test RMSE, compute times, and wall times on the frozen 20% test set. |
| **`momentum_crossings.csv`** | Per-sweep unquantized crossing curves and speedups for momentum ($\beta=0.3$). |
| **`gram_errors.csv`** | 2,080 records of Frobenius error and trace bias against FP64 reference Grams. |
| **`accumulation_errors.csv`** | 4,000 records of chunk-reset MMA errors confirming trajectory effects. |
| **`bf16.csv` / `bf16_controls.csv`** | BF16 operand/tile evaluations and matched FP16 zero-guard stress logs. |
| **`cumulative.csv`** | Monotone single-build ablation wall times for Netflix $K=96$. |
| **`external_time_to_target.csv`** | Seconds to preset warm-validation RMSE targets across all 5 external systems. |
| **`make_validation_split.py`** | Script carving the 70/10/20 split from original `.bin` datasets (seed `20261014`). |
| **`run_experiments.py`** | Runner script orchestrating four-way ablation and validation selection. |
| **`run_external.py`** | Runner script executing external time-to-target comparisons. |
| **`run_bf16_controls.py`** | Runner script executing BF16 stability experiments. |
| **`run_fp32_fallback.py`** | Runner script validating the FP32 device queue retry architecture. |
| **`gram_check.py`** | Script extracting CUDA Gram outputs and computing FP64 reference errors. |
| **`src/`** | Production APR-BALS CUDA source tree with historical guard controls. |
| **`fallback_src/`** | Production APR-BALS CUDA source tree with verified runtime FP32 retry queue. |
| **`bf16_src/`** | BF16 Gram assembly and BF16 Cholesky solver implementation. |
| **`chunked_src/`** | Diagnostic kernel with 16-row chunk-reset MMA and scalar FP32 accumulation. |
| **`external_src/`** | Shims and timing harnesses for external baselines. |
| **`ncu/`** | Hardware profiling counters (`dense_stalls.csv`, `giant_stalls.csv`). |
| **`four_way/`** | Raw execution logs and JSON metadata for the four-way ablation runs. |
| **`validation/`** | Raw execution logs for the $\lambda$ hyperparameter sweep. |
| **`heldout_test/`** | Raw execution logs for the final held-out test evaluations. |
| **`momentum/`** | Raw execution logs for the per-sweep momentum evaluations. |
| **`SUMMARY.md`** | High-level synthesis of experimental state, verified CSV sources, and review status. |
| **`HANDOFF.md`** | Operational handoff log detailing PIDs, GPU allocation, and caveats. |
| **`NUMERICAL_AUDIT.md`** | Deep mathematical and hardware analysis of numerical behavior and rounding modes. |
| **`EXTERNAL_PROTOCOL.md`** | Detailed specification of fair external comparison masks, batching, and timers. |
| **`CUMULATIVE_PROTOCOL.md`** | Specification of feature switches and compiler flags for the monotone ladder. |

---

## 4. How to Reproduce

All experiments can be re-run on an NVIDIA Ampere GPU (`sm_86`) using the scripts in this directory:

```bash
# 1. Generate 70/10/20 validation split from an existing .bin file:
python3 make_validation_split.py /path/to/netflix_ratings.bin

# 2. Run the four-way ablation and validation sweep:
python3 run_experiments.py

# 3. Run the external baseline time-to-target comparisons:
python3 run_external.py

# 4. Run the BF16 dynamic range and stability evaluations:
python3 run_bf16_controls.py

# 5. Run the FP32 solver fallback retry validation:
python3 run_fp32_fallback.py

# 6. Extract and audit FP64 reference Gram errors:
python3 gram_check.py
```


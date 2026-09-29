# Paper-completion measurements — 2026-08-03/04 (GPU1, RTX 3060, sm_86)

Answers the six items in `thesis_latex/TPDS___Hafizur/some_measurements_needed.txt`
(the reviewer's "you need these before submitting" list).

Box: nvcc 12.9, driver 580.173.02, 2× RTX 3060 12 GB. All runs `CUDA_VISIBLE_DEVICES=1`
(GPU0 drives the display), GPU otherwise idle, one session. Every variant is a
compile-flag build of the existing `comparing_baseline/` sources, with **one approved
exception**: the TF32 Gram kernel added for item 2 on 2026-08-04 (user-approved), which
lives entirely behind `-DWMMA_TF32=1`, default OFF, and is verified SASS-identical with
the flag off. No published number changes as a result.

| # | Item | Status | Artifact |
|---|------|--------|----------|
| 1 | Repeat runs for variance | **DONE** | `variance_5x/` + `variance_5x_summary.txt` |
| 2 | Precision-isolating ablation | **DONE — all 3 rungs, 3 datasets + full K curve** (tensor pipe = 87.8–99.6% of the effect; FP16 adds ≤1.32×, and nothing at all at K=64) | `precision_isolation_3way/`, `precision_isolation/` |
| 3 | Nsight counters | **DONE** — regs/thread, nsys shares, theoretical occupancy (all 5 ranks), and **hardware counters collected 2026-08-04: tensor pipe 20.5%, achieved occupancy 15.96% (96% of predicted), DRAM 47.6% ⇒ staging-bound, not tensor-bound** | `ncu/`, `occupancy/`, `ptxas_static_regs.txt`, `nsys_kernel_shares.txt` |
| 4 | BALS ladder at every rank | **DONE — contradicts a published claim**; also showed the paper's flat 1.22× host-level divisor is rank-dependent (1.00× at K=16), and re-derived the symtile mechanism (occupancy loss 0/0/25/37.5/37.5% confirmed, but padding waste 3.20×→1.07× is the other half) | `bals_ladder/` + `bals_ladder_all_K.txt`, `host_factor_per_rank/`, `occupancy/symtile_occupancy.txt` |
| 5 | λ=0.048 provenance | **DONE** | `lambda_sweep_k96.txt` |
| 6 | cuMF time-to-target-RMSE | **DONE** | `time_to_target_rmse.txt` |
| + | θ_g / θ_d threshold provenance (not on the list; 2 paper claims were unbacked) | **DONE — both now backed** | `threshold_sweep/` |

**Four corrections to earlier drafts of this file are recorded inline**, all caught before
they reached the paper: the item-2 Gram numbers were item-side-only (49.7× → **18.29×**
whole-stage, and the Amdahl paragraph built on them is withdrawn); the item-3 nsys
shares needed a wall-clock interval union because 3 CUDA streams overlap; the item-2
claim that TF32's doubled shared memory cost occupancy was **wrong** — both legs are
register-limited to the same 2 blocks/SM at K=96 (see the occupancy section under Item 3);
and the item-4 symmetric-tiling mechanism was **half a mechanism** — the 25–37% occupancy
loss is confirmed but does not explain K=16, where the penalty is zero and symtile still
loses (padding waste, 3.20×). The last two are the same mistake made twice: inferring
occupancy from block or buffer size without checking what actually binds.

---

## Item 1 — Repeat runs for variance (5 runs × 20 configs = 100 runs)

Protocol identical to the Table VI archives (`MAIN3060_WL_20260721_225757` +
`MAIN3060_WL_ML32M_20260722_061135`): `main_experiment.cu` two-leg harness,
weighted-λ 0.048, `CUMF_INIT=1`, FP32 solver on **both** legs, GT = 512 (MovieLens)
/ 1024 (Netflix), converge on train-RMSE Δ<1e-3.

**Run-to-run variation is far smaller than the paper currently claims.**

- Wall-speedup relative std dev: **max 0.44%, mean 0.14%** (n=20 configs × 5 runs).
- Baseline wall std dev ≤0.57%, APR wall std dev ≤0.33%.
- Test RMSE spread across 5 runs: **≤1e-6** (last printed digit) — the APR atomicAdd
  wobble documented in FIXLOG 07-06, nothing more.
- Largest std dev sits at K=16 (shortest runs, ~0.5–1.2 s, so fixed overheads dominate);
  at K≥48 every config is under 0.30%.

**Every one of the 20 published Table VI speedups reproduces within 0.50%**
(max |Δ| = 0.50% at ml20m K=48; 14 of 20 within 0.20%).

→ The paper's "±2% margin from two data points" can be replaced with a measured
**±1% (5-run, max observed 0.44%)**. Recommend quoting mean ± std in Table VI, or
stating the max observed relative std dev once in the protocol subsection.

## Item 5 — λ=0.048 provenance

Sweep over the weighted-λ (ALS-WR) coefficient, K=96, ship recipe
(`-DCHOL_MP=2 -DMOMENTUM=1`, `CUMF_INIT=1`), λ passed as argv[2] (no rebuild needed).
Netflix and ML-20M, 8 values from 0.010 to 0.120.

| λ | Netflix test RMSE | Netflix it | ML-20M test RMSE | ML-20M it |
|------|------|----|------|----|
| 0.010 | 0.879263 | 25 | 0.862641 | 30 |
| 0.020 | 0.833756 | 20 | 0.802707 | 25 |
| 0.030 | 0.817387 | 20 | 0.778736 | 20 |
| **0.040** | **0.814249** | 15 | 0.769808 | 20 |
| 0.048 (shipped) | 0.817146 | 15 | **0.768034** | 15 |
| 0.060 | 0.826253 | 15 | 0.771788 | 15 |
| 0.080 | 0.845177 | 20 | 0.785898 | 20 |
| 0.120 | 0.876239 | 15 | 0.816973 | 15 |

Both datasets show a clean U-shape with the optimum in **λ ∈ [0.04, 0.048]**, which is
the provenance the reviewer asked for: 0.048 is the joint optimum across the two
largest datasets, and it is inside the λ≈0.02–0.06 band that FIXLOG 07-12 identified
when weighted-λ was introduced for cumf_als parity.

**Honest caveat that should go in the paper:** on Netflix alone, λ=0.040 is
*better* than the shipped 0.048 — test RMSE 0.814249 vs 0.817146, i.e. **0.35% lower**,
at the same 15 iterations. λ=0.048 is the better joint choice (it wins on ML-20M,
0.768034 vs 0.769808) and it is what every published number was measured with, but the
paper should not imply 0.048 is Netflix-optimal. Either state it as a single
protocol-wide constant chosen for cross-dataset consistency, or report the per-dataset
optimum. Do **not** re-tune to 0.040 for the paper without re-running every table.

## Item 6 — cuMF_als time-to-target-RMSE

The reviewer's concern: cuMF's CG solver is inexact, so per-iteration comparison
flatters APR-BALS. Measured from `results/logs_justapr_vs_cumf_20260721_045838`
(both systems log test RMSE every 5 iterations, same frozen split, same GPU).

Target = the **worse** of the two final test RMSEs at each rank pair, so both systems
provably reach it. Time-to-target = (first checkpoint at or below target) × per-iter wall.

| pair | common target | APR time-to-target | cuMF time-to-target | **TTT speedup** | per-iter ratio |
|------|------|------|------|------|------|
| K=96 / F=100 | 0.817559 | 7.75 s (it 20) | 41.99 s (it 20) | **5.42×** | 5.42× |
| K=64 / F=60 | 0.819547 | 3.45 s (it 15) | 21.37 s (it 20) | **6.19×** | 4.64× |
| K=48 / F=50 | 0.820556 | 2.68 s (it 20) | 16.52 s (it 20) | **6.15×** | 6.15× |
| K=32 / F=30 | 0.824093 | 1.39 s (it 15) | 9.97 s (it 20) | **7.17×** | 5.38× |
| K=16 / F=20 | 0.831057 | 1.15 s (it 20) | 3.65 s (it 10) | **3.16×** | 6.33× |

**The per-iteration comparison does not flatter APR-BALS at 4 of 5 rank pairs.**
At K=64 and K=32 the time-to-target advantage is *larger* than the per-iteration ratio
(6.19× vs 4.64×, 7.17× vs 5.38×) because APR-BALS reaches the shared target in 15
iterations where cuMF needs 20.

**The K=16 / F=20 pair is the honest exception and should be reported as such:**
cuMF's final test RMSE (0.828840) is better than APR's (0.831057), so cuMF clears the
common target by iteration 10 while APR needs all 20 — time-to-target speedup drops to
3.16× against a 6.33× per-iteration ratio. At the lowest rank cuMF's CG solver converges
in fewer sweeps, and per-iteration numbers do overstate the end-to-end advantage there.

Measurement caveat: RMSE is logged every 5 iterations, so each time-to-target carries
±5 iterations of granularity (±0.3 s at K=16, ±1.9 s at K=96 on the cuMF side). The
ranking is unaffected — every gap is far wider than one checkpoint interval.

---

## Item 4 — BALS ladder at every rank: **two published claims do not survive**

`only_scalar_fp32.cu`, `-DBALS_REORDER=1` ± `-DBALS_SYMTILE=1`, legacy init
(`CUMF_INIT=0`) and weighted-λ 0.048 to match the 07-21d archive. 4 datasets ×
K∈{16,32,48,64,96} × 2 variants = 40 runs, one session.

**Sanity check first:** the K=32 rung reproduces the 07-21d archive
(`BALS_EQUIV_VALIDATION_20260721_205101`) to within **0.3% on all 8 runs**
(base 1.917/4.363/6.933/17.968 vs archived 1.921/4.351/6.933/17.970). The harness is
consistent across sessions, so the K≠32 rungs are trustworthy.

### Finding A — symmetric tiling is a K=32 artifact, not a general win

Reorder-only ÷ symtile wall (>1 = symmetric tiling wins):

| K | ml10m | ml20m | ml32m | netflix | verdict |
|---|---|---|---|---|---|
| 16 | 0.930 | 0.910 | 0.878 | 0.906 | **loses everywhere** |
| 32 | 1.209 | 1.149 | 1.077 | 1.149 | wins (the published rung) |
| 48 | 0.839 | 0.805 | 0.772 | 0.806 | **loses everywhere** |
| 64 | 1.062 | 1.003 | 0.950 | 1.023 | mixed / neutral |
| 96 | 0.908 | 0.898 | 0.898 | 0.916 | **loses everywhere** |

**Convergence is not the explanation: iteration counts are identical between the two
legs at every one of the 20 pairs**, so this is a genuine per-iteration effect.

The paper (`sec/5_extended.tex`) states symmetric tiling buys **"1.07×–1.22×"**. That is
exactly the K=32 row and it does not generalize: across all 20 pairs the true range is
**0.772×–1.209×**, and symmetric tiling is a **net loss at 13 of 20**.

Mechanism, from the kernel's own mapping line (`BALS_SYMTILE mapping: … block=N`):
K=16/32 pad to a full **1024**-thread block, K=48 to **768**, K=64/96 to **640**. The
triangular padding forces a smaller block at K≥48 — a 25–37% occupancy loss against the
base kernel's full 1024 — which outweighs the arithmetic that symmetry removes. At K=16
the padding overhead exceeds the saving even at block=1024. Only K=32 lands on a
geometry where symmetry pays. This *reinforces* the paper's own roofline argument (the
scalar Gram kernel runs at ~2% of FP32 peak, so it is latency/occupancy-bound, not
arithmetic-bound) — the mechanism is right, but it implies symmetry should *not* be
expected to pay in general, which is the opposite of what the current text says.

### Finding B — the isolation ratio is strongly rank-dependent, and K=32 is near its floor

Best faithful BALS-equivalent (min of the two legs) ÷ APR-BALS no-momentum
(APR walls from the archived momentum-OFF grid `logs_justapr_vs_cumf_20260721_045838`):

| K | ml10m | ml20m | ml32m | netflix |
|---|---|---|---|---|
| 16 | 4.10× | 3.47× | 3.74× | 4.17× |
| 32 | 6.35× | 7.00× | 6.42× | **8.43×** |
| 48 | 10.60× | 11.35× | 11.59× | 17.76× |
| 64 | 15.31× | 14.72× | 14.71× | 23.64× |
| 96 | 18.89× | 19.62× | 18.38× | **29.90×** |

The paper reports **6.3×–8.4×** — again exactly the K=32 row. Measured across ranks the
range is **3.47×–29.90×**, rising monotonically with K (the tensor-core dispatch has more
to exploit as the Gram work grows with K while the scalar baseline degrades).

**Direction of the correction:** the published 6.3–8.4× is *conservative* — it understates
the advantage at every rank above 32 and overstates it only at K=16. The paper's
"precision dispatch proper accounts for roughly 5.2×–6.9×" qualification (6.3–8.4 divided
by the 1.22× host-level factor) should likewise be restated per-rank rather than as one
range — see Finding C, which shows the divisor itself is rank-dependent and is **1.00× at
K=16**, so dividing the K=16 ratio by 1.22× removes a speedup that binary never had.

### Finding C — the 1.22× host-level divisor is not a constant across ranks (10 runs)

`sec/5_extended.tex:123` divides the isolation ratio by a flat **1.22×** for the
host-level items of `ssec:nonsolver`. That 1.22× was measured at **Netflix K=96 only**
(FIXLOG 07-21: 9.438 → 7.753 s) and bundles three items: λ-fusion, the fast RMSE check,
and the cp.async giant kernel. Of the three, **only the fast RMSE check is flag-gated**
(`-DFAST_RMSE=0`; grep found no `#ifndef`/`#define` guard for the other two), and
`common.cuh:91` defines `FAST_RMSE (K_DIM >= 32 ? 1 : 0)` — so that component is **off by
construction at K=16**, which is precisely where the paper divides by it anyway.

Measured per rank (`host_factor_per_rank/`, Netflix, λ=0.048, momentum OFF to match the
isolation column's protocol, ship recipe vs `-DFAST_RMSE=0`, 10 runs):

| K | ship (`FAST_RMSE` auto) | `-DFAST_RMSE=0` | factor | RMSE-stage Δ | compute Δ | iters | test RMSE |
|---|---|---|---|---|---|---|---|
| 16 | 1.153 s | 1.153 s | **1.001×** | 0.6 ms | 0.7 ms | 20/20 | 0.831057 / 0.831057 |
| 32 | 1.854 s | 1.901 s | 1.025× | 47.1 ms | 3.8 ms | 20/20 | 0.823282 / 0.823282 |
| 48 | 2.691 s | 2.935 s | 1.091× | 244.1 ms | −0.3 ms | 20/20 | 0.820479 / 0.820479 |
| 64 | 4.611 s | 5.057 s | 1.097× | 445.9 ms | −0.5 ms | 20/20 | 0.819080 / 0.819080 |
| 96 | 8.937 s | 9.795 s | 1.096× | 857.7 ms | 3.5 ms | 20/20 | 0.817558 / 0.817558 |

Three things make this a clean isolation. **Test RMSE is bit-identical to six decimals at
every rank** — the fast path changes only how often the check runs, never the answer.
**Iteration counts are 20/20 everywhere**, so it is per-iteration, not convergence.
And **the entire wall-clock delta lands in the RMSE stage**: at K=96 the stage grows
661.02 → 1515.26 ms (Δ 857.7 ms) against a wall gap of 858 ms, while `compute=` moves
8275.62 → 8279.11 ms (0.04%, i.e. run-to-run noise). Nothing else in the run moved.

**What this does and does not license.** It isolates *one* of the three bundled items, so
1.00×/1.03×/1.09–1.10× is a **lower bound** on the full host-level correction, not a
drop-in replacement for 1.22×. The defensible claim is the shape: the correction is
**nil at K=16, marginal at K=32, and ~1.10× at K≥48**, which is enough to establish that
a single constant must not be applied across the rank sweep. Getting one exact number per
rank would require gating λ-fusion and cp.async behind flags too — a code change, not
requested, and not made.

**Recommended action:** in the prose, state the divisor's rank dependence rather than
quoting a per-rank residual computed from the flat 1.22×. The draft replacement in
`PROSE_REVISIONS_5_extended.md` (Change 2) does this.

**Recommended action:** this needs a prose revision in `sec/5_extended.tex`, not just a
table swap — the subsection's argument is currently built on the K=32 numbers being
representative, and they are not. The paper already anticipates this: it flags
"a cleaner attribution would re-measure the ladder … at every rank rather than at
$K{=}32$ alone, and we flag that as the experiment this subsection most needs."
That experiment is now done and the flag can be removed.

---

## Item 3 — Nsight counters: **COMPLETE** (all three requested numbers)

The reviewer asked for tensor-pipe utilization, occupancy, and registers/thread. All
three are now in hand, from three independent instruments:

- **registers/thread** — `nvcc -Xptxas -v`, static, this section below.
- **theoretical occupancy at all 5 ranks** — computed from ptxas + `deviceQuery` +
  launch config, no profiler and no root (section further down).
- **tensor-pipe utilization + achieved occupancy** — `ncu` hardware counters, collected
  **2026-08-04** once the user supplied sudo (`ncu/`, last section of this item).
  For most of this session `ncu` failed with `ERR_NVGPUCTRPERM` (counters are
  root-restricted by default via `NVreg_RestrictProfilingToAdminUsers=1`); running it
  as root bypassed that with no modprobe change and no reboot.

Plus nsys kernel time-shares (unprivileged), which are a *time* decomposition rather than
a hardware counter and are reported separately below.

Starting with what needed no privileges, from `nvcc -Xptxas -v`
(`ptxas_static_regs.txt`, all five K builds of the ship recipe):

| kernel | regs/thread | smem/block | spills |
|--------|------|------|------|
| `compute_LHS_RHS_wmma<6>` (K=96 dense) | 240 | 12 544 B | 0 |
| `compute_LHS_RHS_wmma<4>` | 163 | 8 448 B | 0 |
| `compute_LHS_RHS_wmma<3>` | 141 | 6 400 B | 0 |
| `compute_LHS_RHS_wmma<2>` | 100 | 4 352 B | 0 |
| `compute_LHS_RHS_wmma<1>` | 60 | 2 304 B | 0 |
| `compute_LHS_RHS_wmma_giant<6,4>` (K=96) | 255 | 31 744 B | 0 (32 B stack) |
| `compute_LHS_RHS_wmma_giant<4,4>` | 191 | 23 040 B | 0 (32 B stack) |
| `compute_LHS_RHS_wmma_giant<3,4>` | 133 | 18 688 B | 0 (32 B stack) |
| `compute_LHS_RHS_wmma_giant<2,4>` | 96 | 14 336 B | 0 (32 B stack) |
| `compute_LHS_RHS_wmma_giant<1,4>` | 62 | 9 984 B | 0 (32 B stack) |
| `cholesky_solve_tiled<K,NTH>` (all K) | 56 | (dynamic) | 0 |

This confirms two claims the paper already makes: the 240-register figure for the K=96
dense WMMA kernel (FIXLOG 07-04c, the "2 blocks/SM" occupancy note) and the solver's
56 registers with **zero spills** at every K (FIXLOG 07-06). The scalar baseline kernel
`compute_LHS_RHS_BALS_block<16,3,3>` still shows 640 B stack / 2 544 B spill stores —
the documented K≥48 baseline spill behaviour that ROW_SPLIT mitigates but does not
eliminate (FIXLOG 07-04b).

### nsys kernel profile (unprivileged, DONE) — `nsys_kernel_shares.txt`

Netflix, `apr_k{96,64,32}`, ship recipe. Tracing overhead **<0.5%** (K=96 wall 5.96 s
traced vs 5.94 s native), so the profile is representative of the shipped run.

**Read this before quoting a percentage.** The run uses **3 CUDA streams and kernels
overlap across classes** — summed kernel duration at K=96 is 7.55 s against a 5.96 s
wall. So `cuda_gpu_kern_sum`'s "Time (%)" column is a share of *summed kernel duration*,
not of wall time. Both framings are below; the wall-time one is what a reader will assume
a percentage means, so **prefer it in the paper and label it**. Within any single class no
two kernels ever overlap (self-overlap exactly 1.00× at every K) — all concurrency is
cross-class, Gram kernels against the solver.

Wall-time footprint (union of kernel intervals ÷ total GPU span):

| K | span | GPU busy | dense | giant | **WMMA path** | solver | scalar (sparse tier) | RMSE | WMMA∥solver |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 5.961 s | 99.7% | 48.5% | 17.1% | **65.5%** | 47.2% | 3.2% | 9.9% | 26.9% |
| 64 | 3.539 s | 99.7% | 49.6% | 22.3% | **71.9%** | 37.8% | 1.5% | 12.7% | 25.2% |
| 32 | 1.431 s | 99.6% | 36.6% | 28.6% | **65.2%** | 31.3% | 0.8% | 22.0% | 20.9% |

Share of summed kernel duration (the raw `cuda_gpu_kern_sum` column, for reference):

| K | dense | giant | WMMA total | solver | scalar | RMSE | other |
|---|---|---|---|---|---|---|---|
| 96 | 38.3% | 13.5% | 51.8% | 37.3% | 2.5% | 7.8% | 0.7% |
| 64 | 39.7% | 17.8% | 57.5% | 30.3% | 1.2% | 10.2% | 0.7% |
| 32 | 30.3% | 23.8% | 54.1% | 26.0% | 0.7% | 18.2% | 1.0% |

Four things the paper can use directly:

1. **The tensor-core path is resident for 65–72% of wall time at every rank**, while the
   scalar sparse tier occupies **0.8–3.2%** — the three-tier dispatch spends almost
   nothing on the fallback and routes the bulk of the work to WMMA. This is the strongest
   available support for the dispatch claim short of the TF32 ablation (item 2).
2. **The GPU is busy 99.6–99.7% of the span** — there is essentially no idle gap between
   kernels, so remaining speedup has to come from making kernels faster, not from filling
   bubbles. Useful against a "your host code is the bottleneck" objection.
3. **The solver's share grows with K** (31.3% → 37.8% → 47.2% of wall for K=32/64/96),
   which is exactly why `CHOL_MP=2` is a K=96-only play (FIXLOG 07-20b): at K=32 there is
   far less solver time to attack.
4. **RMSE evaluation dominates at low rank** (9.9% → 22.0% of wall as K falls). The 6
   `compute_RMSE_csr_half` calls cost a **K-independent ~207 ms** (34.6 ms each at every
   K), so at K=32 fixed evaluation is 22% of the run. FAST_RMSE already targets this;
   worth a sentence if the paper discusses low-rank overheads.

Cross-check against the corrected hardware peak: the K=96 profile's own
`Item LHS+RHS 10 650 GFlop/s` ÷ 25.5 TFLOP/s FP32-accumulate = **41.8% of peak** on the
Gram stage, consistent with the paper's revised "45% of peak".

### Theoretical occupancy — COMPUTED, no profiler and no root needed (`occupancy/`)

Achieved occupancy needs `ncu`. **Theoretical** occupancy does not: it is fully determined
by ptxas output, the launch config, and the documented sm_86 limits, so it can be derived
exactly. It is an *upper bound* on achieved occupancy — labelled as such throughout — and
it answers the structural half of the reviewer's occupancy question today.

Inputs, none guessed: regs/thread and static smem from `nvcc -Xptxas -v`
(`occupancy/ptxas_ship_allK.txt`, `occupancy/ptxas_tf32_allK.txt`, all five ranks);
128 threads/block from the launch sites (`justapr.cu:751-761, 864-874`, and
`GIANT_WARPS*32 = 128` at `736-740`); 1536 threads/SM, 65 536 regs, 28 SMs, cc 8.6 from
`deviceQuery` **on this GPU**; 100 KB smem/SM because every pipeline kernel requests
`cudaSharedmemCarveoutMaxShared` at `justapr.cu:621-623`. Derivation is in
`occupancy/occupancy_calc.py`, output in `occupancy/theoretical_occupancy.txt`.

| K | FP16 regs/smem | blk/SM | occ | TF32 regs/smem | blk/SM | occ | giant regs/smem | blk/SM | occ |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 60 / 2 304 | 8 | **66.7%** | 62 / 4 352 | 8 | **66.7%** | 62 / 9 984 | 8 | 66.7% |
| 32 | 100 / 4 352 | 4 | **33.3%** | 97 / 8 448 | 4 | **33.3%** | 96 / 14 336 | 5 | 41.7% |
| 48 | 141 / 6 400 | 3 | **25.0%** | 114 / 12 544 | 4 | **33.3%** | 133 / 18 688 | 3 | 25.0% |
| 64 | 163 / 8 448 | 3 | **25.0%** | 154 / 16 640 | 3 | **25.0%** | 191 / 23 040 | 2 | 16.7% |
| 96 | 240 / 12 544 | 2 | **16.7%** | 252 / 24 832 | 2 | **16.7%** | 255 / 31 744 | 2 | 16.7% |

**Every dense Gram kernel is register-limited at every rank — shared memory never binds.**
At K=96 the FP16 kernel's 12 544 B/block allows 8 blocks/SM on smem alone, and the TF32
kernel's 24 832 B allows 4; registers cap both at **2**. The conclusion is robust to the
one modelling assumption in the calculation: rerunning without register-allocation
granularity rounding moves only the K=32 row, and moves both legs together
(33.3% → 41.7% for FP16 *and* TF32), so no comparison changes. Both variants are printed
in `theoretical_occupancy.txt`.

Three consequences:

1. **This corrects an Item 2 claim.** An earlier draft said the TF32 kernel's doubled smem
   "caps ~2 blocks/SM" and that part of the FP16 win was therefore occupancy. It is not:
   occupancy is **identical (16.7%) at K=96** and equal at 4 of 5 ranks. At **K=48 TF32
   has the higher occupancy** (33.3% vs 25.0%) and yet that is the row where FP16 posts
   its *largest* precision margin (1.32×), so the residual cannot be an occupancy artifact.
   Removing this caveat does not weaken the headline — the pipe share stands on its own.
2. **Occupancy falls from 66.7% to 16.7% as K goes 16 → 96, and the cause is registers.**
   The kernels are declared `__launch_bounds__(WARPS_PER_BLOCK*32, 1)` — the `1` tells
   ptxas only one block per SM need be resident, which licenses it to spend up to 255
   registers per thread, and at K=96 it spends 240. So low occupancy is a *chosen*
   trade (more accumulator registers per warp, fewer warps) rather than an accident.
   Raising the second `__launch_bounds__` argument would trade registers for warps; that
   is a code change and was **not** made or benchmarked.
3. **It reinforces the roofline story.** The dense Gram kernel delivers its measured
   throughput at 16.7% theoretical occupancy at K=96, i.e. 8 of 48 warp slots. A kernel
   that fast at that little latency-hiding is consistent with the staging/latency-bound
   reading in Item 2 (TF32 loses only 1.01–1.32× despite half the nominal tensor peak),
   not with a throughput-bound one.

**Caveat, and how it resolved:** this is an analytical upper bound — it says nothing about
warp stalls, tail effects, or load imbalance across the 28 SMs, so achieved occupancy can
only be lower. **It was subsequently measured** (next section): 15.96% achieved against
16.7% predicted at K=96, i.e. **96% attainment**, and ncu's own occupancy-limiter fields
independently report the register bound. The model is validated where it could be checked.

### Audit: the symtile "occupancy loss" claim, re-derived properly (`occupancy/symtile_occupancy.txt`)

Item 4's Finding A explains symmetric tiling's losses by block size: 1024 threads at
K=16/32, 768 at K=48, 640 at K=64/96, called "a 25–37% occupancy loss". **A smaller block
does not by itself mean lower occupancy** — that is precisely the inference that proved
wrong for TF32 above, so it was re-derived from ptxas + the run banners rather than
asserted.

**The number survives; the reasoning needed repair.** Both legs use ~64 regs/thread and
the *same* dynamic shared memory (`yb_eff*K*4`, `only_scalar_fp32.cu:480`, identical
`yb_eff` on both legs per the run banners), so shared memory cannot explain any gap.
What binds is registers: at 64 regs × 1024 threads both legs fit **exactly one block per
SM**, and with only one resident block the block size *does* pass straight through to
resident warps.

| K | base block | sym block | base occ | sym occ | occ loss | measured reorder÷symtile |
|---|---|---|---|---|---|---|
| 16 | 1024 | 1024 | 66.7% | 66.7% | **0.0%** | 0.878–0.930× (loses) |
| 32 | 1024 | 1024 | 66.7% | 66.7% | **0.0%** | 1.077–1.209× (**wins**) |
| 48 | 1024 | 768 | 66.7% | 50.0% | 25.0% | 0.772–0.839× (loses) |
| 64 | 1024 | 640 | 66.7% | 41.7% | 37.5% | 0.950–1.062× (neutral) |
| 96 | 1024 | 640 | 66.7% | 41.7% | 37.5% | 0.898–0.916× (loses) |

So "25–37%" is right (25.0% / 37.5%), but the honest statement is **"one resident block
per SM, so block size passes through to warps"**, not "smaller block ⇒ less occupancy" in
general. The claim only holds because of the register bound, and that condition must be
stated or the sentence is wrong for the usual reason.

**Occupancy is not the whole explanation, and the table says so.** It accounts for the
losses at K≥48, where the penalty is 25–37.5% and symtile loses or goes neutral. It does
**not** account for K=16: the penalty there is *exactly zero* and symtile still loses
0.878–0.930×. Any single-mechanism story is therefore wrong, and the earlier draft of this
section asserted one.

**Two mechanisms, pulling opposite ways as K grows.** The second is triangular padding
waste, read off the run banners: the padded thread count over the live-tile count is
**3.20× at K=16** (32 threads for 10 live tiles), then 1.78× / 1.23× / 1.18× / 1.07× at
K=32/48/64/96. It falls monotonically with K, while the occupancy loss rises monotonically
with K. At low rank the padding wastes more threads than the symmetry saves; at high rank
the shrunken block costs more warps than the symmetry saves.

**The one-block-per-SM premise is confirmed on hardware, by accident.** The ncu run below
profiled `compute_LHS_RHS_BALS_block<4,3,3>` — a *different* binary, but the same shape as
the ladder's base leg: **block (32,32,1) = 1024 threads, 64 regs/thread**. ncu reports
`launch__occupancy_limit_registers` = **1 block/SM**, exactly the analytical premise this
section rests on. That is the number the whole derivation hinges on, and it was measured
independently.

**One discrepancy in the same row, flagged rather than explained away.** ncu's *achieved*
occupancy for that kernel is **69.05%**, which is 2.4 pp *above* the 66.7% ceiling that one
1024-thread block per SM implies (32 of 48 warp slots). At 64 regs × 1024 threads the block
consumes exactly the whole 65536-register file, so a second concurrent block should be
impossible, and an achieved value above the residency ceiling should not happen.

I do not have a verified explanation. `sm__warps_active.avg.pct_of_peak_sustained_active`
normalizes over *active* cycles rather than elapsed ones, and its denominator and averaging
across SM sub-partitions are the obvious places for a 2.4 pp discrepancy to come from — but
that is a hypothesis I have not tested, and this section is the wrong place to guess at
hardware behaviour, since guessing is what produced the two withdrawn claims above.

What it does and does not affect: **the ladder comparison is untouched**, because both legs
would be measured the same way and the derivation uses the *ratio* of block sizes under a
shared register bound, not the absolute percentage. But it means the 66.7 / 50.0 / 41.7%
figures in the table above must be quoted as **analytical ceilings**, never as measurements,
and the 2.4 pp gap is a live reason not to over-trust a single counter reading against a
model — in either direction.

**K=32 is the only rank where neither mechanism bites** — full block, so zero occupancy
loss, and padding waste already down to 1.78× — and it is the only rank where symmetric
tiling wins. That is the cleanest available explanation of why the published K=32 row
generalizes so badly: it is not a representative sample of the sweep, it is the single
point where both penalties happen to be small.

### Hardware counters — **COLLECTED 2026-08-04, `ncu/`** (the reviewer's last two numbers)

The user ran the staged script with sudo. **Zero `ERR_NVGPUCTRPERM`; all four kernels
profiled.** Netflix K=96, ship recipe, ncu 2025.2.0.0, 1 profiled launch per kernel with
`--launch-skip 8` (steady state, not cold start). Raw CSV
`ncu/ncu_kernels_netflix_k96.txt`, parsed table `ncu/ncu_parsed_summary.txt`.

| kernel | **tensor pipe** | **achieved occ** | theoretical | ach/theo | SM tput | DRAM | regs | waves/SM |
|---|---|---|---|---|---|---|---|---|
| dense WMMA `compute_LHS_RHS_wmma<6>` | **20.50%** | **15.96%** | 16.7% | **96%** | 25.53% | 47.57% | 240 | 50.71 |
| giant WMMA `..._giant<6,4>` | **24.33%** | **16.62%** | 16.7% | **100%** | 75.07% | 11.38% | 255 | 113.30 |
| sparse scalar `..._BALS_block<4,3,3>` | 0% | 69.05% | — | — | 29.74% | 42.24% | 64 | 3.14 |
| solver `cholesky_solve_tiled_mp<96,128,half>` | 0% | 54.49% | — | — | 58.44% | 11.04% | 40 | 0.96 |

**1. The analytical occupancy model is validated by hardware.** Predicted 16.7% for both
WMMA kernels; measured **15.96%** and **16.62%** — 96% and 100% attainment. ncu's own
`launch__occupancy_limit_registers` = **2 blocks/SM for both**, exactly the analytical
result, and `launch__occupancy_limit_shared_mem` = 7 and 3, both well above 2. **This
independently confirms the register-bound finding and the withdrawal of the smem/occupancy
claim** in Item 2. (One small refinement: ncu says the dense kernel's smem limit is 7
blocks where I computed 8 — the driver reserves ~1 KB/block on Ampere, which my model
omitted. It changes nothing: registers bind at 2 either way, and the same correction
applies to both legs.) Because the K=96 prediction lands within 0.7 pp, **the K=16…64
theoretical numbers in the table above can be quoted with confidence.**

**2. The tensor pipe is only 20.5% utilized — this settles the roofline question.** The
kernel that carries the paper's speedup runs its tensor pipe at **one fifth of peak**.
It is emphatically *not* tensor-throughput-bound, which is the direct hardware
confirmation of the Item 2 inference (TF32 costs only 1.01–1.32× despite half the nominal
tensor peak) and of the paper's staging/latency-bound reading. Both were arguments from
timing; this is the counter.

**3. What the dense kernel *is* bound by: staging.** DRAM throughput **47.57%** is its
highest utilization — more than double its SM throughput (25.53%) and its tensor pipe
(20.50%). The dense Gram kernel spends its time **gathering feature rows from memory**,
not multiplying. That is precisely the "staging-bound" claim the paper makes, now with the
counter that shows which resource is closest to saturation.

**4. The giant tier has the opposite profile, and it explains why it needs cp.async.**
SM throughput **75.07%** against DRAM **11.38%** — nearly inverted from the dense tier.
The giant kernel is busy on non-tensor SM work (addressing, shared-memory traffic,
staging) with its tensor pipe at 24.33%. Two tiers, two different bottlenecks, which is
itself an argument for the tiered dispatch.

**5. The strongest single contrast in the whole measurement set:** the scalar fallback
kernel runs at **69.05% achieved occupancy with 0% tensor pipe**, while the dense WMMA
kernel runs at **15.96% occupancy** and is roughly an order of magnitude faster per unit
work. **4.3× the occupancy and still far slower.** Occupancy is not what makes this
solver fast, and any reviewer instinct that "16% occupancy is a problem" is answered by
the kernel sitting right next to it in the same run.

**6. Two secondary observations worth a sentence each.** The solver runs at **0.96
waves/SM** — 189 blocks over 28 SMs never fills the machine even once, so at K=96 it is
launch-bound rather than compute-bound, which bounds how much `CHOL_MP` can ever buy.
And the sparse tier's **3.14 waves/SM** (grid 11×8) means tail effects are material there,
consistent with its tiny 0.8–3.2% wall share.

**Caveat on scope:** one profiled launch per kernel, one dataset, one rank. These are
steady-state single-launch counters, not distributions. The occupancy figure is
corroborated by the independent analytical model; the tensor-pipe and DRAM figures are
single measurements and should be quoted as such.


## Item 2 — Precision-isolating ablation: **COMPLETE — all three rungs measured**

The reviewer wants TF32-WMMA and FP32-SIMT measured **at the same dataflow**, so that
"tensor cores caused this" does not rest on the dispatch toggle alone. Both rungs are now
measured. The TF32 rung required a code change, which the user approved on 2026-08-04.

### The instrument

Three legs, **one source tree, one flag difference each**, everything else identical
(same tiled FP32 solver, same batching, same momentum setting, same convergence rule,
same frozen split, weighted-λ 0.048, `CUMF_INIT=1`, momentum OFF):

| leg | flags | Gram kernel | tensor opcode in SASS |
|---|---|---|---|
| FP16-WMMA | *(none)* | `compute_LHS_RHS_wmma` | `HMMA.16816.F32` |
| TF32-WMMA | `-DWMMA_TF32=1` | `compute_LHS_RHS_wmma_tf32` | `HMMA.1684.F32.TF32` |
| FP32-SIMT | `-DDENSE_NNZ_THRESH=100000000` | `compute_LHS_RHS_BALS_block` | *(none)* |

The FP32-SIMT rung needed no code change: the WMMA launches are guarded by
`if (use_mixed && n_dense_u > 0)` and `n_dense` counts entities with
`nnz >= DENSE_NNZ_THRESH`, so a huge threshold drives `n_dense` to 0, the guard never
fires, and every entity routes to the scalar FP32 tier **inside the same binary**.
Verified in the banner: `User dense=0 entities, Item dense=0 entities`.

**All three legs run with the giant tier disabled** (`-DGIANT_NNZ_THRESH=2000000000`,
banner `User giants=0, Item giants=0`) so the comparison is dense-tier-only and
apples-to-apples. A TF32 giant kernel would need `2·16·K·4` B/warp of double-buffered
staging ≈ 55.8 KB at K=96, over the 48 KB static smem cap on sm_86 — hence the exclusion
rather than a partial port.

**This is a strictly better isolation instrument than the two-binary comparison the paper
currently uses.** Against `only_scalar_fp32.cu`, *only the Gram kernel changes* — no
difference in solver, host pipeline, or convergence bookkeeping can leak into the ratio.

### Results — 18 runs (3 datasets × 2 ranks × 3 legs), all complete

Wall-clock seconds, and Gram-stage ms/iter (**user + item summed** — see the correction
note below):

| dataset | K | FP16 wall | TF32 wall | SIMT wall | FP16 Gram | TF32 Gram | SIMT Gram | iters |
|---|---|---|---|---|---|---|---|---|
| netflix | 96 | **9.24** | 10.36 | 153.14 | **428.79** | 484.66 | 7 624.05 | 20/20/20 |
| netflix | 32 | **1.79** | 2.04 | 14.29 | **70.21** | 82.54 | 694.98 | 20/20/20 |
| ml20m | 96 | **3.39** | 3.67 | 39.84 | **131.79** | 142.96 | 1 589.93 | 25/25/25 |
| ml20m | 32 | **0.54** | 0.63 | 3.69 | **18.73** | 22.01 | 144.59 | 25/25/25 |
| ml10m | 96 | **1.52** | 1.66 | 16.77 | **59.11** | 64.56 | 668.79 | 25/25/25 |
| ml10m | 32 | **0.25** | 0.29 | 1.62 | **8.56** | 10.22 | 63.40 | 25/25/25 |

**The decomposition the reviewer asked for** — Gram-stage ratios:

| dataset | K | tensor pipe (TF32÷SIMT) | reduced precision (FP16÷TF32) | total (FP16÷SIMT) | share of total from the pipe |
|---|---|---|---|---|---|
| netflix | 96 | **15.73×** | 1.13× | 17.78× | **95.5%** |
| netflix | 32 | **8.42×** | 1.18× | 9.90× | 92.7% |
| ml20m | 96 | **11.12×** | 1.08× | 12.06× | **96.7%** |
| ml20m | 32 | **6.57×** | 1.18× | 7.72× | 92.2% |
| ml10m | 96 | **10.36×** | 1.09× | 11.31× | **96.4%** |
| ml10m | 32 | **6.20×** | 1.19× | 7.41× | 92.2% |

Multiplicative shares in log space (`ln(pipe)/ln(total)`). Either way the reading is the
same and it is the headline of this item:

> **The tensor pipe, not the reduced precision, is what buys the speedup — 92–97% of the
> Gram-stage effect survives at full FP32 range.** FP16 adds only a further 1.08–1.19×.

### The same decomposition across the full rank curve (Netflix, K=16→96, 9 more runs)

| K | FP16 Gram | TF32 Gram | SIMT Gram | pipe (TF32÷SIMT) | precision (FP16÷TF32) | total | pipe share |
|---|---|---|---|---|---|---|---|
| 16 | 44.42 | 54.34 | 231.38 | **4.26×** | 1.22× | 5.21× | 87.8% |
| 32 | 70.21 | 82.54 | 694.98 | **8.42×** | 1.18× | 9.90× | 92.9% |
| 48 | 106.76 | 141.18 | 1 549.65 | **10.98×** | 1.32× | 14.52× | 89.6% |
| 64 | 209.53 | 212.16 | 3 647.07 | **17.19×** | **1.01×** | 17.41× | **99.6%** |
| 96 | 428.79 | 484.66 | 7 624.05 | **15.73×** | 1.13× | 17.78× | 95.7% |

Walls (s), FP16 / TF32 / SIMT: 1.09/1.29/4.83, 1.79/2.04/14.29, 2.59/3.28/31.45,
4.71/4.77/73.46, 9.24/10.36/153.14. 20 iterations on all 15 runs. Test RMSE keeps the
TF32 ≤ FP16 < SIMT ordering at **every one of the 5 ranks**.

**Across the whole curve the pipe share is 87.8–99.6%, and reduced precision never buys
more than 1.32×.** Two points are worth a sentence each in the paper:

- **The pipe ratio rises steeply with rank and then saturates** (4.26× → 17.19× from K=16
  to K=64, then flat/slightly down at K=96). This is a *third* independent instrument
  agreeing with Item 4 Finding B and with the K=32→96 doubling in the cross-dataset table:
  K=32 sits near the floor of a strongly rank-dependent effect, so every K=32-only figure
  in the paper is conservative.
- **At K=64 the FP16 advantage over TF32 disappears entirely (1.01×, pipe share 99.6%).**
  At that rank the ablation says the reduced precision contributes *nothing* measurable and
  the entire Gram speedup is the tensor pipe. The non-monotonic precision column
  (1.22 / 1.18 / 1.32 / 1.01 / 1.13×) tracks the smem-and-occupancy differences between the
  two kernels rather than anything arithmetic — which is the same caveat as point 4 below,
  now visible directly in the data.

Four further things worth stating in the paper:

1. **The causal claim is now measured, not inferred.** "Tensor cores caused this" no
   longer rests on the dispatch toggle: with the dataflow held fixed, the tensor-pipe
   Gram kernel is **6.2×–15.7× faster than the scalar one at FP32 range**, and
   **7.4×–17.8× faster at FP16**.
2. **TF32 costs far less than its nominal peak penalty, and that is a measurement of the
   paper's own roofline argument.** Ampere's TF32 tensor-core peak is *half* the FP16
   FP32-accumulate rate (12.7 vs 25.5 TFLOP/s), so a tensor-throughput-bound kernel would
   lose ≈2×. It loses only **1.08–1.19×**. The Gram kernel is therefore **bound by
   staging/latency, not by tensor-pipe throughput** — the same conclusion the paper draws
   from the scalar kernel's low efficiency, now confirmed on the fast path too, with a
   number attached.
3. **Accuracy ordering is consistent and supports the accumulation-order explanation.**
   Test RMSE at every one of the 6 pairs: **TF32 ≤ FP16 < SIMT** (netflix K=96:
   0.817442 / 0.817457 / 0.817593). Iteration counts are identical (20/20/20 on Netflix,
   25/25/25 on MovieLens) and the total spread is ≤0.018%. Both tensor legs beat the
   scalar leg on *test* error while also being lower on *train* error, which is the
   observation the revised paper already attributes to accumulation order rather than to
   a regularization effect — a regularizer would raise train error. The reduced-precision
   leg being the *worse* of the two tensor legs, by a hair, is consistent with that story.
4. **Register/smem cost of the TF32 rung, from `nvcc -Xptxas -v`** (K=96, zero spills on
   both): FP16 240 regs / 12 544 B smem; TF32 **252 regs / 24 832 B smem**. The float
   staging buffer doubles shared memory. **But that doubling costs no occupancy** — see
   the occupancy section under Item 3: both legs are *register*-limited to 2 blocks/SM at
   K=96 (fp16 needs 8 KB/SM of smem against a 100 KB carveout, tf32 16 KB; neither is
   close to binding), so theoretical occupancy is **16.7% for both**. An earlier draft of
   this bullet claimed part of the FP16 win was occupancy; that was wrong and is withdrawn.
   The residual 1.01–1.32× is **not** explained by occupancy at any rank — occupancy is
   equal at 4 of the 5 ranks and at K=48 it actually **favours TF32** (33.3% vs 25.0%)
   in the very row where FP16 posts its largest margin (1.32×).

### Correction to the earlier draft of this section

An earlier version of this file reported **49.68×** for netflix K=96 and **40.3% of peak**.
Those are the **item-side-only** figures (`Item LHS+RHS` 3 703.31 → 74.54 ms/iter;
10 273.1 GFlop/s). The whole Gram stage is user + item summed, and the correct
same-binary whole-stage numbers for that run are:

| quantity | earlier draft (item side only) | corrected (whole Gram stage) |
|---|---|---|
| netflix K=96 Gram speedup | 49.68× | **18.29×** |
| WMMA Gram efficiency | 40.3% of 25.5 TFLOP/s | **14.4%** (3 670.5 GFlop/s) |
| SIMT Gram efficiency | 0.8% | **0.79%** (200.9 GFlop/s) |
| Gram share of SIMT wall | 48.3% | **99.6%** |
| Gram share of WMMA wall | 16.5% | **92.6%** |

Whole-stage speedups, all six pairs of the earlier two-way run (giants enabled, so these
differ slightly from the giants-disabled table above): netflix 18.29× / 9.47×, ml20m
12.23× / 7.68×, ml10m 11.53× / 7.34× at K=96 / K=32. End-to-end (17.01×, 7.71×, 11.92×,
6.75×, 11.21×, 6.37×) was correct and is unchanged.

**The corrected framing is a simpler and stronger argument than the one it replaces.**
The earlier draft built an Amdahl story on the mislabeled numbers — "the Gram stage
shrinks from half the run to a sixth" — which is not what happens. The stage timers sum
to `compute` and Gram dominates *both* legs (99.6% → 92.6% of wall), so this is
near-pure Amdahl with α≈1: an 18.29× stage speedup yielding 17.01× end-to-end is exactly
what a dominant stage predicts. **Do not use the "half the run to a sixth" sentence.**
What remains true is the *ordering* claim — the solver and RMSE work became the next
targets once Gram was accelerated — but that is now visible in item 3's nsys shares
(solver 47.2% of wall today), not in these stage timers.

Note on the two accounting bases: the stage timers here sum to `compute` (sequential-sum
accounting), while item 3's nsys shares are wall-clock interval unions across 3 concurrent
streams. The two are not interchangeable and should not be mixed in one sentence.

Reconciliation with Item 4 Finding B: the two-binary isolation at netflix K=96 gives
**29.90×** end-to-end, this same-binary one gives **17.01×**. Both are correct and they
measure different things — the two-binary ratio includes host-pipeline and solver
differences between `only_scalar_fp32.cu` and the APR harness, while this one isolates the
Gram kernel alone. **The same-binary number is the honest one for a "tensor cores caused
this" claim;** the two-binary number belongs to the BALS-equivalence ladder where those
host differences are part of what is being compared. Reporting both, labelled, is stronger
than either alone.

Raw logs: `precision_isolation/` (2-way, giants enabled, 12 runs) and
`precision_isolation_3way/` (3-way, giants disabled, 18 runs).

### The code change, for the record

Approved by the user on 2026-08-04. Two files, entirely behind `#if WMMA_TF32`
(default OFF):

- `wmma_kernels.cuh` — new kernel `compute_LHS_RHS_wmma_tf32<KT>`, inserted *before*
  `compute_LHS_RHS_wmma` as a separate kernel rather than an edit to the FP16 path.
- `justapr.cu` — the two dense launch sites (user, item) and one `set_max_shared`
  registration, each wrapped `#if WMMA_TF32 … #else … #endif`.

Three forced differences from the FP16 kernel, all documented in the kernel's own comment:
(1) MMA shape 16×16×**8**, so two `mma_sync` per Gram tile instead of one; (2) `float`
staging buffer, smem 12 544 → 24 832 B at K=96; (3) reads the FP32 `d_X`/`d_Y` arrays
directly, because staging from `d_*_half` would pre-round to fp16 and defeat the purpose
of the rung. `wmma::__float_to_tf32` is applied explicitly to every fragment element after
`load_matrix_sync` — that call does *not* round on its own, and omitting it would have
produced a kernel that ran the TF32 pipe on unrounded input.

**Verified inert with the flag off.** Pristine and modified trees were compiled with the
identical command line in identical directories and `cuobjdump -sass` output is
**byte-identical** (111 222 lines, zero diff hunks). Note for future verification: whole-ELF
md5 is *not* a valid invariance test here — the same source compiled in two different
directories yields different SASS register assignments, so ptxas output depends on the
build path. Compare SASS from like-sited builds instead.

---

## Threshold provenance (not on the reviewer's list, but two paper claims were unbacked)

Netflix K=96, weighted-λ 0.048, `CUMF_INIT=1`, momentum OFF, 9 runs.

### θ_g — `sec/3_method.tex:73` claims "values between 128 and 2048 all perform within a few percent"

**Confirmed.** Full spread across the range is **2.7%**:

| θ_g | user giants | item giants | wall (s) | vs best |
|---|---|---|---|---|
| 128 | 178 606 | 14 298 | 9.186 | 1.027× |
| 256 | 97 234 | 11 029 | 9.056 | 1.012× |
| 512 | 35 721 | 8 478 | 8.963 | 1.002× |
| **1024** | 6 345 | 6 403 | **8.945** | **1.000×** |
| 2048 | 455 | 4 572 | 8.962 | 1.002× |

The shipped Netflix value (1024) is the measured optimum, and the curve is flat to within
2.7% over a 16× range of θ_g — the claim "the choice is not delicate" is now backed. Test
RMSE varies only in the 6th decimal (0.817558–0.817559), and all five runs converge in 20
iterations, so θ_g is a pure work-partitioning knob with no accuracy consequence.

### θ_d — `sec/3_method.tex:73` states θ_d = 4 with a *reason* but no sweep ("we did not sweep θ_d itself")

**Now swept, and θ_d = 4 is the measured optimum — monotonically so:**

| θ_d | user dense | user sparse jobs | wall (s) | vs θ_d=4 |
|---|---|---|---|---|
| **4** | 472 960 | 785 | **8.954** | **1.000×** |
| 8 | 465 856 | 2 146 | 9.371 | 1.047× |
| 16 | 433 472 | 9 813 | 11.762 | 1.314× |
| 32 | 360 352 | 30 097 | 18.236 | 2.037× |

Raising θ_d pushes entities out of the WMMA tier and into the scalar tier, and the cost
rises monotonically — **2.04× worse at θ_d = 32**. The paper's stated reason ("the WMMA
tile depth makes shorter gather loops unprofitable") predicts exactly this direction, and
the sweep bounds the penalty. Test RMSE is identical to 6 decimals (0.817558) at all four
values and every run converges in 20 iterations.

**Why this asymmetry is the interesting part:** θ_g is flat to 2.7% over 16×, while θ_d
costs 2.04× over 8×. The tier *boundary between the two WMMA kernels* barely matters; the
boundary *between WMMA and scalar* matters a great deal. That is the dispatch design's own
claim, measured — and it is worth one sentence in the paper, since it says the sensitivity
lives exactly where the precision decision is made and nowhere else.

Raw logs: `threshold_sweep/`.

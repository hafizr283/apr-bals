# Prose revisions for `sec/5_extended.tex` — prepared 2026-08-04, NOT applied

Two claims in `\subsection` "BALS ladder" (around `sec/5_extended.tex:99–123`) rest on
K=32-only measurements and do not generalize. Both are now re-measured at every rank
(Item 4, `bals_ladder_all_K.txt`), and a third independent instrument (Item 2's precision
ablation across the rank curve) agrees on the direction.

**Nothing here is applied.** The `.tex` tree is untouched this session — these are
ready-to-paste replacements for whoever runs the LaTeX box. Numbers below are quoted from
`SUMMARY.md` Items 2 and 4; the raw logs are in `bals_ladder/` and
`precision_isolation_3way/`.

---

## Change 1 — the symmetric-tiling claim (`5_extended.tex:121`, first two sentences)

**Current text says** symmetric tiling buys "$1.07\times$--$1.22\times$". That is exactly
the K=32 row. Across all 20 dataset×rank pairs the true range is **0.772×–1.209×** and
symmetric tiling is a **net loss at 13 of 20**. Iteration counts are identical between the
two legs at every pair, so it is a genuine per-iteration effect, not convergence.

### Replacement

> Table~\ref{tab:balsladder} reports the ladder, freshly re-measured in a single session on
> the same idle GPU used throughout, and two honest findings follow. BALS's symmetric-tiling
> contribution does not survive measurement outside $K{=}32$: re-measured at every rank it
> ranges from $0.772\times$ to $1.209\times$ over the reorder-only kernel and is a net
> \textit{loss} at 13 of the 20 dataset--rank pairs, winning only in the $K{=}32$ row
> reported here. Iteration counts are identical between the two legs at all 20 pairs, so
> this is a per-iteration effect rather than a convergence artifact. The mechanism is
> visible in the kernel's own thread mapping: triangular padding admits a full
> $1024$-thread block at $K \in \{16,32\}$ but only $768$ at $K{=}48$ and $640$ at
> $K \in \{64,96\}$. Because both legs are register-limited to a single resident block
> per SM, that smaller block translates directly into fewer resident warps --- occupancy
> falls from $66.7\%$ to $50.0\%$ at $K{=}48$ and to $41.7\%$ at $K \in \{64,96\}$,
> a $25$--$37.5\%$ loss, while at $K \in \{16,32\}$ the block is full and the penalty is
> exactly zero. Occupancy alone does not close the argument: at $K{=}16$ the penalty is zero
> and symmetric tiling still loses, because a second effect dominates there --- the
> triangular padding launches $3.20\times$ more threads than there are live tiles at
> $K{=}16$, falling monotonically to $1.07\times$ at $K{=}96$ as the occupancy penalty
> rises. The two costs therefore run in opposite directions with rank, and $K{=}32$ is the
> single rank at which neither is large: the block is still full and the padding waste has
> already fallen to $1.78\times$. This is why the $K{=}32$ row generalizes so poorly --- it
> is not a representative point of the sweep but the one point where both penalties are
> small. This \textit{reinforces} the roofline reading --- at about $2\%$ of the RTX
> 3060's FP32 peak the scalar Gram kernel is bound by latency-bound sparse gather and
> synchronization, not by arithmetic --- while inverting the conclusion usually drawn from
> it: halving the arithmetic of a latency-bound kernel should not be expected to pay, and
> here it generally does not.

**Caption fix needed too** (`tab:balsladder`, line 101): the table is K=32-only, and the
caption currently presents it as if it were the general result. Add after the first
sentence: *"Symmetric tiling wins at this rank; Section~\ref{...} reports the full rank
sweep, where it is a net loss at 13 of 20 dataset--rank pairs."*

---

## Change 2 — the isolation-ratio claim (`5_extended.tex:123`)

**Current text says** $6.3\times$--$8.4\times$ against the best faithful BALS-equivalent,
and derives "precision dispatch proper accounts for roughly $5.2\times$--$6.9\times$".
Both are the K=32 row. Measured at every rank the isolation ratio is **3.47×–29.90×**,
rising monotonically with K.

**The correction favors the paper.** The published figure understates the advantage at
every rank above 32 and overstates it only at K=16.

| K | ml10m | ml20m | ml32m | netflix |
|---|---|---|---|---|
| 16 | 4.10× | 3.47× | 3.74× | 4.17× |
| 32 | 6.35× | 7.00× | 6.42× | **8.43×** (the published row) |
| 48 | 10.60× | 11.35× | 11.59× | 17.76× |
| 64 | 15.31× | 14.72× | 14.71× | 23.64× |
| 96 | 18.89× | 19.62× | 18.38× | **29.90×** |

### Replacement for the second half of the paragraph

> The isolation column accordingly uses a momentum-free run at the same sweep count as the
> BALS-equivalent legs. Measured at every rank rather than at $K{=}32$ alone, the ratio
> against the best faithful BALS-equivalent rises monotonically with rank, from
> $3.5\times$--$4.2\times$ at $K{=}16$ through $6.4\times$--$8.4\times$ at $K{=}32$ to
> $18.4\times$--$29.9\times$ at $K{=}96$: the tensor-core dispatch has more to exploit as
> the Gram work grows with $K$, while the scalar baseline degrades over the same range.
> The $K{=}32$ row reported in Table~\ref{tab:balsladder} therefore sits near the floor of
> the effect, and reading it as representative understates the mechanism at every higher
> rank. One qualification keeps the attribution honest: the APR leg retains the host-level
> items of Section~\ref{ssec:nonsolver}, which the BALS-equivalent legs lack. Those items
> are themselves rank-dependent --- the fast RMSE check is disabled by construction below
> $K{=}32$ and contributes $1.00\times$, $1.03\times$, and $1.09$--$1.10\times$ at
> $K{=}16$, $K{=}32$, and $K \ge 48$ respectively --- so the host-level correction cannot
> be applied as a single constant across ranks, and the residual attributable to precision
> dispatch is correspondingly larger at low rank than a flat division would suggest. Since
> the tiled storage, reordering, and symmetric tiling are present in both legs, that
> residual is what routing the dense majority to tensor cores adds on top of everything
> BALS provides.

**Why the previous derivation needs this fix, and what is still unmeasured.** The current
text divides the isolation ratio by a flat $1.22\times$ to get "roughly 5.2×–6.9×". That
$1.22\times$ was measured at **Netflix K=96 only** (FIXLOG 07-21: 9.438 → 7.753 s), and it
bundles three host-level items — λ-fusion, the fast RMSE check, and the cp.async giant
kernel. Only the fast-RMSE component is flag-gated (`-DFAST_RMSE=0`), and measuring it per
rank (`host_factor_per_rank/`, Netflix, 10 runs, momentum off, identical test RMSE and
20 iterations on every pair) gives:

| K | ship recipe | `FAST_RMSE=0` | factor |
|---|---|---|---|
| 16 | 1.153 s | 1.153 s | **1.001×** |
| 32 | 1.854 s | 1.901 s | 1.025× |
| 48 | 2.691 s | 2.935 s | 1.091× |
| 64 | 4.611 s | 5.057 s | 1.097× |
| 96 | 8.937 s | 9.795 s | 1.096× |

**At K=16 the factor is exactly 1.00× — `common.cuh:91` sets `FAST_RMSE (K_DIM >= 32)`, so
the item is off by construction at that rank.** Dividing the K=16 isolation ratio by a
divisor measured at K=96 therefore removes a speedup the K=16 binary never had.

Be careful with the scope of this measurement: it isolates **one of the three** host-level
items, so it is a lower bound on the full correction, not a replacement for the 1.22×. The
λ-fusion and cp.async components are not flag-gated and were not separately measured. The
defensible claim is the *shape* — the correction is near-zero at K=16 and roughly 1.10× at
K≥48 for the one component that can be isolated — which is enough to justify not applying
a flat constant. If a single number per rank is wanted for the final text, the clean
experiment is to gate the other two items behind flags and repeat this sweep.

**Delete the last sentence of the paragraph.** It reads: *"A cleaner attribution would
re-measure the ladder from the two-leg harness of Section~\ref{sec:eval} at every rank
rather than at $K{=}32$ alone, and we flag that as the experiment this subsection most
needs."* That experiment is now done — the flag should go, and this is the reviewer's
item 4 discharged in the text.

---

## Change 3 — new material available if there is room (Item 2's decomposition)

The reviewer's item 2 asked for TF32-WMMA and FP32-SIMT at the same dataflow specifically
so that "tensor cores caused this" would not rest on the dispatch toggle. That is now
measured three ways, and the result is a cleaner claim than the paper currently makes:

> Holding the dataflow, solver, batching, and convergence rule fixed and changing only the
> Gram kernel, the tensor-pipe rung (TF32, full FP32 range) accounts for $87.8\%$--$99.6\%$
> of the Gram-stage speedup across all five ranks, while dropping to FP16 adds at most a
> further $1.32\times$ and at $K{=}64$ adds nothing measurable ($1.01\times$). The
> advantage is therefore attributable to tensor-core execution rather than to reduced
> precision. Ampere's TF32 tensor peak is half the FP16 FP32-accumulate rate, so a
> throughput-bound kernel would lose about $2\times$ on this substitution; losing only
> $1.01\times$--$1.32\times$ places the Gram kernel firmly in the staging- and
> latency-bound regime, confirming on the fast path the same conclusion the roofline
> analysis draws for the scalar one.

Placement suggestion: this belongs near the existing isolation discussion, and it is the
strongest single answer to the "your speedup might just be lower precision" objection.
If page budget is tight, the two sentences from "Holding the dataflow" to "reduced
precision" carry the argument on their own.

**One number to avoid.** An earlier draft of `SUMMARY.md` reported 49.7× and "40.3% of
peak" for the Netflix K=96 Gram stage. Those were item-side-only; whole-stage they are
**18.29×** and **14.4%**. See the correction block in `SUMMARY.md` Item 2 — and do not
reuse the "shrinks from half the run to a sixth" Amdahl sentence, which was built on them.

---

## Change 4 — item 3's hardware counters (NEW, no drafted text existed for this)

The reviewer asked for tensor-pipe utilization, occupancy, and registers/thread. All three
are now measured on hardware (`ncu/`, Netflix K=96, ship recipe, steady-state launch), and
they do more than discharge the request — they turn two arguments the paper currently makes
*from timing* into arguments *from counters*. This is the strongest evidence in the whole
revision and it currently has nowhere to live in the text.

> Hardware counters collected with Nsight Compute on the steady-state Gram launch place the
> dense tensor-core kernel at $240$ registers per thread, $15.96\%$ achieved occupancy, and
> --- the number that matters for the roofline argument --- only $20.50\%$ tensor-pipe
> utilization. The kernel that carries the speedup runs its tensor pipe at one fifth of
> peak, so it is not tensor-throughput-bound; its highest single utilization is DRAM at
> $47.57\%$, against $25.53\%$ SM throughput, identifying feature-row staging rather than
> arithmetic as the limiter. The giant tier inverts this profile ($75.07\%$ SM against
> $11.38\%$ DRAM), so the two tensor-core tiers are bound by different resources --- which
> is itself an argument for dispatching them separately. Occupancy is a deliberate trade
> rather than a defect: the scalar fallback in the same run reaches $69.05\%$ occupancy
> with the tensor pipe idle and is nonetheless far slower per unit work than the dense
> kernel at $15.96\%$, so warp residency is not what makes this solver fast.

**Why this paragraph is worth the space.** It answers three separate objections at once:
"your speedup is just lower precision" (tensor pipe at 20.5% — there is no throughput
headroom being bought), "16% occupancy is bad" (the 69% kernel next to it is slower), and
"the roofline claim is inferred, not measured" (DRAM 47.57% > SM 25.53% > tensor 20.50%).

**Two numbers to quote carefully.**
- The occupancy figure is corroborated by an independent analytical model (predicted
  $16.7\%$, measured $15.96\%$ — $96\%$ attainment) and by ncu's own
  `launch__occupancy_limit_registers` $= 2$ blocks/SM. Quote it freely.
- Tensor-pipe and DRAM utilization are **single measurements** — one profiled launch, one
  dataset, one rank. They are not distributions and should not be given a $\pm$. If a
  reviewer asks for spread, that is a re-run, not a re-analysis.

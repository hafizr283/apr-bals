#!/usr/bin/env bash
# =============================================================================
#  Paper item 3 — the LAST two missing reviewer numbers:
#     (1) hardware tensor-pipe utilization
#     (2) ACHIEVED occupancy   (theoretical occupancy is already done, no root:
#                               see ../occupancy/theoretical_occupancy.txt)
#
#  HOW TO RUN (one command, from anywhere):
#     sudo /home/pc/Desktop/2007080/latent_factorr_experimen/comparing_baseline/results/PAPER_MEASUREMENTS_20260803/ncu/run_ncu_kernels.sh
#
#  WHY sudo: ncu needs GPU performance counters, which the NVIDIA driver
#  restricts to root by default (NVreg_RestrictProfilingToAdminUsers=1 ->
#  ERR_NVGPUCTRPERM for a normal user). Running ncu AS root bypasses that
#  directly, so this needs NO modprobe edit and NO reboot. The box's security
#  posture is unchanged after the run: nothing is installed, no config touched.
#
#  WHAT IT TOUCHES: reads the dataset + the prebuilt binary in ./bin, writes ONE
#  text file into this directory. No project source is compiled or modified.
#
#  COST: profiles 4 kernels, 1 launch each. See the timing note at the bottom.
#  Safe to re-run; it overwrites its single output file (previous run is kept
#  as *.prev.txt).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU=/usr/local/cuda-12.9/bin/ncu
DS=/home/pc/Desktop/2007080/netflix_ratings.bin
BIN="$HERE/bin/apr_k96"
OUT="$HERE/ncu_kernels_netflix_k96.txt"
OWNER=pc:pc                       # so the output is readable without sudo

export CUDA_VISIBLE_DEVICES=1     # GPU0 drives the display; all paper runs use GPU1

# ---- preflight: fail loudly BEFORE spending time on the GPU -----------------
fail=0
[ -x "$NCU" ] || { echo "MISSING: ncu at $NCU"; fail=1; }
[ -x "$BIN" ] || { echo "MISSING: binary at $BIN"; fail=1; }
[ -r "$DS"  ] || { echo "MISSING: dataset at $DS"; fail=1; }
if [ "$(id -u)" -ne 0 ]; then
  echo "NOT ROOT — ncu will fail with ERR_NVGPUCTRPERM. Re-run with:"
  echo "   sudo $HERE/run_ncu_kernels.sh"
  fail=1
fi
[ "$fail" -eq 0 ] || { echo "preflight failed, nothing was run."; exit 1; }

# ---- metrics ----------------------------------------------------------------
# The first two ARE the reviewer's two missing numbers; the rest are cheap
# context collected in the same pass (no extra kernel replays).
M="sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,launch__waves_per_multiprocessor,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed"

# --launch-skip 8 skips the first 8 launches so we profile a steady-state
# iteration, not a cold-start one. --launch-count 1 keeps the cost to a single
# profiled launch per kernel.
run() { # $1 = human label, $2 = mangled-name regex
  echo ""
  echo "##### $1"
  timeout 1200 "$NCU" --kernel-name-base mangled -k "regex:$2" \
    --launch-count 1 --launch-skip 8 --metrics "$M" --csv "$BIN" "$DS" 0.048 2>&1
}

[ -f "$OUT" ] && cp -f "$OUT" "${OUT%.txt}.prev.txt"

{
  echo "# ncu kernel-only profile — paper item 3 (tensor-pipe utilization, achieved occupancy)"
  echo "# binary: apr_k96  flags: -DK_DIM=96 -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1 -DCHOL_MP=2"
  echo "# dataset: netflix_ratings.bin  lambda 0.048  GPU1 RTX 3060 (sm_86, ga106)"
  echo "# date: $(date -Is)"
  "$NCU" --version 2>/dev/null | grep -i version
  # All 4 regexes were validated against `cuobjdump -symbols` on THIS binary,
  # so no run is wasted on a name miss.
  run "dense WMMA tier    compute_LHS_RHS_wmma<6>"             "compute_LHS_RHS_wmmaILi6E"
  run "giant WMMA tier    compute_LHS_RHS_wmma_giant<6,4>"     "compute_LHS_RHS_wmma_giantILi6ELi4E"
  run "sparse scalar tier compute_LHS_RHS_BALS_block<4,3,3>"   "compute_LHS_RHS_BALS_blockILi4ELi3ELi3E"
  run "solver             cholesky_solve_tiled_mp<96,128,half>" "cholesky_solve_tiled_mpILi96E"
  echo ""
  echo "NCU_DONE $(date +%H:%M:%S)"
} > "$OUT" 2>&1

chown "$OWNER" "$OUT" "${OUT%.txt}.prev.txt" 2>/dev/null

echo "-----------------------------------------------------------------------"
echo "wrote: $OUT   ($(grep -c '' "$OUT") lines)"
if grep -q ERR_NVGPUCTRPERM "$OUT"; then
  echo "RESULT: STILL COUNTER-BLOCKED — ERR_NVGPUCTRPERM is in the output."
elif grep -qi 'sm__pipe_tensor_op_hmma' "$OUT"; then
  echo "RESULT: OK — tensor-pipe metric present. Both missing numbers collected."
else
  echo "RESULT: ran, but no tensor metric found — check the file for errors."
fi
echo "Tell Claude it is done; it will parse this file into SUMMARY.md / FIXLOG.md."

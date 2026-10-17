#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
mkdir -p new_results/ncu

# Keep these measurements separate from the completed section-based profiles.
export CUDA_VISIBLE_DEVICES=1
export MAX_ITERS=10 FIXED_ITERS=1 CHECK_EVERY=1

metrics='smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active'

for kernel in 'compute_LHS_RHS_wmmaILi6E' 'compute_LHS_RHS_wmma_giantILi6ELi4E'; do
  if [[ "$kernel" == *giant* ]]; then
    label=giant
  else
    label=dense
  fi

  /usr/local/cuda/bin/ncu \
    --target-processes all \
    --kernel-name-base mangled \
    --kernel-name "regex:$kernel" \
    --launch-skip 8 \
    --launch-count 1 \
    --metrics "$metrics" \
    --csv \
    new_results/bin/ship_k96 \
    /home/pc/Desktop/2007080/netflix_ratings.bin \
    .048 \
    > "new_results/ncu/${label}_explicit_stalls.csv" 2>&1
done

if [[ -n "${SUDO_USER:-}" ]]; then
  chown "$SUDO_USER":"$(id -gn "$SUDO_USER")" \
    new_results/ncu/dense_explicit_stalls.csv \
    new_results/ncu/giant_explicit_stalls.csv
fi

echo "Explicit long-scoreboard and LG-throttle profiles completed."

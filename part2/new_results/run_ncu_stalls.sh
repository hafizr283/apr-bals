#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
mkdir -p new_results/ncu
# Do not contaminate the serial benchmark timings.
while [[ ! -f new_results/logs/external.done ]]; do sleep 5; done
export CUDA_VISIBLE_DEVICES=1
export MAX_ITERS=10 FIXED_ITERS=1 CHECK_EVERY=1
for kernel in 'compute_LHS_RHS_wmmaILi6E' 'compute_LHS_RHS_wmma_giantILi6ELi4E'; do
 if [[ "$kernel" == *giant* ]]; then label=giant; else label=dense; fi
 /usr/local/cuda/bin/ncu --target-processes all --kernel-name-base mangled --kernel-name "regex:$kernel" --launch-skip 8 --launch-count 1 --section WarpStateStats --section ComputeWorkloadAnalysis --csv new_results/bin/ship_k96 /home/pc/Desktop/2007080/netflix_ratings.bin .048 > "new_results/ncu/${label}_stalls.csv" 2>&1
done

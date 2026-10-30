#!/usr/bin/env python3
"""
Theoretical occupancy of the APR-BALS dense Gram kernels on sm_86 (RTX 3060).

Theoretical occupancy is calculated from ptxas resource usage, documented
sm_86 limits, and the kernel launch configuration. This analytical calculation
does not require a GPU profiler or access to performance counters.
Achieved occupancy is a separate runtime measurement obtained with `ncu`;
counter access may require administrator permissions (ERR_NVGPUCTRPERM).
The values computed here are resource-based upper bounds on achieved occupancy,
not measurements of kernel execution.

INPUT SOURCES:
  - regs/thread and static smem/block: ptxas -v, archived beside this script in
    ptxas_ship_allK.txt (ship recipe) and ptxas_tf32_allK.txt (-DWMMA_TF32=1).
  - threads/block = WARPS_PER_BLOCK*32 = 128, from the launch sites in
    justapr.cu (dense: 751-761, 864-874; giant: GIANT_WARPS*32 = 128, 736-740).
  - per-SM limits: `deviceQuery` on this GPU reports 1536 threads/SM and 65536
    registers, 28 SMs, cc 8.6. Warps/SM = 1536/32 = 48. Max blocks/SM = 16 and
    the allocation granularities are from the CUDA C Programming Guide table for
    compute capability 8.6.
  - smem/SM = 100 KB: every pipeline kernel calls cudaFuncSetAttribute(
    cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)
    at justapr.cu:621-623, which selects the 100 KB carveout on sm_86.

Run:  python3 occupancy_calc.py
"""
import math

# ---- sm_86 per-SM limits -----------------------------------------------------
REGS_SM   = 65536    # 32-bit registers per SM         (deviceQuery)
WARPS_SM  = 48       # = 1536 threads/SM / 32          (deviceQuery)
BLOCKS_SM = 16       # max resident blocks per SM      (CUDA C PG, cc 8.6)
SMEM_SM   = 102400   # 100 KB carveout                 (MaxShared, see above)
REG_UNIT  = 256      # register allocation unit size   (CUDA C PG, cc 8.6)
SMEM_UNIT = 128      # shared memory allocation unit   (CUDA C PG, cc 8.6)

THREADS_PER_BLOCK = 128          # WARPS_PER_BLOCK(4) * 32, all Gram launches
WPB = THREADS_PER_BLOCK // 32    # = 4 warps per block

# ---- ptxas results, K -> (regs/thread, static smem bytes/block) ---------------
FP16 = {16: (60,  2304), 32: (100, 4352), 48: (141,  6400),
        64: (163, 8448), 96: (240, 12544)}
TF32 = {16: (62,  4352), 32: ( 97, 8448), 48: (114, 12544),
        64: (154, 16640), 96: (252, 24832)}
GIANT = {16: (62,  9984), 32: ( 96, 14336), 48: (133, 18688),
         64: (191, 23040), 96: (255, 31744)}


def blocks_per_sm(regs, smem, granular=True):
    """Returns (blocks/SM, by_registers, by_smem, binding_resource)."""
    # registers are allocated per warp, rounded up to REG_UNIT
    regs_per_warp = math.ceil(regs * 32 / REG_UNIT) * REG_UNIT if granular else regs * 32
    by_reg  = REGS_SM // (regs_per_warp * WPB)
    by_smem = SMEM_SM // (math.ceil(smem / SMEM_UNIT) * SMEM_UNIT) if smem else BLOCKS_SM
    by_warp = WARPS_SM // WPB
    b = min(by_reg, by_smem, by_warp, BLOCKS_SM)
    binding = ('registers' if by_reg == b else
               'shared memory' if by_smem == b else 'warp slots')
    return b, by_reg, by_smem, binding


def table(name, data, granular=True):
    print(f"\n{name}  (threads/block={THREADS_PER_BLOCK}, {WPB} warps)")
    print(f"{'K':>4} {'reg/thr':>8} {'smem B':>8} {'blk/SM':>7} "
          f"{'by reg':>7} {'by smem':>8} {'warps':>6} {'occupancy':>10}  binding")
    out = {}
    for K in (16, 32, 48, 64, 96):
        r, s = data[K]
        b, br, bs, lim = blocks_per_sm(r, s, granular)
        occ = 100.0 * b * WPB / WARPS_SM
        out[K] = occ
        print(f"{K:>4} {r:>8} {s:>8} {b:>7} {br:>7} {bs:>8} {b*WPB:>6} "
              f"{occ:>9.1f}%  {lim}")
    return out


if __name__ == '__main__':
    for granular in (True, False):
        lbl = 'WITH' if granular else 'WITHOUT'
        print('=' * 78)
        print(f'Register allocation granularity rounding: {lbl} '
              f'(sanity check -- conclusion must not depend on it)')
        print('=' * 78)
        f = table('FP16-WMMA dense  compute_LHS_RHS_wmma', FP16, granular)
        t = table('TF32-WMMA dense  compute_LHS_RHS_wmma_tf32', TF32, granular)
        g = table('FP16-WMMA giant  compute_LHS_RHS_wmma_giant', GIANT, granular)
        print('\nFP16 vs TF32 theoretical occupancy:')
        for K in (16, 32, 48, 64, 96):
            v = ('EQUAL' if abs(f[K] - t[K]) < 1e-9 else
                 'TF32 HIGHER' if t[K] > f[K] else 'fp16 higher')
            print(f'   K={K:<3} fp16={f[K]:5.1f}%  tf32={t[K]:5.1f}%   -> {v}')
        print()

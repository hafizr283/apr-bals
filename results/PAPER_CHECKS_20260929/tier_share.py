"""Share of training nonzeros routed to the tensor-core tiers under theta_d = 4,
for users and items, directly from the frozen .bin splits (same reader as Fig. 1)."""
import struct, sys
import numpy as np

def degrees(path):
    def skip(f):
        (n,) = struct.unpack("<Q", f.read(8)); f.seek(n * 4, 1)
    def vec(f):
        (n,) = struct.unpack("<Q", f.read(8)); return np.frombuffer(f.read(n * 4), dtype=np.int32)
    with open(path, "rb") as f:
        f.read(20)
        for _ in range(6): skip(f)
        uo = vec(f); skip(f); skip(f); io = vec(f)
    return np.diff(uo).astype(np.int64), np.diff(io).astype(np.int64)

TD = 4
for name in sys.argv[2:]:
    u, i = degrees(sys.argv[1] + "/" + name)
    for side, d in (("user", u), ("item", i)):
        d = np.sort(d)[::-1]                      # frequency-sorted order
        n_dense = int((d >= TD).sum())
        n_dense_r = (n_dense // 32) * 32          # boundary rounded down to a tile row
        tc = d[:n_dense_r].sum() / d.sum()
        print(f"{name:20s} {side}: entities>=theta_d {n_dense}/{len(d)}  "
              f"TC nnz share {100*tc:.3f}%  (unrounded {100*d[:n_dense].sum()/d.sum():.3f}%)")

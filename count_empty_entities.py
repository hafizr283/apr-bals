#!/usr/bin/env python3
"""
count_empty_entities.py -- training-entity coverage and masked test RMSE.

Counts, per dataset, how many items (and users) receive ZERO training ratings
after the frozen 80/20 split, and how many held-out test ratings reference
them. Those items are what make cuMF_als report `nan` test RMSE: under the
weighted regularizer (diag += nnz*lambda) an item with no training rating has an
all-zero system, cuMF's CG step computes alpha = 0/0 (cg.cu) and returns a NaN
factor, and the NaN guard in cuMF's RMSE kernel (SURPASS_NAN, als.cu) is
compiled out by default, so one such item in the test set poisons the total.

Optionally recomputes a MASKED test RMSE from a cuMF factor dump, excluding
test ratings whose item had no training rating, so the affected datasets can
rejoin the held-out accuracy comparison instead of being dropped from it.

Usage
-----
  # census of entities with no training ratings
  python count_empty_entities.py ratings10.bin ratings.bin ratings32.bin \
                                 netflix_ratings.bin ratings100.bin

  # census + masked test RMSE for one dataset from a cuMF factor dump
  python count_empty_entities.py ratings.bin \
      --cumf-x cumf_XT_f100.bin --cumf-theta cumf_thetaT_f100.bin -f 100

Notes
-----
* Reads the same .bin frozen split every system in the harness reads, so the
  counts are properties of the split, not of any implementation.
* cuMF stores XT (f x m) and thetaT (f x n) as row-major float32; a rating is
  predicted as dot(XT[:,u], thetaT[:,i]). Pass -f to match the dump's rank.
* Pure NumPy, no GPU, no CUDA. Runs in seconds on Netflix.

.bin layout (little-endian), per convert_bin_for_ccd_sgd.py:
  int32  version, num_users, num_items, nnz_train, nnz_test
  then size-prefixed vectors (uint64 count + payload):
    train: users(i32) items(i32) ratings(f32)
    test : users(i32) items(i32) ratings(f32)
    csr / csc blocks follow and are not read here
"""
import argparse
import os
import struct
import sys

import numpy as np


def read_bin(path):
    """Read users/items/ratings for the train and test halves of a frozen split."""
    with open(path, "rb") as f:
        ver, nu, ni, ntr, nte = struct.unpack("<5i", f.read(20))

        def rvec(dtype):
            (sz,) = struct.unpack("<Q", f.read(8))
            buf = f.read(np.dtype(dtype).itemsize * sz)
            return np.frombuffer(buf, dtype=dtype)

        tr_u = rvec("<i4")
        tr_i = rvec("<i4")
        tr_r = rvec("<f4")
        te_u = rvec("<i4")
        te_i = rvec("<i4")
        te_r = rvec("<f4")

    return dict(
        version=ver, num_users=nu, num_items=ni,
        nnz_train=ntr, nnz_test=nte,
        tr_u=tr_u, tr_i=tr_i, tr_r=tr_r,
        te_u=te_u, te_i=te_i, te_r=te_r,
    )


def census(d):
    """Zero-training-rating entity counts and their held-out exposure."""
    m, n = d["num_users"], d["num_items"]

    item_train = np.bincount(d["tr_i"], minlength=n)
    user_train = np.bincount(d["tr_u"], minlength=m)

    empty_items = np.flatnonzero(item_train == 0)
    empty_users = np.flatnonzero(user_train == 0)

    # Held-out ratings that reference an item with no training row: these are
    # the ones whose prediction is undefined for a system without a fallback.
    te_bad_item = np.isin(d["te_i"], empty_items)
    te_bad_user = np.isin(d["te_u"], empty_users)
    te_bad_any = te_bad_item | te_bad_user

    return dict(
        users=m, items=n,
        nnz_train=int(d["tr_u"].size), nnz_test=int(d["te_u"].size),
        empty_items=int(empty_items.size),
        empty_users=int(empty_users.size),
        test_hits_empty_item=int(te_bad_item.sum()),
        test_hits_empty_user=int(te_bad_user.sum()),
        test_hits_any=int(te_bad_any.sum()),
        mask_any=te_bad_any,
        empty_item_ids=empty_items,
    )


def masked_rmse(d, c, xt_path, theta_path, f):
    """
    Test RMSE from a cuMF factor dump, reported three ways:
      all      -- every held-out rating (what cuMF itself reports; nan if any
                  empty-row item is referenced and cuMF left it uninitialized)
      masked   -- excluding held-out ratings on zero-training-rating entities
      finite   -- excluding only ratings whose prediction came out non-finite
    """
    m, n = d["num_users"], d["num_items"]
    xt = np.fromfile(xt_path, dtype=np.float32)
    th = np.fromfile(theta_path, dtype=np.float32)

    if xt.size != f * m or th.size != f * n:
        sys.exit(
            "factor dump size mismatch: XT has {} floats (expected f*m = {}*{} = {}), "
            "thetaT has {} (expected f*n = {}*{} = {}). Check -f."
            .format(xt.size, f, m, f * m, th.size, f, n, f * n)
        )

    # cuMF stores the transposes: XT is f x m, thetaT is f x n, row-major.
    X = xt.reshape(f, m)
    T = th.reshape(f, n)

    pred = np.einsum("fu,fu->u", X[:, d["te_u"]], T[:, d["te_i"]]).astype(np.float64)
    err = pred - d["te_r"].astype(np.float64)

    def rmse(sel):
        if sel.sum() == 0:
            return float("nan"), 0
        e = err[sel]
        return float(np.sqrt(np.mean(e * e))), int(sel.sum())

    all_sel = np.ones(err.size, dtype=bool)
    masked_sel = ~c["mask_any"]
    finite_sel = np.isfinite(err)

    return dict(
        all=rmse(all_sel),
        masked=rmse(masked_sel),
        finite=rmse(finite_sel),
        nonfinite=int((~finite_sel).sum()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bins", nargs="+", help="frozen-split .bin files")
    ap.add_argument("--cumf-x", help="cuMF XT factor dump (f x m float32)")
    ap.add_argument("--cumf-theta", help="cuMF thetaT factor dump (f x n float32)")
    ap.add_argument("-f", type=int, help="cuMF rank F matching the dumps")
    args = ap.parse_args()

    if (args.cumf_x or args.cumf_theta) and not (args.cumf_x and args.cumf_theta and args.f):
        sys.exit("--cumf-x, --cumf-theta and -f must be given together")
    if args.cumf_x and len(args.bins) != 1:
        sys.exit("masked-RMSE mode takes exactly one .bin")

    hdr = ("{:<22} {:>9} {:>8} {:>7} {:>7} {:>9} {:>9}"
           .format("dataset", "items", "empty", "users", "empty", "test nnz", "affected"))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for path in args.bins:
        d = read_bin(path)
        c = census(d)
        rows.append((path, d, c))
        print("{:<22} {:>9,} {:>8,} {:>7,} {:>7,} {:>9,} {:>9,}".format(
            os.path.basename(path), c["items"], c["empty_items"],
            c["users"], c["empty_users"], c["nnz_test"], c["test_hits_any"]))

    print()
    for path, d, c in rows:
        frac = 100.0 * c["test_hits_any"] / max(1, c["nnz_test"])
        print("{}: {} of {:,} items ({:.3f}%) have zero training ratings; "
              "{:,} of {:,} held-out ratings ({:.4f}%) reference an empty-row entity."
              .format(os.path.basename(path), c["empty_items"], c["items"],
                      100.0 * c["empty_items"] / max(1, c["items"]),
                      c["test_hits_any"], c["nnz_test"], frac))
        if c["empty_items"] == 0:
            print("  -> no empty-row items: a system without a zero-count fallback "
                  "is safe on this split.")
        else:
            print("  -> a system without a zero-count fallback yields a non-finite "
                  "aggregate here.")

    if args.cumf_x:
        path, d, c = rows[0]
        r = masked_rmse(d, c, args.cumf_x, args.cumf_theta, args.f)
        print()
        print("cuMF test RMSE from factor dump ({}), F={}".format(
            os.path.basename(args.cumf_x), args.f))
        for k in ("all", "masked", "finite"):
            val, cnt = r[k]
            print("  {:<8} {:>12}   over {:,} ratings".format(
                k, "nan" if np.isnan(val) else "{:.6f}".format(val), cnt))
        print("  non-finite predictions: {:,}".format(r["nonfinite"]))
        print("\n  'masked' is the figure the paper needs: held-out RMSE with "
              "zero-training-rating entities excluded,\n  so the dataset rejoins the "
              "accuracy comparison instead of being dropped.")


if __name__ == "__main__":
    main()

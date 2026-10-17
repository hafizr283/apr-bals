#!/usr/bin/env python3
"""Build and run the FP16->FP32 solver-fallback experiment on GPU 1.

All timed runs are serial. The test-split binary supplies the fixed training
set and preserved original test set. Validation RMSE is scored from each run's
dumped factors against the companion validation split, whose training set is
identical.
"""
import csv
import json
import math
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
ROOT = R.parent
CUDA = Path("/usr/local/cuda/bin/nvcc")
TEST_DATA = R / "splits/netflix_ratings_test.bin"
VALID_DATA = R / "splits/netflix_ratings_validation.bin"
LOGDIR = R / "fallback_logs"
CMDDIR = R / "fallback_commands"
BINDIR = R / "fallback_bin"
TMPDIR = R / "fallback_factors_tmp"
for directory in (LOGDIR, CMDDIR, BINDIR, TMPDIR):
    directory.mkdir(exist_ok=True)

BASE = [str(CUDA), "-O3", "-arch=sm_86", "-std=c++14",
        "-DWEIGHTED_LAMBDA=1", "-DCUMF_INIT=1", "-DK_DIM=96",
        "-DMOMENTUM=1", "-DFAST_RMSE=0", "-Xptxas=-v"]

BUILDS = {
    "fallback_noref": (R / "fallback_src/justapr.cu", ["-DCHOL_MP=2"]),
    "fallback_ir1": (R / "fallback_src/justapr.cu", ["-DCHOL_MP=2", "-DCHOL_REFINE=1"]),
    "zero_noref": (R / "src/justapr.cu", ["-DCHOL_MP=2"]),
    "zero_ir1": (R / "src/justapr.cu", ["-DCHOL_MP=2", "-DCHOL_REFINE=1"]),
    "fp32": (R / "src/justapr.cu", ["-DCHOL_MP=0"]),
}

CASES = [
    ("stable_selected", "fallback_noref", 0.04, 0.3),
    ("stable_selected", "zero_noref", 0.04, 0.3),
    ("stable_selected", "fp32", 0.04, 0.3),
    ("hard_beta05", "fallback_noref", 0.048, 0.5),
    ("hard_beta05", "fallback_ir1", 0.048, 0.5),
    ("hard_beta05", "zero_noref", 0.048, 0.5),
    ("hard_beta05", "zero_ir1", 0.048, 0.5),
    ("hard_beta05", "fp32", 0.048, 0.5),
]


def run_checked(cmd, **kwargs):
    result = subprocess.run(cmd, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {cmd}")
    return result


def build_all():
    for name, (source, flags) in BUILDS.items():
        binary = BINDIR / name
        cmd = BASE + flags + [str(source), "-o", str(binary)]
        (CMDDIR / f"build_{name}.json").write_text(json.dumps({"command": cmd}, indent=2))
        with (LOGDIR / f"build_{name}.txt").open("w") as out:
            run_checked(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def read_header(path):
    with path.open("rb") as f:
        version, users, items, train_n, test_n = struct.unpack("<5i", f.read(20))
    return {"version": version, "users": users, "items": items,
            "train_n": train_n, "test_n": test_n}


def read_heldout(path):
    dtypes = (np.int32, np.int32, np.float32, np.int32, np.int32, np.float32)
    arrays = []
    with path.open("rb") as f:
        f.seek(20)
        for index, dtype in enumerate(dtypes):
            n = struct.unpack("<Q", f.read(8))[0]
            nbytes = n * np.dtype(dtype).itemsize
            if index < 3:
                f.seek(nbytes, 1)
            else:
                arrays.append(np.fromfile(f, dtype=dtype, count=n))
    return arrays


def score_factors(prefix, heldout, users, items, k=96):
    x_path = Path(str(prefix) + "_X.bin")
    y_path = Path(str(prefix) + "_Y.bin")
    x = np.memmap(x_path, dtype=np.float32, mode="r", shape=(users, k))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(items, k))
    factors_finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    hu, hi, hr = heldout
    sq = 0.0
    predictions_finite = True
    chunk = 100_000
    for start in range(0, len(hr), chunk):
        stop = min(start + chunk, len(hr))
        pred = np.einsum("ij,ij->i", x[hu[start:stop]], y[hi[start:stop]],
                         dtype=np.float64, optimize=True)
        predictions_finite &= bool(np.isfinite(pred).all())
        err = pred - hr[start:stop]
        sq += float(np.dot(err, err))
    rmse = math.sqrt(sq / len(hr))
    del x, y
    x_path.unlink()
    y_path.unlink()
    return rmse, factors_finite, predictions_finite


def parse_log(text, total_expected, solver):
    final = re.findall(r"Train RMSE: ([0-9.eE+-]+) \| Test RMSE: ([0-9.eE+-]+) \| compute=([0-9.eE+-]+) ms \| wall=([0-9.eE+-]+) ms", text)
    if not final:
        raise ValueError("missing final metrics")
    train, test, compute_ms, wall_ms = map(float, final[-1])
    counter = re.findall(
        r"FALLBACK_COUNTERS total=(\d+) fp16_success=(\d+) fp16_failures=(\d+) "
        r"fp32_attempts=(\d+) fp32_recoveries=(\d+) fp32_failures=(\d+)", text)
    stream = re.findall(r"SOLVER_STREAM_MS total=([0-9.eE+-]+)", text)
    if counter:
        total, ok16, fail16, attempts, recovered, fail32 = map(int, counter[-1])
    elif solver.startswith("zero"):
        failed = re.findall(r"non-finite solves zeroed = (\d+)", text)
        fail16 = int(failed[-1]) if failed else 0
        total, ok16, attempts, recovered, fail32 = total_expected, total_expected - fail16, 0, 0, 0
    else:
        total, ok16, fail16, attempts, recovered, fail32 = total_expected, 0, 0, 0, 0, 0
    return dict(train_rmse=train, test_rmse=test, compute_s=compute_ms / 1000.0,
                wall_s=wall_ms / 1000.0,
                solver_stream_s=(float(stream[-1]) / 1000.0 if stream else None),
                total_solves=total, fp16_successes=ok16, fp16_failures=fail16,
                fp32_attempts=attempts, fp32_recoveries=recovered,
                fp32_failures=fail32)


def main():
    build_all()
    header = read_header(TEST_DATA)
    validation = read_heldout(VALID_DATA)
    total_expected = (header["users"] + header["items"]) * 30
    rows = []
    for case, solver, lam, beta in CASES:
        for repetition in range(1, 4):
            tag = f"{case}_{solver}_rep{repetition}"
            log = LOGDIR / f"{tag}.txt"
            prefix = TMPDIR / tag
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CHECK_EVERY="30",
                       MAX_ITERS="30", FIXED_ITERS="1", MBETA=str(beta),
                       DUMP_FACTORS=str(prefix))
            cmd = [str(BINDIR / solver), str(TEST_DATA), str(lam)]
            snapshot = run_checked(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                                    "--format=csv,noheader"], capture_output=True, text=True).stdout
            metadata = {"command": cmd,
                        "environment": {key: env[key] for key in
                                        ("CUDA_VISIBLE_DEVICES", "CHECK_EVERY", "MAX_ITERS",
                                         "FIXED_ITERS", "MBETA", "DUMP_FACTORS")},
                        "gpu_snapshot_before": snapshot, "start_unix": time.time()}
            with log.open("w") as out:
                run_checked(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
            metadata["end_unix"] = time.time()
            (CMDDIR / f"{tag}.json").write_text(json.dumps(metadata, indent=2))
            text = log.read_text()
            metrics = parse_log(text, total_expected, solver)
            val_rmse, factors_finite, val_predictions_finite = score_factors(
                prefix, validation, header["users"], header["items"])
            row = dict(case=case, solver=solver, repetition=repetition, gpu=1,
                       K=96, lambda_value=lam, beta=beta, refinement_steps=(1 if "ir1" in solver else 0),
                       validation_rmse=val_rmse, factors_finite=factors_finite,
                       validation_predictions_finite=val_predictions_finite,
                       test_predictions_finite=math.isfinite(metrics["test_rmse"]),
                       log=str(log.relative_to(R)), command_record=str((CMDDIR / f"{tag}.json").relative_to(R)),
                       **metrics)
            if solver.startswith("fallback"):
                assert row["total_solves"] == row["fp16_successes"] + row["fp16_failures"]
                assert row["fp16_failures"] == row["fp32_attempts"]
                assert row["fp32_attempts"] == row["fp32_recoveries"] + row["fp32_failures"]
            assert factors_finite and val_predictions_finite and row["test_predictions_finite"]
            rows.append(row)
            fields = list(rows[0])
            with (R / "fp32_fallback_runs.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({k: row[k] for k in ("case", "solver", "repetition", "wall_s",
                                                   "test_rmse", "fp16_failures", "fp32_recoveries",
                                                   "fp32_failures")}), flush=True)
    (LOGDIR / "fp32_fallback.done").write_text("completed\n")


if __name__ == "__main__":
    main()

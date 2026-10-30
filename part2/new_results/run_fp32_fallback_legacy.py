#!/usr/bin/env python3
"""Repeat the documented original-split Netflix beta=.5 failure protocol."""
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

from run_fp32_fallback import R, ROOT, BINDIR, LOGDIR, CMDDIR, TMPDIR, read_header, read_heldout, score_factors, parse_log

DATA = Path("/home/pc/Desktop/2007080/netflix_ratings.bin")
SOLVERS = ("fallback_noref", "fallback_ir1", "zero_noref", "zero_ir1", "fp32")


def main():
    header = read_header(DATA)
    heldout = read_heldout(DATA)
    total_expected = (header["users"] + header["items"]) * 30
    rows = []
    for solver in SOLVERS:
        for repetition in range(1, 4):
            tag = f"legacy_original_{solver}_rep{repetition}"
            log = LOGDIR / f"{tag}.txt"
            prefix = TMPDIR / tag
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CHECK_EVERY="30",
                       MAX_ITERS="30", FIXED_ITERS="1", MBETA="0.5",
                       DUMP_FACTORS=str(prefix))
            cmd = [str(BINDIR / solver), str(DATA), ".048"]
            snapshot = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout
            metadata = {"command": cmd,
                        "environment": {key: env[key] for key in
                                        ("CUDA_VISIBLE_DEVICES", "CHECK_EVERY", "MAX_ITERS",
                                         "FIXED_ITERS", "MBETA", "DUMP_FACTORS")},
                        "gpu_snapshot_before": snapshot, "start_unix": time.time()}
            with log.open("w") as out:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=out,
                               stderr=subprocess.STDOUT, check=True)
            metadata["end_unix"] = time.time()
            command_record = CMDDIR / f"{tag}.json"
            command_record.write_text(json.dumps(metadata, indent=2))
            metrics = parse_log(log.read_text(), total_expected, solver)
            cpu_test_rmse, factors_finite, predictions_finite = score_factors(
                prefix, heldout, header["users"], header["items"])
            row = dict(case="legacy_original_split", solver=solver,
                       repetition=repetition, gpu=1, K=96, lambda_value=.048,
                       beta=.5, refinement_steps=(1 if "ir1" in solver else 0),
                       cpu_test_rmse=cpu_test_rmse, factors_finite=factors_finite,
                       test_predictions_finite=predictions_finite,
                       gpu_cpu_test_rmse_abs_diff=abs(cpu_test_rmse - metrics["test_rmse"]),
                       log=str(log.relative_to(R)),
                       command_record=str(command_record.relative_to(R)), **metrics)
            if solver.startswith("fallback"):
                assert row["total_solves"] == row["fp16_successes"] + row["fp16_failures"]
                assert row["fp16_failures"] == row["fp32_attempts"]
                assert row["fp32_attempts"] == row["fp32_recoveries"] + row["fp32_failures"]
            assert factors_finite and predictions_finite and math.isfinite(cpu_test_rmse)
            rows.append(row)
            with (R / "fp32_fallback_legacy_runs.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            print(json.dumps({k: row[k] for k in ("solver", "repetition", "wall_s",
                                                   "test_rmse", "fp16_failures",
                                                   "fp32_recoveries", "fp32_failures")}), flush=True)
    (LOGDIR / "fp32_fallback_legacy.done").write_text("completed\n")


if __name__ == "__main__":
    main()

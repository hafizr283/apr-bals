#!/usr/bin/env python3
"""Verify raw fallback runs and emit aggregate CSV, audit, and paper tables."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent
PRIMARY = R / "fp32_fallback_runs.csv"
LEGACY = R / "fp32_fallback_legacy_runs.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(frame, scope):
    rows = []
    for (case, solver), group in frame.groupby(["case", "solver"], sort=False):
        med = group.median(numeric_only=True)
        row = {
            "scope": scope, "case": case, "solver": solver, "runs": len(group),
            "wall_median_s": med.wall_s, "wall_min_s": group.wall_s.min(),
            "wall_max_s": group.wall_s.max(), "solver_stream_median_s": med.get("solver_stream_s", math.nan),
            "validation_rmse_median": med.get("validation_rmse", math.nan),
            "test_rmse_median": med.test_rmse, "test_rmse_min": group.test_rmse.min(),
            "test_rmse_max": group.test_rmse.max(), "total_solves": int(med.total_solves),
            "fp16_failures_median": med.fp16_failures,
            "fp16_failures_min": int(group.fp16_failures.min()),
            "fp16_failures_max": int(group.fp16_failures.max()),
            "fallback_attempts_median": med.fp32_attempts,
            "fp32_recoveries_median": med.fp32_recoveries,
            "fp32_failures_max": int(group.fp32_failures.max()),
            "fp16_failure_percent": 100.0 * med.fp16_failures / med.total_solves,
            "fp32_recovery_percent": (100.0 * med.fp32_recoveries / med.fp32_attempts
                                      if med.fp32_attempts else math.nan),
        }
        rows.append(row)
    return rows


def main():
    primary = pd.read_csv(PRIMARY)
    legacy = pd.read_csv(LEGACY)
    checks = []
    checks.append({"name": "primary_run_count", "passed": len(primary) == 24, "value": len(primary)})
    checks.append({"name": "legacy_run_count", "passed": len(legacy) == 15, "value": len(legacy)})
    for label, frame in (("primary", primary), ("legacy", legacy)):
        for _, row in frame.iterrows():
            log = R / row.log
            command = R / row.command_record
            text = log.read_text()
            metadata = json.loads(command.read_text())
            gpu_lines = metadata.get("gpu_snapshot_before", "").strip().splitlines()
            gpu1_idle = any(line.startswith("1,") and ", 0 %" in line for line in gpu_lines)
            ok = log.exists() and command.exists() and "=== APR-BALS final ===" in text
            ok &= gpu1_idle
            ok &= bool(row.factors_finite and row.test_predictions_finite)
            if "validation_predictions_finite" in frame:
                ok &= bool(row.validation_predictions_finite)
            if row.solver.startswith("fallback"):
                ok &= row.total_solves == row.fp16_successes + row.fp16_failures
                ok &= row.fp16_failures == row.fp32_attempts
                ok &= row.fp32_attempts == row.fp32_recoveries + row.fp32_failures
                match = re.findall(r"FALLBACK_COUNTERS total=(\d+) fp16_success=(\d+) fp16_failures=(\d+) fp32_attempts=(\d+) fp32_recoveries=(\d+) fp32_failures=(\d+)", text)
                ok &= bool(match and tuple(map(int, match[-1])) ==
                           tuple(int(row[k]) for k in ("total_solves", "fp16_successes", "fp16_failures",
                                                       "fp32_attempts", "fp32_recoveries", "fp32_failures")))
            checks.append({"name": f"{label}:{row.case}:{row.solver}:rep{row.repetition}",
                           "passed": bool(ok), "gpu1_idle_before": gpu1_idle, "log": row.log})

    summary = summarize(primary, "new_70_10_20_split") + summarize(legacy, "legacy_original_split")
    lookup = {(r["scope"], r["case"], r["solver"]): r for r in summary}
    for scope, case in (("new_70_10_20_split", "stable_selected"),
                        ("new_70_10_20_split", "hard_beta05"),
                        ("legacy_original_split", "legacy_original_split")):
        for suffix in ("noref", "ir1"):
            a = lookup.get((scope, case, f"fallback_{suffix}"))
            b = lookup.get((scope, case, f"zero_{suffix}"))
            if a and b:
                a["wall_overhead_vs_zero_percent"] = 100.0 * (a["wall_median_s"] / b["wall_median_s"] - 1.0)
    fields = list(summary[0])
    if "wall_overhead_vs_zero_percent" not in fields:
        fields.append("wall_overhead_vs_zero_percent")
    with (R / "fp32_fallback_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)

    selected = []
    order = [
        ("new_70_10_20_split", "stable_selected", "fallback_noref", "New split, stable", "FP16 + FP32 fallback", "0"),
        ("new_70_10_20_split", "hard_beta05", "zero_noref", "New split, beta=.5", "FP16 zero, no IR", "0"),
        ("new_70_10_20_split", "hard_beta05", "fallback_noref", "New split, beta=.5", "FP16 fallback, no IR", "0"),
        ("new_70_10_20_split", "hard_beta05", "zero_ir1", "New split, beta=.5", "FP16 zero, IR1", "1"),
        ("new_70_10_20_split", "hard_beta05", "fallback_ir1", "New split, beta=.5", "FP16 fallback, IR1", "1"),
        ("new_70_10_20_split", "hard_beta05", "fp32", "New split, beta=.5", "FP32", "--"),
        ("legacy_original_split", "legacy_original_split", "zero_noref", "Original split, beta=.5", "FP16 zero, no IR", "0"),
        ("legacy_original_split", "legacy_original_split", "fallback_noref", "Original split, beta=.5", "FP16 fallback, no IR", "0"),
        ("legacy_original_split", "legacy_original_split", "zero_ir1", "Original split, beta=.5", "FP16 zero, IR1", "1"),
        ("legacy_original_split", "legacy_original_split", "fallback_ir1", "Original split, beta=.5", "FP16 fallback, IR1", "1"),
        ("legacy_original_split", "legacy_original_split", "fp32", "Original split, beta=.5", "FP32", "--"),
    ]
    for scope, case, solver, label, method, ir in order:
        r = lookup[(scope, case, solver)]
        selected.append((label, method, ir, r))

    tex = [r"\begin{table*}[t]", r"\centering", r"\caption{FP32 solver fallback at Netflix $K=96$ (median of three GPU-1 runs, 30 sweeps).}",
           r"\label{tab:fp32-fallback}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Protocol & Solver & FP16 fail (\%) & FP32 rec. & FP32 fail & Val. RMSE & Test RMSE & Wall (s) \\", r"\midrule"]
    for label, method, ir, r in selected:
        validation = "--" if math.isnan(r["validation_rmse_median"]) else f'{r["validation_rmse_median"]:.6f}'
        solver_label = method.replace("FP32 fallback", "FP32 fb")
        protocol_label = label.replace("beta", r"$\beta$")
        is_fallback = "fallback" in method.lower()
        recovered = f'{r["fp32_recoveries_median"]:.0f}' if is_fallback else "--"
        fp32_failed = str(r["fp32_failures_max"]) if is_fallback else "--"
        tex.append(f'{protocol_label} & {solver_label} & {r["fp16_failures_median"]:.0f} ({r["fp16_failure_percent"]:.6f}) & {recovered} & {fp32_failed} & {validation} & {r["test_rmse_median"]:.6f} & {r["wall_median_s"]:.5f} \\\\')
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (R / "paper_tables/fp32_fallback.tex").write_text("\n".join(tex) + "\n")

    audit = {"all_passed": all(c["passed"] for c in checks), "checks": checks,
             "source_sha256": {str(p.relative_to(R)): sha256(p) for p in
                               (R / "fallback_src/common.cuh", R / "fallback_src/cholesky_kernels.cuh",
                                R / "fallback_src/justapr.cu", R / "run_fp32_fallback.py",
                                R / "run_fp32_fallback_legacy.py")},
             "csv_sha256": {PRIMARY.name: sha256(PRIMARY), LEGACY.name: sha256(LEGACY)}}
    deliverables = (R / "fp32_fallback_summary.csv", R / "paper_tables/fp32_fallback.tex")
    audit["protocol_document"] = "docs/reproduction.md#fp32-fallback-protocol"
    audit["deliverable_sha256"] = {str(p.relative_to(R)): sha256(p) for p in deliverables}
    (R / "fp32_fallback_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["all_passed"]:
        raise SystemExit("fallback audit failed")
    print(json.dumps({"runs_verified": len(primary) + len(legacy), "all_passed": True}))


if __name__ == "__main__":
    main()

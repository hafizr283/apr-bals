#!/usr/bin/env python3
"""Read-only, offline consistency checks for the curated paper evidence.

Only the Python standard library is used. No experiment module is imported.
Fingerprints normalize CRLF/CR to LF so Git checkouts are portable.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "part2/new_results/"
DATASETS = ("ml10m", "ml20m", "ml32m", "netflix")
NAMES = dict(zip(DATASETS, ("ML-10M", "ML-20M", "ML-32M", "Netflix")))
RANKS = (16, 32, 48, 64, 96)
ALS_RANKS = dict(zip(RANKS, (20, 30, 50, 60, 100)))
COMMIT = "e798aa242dc5026247426bd96ca6055be26b2354"
GITHUB = "https://github.com/hafizr283/apr-bals/blob/" + COMMIT + "/"


class EvidenceError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read(root, path):
    p = root / path
    require(p.is_file(), "missing required source: " + path)
    return p.read_text(encoding="utf-8-sig")


def fingerprint(text):
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()


def table(root, name):
    path = RESULTS + name
    lines = read(root, path).splitlines()
    reader = csv.DictReader(lines)
    rows, locations = [], []
    for row in reader:
        rows.append(row)
        locations.append(reader.line_num)
    return rows, locations, reader.fieldnames


def close(a, b, context, tolerance=1e-9):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=tolerance),
            "numerical mismatch: " + context + f" ({a} != {b})")


def link(label, path, line):
    return f"[{label}]({GITHUB}{path}?plain=1#L{line})"


def external_curve(text, method):
    if method == "ccd":
        pattern = r"iter (\d+) time [\d.]+ RMSE ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)"
    elif method == "sgd":
        pattern = r"Iter (\d+) \| Validation RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)"
    else:
        pattern = r"Iter\s+(\d+) \| Train RMSE: [\d.eE+-]+ \| Test RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)"
    return [(int(i), float(r), float(t) / 1000) for i, r, t in re.findall(pattern, text)]


def sequence(curve, count, context):
    require([r[0] for r in curve] == list(range(1, count + 1)), "sweep sequence: " + context)
    require(all(all(math.isfinite(v) for v in r) for r in curve), "non-finite curve: " + context)
    require(all(r[-1] >= 0 for r in curve) and
            all(b[-1] >= a[-1] for a, b in zip(curve, curve[1:])), "wall sequence: " + context)


def calculate(root, manifest):
    rows, lines, _ = table(root, "external_time_to_target.csv")
    idx = {(r["dataset"], r["method"], int(r["K"])): r for r in rows}
    loc = {k: n for k, n in zip(idx, lines)}
    expected = {(d, m, k) for d in DATASETS for m in ("apr", "als", "ccd", "bals") for k in RANKS}
    expected |= {(d, "sgd", 128) for d in DATASETS}
    require(len(rows) == len(idx) and set(idx) == expected, "external CSV key set or duplicate key")
    ratios = {}
    accuracy = {}
    raw_count = 0
    for key, row in idx.items():
        d, m, k = key
        close(row["target"], manifest["targets"][d], str(key) + " target")
        require(row["reached"] == "True" and float(row["seconds_to_target"]) > 0, "target not reached: " + str(key))
        count = 40 if m == "sgd" else 30
        require(int(row["observed_sweeps"]) == count and 1 <= int(row["first_sweep"]) <= count,
                "external sweep count: " + str(key))
        raw = manifest["external_raw"]["|".join(map(str, key))]
        path = raw["path"]
        if raw["status"] == "missing":
            require(not (root / path).exists(), "declared gap is now present; review manifest: " + path)
            continue
        c = external_curve(read(root, path), m)
        sequence(c, count, path)
        hits = [r for r in c if r[1] <= float(row["target"])]
        require(bool(hits), "raw target not reached: " + path)
        close(hits[0][0], row["first_sweep"], path + " crossing sweep")
        close(hits[0][2], row["seconds_to_target"], path + " crossing wall")
        close(c[-1][1], row["final_validation_rmse"], path + " final RMSE")
        raw_count += 1
    for d in DATASETS:
        for m in ("als", "ccd", "bals"):
            for k in RANKS:
                ratios[d, m, k] = float(idx[d, m, k]["seconds_to_target"]) / float(idx[d, "apr", k]["seconds_to_target"])
        for m in ("als", "ccd"):
            a, b = (float(idx[d, x, 96]["final_validation_rmse"]) for x in ("apr", m))
            accuracy[d, m] = 100 * (a - b) / b
    require(raw_count == manifest["external_raw_available"], "external raw coverage drift")
    mom, mom_lines, _ = table(root, "momentum_crossings.csv")
    mi = {}
    for row, line in zip(mom, mom_lines):
        key = row["dataset"], int(row["K"]), row["beta"]
        path = RESULTS + f"momentum/{key[0]}_ship_k{key[1]}_beta{key[2]}.txt"
        c = [(int(i), float(tr), float(te), float(t) / 1000) for i, tr, te, t in re.findall(
            r"Iter\s+(\d+) \| Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)", read(root, path))]
        sequence(c, 30, path)
        for window in (1, 5):
            hits = [b for a, b in zip(c, c[window:]) if abs(a[1] - b[1]) < .001]
            require(bool(hits), "missing momentum crossing: " + path)
            h = hits[0]
            close(h[0], row[f"first_delta{window}_sweep"], path + " crossing sweep")
            close(h[3], row[f"first_delta{window}_wall_s"], path + " crossing wall")
            if window == 1:
                close(h[2], row["first_delta1_test"], path + " crossing RMSE")
        mi[key] = (row, line)
    pairs = sorted((d, k) for d, k, b in mi if b == "0")
    require(len(mi) == len(mom) == 2 * len(pairs) and all((d, k, "0.3") in mi for d, k in pairs),
            "momentum duplicate key or incomplete pair")
    mr = {(d, k): float(mi[d, k, "0"][0]["first_delta1_wall_s"]) / float(mi[d, k, "0.3"][0]["first_delta1_wall_s"]) for d, k in pairs}
    selections = {}
    for d in DATASETS:
        p = RESULTS + f"external/{d}_ccd_selection.json"
        selection = json.loads(read(root, p))
        best = min(selection["candidates"])
        close(selection["selected_lambda"], best[1], d + " CCD selection minimum")
        close(selection["selected_lambda"], manifest["ccd_lambdas"][d], d + " CCD selected lambda")
        close(best[0], idx[d, "ccd", 96]["final_validation_rmse"], d + " CCD selected RMSE")
        selections[d] = selection["selected_lambda"]
    metrics = {}
    for m in ("als", "ccd", "bals"):
        values = [v for (d, method, k), v in ratios.items() if method == m]
        metrics[m + "_min"], metrics[m + "_max"] = min(values), max(values)
        for d in DATASETS:
            metrics[d + "_" + m + "_mean"] = statistics.mean(ratios[d, m, k] for k in RANKS)
    for m in ("als", "ccd"):
        metrics[m + "_accuracy_max_percent"] = max(abs(accuracy[d, m]) for d in DATASETS)
        for d in DATASETS:
            metrics[d + "_" + m + "_accuracy_percent"] = accuracy[d, m]
    for label, values in (
        ("netflix_als", [ratios["netflix", "als", k] for k in RANKS]),
        ("k96_als", [ratios[d, "als", 96] for d in DATASETS]),
    ):
        metrics[label + "_min"], metrics[label + "_max"] = min(values), max(values)
    metrics["momentum_min"], metrics["momentum_max"] = min(mr.values()), max(mr.values())
    require(metrics["als_accuracy_max_percent"] < .1 and metrics["ccd_accuracy_max_percent"] < .2, "K96 RMSE bound")
    for key, value in manifest.get("metrics", {}).items():
        close(metrics[key], value, "documented metric " + key)
    return dict(index=idx, lines=loc, ratios=ratios, accuracy=accuracy, momentum=mi,
                momentum_ratios=mr, selections=selections, metrics=metrics, raw_count=raw_count)


def check_other_raw(root, manifest):
    """Reparse final summaries for run tables; other tables retain stated scope."""
    for name, records in manifest["run_raw"].items():
        rows = table(root, name)[0]
        require(len(rows) == len(records), "run raw inventory drift: " + name)
        for row, spec in zip(rows, records):
            path = spec["path"]
            if spec["status"] == "missing":
                require(not (root / path).exists(), "declared run gap now present: " + path)
                continue
            text = read(root, path)
            final = re.findall(r"Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+) \| compute=([\d.]+) ms \| wall=([\d.]+) ms", text)
            require(bool(final), "missing final summary: " + path)
            tr, te, comp, wall = map(float, final[-1])
            values = dict(train=tr, test=te, compute_s=comp/1000, wall_s=wall/1000)
            values["gram_s"] = sum(float(v) for v in re.findall(r"(?:User|Item) LHS\+RHS:\s*([\d.]+) ms total", text)) / 1000
            failures = re.findall(r"non-finite solves zeroed = (\d+)", text)
            values["zeroed_solves"] = int(failures[-1]) if failures else 0
            for key, value in values.items():
                close(value, row[key], path + " " + key)
            if "sweeps" in row:
                close(len(external_curve(text, "apr")), row["sweeps"], path + " sweeps")
    t4 = read(root, manifest["historical"]["t4"])
    walls = list(map(float, re.findall(r"Training Complete in ([\d.]+) seconds", t4)))
    require(len(walls) == 2 and f"{walls[0]/walls[1]:.2f}" == "10.78", "historical T4 wall ratio")
    require("0.0480" in t4 and "FP32 tiled" in t4, "historical T4 configuration")
    hist = manifest["historical"]["variance"]
    require(len(hist) == 100, "historical variance run coverage")
    for path in hist:
        text = read(root, path)
        values = re.findall(r"Training Complete in ([\d.]+) seconds", text)
        require(len(values) == 2 and all(float(v) > 0 for v in values), "historical run incomplete: " + path)
    summary = read(root, manifest["historical"]["summary"])
    require("20 configurations x 5 runs = 100 runs" in summary and "max 0.44%, mean 0.14%" in summary,
            "historical variance summary drift")


def rendered_blocks(root, manifest, data):
    idx, loc, ratios = data["index"], data["lines"], data["ratios"]
    blocks = {}
    def cell(d, m, k, field):
        return link(idx[d, m, k][field], RESULTS + "external_time_to_target.csv", loc[d, m, k])
    speed = ["| Dataset | APR K / ALS F | APR seconds | ALS seconds | ALS / APR | CCD seconds | CCD / APR | BALS seconds | BALS / APR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for k in RANKS:
            speed.append(f"| {NAMES[d]} | {k} / {ALS_RANKS[k]} | {cell(d, 'apr', k, 'seconds_to_target')} | {cell(d, 'als', k, 'seconds_to_target')} | {ratios[d,'als',k]:.9f} | {cell(d, 'ccd', k, 'seconds_to_target')} | {ratios[d,'ccd',k]:.9f} | {cell(d, 'bals', k, 'seconds_to_target')} | {ratios[d,'bals',k]:.9f} |")
    blocks["external"] = "\n".join(speed)
    means = ["| Dataset | Mean ALS / APR | Mean CCD / APR | Mean BALS / APR |", "|---|---:|---:|---:|"]
    for d in DATASETS:
        means.append(f"| {NAMES[d]} | " + " | ".join(f"{statistics.mean(ratios[d,m,k] for k in RANKS):.9f}" for m in ("als", "ccd", "bals")) + " |")
    blocks["means"] = "\n".join(means)
    sgd = ["| Dataset | APR K=96 seconds | SGD K=128 seconds | SGD / APR | SGD final validation RMSE |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        ratio = float(idx[d,'sgd',128]['seconds_to_target']) / float(idx[d,'apr',96]['seconds_to_target'])
        sgd.append(f"| {NAMES[d]} | {cell(d,'apr',96,'seconds_to_target')} | {cell(d,'sgd',128,'seconds_to_target')} | {ratio:.9f} | {cell(d,'sgd',128,'final_validation_rmse')} |")
    blocks["sgd"] = "\n".join(sgd)
    accuracy = ["| Dataset | APR K=96 RMSE | ALS F=100 RMSE | APR vs ALS (%) | CCD K=96 RMSE | APR vs CCD (%) |", "|---|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        accuracy.append(f"| {NAMES[d]} | {cell(d,'apr',96,'final_validation_rmse')} | {cell(d,'als',96,'final_validation_rmse')} | {data['accuracy'][d,'als']:+.9f} | {cell(d,'ccd',96,'final_validation_rmse')} | {data['accuracy'][d,'ccd']:+.9f} |")
    blocks["accuracy"] = "\n".join(accuracy)
    sels = ["| Dataset | Selected CCD lambda |", "|---|---:|"]
    for d in DATASETS:
        sels.append(f"| {NAMES[d]} | {link(str(data['selections'][d]), RESULTS + f'external/{d}_ccd_selection.json', 2)} |")
    blocks["lambdas"] = "\n".join(sels)
    mom = ["| Dataset | K | Beta 0 crossing sweep | Beta 0.3 crossing sweep | Wall ratio | Five-sweep window crossings (0 / 0.3) |", "|---|---:|---:|---:|---:|---:|"]
    for (d, k), ratio in data["momentum_ratios"].items():
        a, la = data["momentum"][d, k, "0"]
        b, lb = data["momentum"][d, k, "0.3"]
        mom.append(f"| {NAMES[d]} | {k} | {link(a['first_delta1_sweep'], RESULTS+'momentum_crossings.csv', la)} | {link(b['first_delta1_sweep'], RESULTS+'momentum_crossings.csv', lb)} | {ratio:.9f} | {a['first_delta5_sweep']} / {b['first_delta5_sweep']} |")
    blocks["momentum"] = "\n".join(mom)
    coverage = ["| Evidence table | Available records | Planned records | Protocol | Verification scope |", "|---|---:|---:|---|---|"]
    for name, spec in manifest["tables"].items():
        count = len(table(root, name)[0])
        coverage.append(f"| {link(name, RESULTS+name, 1)} | {count:,} | {spec['planned'] or 'Not a run total'} | {spec['protocol']} | {spec['scope']} |")
    blocks["coverage"] = "\n".join(coverage)
    gaps = ["| Dataset | Method | APR comparison K | Expected missing raw filename |",
            "|---|---|---:|---|"]
    for key, spec in sorted(manifest["external_raw"].items()):
        if spec["status"] == "missing":
            d, method, k = key.split("|")
            gaps.append(f"| {NAMES[d]} | {method} | {k} | {Path(spec['path']).name} |")
    blocks["gaps"] = "\n".join(gaps)
    m = data["metrics"]
    blocks["headlines"] = "\n".join([
        "| Comparison | Protocol and ranks | Result | Evidence |",
        "|---|---|---|---|",
        f"| APR vs cuMF_als | October warm-validation time to target; 20 documented K/F pairings | **{m['als_min']:.1f}–{m['als_max']:.1f}×** | [Rows and calculation](docs/evidence.md#external-time-to-target) |",
        f"| APR vs cuMF_CCD | October warm-validation time to target; 20 matched-rank comparisons | **{m['ccd_min']:.1f}–{m['ccd_max']:.1f}×** | [Rows and calculation](docs/evidence.md#external-time-to-target) |",
        "| Final warm-validation RMSE | APR K=96; ALS F=100; CCD K=96 | Absolute relative difference **<0.1% vs ALS**, **<0.2% vs CCD** | [Exact RMSEs and denominators](docs/evidence.md#final-validation-accuracy) |",
    ])
    return blocks


def heading_ids(text):
    ids, seen = set(), {}
    for title in re.findall(r"^#{1,6}\s+(.+)$", text, re.M):
        base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        n = seen.get(base, 0)
        ids.add(base + (f"-{n}" if n else ""))
        seen[base] = n + 1
    return ids


def validate_links(root, manifest):
    checked = set()
    for doc in manifest["documents"]:
        text = read(root, doc)
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
            url = urlsplit(target)
            if url.scheme:
                if target.startswith("https://github.com/hafizr283/apr-bals/blob/"):
                    require(target.startswith(GITHUB), "unpinned or unexpected GitHub commit: " + target)
                    require(target in manifest["pinned_links"], "uncurated pinned reference: " + target)
                    spec = manifest["pinned_links"][target]
                    match = re.fullmatch(r"L(\d+)(?:-L(\d+))?", url.fragment)
                    require(bool(match), "invalid GitHub line anchor: " + target)
                    start, end = int(match[1]), int(match[2] or match[1])
                    path = unquote(url.path.split("/blob/" + COMMIT + "/", 1)[1])
                    lines = read(root, path).splitlines()
                    require(1 <= start <= end <= len(lines), "out-of-range line anchor: " + target)
                    require(lines[start-1:end] == spec["lines"], "pinned line content drift: " + target)
                    checked.add(target)
                continue
            relative = unquote(url.path)
            path = ((root / doc).parent / relative).resolve() if relative else root / doc
            require(path.is_relative_to(root.resolve()), "reference escapes repository: " + target)
            require(path.exists(), "broken relative reference: " + doc + " -> " + target)
            if url.fragment and path.suffix.lower() == ".md":
                require(unquote(url.fragment) in heading_ids(path.read_text(encoding="utf-8")), "broken heading: " + target)
    require(checked == set(manifest["pinned_links"]), "unused pinned reference in manifest")
    return len(checked)


def verify(root=ROOT):
    manifest = json.loads(read(root, "docs/evidence_manifest.json"))
    require(manifest["version"] == 1 and manifest["source_commit"] == COMMIT, "unsupported manifest")
    for path in manifest["declared_missing_sources"]:
        require(not (root / path).exists(), "declared source gap now present; review manifest: " + path)
    actual_tables = {p.name for p in (root / RESULTS).glob("*.csv")}
    require(actual_tables == set(manifest["tables"]), "unindexed or missing October CSV")
    for name, spec in manifest["tables"].items():
        rows, _, headers = table(root, name)
        require(headers == spec["columns"], "CSV schema drift: " + name)
        keys = []
        for r in rows:
            require(None not in r and all(v is not None for v in r.values()), "malformed row: " + name)
            keys.append([r[k] for k in spec["keys"]])
            for col in spec["numeric"]:
                require(r[col] == "" or math.isfinite(float(r[col])), "non-finite CSV value: " + name)
        require(len(keys) == len({tuple(k) for k in keys}), "duplicate CSV key: " + name)
        require(sorted(keys) == spec["expected_keys"], "coverage/key drift: " + name)
    data = calculate(root, manifest)
    check_other_raw(root, manifest)
    blocks = rendered_blocks(root, manifest, data)
    for doc, ids in manifest["generated_blocks"].items():
        text = read(root, doc)
        for block in ids:
            pattern = "<!-- evidence:" + block + ":start -->\n(.*?)\n<!-- evidence:" + block + ":end -->"
            matches = re.findall(pattern, text, re.S)
            require(matches == [blocks[block]], "documentation numerical drift: " + doc + " / " + block)
    anchors = validate_links(root, manifest)
    for path, sha in manifest["sources"].items():
        require(fingerprint(read(root, path)) == sha, "source fingerprint changed: " + path)
    for path, sha in manifest["documents"].items():
        require(fingerprint(read(root, path)) == sha, "documentation drift: " + path)
    for path in manifest["small_evidence"]:
        require(path in manifest["sources"], "unverified small evidence: " + path)
        require((root / path).stat().st_size < 100000, "large artifact in small evidence: " + path)
    require(len(manifest["small_evidence"]) == 99, "small evidence inventory drift")
    require(150 <= len(read(root, "README.md").splitlines()) <= 220, "root README line budget")
    warnings = ["declared missing external raw log: " + x["path"] for x in manifest["external_raw"].values() if x["status"] == "missing"]
    warnings += manifest["warnings"]
    warnings += ["declared missing reproduction input: " + p for p in manifest["declared_missing_sources"]]
    return warnings, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True, help="verify archived evidence without writing files")
    parser.parse_args()
    try:
        warnings, anchors = verify()
    except (EvidenceError, ValueError, KeyError, OSError, TypeError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 1
    for warning in warnings:
        print("WARNING: " + warning)
    print(f"PASS: archived evidence consistent; 84 external records, 59 raw trajectories, 6 momentum pairs; {anchors} pinned links checked.")
    print("No models trained; no result files modified. Declared gaps remain; this is not a full reproduction check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

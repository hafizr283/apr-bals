"""Failure tests operate on a small, isolated evidence-only checkout."""
import hashlib
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence", ROOT / "scripts/verify_paper_evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        # Respect the workstation's storage policy; Linux CI uses its normal temp.
        temp_base = Path("F:/CodexData/temp") if os.name == "nt" and Path("F:/CodexData/temp").is_dir() else None
        self.temp = tempfile.TemporaryDirectory(prefix="apr-evidence-test-", dir=temp_base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = json.loads((ROOT / "docs/evidence_manifest.json").read_text(encoding="utf-8"))
        files = set(self.manifest["sources"]) | set(self.manifest["documents"]) | set(self.manifest.get("binary_sources", {}))
        files |= {"docs/evidence_manifest.json", "scripts/verify_paper_evidence.py"}
        for name in files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for directory in self.manifest["required_directories"]:
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def check(self):
        return subprocess.run([sys.executable, "scripts/verify_paper_evidence.py", "--check"],
                              cwd=self.root, text=True, capture_output=True)

    def edit(self, name, change):
        p = self.root / name
        p.write_text(change(p.read_text(encoding="utf-8")), encoding="utf-8")

    def test_declared_gaps_pass_and_check_is_read_only(self):
        def hashes():
            return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.root.rglob("*") if p.is_file()}
        before = hashes()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("WARNING: declared missing external raw log:"), 0)
        self.assertIn("84 raw trajectories, 20 momentum pairs", result.stdout)
        self.assertIn("PASS: archived evidence consistent", result.stdout)
        self.assertEqual(before, hashes())

    def test_changed_numerical_value_fails(self):
        self.edit("part2/new_results/external_time_to_target.csv",
                  lambda text: text.replace("0.065774818", "9.065774818", 1))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("numerical mismatch", result.stderr)

    def test_declared_missing_log_warns_but_unexpected_missing_fails(self):
        record = next(r for r in self.manifest["external_raw"].values() if r["status"] == "available")
        record["status"] = "missing"
        (self.root / record["path"]).unlink()
        self.manifest["sources"].pop(record["path"])
        self.manifest["external_raw_available"] -= 1
        # The fixture is a deliberate partial archive, with its ledger regenerated.
        data = evidence.calculate(self.root, self.manifest)
        blocks = evidence.rendered_blocks(self.root, self.manifest, data)
        import re
        p = self.root / "docs/evidence.md"
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"(<!-- evidence:gaps:start -->\n).*?(\n<!-- evidence:gaps:end -->)",
                   lambda m: m[1] + blocks["gaps"] + m[2], t, flags=re.S)
        p.write_text(t, encoding="utf-8")
        self.manifest["documents"]["docs/evidence.md"] = evidence.fingerprint(t)
        (self.root / "docs/evidence_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("WARNING: declared missing external raw log:"), 1)

    def test_recovered_target_manifest_mismatch_fails(self):
        self.edit("part2/new_results/external_targets.json", lambda s: s.replace('0.80', '0.81', 1))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external target manifest mismatch", result.stderr)

    def test_duplicate_csv_key_fails(self):
        self.edit("part2/new_results/external_time_to_target.csv",
                  lambda text: text + text.splitlines()[1] + "\n")
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate CSV key", result.stderr)

    def test_missing_required_source_fails(self):
        (self.root / "part2/new_results/external/ml10m_ccd_selection.json").unlink()
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required source", result.stderr)

    def test_unexpected_missing_raw_log_fails(self):
        record = next(r for r in self.manifest["external_raw"].values() if r["status"] == "available")
        (self.root / record["path"]).unlink()
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required source", result.stderr)

    def test_documented_number_drift_fails(self):
        self.edit("README.md", lambda text: text.replace("2.6–5.2", "2.6–5.9", 1))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("documentation numerical drift", result.stderr)

    def test_broken_relative_link_fails(self):
        self.edit("README.md", lambda text: text.replace("(results/README.md)", "(results/missing.md)", 1))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broken relative reference", result.stderr)

    def test_bad_github_anchor_fails(self):
        self.edit("docs/evidence.md", lambda text: text.replace("#L42-L60)", "#L999999)", 1))
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference", result.stderr)

    def test_numerical_replay_does_not_only_check_hashes(self):
        # Change a crossing and also bless its CSV hash: raw replay must still fail.
        name = "part2/new_results/momentum_crossings.csv"
        self.edit(name, lambda text: text.replace("0.645083862", "9.645083862", 1))
        self.manifest["sources"][name] = evidence.fingerprint((self.root / name).read_text(encoding="utf-8"))
        (self.root / "docs/evidence_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("numerical mismatch", result.stderr)

    def test_public_markdown_inventory(self):
        # Inspect only Git-visible files; local manuscript/scratch directories may be ignored.
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        actual = {p for p in result.stdout.split("\0") if p.endswith(".md") and (ROOT / p).is_file()}
        self.assertEqual(actual, set(self.manifest["public_markdown"]))

    def test_summarizers_preserve_numerical_records_without_markdown(self):
        paths = ("part2/new_results/summarize.py", "part2/new_results/summarize_external.py")
        names = [n for n in self.manifest["tables"] if not n.startswith("fp32_fallback")]
        def records(name):
            with (self.root / "part2/new_results" / name).open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            keys = self.manifest["tables"][name]["keys"]
            return {tuple(row[k] for k in keys): row for row in rows}
        before = {name: records(name) for name in names}
        markdown = {str(p.relative_to(self.root)) for p in self.root.rglob("*.md")}
        for path in paths:
            result = subprocess.run([sys.executable, path], cwd=self.root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(markdown, {str(p.relative_to(self.root)) for p in self.root.rglob("*.md")})
        for name, expected in before.items():
            actual = records(name)
            self.assertEqual(expected.keys(), actual.keys(), name)
            numeric = self.manifest["tables"][name]["numeric"]
            for key, row in expected.items():
                for column, value in row.items():
                    got = actual[key][column]
                    if column in numeric and value and got:
                        self.assertTrue(math.isclose(float(value), float(got), rel_tol=1e-10, abs_tol=1e-9),
                                        (name, key, column, value, got))
                    else:
                        self.assertEqual(value, got, (name, key, column))


if __name__ == "__main__":
    unittest.main()

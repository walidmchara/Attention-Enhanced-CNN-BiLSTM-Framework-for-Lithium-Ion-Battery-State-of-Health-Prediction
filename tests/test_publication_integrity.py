"""Pre-publication integrity checks for recorded NASA SOH experiments.

These checks validate published artifacts and documentation. They do not
establish methodological validity, independent forecasting performance, or
bitwise replay of the historical TensorFlow training run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "B0018": ("B0005", "B0006", "B0007", "B0018", 132),
    "B0030": ("B0029", "B0032", "B0031", "B0030", 40),
}
FIGURES = (
    "training_convergence.png", "soh_prediction_{cell}.png",
    "parity_{cell}.svg", "signed_residuals_{cell}.svg",
    "absolute_error_distribution_{cell}.svg",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RecordedResultsIntegrity(unittest.TestCase):
    def test_published_artifacts_and_metrics(self):
        for cell, (*train_cells, target, ncycles) in RUNS.items():
            with self.subTest(cell=cell):
                root = ROOT / "results" / f"legacy_mat_{cell.lower()}_e30_seed42"
                self.assertTrue(root.is_dir())
                metrics = load_json(root / "metrics.json")
                config = load_json(root / "config.json")
                manifest = load_json(root / "manifest.json")
                prediction_rows = load_csv(root / "test_predictions.csv")
                history = load_csv(root / "training_history.csv")

                self.assertEqual(metrics["experiment"], cell)
                self.assertEqual(metrics["epochs_completed"], 30)
                self.assertEqual(metrics["test"]["discharge_cycles"], ncycles)
                self.assertEqual(config["training_cells"], train_cells)
                self.assertEqual(config["test_cell"], target)
                self.assertEqual(config["epochs"], 30)
                self.assertEqual(config["batch_size"], 32)
                self.assertEqual(config["test_size"], 0.2)
                self.assertEqual(config["train_test_random_state"], 0)
                self.assertEqual(config["seed_added_for_reproducibility"], 42)
                self.assertIn("capacity is input feature", " ".join(config["historical_leakage"]))
                self.assertEqual(len(config["mat_sha256"]), 4)
                for sha in config["mat_sha256"].values():
                    self.assertEqual(len(sha), 64)
                    self.assertTrue(all(c in "0123456789abcdef" for c in sha))

                self.assertEqual(len(prediction_rows), ncycles)
                self.assertEqual(len(history), 30)
                self.assertEqual(
                    [int(r["cycle"]) for r in prediction_rows],
                    list(range(1, ncycles + 1)),
                )
                self.assertEqual(
                    [int(row["epoch"]) for row in history],
                    list(range(1, 31)),
                )
                for row in history:
                    self.assertTrue(math.isfinite(float(row["loss"])))
                    self.assertTrue(math.isfinite(float(row["val_loss"])))

                actual = [float(row["SoH"]) for row in prediction_rows]
                predicted = [float(row["NewSoH"]) for row in prediction_rows]
                self.assertTrue(all(math.isfinite(x) and x > 0 for x in actual))
                self.assertTrue(all(math.isfinite(x) for x in predicted))
                differences = [a - b for a, b in zip(actual, predicted)]
                mse = sum(x*x for x in differences) / ncycles
                mae = sum(abs(x) for x in differences) / ncycles
                rmse = math.sqrt(mse)
                mape = sum(abs((a-b)/a) for a,b in zip(actual,predicted)) / ncycles * 100
                mean_actual = sum(actual) / ncycles
                sst = sum((x-mean_actual)**2 for x in actual)
                self.assertGreater(sst, 0)
                r2 = 1 - sum(x*x for x in differences) / sst
                expected = metrics["test"]
                for key, value in (
                    ("mse", mse), ("mae", mae), ("rmse", rmse),
                    ("mape_percent", mape), ("r2_additional", r2),
                ):
                    self.assertTrue(
                        math.isclose(float(expected[key]), value,
                                     abs_tol=5e-6, rel_tol=1e-4),
                        msg=f"{cell} {key}: CSV-derived={value}, metrics.json={expected[key]}",
                    )

                # Original Colab publication manifest covers the six initial
                # artifacts. Subsequent documented diagnostic figures are
                # intentionally outside that original provenance manifest.
                self.assertEqual(manifest["run_id"], root.name)
                self.assertEqual(len(manifest["files"]), 6)
                self.assertEqual(manifest.get("skipped_files"), [])
                for entry in manifest["files"]:
                    path = root / entry["name"]
                    self.assertTrue(path.is_file(), msg=str(path))
                    self.assertEqual(path.stat().st_size, entry["size_bytes"])
                    self.assertEqual(digest(path), entry["sha256"])

                report = (root / "README.md").read_text(encoding="utf-8").lower()
                self.assertIn("capacity", report)
                self.assertIn("leakage", report)
                for template in FIGURES:
                    fig = root / template.format(cell=cell)
                    self.assertTrue(fig.is_file(), msg=str(fig))
                    if fig.suffix == ".png":
                        self.assertEqual(fig.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    else:
                        tree = ET.parse(fig)
                        self.assertEqual(tree.getroot().tag,
                                         "{http://www.w3.org/2000/svg}svg")
                        xml = fig.read_text(encoding="utf-8").lower()
                        self.assertNotIn("<script", xml)
                        self.assertNotIn("<foreignobject", xml)
                        self.assertNotIn("javascript:", xml)
                        self.assertNotIn("https://", xml)

    def test_repository_presentation_and_private_artifacts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("b0018", readme)
        self.assertIn("b0030", readme)
        self.assertIn("capacity", readme)
        self.assertIn("leakage", readme)
        self.assertIn("original", readme)
        self.assertIn("validation", readme)

        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT,
            capture_output=True, check=True,
        ).stdout.decode("utf-8").split("\0")
        for name in filter(None, listed):
            path = Path(name)
            self.assertNotIn(path.suffix.lower(), {".mat", ".keras", ".h5", ".p12", ".pem"})
            self.assertNotIn(path.name.lower(), {".env", ".env.local", "id_rsa"})
            self.assertNotIn("raw", path.parts if path.parts and path.parts[0]=="data" else ())
        self.assertTrue((ROOT / ".gitignore").is_file())


if __name__ == "__main__":
    unittest.main()

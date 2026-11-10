"""Smoke test: NASA measured CSV splits and Keras model+optimizer recovery.

Not a scientific accuracy benchmark. Runs two very small CPU epochs only.
"""
import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_resumable import (
    FEATURES,
    EpochSnapshots,
    build_model,
    load_snapshot,
    prepare_data,
)


class RealDataAndRecoveryTest(unittest.TestCase):
    def test_measured_csv_separation(self):
        train, valid, test, manifest, scaler, held_out = prepare_data(
            ROOT / "data/processed/nasa", "B0018"
        )
        self.assertEqual(manifest["split"]["train"], ("B0005", "B0006"))
        self.assertEqual(manifest["split"]["validation"], ("B0007",))
        self.assertEqual(manifest["split"]["test"], ("B0018",))
        self.assertEqual(train[0].shape, (336, 7, 1))
        self.assertEqual(valid[0].shape, (168, 7, 1))
        self.assertEqual(test[0].shape, (132, 7, 1))
        self.assertTrue(all("capacity" not in col and "soh" not in col for col in FEATURES))
        self.assertFalse("target" in FEATURES)
        self.assertEqual(len(held_out), 132)
        self.assertAlmostEqual(float(held_out["soh_reference"].iloc[0]), 1.0, places=5)
        self.assertEqual(int(scaler.n_samples_seen_), 336)

    def test_complete_model_adam_round_trip(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(44)
        # Small synthetic arrays test the saving mechanism, not predictive scores.
        x = np.random.default_rng(44).normal(size=(12, 7, 1)).astype("float32")
        y = np.random.default_rng(45).uniform(0.7, 1, size=(12,)).astype("float32")
        with tempfile.TemporaryDirectory() as temp:
            ckpt = Path(temp)
            model = build_model(0.001)
            callback = EpochSnapshots(ckpt, signature="test-protocol")
            model.fit(x[:8], y[:8], batch_size=4, validation_data=(x[8:], y[8:]),
                      epochs=1, callbacks=[callback], verbose=0)
            state = json.loads((ckpt / "latest.json").read_text())
            self.assertEqual(state["completed_epoch"], 1)
            iterations_at_save = int(model.optimizer.iterations.numpy())
            restored, state = load_snapshot(ckpt, "test-protocol", resume=True)
            self.assertEqual(int(restored.optimizer.iterations.numpy()), iterations_at_save)
            self.assertEqual(restored.output_shape, (None, 1))
            continued = EpochSnapshots(ckpt, "test-protocol", state)
            restored.fit(x[:8], y[:8], batch_size=4, validation_data=(x[8:], y[8:]),
                         initial_epoch=1, epochs=2, callbacks=[continued], verbose=0)
            last = json.loads((ckpt / "latest.json").read_text())
            self.assertEqual(last["completed_epoch"], 2)
            self.assertEqual(len(last["history"]), 2)
            self.assertGreater(int(restored.optimizer.iterations.numpy()), iterations_at_save)
            with self.assertRaises(ValueError):
                load_snapshot(ckpt, "unrelated-dataset", resume=True)


if __name__ == "__main__":
    unittest.main()

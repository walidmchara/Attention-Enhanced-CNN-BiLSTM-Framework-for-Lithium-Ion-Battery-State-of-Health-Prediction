"""Software-only tests for faithful original NASA MATLAB notebook reproduction.

These synthetic CPU inputs test checkpoints, optimizer restoration, and source
parity; they DO NOT claim to reproduce numerical accuracy without real MATs.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OriginalMatReproductionTests(unittest.TestCase):
    def test_code_parity_with_archived_notebooks(self):
        for cell in ("B0018", "B0030"):
            historic = json.loads(
                (ROOT / "notebooks" / f"CNN_BiLSTM_Attention_{cell}.ipynb").read_text()
            )
            adapted = json.loads(
                (ROOT / "notebooks" / f"REPRO_ORIGINAL_MAT_{cell}_COLAB.ipynb").read_text()
            )
            historic_codes = ["".join(c["source"]) for c in historic["cells"]]
            adapted_codes = ["".join(c["source"]) for c in adapted["cells"]]
            for unchanged_cell in (0, 1, 2, 3, 4, 5, 6, 9):
                self.assertIn(
                    historic_codes[unchanged_cell], adapted_codes,
                    f"{cell}: historic code cell {unchanged_cell} changed"
                )
            self.assertTrue(any("fit_exact_notebook_protocol" in c for c in adapted_codes))
            self.assertTrue(any("sc.fit_transform(dataset_val[attrib])" in c for c in adapted_codes))
            self.assertTrue(any("test_size=0.2, random_state=0" in c for c in adapted_codes))
            self.assertTrue(any("attribs=['capacity'" in c for c in adapted_codes))

    def test_full_model_and_optimizer_resume(self):
        import tensorflow as tf
        from scripts.legacy_mat_recovery import _KeepLastEpoch, fit_exact_notebook_protocol, fingerprint

        tf.keras.utils.set_random_seed(3)
        x = np.random.default_rng(3).normal(size=(16, 7, 1)).astype(np.float32)
        y = np.random.default_rng(9).uniform(0.7, 1.0, size=(16, 1)).astype(np.float32)

        def make_model():
            inputs = tf.keras.Input(shape=(7, 1))
            z = tf.keras.layers.Conv1D(64, 3, activation="relu")(inputs)
            z = tf.keras.layers.MaxPooling1D(2)(z)
            z = tf.keras.layers.Dropout(0.3)(z)
            z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, activation="relu"))(z)
            weights = tf.keras.layers.Dense(200, activation="sigmoid")(z)
            z = tf.keras.layers.Multiply()([z, weights])
            model = tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation="relu")(z))
            model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
            return model

        with tempfile.TemporaryDirectory() as d:
            checkpoint_dir = Path(d)
            metadata = {"protocol": "unit-test-original-mat", "seed": 3}
            signature = fingerprint(metadata)
            model = make_model()
            # Simulate Colab ending after epoch one.
            model.fit(x[:12], y[:12], validation_data=(x[12:], y[12:]),
                      epochs=1, verbose=0,
                      callbacks=[_KeepLastEpoch(checkpoint_dir, signature)])
            first_iterations = int(model.optimizer.iterations.numpy())
            first = json.loads((checkpoint_dir / "latest.json").read_text())
            self.assertEqual(first["epoch"], 1)

            # New session: a newly built model gets replaced by the saved full
            # model (including optimizer) and continues only the missing epoch.
            restored, record = fit_exact_notebook_protocol(
                make_model(), x[:12], y[:12], x[12:], y[12:],
                checkpoint_dir=checkpoint_dir, signature_settings=metadata, epochs=2,
            )
            final = json.loads((checkpoint_dir / "latest.json").read_text())
            self.assertEqual(final["epoch"], 2)
            self.assertEqual(len(final["history"]), 2)
            self.assertEqual(len(record.history["loss"]), 2)
            self.assertGreater(int(restored.optimizer.iterations.numpy()), first_iterations)
            self.assertTrue((checkpoint_dir / "epoch_0002.keras").is_file())
            with self.assertRaises(ValueError):
                fit_exact_notebook_protocol(
                    make_model(), x[:12], y[:12], x[12:], y[12:],
                    checkpoint_dir=checkpoint_dir,
                    signature_settings={"protocol": "different-experiment"},
                    epochs=2
                )


if __name__ == "__main__":
    unittest.main()

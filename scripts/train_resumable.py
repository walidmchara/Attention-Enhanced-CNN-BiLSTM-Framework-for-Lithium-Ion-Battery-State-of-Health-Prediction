"""Reproducible, resumable NASA cycle-level CNN-BiLSTM-Attention SOH experiment.

This is a NEW measured-CSV experiment. The original notebooks operate on
sample-level MATLAB traces and are preserved without modification.

Use --checkpoint-dir on persistent storage (Google Drive / RunPod volume).
Only restore model snapshots created by trusted runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

FEATURES = (
    "voltage_mean",
    "voltage_std",
    "current_mean",
    "current_std",
    "temperature_mean",
    "discharge_duration_s",
    "ambient_temperature",
)
EXPERIMENTS = {
    "B0018": {"train": ("B0005", "B0006"), "validation": ("B0007",), "test": ("B0018",)},
    "B0030": {"train": ("B0029", "B0032"), "validation": ("B0031",), "test": ("B0030",)},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write_atomic(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_battery(path: Path, cell: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing real NASA CSV for {cell}: {path}")
    frame = pd.read_csv(path)
    needed = {"cycle", "capacity_ah", *FEATURES}
    if not needed.issubset(frame):
        raise ValueError(f"{cell} missing columns: {sorted(needed.difference(frame))}")
    if "cell_id" in frame and not frame["cell_id"].astype(str).eq(cell).all():
        raise ValueError(f"{cell}: cell_id in CSV is inconsistent")
    frame = frame.sort_values("cycle", kind="stable").reset_index(drop=True).copy()
    if frame["cycle"].isna().any() or frame["cycle"].duplicated().any():
        raise ValueError(f"{cell}: invalid or duplicate cycles")
    for name in ("capacity_ah", *FEATURES):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if (frame["capacity_ah"] <= 0).any() or not np.isfinite(frame["capacity_ah"]).all():
        raise ValueError(f"{cell}: capacity values must be positive and finite")
    if not np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError(f"{cell}: non-finite measured input features")
    frame["soh_reference"] = frame["capacity_ah"] / float(frame["capacity_ah"].iloc[0])
    frame["battery_id"] = cell
    return frame


def prepare_data(root: Path, experiment: str):
    groups = EXPERIMENTS[experiment]
    cells = set(groups["train"] + groups["validation"] + groups["test"])
    if sum(len(groups[k]) for k in ("train", "validation", "test")) != len(cells):
        raise ValueError("Train, validation and test batteries must be disjoint")
    paths = {cell: root / f"{cell}.csv" for cell in sorted(cells)}
    battery = {cell: load_battery(paths[cell], cell) for cell in paths}

    train = pd.concat([battery[c] for c in groups["train"]], ignore_index=True)
    valid = pd.concat([battery[c] for c in groups["validation"]], ignore_index=True)
    test = pd.concat([battery[c] for c in groups["test"]], ignore_index=True)

    # Never fit transforms to validation/test. Capacity and all SOH/RUL target
    # columns are absent from predictors.
    scaler = MinMaxScaler()
    scaler.fit(train[list(FEATURES)].to_numpy(dtype=np.float32))

    def arrays(frame):
        features = scaler.transform(frame[list(FEATURES)].to_numpy(dtype=np.float32))
        return features[..., np.newaxis].astype(np.float32), frame["soh_reference"].to_numpy(dtype=np.float32)

    provenance = {
        "split": groups,
        "n_samples": {"train": len(train), "validation": len(valid), "test": len(test)},
        "source_csv_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "target_definition": "capacity_ah / first_observed_capacity_ah per battery",
        "features": list(FEATURES),
        "scaler": "MinMaxScaler fitted on development training cells only",
        "input_scope": "same-cycle measured discharge summary; not early-cycle forecasting",
        "source": "NASA PCoE processed real cycle-level CSVs supplied in this repository",
        "protocol_notice": "New measured-CSV protocol; not exact original MATLAB-trace paper reproduction",
    }
    return arrays(train), arrays(valid), arrays(test), provenance, scaler, test


def build_model(learning_rate: float):
    # Same layer architecture as the archived original notebooks; the CSV
    # predictor semantics differ from the raw MATLAB sample-level input.
    inputs = tf.keras.Input(shape=(len(FEATURES), 1))
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(100, activation="relu"), name="bilstm"
    )(x)
    weights = tf.keras.layers.Dense(200, activation="sigmoid", name="attention_vec")(x)
    x = tf.keras.layers.Multiply()([x, weights])
    outputs = tf.keras.layers.Dense(1, activation="relu")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


class EpochSnapshots(tf.keras.callbacks.Callback):
    """Commit each full-model snapshot before atomically updating latest.json."""

    def __init__(self, directory: Path, signature: str, previous: dict | None = None):
        super().__init__()
        self.directory = directory
        self.signature = signature
        self.history = list(previous["history"]) if previous else []
        self.best_epoch = int(previous["best_epoch"]) if previous else 0
        self.best_val_loss = float(previous["best_val_loss"]) if previous else float("inf")
        directory.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs=None):
        metrics = {
            "epoch": epoch + 1,
            **{name: float(value) for name, value in (logs or {}).items()},
        }
        self.history.append(metrics)
        val_loss = float(metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

        filename = f"epoch_{epoch + 1:04d}.keras"
        temp = self.directory / f".partial_{filename}"
        destination = self.directory / filename
        self.model.save(str(temp))
        os.replace(temp, destination)
        digest = sha256_file(destination)
        manifest = {
            "format": "keras-full-model-epoch-v1",
            "signature": self.signature,
            "completed_epoch": epoch + 1,
            "snapshot": filename,
            "snapshot_sha256": digest,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        json_write_atomic(self.directory / "latest.json", manifest)

        # Keep last two fully committed epochs and the best model. An
        # interrupted save never removes the previous committed checkpoint.
        keep = {epoch + 1, max(1, epoch), self.best_epoch}
        for old in self.directory.glob("epoch_*.keras"):
            try:
                old_epoch = int(old.stem.split("_")[1])
            except (ValueError, IndexError):
                continue
            if old_epoch not in keep:
                old.unlink()
        print(f"Saved persistent checkpoint at completed epoch {epoch + 1}: {destination}", flush=True)


def load_snapshot(directory: Path, signature: str, *, resume: bool):
    manifest_path = directory / "latest.json"
    if not resume:
        if manifest_path.exists():
            raise FileExistsError(
                f"Existing training recovery in {directory}. Use --resume or a new checkpoint directory."
            )
        return None, None

    if not manifest_path.is_file():
        raise FileNotFoundError(f"No committed recovery manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("format") != "keras-full-model-epoch-v1":
        raise ValueError("Dataset/configuration mismatch; refusing unsafe recovery.")
    snapshot = directory / manifest["snapshot"]
    if not snapshot.is_file() or sha256_file(snapshot) != manifest["snapshot_sha256"]:
        raise IOError(f"Checkpoint missing or corrupted: {snapshot}")
    model = tf.keras.models.load_model(str(snapshot))
    print(f"Restored model AND optimizer from completed epoch {manifest['completed_epoch']}", flush=True)
    return model, manifest


def run(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test), provenance, scaler, test_df = (
        prepare_data(args.data_dir, args.experiment)
    )
    metadata = {
        "experiment": args.experiment,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_architecture": "Conv1D(64,k3)-MaxPool(2)-Dropout(.3)-BiLSTM(100,relu)-Dense(200,sigmoid)-Multiply-Dense(1,relu)",
        **provenance,
    }
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()

    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, saved = load_snapshot(checkpoint_dir, signature, resume=args.resume)
    if model is None:
        model = build_model(args.learning_rate)
    initial_epoch = saved["completed_epoch"] if saved else 0
    callback = EpochSnapshots(checkpoint_dir, signature, saved)
    print(json.dumps({
        "experiment": args.experiment, "train": list(provenance["split"]["train"]),
        "validation": list(provenance["split"]["validation"]),
        "test": list(provenance["split"]["test"]),
        "sample_counts": provenance["n_samples"],
        "start_epoch": initial_epoch, "target_epochs": args.epochs,
        "checkpoint_dir": str(checkpoint_dir),
    }), flush=True)
    if initial_epoch < args.epochs:
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            batch_size=args.batch_size,
            callbacks=[callback],
            verbose=2,
            shuffle=True,
        )
    else:
        print("All requested epochs already completed; generating final evaluation.", flush=True)

    latest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    if latest["signature"] != signature:
        raise RuntimeError("Unexpected checkpoint signature during evaluation")
    best_path = checkpoint_dir / f"epoch_{latest['best_epoch']:04d}.keras"
    if not best_path.exists():
        raise FileNotFoundError(f"Best validation model missing: {best_path}")
    best_model = tf.keras.models.load_model(str(best_path))
    predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    err = predictions - y_test
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    total = float(np.square(y_test - y_test.mean()).sum())
    r2 = float(1 - np.square(err).sum() / total) if total > 1e-12 else None

    prediction_frame = pd.DataFrame({
        "battery_id": test_df["battery_id"],
        "cycle": test_df["cycle"],
        "soh_true": y_test,
        "soh_pred": predictions,
    })
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(latest["history"]).to_csv(output_dir / "training_history.csv", index=False)
    report = {
        "experiment": args.experiment,
        "completed_epochs": latest["completed_epoch"],
        "best_validation_epoch": latest["best_epoch"],
        "best_validation_mse": latest["best_val_loss"],
        "test": {"mae": mae, "rmse": rmse, "r2": r2, "samples": len(y_test)},
        "provenance": provenance,
        "protocol": "held-out battery, validation battery distinct, features exclude capacity/target",
        "scientific_notice": "New NASA cycle-level experiment, not the original raw MATLAB notebooks",
        "checkpoint_manifest": str(checkpoint_dir / "latest.json"),
    }
    json_write_atomic(output_dir / "metrics.json", report)
    json_write_atomic(output_dir / "config.json", metadata)
    json_write_atomic(output_dir / "scaler.json", {
        "feature_names": list(FEATURES),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    })
    print(json.dumps({"final_test": report["test"], "best_validation_epoch": latest["best_epoch"]}, indent=2), flush=True)
    print(f"Final outputs saved to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="NASA CNN-BiLSTM-Attention with persistent Keras checkpoints")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="B0018")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/nasa"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Mounted persistent Drive/volume checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where CSV + JSON reports are stored")
    parser.add_argument("--resume", action="store_true", help="Recover model + optimizer at last committed epoch")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

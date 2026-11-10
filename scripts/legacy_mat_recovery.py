"""Restore the archived NASA MATLAB experiment after a Colab interruption.

The original source notebooks train 30 epochs and evaluate the LAST epoch.
This helper preserves that choice, including the historical data protocol,
while storing a complete .keras model and Adam state each completed epoch.
It does not correct the original target leakage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(settings: dict) -> str:
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, data: dict):
    tmp = path.with_name("." + path.name + ".partial")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class _KeepLastEpoch(tf.keras.callbacks.Callback):
    def __init__(self, root: Path, signature: str, history=None):
        super().__init__()
        self.root = root
        self.signature = signature
        self.rows = list(history or [])
        root.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        row = {"epoch": completed}
        row.update({key: float(value) for key, value in (logs or {}).items()})
        # Persist the full compiled network (includes optimizer and its moments).
        dst = self.root / f"epoch_{completed:04d}.keras"
        tmp = self.root / f".partial_epoch_{completed:04d}.keras"
        try:
            self.model.save(str(tmp))
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()
        self.rows.append(row)
        atomic_json(self.root / "latest.json", {
            "format": "original-mat-keras-last-epoch-v1",
            "signature": self.signature,
            "epoch": completed,
            "model": dst.name,
            "model_sha256": sha256(dst),
            "history": self.rows,
        })
        # Preserve a second backup. The original notebook evaluates the
        # last epoch (not the minimum validation epoch).
        for old in self.root.glob("epoch_*.keras"):
            old_epoch = int(old.stem.rsplit("_", 1)[1])
            if old_epoch < completed - 1:
                old.unlink()
        print(f"Saved original-MAT epoch {completed} to persistent storage", flush=True)


def fit_exact_notebook_protocol(
    model, X_train_series, Y_train, X_valid_series, Y_valid,
    *, checkpoint_dir: str | Path, signature_settings: dict,
    epochs: int = 30,
):
    """Same model.fit parameters as the archived notebooks (+ callback/resume)."""
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    sig = fingerprint(signature_settings)
    state = root / "latest.json"
    prior = None
    if state.is_file():
        prior = json.loads(state.read_text(encoding="utf-8"))
        if prior.get("format") != "original-mat-keras-last-epoch-v1":
            raise ValueError("Unsupported checkpoint format")
        if prior.get("signature") != sig:
            raise ValueError("Dataset / configuration mismatch. Use a fresh run directory.")
        checkpoint = root / prior["model"]
        if not checkpoint.is_file() or sha256(checkpoint) != prior["model_sha256"]:
            raise IOError("Latest Keras checkpoint is missing or corrupted")
        # For a trusted checkpoint from this run, load the FULL model + Adam.
        model = tf.keras.models.load_model(str(checkpoint), compile=True)
        print(f"Resuming original notebook after completed epoch {prior['epoch']}")
    initial_epoch = int(prior["epoch"]) if prior else 0
    if initial_epoch > epochs:
        raise ValueError("Checkpoint already exceeds the requested number of epochs")
    cb = _KeepLastEpoch(root, sig, prior["history"] if prior else [])
    if initial_epoch < epochs:
        # Original fit() defaults: batch_size=32, shuffle=True, no early stopping,
        # no extra metrics; we change only initial_epoch and callbacks.
        model.fit(
            X_train_series, Y_train,
            validation_data=(X_valid_series, Y_valid),
            epochs=epochs, initial_epoch=initial_epoch,
            verbose=2, callbacks=[cb],
        )
    else:
        print(f"All {epochs} historical epochs have already finished")
    # Historical learning-curve cell expects .history['loss'], etc.
    aggregated = {}
    for row in cb.rows:
        for name, value in row.items():
            if name != "epoch":
                aggregated.setdefault(name, []).append(value)
    return model, SimpleNamespace(history=aggregated, epoch=list(range(epochs)))


def provenance_for_legacy(repo: str | Path, experiment: str, *, seed: int = 42) -> dict:
    repo = Path(repo)
    groups = {
        "B0018": ("B0005", "B0006", "B0007", "B0018"),
        "B0030": ("B0029", "B0032", "B0031", "B0030"),
    }
    if experiment not in groups:
        raise ValueError("Unsupported notebook experiment")
    source_sha = {}
    for cell in groups[experiment]:
        mat = repo / "data" / (cell + ".mat")
        if not mat.is_file():
            raise FileNotFoundError(f"Original NASA source missing: {mat}")
        source_sha[cell] = sha256(mat)
    return {
        "protocol": "original NASA sample-level MATLAB notebook, leakage-retaining",
        "experiment": experiment,
        "mat_sha256": source_sha,
        "training_cells": list(groups[experiment][:-1]),
        "test_cell": groups[experiment][-1],
        "model": "Conv1D(64,k3)-MaxPooling(2)-Dropout(.3)-BiLSTM(100,relu)-Dense(200,sigmoid)-Multiply-Dense(1,relu)",
        "optimizer": "Adam default", "loss": "mse", "metric": "accuracy (as in original)",
        "epochs": 30, "batch_size": 32, "test_size": 0.2, "train_test_random_state": 0,
        "seed_added_for_reproducibility": seed,
        "historical_leakage": [
            "capacity is input feature and determines target SOH",
            "MinMaxScaler fit on all development samples before train/validation split",
            "MinMaxScaler re-fitted on independent test battery",
            "random sample-level validation split groups repeated cycles together",
        ],
        "evaluation": "final-epoch predictions, mean over individual discharge cycles",
        "scientific_caution": "Historical reproduction only, NOT independent leakage-free generalization",
    }

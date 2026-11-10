"""Cycle-aggregated evaluation of the final-epoch NASA SOH model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.legacy_mat_recovery import atomic_json
from .data import testing_frame


def evaluate_and_save(
    *, model, history, data_dir: Path, test_cell: str,
    scaler, output_dir: Path, experiment: str, provenance: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_test, per_sample = testing_frame(data_dir, test_cell, scaler)
    predicted = model.predict(x_test, verbose=1).reshape(-1)
    per_sample["NewSoH"] = predicted
    # Report metrics after aggregation of sample-level predictions by cycle.
    per_cycle = per_sample.groupby("cycle").mean().reset_index()
    true = per_cycle["SoH"]
    estimated = per_cycle["NewSoH"]
    mse = mean_squared_error(true, estimated)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(true, estimated)
    mape = float(np.mean(np.abs((true - estimated) / true)) * 100)

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.to_csv(output_dir / "training_history.csv", index=False)
    per_cycle.to_csv(output_dir / "test_predictions.csv", index=False)

    scores = {
        "experiment": experiment,
        "protocol": "original-mat-legacy-with-target-leakage",
        "epochs_completed": len(frame),
        "evaluation": "last epoch; raw discharge samples averaged within cycle",
        "test": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mse": float(mse),
            "mape_percent": float(mape),
            "r2_additional": float(r2_score(true, estimated)),
            "discharge_cycles": int(len(per_cycle)),
        },
        "scientific_warning": (
            "The original notebook inputs include same-cycle capacity (target leakage), "
            "normalizes test data with a new MinMaxScaler fit, and uses sample-level "
            "random validation. These values are not leakage-free generalization scores."
        ),
        "source": "NASA PCoE original MATLAB notebook protocol",
    }
    atomic_json(output_dir / "metrics.json", scores)
    atomic_json(output_dir / "config.json", provenance)
    model.save(str(output_dir / f"AM_CNN_BILSTM_{experiment}.keras"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(true.to_numpy(), label="Actual-SoH", marker="o")
    ax.plot(estimated.to_numpy(), label="Predicted-SoH", marker="o")
    ax.set(xlabel="cycle", ylabel="SoH",
           title=f"Actual vs Predicted SoH -AM-CNN-BILSTM {experiment}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"soh_prediction_{experiment}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame["epoch"], frame["loss"], label="Train loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="Validation loss")
    ax.set(xlabel="Epochs", ylabel="MSE", title="AM CNN BILSTM Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_convergence.png", dpi=200)
    plt.close(fig)
    print(json.dumps(scores, indent=2))
    return scores

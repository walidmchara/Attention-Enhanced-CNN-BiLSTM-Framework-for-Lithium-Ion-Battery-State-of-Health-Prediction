"""Run a legacy raw-MAT experiment without executing a notebook.

Example:
  python -m legacy_mat_pipeline --experiment B0018 --run-root /content/drive/MyDrive/NASA_CNN_BiLSTM_Attention
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import EXPERIMENTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="B0018")
    parser.add_argument(
        "--run-root", type=Path, default=Path("legacy_mat_runs"),
        help="Root for checkpoints, final model, and metrics. Use a mounted Drive path in Colab.",
    )
    parser.add_argument(
        "--archive", type=Path, default=None,
        help="Optional pre-downloaded verified NASA 5.+Battery+Data+Set.zip",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    archive = args.archive or data_dir / "raw" / "5.+Battery+Data+Set.zip"
    cells = EXPERIMENTS[args.experiment]

    # This script preserves the historically fixed 30-epoch protocol.
    epochs = 30
    run_id = f"legacy_mat_{args.experiment.lower()}_e{epochs}_seed{args.seed}"
    checkpoint_dir = args.run_root.expanduser().resolve() / run_id / "checkpoints"
    output_dir = args.run_root.expanduser().resolve() / run_id / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(root / "scripts" / "restore_original_nasa_mat.py"),
        "--dest", str(data_dir), "--archive", str(archive),
        "--cells", *cells,
    ]
    subprocess.run(command, cwd=root, check=True)

    import tensorflow as tf
    from scripts.legacy_mat_recovery import (
        fit_exact_notebook_protocol, provenance_for_legacy,
    )
    from .data import training_arrays
    from .evaluation import evaluate_and_save
    from .network import build_model

    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train, x_valid, y_valid, scaler = training_arrays(data_dir, cells[:3])
    provenance = provenance_for_legacy(root, args.experiment, seed=args.seed)
    model = build_model(n_features=x_train.shape[1])
    model, history = fit_exact_notebook_protocol(
        model, x_train, y_train, x_valid, y_valid,
        checkpoint_dir=checkpoint_dir, signature_settings=provenance,
        epochs=epochs,
    )
    metrics = evaluate_and_save(
        model=model, history=history, data_dir=data_dir,
        test_cell=cells[3], scaler=scaler, output_dir=output_dir,
        experiment=args.experiment, provenance=provenance,
    )
    print(f"Complete: {run_id}")
    print(f"Outputs: {output_dir}")
    print(f"MAE={metrics['test']['mae']:.6f}; "
          f"RMSE={metrics['test']['rmse']:.6f}; "
          f"R2={metrics['test']['r2_additional']:.6f}")


if __name__ == "__main__":
    main()

# Colab end-to-end SOH training with persistent checkpoints

This repository contains **16 processed real battery datasets**: 12 NASA cycle-level
CSVs (including B0005/B0006/B0007/B0018 and B0029/B0030/B0031/B0032)
and four CALCE CSVs. These are **not original sample-level MATLAB traces**.

## Recommended first experiment: NASA B0018

| Partition | Cells | Cycles |
|---|---|---:|
| Training | B0005, B0006 | 336 |
| Validation (model selection only) | B0007 | 168 |
| Independent test | B0018 | 132 |

Model: same **Conv1D(64,k=3) → MaxPool(2) → Dropout(0.3) →
BiLSTM(100) → Dense(200, sigmoid) attention gate → Multiply →
Dense(1, ReLU)** architecture as the original notebooks. Adam/MSE are
unchanged. The model uses seven measured, non-target **cycle-level summary
features**, not the original sample-level MATLAB inputs. Capacity-derived SOH
and target columns are excluded from predictors; feature scaling is fitted
on the training batteries alone. Held-out battery data are never used to fit
the scaler, optimizer or to select the best epoch.

**Important scientific distinction:** this script is a new independent
measured-cycle experiment, **not** an exact reproduction of the MATLAB
notebook study. The regression estimates SOH from the measurements collected
during the same discharge cycle; it is not a before-discharge forecast.

## Open in Colab

[Open the resumable Colab notebook](../notebooks/train_resumable_colab.ipynb)

Or use:
https://colab.research.google.com/github/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction/blob/main/notebooks/train_resumable_colab.ipynb

Select a GPU runtime (optional for this light experiment). The notebook
clones this **public repository**, checks the actual datasets, mounts Google
Drive and launches 30 epochs.

### Manual first run

After mounting Drive, from the root of the GitHub clone:

```bash
python scripts/train_resumable.py \
  --experiment B0018 --epochs 30 --batch-size 16 --seed 42 \
  --data-dir data/processed/nasa \
  --checkpoint-dir /content/drive/MyDrive/NASA_CNN_BiLSTM_Attention/b0018_e30_seed42/checkpoints \
  --output-dir /content/drive/MyDrive/NASA_CNN_BiLSTM_Attention/b0018_e30_seed42/outputs
```

### Reconnect to Colab after an interruption

Rerun **the same command**, adding `--resume`. The runner reads
`latest.json`, verifies its model SHA-256 and experiment fingerprint,
then restores the full Keras model and Adam optimizer state from
`epoch_NNNN.keras` with `initial_epoch` set to the last fully
completed epoch. The last incomplete epoch is repeated.

Snapshots are committed atomically: the current snapshot and metadata
are published only after the full model is written. Only the latest two
snapshots and the best validation model are retained to reduce storage.

The experiment fingerprint covers training options, cell-wise split and
the SHA-256 digest of each source CSV. Do not change the target epoch count,
data, optimizer, seed, architecture or protocol during a resume.
The optimizer state is restored, but fully bit-identical random trajectories
across sessions/devices are **not guaranteed** because accelerator-level RNG
and nondeterministic operations are not fully captured.

Checkpoints must reside on **Drive or a persistent RunPod volume**. Saving
only to `/content` does not protect against session deletion.

### Outputs after training

The persistent outputs folder contains:
- `metrics.json` — held-out B0018 MAE/RMSE/R² and best validation epoch
- `config.json` — experiment settings, input SHA-256 and splits
- `scaler.json` — parameters fitted on training cells only
- `training_history.csv` — per-epoch losses
- `test_predictions.csv` — real test cycle, observed and estimated SOH

The notebook can optionally commit **only these small artifacts** to
`results/<run-id>/` in GitHub. To push from Colab, the fine-grained PAT
stored as Colab secret `GithubToken` must have **Contents: Read and write**
authorization for this public repository. Existing permission limited to
another repo (e.g. SG-PIWT) is not sufficient. Never commit or share the
token. The full `.keras` checkpoints are not uploaded to ordinary Git.

Run the fast input + checkpoint test from the repository root:

```bash
python -m pytest -q tests/test_resumable_training.py
```

## Original notebooks

`notebooks/CNN_BiLSTM_Attention_B0018.ipynb` and
`notebooks/CNN_BiLSTM_Attention_B0030.ipynb` are preserved for historical
provenance. They read raw NASA `.mat` discharge traces under `data/`,
which are **not included** in this repository. They also contain a
same-cycle target-feature leakage issue and re-fit their scaler on
held-out data. Do not quote their outputs as independent, leakage-free
generalization metrics without a separate methodological correction.

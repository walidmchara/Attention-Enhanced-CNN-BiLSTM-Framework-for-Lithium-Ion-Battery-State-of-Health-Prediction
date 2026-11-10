# Original NASA MATLAB experiment — Python entry point

Run the archived CNN–BiLSTM–Attention experiments as a **Python package**
without executing the original or Colab notebooks. The notebook files remain
in `notebooks/` as reference material.

## Run in Colab

First select a GPU runtime, then execute these setup commands in Colab (a
terminal is fine for pip and git; Drive must be mounted in a Colab Python cell).

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone https://github.com/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction.git
%cd /content/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction
!pip install -r requirements.txt
!python -m legacy_mat_pipeline --experiment B0018 --run-root /content/drive/MyDrive/NASA_CNN_BiLSTM_Attention
```

Replace `B0018` with `B0030` for the second historical experiment.
For subsequent sessions, update the clone with `git pull --ff-only` rather
than cloning it again. Running the same command resumes from a verified full
Keras checkpoint after the last completed epoch. No notebook needs to be
executed for training.

If the archive was already downloaded, add
`--archive "/content/drive/MyDrive/NASA_original_source/5.+Battery+Data+Set.zip"`.
Otherwise the NASA official archive is retrieved automatically, checked
against its recorded SHA-256 and extracted into `data/Bxxxx.mat`.

## Files and parity

- `data.py`: original discharge-sample extraction, 7 inputs including
  `capacity`, MinMaxScaler fit on all development rows, 80/20 random
  train/validation split (`random_state=0`), and separate test-time scaler fit.
- `network.py`: Conv1D(64, 3) → MaxPool(2) → Dropout(.3) →
  BiLSTM(100, ReLU) → Dense(200, sigmoid) → Multiply → Dense(1, ReLU),
  MSE/Adam/accuracy, default batch size 32.
- `__main__.py`: 30 epochs on the original training batteries, with
  checkpoints and full optimizer state from `scripts/legacy_mat_recovery.py`.
- `evaluation.py`: final-epoch predictions averaged by discharge cycle;
  MAE, RMSE, MSE, MAPE and additional R², figures and model.

Output goes to `<run-root>/legacy_mat_b0018_e30_seed42/` or
`<run-root>/legacy_mat_b0030_e30_seed42/`. Per-run `outputs/`
contains `metrics.json`, `config.json`, `training_history.csv`,
`test_predictions.csv`, two PNG plots, and a full `.keras` model.
The two most recent checkpoints are retained under `checkpoints/`.
The Python version reuses the exact same run IDs and checkpoint signatures
as the revised Colab notebooks.

## Scientific interpretation

This is an **archival replication**, not a corrected validation experiment:
same-cycle capacity (from which SOH is calculated) remains a model input,
validation contains randomly split measurements from shared cycles, and
the test scaler is re-fitted on the test battery. MAE/RMSE cannot be
interpreted as independent leakage-free forecasting metrics. Seed 42 and
Keras compatibility changes are technical differences; numerical identity
with an earlier run is not guaranteed. Full 30-epoch parity must be checked
with a fresh training run; the implementation itself cannot guarantee the
paper's historical numeric scores.

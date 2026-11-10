# Reproduce the original NASA MATLAB notebooks

This repository offers **two separate experimental protocols**:

- **Historic raw-MAT reproduction**: faithfully follows the two original
  sample-level CNN–BiLSTM–Attention notebooks, including their capacity
  predictor, split, model, 30 epochs, test normalization and per-cycle
  averaging. This retains **known target leakage** and is NOT a valid
  independent generalization claim.
- **Modern cycle-level CSV**: the different leakage-aware experiment in
  `notebooks/train_resumable_colab.ipynb`, which cannot be compared
  numerically to the historic MATLAB experiment as though the protocols
  were identical.

## Open the original experiments

- [B0018 MATLAB original on Colab](https://colab.research.google.com/github/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction/blob/main/notebooks/REPRO_ORIGINAL_MAT_B0018_COLAB.ipynb)
- [B0030 MATLAB original on Colab](https://colab.research.google.com/github/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction/blob/main/notebooks/REPRO_ORIGINAL_MAT_B0030_COLAB.ipynb)

Execute cells **in order**. The setup cell clones the public GitHub repo,
mounts Google Drive, and downloads and verifies the original NASA archive.
The raw NASA `.mat` files are extracted locally into `data/Bxxxx.mat`,
excluded from ordinary Git history.

NASA PCoE source: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

Expected ZIP SHA-256 from the dataset manifest:
`82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`

The actual NASA ZIP contains **six nested ZIP archives**; the new
`scripts/restore_original_nasa_mat.py` retrieves the specific cells and
checks their discharge-cycle counts before accepting their files.

If downloading the official archive fails, manually save the original ZIP
to `MyDrive/NASA_original_source/5.+Battery+Data+Set.zip` and rerun the
setup cell. The official source ZIP download, exact SHA-256 verification, and
extraction of all eight requested MATLAB files **passed GitHub Actions**:
https://github.com/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction/actions/runs/42798834919
This does not claim that a full 30-epoch run has already been carried out
in the user's Colab runtime.

| Original notebook | Development cells | Test battery |
|---|---|---|
| B0018 | B0005, B0006, B0007 | B0018 |
| B0030 | B0029, B0032, B0031 | B0030 |

## Exact historical protocol retained

1. Extract *every sample* from each original MATLAB discharge cycle.
2. Calculate SOH from the original battery capacity convention, and
   include **`capacity`** in the seven predictors.
3. Apply MinMax scaling on the entire development data set before
   `train_test_split(test_size=0.2, random_state=0)`.
4. Train the original Conv1D(64,k=3) → MaxPooling1D(2) → Dropout(0.3) →
   Bidirectional(LSTM(100, activation='relu')) → Dense(200,sigmoid) →
   Multiply → Dense(1,relu) architecture.
5. Use original Adam, MSE, accuracy metric (historically inappropriate
   for regression), 30 epochs and default batch size 32.
6. Re-fit the same scaler on test battery with `sc.fit_transform`, as
   in the original, and then predict.
7. Average sample-level predictions **within each discharge cycle**,
   then calculate MAE, RMSE, MSE and MAPE.
8. Test the **last** completed epoch, rather than choosing the best
   validation epoch.

**Technical-only changes**: the original 2D test input is reshaped to
`(N, 7, 1)` for Keras 3, the final model is saved in the modern full
`.keras` format, a seed 42 is specified for better repeatability, and
per-epoch Drive checkpoints are added.

## Google Drive checkpoints and resume

These separate locations protect the earlier cycle-level experiment:

- `MyDrive/NASA_CNN_BiLSTM_Attention/legacy_mat_b0018_e30_seed42/checkpoints`
- `MyDrive/NASA_CNN_BiLSTM_Attention/legacy_mat_b0030_e30_seed42/checkpoints`

The helper `scripts/legacy_mat_recovery.py` saves the full Keras model
and optimizer after each completed epoch, verifies SHA-256 and the source
`.mat` hashes, and continues unfinished epochs when the same notebook is
rerun. The last two completed epochs are retained. An interrupted current
epoch must run again. Bitwise-identical results across runtimes and GPUs
cannot be guaranteed, even with the added seed.

Each notebook saves a Drive output folder containing `metrics.json`,
`config.json`, `training_history.csv` and `test_predictions.csv` and the
final `.keras` model. An optional final notebook cell publishes only small
reports to an explicitly labeled `results/legacy_mat_*` directory on
GitHub; it never publishes the raw .mat files or the trained .keras models.
To publish, Colab's `GithubToken` must have Contents: Read and write for
this public repository. Never display or share the token.

## Scientific limitation

The original notebook passes same-cycle capacity to a model whose target
is SOH calculated from that capacity. The test scaler is fitted on
held-out test data, and train/validation rows from the same discharge
cycle can overlap across the random split. The original BiLSTM steps
traverse seven different features rather than chronological cycles.

**Therefore, even when the low historical MAE and RMSE are reproduced,
they cannot be presented as independent leakage-free forecasting results.**
This archival reproduction is for understanding the source paper's
reported numbers. Keep it separate from scientifically corrected protocols.

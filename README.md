# CNN–BiLSTM with Attention for Lithium-Ion Battery SOH

TensorFlow notebook experiments for estimating lithium-ion battery State of Health (SOH) from NASA PCoE battery-aging measurements.

## Network architecture

`Discharge measurements → Conv1D → MaxPooling → Dropout → BiLSTM → Attention → SOH regression`

The experimental notebook architecture uses a 64-filter convolutional layer, 0.30 dropout, a bidirectional LSTM with 100 units, an attention stage and dense regression output. The optimizer is Adam with mean-squared-error loss.

## Battery-level experiments

| Experiment | Development cells | Independent test cell | Notebook |
| --- | --- | --- | --- |
| NASA B0018 | B0005, B0006, B0007 | **B0018** | [B0018 experiment](notebooks/CNN_BiLSTM_Attention_B0018.ipynb) |
| NASA B0030 | B0029, B0031, B0032 | **B0030** | [B0030 experiment](notebooks/CNN_BiLSTM_Attention_B0030.ipynb) |

The MATLAB-based notebooks extract sample-level voltage, current, temperature, load measurements and time information. SOH is defined relative to the initial measured capacity. Feature selection must be reviewed for target leakage in any independent evaluation.

## Datasets

The repository includes 12 NASA PCoE battery series and four CALCE CS2 series as cycle-level CSV tables in [data/processed/](data/processed/). The [dataset manifest](data/processed/manifest.json) records source checksums and conversion counts.

The two existing TensorFlow notebooks use the **original NASA MATLAB discharge traces**, not the CSV aggregates. Their MATLAB inputs are read from repository-relative paths under `data/` (for example `data/B0005.mat`). These source files are maintained separately from the versioned CSV tables. See [data documentation](data/README.md) and [source extraction guide](docs/REAL_DATA_SETUP.md) for their precise format and provenance.

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
jupyter notebook
```

Activate the environment according to the operating system before starting Jupyter. The notebooks are located in [notebooks/](notebooks/). Evaluation includes SOH curves, prediction error and training history.

## Research notes

For reproducible battery-level evaluation, keep the held-out cell outside feature selection, normalization fitting and training. The included processed tables and notebook workflows do not by themselves establish correspondence with any publication's reported benchmark scores.

The underlying battery datasets are maintained by the [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/University of Maryland](https://calce.umd.edu/data). The project is distributed under the terms of its [LICENSE](LICENSE).

## Fully resumable Colab training from the included NASA CSV data

For a **new, leakage-aware cycle-level evaluation**, use the
[resumable Colab notebook](notebooks/train_resumable_colab.ipynb) and
[training/recovery guide](TRAINING_RECOVERY.md). The 30-epoch B0018 trial
trains on B0005/B0006, validates on B0007 and tests only on B0018;
features exclude battery capacity and target columns, and MinMax scaling
uses training cells only. Keras saves model + optimizer state to
**persistent Google Drive** after each epoch. It can resume with
`--resume`; small scientific reports can be published to GitHub.

This is **not** an exact re-run of the historical notebooks: they require
original NASA MATLAB discharge samples, which are not versioned in this
repository, and use a different input representation. Do not attribute
results of the new CSV workflow to the original paper.

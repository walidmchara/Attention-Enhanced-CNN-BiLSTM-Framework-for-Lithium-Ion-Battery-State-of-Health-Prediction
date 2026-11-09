# Real datasets installed: NASA PCoE and CALCE CS2

## Files checked into the repository
- `data/processed/nasa/B0005.csv` through the 12 selected batteries listed in `data/processed/manifest.json` (1,157 discharge-cycle rows across 12 cells).
- `data/processed/calce/CS2_35.csv`, `CS2_36.csv`, `CS2_37.csv`, `CS2_38.csv` (3,921 cycle-level rows across **all 106 CALCE XLSX workbooks**).
- `data/processed/manifest.json`: input archive SHA-256 fingerprints, cell IDs, output record counts.
- `scripts/prepare_battery_data.py`: the code used to regenerate derived CSVs.

The original `.mat` and Excel/ZIP archives are **not** tracked because they are very large; derived CSVs are tracked. Source links: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). NASA asks researchers to acknowledge data donors; CALCE asks publications to cite the relevant experimental studies. Verify conditions for subsequent redistribution.

## Derived columns and warnings

Each CSV has `cycle`, `time`, `capacity_ah`, `soh_first_observed`, `target`, `cell_id`, diagnostic statistics and `remaining_observed_cycles`. SOH here is relative to the **first observed** discharge capacity, not necessarily the published nominal rating. `remaining_observed_cycles` counts records until *the end of the observed file* and is **not** confirmed EOL-based RUL.

NASA: one row per valid discharge cycle. Some cells have `target > 1` because the first measured capacity is lower than later measurements; do not clip these silently. CALCE: discharge capacity is cumulative **within each workbook**; conversion uses capacity maximum minus minimum per valid cycle, corrects identical-time duplicates, and flags workbook boundary cycles. CALCE measurement sheets lack an observed temperature channel, so `temperature_mean` is missing.

**Avoid target leakage.** Do not feed the same-cycle `capacity_ah`, `soh_first_observed`, `soh_nominal`, `target`, or `remaining_observed_cycles` as predictor features for an SOH model. The current reconstruction's `train.py` appends *past* target values to historical windows; this is an autoregressive research assumption, not an independently validated paper implementation. Split by cell for cross-battery benchmarks; the default one-CSV chronology is a pipeline test only.

## Notebooks and use of checked-in data

The existing TensorFlow notebooks in `notebooks/` use `scipy.io.loadmat` to read **the original NASA `.mat` time-series files** from the local `data/` directory. They cannot consume cycle-level CSV aggregates without rewriting their feature-extraction cells.

The following original files are needed locally to run both notebooks: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`, `B0029.mat`, `B0030.mat`, `B0031.mat`, `B0032.mat`. Extract them from the original NASA ZIP, e.g.:

```bash
python scripts/prepare_battery_data.py --nasa 'data/raw/5.+Battery+Data+Set.zip' --nasa-cells B0005 B0006 B0007 B0018 B0029 B0030 B0031 B0032 --out data/processed --extract-nasa-mat-to data
```

The 12 NASA and four CALCE **processed CSVs are already checked in** under `data/processed/`, independently of the notebooks. For future cycle-level models, these CSVs are a compact input with the target leakage precautions described above. Notebook experiments remain dependent on the full original `.mat` signals.

## Regenerating from original archives

Put the downloaded `5.+Battery+Data+Set.zip` and `CS2_35.zip`–`CS2_38.zip` into `data/raw/`, then:

```bash
python scripts/prepare_battery_data.py --nasa 'data/raw/5.+Battery+Data+Set.zip' --calce data/raw/CS2_35.zip data/raw/CS2_36.zip data/raw/CS2_37.zip data/raw/CS2_38.zip --out data/processed --extract-nasa-mat-to data/raw/nasa
```

CSV data are derived from real measurements, **not** synthetic demonstration data. They have not been used here to verify published paper-level accuracy.

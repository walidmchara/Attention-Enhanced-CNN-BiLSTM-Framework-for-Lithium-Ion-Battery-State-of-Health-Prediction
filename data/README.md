# Battery datasets

The repository contains cycle-level measurements from the NASA PCoE and CALCE CS2 battery-aging benchmarks. Their provenance, conversion settings and file checksums are recorded in [processed/manifest.json](processed/manifest.json).

## Available processed data

| Source | Dataset | Directory |
| --- | --- | --- |
| NASA PCoE | B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053, B0054, B0055, B0056 | [processed/nasa/](processed/nasa/) |
| CALCE (University of Maryland) | CS2_35, CS2_36, CS2_37, CS2_38 | [processed/calce/](processed/calce/) |

The CSV files contain cycle-level statistics and capacity-derived health indicators. They are suitable for cycle-level analyses; they do not retain every time sample from the original experiments.

## Notebook experiments

The two notebooks in `notebooks/` use MATLAB discharge traces through `scipy.io.loadmat`, rather than the processed CSV aggregates. Their expected MATLAB inputs are:

- **B0018 experiment:** B0005, B0006, B0007 for model development; B0018 for independent evaluation.
- **B0030 experiment:** B0029, B0031, B0032 for model development; B0030 for independent evaluation.

The original `.mat` files are maintained separately from the versioned cycle-level tables. For MATLAB-based notebook runs, the expected local paths are `data/B0005.mat`, etc. The extraction procedure is documented in [../docs/REAL_DATA_SETUP.md](../docs/REAL_DATA_SETUP.md). Both notebooks use repository-relative paths instead of machine-specific Windows paths.

## Data provenance and evaluation

Original sources: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). For health prediction, capacity-derived labels must be kept separate from same-cycle predictors; cross-cell comparisons require a held-out battery. The included processed tables provide measured data, but their availability alone does not establish reproduction of any published performance figures.

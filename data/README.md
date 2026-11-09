# Dataset

Place the NASA lithium-ion battery `.mat` files here.

Experiments in this repository require:

- B0005.mat
- B0006.mat
- B0007.mat
- B0018.mat
- B0029.mat
- B0030.mat
- B0031.mat
- B0032.mat

Dataset files are intentionally excluded from Git by `.gitignore`.


## User-supplied NASA + CALCE source archives

The NASA outer ZIP contains nested archives with the `.mat` files. CALCE CS2_35–38 contain 25/26/27/28 `.xlsx` files; the original CALCE discharge capacity column is cumulative, so it must be differenced per cycle. See [`../docs/REAL_DATA_SETUP.md`](../docs/REAL_DATA_SETUP.md). Run `python scripts/prepare_battery_data.py --help` from the project root. Never commit the original archives or processed CSVs.

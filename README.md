# Attention-Based CNN-BiLSTM for Li-ion Battery SOH Prediction

A deep-learning project for lithium-ion battery **State of Health (SOH)** estimation using an
**Attention-based CNN-BiLSTM** architecture and NASA battery aging datasets.

## Model

The notebooks implement the following pipeline:

`Battery measurements → 1D CNN → Max Pooling → Dropout → BiLSTM → Attention → Dense SOH output`

The model uses:
- Conv1D with 64 filters
- MaxPooling1D
- Dropout = 0.30
- Bidirectional LSTM with 100 units
- Sigmoid attention vector
- Dense regression output
- Adam optimizer and MSE loss

## Experiments

### Experiment 1 — B0018
Training batteries:
- B0005
- B0006
- B0007

Independent test battery:
- **B0018**

Notebook: `notebooks/CNN_BiLSTM_Attention_B0018.ipynb`

### Experiment 2 — B0030
Training batteries:
- B0029
- B0031
- B0032

Independent test battery:
- **B0030**

Notebook: `notebooks/CNN_BiLSTM_Attention_B0030.ipynb`

## Input features

The original implementation uses:
- Capacity
- Measured voltage
- Measured current
- Measured temperature
- Load current
- Load voltage
- Time

SOH is calculated relative to the initial battery capacity.

## Repository structure

```text
Attention-CNN-BiLSTM-SOH-Prediction/
├── notebooks/
│   ├── CNN_BiLSTM_Attention_B0018.ipynb
│   └── CNN_BiLSTM_Attention_B0030.ipynb
├── src/
│   └── README.md
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── results/
│   └── figures/
│       └── .gitkeep
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset setup

Place the NASA `.mat` battery files in `data/`.

Required for the included experiments:

`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`,
`B0029.mat`, `B0030.mat`, `B0031.mat`, `B0032.mat`.

> The original notebooks contain a local Windows path (`E:/`). Update the dataset path before running,
> or refactor the loader to point to the repository `data/` directory.

## Installation

```bash
git clone <your-repository-url>
cd Attention-CNN-BiLSTM-SOH-Prediction
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Evaluation

The notebooks calculate SOH predictions per cycle and include error metrics such as:
- RMSE
- MAPE

They also plot training and validation loss.

## Suggested GitHub topics

`battery-health` `state-of-health` `lithium-ion-battery` `deep-learning`
`cnn` `bilstm` `attention-mechanism` `remaining-useful-life`
`predictive-maintenance` `tensorflow` `nasa-dataset`

## License

MIT License.

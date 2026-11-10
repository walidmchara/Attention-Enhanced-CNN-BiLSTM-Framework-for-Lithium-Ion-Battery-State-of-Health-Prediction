"""NASA discharge-trace extraction and historical feature preparation."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURES = (
    "capacity", "voltage_measured", "current_measured",
    "temperature_measured", "current_load", "voltage_load", "time",
)
MEASUREMENT_COLUMNS = (
    "cycle", "ambient_temperature", "datetime", "capacity",
    "voltage_measured", "current_measured", "temperature_measured",
    "current_load", "voltage_load", "time",
)
CAPACITY_COLUMNS = ("cycle", "ambient_temperature", "datetime", "capacity")


def load_data(data_dir: Path, battery: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract ordered discharge measurements and per-cycle capacities."""
    mat_path = Path(data_dir) / f"{battery}.mat"
    mat = loadmat(str(mat_path))
    cycles = mat[battery][0, 0]["cycle"][0]
    print(f"{battery}: total raw cycles = {len(cycles)}", flush=True)
    dataset, capacity_data = [], []
    counter = 0
    for i in range(len(cycles)):
        row = cycles[i]
        if row["type"][0] != "discharge":
            continue
        ambient_temperature = row["ambient_temperature"][0][0]
        t = row["time"][0]
        date_time = datetime.datetime(
            int(t[0]), int(t[1]), int(t[2]),
            int(t[3]), int(t[4])
        ) + datetime.timedelta(seconds=int(t[5]))
        data = row["data"][0][0]
        capacity = data["Capacity"][0][0]
        for j in range(len(data["Voltage_measured"][0])):
            dataset.append([
                counter + 1, ambient_temperature, date_time, capacity,
                data["Voltage_measured"][0][j],
                data["Current_measured"][0][j],
                data["Temperature_measured"][0][j],
                data["Current_load"][0][j],
                data["Voltage_load"][0][j],
                data["Time"][0][j],
            ])
        capacity_data.append([
            counter + 1, ambient_temperature, date_time, capacity
        ])
        counter += 1
    return (
        pd.DataFrame(dataset, columns=MEASUREMENT_COLUMNS),
        pd.DataFrame(capacity_data, columns=CAPACITY_COLUMNS),
    )


def training_arrays(data_dir: Path, cells: tuple[str, ...]):
    """Construct the seven-feature sample-level training/validation split."""
    loaded = [load_data(data_dir, cell) for cell in cells]
    dataset = pd.concat([part[0] for part in loaded], axis=0).reset_index(drop=True)

    # SOH normalization follows the first development discharge capacity.
    first_capacity = dataset["capacity"].iloc[0]
    soh = pd.DataFrame({"SoH": dataset["capacity"] / first_capacity})
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset[list(FEATURES)])
    x_train, x_valid, y_train, y_valid = train_test_split(
        scaled, soh, test_size=0.2, random_state=0
    )
    print(
        f"Development measurements: {len(dataset)}; "
        f"train: {len(x_train)}, validation: {len(x_valid)}",
        flush=True,
    )
    return (
        x_train.reshape((x_train.shape[0], x_train.shape[1], 1)),
        y_train,
        x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1)),
        y_valid,
        scaler,
    )


def testing_frame(data_dir: Path, cell: str, scaler: MinMaxScaler):
    """Prepare the original test-battery scaling and SOH reference."""
    dataset, _ = load_data(data_dir, cell)
    # The historical test protocol fits its scaler on the test battery.
    scaled = scaler.fit_transform(dataset[list(FEATURES)])
    x_test = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))
    actual = dataset["capacity"].to_numpy() / dataset["capacity"].iloc[0]
    return x_test, pd.DataFrame({
        "cycle": dataset["cycle"].to_numpy(), "SoH": actual
    })

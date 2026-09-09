import pandas as pd
import numpy as np

DATA_PATH = "/workspaces/XO10_DS02/development_train.csv"

df = pd.read_csv(DATA_PATH)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

sensor_columns = [
    "CO_Channel",
    "NOx_Channel",
    "NO2_Channel",
    "Ambient_Temperature",
    "Relative_Humidity",
    "Absolute_Humidity"
]

print("=" * 60)
print("SENTINEL - ANOMALY DETECTOR")
print("=" * 60)

for column in sensor_columns:

    # Calculate mean and standard deviation
    mean = df[column].mean()
    std = df[column].std()

    # Calculate Z-score
    df[f"{column}_zscore"] = (
        (df[column] - mean) / std
    )

    # Count unusual values
    anomalies = (
        df[f"{column}_zscore"].abs() > 3
    ).sum()

    print(
        f"{column:25} : "
        f"{anomalies} anomalies"
    )

print("\n" + "=" * 60)
print("ANOMALY DETECTION COMPLETE")
print("=" * 60)
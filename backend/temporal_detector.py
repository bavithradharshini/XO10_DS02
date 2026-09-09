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
print("SENTINEL - TEMPORAL CONSISTENCY DETECTOR")
print("=" * 60)

for column in sensor_columns:

    # Calculate change from previous reading
    change = df[column].diff().abs()

    # Calculate normal change statistics
    mean_change = change.mean()
    std_change = change.std()

    # Detect unusually large sudden changes
    threshold = mean_change + (3 * std_change)

    sudden_changes = (change > threshold).sum()

    print(
        f"{column:25} : "
        f"{sudden_changes} sudden changes"
    )

print("\n" + "=" * 60)
print("TEMPORAL ANALYSIS COMPLETE")
print("=" * 60)
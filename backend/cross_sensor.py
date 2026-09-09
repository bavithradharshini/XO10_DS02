import pandas as pd
import numpy as np

DATA_PATH = "/workspaces/XO10_DS02/development_train.csv"

df = pd.read_csv(DATA_PATH)

sensor_columns = [
    "CO_Channel",
    "NOx_Channel",
    "NO2_Channel",
    "Ambient_Temperature",
    "Relative_Humidity",
    "Absolute_Humidity"
]

print("=" * 60)
print("SENTINEL - CROSS SENSOR CONSISTENCY")
print("=" * 60)

# Calculate correlation between sensors
correlation = df[sensor_columns].corr()

print("\nSENSOR CORRELATION MATRIX")
print(correlation.round(2).to_string())

# Find highly related sensor pairs
print("\nRELATED SENSOR PAIRS")

for i in range(len(sensor_columns)):
    for j in range(i + 1, len(sensor_columns)):

        sensor_a = sensor_columns[i]
        sensor_b = sensor_columns[j]

        value = correlation.loc[sensor_a, sensor_b]

        if abs(value) >= 0.5:
            print(
                f"{sensor_a} <-> {sensor_b} : "
                f"{value:.2f}"
            )

print("\n" + "=" * 60)
print("CROSS SENSOR ANALYSIS COMPLETE")
print("=" * 60)
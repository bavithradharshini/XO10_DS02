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

# Start every observation with perfect trust
df["Trust_Score"] = 100.0

for column in sensor_columns:

    mean = df[column].mean()
    std = df[column].std()

    # Z-score
    z_score = ((df[column] - mean) / std).abs()

    # Penalize unusual readings
    penalty = np.where(
        z_score > 3,
        30,
        np.where(z_score > 2, 15, 0)
    )

    df["Trust_Score"] -= penalty


# Keep score between 0 and 100
df["Trust_Score"] = df["Trust_Score"].clip(0, 100)


# Assign reliability level
def reliability_level(score):

    if score >= 80:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    else:
        return "LOW"


df["Reliability"] = df["Trust_Score"].apply(
    reliability_level
)


print("=" * 60)
print("SENTINEL - TRUST SCORE ENGINE")
print("=" * 60)

print("\nTOTAL OBSERVATIONS:", len(df))

print("\nRELIABILITY DISTRIBUTION")
print(df["Reliability"].value_counts())

print("\nSAMPLE RESULTS")

print(
    df[
        [
            "Station_ID",
            "Timestamp",
            "Trust_Score",
            "Reliability"
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)
print("TRUST SCORE GENERATION COMPLETE")
print("=" * 60)
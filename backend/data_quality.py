import pandas as pd

DATA_PATH = "/workspaces/XO10_DS02/development_train.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sensor columns
sensor_columns = [
    "CO_Channel",
    "NOx_Channel",
    "NO2_Channel",
    "Ambient_Temperature",
    "Relative_Humidity",
    "Absolute_Humidity"
]

print("=" * 60)
print("SENTINEL - DATA QUALITY ANALYZER")
print("=" * 60)

# 1. Missing values
print("\n[1] MISSING VALUES")

for column in sensor_columns:
    missing = df[column].isna().sum()

    print(f"{column:25} : {missing}")

# 2. Duplicate records
print("\n[2] DUPLICATE RECORDS")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# 3. Negative values
print("\n[3] NEGATIVE SENSOR VALUES")

for column in sensor_columns:
    negative = (df[column] < 0).sum()

    print(f"{column:25} : {negative}")

# 4. Basic range information
print("\n[4] SENSOR RANGES")

for column in sensor_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    print(
        f"{column:25} : "
        f"Min = {minimum:.2f}, "
        f"Max = {maximum:.2f}"
    )

print("\n" + "=" * 60)
print("DATA QUALITY ANALYSIS COMPLETE")
print("=" * 60)
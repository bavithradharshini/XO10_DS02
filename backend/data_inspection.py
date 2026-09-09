import pandas as pd

# Location of our dataset
DATA_PATH = "../dataset/development_train.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SENTINEL - DATASET INSPECTION")
print("=" * 60)

print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST 5 OBSERVATIONS")
print(df.head().to_string())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. BASIC STATISTICS")
print(df.describe().round(2).to_string())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
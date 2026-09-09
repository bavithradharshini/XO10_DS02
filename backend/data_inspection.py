import os
import pandas as pd

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "development_train.csv")
)
DATA_PATH = os.environ.get("SENTINEL_DATA_PATH", DEFAULT_DATA_PATH)


def load_dataset(filepath=None):
    path = filepath or DATA_PATH
    if not os.path.exists(path):
        for candidate in [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "development_train.csv")),
            os.path.abspath("development_train.csv"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "development_train.csv")),
        ]:
            if os.path.exists(candidate):
                path = candidate
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def detect_columns(df):
    cols = df.columns.tolist()

    # 1. Detect timestamp column first
    timestamp_col = None
    for c in cols:
        if any(keyword in c.lower() for keyword in ["timestamp", "datetime", "observed", "time", "date"]):
            timestamp_col = c
            break

    # 2. Detect station / location identifier column
    station_col = None
    station_keywords = ["station", "stn", "location", "site", "device", "node"]
    for c in cols:
        if any(keyword in c.lower() for keyword in station_keywords):
            station_col = c
            break

    # If no keyword matched, look for categorical object column (excluding timestamp)
    if station_col is None:
        for c in cols:
            if c != timestamp_col and (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype)):
                station_col = c
                break

    # 3. Detect sensor numeric columns (exclude station and timestamp)
    excluded = {c for c in [station_col, timestamp_col] if c is not None}
    sensor_cols = []
    for c in cols:
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c]):
            sensor_cols.append(c)

    return {
        "station_col": station_col,
        "timestamp_col": timestamp_col,
        "sensor_cols": sensor_cols
    }


def inspect_dataset(df):
    col_info = detect_columns(df)
    summary_stats = df.describe().round(2).to_dict()

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(val) for col, val in df.isnull().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "statistics": summary_stats,
        "detected_columns": col_info
    }


if __name__ == "__main__":
    df = load_dataset()

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
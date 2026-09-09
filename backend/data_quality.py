import os
import pandas as pd
try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def analyze_data_quality(df, sensor_columns=None):
    if sensor_columns is None:
        detected = detect_columns(df)
        sensor_columns = detected["sensor_cols"]

    total_records = len(df)
    duplicates = int(df.duplicated().sum())

    missing_values = {}
    missing_pct = {}
    negative_values = {}
    sensor_ranges = {}

    total_cells = total_records * max(1, len(sensor_columns))
    total_missing = 0
    total_negatives = 0

    for col in sensor_columns:
        if col in df.columns:
            m = int(df[col].isna().sum())
            missing_values[col] = m
            missing_pct[col] = round((m / total_records) * 100, 2) if total_records > 0 else 0
            total_missing += m

            neg = int((df[col] < 0).sum()) if pd.api.types.is_numeric_dtype(df[col]) else 0
            negative_values[col] = neg
            total_negatives += neg

            col_clean = df[col].dropna()
            if len(col_clean) > 0:
                sensor_ranges[col] = {
                    "min": round(float(col_clean.min()), 2),
                    "max": round(float(col_clean.max()), 2),
                    "mean": round(float(col_clean.mean()), 2),
                    "std": round(float(col_clean.std()), 2)
                }
            else:
                sensor_ranges[col] = {"min": None, "max": None, "mean": None, "std": None}

    # Overall Quality Score out of 100
    missing_ratio = (total_missing / max(1, total_cells))
    quality_score = max(0, min(100, round(100 - (missing_ratio * 100 * 2) - (duplicates * 5), 1)))

    return {
        "total_records": total_records,
        "duplicate_records": duplicates,
        "missing_values": missing_values,
        "missing_percentage": missing_pct,
        "negative_values": negative_values,
        "sensor_ranges": sensor_ranges,
        "overall_quality_score": quality_score,
        "sensor_columns": sensor_columns
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    sensor_columns = col_info["sensor_cols"]
    results = analyze_data_quality(df, sensor_columns)

    print("=" * 60)
    print("SENTINEL - DATA QUALITY ANALYZER")
    print("=" * 60)

    # 1. Missing values
    print("\n[1] MISSING VALUES")
    for column in sensor_columns:
        print(f"{column:25} : {results['missing_values'].get(column, 0)}")

    # 2. Duplicate records
    print("\n[2] DUPLICATE RECORDS")
    print("Duplicate rows:", results["duplicate_records"])

    # 3. Negative values
    print("\n[3] NEGATIVE SENSOR VALUES")
    for column in sensor_columns:
        print(f"{column:25} : {results['negative_values'].get(column, 0)}")

    # 4. Basic range information
    print("\n[4] SENSOR RANGES")
    for column in sensor_columns:
        r = results["sensor_ranges"].get(column, {})
        min_val = r.get("min", 0.0)
        max_val = r.get("max", 0.0)
        print(
            f"{column:25} : "
            f"Min = {min_val:.2f}, "
            f"Max = {max_val:.2f}"
        )

    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS COMPLETE")
    print("=" * 60)
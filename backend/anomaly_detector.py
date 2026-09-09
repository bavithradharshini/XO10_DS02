import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def detect_anomalies(df, sensor_columns=None):
    if sensor_columns is None:
        detected = detect_columns(df)
        sensor_columns = detected["sensor_cols"]

    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    sensor_stats = {}
    z_scores = pd.DataFrame(index=df.index)
    anomaly_mask = pd.DataFrame(index=df.index)
    severe_anomaly_mask = pd.DataFrame(index=df.index)

    for column in valid_cols:
        col_series = df[column]
        mean = float(col_series.mean()) if not col_series.dropna().empty else 0.0
        std = float(col_series.std()) if not col_series.dropna().empty else 1.0

        if std > 0:
            z = ((col_series - mean) / std)
        else:
            z = pd.Series(0.0, index=df.index)

        z_abs = z.abs()
        is_mod = (z_abs > 2) & z_abs.notna()
        is_sev = (z_abs > 3) & z_abs.notna()

        z_scores[column] = z
        anomaly_mask[column] = is_mod
        severe_anomaly_mask[column] = is_sev

        sensor_stats[column] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "moderate_anomalies": int(is_mod.sum()),
            "severe_anomalies": int(is_sev.sum()),
            "anomalies": int(is_sev.sum())  # to match existing anomaly count
        }

    total_anomalous_records = int(severe_anomaly_mask.any(axis=1).sum()) if not severe_anomaly_mask.empty else 0

    return {
        "sensor_stats": sensor_stats,
        "z_scores": z_scores,
        "anomaly_mask": anomaly_mask,
        "severe_anomaly_mask": severe_anomaly_mask,
        "total_anomalous_records": total_anomalous_records
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    sensor_columns = col_info["sensor_cols"]

    print("=" * 60)
    print("SENTINEL - ANOMALY DETECTOR")
    print("=" * 60)

    for column in sensor_columns:
        mean = df[column].mean()
        std = df[column].std()
        df[f"{column}_zscore"] = (df[column] - mean) / std
        anomalies = (df[f"{column}_zscore"].abs() > 3).sum()

        print(
            f"{column:25} : "
            f"{anomalies} anomalies"
        )

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 60)
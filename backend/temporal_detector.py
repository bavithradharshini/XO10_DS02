import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def detect_temporal_inconsistencies(df, sensor_columns=None, station_col=None, timestamp_col=None):
    if sensor_columns is None or station_col is None or timestamp_col is None:
        detected = detect_columns(df)
        sensor_columns = sensor_columns or detected["sensor_cols"]
        station_col = station_col or detected["station_col"]
        timestamp_col = timestamp_col or detected["timestamp_col"]

    stats_by_sensor = {}
    sudden_change_mask = pd.DataFrame(index=df.index)

    for column in sensor_columns:
        if column not in df.columns:
            continue

        if station_col and station_col in df.columns:
            # Group by station to avoid cross-station difference jump
            change = df.groupby(station_col)[column].diff().abs()
        else:
            change = df[column].diff().abs()

        mean_change = float(change.mean()) if not change.dropna().empty else 0.0
        std_change = float(change.std()) if not change.dropna().empty else 0.0
        threshold = mean_change + (3 * std_change)

        is_sudden = (change > threshold) & change.notna()
        sudden_change_mask[column] = is_sudden

        stats_by_sensor[column] = {
            "mean_change": round(mean_change, 3),
            "std_change": round(std_change, 3),
            "threshold": round(threshold, 3),
            "sudden_changes": int(is_sudden.sum())
        }

    return {
        "sensor_stats": stats_by_sensor,
        "sudden_change_mask": sudden_change_mask,
        "total_sudden_events": int(sudden_change_mask.any(axis=1).sum()) if not sudden_change_mask.empty else 0
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    sensor_columns = col_info["sensor_cols"]

    print("=" * 60)
    print("SENTINEL - TEMPORAL CONSISTENCY DETECTOR")
    print("=" * 60)

    # Calculate global differences to match original standalone CLI baseline
    for column in sensor_columns:
        change = df[column].diff().abs()
        mean_change = change.mean()
        std_change = change.std()
        threshold = mean_change + (3 * std_change)
        sudden_changes = (change > threshold).sum()

        print(
            f"{column:25} : "
            f"{sudden_changes} sudden changes"
        )

    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 60)
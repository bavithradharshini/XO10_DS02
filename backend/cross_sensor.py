import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def analyze_cross_sensor(df, sensor_columns=None):
    if sensor_columns is None:
        detected = detect_columns(df)
        sensor_columns = detected["sensor_cols"]

    # Filter to existing numeric columns
    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(valid_cols) < 2:
        return {
            "correlation_matrix": {},
            "related_pairs": [],
            "inconsistency_mask": pd.Series(False, index=df.index),
            "inconsistent_rows_count": 0
        }

    correlation = df[valid_cols].corr()

    # Identify highly correlated pairs (|r| >= 0.5)
    related_pairs = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            sa = valid_cols[i]
            sb = valid_cols[j]
            val = correlation.loc[sa, sb]
            if pd.notna(val) and abs(val) >= 0.5:
                related_pairs.append({
                    "sensor_a": sa,
                    "sensor_b": sb,
                    "correlation": round(float(val), 2)
                })

    # Row-level inconsistency detection using standardized Z-scores
    # If two sensors have strong correlation, their normalized values should not diverge violently
    z_scores = pd.DataFrame(index=df.index)
    for col in valid_cols:
        col_series = df[col]
        mean = col_series.mean()
        std = col_series.std()
        if pd.notna(std) and std > 0:
            z_scores[col] = (col_series - mean) / std
        else:
            z_scores[col] = 0.0

    inconsistency_flags = pd.Series(False, index=df.index)
    inconsistency_details = [[] for _ in range(len(df))]

    for pair in related_pairs:
        sa = pair["sensor_a"]
        sb = pair["sensor_b"]
        r = pair["correlation"]

        za = z_scores[sa]
        zb = z_scores[sb]

        # Valid rows where both are non-null
        valid_mask = za.notna() & zb.notna()

        if r > 0:
            # Positive correlation: difference should not be extreme
            diff = (za - zb).abs()
            conflict = valid_mask & (diff > 2.5)
        else:
            # Negative correlation: sum should not be extreme
            diff = (za + zb).abs()
            conflict = valid_mask & (diff > 2.5)

        if conflict.any():
            inconsistency_flags = inconsistency_flags | conflict
            conflict_indices = np.where(conflict)[0]
            for idx in conflict_indices:
                inconsistency_details[idx].append(f"Conflict between {sa} and {sb} (r={r})")

    corr_dict = {
        col: {c: round(float(correlation.loc[col, c]), 2) if pd.notna(correlation.loc[col, c]) else None
              for c in valid_cols}
        for col in valid_cols
    }

    return {
        "correlation_matrix": corr_dict,
        "related_pairs": related_pairs,
        "inconsistency_mask": inconsistency_flags,
        "inconsistency_details": inconsistency_details,
        "inconsistent_rows_count": int(inconsistency_flags.sum())
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    sensor_columns = col_info["sensor_cols"]

    print("=" * 60)
    print("SENTINEL - CROSS SENSOR CONSISTENCY")
    print("=" * 60)

    correlation = df[sensor_columns].corr()

    print("\nSENSOR CORRELATION MATRIX")
    print(correlation.round(2).to_string())

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
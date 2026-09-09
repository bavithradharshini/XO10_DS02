import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
    from backend.anomaly_detector import detect_anomalies
    from backend.temporal_detector import detect_temporal_inconsistencies
    from backend.cross_sensor import analyze_cross_sensor
except ImportError:
    from data_inspection import load_dataset, detect_columns
    from anomaly_detector import detect_anomalies
    from temporal_detector import detect_temporal_inconsistencies
    from cross_sensor import analyze_cross_sensor


def reliability_level(score):
    if score >= 80:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_trust_scores(df, sensor_columns=None):
    orig_index = df.index
    df_result = df.reset_index(drop=True).copy()

    detected = detect_columns(df_result)
    if sensor_columns is None:
        sensor_columns = detected["sensor_cols"]
    station_col = detected["station_col"]
    timestamp_col = detected["timestamp_col"]

    valid_cols = [c for c in sensor_columns if c in df_result.columns and pd.api.types.is_numeric_dtype(df_result[c])]

    # 1. Run detectors
    anomaly_res = detect_anomalies(df_result, valid_cols)
    temporal_res = detect_temporal_inconsistencies(df_result, valid_cols, station_col, timestamp_col)
    cross_res = analyze_cross_sensor(df_result, valid_cols)

    z_scores = anomaly_res["z_scores"]
    sudden_mask = temporal_res["sudden_change_mask"]
    cross_mask = cross_res["inconsistency_mask"]
    cross_details = cross_res["inconsistency_details"]

    n_records = len(df_result)
    trust_scores = np.full(n_records, 100.0)
    deductions_list = [[] for _ in range(n_records)]

    # 2. Deduct points with full explainability
    for i in range(n_records):
        # A. Missing Data Deductions (-20 pts per missing sensor, max 40 pts)
        missing_penalties = 0
        for col in valid_cols:
            if pd.isna(df_result[col].iloc[i]):
                if missing_penalties < 40:
                    trust_scores[i] -= 20
                    missing_penalties += 20
                    deductions_list[i].append({
                        "factor": "Missing Data",
                        "sensor": col,
                        "penalty": 20,
                        "reason": f"{col} value is missing (-20 pts)"
                    })

        # B. Statistical Outliers Deductions
        for col in valid_cols:
            if col in z_scores.columns:
                z = z_scores[col].iloc[i]
                if pd.notna(z):
                    z_abs = abs(z)
                    if z_abs > 3:
                        trust_scores[i] -= 30
                        deductions_list[i].append({
                            "factor": "Statistical Outlier",
                            "sensor": col,
                            "penalty": 30,
                            "reason": f"{col} extreme outlier (|Z|={z_abs:.2f} > 3) (-30 pts)"
                        })
                    elif z_abs > 2:
                        trust_scores[i] -= 15
                        deductions_list[i].append({
                            "factor": "Statistical Outlier",
                            "sensor": col,
                            "penalty": 15,
                            "reason": f"{col} moderate outlier (|Z|={z_abs:.2f} > 2) (-15 pts)"
                        })

        # C. Sudden Temporal Changes Deductions (-15 pts per sensor)
        for col in valid_cols:
            if col in sudden_mask.columns and bool(sudden_mask[col].iloc[i]):
                trust_scores[i] -= 15
                deductions_list[i].append({
                    "factor": "Temporal Inconsistency",
                    "sensor": col,
                    "penalty": 15,
                    "reason": f"{col} sudden rate-of-change spike (-15 pts)"
                })

        # D. Cross-Sensor Inconsistency Deductions (-15 pts)
        if bool(cross_mask.iloc[i]):
            trust_scores[i] -= 15
            details_str = ", ".join(cross_details[i]) if cross_details[i] else "Channel correlation conflict"
            deductions_list[i].append({
                "factor": "Cross-Sensor Inconsistency",
                "sensor": "Multiple",
                "penalty": 15,
                "reason": f"{details_str} (-15 pts)"
            })

    # Clip scores to [0, 100]
    trust_scores = np.clip(trust_scores, 0, 100)

    df_result["Trust_Score"] = np.round(trust_scores, 1)
    df_result["Reliability"] = df_result["Trust_Score"].apply(reliability_level)
    df_result["Deductions"] = deductions_list
    df_result["Deduction_Reasons"] = [
        "; ".join([d["reason"] for d in dlist]) if dlist else "None (Optimal confidence)"
        for dlist in deductions_list
    ]

    df_result.index = orig_index
    return df_result


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    df_scored = calculate_trust_scores(df)

    print("=" * 60)
    print("SENTINEL - TRUST SCORE ENGINE")
    print("=" * 60)

    print("\nTOTAL OBSERVATIONS:", len(df_scored))

    print("\nRELIABILITY DISTRIBUTION")
    print(df_scored["Reliability"].value_counts().to_string())

    print("\nSAMPLE RESULTS WITH EXPLAINABILITY")
    sample_cols = ["Station_ID", "Timestamp", "Trust_Score", "Reliability", "Deduction_Reasons"]
    available_sample_cols = [c for c in sample_cols if c in df_scored.columns]
    print(df_scored[available_sample_cols].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TRUST SCORE GENERATION COMPLETE")
    print("=" * 60)
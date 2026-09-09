import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
    from backend.trust_score import calculate_trust_scores
    from backend.drift_detector import detect_drift
except ImportError:
    from data_inspection import load_dataset, detect_columns
    from trust_score import calculate_trust_scores
    from drift_detector import detect_drift


def generate_alerts(df_scored=None, drift_results=None, max_observation_alerts=50):
    if df_scored is None:
        df = load_dataset()
        df_scored = calculate_trust_scores(df)

    if drift_results is None:
        drift_results = detect_drift(df_scored)

    detected = detect_columns(df_scored)
    station_col = detected["station_col"] or "Station_ID"
    timestamp_col = detected["timestamp_col"] or "Timestamp"

    alerts = []
    alert_counter = 1

    # 1. System-level Drift Alerts
    sensors_drift = drift_results.get("sensors", {})
    for col, dinfo in sensors_drift.items():
        status = dinfo.get("drift_status")
        if status == "HIGH_DRIFT":
            alerts.append({
                "id": f"ALT-{alert_counter:04d}",
                "timestamp": str(df_scored[timestamp_col].iloc[-1]) if timestamp_col in df_scored else "Recent",
                "station_id": "All Stations",
                "sensor": col,
                "severity": "CRITICAL",
                "category": "SENSOR_DRIFT",
                "message": f"Significant baseline drift detected in {col} ({dinfo['mean_shift']:+0.2f}, Z={dinfo['shift_zscore']})",
                "recommendation": "Recalibrate sensor hardware and inspect zero-point baseline."
            })
            alert_counter += 1
        elif status == "MODERATE_DRIFT":
            alerts.append({
                "id": f"ALT-{alert_counter:04d}",
                "timestamp": str(df_scored[timestamp_col].iloc[-1]) if timestamp_col in df_scored else "Recent",
                "station_id": "All Stations",
                "sensor": col,
                "severity": "WARNING",
                "category": "SENSOR_DRIFT",
                "message": f"Moderate baseline drift trending {dinfo['drift_direction']} in {col} ({dinfo['mean_shift']:+0.2f})",
                "recommendation": "Monitor sensor drift rate; schedule calibration check."
            })
            alert_counter += 1

    # 2. Critical Observation Alerts (Low Trust Score < 50)
    low_trust_df = df_scored[df_scored["Trust_Score"] < 50]
    # Sample or take most severe / most recent
    selected_low = low_trust_df.sort_values("Trust_Score").head(max_observation_alerts)

    for idx, row in selected_low.iterrows():
        stn = str(row[station_col]) if station_col in row else "Unknown"
        ts = str(row[timestamp_col]) if timestamp_col in row else f"Row {idx}"
        score = float(row["Trust_Score"])
        reasons = str(row.get("Deduction_Reasons", "Severe measurement anomalies detected"))

        alerts.append({
            "id": f"ALT-{alert_counter:04d}",
            "row_index": int(idx),
            "trust_score": round(score, 1),
            "timestamp": ts,
            "station_id": stn,
            "sensor": "Multi-Sensor",
            "severity": "CRITICAL" if score <= 30 else "WARNING",
            "category": "TRUST_DEGRADATION",
            "message": f"Low data trust score ({score}/100) at station {stn}: {reasons}",
            "recommendation": "Exclude measurement from downstream analytics and review station telemetry."
        })
        alert_counter += 1

    # Sort alerts by severity (CRITICAL first, then WARNING, then INFO)
    severity_rank = {"CRITICAL": 1, "WARNING": 2, "INFO": 3}
    alerts.sort(key=lambda x: severity_rank.get(x["severity"], 99))

    summary = {
        "total_alerts": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
        "warning_count": sum(1 for a in alerts if a["severity"] == "WARNING"),
        "info_count": sum(1 for a in alerts if a["severity"] == "INFO")
    }

    return {
        "summary": summary,
        "alerts": alerts
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    df_scored = calculate_trust_scores(df)
    drift = detect_drift(df_scored)
    result = generate_alerts(df_scored, drift)

    print("=" * 60)
    print("SENTINEL - ALERT ENGINE")
    print("=" * 60)

    print(f"\nALERT SUMMARY:")
    print(f"Total Active Alerts: {result['summary']['total_alerts']}")
    print(f"Critical Alerts:     {result['summary']['critical_count']}")
    print(f"Warnings:            {result['summary']['warning_count']}")

    print("\nSAMPLE RECENT ALERTS:")
    for alert in result["alerts"][:5]:
        print(f"[{alert['severity']}] {alert['id']} | {alert['station_id']} | {alert['message']}")

    print("\n" + "=" * 60)
    print("ALERT GENERATION COMPLETE")
    print("=" * 60)

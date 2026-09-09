import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
    from backend.trust_score import calculate_trust_scores
    from backend.alert_engine import generate_alerts
except ImportError:
    from data_inspection import load_dataset, detect_columns
    from trust_score import calculate_trust_scores
    from alert_engine import generate_alerts


def calculate_station_health(df_scored=None, alerts_list=None):
    if df_scored is None:
        df = load_dataset()
        df_scored = calculate_trust_scores(df)

    if alerts_list is None:
        alerts_data = generate_alerts(df_scored)
        alerts_list = alerts_data["alerts"]

    detected = detect_columns(df_scored)
    station_col = detected["station_col"]
    sensor_cols = detected["sensor_cols"]

    if not station_col or station_col not in df_scored.columns:
        # Fallback if no station column exists
        avg_trust = float(df_scored["Trust_Score"].mean())
        return [{
            "station_id": "DEFAULT_STATION",
            "health_score": round(avg_trust, 1),
            "status": "OPTIMAL" if avg_trust >= 85 else "GOOD",
            "total_readings": len(df_scored),
            "average_trust_score": round(avg_trust, 1),
            "reliability_distribution": df_scored["Reliability"].value_counts().to_dict(),
            "active_alerts": len(alerts_list)
        }]

    stations = sorted(df_scored[station_col].dropna().unique().tolist())
    station_health_list = []

    for stn in stations:
        stn_df = df_scored[df_scored[station_col] == stn]
        total_obs = len(stn_df)
        if total_obs == 0:
            continue

        avg_trust = float(stn_df["Trust_Score"].mean())
        low_count = int((stn_df["Reliability"] == "LOW").sum())
        med_count = int((stn_df["Reliability"] == "MEDIUM").sum())
        high_count = int((stn_df["Reliability"] == "HIGH").sum())

        # Count active alerts for this station
        stn_alerts = [a for a in alerts_list if a.get("station_id") == stn]
        crit_alerts = sum(1 for a in stn_alerts if a.get("severity") == "CRITICAL")

        # Health score: base average trust with penalty for critical alerts
        health_score = max(0.0, min(100.0, round(avg_trust - (crit_alerts * 0.5), 1)))

        if health_score >= 90:
            status = "OPTIMAL"
        elif health_score >= 75:
            status = "GOOD"
        elif health_score >= 50:
            status = "DEGRADED"
        else:
            status = "CRITICAL"

        # Sensor averages for this station
        sensor_averages = {}
        for col in sensor_cols:
            if col in stn_df.columns and pd.api.types.is_numeric_dtype(stn_df[col]):
                s_mean = stn_df[col].dropna().mean()
                sensor_averages[col] = round(float(s_mean), 2) if pd.notna(s_mean) else None

        station_health_list.append({
            "station_id": stn,
            "health_score": health_score,
            "status": status,
            "total_readings": total_obs,
            "average_trust_score": round(avg_trust, 1),
            "reliability_distribution": {
                "HIGH": high_count,
                "MEDIUM": med_count,
                "LOW": low_count
            },
            "active_alerts": len(stn_alerts),
            "critical_alerts": crit_alerts,
            "sensor_averages": sensor_averages
        })

    return station_health_list


def get_station_detail(station_id, df_scored=None, alerts_list=None, recent_limit=20):
    if df_scored is None:
        df = load_dataset()
        df_scored = calculate_trust_scores(df)

    if alerts_list is None:
        alerts_data = generate_alerts(df_scored)
        alerts_list = alerts_data["alerts"]

    detected = detect_columns(df_scored)
    station_col = detected["station_col"]

    all_stations_health = calculate_station_health(df_scored, alerts_list)
    match_health = next((s for s in all_stations_health if s["station_id"] == station_id), None)

    if not match_health:
        return None

    stn_df = df_scored[df_scored[station_col] == station_id] if station_col else df_scored
    stn_alerts = [a for a in alerts_list if a.get("station_id") == station_id]

    # Convert recent readings to dict records
    recent_readings = (
        stn_df.tail(recent_limit)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "station_id": station_id,
        "health_summary": match_health,
        "recent_readings": recent_readings,
        "alerts": stn_alerts
    }


if __name__ == "__main__":
    df = load_dataset()
    col_info = detect_columns(df)
    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])

    df_scored = calculate_trust_scores(df)
    health_data = calculate_station_health(df_scored)

    print("=" * 60)
    print("SENTINEL - STATION HEALTH ANALYZER")
    print("=" * 60)

    for stn in health_data:
        print(
            f"Station: {stn['station_id']:8} | Health: {stn['health_score']:5.1f} | "
            f"Status: {stn['status']:9} | Avg Trust: {stn['average_trust_score']:5.1f} | "
            f"Alerts: {stn['active_alerts']}"
        )

    print("\n" + "=" * 60)
    print("STATION HEALTH COMPLETE")
    print("=" * 60)

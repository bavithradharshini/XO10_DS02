import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns, inspect_dataset
    from backend.data_quality import analyze_data_quality
    from backend.trust_score import calculate_trust_scores
    from backend.drift_detector import detect_drift, analyze_channel_drift
    from backend.alert_engine import generate_alerts
    from backend.station_health import calculate_station_health, get_station_detail
    from backend.false_alarm_classifier import classify_false_alarms
except ImportError:
    from data_inspection import load_dataset, detect_columns, inspect_dataset
    from data_quality import analyze_data_quality
    from trust_score import calculate_trust_scores
    from drift_detector import detect_drift, analyze_channel_drift
    from alert_engine import generate_alerts
    from station_health import calculate_station_health, get_station_detail
    from false_alarm_classifier import classify_false_alarms

app = Flask(__name__)
CORS(app)

# Pipeline In-Memory Cache
_PIPELINE_CACHE = {
    "df_raw": None,
    "df_scored": None,
    "col_info": None,
    "quality": None,
    "drift": None,
    "alerts": None,
    "stations": None,
    "false_alarms": None
}


def process_pipeline_data(df):
    col_info = detect_columns(df)

    if col_info["timestamp_col"] and col_info["timestamp_col"] in df.columns:
        try:
            df[col_info["timestamp_col"]] = pd.to_datetime(df[col_info["timestamp_col"]])
        except Exception:
            pass

    df_scored = calculate_trust_scores(df, col_info["sensor_cols"])
    quality = analyze_data_quality(df, col_info["sensor_cols"])
    drift = detect_drift(df, col_info["sensor_cols"])
    alerts_data = generate_alerts(df_scored, drift)
    stations = calculate_station_health(df_scored, alerts_data["alerts"])
    false_alarms = classify_false_alarms(df, col_info["sensor_cols"])

    return {
        "df_raw": df,
        "df_scored": df_scored,
        "col_info": col_info,
        "quality": quality,
        "drift": drift,
        "alerts": alerts_data,
        "stations": stations,
        "false_alarms": false_alarms
    }


def get_pipeline(force_reload=False):
    global _PIPELINE_CACHE
    if force_reload or _PIPELINE_CACHE["df_scored"] is None:
        df = load_dataset()
        _PIPELINE_CACHE = process_pipeline_data(df)

    return _PIPELINE_CACHE


@app.route("/")
def home():
    return jsonify({
        "message": "SENTINEL API is running",
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/summary",
            "/api/readings",
            "/api/quality",
            "/api/trust",
            "/api/drift",
            "/api/drift/analysis",
            "/api/alerts",
            "/api/stations",
            "/api/stations/<station_id>",
            "/api/upload",
            "/api/classify",
            "/api/false-alarms",
            "/dashboard"
        ]
    })


@app.route("/dashboard")
def dashboard():
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    return send_from_directory(frontend_dir, "index.html")


@app.route("/api/health")
def health():
    pipeline = get_pipeline()
    df_scored = pipeline["df_scored"]
    return jsonify({
        "status": "healthy",
        "service": "SENTINEL Sensor Intelligence Engine",
        "version": "1.0.0",
        "dataset_loaded": True,
        "total_records": len(df_scored),
        "total_stations": len(pipeline["stations"]),
        "active_alerts": pipeline["alerts"]["summary"]["total_alerts"]
    })


@app.route("/api/classify")
@app.route("/api/false-alarms")
def get_false_alarms():
    pipeline = get_pipeline()
    fa_data = pipeline.get("false_alarms")
    if fa_data is None:
        fa_data = classify_false_alarms(pipeline["df_raw"], pipeline["col_info"]["sensor_cols"])
        pipeline["false_alarms"] = fa_data

    station_id = request.args.get("station_id", default=None, type=str)
    channel = request.args.get("channel", default=None, type=str)
    classification = request.args.get("classification", default=None, type=str)
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)

    items = fa_data["classifications"]
    if station_id:
        items = [x for x in items if x.get("station_id") == station_id]
    if channel:
        items = [x for x in items if x.get("channel", "").lower() == channel.lower()]
    if classification:
        items = [x for x in items if x.get("classification", "").upper() == classification.upper()]

    paged = items[offset: offset + limit]

    return jsonify({
        "summary": fa_data["summary"],
        "filtered_count": len(items),
        "limit": limit,
        "offset": offset,
        "classifications": paged
    })


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    global _PIPELINE_CACHE
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded. Please provide a CSV file under form-data key 'file'."
            }), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected for upload."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Unsupported file format. Only CSV (.csv) files are supported."}), 400

        try:
            new_df = pd.read_csv(file)
        except Exception as read_err:
            return jsonify({"error": f"Failed to parse CSV file: {str(read_err)}"}), 400

        if new_df.empty:
            return jsonify({"error": "The uploaded CSV file contains no data rows."}), 400

        # Validate minimum required columns:
        # Station_ID, Timestamp, and at least one sensor channel
        detected = detect_columns(new_df)
        station_col = detected.get("station_col")
        timestamp_col = detected.get("timestamp_col")
        sensor_cols = detected.get("sensor_cols", [])

        # If sensor_cols is empty, attempt to convert potential numeric object columns
        if not sensor_cols:
            for c in new_df.columns:
                if c not in [station_col, timestamp_col]:
                    try:
                        converted = pd.to_numeric(new_df[c], errors="coerce")
                        if converted.notna().sum() > 0:
                            new_df[c] = converted
                            sensor_cols.append(c)
                    except Exception:
                        pass

        missing_fields = []
        if not station_col or station_col not in new_df.columns:
            missing_fields.append("Station identifier (e.g. 'Station_ID' or 'Station')")

        if not timestamp_col or timestamp_col not in new_df.columns:
            missing_fields.append("Timestamp (e.g. 'Timestamp' or 'DateTime')")

        if not sensor_cols:
            missing_fields.append("At least one numeric sensor channel")

        if missing_fields:
            return jsonify({
                "error": f"CSV validation failed. Missing required fields: {', '.join(missing_fields)}.",
                "detected_columns": list(new_df.columns),
                "required": ["Station_ID", "Timestamp", "At least one numeric sensor channel"]
            }), 400

        # Run through identical pipeline
        new_pipeline = process_pipeline_data(new_df)
        _PIPELINE_CACHE = new_pipeline

        df_scored = new_pipeline["df_scored"]
        col_info = new_pipeline["col_info"]
        alerts = new_pipeline["alerts"]
        stations = new_pipeline["stations"]

        return jsonify({
            "success": True,
            "message": f"Successfully loaded and analyzed '{file.filename}'.",
            "filename": file.filename,
            "total_records": len(df_scored),
            "stations_count": len(stations),
            "sensors_count": len(col_info["sensor_cols"]),
            "sensor_names": col_info["sensor_cols"],
            "station_names": [s["station_id"] for s in stations],
            "average_trust_score": round(float(df_scored["Trust_Score"].mean()), 1),
            "active_alerts": alerts["summary"]["total_alerts"]
        })

    except Exception as e:
        return jsonify({"error": f"Error running data intelligence pipeline on uploaded CSV: {str(e)}"}), 500


@app.route("/api/summary")
def summary():
    pipeline = get_pipeline()
    df = pipeline["df_scored"]
    col_info = pipeline["col_info"]
    stations_data = pipeline["stations"]

    station_col = col_info["station_col"]
    stations_count = int(df[station_col].nunique()) if station_col and station_col in df.columns else 1
    sensor_cols = col_info["sensor_cols"]

    avg_trust = round(float(df["Trust_Score"].mean()), 1)
    reliability_dist = df["Reliability"].value_counts().to_dict()

    return jsonify({
        "total_records": len(df),
        "stations": stations_count,
        "sensors": len(sensor_cols),
        "sensor_names": sensor_cols,
        "station_names": [s["station_id"] for s in stations_data],
        "average_trust_score": avg_trust,
        "reliability_distribution": reliability_dist,
        "system_drift_status": pipeline["drift"].get("overall_status", "STABLE"),
        "critical_alerts": pipeline["alerts"]["summary"]["critical_count"]
    })


@app.route("/api/readings")
def readings():
    pipeline = get_pipeline()
    df_scored = pipeline["df_scored"]
    col_info = pipeline["col_info"]

    limit = request.args.get("limit", default=20, type=int)
    offset = request.args.get("offset", default=0, type=int)
    station_id = request.args.get("station_id", default=None, type=str)

    df_filtered = df_scored
    station_col = col_info["station_col"]
    if station_id and station_col and station_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[station_col] == station_id]

    # Convert timestamps to string format for clean JSON serialization
    paged_df = df_filtered.iloc[offset: offset + limit].copy()
    paged_df["row_index"] = paged_df.index
    ts_col = col_info["timestamp_col"]
    if ts_col and ts_col in paged_df.columns:
        paged_df[ts_col] = paged_df[ts_col].astype(str)

    data = paged_df.replace({np.nan: None}).to_dict(orient="records")

    return jsonify(data)


@app.route("/api/quality")
def quality():
    pipeline = get_pipeline()
    return jsonify(pipeline["quality"])


@app.route("/api/trust")
def trust():
    pipeline = get_pipeline()
    df_scored = pipeline["df_scored"]
    col_info = pipeline["col_info"]

    limit = request.args.get("limit", default=20, type=int)
    offset = request.args.get("offset", default=0, type=int)
    station_id = request.args.get("station_id", default=None, type=str)
    reliability_filter = request.args.get("reliability", default=None, type=str)

    df_view = df_scored
    station_col = col_info["station_col"]
    if station_id and station_col and station_col in df_view.columns:
        df_view = df_view[df_view[station_col] == station_id]

    if reliability_filter:
        df_view = df_view[df_view["Reliability"] == reliability_filter.upper()]

    paged_df = df_view.iloc[offset: offset + limit].copy()
    paged_df["row_index"] = paged_df.index
    ts_col = col_info["timestamp_col"]
    if ts_col and ts_col in paged_df.columns:
        paged_df[ts_col] = paged_df[ts_col].astype(str)

    sample_cols = [c for c in [station_col, ts_col, "Trust_Score", "Reliability", "Deduction_Reasons", "Deductions", "row_index"] if c and c in paged_df.columns]
    sample_records = paged_df[sample_cols].replace({np.nan: None}).to_dict(orient="records")

    return jsonify({
        "average_trust_score": round(float(df_scored["Trust_Score"].mean()), 1),
        "reliability_distribution": df_scored["Reliability"].value_counts().to_dict(),
        "total_scored_records": len(df_scored),
        "observations": sample_records
    })


@app.route("/api/trust/<int:reading_id>/explain")
@app.route("/api/trust/explain")
def trust_explain(reading_id=None):
    pipeline = get_pipeline()
    df_scored = pipeline["df_scored"]
    col_info = pipeline["col_info"]
    station_col = col_info["station_col"]
    timestamp_col = col_info["timestamp_col"]
    sensor_cols = col_info["sensor_cols"]

    index = reading_id if reading_id is not None else request.args.get("index", default=None, type=int)
    station_id = request.args.get("station_id", default=None, type=str)
    timestamp = request.args.get("timestamp", default=None, type=str)

    if index is not None:
        if 0 <= index < len(df_scored):
            row = df_scored.iloc[index]
        else:
            return jsonify({"error": f"Index {index} out of range"}), 404
    elif station_id and timestamp:
        matches = df_scored
        if station_col and station_col in matches.columns:
            matches = matches[matches[station_col] == station_id]
        if timestamp_col and timestamp_col in matches.columns:
            matches = matches[matches[timestamp_col].astype(str) == str(timestamp)]
        if matches.empty:
            return jsonify({"error": f"No observation found for {station_id} at {timestamp}"}), 404
        row = matches.iloc[0]
    elif station_id:
        matches = df_scored
        if station_col and station_col in matches.columns:
            matches = matches[matches[station_col] == station_id]
        if matches.empty:
            return jsonify({"error": f"Station '{station_id}' not found"}), 404

        avg_score = round(float(matches["Trust_Score"].mean()), 1)
        rel_dist = matches["Reliability"].value_counts().to_dict()

        factor_summary = {
            "Statistical Outlier": 0,
            "Temporal Inconsistency": 0,
            "Missing Data": 0,
            "Cross-Sensor Inconsistency": 0
        }
        for dlist in matches["Deductions"]:
            for d in dlist:
                f = d.get("factor")
                if f in factor_summary:
                    factor_summary[f] += 1

        sample_lowest = (
            matches.sort_values("Trust_Score")
            .head(5)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
        for r in sample_lowest:
            if timestamp_col and timestamp_col in r:
                r[timestamp_col] = str(r[timestamp_col])

        return jsonify({
            "type": "station",
            "station_id": station_id,
            "base_score": 100.0,
            "average_trust_score": avg_score,
            "reliability_distribution": rel_dist,
            "factor_summary": factor_summary,
            "sample_lowest_readings": sample_lowest
        })
    else:
        avg_score = round(float(df_scored["Trust_Score"].mean()), 1)
        rel_dist = df_scored["Reliability"].value_counts().to_dict()
        factor_summary = {
            "Statistical Outlier": 0,
            "Temporal Inconsistency": 0,
            "Missing Data": 0,
            "Cross-Sensor Inconsistency": 0
        }
        for dlist in df_scored["Deductions"]:
            for d in dlist:
                f = d.get("factor")
                if f in factor_summary:
                    factor_summary[f] += 1

        sample_lowest = (
            df_scored.sort_values("Trust_Score")
            .head(5)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
        for r in sample_lowest:
            if timestamp_col and timestamp_col in r:
                r[timestamp_col] = str(r[timestamp_col])

        return jsonify({
            "type": "platform",
            "station_id": "All Stations",
            "base_score": 100.0,
            "average_trust_score": avg_score,
            "reliability_distribution": rel_dist,
            "factor_summary": factor_summary,
            "sample_lowest_readings": sample_lowest
        })

    raw_deductions = row.get("Deductions", [])
    deductions_formatted = []
    total_penalty = 0

    for d in raw_deductions:
        p = d.get("penalty", 0)
        total_penalty += p
        deductions_formatted.append({
            "factor": d.get("factor", "Anomaly"),
            "sensor": d.get("sensor", "Unknown"),
            "penalty": -abs(p),
            "reason": d.get("reason", "")
        })

    sensor_values = {}
    for sc in sensor_cols:
        if sc in row and pd.notna(row[sc]):
            sensor_values[sc] = round(float(row[sc]), 2)
        else:
            sensor_values[sc] = None

    stn_val = str(row[station_col]) if station_col and station_col in row else "Unknown"
    ts_val = str(row[timestamp_col]) if timestamp_col and timestamp_col in row else "Unknown"
    final_score = round(float(row["Trust_Score"]), 1)
    reliability = str(row.get("Reliability", "HIGH"))

    return jsonify({
        "type": "observation",
        "station_id": stn_val,
        "timestamp": ts_val,
        "base_score": 100.0,
        "final_score": final_score,
        "reliability": reliability,
        "total_penalty": -abs(total_penalty),
        "deductions": deductions_formatted,
        "deduction_reasons": row.get("Deduction_Reasons", "None (Optimal confidence)"),
        "sensor_values": sensor_values
    })


@app.route("/api/drift")
def drift():
    pipeline = get_pipeline()
    station_id = request.args.get("station_id", default=None, type=str)
    channel = request.args.get("channel", default=None, type=str)
    if station_id:
        df = pipeline["df_scored"]
        col_info = pipeline["col_info"]
        station_col = col_info["station_col"]
        if station_col and station_col in df.columns:
            stn_df = df[df[station_col] == station_id]
            if not stn_df.empty:
                res = detect_drift(stn_df, col_info["sensor_cols"])
                if channel:
                    res["channel_drift_analysis"] = [
                        c for c in res.get("channel_drift_analysis", [])
                        if c.get("affected_channel", "").lower() == channel.lower()
                    ]
                return jsonify(res)
    res = pipeline["drift"]
    if channel:
        res_copy = dict(res)
        res_copy["channel_drift_analysis"] = [
            c for c in res.get("channel_drift_analysis", [])
            if c.get("affected_channel", "").lower() == channel.lower()
        ]
        return jsonify(res_copy)
    return jsonify(res)


@app.route("/api/drift/analysis")
def drift_analysis():
    pipeline = get_pipeline()
    station_id = request.args.get("station_id", default=None, type=str)
    channel = request.args.get("channel", default=None, type=str)
    drift_status_filter = request.args.get("drift_status", default=None, type=str)

    col_info = pipeline["col_info"]
    df = pipeline["df_scored"]
    station_col = col_info["station_col"]

    if station_id and station_col and station_col in df.columns:
        stn_df = df[df[station_col] == station_id]
        if stn_df.empty:
            return jsonify({"error": f"Station '{station_id}' not found"}), 404
        analysis_data = analyze_channel_drift(stn_df, col_info["sensor_cols"], station_id=station_id)
    else:
        drift_data = pipeline["drift"]
        analysis_data = {
            "drift_status": drift_data.get("overall_status", "SYSTEM_STABLE"),
            "total_channels": len(drift_data.get("channel_drift_analysis", [])),
            "drift_detected_count": len(drift_data.get("detected_drift_channels", [])),
            "station_id": station_id or "All Stations",
            "channels": drift_data.get("channel_drift_analysis", []),
            "by_station": drift_data.get("by_station", {})
        }

    channels = list(analysis_data.get("channels", []))
    if channel:
        channels = [c for c in channels if c.get("affected_channel", "").lower() == channel.lower()]
    if drift_status_filter:
        channels = [c for c in channels if c.get("drift_status", "").upper() == drift_status_filter.upper()]

    return jsonify({
        "overall_drift_status": analysis_data.get("drift_status"),
        "total_channels_analyzed": len(channels),
        "drift_detected_count": sum(1 for c in channels if c.get("drift_status") == "DRIFT_DETECTED"),
        "station_id": analysis_data.get("station_id", "All Stations"),
        "channels": channels,
        "by_station": analysis_data.get("by_station", {})
    })


@app.route("/api/alerts")
def alerts():
    pipeline = get_pipeline()
    alerts_data = pipeline["alerts"]

    severity = request.args.get("severity", default=None, type=str)
    station_id = request.args.get("station_id", default=None, type=str)

    all_alerts = alerts_data["alerts"]

    if severity:
        all_alerts = [a for a in all_alerts if a["severity"].upper() == severity.upper()]

    if station_id:
        all_alerts = [a for a in all_alerts if a["station_id"] == station_id or a["station_id"] == "All Stations"]

    return jsonify({
        "summary": alerts_data["summary"],
        "filtered_count": len(all_alerts),
        "alerts": all_alerts
    })


@app.route("/api/stations")
def stations():
    pipeline = get_pipeline()
    return jsonify(pipeline["stations"])


@app.route("/api/stations/<station_id>")
def station_detail(station_id):
    pipeline = get_pipeline()
    detail = get_station_detail(
        station_id=station_id,
        df_scored=pipeline["df_scored"],
        alerts_list=pipeline["alerts"]["alerts"]
    )
    if detail is None:
        return jsonify({"error": f"Station '{station_id}' not found"}), 404
    return jsonify(detail)


if __name__ == "__main__":
    # Preload pipeline on startup for instant responses
    print("Preloading SENTINEL data intelligence pipeline...")
    get_pipeline()
    print("SENTINEL pipeline ready. Starting server on http://0.0.0.0:5001")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
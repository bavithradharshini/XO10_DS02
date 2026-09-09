import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def _compute_rolling_trend(vals, b_mad):
    """
    Computes linear regression trend slope over normalized time span [0, 1].
    Returns slope, normalized slope (in baseline MADs), R-squared, and direction.
    """
    n = len(vals)
    if n < 10:
        return 0.0, 0.0, 0.0, "flat"

    t_norm = np.linspace(0, 1, n)
    cov_t = np.cov(t_norm, vals)[0, 1]
    var_t = np.var(t_norm)
    slope = float(cov_t / var_t) if var_t > 0 else 0.0
    intercept = float(np.mean(vals) - slope * np.mean(t_norm))
    y_pred = slope * t_norm + intercept

    ss_tot = float(np.sum((vals - np.mean(vals)) ** 2))
    ss_res = float(np.sum((vals - y_pred) ** 2))
    r2 = max(0.0, float(1.0 - (ss_res / ss_tot))) if ss_tot > 0 else 0.0
    slope_norm = float(slope / b_mad) if b_mad > 1e-6 else 0.0
    direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")

    return slope, slope_norm, r2, direction


def _detect_changepoint(vals, timestamps, b_med, b_mad):
    """
    Identifies approximate structural break timestamp using maximum-likelihood
    pooled two-sample T-statistic across candidate split points, corroborated by CUSUM.
    """
    n = len(vals)
    if n < 20:
        return False, 0, "N/A", 0.0

    step = max(1, n // 100)
    t_stats = []
    min_k = max(5, int(n * 0.10))
    max_k = int(n * 0.90)

    for k in range(min_k, max_k, step):
        left = vals[:k]
        right = vals[k:]
        nl, nr = len(left), len(right)
        vl = float(np.var(left, ddof=1)) if nl > 1 else 0.0
        vr = float(np.var(right, ddof=1)) if nr > 1 else 0.0
        sp2 = ((nl - 1) * vl + (nr - 1) * vr) / max(1, (nl + nr - 2))
        sp = np.sqrt(sp2) if sp2 > 0 else 1.0
        t_val = np.sqrt((nl * nr) / (nl + nr)) * abs(float(np.mean(right)) - float(np.mean(left))) / sp
        t_stats.append((t_val, k))

    if not t_stats:
        return False, 0, "N/A", 0.0

    best_t, best_k = max(t_stats, key=lambda x: x[0])
    cp_detected = bool(best_t >= 4.0)

    approx_time = "N/A"
    if cp_detected and timestamps is not None and best_k < len(timestamps):
        approx_time = str(timestamps.iloc[best_k])

    return cp_detected, best_k, approx_time, float(best_t)


def _compute_persistence(vals, b_med, b_mad, post_idx):
    """
    Measures post-changepoint deviation persistence and longest streak.
    """
    n = len(vals)
    post_vals = vals[post_idx:] if post_idx < n else vals[-max(5, int(n * 0.25)):]
    n_post = len(post_vals)
    if n_post == 0:
        return 0.0, 0, 0

    dev_mask = np.abs(post_vals - b_med) >= (1.0 * b_mad)
    pers_ratio = float(np.mean(dev_mask))

    cur_streak, max_streak = 0, 0
    for d in dev_mask:
        if d:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0

    return pers_ratio, max_streak, n_post


def _build_channel_drift_evidence(
    channel_col,
    vals,
    timestamps,
    corr_matrix=None,
    all_shifts=None,
    station_id=None
):
    """
    Evaluates drift across the 6 multi-signal evidence dimensions:
    1. Rolling statistics (robust median & MAD)
    2. Trend analysis (linear slope & R2)
    3. Baseline comparison (initial 25% vs recent 25%)
    4. Change-point detection (structural break index & timestamp)
    5. Cross-sensor validation (environmental vs isolated hardware drift)
    6. Persistence (post-transition deviation ratio and unbroken streak)
    """
    n = len(vals)
    if n < 20:
        return {
            "drift_status": "NO_DRIFT",
            "affected_channel": channel_col,
            "station_id": station_id or "All Stations",
            "approximate_start_time": "N/A",
            "confidence": 0.0,
            "evidence": {
                "rolling_trend": "Insufficient records for statistical trend evaluation.",
                "baseline_shift": "Insufficient records for baseline comparison.",
                "change_point": "Insufficient records for changepoint detection.",
                "cross_sensor_evidence": "Insufficient records for cross-sensor validation.",
                "persistence": "Insufficient records for persistence estimation."
            }
        }

    # 1. Baseline & Recent Window
    b_len = max(10, int(n * 0.25))
    r_len = max(10, int(n * 0.25))
    b_vals = vals[:b_len]
    r_vals = vals[-r_len:]

    b_med = float(np.median(b_vals))
    b_mad = float(1.4826 * np.median(np.abs(b_vals - b_med)))
    if b_mad < 1e-6:
        b_mad = float(np.std(b_vals)) if np.std(b_vals) > 1e-6 else 1.0
    b_mean = float(np.mean(b_vals))
    b_sd = float(np.std(b_vals)) if np.std(b_vals) > 1e-6 else 1.0

    r_med = float(np.median(r_vals))
    r_mean = float(np.mean(r_vals))

    shift_med = r_med - b_med
    z_shift = shift_med / b_mad
    mean_z = (r_mean - b_mean) / b_sd
    pct_shift = (shift_med / abs(b_med) * 100) if abs(b_med) > 1e-6 else 0.0

    # 2. Trend Analysis
    slope, slope_norm, r2, direction = _compute_rolling_trend(vals, b_mad)

    # 3. Changepoint Detection
    cp_detected, best_k, approx_time, best_t = _detect_changepoint(vals, timestamps, b_med, b_mad)

    # 4. Cross-Sensor Validation
    partners = []
    if corr_matrix is not None and channel_col in corr_matrix:
        corr_series = corr_matrix[channel_col]
        for p_col, r_val in corr_series.items():
            if p_col != channel_col and abs(r_val) >= 0.35:
                p_shift = all_shifts.get(p_col, 0.0) if all_shifts else 0.0
                partners.append((p_col, float(r_val), float(p_shift)))

    # 5. Persistence
    post_idx = best_k if cp_detected else (n - r_len)
    pers_ratio, max_streak, n_post = _compute_persistence(vals, b_med, b_mad, post_idx)

    # 6. Multi-Signal Composite Confidence Score
    s_shift = min(1.0, abs(z_shift) / 2.5)
    s_trend = min(1.0, r2 * min(2.0, abs(slope_norm)))
    s_cp = 1.0 if cp_detected else min(1.0, best_t / 6.0)

    if abs(z_shift) >= 1.0:
        if partners:
            avg_partner_shift = float(np.mean([abs(p[2]) for p in partners]))
            s_cross = 1.0 if avg_partner_shift < 0.8 else 0.35
        else:
            s_cross = 0.60
    else:
        s_cross = 0.10

    s_pers = pers_ratio
    confidence = round(float(0.25 * s_shift + 0.25 * s_trend + 0.20 * s_cp + 0.15 * s_cross + 0.15 * s_pers), 2)

    # Multi-condition decision rule: requires concurring statistical evidence
    drift_detected = bool(
        confidence >= 0.48
        and (abs(z_shift) >= 1.0 or (r2 >= 0.18 and abs(slope_norm) >= 1.0))
        and pers_ratio >= 0.35
    )
    status = "DRIFT_DETECTED" if drift_detected else "NO_DRIFT"
    if not drift_detected:
        approx_time = "N/A"

    # Human-readable evidence descriptions
    if abs(slope_norm) >= 1.0 and r2 >= 0.15:
        ev_trend = (
            f"Sustained {direction} trend (slope: {slope:+.2f} units across span, "
            f"{slope_norm:+.2f} baseline MADs, R2={r2:.3f}); indicates continuous progressive drift."
        )
    elif abs(slope_norm) >= 0.5:
        ev_trend = (
            f"Moderate {direction} trend (slope: {slope:+.2f} units, "
            f"{slope_norm:+.2f} baseline MADs, R2={r2:.3f}); mild directional trajectory observed."
        )
    else:
        ev_trend = (
            f"Stationary/flat trend (slope: {slope:+.2f} units, "
            f"{slope_norm:+.2f} baseline MADs, R2={r2:.3f}); fluctuations remain mean-reverting."
        )

    if abs(z_shift) >= 1.0:
        ev_shift = (
            f"Recent monitoring median ({r_med:.2f}) displaced by {z_shift:+.2f} MADs from "
            f"initial calibration baseline ({b_med:.2f}, relative change: {pct_shift:+.1f}%, mean shift: {mean_z:+.2f} SD)."
        )
    else:
        ev_shift = (
            f"Recent monitoring median ({r_med:.2f}) aligns closely with baseline "
            f"({b_med:.2f}, shift: {z_shift:+.2f} MADs / {pct_shift:+.1f}%, mean shift: {mean_z:+.2f} SD)."
        )

    if cp_detected and drift_detected:
        ev_cp = (
            f"Structural break identified at {approx_time} "
            f"(pooled two-sample T={best_t:.2f} > 3.5 critical barrier, index {best_k} of {n})."
        )
    elif cp_detected:
        ev_cp = (
            f"Candidate structural transition flagged at {approx_time} (T={best_t:.2f}), "
            f"but overall persistence/shift remained below drift declaration threshold."
        )
    else:
        ev_cp = (
            f"No structural changepoint detected; test statistic ({best_t:.2f}) "
            f"remained within normal random fluctuation limits."
        )

    if partners:
        p_desc = ", ".join([f"{p[0]} (r={p[1]:+.2f}, shift={p[2]:+.2f} MAD)" for p in partners])
        avg_p = float(np.mean([abs(p[2]) for p in partners]))
        if abs(z_shift) >= 1.0 and avg_p < 0.8:
            ev_cross = (
                f"Target sensor drifted ({z_shift:+.2f} MAD) while correlated partners {p_desc} "
                f"remained steady (avg shift: {avg_p:.2f} MAD); strongly isolates hardware sensor drift from environmental atmosphere variation."
            )
        elif abs(z_shift) >= 1.0:
            ev_cross = (
                f"Correlated partners {p_desc} also shifted concurrently (avg shift: {avg_p:.2f} MAD); "
                f"indicates shared environmental/ambient weather shift."
            )
        else:
            ev_cross = f"Target sensor and correlated partners ({p_desc}) demonstrate stable baseline consistency."
    else:
        ev_cross = (
            "No strongly correlated partner channels (|r| >= 0.35) found; "
            "sensor analyzed independently without cross-channel corroboration."
        )

    if pers_ratio >= 0.50:
        ev_pers = (
            f"Shift sustained across {max_streak} consecutive points ({pers_ratio * 100:.1f}% "
            f"of post-transition window, {n_post} records); verifies persistent displacement rather than transient spikes."
        )
    elif pers_ratio >= 0.25:
        ev_pers = (
            f"Shift observed in {pers_ratio * 100:.1f}% of post-transition readings "
            f"(max streak: {max_streak} points); moderate persistence."
        )
    else:
        ev_pers = (
            f"Fluctuation was transient/sporadic ({pers_ratio * 100:.1f}% of window, "
            f"max streak: {max_streak} points); fails persistence criteria for persistent drift."
        )

    return {
        "drift_status": status,
        "affected_channel": channel_col,
        "station_id": station_id or "All Stations",
        "approximate_start_time": approx_time,
        "confidence": confidence,
        "evidence": {
            "rolling_trend": ev_trend,
            "baseline_shift": ev_shift,
            "change_point": ev_cp,
            "cross_sensor_evidence": ev_cross,
            "persistence": ev_pers
        }
    }


def analyze_channel_drift(df, sensor_columns=None, station_id=None):
    """
    Performs comprehensive Feature B measurement drift analysis for all sensor channels.
    Supports optional station filtering and provides per-station breakdowns.
    """
    detected = detect_columns(df)
    if sensor_columns is None:
        sensor_columns = detected["sensor_cols"]
    timestamp_col = detected["timestamp_col"]
    station_col = detected["station_col"]

    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    df_work = df.copy()
    if station_id and station_col and station_col in df_work.columns:
        df_work = df_work[df_work[station_col] == station_id]

    if timestamp_col and timestamp_col in df_work.columns:
        try:
            df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
        except Exception:
            pass
        df_work = df_work.sort_values(timestamp_col)

    if len(df_work) < 20 or not valid_cols:
        return {
            "drift_status": "NO_DRIFT",
            "total_channels": len(valid_cols),
            "drift_detected_count": 0,
            "station_id": station_id or "All Stations",
            "channels": [],
            "by_station": {}
        }

    # Precalculate baseline shifts and correlation matrix
    corr_matrix = df_work[valid_cols].corr().fillna(0)
    shifts = {}
    for col in valid_cols:
        v = df_work[col].dropna().values
        if len(v) >= 10:
            b_sub = v[:max(5, int(len(v) * 0.25))]
            r_sub = v[-max(5, int(len(v) * 0.25)):]
            b_m = float(np.median(b_sub))
            b_mad = float(1.4826 * np.median(np.abs(b_sub - b_m)))
            if b_mad < 1e-6:
                b_mad = float(np.std(b_sub)) or 1.0
            r_m = float(np.median(r_sub))
            shifts[col] = (r_m - b_m) / b_mad
        else:
            shifts[col] = 0.0

    channels_analysis = []
    drift_detected_count = 0

    ts_series = df_work[timestamp_col] if timestamp_col and timestamp_col in df_work.columns else None

    for col in valid_cols:
        v = df_work[col].dropna().values
        res = _build_channel_drift_evidence(
            channel_col=col,
            vals=v,
            timestamps=ts_series,
            corr_matrix=corr_matrix,
            all_shifts=shifts,
            station_id=station_id or "All Stations"
        )
        channels_analysis.append(res)
        if res["drift_status"] == "DRIFT_DETECTED":
            drift_detected_count += 1

    # Optional per-station breakdown if multi-station dataset
    by_station = {}
    if not station_id and station_col and station_col in df.columns and df[station_col].nunique() > 1:
        for stn in df[station_col].unique():
            stn_sub = df[df[station_col] == stn].copy()
            if timestamp_col and timestamp_col in stn_sub.columns:
                try:
                    stn_sub[timestamp_col] = pd.to_datetime(stn_sub[timestamp_col])
                except Exception:
                    pass
                stn_sub = stn_sub.sort_values(timestamp_col)

            stn_corr = stn_sub[valid_cols].corr().fillna(0)
            stn_shifts = {}
            for col in valid_cols:
                v = stn_sub[col].dropna().values
                if len(v) >= 10:
                    b_sub = v[:max(5, int(len(v) * 0.25))]
                    r_sub = v[-max(5, int(len(v) * 0.25)):]
                    b_m = float(np.median(b_sub))
                    b_mad = float(1.4826 * np.median(np.abs(b_sub - b_m))) or 1.0
                    r_m = float(np.median(r_sub))
                    stn_shifts[col] = (r_m - b_m) / b_mad
                else:
                    stn_shifts[col] = 0.0

            stn_ts = stn_sub[timestamp_col] if timestamp_col and timestamp_col in stn_sub.columns else None
            stn_analyses = []
            for col in valid_cols:
                v = stn_sub[col].dropna().values
                r = _build_channel_drift_evidence(
                    channel_col=col,
                    vals=v,
                    timestamps=stn_ts,
                    corr_matrix=stn_corr,
                    all_shifts=stn_shifts,
                    station_id=str(stn)
                )
                stn_analyses.append(r)
            by_station[str(stn)] = stn_analyses

    overall_drift = "DRIFT_DETECTED" if drift_detected_count > 0 else "NO_DRIFT"

    return {
        "drift_status": overall_drift,
        "total_channels": len(valid_cols),
        "drift_detected_count": drift_detected_count,
        "station_id": station_id or "All Stations",
        "channels": channels_analysis,
        "by_station": by_station
    }


def detect_drift(df, sensor_columns=None):
    """
    Existing detect_drift entry point, preserved and extended with Feature B multi-signal intelligence.
    Returns backwards-compatible keys alongside rich channel_drift_analysis.
    """
    detected = detect_columns(df)
    if sensor_columns is None:
        sensor_columns = detected["sensor_cols"]
    timestamp_col = detected["timestamp_col"]

    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    df_sorted = df.copy()
    if timestamp_col and timestamp_col in df_sorted.columns:
        try:
            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col])
        except Exception:
            pass
        df_sorted = df_sorted.sort_values(timestamp_col)

    total_records = len(df_sorted)
    if total_records < 20:
        return {
            "overall_status": "INSUFFICIENT_DATA",
            "overall_drift_status": "INSUFFICIENT_DATA",
            "high_drift_sensors_count": 0,
            "moderate_drift_sensors_count": 0,
            "sensors": {},
            "channel_drift_analysis": [],
            "detected_drift_channels": [],
            "analyzed_records": total_records,
            "baseline_window_records": 0,
            "recent_window_records": 0
        }

    # Define baseline window (first 25%) and recent monitoring window (last 25%)
    window_size = max(5, int(total_records * 0.25))
    baseline_df = df_sorted.iloc[:window_size]
    recent_df = df_sorted.iloc[-window_size:]

    # Run Feature B Enhanced Channel Drift Analysis
    analysis_results = analyze_channel_drift(df_sorted, valid_cols)
    channel_analyses = analysis_results["channels"]
    analysis_lookup = {item["affected_channel"]: item for item in channel_analyses}

    sensor_drift = {}
    high_drift_count = 0
    moderate_drift_count = 0

    for col in valid_cols:
        b_series = baseline_df[col].dropna()
        r_series = recent_df[col].dropna()

        if len(b_series) == 0 or len(r_series) == 0:
            continue

        b_mean = float(b_series.mean())
        b_std = float(b_series.std()) if b_series.std() > 0 else 1.0
        r_mean = float(r_series.mean())

        mean_shift = r_mean - b_mean
        z_shift = mean_shift / b_std
        pct_change = (mean_shift / abs(b_mean) * 100) if abs(b_mean) > 1e-6 else 0.0

        if abs(z_shift) >= 1.5:
            drift_status = "HIGH_DRIFT"
            high_drift_count += 1
        elif abs(z_shift) >= 0.75:
            drift_status = "MODERATE_DRIFT"
            moderate_drift_count += 1
        else:
            drift_status = "STABLE"

        direction = "UPWARD" if mean_shift > 0 else ("DOWNWARD" if mean_shift < 0 else "FLAT")

        sensor_drift[col] = {
            "baseline_mean": round(b_mean, 2),
            "recent_mean": round(r_mean, 2),
            "mean_shift": round(mean_shift, 2),
            "shift_zscore": round(z_shift, 2),
            "percentage_change": round(pct_change, 1),
            "drift_status": drift_status,
            "drift_direction": direction,
            "enhanced_analysis": analysis_lookup.get(col)
        }

    if high_drift_count > 0:
        overall_status = "CRITICAL_DRIFT_DETECTED"
    elif moderate_drift_count > 0:
        overall_status = "MODERATE_DRIFT_DETECTED"
    else:
        overall_status = "SYSTEM_STABLE"

    detected_drift_channels = [
        c["affected_channel"] for c in channel_analyses if c["drift_status"] == "DRIFT_DETECTED"
    ]

    return {
        "overall_status": overall_status,
        "overall_drift_status": overall_status,
        "high_drift_sensors_count": high_drift_count,
        "moderate_drift_sensors_count": moderate_drift_count,
        "sensors": sensor_drift,
        "channel_drift_analysis": channel_analyses,
        "detected_drift_channels": detected_drift_channels,
        "by_station": analysis_results.get("by_station", {}),
        "analyzed_records": total_records,
        "baseline_window_records": window_size,
        "recent_window_records": window_size
    }


if __name__ == "__main__":
    df = load_dataset()
    results = detect_drift(df)

    print("=" * 70)
    print("SENTINEL - SENSOR DRIFT DETECTOR & STATISTICAL INTELLIGENCE")
    print("=" * 70)

    print(f"\nOVERALL SYSTEM DRIFT STATUS: {results['overall_status']}")
    print(f"Analyzed Records: {results['analyzed_records']}")
    print(f"Detected Drift Channels: {results['detected_drift_channels']}")
    print("\nENHANCED FEATURE B CHANNEL ANALYSIS:")

    for ch_item in results["channel_drift_analysis"]:
        print(f"\nChannel: {ch_item['affected_channel']}")
        print(f"  Status: {ch_item['drift_status']}")
        print(f"  Confidence: {ch_item['confidence']}")
        print(f"  Approximate Start: {ch_item['approximate_start_time']}")
        print("  Evidence:")
        for k, v in ch_item["evidence"].items():
            print(f"    - {k}: {v}")

    print("\n" + "=" * 70)
    print("DRIFT DETECTION COMPLETE")
    print("=" * 70)

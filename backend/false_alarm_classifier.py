import os
import pandas as pd
import numpy as np

try:
    from backend.data_inspection import load_dataset, detect_columns
except ImportError:
    from data_inspection import load_dataset, detect_columns


def compute_sensor_correlations(df, sensor_columns, min_abs_corr=0.40):
    """
    Dynamically identifies correlated partner channels for each sensor
    based on Pearson correlation matrix without hardcoding channel names.
    """
    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return {}

    corr = df[valid_cols].corr()
    partners_map = {}

    for col in valid_cols:
        partners = []
        for other in valid_cols:
            if col != other:
                r_val = corr.loc[col, other]
                if pd.notna(r_val) and abs(r_val) >= min_abs_corr:
                    partners.append({
                        "channel": other,
                        "correlation": round(float(r_val), 3)
                    })
        # Sort by strongest correlation magnitude
        partners.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        partners_map[col] = partners

    return partners_map


def classify_false_alarms(df=None, sensor_columns=None, max_flagged=150):
    """
    Multi-signal False Alarm Classifier for extreme observations.
    Classifies each flagged observation as:
      - 'VALID_EXTREME_EVENT'
      - 'UNRELIABLE_READING'

    Combines 4 statistical evidence signals:
      1. Cross-sensor correlation agreement
      2. Temporal persistence (consecutive intervals vs isolated impulse)
      3. Rate of change (vs local rolling difference MAD)
      4. Local/rolling robust statistics (rolling median & rolling MAD)
    """
    if df is None:
        df = load_dataset()

    detected = detect_columns(df)
    station_col = detected.get("station_col") or "Station_ID"
    timestamp_col = detected.get("timestamp_col") or "Timestamp"
    if sensor_columns is None:
        sensor_columns = detected.get("sensor_cols", [])

    valid_cols = [c for c in sensor_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return {"summary": {"total_flagged": 0, "valid_extreme_events": 0, "unreliable_readings": 0}, "classifications": []}

    df_work = df.copy()
    if timestamp_col in df_work.columns:
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col], errors="coerce")

    sort_cols = [c for c in [station_col, timestamp_col] if c in df_work.columns]
    if sort_cols:
        df_work = df_work.sort_values(sort_cols).reset_index(drop=True)
    else:
        df_work = df_work.reset_index(drop=True)

    # 1. Global standardizations for baseline reference
    z_global = pd.DataFrame(index=df_work.index)
    col_medians = {}
    col_mads = {}

    for c in valid_cols:
        series = df_work[c].dropna()
        c_mean = float(series.mean()) if not series.empty else 0.0
        c_std = float(series.std()) if not series.empty and series.std() > 0 else 1.0
        c_med = float(series.median()) if not series.empty else 0.0
        c_mad = float((series - c_med).abs().median() * 1.4826)
        if c_mad <= 1e-6:
            c_mad = c_std

        z_global[c] = (df_work[c] - c_mean) / c_std
        col_medians[c] = c_med
        col_mads[c] = c_mad

    # 2. Dynamic cross-sensor correlation partner map
    partners_map = compute_sensor_correlations(df_work, valid_cols, min_abs_corr=0.40)

    # 3. Rolling robust local context per station
    # Window of 24 observations for local diurnal/short-term context
    local_medians = pd.DataFrame(index=df_work.index)
    local_mads = pd.DataFrame(index=df_work.index)
    diff_mads = pd.DataFrame(index=df_work.index)

    for c in valid_cols:
        if station_col in df_work.columns:
            # Grouped rolling stats
            grouped = df_work.groupby(station_col)[c]
            roll_med = grouped.rolling(24, min_periods=3).median().reset_index(level=0, drop=True)
            roll_diff = grouped.diff().abs()
            roll_diff_mad = roll_diff.groupby(df_work[station_col]).rolling(24, min_periods=3).median().reset_index(level=0, drop=True) * 1.4826
        else:
            roll_med = df_work[c].rolling(24, min_periods=3).median()
            roll_diff = df_work[c].diff().abs()
            roll_diff_mad = roll_diff.rolling(24, min_periods=3).median() * 1.4826

        # Fill edge cases with global robust stats
        local_medians[c] = roll_med.fillna(col_medians[c])
        # Local deviation from rolling median
        dev_from_roll = (df_work[c] - local_medians[c]).abs()
        if station_col in df_work.columns:
            roll_mad = dev_from_roll.groupby(df_work[station_col]).rolling(24, min_periods=3).median().reset_index(level=0, drop=True) * 1.4826
        else:
            roll_mad = dev_from_roll.rolling(24, min_periods=3).median() * 1.4826

        local_mads[c] = roll_mad.fillna(col_mads[c]).replace(0, col_mads[c])
        diff_mads[c] = roll_diff_mad.fillna(col_mads[c] * 0.5).replace(0, 1e-4)

    # 4. Identify candidate extreme observations (|Z_global| >= 2.5 or |Z_local| >= 2.5)
    flagged_candidates = []
    for idx in range(len(df_work)):
        for c in valid_cols:
            val = df_work[c].iloc[idx]
            if pd.isna(val):
                continue

            zg = z_global[c].iloc[idx]
            l_med = local_medians[c].iloc[idx]
            l_mad = local_mads[c].iloc[idx]
            zl = (val - l_med) / max(l_mad, 1e-4)

            # Flag if substantially extreme in either global or robust local frame
            if abs(zg) >= 2.5 or abs(zl) >= 2.8:
                severity_score = max(abs(zg), abs(zl))
                flagged_candidates.append({
                    "row_index": idx,
                    "channel": c,
                    "value": float(val),
                    "zg": float(zg),
                    "zl": float(zl),
                    "severity_score": severity_score
                })

    # Sort candidates by severity and prioritize diverse rows
    flagged_candidates.sort(key=lambda x: x["severity_score"], reverse=True)
    selected_candidates = flagged_candidates[:max_flagged]

    classifications = []
    valid_events_count = 0
    unreliable_count = 0

    for cand in selected_candidates:
        idx = cand["row_index"]
        c = cand["channel"]
        val = cand["value"]
        zg = cand["zg"]
        zl = cand["zl"]

        row = df_work.iloc[idx]
        ts_str = str(row[timestamp_col]) if timestamp_col in row and pd.notna(row[timestamp_col]) else f"Idx {idx}"
        stn_str = str(row[station_col]) if station_col in row and pd.notna(row[station_col]) else "Unknown"

        # -------------------------------------------------------------------
        # SIGNAL 1: Cross-Sensor Correlation Agreement
        # -------------------------------------------------------------------
        partners = partners_map.get(c, [])
        if partners:
            partner_supports = []
            partner_details = []
            direction = 1 if zg >= 0 else -1

            for p in partners:
                p_col = p["channel"]
                r = p["correlation"]
                p_zg = z_global[p_col].iloc[idx] if p_col in z_global.columns else 0.0

                if pd.isna(p_zg):
                    continue

                # Expected direction based on correlation sign
                expected_dir = direction if r > 0 else -direction
                # Check agreement: does partner deviate in expected direction?
                normalized_agreement = (p_zg * expected_dir)

                if normalized_agreement >= 1.5:
                    # Strong agreement
                    partner_supports.append(1.0)
                    partner_details.append(f"{p_col} (+{abs(p_zg):.1f} SD, r={r:+.2f}) confirmed surge")
                elif normalized_agreement >= 0.6:
                    # Moderate agreement
                    partner_supports.append(0.70)
                    partner_details.append(f"{p_col} (+{abs(p_zg):.1f} SD, r={r:+.2f}) showed matching movement")
                elif abs(p_zg) < 0.5:
                    # Partner remained completely baseline while this sensor went extreme
                    partner_supports.append(0.15)
                    partner_details.append(f"{p_col} remained at baseline (+{p_zg:.1f} SD, r={r:+.2f})")
                else:
                    # Partner moved opposite to correlation
                    partner_supports.append(0.0)
                    partner_details.append(f"{p_col} diverged opposite ({p_zg:+.1f} SD, r={r:+.2f})")

            if partner_supports:
                cross_score = float(np.mean(partner_supports))
                if cross_score >= 0.65:
                    cross_msg = f"Multi-channel atmospheric agreement: {'; '.join(partner_details[:2])}."
                elif cross_score >= 0.40:
                    cross_msg = f"Partial channel agreement: {'; '.join(partner_details[:2])}."
                else:
                    cross_msg = f"Isolated channel deviation: {'; '.join(partner_details[:2])}."
            else:
                cross_score = 0.50
                cross_msg = "Correlated partner channel data unavailable at this timestamp."
        else:
            cross_score = 0.50
            cross_msg = f"No strongly correlated partner sensors (|r|>=0.40) detected in dataset for {c}; reliance shifted to temporal and local signals."

        # -------------------------------------------------------------------
        # SIGNAL 2: Temporal Persistence
        # -------------------------------------------------------------------
        stn_mask = (df_work[station_col] == stn_str) if station_col in df_work.columns else pd.Series(True, index=df_work.index)
        stn_indices = df_work[stn_mask].index.tolist()

        try:
            curr_pos = stn_indices.index(idx)
        except ValueError:
            curr_pos = 0

        # Check consecutive window around this observation
        # Window of [-2, -1, 0, +1, +2]
        consecutive_count = 1
        # Check backward
        p_back = curr_pos - 1
        while p_back >= 0 and (curr_pos - p_back) <= 3:
            b_idx = stn_indices[p_back]
            b_zg = z_global[c].iloc[b_idx]
            if pd.notna(b_zg) and (abs(b_zg) >= 1.5) and ((b_zg >= 0) == (zg >= 0)):
                consecutive_count += 1
                p_back -= 1
            else:
                break

        # Check forward
        p_fwd = curr_pos + 1
        while p_fwd < len(stn_indices) and (p_fwd - curr_pos) <= 3:
            f_idx = stn_indices[p_fwd]
            f_zg = z_global[c].iloc[f_idx]
            if pd.notna(f_zg) and (abs(f_zg) >= 1.5) and ((f_zg >= 0) == (zg >= 0)):
                consecutive_count += 1
                p_fwd += 1
            else:
                break

        if consecutive_count >= 3:
            persist_score = 1.0
            persist_msg = f"Sustained event: extreme elevation persisted across {consecutive_count} consecutive observation periods."
        elif consecutive_count == 2:
            persist_score = 0.65
            persist_msg = f"Moderate persistence: elevated reading held across 2 consecutive observation periods."
        else:
            persist_score = 0.10
            persist_msg = "Transient impulse: isolated single-timestamp spike with immediate return to baseline."

        # -------------------------------------------------------------------
        # SIGNAL 3: Rate of Change (Local Physical Plausibility)
        # -------------------------------------------------------------------
        if curr_pos > 0:
            prev_idx = stn_indices[curr_pos - 1]
            prev_val = df_work[c].iloc[prev_idx]
            step_change = abs(val - prev_val) if pd.notna(prev_val) else 0.0
        else:
            step_change = 0.0

        local_d_mad = diff_mads[c].iloc[idx]
        jump_ratio = (step_change / local_d_mad) if local_d_mad > 0 else 1.0

        if jump_ratio <= 2.8:
            rate_score = 0.90
            rate_msg = f"Plausible dynamic transition: step change ({step_change:+.2f}) is consistent with local rate-of-change variance ({jump_ratio:.1f}x local MAD)."
        elif jump_ratio <= 5.0:
            rate_score = 0.60
            rate_msg = f"Elevated transition speed: step change ({step_change:+.2f}) represents a rapid change ({jump_ratio:.1f}x local MAD)."
        else:
            rate_score = max(0.05, 0.50 - ((jump_ratio - 5.0) * 0.05))
            rate_msg = f"Physically implausible jump: instantaneous jump of {step_change:+.2f} ({jump_ratio:.1f}x local dynamic variance) with zero gradual ramp."

        # -------------------------------------------------------------------
        # SIGNAL 4: Local Robust Statistics (Rolling Median & MAD)
        # -------------------------------------------------------------------
        l_med = local_medians[c].iloc[idx]
        l_mad = local_mads[c].iloc[idx]
        g_med = col_medians[c]

        # Check if local baseline has been organically elevated vs detached spike
        baseline_elevation = (l_med - g_med) / max(col_mads[c], 1e-4)

        if abs(zl) >= 4.0 and abs(baseline_elevation) < 0.5:
            # Extreme local deviation from a baseline that never moved
            local_score = 0.20
            local_msg = f"Severe local excursion: reading ({val:.2f}) is {zl:+.1f} SD detached from flat rolling 24h median ({l_med:.2f}, robust MAD={l_mad:.2f})."
        elif abs(baseline_elevation) >= 1.0:
            # Rolling baseline itself was elevated, supporting wider environmental accumulation
            local_score = 0.85
            local_msg = f"Environmental baseline accumulation: reading ({val:.2f}) aligns with an elevated rolling median ({l_med:.2f}, +{baseline_elevation:.1f} SD above global median)."
        else:
            local_score = 0.50
            local_msg = f"Local departure: reading ({val:.2f}) is {zl:+.1f} SD relative to rolling 24h median ({l_med:.2f}, robust MAD={l_mad:.2f})."

        # -------------------------------------------------------------------
        # 5. Multi-Signal Composite Fusion & Decision
        # -------------------------------------------------------------------
        # Weights: Cross-Sensor 35%, Persistence 30%, Rate of Change 20%, Local Context 15%
        event_support = (0.35 * cross_score) + (0.30 * persist_score) + (0.20 * rate_score) + (0.15 * local_score)

        if event_support >= 0.50:
            classification = "VALID_EXTREME_EVENT"
            confidence = round(float(event_support), 2)
            valid_events_count += 1
        else:
            classification = "UNRELIABLE_READING"
            confidence = round(float(1.0 - event_support), 2)
            unreliable_count += 1

        classifications.append({
            "timestamp": ts_str,
            "station_id": stn_str,
            "channel": c,
            "value": round(float(val), 2),
            "classification": classification,
            "confidence": confidence,
            "event_support_score": round(float(event_support), 3),
            "evidence": {
                "cross_sensor_agreement": cross_msg,
                "persistence": persist_msg,
                "rate_of_change": rate_msg,
                "local_statistics": local_msg
            }
        })

    # Sort final list chronologically or by confidence
    classifications.sort(key=lambda x: x["timestamp"])

    return {
        "summary": {
            "total_flagged": len(classifications),
            "valid_extreme_events": valid_events_count,
            "unreliable_readings": unreliable_count,
            "validation_method": "Multi-Signal Evidence Fusion (Cross-Sensor, Persistence, Dynamic Rate, Robust Rolling MAD)"
        },
        "classifications": classifications
    }


if __name__ == "__main__":
    df = load_dataset()
    result = classify_false_alarms(df, max_flagged=20)

    print("=" * 70)
    print("SENTINEL - FALSE ALARM CLASSIFIER")
    print("=" * 70)
    print(f"Total Flagged Observations: {result['summary']['total_flagged']}")
    print(f"Valid Extreme Events:       {result['summary']['valid_extreme_events']}")
    print(f"Unreliable Readings:        {result['summary']['unreliable_readings']}")
    print("\nSAMPLE CLASSIFICATIONS WITH VISIBLE EVIDENCE:\n")

    for item in result["classifications"][:5]:
        print(f"[{item['classification']}] (Conf: {item['confidence']}) {item['station_id']} | {item['timestamp']} | {item['channel']} = {item['value']}")
        print(f"  • Cross-Sensor: {item['evidence']['cross_sensor_agreement']}")
        print(f"  • Persistence:  {item['evidence']['persistence']}")
        print(f"  • Rate-of-Change: {item['evidence']['rate_of_change']}")
        print(f"  • Local Stats:  {item['evidence']['local_statistics']}\n")

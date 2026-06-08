import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any


def _smart_contamination(n_rows: int, cv: float) -> float:
    """
    Adaptive contamination for IsolationForest.
    Noisier data gets a lower contamination rate so we don't flag
    everything as anomalous on high-variance datasets like daily profit.

    Targets ~3-8% of points flagged maximum.
    """
    if cv > 2.0:
        return 0.03   # Very noisy — extreme outliers only
    elif cv > 1.0:
        return 0.05   # Moderately noisy
    else:
        return 0.08   # Clean data — standard sensitivity


def detect_anomalies(df: pd.DataFrame, forecast_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects anomalies in historical data using IsolationForest.
    Also flags points outside the forecast confidence band.

    Improvements over naive approach:
    - Contamination adapts to data noisiness (CV-based)
    - Minimum 15% deviation threshold — ignores statistical noise
    - Z-score floor of 2.5 required for HIGH severity
    - Direction labelled as spike/drop

    Args:
        df:            DataFrame with 'date' and '_target_' columns.
        forecast_dict: Output from run_forecast(); provides lower/upper bands.

    Returns:
        List of anomaly dicts with date, actual, expected, severity,
        deviation_percent, is_outside_band, direction.
    """
    try:
        series = df["_target_"]
        dates  = df["date"].tolist()
        sales  = series.values.reshape(-1, 1)

        # Adaptive contamination — key fix for noisy retail/transactional data
        cv            = float(series.std() / (abs(series.mean()) + 1e-8))
        contamination = _smart_contamination(len(series), cv)

        iso    = IsolationForest(contamination=contamination, random_state=42)
        labels = iso.fit_predict(sales)

        rolling_mean = series.rolling(window=4, min_periods=1).mean().tolist()

        # Build forecast-band lookup keyed by date string
        fc_lower  = forecast_dict.get("lower", [])
        fc_upper  = forecast_dict.get("upper", [])
        fc_dates  = forecast_dict.get("dates", [])
        band_lookup: Dict[str, Dict] = {}
        for i, d in enumerate(fc_dates):
            d_str = str(d)[:10]
            band_lookup[d_str] = {
                "lower": fc_lower[i] if i < len(fc_lower) else None,
                "upper": fc_upper[i] if i < len(fc_upper) else None,
            }

        std_dev   = float(series.std() + 1e-8)
        anomalies = []

        for i, label in enumerate(labels):
            if label != -1:
                continue   # IsolationForest: -1 = anomaly, 1 = normal

            actual   = float(sales[i][0])
            expected = float(rolling_mean[i])
            expected_denom   = abs(expected) if expected != 0 else 1.0
            deviation_percent = abs(actual - expected) / expected_denom * 100

            # Minimum deviation floor — ignores statistical noise on noisy datasets
            if deviation_percent < 15:
                continue

            z_score   = abs(actual - expected) / std_dev
            direction = "spike" if actual > expected else "drop"

            # Severity: require BOTH high z-score AND high deviation for HIGH
            if z_score > 2.5 and deviation_percent > 20:
                severity = "high"
            elif deviation_percent > 15:
                severity = "medium"
            else:
                severity = "low"

            # Check against forecast confidence band
            date_str = str(dates[i])[:10]
            band     = band_lookup.get(date_str, {})
            is_outside_band = False
            if band:
                lo = band.get("lower")
                hi = band.get("upper")
                if lo is not None and actual < lo:
                    is_outside_band = True
                elif hi is not None and actual > hi:
                    is_outside_band = True

            anomalies.append({
                "date":             date_str,
                "actual":           actual,
                "expected":         expected,
                "severity":         severity,
                "deviation_percent": round(deviation_percent, 2),
                "is_outside_band":  is_outside_band,
                "direction":        direction,
            })

        return anomalies

    except Exception as e:
        import logging
        logging.error(f"Error detecting anomalies: {e}")
        return []


def compute_truth_meter(model_mape: float, baseline_mape: float) -> Dict[str, Any]:
    """
    Compares model SMAPE to baseline SMAPE and returns a human-readable
    reliability assessment.

    Note: model_mape and baseline_mape are now SMAPE values (passed from
    forecaster.py after the SMAPE fix). The parameter names are kept for
    backwards compatibility.

    Returns:
        dict with reliable (bool), color (str), message (str), score (float).
    """
    try:
        if not baseline_mape or baseline_mape == 0:
            return {
                "reliable": False,
                "color":    "red",
                "message":  "Baseline unavailable — cannot compute truth score.",
                "score":    0.0,
            }

        score = (baseline_mape - model_mape) / baseline_mape * 100.0

        if score >= 10:
            return {
                "reliable": True,
                "color":    "green",
                "message":  (
                    f"Model is {score:.1f}% better than baseline "
                    f"(SMAPE: {model_mape:.1f}% vs baseline {baseline_mape:.1f}%)"
                ),
                "score": score,
            }
        elif score >= 0:
            return {
                "reliable": False,
                "color":    "orange",
                "message":  (
                    f"Model is {score:.1f}% better than baseline "
                    f"(SMAPE: {model_mape:.1f}% vs baseline {baseline_mape:.1f}%)"
                ),
                "score": score,
            }
        else:
            return {
                "reliable": False,
                "color":    "red",
                "message":  (
                    f"Model close to baseline — consider uploading more data "
                    f"for better accuracy (SMAPE: {model_mape:.1f}%)"
                ),
                "score": score,
            }

    except Exception as e:
        import logging
        logging.error(f"Error in truth meter: {e}")
        return {
            "reliable": False,
            "color":    "red",
            "message":  "Error calculating truth meter.",
            "score":    0.0,
        }
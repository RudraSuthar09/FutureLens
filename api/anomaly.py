import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any


def detect_anomalies(df: pd.DataFrame, forecast_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects anomalies in historical data using IsolationForest.
    Also flags points that fall outside the forecast confidence band.

    Args:
        df: DataFrame with 'date' and 'sales' columns.
        forecast_dict: Output from run_forecast(); used to extract lower/upper bands.

    Returns:
        List of anomaly dicts with date, actual, expected, severity,
        deviation_percent, is_outside_band, and direction.
    """
    try:
        sales = df["sales"].values.reshape(-1, 1)
        dates = df["date"].tolist()

        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(sales)  # -1 = anomaly, 1 = normal

        rolling_mean = df["sales"].rolling(window=4, min_periods=1).mean().tolist()

        # Build a lookup for forecast band by date string
        fc_lower = forecast_dict.get("lower", [])
        fc_upper = forecast_dict.get("upper", [])
        fc_dates = forecast_dict.get("dates", [])
        band_lookup: Dict[str, Dict] = {}
        for i, d in enumerate(fc_dates):
            d_str = str(d)[:10]
            band_lookup[d_str] = {
                "lower": fc_lower[i] if i < len(fc_lower) else None,
                "upper": fc_upper[i] if i < len(fc_upper) else None,
            }

        anomalies = []
        for i, label in enumerate(labels):
            if label == -1:
                actual = float(sales[i][0])
                expected = float(rolling_mean[i])
                expected_denom = abs(expected) if expected != 0 else 1.0
                deviation_percent = abs(actual - expected) / expected_denom * 100

                severity = "high" if deviation_percent > 30 else ("medium" if deviation_percent > 15 else "low")
                direction = "spike" if actual > expected else "drop"

                # Check against forecast confidence band
                date_str = str(dates[i])[:10]
                band = band_lookup.get(date_str, {})
                is_outside_band = False
                if band:
                    lo = band.get("lower")
                    hi = band.get("upper")
                    if lo is not None and actual < lo:
                        is_outside_band = True
                    elif hi is not None and actual > hi:
                        is_outside_band = True

                anomalies.append(
                    {
                        "date": date_str,
                        "actual": actual,
                        "expected": expected,
                        "severity": severity,
                        "deviation_percent": round(deviation_percent, 2),
                        "is_outside_band": is_outside_band,
                        "direction": direction,
                    }
                )

        return anomalies
    except Exception as e:
        import logging

        logging.error(f"Error detecting anomalies: {e}")
        return []


def compute_truth_meter(model_mape: float, baseline_mape: float) -> Dict[str, Any]:
    """Compares model MAPE to baseline MAPE and returns a confidence assessment."""
    try:
        if baseline_mape == 0:
            return {"reliable": False, "color": "red", "message": "Baseline MAPE is 0 — invalid test"}

        improvement = (baseline_mape - model_mape) / baseline_mape

        if improvement > 0.10:
            return {
                "reliable": True,
                "color": "green",
                "message": f"High confidence — MAPE {model_mape:.1f}%",
            }
        else:
            diff = max(0.0, improvement * 100)
            return {
                "reliable": False,
                "color": "red",
                "message": f"Model only {diff:.1f}% better — low trust",
            }
    except Exception as e:
        import logging

        logging.error(f"Error in truth meter: {e}")
        return {"reliable": False, "color": "red", "message": "Error calculating truth meter."}

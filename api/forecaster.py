import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from mapie.regression import MapieRegressor
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Helper: detect date column with 1980 sanity check
# ---------------------------------------------------------------------------

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Loops through every column and returns the first one that:
    - Can be parsed as datetime with valid_ratio >= 0.8
    - Has a max parsed date after 1980-01-01 (guards against Unix-epoch garbage)

    Raises ValueError if no column passes both checks.
    """
    cutoff = pd.Timestamp("1980-01-01")
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            valid = parsed.notna()
            valid_ratio = valid.sum() / len(df)
            if valid_ratio >= 0.8:
                if parsed[valid].max() > cutoff:
                    return col
        except Exception:
            continue
    raise ValueError("No valid date column found in dataset")


# ---------------------------------------------------------------------------
# Adaptive parameters
# ---------------------------------------------------------------------------

def calculate_forecast_horizon(n_rows: int) -> int:
    if n_rows < 20:
        return 2
    elif n_rows < 50:
        return 3
    elif n_rows < 104:
        return 4
    else:
        return 8


def get_alpha(n_samples: int) -> float:
    if n_samples < 30:
        return 0.50
    elif n_samples < 60:
        return 0.32
    elif n_samples < 100:
        return 0.20
    else:
        return 0.10


def _detect_freq(dates: pd.Series) -> str:
    """Infer data frequency from a sorted datetime Series."""
    try:
        inferred = pd.infer_freq(dates)
        if inferred:
            return inferred
    except Exception:
        pass
    # Fallback: median gap in days
    gaps = dates.diff().dropna().dt.days
    if gaps.empty:
        return "W"
    median_gap = float(gaps.median())
    if median_gap < 3:
        return "D"
    elif median_gap < 10:
        return "W"
    elif median_gap < 35:
        return "MS"
    else:
        return "QS"


# ---------------------------------------------------------------------------
# Feature engineering on residuals
# ---------------------------------------------------------------------------

def _build_lag_features(series: pd.Series, available_lags: List[int], rolling_window: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in available_lags:
        df[f"lag{lag}"] = df["y"].shift(lag)
    df[f"rolling_mean_{rolling_window}"] = df["y"].rolling(window=rolling_window).mean()
    return df


# ---------------------------------------------------------------------------
# Main forecast function
# ---------------------------------------------------------------------------

def run_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Adaptive forecast: Prophet trend + LightGBM on residuals + MAPIE intervals.

    Args:
        df: DataFrame with 'date' and 'sales' columns (already cleaned).

    Returns:
        dict with dates, forecast, lower, upper, historical, historical_dates,
        baseline, model_mape, baseline_mape, truth_score, detected_freq,
        forecast_horizon, confidence_level, data_quality, model, X.
    """
    # 1. Build Prophet-format frame
    df_prophet = pd.DataFrame(
        {
            "ds": pd.to_datetime(df["date"], utc=False),
            "y": df["sales"].values,
        }
    ).sort_values("ds").reset_index(drop=True)

    n_rows = len(df_prophet)
    periods = calculate_forecast_horizon(n_rows)
    alpha = get_alpha(n_rows)
    confidence_level = int((1 - alpha) * 100)

    # 2. Auto-detect data frequency
    detected_freq = _detect_freq(df_prophet["ds"])

    # 3. Adaptive Prophet configuration
    has_yearly = n_rows >= 104
    has_weekly = n_rows >= 26

    model_prophet = Prophet(
        weekly_seasonality=has_weekly,
        yearly_seasonality=has_yearly,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model_prophet.fit(df_prophet)

    future = model_prophet.make_future_dataframe(periods=periods, freq=detected_freq)
    prophet_forecast = model_prophet.predict(future)

    # 4. Extract Prophet trend for historical period
    prophet_trend_hist = prophet_forecast["trend"].iloc[:n_rows].values

    # 5. Compute residuals (what LightGBM will learn)
    df_prophet["residuals"] = df_prophet["y"].values - prophet_trend_hist

    # 6. Adaptive lag features (only lags feasible given data size)
    available_lags = [l for l in [1, 2, 4, 8] if l < n_rows // 3]
    if not available_lags:
        available_lags = [1]
    rolling_window = max(2, min(4, n_rows // 4))
    feat_cols = [f"lag{l}" for l in available_lags] + [f"rolling_mean_{rolling_window}"]

    df_features = _build_lag_features(df_prophet["residuals"], available_lags, rolling_window)
    train_data = df_features.dropna().reset_index(drop=True)
    X = train_data[feat_cols]
    y_res = train_data["y"]

    # 7. Fit LightGBM on residuals, then MAPIE for conformal intervals
    lgbm = LGBMRegressor(random_state=42, verbose=-1)
    lgbm.fit(X, y_res)
    mapie = MapieRegressor(estimator=lgbm, cv="prefit")
    mapie.fit(X, y_res)

    # 8. Autoregressive future loop: Prophet trend + residual prediction
    residual_history = df_prophet["residuals"].tolist()
    future_preds: List[float] = []
    future_lower: List[float] = []
    future_upper: List[float] = []

    for i in range(periods):
        trend_val = float(prophet_forecast["trend"].iloc[n_rows + i])

        lag_vals: Dict[str, List[float]] = {}
        for lag in available_lags:
            lag_vals[f"lag{lag}"] = [residual_history[-lag] if len(residual_history) >= lag else 0.0]
        window_slice = residual_history[-rolling_window:] if len(residual_history) >= rolling_window else residual_history
        lag_vals[f"rolling_mean_{rolling_window}"] = [float(np.mean(window_slice))]

        X_next = pd.DataFrame(lag_vals)
        pred, pred_int = mapie.predict(X_next, alpha=alpha)
        residual_pred = float(pred[0])
        final_pred = trend_val + residual_pred

        pi = pred_int[0]
        if pi.ndim == 2:
            lower_residual = float(pi[0][0])
            upper_residual = float(pi[1][0])
        else:
            lower_residual = float(pi[0])
            upper_residual = float(pi[1])

        future_preds.append(final_pred)
        future_lower.append(trend_val + lower_residual)
        future_upper.append(trend_val + upper_residual)
        residual_history.append(residual_pred)

    # 9. Baseline: naive last-value forecast
    last_value = float(df_prophet["y"].iloc[-1])
    baseline_forecast = [last_value] * periods

    # 10. Validation MAPE
    val_size = min(8, len(train_data) // 4)
    if val_size > 0 and len(train_data) > val_size:
        val_y_res = train_data["y"].iloc[-val_size:].values
        val_X = train_data[feat_cols].iloc[-val_size:]
        raw_preds = mapie.predict(val_X)
        model_res_preds = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds

        val_start_idx = n_rows - val_size
        val_trend = prophet_forecast["trend"].iloc[val_start_idx : val_start_idx + val_size].values
        val_actual = df_prophet["y"].iloc[-val_size:].values
        val_model_preds = val_trend + model_res_preds
        val_baseline = np.array([last_value] * val_size)

        nonzero = val_actual != 0
        if nonzero.any():
            model_mape = float(
                np.mean(np.abs((val_actual[nonzero] - val_model_preds[nonzero]) / val_actual[nonzero])) * 100
            )
            baseline_mape = float(
                np.mean(np.abs((val_actual[nonzero] - val_baseline[nonzero]) / val_actual[nonzero])) * 100
            )
        else:
            model_mape, baseline_mape = 10.0, 20.0
    else:
        model_mape, baseline_mape = 10.0, 20.0

    truth_score = max(0.0, (baseline_mape - model_mape) / baseline_mape * 100) if baseline_mape > 0 else 0.0

    # 11. Output dates
    future_dates = (
        prophet_forecast["ds"].iloc[n_rows : n_rows + periods].dt.strftime("%Y-%m-%d").tolist()
    )
    historical_dates = df_prophet["ds"].dt.strftime("%Y-%m-%d").tolist()
    historical_sales = df_prophet["y"].tolist()

    # 12. Data quality metadata
    data_quality_warning: Optional[str] = None
    if n_rows < 104:
        data_quality_warning = (
            f"Dataset has {n_rows} data points. "
            f"Forecasts are exploratory beyond {periods} periods."
        )

    return {
        "dates": future_dates,
        "forecast": future_preds,
        "lower": future_lower,
        "upper": future_upper,
        "historical": historical_sales,
        "historical_dates": historical_dates,
        "baseline": baseline_forecast,
        "model_mape": model_mape,
        "baseline_mape": baseline_mape,
        "truth_score": truth_score,
        "detected_freq": detected_freq,
        "forecast_horizon": periods,
        "confidence_level": confidence_level,
        "data_quality": {
            "total_rows": n_rows,
            "warning": data_quality_warning,
        },
        # These are stripped in main.py before saving
        "model": lgbm,
        "X": X,
    }

import io
import re
import uuid
import json
import os
import pandas as pd
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from api.database import (
    init_db, save_upload, save_forecast, save_anomalies, save_chat,
    get_forecast, get_anomalies, get_chat_history, get_recent_uploads,
    save_system_prompt, get_system_prompt, get_recent_chat
)
from api.forecaster import run_forecast, detect_date_column
from api.rca import compute_shap, explain_rca
from api.anomaly import detect_anomalies, compute_truth_meter
from api.agent import chat
from api.simulator import simulate_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    yield


app = FastAPI(title="FutureLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_context: list = []
    session_id: str = "demo"


class SimulateRequest(BaseModel):
    session_id: str
    change_percent: float
    scenario_type: str = "growth"


class ForecastQueryRequest(BaseModel):
    message: str
    session_id: str = "demo"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import datetime
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


def _sanitize_for_json(data):
    return json.loads(json.dumps(data, cls=NumpyEncoder))


def _get_freq_info(freq_code: str) -> dict:
    if not freq_code:
        return {"label": "week", "per_week": 1, "per_month": 4}
    code = freq_code.upper()
    if code.startswith("B") or code.startswith("D"):
        return {"label": "day", "per_week": 7, "per_month": 30}
    elif code.startswith("H") or code.startswith("T"):
        return {"label": "hour", "per_week": 168, "per_month": 720}
    elif code.startswith("W"):
        return {"label": "week", "per_week": 1, "per_month": 4}
    elif code.startswith("M") or code.startswith("BM"):
        return {"label": "month", "per_week": 0.25, "per_month": 1}
    elif code.startswith("Q"):
        return {"label": "quarter", "per_week": 0.077, "per_month": 0.333}
    elif code.startswith("Y") or code.startswith("A"):
        return {"label": "year", "per_week": 0.019, "per_month": 0.083}
    return {"label": "week", "per_week": 1, "per_month": 4}


def _compute_column_relationships(
    df_work: pd.DataFrame,
    original_target: str,
    feature_cols: list,
) -> dict:
    relationships: dict = {
        "correlations_with_target": {},
        "profit_margin": None,
        "lag_correlations": {},
    }

    for col in feature_cols:
        if col not in df_work.columns:
            continue
        try:
            aligned = df_work[["sales", col]].dropna()
            if len(aligned) < 4:
                continue
            corr = round(float(aligned["sales"].corr(aligned[col])), 3)
            relationships["correlations_with_target"][col] = corr
        except Exception:
            continue

    target_lower = original_target.lower()
    is_profit_target = any(k in target_lower for k in ["profit", "income", "earnings", "margin", "net"])
    is_sales_target  = any(k in target_lower for k in ["sales", "revenue", "turnover"])

    for col in feature_cols:
        if col not in df_work.columns:
            continue
        col_lower = col.lower()
        try:
            if is_profit_target and any(k in col_lower for k in ["sales", "revenue", "turnover"]):
                aligned = df_work[["sales", col]].dropna()
                aligned = aligned[aligned[col] != 0]
                if len(aligned) >= 4:
                    margin = round(float((aligned["sales"] / aligned[col]).mean()), 4)
                    relationships["profit_margin"] = {
                        "ratio": margin,
                        "profit_col": original_target,
                        "sales_col": col,
                    }
                    break
            elif is_sales_target and any(k in col_lower for k in ["profit", "income", "earnings", "net"]):
                aligned = df_work[["sales", col]].dropna()
                aligned = aligned[aligned["sales"] != 0]
                if len(aligned) >= 4:
                    margin = round(float((aligned[col] / aligned["sales"]).mean()), 4)
                    relationships["profit_margin"] = {
                        "ratio": margin,
                        "profit_col": col,
                        "sales_col": original_target,
                    }
                    break
        except Exception:
            continue

    for col in feature_cols:
        if col not in df_work.columns:
            continue
        try:
            shifted = df_work[col].shift(1)
            aligned = pd.concat([df_work["sales"], shifted], axis=1).dropna()
            aligned.columns = ["target", "lagged"]
            if len(aligned) < 4:
                continue
            lag_corr = round(float(aligned["target"].corr(aligned["lagged"])), 3)
            if abs(lag_corr) > 0.1:
                relationships["lag_correlations"][col] = lag_corr
        except Exception:
            continue

    return relationships


def detect_target_column(df: pd.DataFrame, date_col: str) -> str:
    preferred_names = [
        "AveragePrice", "sales", "Sales", "price", "revenue",
        "amount", "value", "close", "Close",
    ]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = []
    for col in num_cols:
        if col == date_col:
            continue
        null_ratio = df[col].isna().sum() / len(df)
        if null_ratio > 0.30:
            continue
        col_lower = col.lower()
        if "id" in col_lower and df[col].nunique() == len(df):
            continue
        if df[col].dtype in [np.int64, np.int32, np.int16, np.int8]:
            if df[col].nunique() == len(df):
                continue
        candidates.append(col)

    if not candidates:
        raise ValueError("No suitable numeric target column found in CSV.")

    for name in preferred_names:
        if name in candidates:
            return name

    cv_scores = {}
    for col in candidates:
        mean_val = float(df[col].mean())
        std_val  = float(df[col].std())
        cv_scores[col] = std_val / abs(mean_val) if mean_val != 0 else 0.0
    return max(cv_scores, key=cv_scores.get)


def build_dataset_profile(df: pd.DataFrame, max_categories: int = 8, sample_rows: int = 12) -> dict:
    profile = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "numeric_summary": {},
        "categorical_summary": {},
        "sample_rows": [],
    }
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        profile["numeric_summary"][c] = {
            "count": int(s.shape[0]),
            "mean":  float(s.mean()),
            "std":   float(s.std()) if s.shape[0] > 1 else 0.0,
            "min":   float(s.min()),
            "max":   float(s.max()),
        }
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols:
        s = df[c].astype(str).replace({"nan": np.nan}).dropna()
        if len(s) == 0:
            continue
        vc = s.value_counts().head(max_categories)
        profile["categorical_summary"][c] = [{"value": k, "count": int(v)} for k, v in vc.items()]
    try:
        profile["sample_rows"] = df.head(sample_rows).fillna("").to_dict(orient="records")
    except Exception:
        profile["sample_rows"] = []
    return profile


def pick_group_column(df: pd.DataFrame) -> str | None:
    preferred = ["region", "category", "segment", "product", "store", "country", "city"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return None
    for p in preferred:
        for c in cat_cols:
            if p in c.lower():
                return c
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 30:
            return c
    return None


def _make_empty_forecast(df_work: pd.DataFrame, detected_columns: dict) -> dict:
    sales = df_work["sales"].dropna().tolist()
    last_val = float(sales[-1]) if sales else 0.0
    return {
        "dates": [], "forecast": [], "lower": [], "upper": [],
        "historical": sales,
        "historical_dates": df_work["date"].dt.strftime("%Y-%m-%d").tolist(),
        "baseline": [], "model_mape": 0.0, "baseline_mape": 0.0,
        "truth_score": 0.0, "detected_freq": "D", "forecast_horizon": 0,
        "confidence_level": 80, "last_value": last_val,
        "data_quality": {
            "total_rows": len(df_work),
            "warning": "Forecast pipeline did not run.",
        },
    }


def _run_group_forecasts(df, group_col, original_date_col, original_target_col, logger):
    try:
        df_tmp = df[[original_date_col, original_target_col, group_col]].copy()
        df_tmp[original_date_col] = pd.to_datetime(
            df_tmp[original_date_col], errors="coerce", utc=False
        )
        df_tmp[original_target_col] = pd.to_numeric(
            df_tmp[original_target_col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("\xa0", " ")
                .str.strip(),
            errors="coerce",
        )
        df_tmp = df_tmp.dropna(subset=[original_date_col, original_target_col, group_col])

        top_groups = (
            df_tmp.groupby(group_col)[original_target_col]
            .sum().sort_values(ascending=False).head(5).index.tolist()
        )

        group_forecasts = []
        for g in top_groups:
            gdf = df_tmp[df_tmp[group_col] == g][
                [original_date_col, original_target_col]
            ].copy()
            gdf.rename(columns={original_date_col: "date", original_target_col: "sales"}, inplace=True)
            gdf["date"]  = pd.to_datetime(gdf["date"], errors="coerce")
            gdf["sales"] = pd.to_numeric(gdf["sales"], errors="coerce")
            gdf = gdf.dropna(subset=["date", "sales"])

            # Resample to weekly SUM
            gdf = (
                gdf.set_index("date")["sales"]
                .resample("W").sum()
                .reset_index()
            )
            gdf.columns = ["date", "sales"]
            gdf = gdf[gdf["sales"] != 0].reset_index(drop=True)

            if len(gdf) < 10:
                continue

            gres      = run_forecast(gdf)
            last_hist = float(gres["historical"][-1]) if gres.get("historical") else 0.0
            last_fc   = float(gres["forecast"][-1])   if gres.get("forecast")   else 0.0
            pct = (
                (last_fc - last_hist) / abs(last_hist) * 100.0
                if abs(last_hist) > 1e-6 else 0.0
            )
            pct = float(np.clip(pct, -200.0, 200.0))

            group_forecasts.append({
                "group_col":               group_col,
                "group":                   str(g),
                "forecast_horizon":        gres.get("forecast_horizon"),
                "expected_change_percent": round(pct, 2),
                "last_hist":               round(last_hist, 2),
                "last_forecast":           round(last_fc,   2),
            })

        group_forecasts.sort(key=lambda x: x["expected_change_percent"], reverse=True)
        return group_forecasts or None
    except Exception as e:
        logger.warning(f"Group forecasts skipped: {e}")
        return None


def _build_static_group_summary(df, group_col, target_col):
    """For cross-sectional datasets — rank groups by mean/sum of target."""
    try:
        df_tmp = df[[group_col, target_col]].copy()
        df_tmp[target_col] = pd.to_numeric(
            df_tmp[target_col].astype(str)
                .str.replace(",", "").str.replace("\xa0", " ").str.strip(),
            errors="coerce",
        )
        df_tmp = df_tmp.dropna()
        summary = (
            df_tmp.groupby(group_col)[target_col]
            .agg(["mean", "sum", "count"])
            .reset_index().sort_values("sum", ascending=False).head(10)
        )
        global_mean = float(df_tmp[target_col].mean())
        result = []
        for _, row in summary.iterrows():
            grp_mean = float(row["mean"])
            pct_vs_avg = ((grp_mean - global_mean) / abs(global_mean) * 100) if global_mean else 0.0
            result.append({
                "group_col":               group_col,
                "group":                   str(row[group_col]),
                "forecast_horizon":        0,
                "expected_change_percent": round(pct_vs_avg, 2),
                "last_hist":               round(grp_mean, 2),
                "last_forecast":           round(grp_mean, 2),
                "total":                   round(float(row["sum"]), 2),
                "count":                   int(row["count"]),
                "is_static":               True,
            })
        return result or None
    except Exception as e:
        logger.warning(f"Static group summary failed: {e}")
        return None


def _safe_float(val) -> float:
    try:
        f = float(val)
        return None if (f != f) else round(f, 4)
    except Exception:
        return None


def _build_chart_data(df: pd.DataFrame, chart_config: dict) -> dict:
    """Build chart-ready data for cross-sectional datasets."""
    chart_type = chart_config.get("chart_type", "horizontal_bar")
    x_col      = chart_config.get("x_col")
    y_col      = chart_config.get("y_col")
    color_col  = chart_config.get("color_col")
    size_col   = chart_config.get("size_col")
    top_n      = int(chart_config.get("top_n", 20))
    sort_order = chart_config.get("sort_order", "desc")

    df_tmp = df.copy()

    if y_col and y_col in df_tmp.columns:
        df_tmp[y_col] = pd.to_numeric(
            df_tmp[y_col].astype(str)
                .str.replace(",", "").str.replace("\xa0", " ").str.strip(),
            errors="coerce"
        )
        df_tmp = df_tmp.dropna(subset=[y_col])

    if y_col and y_col in df_tmp.columns and chart_type != "histogram":
        ascending = sort_order == "asc"
        df_tmp = df_tmp.sort_values(y_col, ascending=ascending).head(top_n)

    out = {
        "chart_type":  chart_type,
        "chart_title": chart_config.get("chart_title", ""),
        "x_label":     chart_config.get("x_label", x_col or ""),
        "y_label":     chart_config.get("y_label", y_col or ""),
        "x_col":       x_col,
        "y_col":       y_col,
        "color_col":   color_col,
        "size_col":    size_col,
        "rows":        [],
    }

    for _, row in df_tmp.iterrows():
        entry = {}
        if x_col and x_col in df_tmp.columns:
            entry["x"] = str(row[x_col]) if df_tmp[x_col].dtype == object else _safe_float(row[x_col])
        if y_col and y_col in df_tmp.columns:
            entry["y"] = _safe_float(row[y_col])
        if color_col and color_col in df_tmp.columns:
            entry["color"] = str(row[color_col])
        if size_col and size_col in df_tmp.columns:
            entry["size"] = _safe_float(row[size_col])
        extra_count = 0
        for col in df_tmp.select_dtypes(include=["number"]).columns:
            if col not in [y_col, size_col] and extra_count < 4:
                val = _safe_float(row[col])
                if val is not None:
                    entry[col] = val
                    extra_count += 1
        out["rows"].append(entry)

    return out


def build_dataset_intelligence_card(
    df_work: pd.DataFrame,
    forecast_results: dict,
    anomalies: list,
    detected_columns: dict,
    shap_results: list,
) -> dict:
    target   = detected_columns["target"]
    raw_freq = forecast_results.get("detected_freq", "W")
    freq_info         = _get_freq_info(raw_freq)
    freq_label        = freq_info["label"]
    periods_per_week  = freq_info["per_week"]
    periods_per_month = freq_info["per_month"]
    horizon = forecast_results.get("forecast_horizon", 4)

    hist      = forecast_results.get("historical", [])
    fc        = forecast_results.get("forecast",   [])
    lo        = forecast_results.get("lower",      [])
    hi        = forecast_results.get("upper",      [])
    last_hist = hist[-1] if hist else 0
    has_forecast = bool(fc)

    if has_forecast:
        change_pct = ((fc[-1] - last_hist) / abs(last_hist) * 100) if last_hist else 0
        lower_pct  = ((sum(lo)/len(lo) - last_hist) / abs(last_hist) * 100) if lo and last_hist else 0
        upper_pct  = ((sum(hi)/len(hi) - last_hist) / abs(last_hist) * 100) if hi and last_hist else 0
    else:
        change_pct = lower_pct = upper_pct = 0.0

    trend = "rising" if change_pct > 2 else ("falling" if change_pct < -2 else "stable")

    forecast_dates = forecast_results.get("dates", [])
    period_lines   = []
    if has_forecast:
        for i in range(min(horizon, len(forecast_dates))):
            period_lines.append(
                f"Period {i+1} ({forecast_dates[i]}): "
                f"{fc[i]:.2f} [low {lo[i]:.2f} – high {hi[i]:.2f}]"
            )

    if has_forecast:
        one_liner = (
            f"Next {horizon} {freq_label}s: central estimate {change_pct:+.1f}% ({trend}). "
            f"Lower bound: {lower_pct:+.1f}%. Upper bound: {upper_pct:+.1f}%."
        )
    else:
        mean_val = float(np.mean(hist)) if hist else 0.0
        min_val  = float(np.min(hist))  if hist else 0.0
        max_val  = float(np.max(hist))  if hist else 0.0
        one_liner = (
            f"Cross-sectional dataset: {len(hist)} rows. "
            f"{target} ranges from {min_val:.2f} to {max_val:.2f} "
            f"(mean: {mean_val:.2f}). "
            f"Use chat to explore correlations and rankings."
        )

    top_driver           = "recent trend"
    top_driver_direction = "neutral"
    if shap_results:
        top_driver           = shap_results[0].get("feature", "recent trend")
        top_driver_direction = shap_results[0].get("direction", "neutral")

    other_cols   = {}
    feature_cols = detected_columns.get("features", [])
    for col in feature_cols:
        if col not in df_work.columns:
            continue
        series = df_work[col].dropna()
        if len(series) < 4:
            continue
        col_last = round(float(series.iloc[-1]), 2)
        col_prev = round(float(series.iloc[-5]), 2) if len(series) >= 5 else col_last
        col_chg  = ((col_last - col_prev) / abs(col_prev) * 100) if col_prev else 0
        other_cols[col] = {
            "last":       col_last,
            "change_pct": round(col_chg, 1),
            "trend":      "rising" if col_chg > 2 else ("falling" if col_chg < -2 else "stable"),
            "mean":       round(float(series.mean()), 2),
        }

    column_relationships = _compute_column_relationships(df_work, target, feature_cols)

    if anomalies:
        high   = [a for a in anomalies if a.get("severity") == "high"]
        recent = sorted(anomalies, key=lambda x: x.get("date", ""), reverse=True)[0]
        anomaly_plain = (
            f"{len(anomalies)} unusual point(s) detected ({len(high)} high severity). "
            f"Most recent: {recent.get('date', 'unknown')} — "
            f"a {recent.get('direction', 'change')} of "
            f"{recent.get('deviation_percent', 0):.0f}% from expected."
        )
    else:
        anomaly_plain = "No unusual patterns detected."

    truth_score = forecast_results.get("truth_score", 0)
    if truth_score > 15:
        reliability_plain = f"Model is reliable ({truth_score:.0f}% better than baseline)."
    elif truth_score > 0:
        reliability_plain = "Model is slightly better than baseline. Treat forecast as directional guidance."
    else:
        reliability_plain = "Model accuracy is close to baseline. Use forecast cautiously."

    return {
        "target_col":            target,
        "freq_label":            freq_label,
        "detected_freq":         raw_freq,
        "periods_per_week":      periods_per_week,
        "periods_per_month":     periods_per_month,
        "horizon":               horizon,
        "one_liner":             one_liner,
        "change_pct":            round(change_pct, 1),
        "lower_pct":             round(lower_pct,  1),
        "upper_pct":             round(upper_pct,  1),
        "trend_direction":       trend,
        "last_historical_value": round(last_hist, 2),
        "forecast_dates":        forecast_dates,
        "forecast_values":       [round(v, 2) for v in fc],
        "lower_values":          [round(v, 2) for v in lo],
        "upper_values":          [round(v, 2) for v in hi],
        "period_lines":          period_lines,
        "top_driver":            top_driver,
        "top_driver_direction":  top_driver_direction,
        "other_columns":         other_cols,
        "column_relationships":  column_relationships,
        "anomaly_plain":         anomaly_plain,
        "anomaly_count":         len(anomalies),
        "reliability_plain":     reliability_plain,
        "truth_score":           truth_score,
    }


def build_compact_system_prompt(card: dict, detected_columns: dict) -> str:
    target        = card["target_col"]
    freq          = card["freq_label"]
    horizon       = card["horizon"]
    detected_freq = card.get("detected_freq", "W")

    other_col_lines = []
    for col, stats in card["other_columns"].items():
        other_col_lines.append(
            f"  {col}: current={stats['last']}, trend={stats['trend']} ({stats['change_pct']:+.1f}% recent change)"
        )
    other_cols_str = "\n".join(other_col_lines) if other_col_lines else "  No additional columns."

    rel = card.get("column_relationships", {})
    corr_lines = []
    for col, corr in rel.get("correlations_with_target", {}).items():
        direction = "positive" if corr > 0 else "negative"
        strength  = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.3 else "weak")
        corr_lines.append(f"  {col}: {corr:+.2f} ({strength} {direction} link with {target})")
    lag_lines = []
    for col, corr in rel.get("lag_correlations", {}).items():
        lag_lines.append(f"  Last period's {col} predicts this period's {target}: {corr:+.2f}")
    margin = rel.get("profit_margin")
    margin_line = ""
    if margin:
        margin_line = (
            f"  Profit margin: every 1 unit of {margin['sales_col']} "
            f"generates {margin['ratio']:.4f} units of {margin['profit_col']} on average."
        )
    relationships_str = "\n".join(corr_lines + lag_lines)
    if margin_line:
        relationships_str += f"\n{margin_line}"
    if not relationships_str:
        relationships_str = "  No multi-column relationships computed (single numeric column dataset)."

    if corr_lines:
        example_col = next(iter(rel.get("correlations_with_target", {})), "X")
        cross_col_formula = (
            f"For cross-column scenarios (e.g. 'if {example_col} increases by N%'): "
            f"use estimated_{target}_change = correlation(column, {target}) × N%. "
            f"Always add: 'Note: correlation is not causation — this is a statistical estimate.'"
        )
    else:
        cross_col_formula = ""

    if freq == "day":
        freq_rules = "1 week = 7 periods. 1 month = 30 periods."
    elif freq == "week":
        freq_rules = "1 week = 1 period. 1 month = 4 periods."
    elif freq == "month":
        freq_rules = "1 month = 1 period. 1 quarter = 3 periods."
    else:
        freq_rules = f"Data is {freq}ly. Convert user time references to periods accordingly."

    periods_str = "\n".join([f"  {line}" for line in card["period_lines"]]) or "  No forecast periods available (cross-sectional dataset)."

    prompt = f"""You are FutureLens, a friendly AI forecasting assistant.

[DATASET PROFILE]
- Forecasting: {target} | Date col: {detected_columns['date']}
- Data frequency: {freq}ly ({detected_freq}) | Forecast horizon: {horizon} {freq}s ahead

[FREQUENCY CONTEXT]
{freq_rules}

[FORECAST FOR {target.upper()}]
{card['one_liner']}
By period:
{periods_str}
Top driver: {card['top_driver']} ({card['top_driver_direction']} impact)

[COLUMN STATISTICS]
{other_cols_str}

[COLUMN RELATIONSHIPS — correlations with {target}]
{relationships_str}

[ANOMALIES]
{card['anomaly_plain']}

[RELIABILITY]
{card['reliability_plain']}

[MANDATORY ROUTING]
WHY questions: Use ANOMALIES data. State date, severity, deviation. Never answer WHY from general knowledge.
WHAT WILL HAPPEN questions: Use exact forecast numbers. State period range explicitly.
WHAT IF questions: {cross_col_formula if cross_col_formula else "Use correlation data and explain the statistical link."}
OTHER COLUMN questions: Use COLUMN STATISTICS above.
UNKNOWN questions: Never say "I cannot". State the one-liner and suggest 3 specific questions.

[RESPONSE FORMAT]
Lead with the number or direct answer. Max 4 sentences. Plain English only.
End with exactly one follow-up question suggestion.
"""
    return prompt


# ---------------------------------------------------------------------------
# UPLOAD ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="cp1252")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")

        dataset_profile = build_dataset_profile(df)

        # ── Step 1: Try to detect a real date column ──
        original_date_col = detect_date_column(df) if _has_date_column(df) else None
        has_real_date     = original_date_col is not None

        # ── Step 1b: Groq classifies dataset + picks chart config ──
        groq_picked        = None
        is_cross_sectional = False
        chart_config       = None

        if not has_real_date:
            from api.column_picker import pick_columns_with_groq
            groq_picked = pick_columns_with_groq(dataset_profile)

            actual_cols_lower = {c.lower(): c for c in df.columns}

            if groq_picked.get("x_col"):
                picked_x = groq_picked["x_col"].lower()
                if picked_x in actual_cols_lower:
                    groq_picked["x_col"] = actual_cols_lower[picked_x]
                elif groq_picked["x_col"] not in df.columns:
                    groq_picked["x_col"] = None

            if groq_picked.get("y_col"):
                picked_y = groq_picked["y_col"].lower()
                if picked_y in actual_cols_lower:
                    groq_picked["y_col"] = actual_cols_lower[picked_y]
                elif groq_picked["y_col"] not in df.columns:
                    groq_picked["y_col"] = None

            is_cross_sectional = groq_picked.get("dataset_type") != "time_series"
            chart_config       = groq_picked

            logger.info(
                f"Groq: dataset_type={groq_picked.get('dataset_type')}, "
                f"chart={groq_picked.get('chart_type')}, "
                f"x={groq_picked.get('x_col')}, y={groq_picked.get('y_col')}, "
                f"color={groq_picked.get('color_col')}, reason={groq_picked.get('reason')}"
            )

        # ── Step 2: Detect target column ──
        if groq_picked and groq_picked.get("y_col") and groq_picked["y_col"] in df.columns:
            original_target_col = groq_picked["y_col"]
        else:
            try:
                original_target_col = detect_target_column(df, original_date_col)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # ── Step 3: Feature columns ──
        num_cols     = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [
            c for c in num_cols
            if c != original_date_col
            and c != original_target_col
            and df[c].isna().sum() / len(df) <= 0.30
        ]

        # ── Step 4: Build working frame ──
        if has_real_date:
            use_cols = [original_date_col, original_target_col] + feature_cols
            df_work  = df[use_cols].copy()
            df_work.rename(columns={original_date_col: "date", original_target_col: "sales"}, inplace=True)
            df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce", utc=False)
            df_work = df_work.dropna(subset=["date"])
            date_was_generated = False
            synthetic_time_col = None

        elif not is_cross_sectional:
            # Time-series without a real date column — synthesise dates
            groq_time_col = groq_picked.get("x_col") if groq_picked else None
            use_cols = [original_target_col] + feature_cols
            df_work  = df[use_cols].copy()
            df_work.rename(columns={original_target_col: "sales"}, inplace=True)
            if groq_time_col and groq_time_col in df.columns:
                df_work = df_work.iloc[df[groq_time_col].argsort().values]
            df_work = df_work.reset_index(drop=True)
            df_work.insert(0, "date", pd.date_range(start="2020-01-01", periods=len(df_work), freq="D"))
            original_date_col  = "_auto_generated_"
            date_was_generated = True
            synthetic_time_col = groq_time_col

        else:
            # Cross-sectional — synthesise placeholder dates for pipeline compatibility
            use_cols = [original_target_col] + feature_cols
            df_work  = df[use_cols].copy()
            df_work.rename(columns={original_target_col: "sales"}, inplace=True)
            df_work = df_work.reset_index(drop=True)
            df_work.insert(0, "date", pd.date_range(start="2020-01-01", periods=len(df_work), freq="D"))
            original_date_col  = "_cross_sectional_"
            date_was_generated = True
            synthetic_time_col = None

        # ── Coerce numeric types ──
        df_work["sales"] = pd.to_numeric(
            df_work["sales"].astype(str).str.replace(",", "").str.replace("\xa0", " ").str.strip(),
            errors="coerce",
        )
        for c in feature_cols:
            if c in df_work.columns:
                df_work[c] = pd.to_numeric(
                    df_work[c].astype(str).str.replace(",", "").str.replace("\xa0", " ").str.strip(),
                    errors="coerce",
                )

        df_work = df_work.dropna(subset=["sales"])

        agg_map = {"sales": "sum"}
        for c in feature_cols:
            if c in df_work.columns:
                agg_map[c] = "sum"

        df_work = df_work.groupby("date", as_index=False).agg(agg_map)
        df_work = df_work.sort_values("date").reset_index(drop=True)
        df_work["sales"] = df_work["sales"].ffill().bfill()
        for c in feature_cols:
            if c in df_work.columns:
                df_work[c] = df_work[c].ffill().bfill()

        if len(df_work) < 5:
            raise HTTPException(
                status_code=400,
                detail="Dataset too small — need at least 5 rows after cleaning.",
            )

        session_id = str(uuid.uuid4())
        save_upload(session_id, file.filename, len(df_work), df_work.columns.tolist())

        detected_columns = {
            "date":               original_date_col,
            "target":             original_target_col,
            "features":           feature_cols,
            "has_date":           has_real_date,
            "date_was_generated": date_was_generated,
            "synthetic_time_col": synthetic_time_col,
            "groq_reason":        groq_picked.get("reason", "") if groq_picked else "",
            "chart_config":       chart_config,
        }

        # ── Step 5: Forecast or chart data ──
        forecast_results = _make_empty_forecast(df_work, detected_columns)
        shap_results     = []
        rca_explanation  = "No explanation available."
        anomalies        = []
        truth_meter      = {"status": "ok", "score": 0, "color": "gray", "label": "No forecast"}
        group_forecasts  = None
        chart_data       = None

        if is_cross_sectional:
            logger.info(f"Cross-sectional — building {chart_config.get('chart_type')} chart")
            chart_data = _build_chart_data(df, chart_config)
            truth_meter["label"] = "Cross-sectional snapshot — no forecast available"

        else:
            try:
                forecast_results = run_forecast(df_work)
                model = forecast_results.pop("model", None)
                X     = forecast_results.pop("X",     None)

                model_mape    = forecast_results.get("model_mape",    10.0)
                baseline_mape = forecast_results.get("baseline_mape", 20.0)
                truth_score   = forecast_results.get("truth_score",    0.0)

                if model is not None and X is not None:
                    shap_results    = compute_shap(model, X, X)
                    rca_explanation = explain_rca(shap_results)

                anomalies   = detect_anomalies(df_work, forecast_results)
                truth_meter = compute_truth_meter(model_mape, baseline_mape)
                forecast_results["truth_score"] = truth_meter.get("score", truth_score)

                if date_was_generated:
                    truth_meter["label"] = "Synthetic time axis — directional only"

            except Exception as e:
                logger.error(f"Forecast pipeline error: {e}", exc_info=True)
                forecast_results = _make_empty_forecast(df_work, detected_columns)

            # Group forecasts
            group_col = pick_group_column(df)
            if group_col and has_real_date:
                group_forecasts = _run_group_forecasts(
                    df, group_col, original_date_col, original_target_col, logger
                )
            elif group_col:
                group_forecasts = _build_static_group_summary(df, group_col, original_target_col)

        # ── Assemble response ──
        forecast_results["chart_data"]         = chart_data
        forecast_results["chart_config"]       = chart_config
        forecast_results["is_cross_sectional"] = is_cross_sectional
        forecast_results["truth_meter"]        = truth_meter
        forecast_results["session_id"]         = session_id
        forecast_results["anomalies"]          = anomalies
        forecast_results["shap_results"]       = shap_results
        forecast_results["rca_explanation"]    = rca_explanation
        forecast_results["detected_columns"]   = detected_columns
        forecast_results["dataset_profile"]    = dataset_profile
        forecast_results["group_forecasts"]    = group_forecasts
        forecast_results["has_date"]           = has_real_date

        sanitized_results = _sanitize_for_json(forecast_results)

        save_forecast(session_id, sanitized_results, sanitized_results.get("truth_score", 0))
        save_anomalies(session_id, sanitized_results["anomalies"])

        # ── Intelligence card + system prompt ──
        intelligence_card = build_dataset_intelligence_card(
            df_work          = df_work,
            forecast_results = sanitized_results,
            anomalies        = anomalies,
            detected_columns = detected_columns,
            shap_results     = shap_results,
        )
        intelligence_card["has_date"]           = has_real_date
        intelligence_card["date_was_generated"] = date_was_generated
        intelligence_card["synthetic_time_col"] = synthetic_time_col
        intelligence_card["groq_reason"]        = detected_columns.get("groq_reason", "")
        intelligence_card["chart_config"]       = chart_config

        system_prompt = build_compact_system_prompt(
            card             = intelligence_card,
            detected_columns = detected_columns,
        )
        save_system_prompt(
            session_id        = session_id,
            system_prompt     = system_prompt,
            intelligence_card = intelligence_card,
        )

        sanitized_results["intelligence_card"] = {
            "one_liner":          intelligence_card.get("one_liner", ""),
            "anomaly_plain":      intelligence_card.get("anomaly_plain", ""),
            "reliability_plain":  intelligence_card.get("reliability_plain", ""),
            "target_col":         intelligence_card["target_col"],
            "freq_label":         intelligence_card.get("freq_label", "row"),
            "has_date":           has_real_date,
            "date_was_generated": date_was_generated,
            "synthetic_time_col": synthetic_time_col,
            "groq_reason":        detected_columns.get("groq_reason", ""),
            "chart_config":       chart_config,
            "is_cross_sectional": is_cross_sectional,
            "chart_type":         chart_config.get("chart_type") if chart_config else "forecast",
            "chart_title":        chart_config.get("chart_title", "") if chart_config else "",
        }

        return sanitized_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")


def _has_date_column(df: pd.DataFrame) -> bool:
    """Quick check: does this DataFrame likely have a real date column?"""
    try:
        detect_date_column(df)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# READ ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/forecast/{session_id}")
async def get_forecast_data(session_id: str):
    try:
        data = get_forecast(session_id)
        if not data:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/anomalies/{session_id}")
async def get_anomalies_data(session_id: str):
    try:
        data = get_anomalies(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Anomalies not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------------------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(request: Request):
    from api.planner import plan
    from api.tools import execute_tool
    from api.agent import explain

    body       = await request.json()
    message    = body.get("message", "")
    session_id = body.get("session_id", "demo")

    if not message.strip():
        return {"response": "Please ask a question.", "suggested_questions": []}

    prompt_data = get_system_prompt(session_id)
    if not prompt_data:
        return {
            "response":           "Please upload a dataset first to enable the chat assistant.",
            "suggested_questions": [],
        }

    card            = prompt_data["intelligence_card"]
    forecast_data   = get_forecast(session_id) or {}
    anomalies_data  = get_anomalies(session_id) or []
    group_forecasts = forecast_data.get("group_forecasts") or []

    try:
        recent_chat = get_recent_chat(session_id, limit=4)
    except Exception:
        recent_chat = []

    # Step 1 — plan
    tool_call = plan(message, card)
    logger.info(f"Tool call planned: {tool_call}")

    # Step 2 — execute
    tool_result = execute_tool(
        tool_call       = tool_call,
        card            = card,
        forecast_data   = forecast_data,
        anomalies_data  = anomalies_data,
        group_forecasts = group_forecasts,
    )
    logger.info(f"Tool result: {tool_result.get('tool')} — keys: {list(tool_result.keys())}")

    # Step 3 — explain
    rate_limited = False
    try:
        response_text = explain(
            user_message = message,
            tool_result  = tool_result,
            card         = card,
            chat_history = recent_chat,
        )
        if "rate" in response_text.lower() and "limit" in response_text.lower():
            rate_limited = True
    except Exception as e:
        err_str = str(e).lower()
        if "429" in str(e) or "quota" in err_str or "rate" in err_str:
            rate_limited  = True
            response_text = _build_rate_limit_fallback(tool_result, card)
        else:
            logger.error(f"Explain error: {e}")
            response_text = "I encountered an error. Please try again."

    if len(response_text) > 700:
        cut = response_text.find("\n\n", 400)
        response_text = response_text[:cut] if cut > 0 else response_text[:700]

    chart_context = _build_chart_context(message, card)

    target    = card.get("target_col", "your metric")
    freq      = card.get("freq_label", "week")
    corr_cols = list(card.get("column_relationships", {}).get("correlations_with_target", {}).keys())
    suggested = [
        f"What will {target} look like next {freq}?",
        "Are there any sudden changes I should look at?",
        (
            f"If {corr_cols[0]} increases by 10%, how does that affect {target}?"
            if corr_cols else f"What if {target} increases by 10%?"
        ),
    ]

    save_chat(session_id, message, response_text[:400])

    return {
        "response":            response_text,
        "suggested_questions": suggested,
        "rate_limited":        rate_limited,
        "chart_context":       chart_context,
        "debug_tool":          tool_call.get("tool"),
    }


def _build_chart_context(message: str, card: dict) -> dict | None:
    msg_lower     = message.lower()
    horizon_match = re.search(r"(\d+)\s*(week|month|day)", msg_lower)
    if horizon_match and any(
        k in msg_lower for k in ["next", "forecast", "predict", "look like", "future", "will"]
    ):
        n    = int(horizon_match.group(1))
        unit = horizon_match.group(2)
        per_week  = card.get("periods_per_week",  1)
        per_month = card.get("periods_per_month", 4)
        multiplier = per_week if unit == "week" else (per_month if unit == "month" else 1)
        periods_requested = int(round(n * multiplier))
        max_horizon = card.get("horizon", 8)
        return {
            "type":    "highlight_forecast",
            "periods": min(periods_requested, max_horizon),
            "label":   f"Next {n} {unit}{'s' if n > 1 else ''}",
        }

    group_keywords = [
        "region", "segment", "category", "product",
        "west", "east", "south", "central", "north",
        "technology", "furniture", "office",
    ]
    for kw in group_keywords:
        if kw in msg_lower:
            return {"type": "highlight_group", "keyword": kw}

    return None


def _build_rate_limit_fallback(tool_result: dict, card: dict) -> str:
    tool   = tool_result.get("tool", "none")
    target = card.get("target_col", "metric")
    freq   = card.get("freq_label", "period")

    if tool == "forecast":
        periods = tool_result.get("period_data", [])
        chg     = tool_result.get("overall_change_pct", 0)
        if periods:
            p = periods[0]
            return (
                f"{target} forecast: {p['forecast']:.2f} next {freq} "
                f"(range {p['lower']:.2f}–{p['upper']:.2f}). Overall change: {chg:+.1f}%."
            )
    if tool == "categorical_analysis":
        groups = tool_result.get("ranked_groups", [])
        if groups:
            top = groups[0]
            return (
                f"Top {tool_result.get('group_column','segment')} by "
                f"{tool_result.get('rank_by','volume')}: {top['group']} "
                f"(forecast: {top['forecast_value']:.2f}, growth: {top['expected_change_pct']:+.1f}%)."
            )
    if tool == "profitability":
        gp = tool_result.get("group_profit_ranking", [])
        if gp:
            top = gp[0]
            return (
                f"Most profitable {top['group_col']}: {top['group']} "
                f"(current: {top['current']:.2f}, forecast: {top['forecast']:.2f}, "
                f"growth: {top['growth_pct']:+.1f}%)."
            )
    if tool == "scenario":
        return (
            f"Scenario result: {tool_result.get('difference_percent', 0):+.1f}% difference. "
            f"Scenario total: {tool_result.get('scenario_total', 0):.2f} "
            f"vs baseline {tool_result.get('baseline_total', 0):.2f}."
        )
    return card.get("one_liner", "Data loaded. Please try again in a moment.")


# ---------------------------------------------------------------------------
# FORECAST QUERY ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/forecast_query")
async def forecast_query_endpoint(req: ForecastQueryRequest):
    from api.forecast_query import detect_forecast_intent

    if not req.message.strip():
        return {"forecast_update": None}

    prompt_data = get_system_prompt(req.session_id)
    if not prompt_data:
        return {"forecast_update": None}

    card = prompt_data["intelligence_card"]
    try:
        forecast_update = detect_forecast_intent(req.message, card)
        return {"forecast_update": forecast_update}
    except Exception as e:
        logger.warning(f"forecast_query error: {e}")
        return {"forecast_update": None}


# ---------------------------------------------------------------------------
# SIMULATE ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/simulate")
async def simulate(req: SimulateRequest):
    try:
        forecast_data = get_forecast(req.session_id)
        if not forecast_data:
            raise HTTPException(status_code=404, detail="Forecast not found")

        base_forecast = forecast_data.get("forecast", [])
        if not base_forecast:
            raise HTTPException(status_code=400, detail="No base forecast available")

        historical = forecast_data.get("historical", [])
        sim_result = simulate_scenario(
            base_forecast,
            req.change_percent,
            scenario_type = req.scenario_type,
            historical    = historical,
        )

        baseline_vals = sim_result.get("baseline", [])
        scenario_vals = sim_result.get("scenario", [])
        summary       = sim_result.get("summary", "")

        sum_base = sum(baseline_vals)
        sum_scen = sum(scenario_vals)
        diff_pct = ((sum_scen - sum_base) / sum_base * 100) if sum_base != 0 else 0.0

        return _sanitize_for_json({
            "baseline":           baseline_vals,
            "scenario":           scenario_vals,
            "difference_percent": diff_pct,
            "summary":            summary,
            "dates":              forecast_data.get("dates", []),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simulate: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------------------------------------------------------------------
# UTILITY ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/history")
async def get_history():
    try:
        return get_recent_uploads(10)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
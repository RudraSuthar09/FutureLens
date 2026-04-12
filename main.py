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
    save_system_prompt, get_system_prompt
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


def build_dataset_intelligence_card(
    df_work: pd.DataFrame,
    forecast_results: dict,
    anomalies: list,
    detected_columns: dict,
    shap_results: list,
) -> dict:
    target   = detected_columns["target"]
    raw_freq = forecast_results.get("detected_freq", "W")
    freq_info        = _get_freq_info(raw_freq)
    freq_label       = freq_info["label"]
    periods_per_week  = freq_info["per_week"]
    periods_per_month = freq_info["per_month"]
    horizon = forecast_results.get("forecast_horizon", 4)

    hist     = forecast_results["historical"]
    fc       = forecast_results["forecast"]
    lo       = forecast_results["lower"]
    hi       = forecast_results["upper"]
    last_hist = hist[-1] if hist else 0

    change_pct = ((fc[-1] - last_hist) / abs(last_hist) * 100) if last_hist else 0
    lower_pct  = ((sum(lo) / len(lo) - last_hist) / abs(last_hist) * 100) if last_hist and lo else 0
    upper_pct  = ((sum(hi) / len(hi) - last_hist) / abs(last_hist) * 100) if last_hist and hi else 0
    trend = "rising" if change_pct > 2 else ("falling" if change_pct < -2 else "stable")

    forecast_dates = forecast_results.get("dates", [])
    period_lines = []
    for i in range(min(horizon, len(forecast_dates))):
        period_lines.append(
            f"Period {i+1} ({forecast_dates[i]}): "
            f"{fc[i]:.2f} [low {lo[i]:.2f} – high {hi[i]:.2f}]"
        )

    one_liner = (
        f"Next {horizon} {freq_label}s: central estimate {change_pct:+.1f}% ({trend}). "
        f"Lower bound: {lower_pct:+.1f}%. Upper bound: {upper_pct:+.1f}%."
    )

    top_driver = "recent trend"
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
        "target_col":           target,
        "freq_label":           freq_label,
        "detected_freq":        raw_freq,
        "periods_per_week":     periods_per_week,
        "periods_per_month":    periods_per_month,
        "horizon":              horizon,
        "one_liner":            one_liner,
        "change_pct":           round(change_pct, 1),
        "lower_pct":            round(lower_pct, 1),
        "upper_pct":            round(upper_pct, 1),
        "trend_direction":      trend,
        "last_historical_value": round(last_hist, 2),
        "forecast_dates":       forecast_dates,
        "forecast_values":      [round(v, 2) for v in fc],
        "lower_values":         [round(v, 2) for v in lo],
        "upper_values":         [round(v, 2) for v in hi],
        "period_lines":         period_lines,
        "top_driver":           top_driver,
        "top_driver_direction": top_driver_direction,
        "other_columns":        other_cols,
        "column_relationships": column_relationships,
        "anomaly_plain":        anomaly_plain,
        "anomaly_count":        len(anomalies),
        "reliability_plain":    reliability_plain,
        "truth_score":          truth_score,
    }


def build_compact_system_prompt(card: dict, detected_columns: dict) -> str:
    target       = card["target_col"]
    freq         = card["freq_label"]
    horizon      = card["horizon"]
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
        freq_rules = "1 week = 7 periods. 1 month = 30 periods. 'Next N weeks' means N×7 periods."
    elif freq == "week":
        freq_rules = "1 week = 1 period. 1 month = 4 periods. 'Next N months' means N×4 periods."
    elif freq == "month":
        freq_rules = "1 month = 1 period. 1 quarter = 3 periods. 'Next N weeks' means N×0.25 periods (round up)."
    else:
        freq_rules = f"Data is {freq}ly. Convert user time references to periods accordingly."

    periods_str = "\n".join([f"  {line}" for line in card["period_lines"]])

    prompt = f"""You are FutureLens, a friendly AI forecasting assistant.

[DATASET PROFILE]
- Forecasting: {target} | Date col: {detected_columns['date']}
- Data frequency: {freq}ly ({detected_freq}) | Forecast horizon: {horizon} {freq}s ahead

[FREQUENCY CONTEXT]
{freq_rules}
When user says "next N weeks/months", convert N to periods using the rules above before answering.
Always state how many periods your answer covers.

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

[MANDATORY ROUTING — follow exactly]
WHY questions (why did X happen / what caused the drop): Use ANOMALIES data above first. State the anomaly date, severity and deviation. Then explain in plain English. Never answer WHY from general knowledge.
WHAT WILL HAPPEN questions (next N weeks/months/forecast): Convert N to periods using FREQUENCY CONTEXT. Use exact forecast numbers from FORECAST section above. State period range explicitly.
WHAT IF / SCENARIO questions (if X increases by N%):
  - If question is about {target} itself: "Under a +N% scenario, {target} reaches [{card['last_historical_value']:.2f} × (1+N/100)] (vs {card['last_historical_value']:.2f} baseline). Range: [value×0.95]–[value×1.05]."
  - If question links another column to {target}: {cross_col_formula if cross_col_formula else "Use the correlation data in COLUMN RELATIONSHIPS and explain the statistical link."}
OTHER COLUMN questions: Use COLUMN STATISTICS above. "While my full forecast is for {target}, here is what I see for [col]: current=[last], trend=[trend]."
UNKNOWN questions: Never say "I cannot". State the one-liner and suggest 3 specific questions.

[RESPONSE FORMAT]
Lead with the number or direct answer. Max 4 sentences. Plain English only — no technical terms.
End with exactly one follow-up question suggestion.
Never refuse when data is available above. Never use general business knowledge when specific data is provided.
"""

    word_count = len(prompt.split())
    if word_count > 700:
        logger.warning(
            f"System prompt is {word_count} words (target ≤700). "
            f"Reduce other_columns count (currently {len(card['other_columns'])})."
        )
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

        try:
            original_date_col = detect_date_column(df)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            original_target_col = detect_target_column(df, original_date_col)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [
            c for c in num_cols
            if c != original_date_col and c != original_target_col
            and df[c].isna().sum() / len(df) <= 0.30
        ]

        use_cols = [original_date_col, original_target_col] + feature_cols
        df_work  = df[use_cols].copy()
        df_work.rename(columns={original_date_col: "date", original_target_col: "sales"}, inplace=True)

        df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce", utc=False)
        df_work = df_work.dropna(subset=["date"])

        df_work["sales"] = pd.to_numeric(
            df_work["sales"].astype(str)
                            .str.replace(",", "")
                            .str.replace("\xa0", " ")
                            .str.strip(),
            errors="coerce",
        )

        for c in feature_cols:
            if c in df_work.columns:
                df_work[c] = pd.to_numeric(
                    df_work[c].astype(str)
                               .str.replace(",", "")
                               .str.replace("\xa0", " ")
                               .str.strip(),
                    errors="coerce",
                )

        df_work = df_work.dropna(subset=["sales"])

        # FIX 1 — use SUM not MEAN when aggregating duplicate dates.
        # Transactional CSVs (Superstore, retail) have many rows per day;
        # summing gives the correct daily total, not an average of individual orders.
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

        if len(df_work) < 20:
            raise HTTPException(
                status_code=400,
                detail="Dataset too small — need at least 20 rows after cleaning.",
            )

        session_id = str(uuid.uuid4())
        save_upload(session_id, file.filename, len(df_work), df_work.columns.tolist())

        # Step 5 — forecast
        forecast_results = run_forecast(df_work)
        model = forecast_results.pop("model", None)
        X     = forecast_results.pop("X", None)

        model_mape    = forecast_results.get("model_mape",    10.0)
        baseline_mape = forecast_results.get("baseline_mape", 20.0)
        truth_score   = forecast_results.get("truth_score",    0.0)

        # Step 6 — SHAP / RCA
        shap_results    = []
        rca_explanation = "No explanation provided"
        if model is not None and X is not None:
            shap_results    = compute_shap(model, X, X)
            rca_explanation = explain_rca(shap_results)

        # Step 7 — anomaly detection
        anomalies = detect_anomalies(df_work, forecast_results)

        # Step 8 — truth meter
        truth_meter = compute_truth_meter(model_mape, baseline_mape)
        forecast_results["truth_score"] = truth_meter.get("score", truth_score)

        # Step 8.5 — group forecasts
        # FIX 2 — resample each group to weekly SUM before forecasting.
        # Previously used daily mean which produced tiny last_hist values
        # (e.g. 3.46) causing nonsensical ±500% change estimates.
        group_forecasts = None
        group_col = pick_group_column(df)

        if group_col:
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
                df_tmp = df_tmp.dropna(
                    subset=[original_date_col, original_target_col, group_col]
                )

                top_groups = (
                    df_tmp.groupby(group_col)[original_target_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .index
                    .tolist()
                )

                group_forecasts = []
                for g in top_groups:
                    gdf = df_tmp[df_tmp[group_col] == g][
                        [original_date_col, original_target_col]
                    ].copy()
                    gdf.rename(
                        columns={
                            original_date_col:    "date",
                            original_target_col:  "sales",
                        },
                        inplace=True,
                    )
                    gdf["date"]  = pd.to_datetime(gdf["date"], errors="coerce")
                    gdf["sales"] = pd.to_numeric(gdf["sales"], errors="coerce")
                    gdf = gdf.dropna(subset=["date", "sales"])

                    # Resample to weekly SUM — matches main forecast pipeline
                    gdf = (
                        gdf.set_index("date")["sales"]
                        .resample("W")
                        .sum()
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
                    # Cap at ±200%: anything beyond is a data artefact
                    pct = float(np.clip(pct, -200.0, 200.0))

                    group_forecasts.append({
                        "group_col":              group_col,
                        "group":                  str(g),
                        "forecast_horizon":       gres.get("forecast_horizon"),
                        "expected_change_percent": round(pct, 2),
                        "last_hist":              round(last_hist, 2),
                        "last_forecast":          round(last_fc,   2),
                    })

                group_forecasts.sort(
                    key=lambda x: x["expected_change_percent"], reverse=True
                )
            except Exception as e:
                logger.warning(f"Group forecasts skipped: {e}")
                group_forecasts = None

        # Step 9 — assemble
        forecast_results["truth_meter"]      = truth_meter
        forecast_results["session_id"]       = session_id
        forecast_results["anomalies"]        = anomalies
        forecast_results["shap_results"]     = shap_results
        forecast_results["rca_explanation"]  = rca_explanation
        forecast_results["detected_columns"] = {
            "date":     original_date_col,
            "target":   original_target_col,
            "features": feature_cols,
        }
        forecast_results["dataset_profile"]  = dataset_profile
        forecast_results["group_forecasts"]  = group_forecasts

        sanitized_results = _sanitize_for_json(forecast_results)

        save_forecast(
            session_id,
            sanitized_results,
            sanitized_results.get("truth_score", truth_score),
        )
        save_anomalies(session_id, sanitized_results["anomalies"])

        detected_columns = {
            "date":     original_date_col,
            "target":   original_target_col,
            "features": feature_cols,
        }
        intelligence_card = build_dataset_intelligence_card(
            df_work          = df_work,
            forecast_results = sanitized_results,
            anomalies        = anomalies,
            detected_columns = detected_columns,
            shap_results     = shap_results,
        )
        system_prompt = build_compact_system_prompt(
            card             = intelligence_card,
            detected_columns = detected_columns,
        )
        save_system_prompt(
            session_id     = session_id,
            system_prompt  = system_prompt,
            intelligence_card = intelligence_card,
        )

        sanitized_results["intelligence_card"] = {
            "one_liner":        intelligence_card["one_liner"],
            "anomaly_plain":    intelligence_card["anomaly_plain"],
            "reliability_plain": intelligence_card["reliability_plain"],
            "target_col":       intelligence_card["target_col"],
            "freq_label":       intelligence_card["freq_label"],
        }
        return sanitized_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")


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
            "response": "Please upload a dataset first to enable the chat assistant.",
            "suggested_questions": [],
        }

    card           = prompt_data["intelligence_card"]
    forecast_data  = get_forecast(session_id) or {}
    anomalies_data = get_anomalies(session_id) or []
    group_forecasts = forecast_data.get("group_forecasts") or []

    # Step 1 — planner picks the tool
    tool_call = plan(message, card)
    logger.info(f"Tool planned: {tool_call}")

    # Step 2 — Python executes the tool (no LLM involved)
    tool_result = execute_tool(
        tool_call      = tool_call,
        card           = card,
        forecast_data  = forecast_data,
        anomalies_data = anomalies_data,
        group_forecasts = group_forecasts,
    )
    logger.info(f"Tool result: {tool_result.get('tool')} — keys: {list(tool_result.keys())}")

    # Step 3 — Gemini explains the tool result in plain English
    rate_limited = False
    try:
        response_text = explain(
            user_message = message,
            tool_result  = tool_result,
            card         = card,
        )
        if "rate" in response_text.lower() and "limit" in response_text.lower():
            rate_limited = True
    except Exception as e:
        err_str = str(e).lower()
        if "429" in str(e) or "quota" in err_str or "rate" in err_str:
            rate_limited    = True
            response_text   = _build_rate_limit_fallback(tool_result, card)
        else:
            logger.error(f"Explain error: {e}")
            response_text = "I encountered an error. Please try again."

    if len(response_text) > 700:
        cut = response_text.find("\n\n", 400)
        response_text = response_text[:cut] if cut > 0 else response_text[:700]

    # FIX 3 — chart_context: tells the frontend what to highlight / render inline.
    # Detects horizon requests ("next 3 weeks") and group/segment questions.
    chart_context = _build_chart_context(message, card)

    # Suggested follow-up questions (dynamic, not hardcoded)
    target    = card.get("target_col", "your metric")
    freq      = card.get("freq_label", "week")
    corr_cols = list(
        card.get("column_relationships", {})
            .get("correlations_with_target", {})
            .keys()
    )
    suggested = [
        f"What will {target} look like next {freq}?",
        "Are there any sudden changes I should look at?",
        (
            f"If {corr_cols[0]} increases by 10%, how does that affect {target}?"
            if corr_cols else f"What if {target} increases by 10%?"
        ),
    ]

    save_chat(session_id, message, response_text[:250])

    return {
        "response":           response_text,
        "suggested_questions": suggested,
        "rate_limited":       rate_limited,
        "chart_context":      chart_context,   # consumed by app.py for inline charts
        "debug_tool":         tool_call.get("tool"),
    }


def _build_chart_context(message: str, card: dict) -> dict | None:
    """
    Parses the user message to produce a chart_context dict that app.py
    uses to render an inline mini-chart in the chat panel.

    Returns None when no chart update is warranted.
    """
    msg_lower = message.lower()

    # Detect explicit time horizon ("next 3 weeks", "next 2 months", etc.)
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

    # Detect group/segment keyword
    group_keywords = [
        "region", "segment", "category", "product",
        "west", "east", "south", "central", "north",
        "technology", "furniture", "office",
    ]
    for kw in group_keywords:
        if kw in msg_lower:
            return {
                "type":    "highlight_group",
                "keyword": kw,
            }

    return None


def _build_rate_limit_fallback(tool_result: dict, card: dict) -> str:
    """Template answer using real tool-result numbers — never leaks system prompt."""
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
                f"(range {p['lower']:.2f}–{p['upper']:.2f}). "
                f"Overall change: {chg:+.1f}%."
            )
    if tool == "categorical_analysis":
        groups = tool_result.get("ranked_groups", [])
        if groups:
            top = groups[0]
            return (
                f"Top {tool_result.get('group_column','segment')} by "
                f"{tool_result.get('rank_by','volume')}: {top['group']} "
                f"(forecast: {top['forecast_value']:.2f}, "
                f"growth: {top['expected_change_pct']:+.1f}%)."
            )
    if tool == "profitability":
        gp = tool_result.get("group_profit_ranking", [])
        if gp:
            top = gp[0]
            return (
                f"Most profitable {top['group_col']}: {top['group']} "
                f"(current: {top['current']:.2f}, "
                f"forecast: {top['forecast']:.2f}, "
                f"growth: {top['growth_pct']:+.1f}%)."
            )
    if tool == "scenario":
        return (
            f"Scenario result: {tool_result.get('difference_percent', 0):+.1f}% difference. "
            f"Scenario total: {tool_result.get('scenario_total', 0):.2f} "
            f"vs baseline {tool_result.get('baseline_total', 0):.2f}."
        )

    return card.get("one_liner", "Data loaded. Please try again in a moment.")

from pydantic import BaseModel
 
class ForecastQueryRequest(BaseModel):
    message:    str
    session_id: str = "demo"
 
 
@app.post("/forecast_query")
async def forecast_query_endpoint(req: ForecastQueryRequest):
    """
    Groq reads the user's message and returns a structured forecast_update
    telling the frontend how to update the Forecast tab chart.
 
    Completely independent of the chat endpoint.
    Returns {"forecast_update": {...}} or {"forecast_update": null}.
    """
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
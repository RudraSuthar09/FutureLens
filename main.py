import io
import uuid
import json
import os
import pandas as pd
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from api.database import (
    init_db, save_upload, save_forecast, save_anomalies, save_chat,
    get_forecast, get_anomalies, get_chat_history, get_recent_uploads
)
from api.forecaster import run_forecast, detect_date_column
from api.rca import compute_shap, explain_rca
from api.anomaly import detect_anomalies, compute_truth_meter
from api.agent import chat
from api.simulator import simulate_scenario

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    init_db()
    yield
    # Shutdown

app = FastAPI(title="FutureLens API", lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for Streamlit)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_context: list = []  # Maintain compatibility with frontend
    session_id: str = "demo"

class SimulateRequest(BaseModel):
    session_id: str
    change_percent: float
    scenario_type: str = "growth"

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types and datetimes """
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


def detect_target_column(df: pd.DataFrame, date_col: str) -> str:
    """
    Finds the best numeric column to forecast:
    1. First tries well-known domain names (AveragePrice, sales, price, …)
    2. Falls back to the column with the highest coefficient of variation (std/mean)
       — normalised so scale does not dominate the selection.
    - Excludes the date column
    - Excludes columns with >30% nulls
    - Excludes probable ID columns (all-unique integers, or name contains 'id')
    Raises ValueError if none found.
    """
    preferred_names = ["AveragePrice", "sales", "price", "revenue", "amount", "value", "close", "Close"]

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

    # 1. Preferred name match
    for name in preferred_names:
        if name in candidates:
            return name

    # 2. Highest coefficient of variation (relative variability, scale-independent)
    cv_scores = {}
    for col in candidates:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
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

    # numeric summary
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        profile["numeric_summary"][c] = {
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std()) if s.shape[0] > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }

    # categorical summary (object/category/bool)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols:
        s = df[c].astype(str).replace({"nan": np.nan}).dropna()
        if len(s) == 0:
            continue
        vc = s.value_counts().head(max_categories)
        profile["categorical_summary"][c] = [{"value": k, "count": int(v)} for k, v in vc.items()]

    # sample rows (sanitized)
    try:
        profile["sample_rows"] = df.head(sample_rows).fillna("").to_dict(orient="records")
    except Exception:
        profile["sample_rows"] = []

    return profile

def pick_group_column(df: pd.DataFrame) -> str | None:
    """
    Pick a categorical column suitable for group forecasting (Region/Category/etc.).
    """
    preferred = ["region", "category", "segment", "product", "store", "country", "city"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return None

    # prefer by name
    for p in preferred:
        for c in cat_cols:
            if p in c.lower():
                return c

    # fallback: reasonable cardinality
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 30:
            return c

    return None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts CSV, auto-detects columns, runs forecast, RCA, anomalies, and saves to DB.
    """
    try:
        contents = await file.read()

        # Robust CSV decoding (handles Windows/Excel CSVs with bytes like 0xA0)
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="cp1252")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")

        dataset_profile = build_dataset_profile(df)

        # --- Step 1: detect date column (with 1980 guard) ---
        try:
            original_date_col = detect_date_column(df)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # --- Step 2: detect numeric target column (highest variance) ---
        try:
            original_target_col = detect_target_column(df, original_date_col)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # --- Step 3: detect additional feature columns (numeric only) ---
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [
            c for c in num_cols
            if c != original_date_col and c != original_target_col
            and df[c].isna().sum() / len(df) <= 0.30
        ]

        # --- Step 4: build working frame (date + sales + optional numeric features) ---
        use_cols = [original_date_col, original_target_col] + feature_cols
        df_work = df[use_cols].copy()

        df_work.rename(columns={original_date_col: "date", original_target_col: "sales"}, inplace=True)

        # Parse date (remove deprecated infer_datetime_format)
        df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce", utc=False)
        df_work = df_work.dropna(subset=["date"])

        # Coerce sales to numeric (handles commas and NBSP)
        df_work["sales"] = pd.to_numeric(
            df_work["sales"].astype(str).str.replace(",", "").str.replace("\xa0", " ").str.strip(),
            errors="coerce",
        )

        # Coerce feature cols too (safe)
        for c in feature_cols:
            if c in df_work.columns:
                df_work[c] = pd.to_numeric(
                    df_work[c].astype(str).str.replace(",", "").str.replace("\xa0", " ").str.strip(),
                    errors="coerce",
                )

        df_work = df_work.dropna(subset=["sales"])

        # Aggregate duplicate dates (mean)
        agg_map = {"sales": "mean"}
        for c in feature_cols:
            if c in df_work.columns:
                agg_map[c] = "mean"

        df_work = df_work.groupby("date", as_index=False).agg(agg_map)
        df_work = df_work.sort_values("date").reset_index(drop=True)

        # Fill missing
        df_work["sales"] = df_work["sales"].ffill().bfill()
        for c in feature_cols:
            if c in df_work.columns:
                df_work[c] = df_work[c].ffill().bfill()

        if len(df_work) < 20:
            raise HTTPException(status_code=400, detail="Dataset too small — need at least 20 rows after cleaning.")

        session_id = str(uuid.uuid4())
        save_upload(session_id, file.filename, len(df_work), df_work.columns.tolist())

        # --- Step 5: run forecast pipeline ---
        forecast_results = run_forecast(df_work)

        # Strip non-serialisable model objects, capture for RCA
        model = forecast_results.pop("model", None)
        X = forecast_results.pop("X", None)

        model_mape = forecast_results.get("model_mape", 10.0)
        baseline_mape = forecast_results.get("baseline_mape", 20.0)
        truth_score = forecast_results.get("truth_score", 0.0)

        # --- Step 6: SHAP / RCA ---
        shap_results = []
        rca_explanation = "No explanation provided"
        if model is not None and X is not None:
            shap_results = compute_shap(model, X, X)
            rca_explanation = explain_rca(shap_results)

        # --- Step 7: anomaly detection ---
        anomalies = detect_anomalies(df_work, forecast_results)

        # --- Step 8: truth meter ---
        truth_meter = compute_truth_meter(model_mape, baseline_mape)
        forecast_results["truth_score"] = truth_meter.get("score", truth_score)

        # --- Step 8.5: lightweight group forecasts (top 5 groups) ---
        group_forecasts = None
        group_col = pick_group_column(df)

        if group_col:
            try:
                df_tmp = df[[original_date_col, original_target_col, group_col]].copy()
                df_tmp[original_date_col] = pd.to_datetime(df_tmp[original_date_col], errors="coerce", utc=False)
                df_tmp[original_target_col] = pd.to_numeric(
                    df_tmp[original_target_col].astype(str).str.replace(",", "").str.replace("\xa0", " ").str.strip(),
                    errors="coerce",
                )
                df_tmp = df_tmp.dropna(subset=[original_date_col, original_target_col, group_col])

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
                    gdf = df_tmp[df_tmp[group_col] == g][[original_date_col, original_target_col]].copy()
                    gdf.rename(columns={original_date_col: "date", original_target_col: "sales"}, inplace=True)
                    gdf = gdf.dropna(subset=["date", "sales"])
                    gdf = gdf.groupby("date")["sales"].mean().reset_index().sort_values("date").reset_index(drop=True)
                    gdf["sales"] = gdf["sales"].ffill().bfill()

                    if len(gdf) < 20:
                        continue

                    gres = run_forecast(gdf)
                    last_hist = float(gres["historical"][-1]) if gres.get("historical") else 0.0
                    last_fc = float(gres["forecast"][-1]) if gres.get("forecast") else 0.0
                    pct = ((last_fc - last_hist) / abs(last_hist) * 100.0) if last_hist != 0 else 0.0

                    group_forecasts.append(
                        {
                            "group_col": group_col,
                            "group": str(g),
                            "forecast_horizon": gres.get("forecast_horizon"),
                            "expected_change_percent": round(pct, 2),
                            "last_hist": last_hist,
                            "last_forecast": last_fc,
                        }
                    )

                group_forecasts.sort(key=lambda x: x["expected_change_percent"], reverse=True)
            except Exception as e:
                logger.warning(f"Group forecasts skipped due to error: {e}")
                group_forecasts = None

        # --- Step 9: assemble full response ---
        forecast_results["truth_meter"] = truth_meter
        forecast_results["session_id"] = session_id
        forecast_results["anomalies"] = anomalies
        forecast_results["shap_results"] = shap_results
        forecast_results["rca_explanation"] = rca_explanation
        forecast_results["detected_columns"] = {
            "date": original_date_col,
            "target": original_target_col,
            "features": feature_cols,
        }
        forecast_results["dataset_profile"] = dataset_profile
        forecast_results["group_forecasts"] = group_forecasts

        sanitized_results = _sanitize_for_json(forecast_results)

        # IMPORTANT: save forecast using the same truth_score the UI uses
        save_forecast(session_id, sanitized_results, sanitized_results.get("truth_score", truth_score))
        save_anomalies(session_id, sanitized_results["anomalies"])

        return sanitized_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/forecast/{session_id}")
async def get_forecast_data(session_id: str):
    """Load forecast from DB by session_id."""
    try:
        data = get_forecast(session_id)
        if not data:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return data
    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/anomalies/{session_id}")
async def get_anomalies_data(session_id: str):
    """Load anomalies from DB by session_id."""
    try:
        data = get_anomalies(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Anomalies not found")
        return data
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.post("/chat")
async def chat_interaction(req: ChatRequest):
    """Interact with agent using session context."""
    try:
        history_from_db = get_chat_history(req.session_id)
        context = req.session_context or []

        if len(context) == 0 and history_from_db:
            for row in history_from_db[-5:]:
                context.append({"role": "user", "content": row["user_message"]})
                context.append({"role": "model", "content": row["agent_response"]})

        forecast_data = get_forecast(req.session_id) or {}
        anomalies_data = get_anomalies(req.session_id) or []

        dataset_profile = forecast_data.get("dataset_profile", {})
        group_forecasts = forecast_data.get("group_forecasts", None)

        # --- Build rich, accurate system prompt ---
        historical = forecast_data.get("historical", [])
        hist_dates = forecast_data.get("historical_dates", [])
        future_dates = forecast_data.get("dates", [])
        truth_score = forecast_data.get("truth_score", 0.0)
        truth_meter = forecast_data.get("truth_meter", {})
        rca = forecast_data.get("rca_explanation", "Not available.")
        detected_freq = forecast_data.get("detected_freq", "unknown")
        forecast_horizon = forecast_data.get("forecast_horizon", len(future_dates))
        confidence_level = forecast_data.get("confidence_level", 90)
        data_quality = forecast_data.get("data_quality", {})
        detected_cols = forecast_data.get("detected_columns", {})
        target_col = detected_cols.get("target", "the target metric")
        total_rows = data_quality.get("total_rows", len(historical))
        dq_warning = data_quality.get("warning", "")

        # Actual date boundaries
        start_date = hist_dates[0] if hist_dates else "unknown"
        end_date = hist_dates[-1] if hist_dates else "unknown"
        forecast_start = future_dates[0] if future_dates else "unknown"
        forecast_end = future_dates[-1] if future_dates else "unknown"

        # Most severe anomaly
        most_severe = None
        if anomalies_data:
            most_severe = max(anomalies_data, key=lambda a: a.get("deviation_percent", 0))

        if anomalies_data:
            anom_lines = []
            for a in anomalies_data[:10]:
                anom_lines.append(
                    f"  {a.get('date','?')}: actual={a.get('actual',0):.2f}, "
                    f"expected={a.get('expected',0):.2f}, "
                    f"{a.get('severity','?')} severity {a.get('direction','?')}"
                )
            anomaly_summary = f"{len(anomalies_data)} anomaly/anomalies detected:\n" + "\n".join(anom_lines)
            if most_severe:
                anomaly_summary += (
                    f"\nMost severe: {most_severe.get('date')} "
                    f"({most_severe.get('deviation_percent',0):.1f}% deviation, "
                    f"{most_severe.get('direction','?')})"
                )
        else:
            anomaly_summary = "No anomalies detected."

        warning_line = f"\n⚠️ Data quality note: {dq_warning}" if dq_warning else ""

        # Keep prompt lightweight: include only small summaries
        cols_list = dataset_profile.get("columns", [])
        cat_summary = dataset_profile.get("categorical_summary", {})
        num_summary = dataset_profile.get("numeric_summary", {})

        group_hint = ""
        if group_forecasts:
            group_col = group_forecasts[0].get("group_col", "group")
            top_lines = []
            for gf in group_forecasts[:5]:
                top_lines.append(f"- {gf.get('group')}: {gf.get('expected_change_percent')}% expected change")
            group_hint = (
                f"\nGroup forecast available by `{group_col}` (top growth):\n" + "\n".join(top_lines)
            )

        system_msg = f"""You are FutureLens, an AI forecasting assistant.

You must:
- Handle typos and minor misspellings.
- Answer concisely for non-experts (2���6 bullets).
- Be transparent about uncertainty and limitations.

Dataset context:
- Columns available: {cols_list}
- Historical data: {start_date} to {end_date}
- Data frequency: {detected_freq}
- Total rows: {total_rows}
- Forecasting: {target_col} column

Forecast generated:
- Forecast period: {forecast_start} to {forecast_end}
- Horizon: {forecast_horizon} {detected_freq} periods ahead
- Confidence level: {confidence_level}%
- Truth meter: {truth_meter.get("message", f"{truth_score:.1f}% vs baseline")}
- Root Cause Analysis: {rca}
{warning_line}

Categorical columns (top values): {list(cat_summary.keys())}
Numeric columns: {list(num_summary.keys())}
{group_hint}

{anomaly_summary}

If user asks "which region/category will expand", use group forecast if present.
If no group forecast is available, explain that grouping requires a categorical column and suggest which columns could be used.
"""

        response = chat(req.message, context, system_instruction=system_msg)
        save_chat(req.session_id, req.message, response)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """Simulates a future scenario by adjusting base forecast."""
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
            scenario_type=req.scenario_type,
            historical=historical,
        )

        baseline_vals = sim_result.get("baseline", [])
        scenario_vals = sim_result.get("scenario", [])
        summary = sim_result.get("summary", "")

        sum_base = sum(baseline_vals)
        sum_scen = sum(scenario_vals)
        diff_pct = 0.0
        if sum_base != 0:
            diff_pct = ((sum_scen - sum_base) / sum_base) * 100

        res = {
            "baseline": baseline_vals,
            "scenario": scenario_vals,
            "difference_percent": diff_pct,
            "summary": summary,
            "dates": forecast_data.get("dates", []),
        }
        return _sanitize_for_json(res)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simulate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/history")
async def get_history():
    """Return last 10 uploads from DB."""
    try:
        return get_recent_uploads(10)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
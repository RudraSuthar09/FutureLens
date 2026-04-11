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
from api.forecaster import run_forecast
from api.rca import compute_shap, explain_rca
from api.anomaly import detect_anomalies
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

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types and datetimes """
    def default(self, obj):
        import datetime
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Check for NaN and return None (which serializes to null in JSON)
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts CSV, cleans it, runs forecast, RCA, anomalies, and saves to DB.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # --- Smart column detection ---
        # 1. Find date column (case-insensitive, tries to parse as datetime)
        date_col = None
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() >= 0.8 * len(df):
                    date_col = col
                    df[col] = parsed
                    break
            except Exception:
                continue
        if date_col is None:
            raise HTTPException(status_code=400, detail="No date column found in CSV.")

        # 2. Find numeric target column
        # Prefer known names; fall back to first remaining numeric column
        preferred_names = ['AveragePrice', 'sales', 'price', 'value', 'revenue', 'amount']
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        target_col = None
        for name in preferred_names:
            if name in df.columns:
                target_col = name
                break
        if target_col is None:
            if num_cols:
                target_col = num_cols[0]
            else:
                raise HTTPException(status_code=400, detail="No numeric target column found in CSV.")

        # 3. Keep only date + target, group by date (handles multi-row-per-date CSVs)
        df = df[[date_col, target_col]].copy()
        df.rename(columns={date_col: 'date', target_col: 'sales'}, inplace=True)
        df = df.groupby('date')['sales'].mean().reset_index()
        df = df.sort_values('date').reset_index(drop=True)

        # 4. Fill gaps
        df['sales'] = df['sales'].ffill().bfill()

        session_id = str(uuid.uuid4())
        row_count = len(df)
        column_names = df.columns.tolist()
        
        save_upload(session_id, file.filename, row_count, column_names)
        
        # Call run_forecast()
        forecast_results = run_forecast(df)
        
        # Extract model & X for RCA and truth score for saving
        model = forecast_results.pop("model", None)
        X = forecast_results.pop("X", None)
        truth_score = forecast_results.get("truth_score", 0.0)
        
        # Call compute_shap()
        shap_results = []
        rca_explanation = "No explanation provided"
        if model is not None and X is not None:
            shap_results = compute_shap(model, X, X)
            rca_explanation = explain_rca(shap_results)
            
        # Call detect_anomalies()
        anomalies = detect_anomalies(df, forecast_results)
        
        # Merge all into forecast_results so frontend doesn't break
        forecast_results["session_id"] = session_id
        forecast_results["anomalies"] = anomalies
        forecast_results["shap_results"] = shap_results
        forecast_results["rca_explanation"] = rca_explanation
        
        # Sanitize numpy arrays/floats before database saving and FastAPI response
        sanitized_results = _sanitize_for_json(forecast_results)
        
        save_forecast(session_id, sanitized_results, truth_score)
        save_anomalies(session_id, sanitized_results["anomalies"])
        
        return sanitized_results
        
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
                context.append({"role": "user", "content": row['user_message']})
                context.append({"role": "model", "content": row['agent_response']})
            
        forecast_data = get_forecast(req.session_id) or {}
        anomalies_data = get_anomalies(req.session_id) or []

        # Build a rich system message so Gemini has full context about the loaded data
        historical = forecast_data.get("historical", [])
        hist_dates = forecast_data.get("historical_dates", [])
        future_dates = forecast_data.get("dates", [])
        forecast_vals = forecast_data.get("forecast", [])
        truth_score = forecast_data.get("truth_score", 0.0)
        rca = forecast_data.get("rca_explanation", "Not available.")

        # Summarise anomalies as human-readable text for the LLM
        if anomalies_data:
            anom_lines = []
            for a in anomalies_data[:10]:
                anom_lines.append(
                    f"{a.get('date','?')}: actual={a.get('actual','?'):.2f}, "
                    f"expected={a.get('expected','?'):.2f}, severity={a.get('severity','?')}"
                )
            anomaly_summary = f"{len(anomalies_data)} anomaly/anomalies detected:\n" + "\n".join(anom_lines)
        else:
            anomaly_summary = "No anomalies detected."

        data_summary = (
            f"Dataset covers {len(historical)} historical data points "
            f"from {hist_dates[0] if hist_dates else 'unknown'} to {hist_dates[-1] if hist_dates else 'unknown'}. "
            f"Forecasting {len(future_dates)} periods ahead "
            f"({future_dates[0] if future_dates else '?'} to {future_dates[-1] if future_dates else '?'}). "
            f"Model Truth Score: {truth_score:.1f}% better than naive baseline. "
            f"Root Cause Analysis: {rca}. "
            f"{anomaly_summary}"
        )

        system_msg = (
            "You are FutureLens, an expert AI forecasting assistant. "
            "You have already analyzed the user's uploaded dataset. "
            "You MUST use the data provided below to answer questions — "
            "do NOT ask the user to provide data you already have. "
            f"{data_summary}"
        )
        
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
            
        sim_result = simulate_scenario(base_forecast, req.change_percent)
        
        baseline_vals = sim_result.get("baseline", [])
        scenario_vals = sim_result.get("scenario", [])
        
        sum_base = sum(baseline_vals)
        sum_scen = sum(scenario_vals)
        diff_pct = 0.0
        if sum_base != 0:
            diff_pct = ((sum_scen - sum_base) / sum_base) * 100
            
        res = {
            "baseline": baseline_vals,
            "scenario": scenario_vals,
            "difference_percent": diff_pct
        }
        return _sanitize_for_json(res)
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

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
        
        # CRITICAL FIX 2 Exact steps
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.groupby('Date')['AveragePrice'].mean().reset_index()
        
        # Map back to standardized internal schema
        df.rename(columns={'Date': 'date', 'AveragePrice': 'sales'}, inplace=True)

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        
        system_msg = f"You are an AI forecasting assistant analyzing the user's uploaded data. The data has a baseline model and a LightGBM model. RCA Explanation: {forecast_data.get('rca_explanation', 'None')}. Detected Anomalies: {json.dumps(anomalies_data)}."
        
        response = chat(req.message, context, system_instruction=system_msg)
        save_chat(req.session_id, req.message, response)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Return last 10 uploads from DB."""
    try:
        return get_recent_uploads(10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import io
import uuid
import pandas as pd
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.database import (
    init_db, save_upload, save_forecast, save_anomalies, save_chat,
    get_forecast, get_anomalies, get_chat_history, get_recent_uploads
)
from api.forecaster import run_forecast
from api.rca import compute_shap
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
    session_id: str

class SimulateRequest(BaseModel):
    session_id: str
    change_percent: float

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts CSV, cleans it, runs forecast, RCA, anomalies, and saves to DB.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Auto-detect date column
        date_col = None
        for col in df.columns:
            # Try to convert to datetime to see if it works
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > (0.5 * len(df)): # if more than 50% are valid dates
                date_col = col
                df[col] = parsed
                break
                
        if date_col and date_col.lower() != 'date':
            df.rename(columns={date_col: 'date'}, inplace=True)
            
        # Clean nulls (forward fill)
        df.ffill(inplace=True)
        # fallback to backfill if first row was null
        df.bfill(inplace=True)
        
        # We need a 'sales' col for forecaster, check if it exists, else assume 2nd num col
        if 'sales' not in df.columns:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                df.rename(columns={num_cols[0]: 'sales'}, inplace=True)

        session_id = str(uuid.uuid4())
        row_count = len(df)
        column_names = df.columns.tolist()
        
        # Save upload to DB
        save_upload(session_id, file.filename, row_count, column_names)
        
        # Call run_forecast()
        forecast_results = run_forecast(df)
        
        # Extract model & X for RCA and truth score for saving
        model = forecast_results.get("model", None)
        X = forecast_results.get("X", None)
        truth_score = forecast_results.get("truth_score", 0.0)
        
        # Call compute_shap()
        shap_results = []
        if model is not None and X is not None:
            shap_results = compute_shap(model, X, X)
            
        # Call detect_anomalies()
        anomalies = detect_anomalies(df, forecast_results)
        
        # Save forecast and anomalies to DB
        save_forecast(session_id, forecast_results, truth_score)
        save_anomalies(session_id, anomalies)
        
        return {
            "session_id": session_id,
            "forecast": forecast_results.get("forecast", []),
            "anomalies": anomalies,
            "truth_score": truth_score,
            "shap_results": shap_results
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
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
        # Load session context from DB
        history = get_chat_history(req.session_id)
        
        # Format history for the agent (List[Dict[str, str]]) if agent supports it
        # Based on agent.py, chat(message, session_context)
        context = []
        for row in history[-5:]:  # Send last 5 exchanges to limit context window
            context.append({"role": "user", "content": row['user_message']})
            context.append({"role": "model", "content": row['agent_response']})
            
        # Call agent.chat
        response = chat(req.message, context)
        
        # Save chat message to DB
        save_chat(req.session_id, req.message, response)
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """Simulates a future scenario by adjusting base forecast."""
    try:
        # Load base forecast from DB
        forecast_data = get_forecast(req.session_id)
        if not forecast_data:
            raise HTTPException(status_code=404, detail="Forecast not found")
            
        base_forecast = forecast_data.get("forecast", [])
        if not base_forecast:
            raise HTTPException(status_code=400, detail="No base forecast available")
            
        # Call simulate_scenario
        sim_result = simulate_scenario(base_forecast, req.change_percent)
        
        baseline_vals = sim_result.get("baseline", [])
        scenario_vals = sim_result.get("scenario", [])
        
        # Calculate difference percent dynamically
        sum_base = sum(baseline_vals)
        sum_scen = sum(scenario_vals)
        diff_pct = 0.0
        if sum_base != 0:
            diff_pct = ((sum_scen - sum_base) / sum_base) * 100
            
        return {
            "baseline": baseline_vals,
            "scenario": scenario_vals,
            "difference_percent": diff_pct
        }
    except Exception as e:
        logger.error(f"Error in simulate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/history")
async def get_history():
    """Return last 10 uploads from DB."""
    try:
        uploads = get_recent_uploads(10)
        return uploads
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

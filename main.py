import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any

from api.forecaster import run_forecast
from api.rca import compute_shap, explain_rca
from api.agent import chat

app = FastAPI(title="FutureLens API")

class ChatRequest(BaseModel):
    message: str
    session_context: List[Dict[str, str]] = []

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Endpoint to handle data upload, run forecast modeling, detect anomalies, 
    and perform RCA.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # 1. Run Forecast
    forecast_results = run_forecast(df, periods=8)
    
    # Extract model and data for RCA (removed from the response payload manually)
    model = forecast_results.pop("model", None)
    X = forecast_results.pop("X", None)
    
    # 2. Run Root Cause Analysis (SHAP)
    shap_results = []
    rca_explanation = "No model available for RCA."
    
    if model is not None and X is not None:
        shap_results = compute_shap(model, X_train=X, X_test=X)
        rca_explanation = explain_rca(shap_results)
    
    # 3. Detect Anomalies
    # Simple rolling standard deviation approach
    sales = df['sales'].values
    dates = df['date'].values
    
    window = 10
    anomalies = []
    
    for i in range(window, len(sales)):
        history = sales[i-window:i]
        mean = np.mean(history)
        std = np.std(history)
        
        # Flag as anomaly if more than 3 standard deviations away
        if std > 0 and abs(sales[i] - mean) > 3 * std:
            # We specifically target drops for our rules
            direction = 'drop' if sales[i] < mean else 'spike'
            anomalies.append({
                'date': dates[i],
                'value': float(sales[i]),
                'expected': float(mean),
                'direction': direction
            })
            
    # Include RCA and Anomalies with the forecast dictionary
    forecast_results['shap_results'] = shap_results
    forecast_results['rca_explanation'] = rca_explanation
    forecast_results['anomalies'] = anomalies
    
    return forecast_results

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Endpoint to interact with the Gemini Agent.
    """
    response_text = chat(req.message, req.session_context)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

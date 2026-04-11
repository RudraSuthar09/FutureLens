import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Tuple

def detect_anomalies(df: pd.DataFrame, forecast_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects anomalies in the historical sales data using IsolationForest.
    
    Args:
        df: DataFrame containing historical data including 'date' and 'sales'.
        forecast_dict: Optional forecast dictionary (may contain baseline/expected if available).
        
    Returns:
        List of dictionaries with anomaly details.
    """
    try:
        # Prepare data
        sales = df['sales'].values.reshape(-1, 1)
        dates = df['date'].tolist()
        
        # Fit IsolationForest
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(sales) # -1 = anomaly, 1 = normal
        
        anomalies = []
        
        # Calculate a simple expected value (rolling mean) for historical context
        # if the forecast_dict does not provide historical expectations.
        rolling_mean = df['sales'].rolling(window=4, min_periods=1).mean().tolist()
        
        for i, label in enumerate(labels):
            if label == -1:
                actual = float(sales[i][0])
                # Provide a fallback expected value
                expected = float(rolling_mean[i])
                
                # Protect against division by zero
                expected_denom = expected if expected > 0 else 1.0
                deviation_diff = abs(actual - expected)
                deviation_percent = deviation_diff / expected_denom
                
                if deviation_percent > 0.3:
                    severity = "high"
                elif deviation_percent > 0.15:
                    severity = "medium"
                else:
                    severity = "low"
                    
                anomalies.append({
                    "date": dates[i],
                    "actual": actual,
                    "expected": expected,
                    "severity": severity,
                    "deviation_percent": deviation_percent * 100
                })
                
        return anomalies
    except Exception as e:
        import logging
        logging.error(f"Error detecting anomalies: {e}")
        return []

def compute_truth_meter(model_mape: float, baseline_mape: float) -> Dict[str, Any]:
    """
    Compares model MAPE to baseline MAPE to provide a confidence score.
    """
    # model_mape is more than 10% better meaning it's 10 percentage points lower?
    # Or 10% relatively lower? Let's use relative difference.
    # relative_improvement = (baseline_mape - model_mape) / baseline_mape
    # Wait, instructions: "If model_mape is more than 10% better than baseline"
    # Usually meaning baseline_mape - model_mape > 10 (percentage points) 
    # OR model_mape <= 0.9 * baseline_mape. Let's assume standard relative ratio.
    try:
        if baseline_mape == 0:
            return {"reliable": False, "color": "red", "message": "Baseline MAPE is 0 - invalid test"}
            
        improvement = (baseline_mape - model_mape) / baseline_mape
        
        if improvement > 0.10:
            return {
                "reliable": True,
                "color": "green",
                "message": f"High confidence — MAPE {model_mape:.1f}%"
            }
        else:
            diff = max(0.0, improvement * 100)
            return {
                "reliable": False,
                "color": "red",
                "message": f"Model only {diff:.1f}% better — low trust"
            }
    except Exception as e:
        import logging
        logging.error(f"Error in truth meter: {e}")
        return {"reliable": False, "color": "red", "message": "Error calculating truth meter."}

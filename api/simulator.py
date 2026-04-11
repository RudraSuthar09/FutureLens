import pandas as pd
import numpy as np
from typing import List, Dict, Any

from api.forecaster import run_forecast

def simulate_scenario(base_forecast: List[float], change_percent: float, scenario_type: str = "growth") -> Dict[str, Any]:
    """
    Applies a percentage change across a forecast, adding realistic noise.
    
    Args:
        base_forecast: List of future forecasted values.
        change_percent: The percentage to change (e.g., 10 for +10%).
        scenario_type: Type of scenario (growth, decline, etc).
        
    Returns:
        Dict with baseline, scenario, and summary details.
    """
    try:
        import random
        
        scenario_values = []
        for val in base_forecast:
            # apply change
            adjusted = val * (1 + (change_percent / 100.0))
            
            # Add realistic noise of +/- 2%
            noise_factor = random.uniform(-0.02, 0.02)
            adjusted += adjusted * noise_factor
            
            scenario_values.append(adjusted)
            
        baseline_sum = sum(base_forecast)
        scenario_sum = sum(scenario_values)
        
        # summary string
        sign = "+" if change_percent >= 0 else ""
        summary = (f"Under {sign}{change_percent}% scenario, "
                   f"forecast reaches {scenario_sum:.0f} (vs {baseline_sum:.0f} baseline). "
                   f"Range: {min(scenario_values):.0f}–{max(scenario_values):.0f}.")
        
        # In a real setup, we would return dates too, but we might not have them here
        return {
            "baseline": base_forecast,
            "scenario": scenario_values,
            "summary": summary
        }
    except Exception as e:
        import logging
        logging.error(f"Error simulating scenario: {e}")
        return {"baseline": base_forecast, "scenario": base_forecast, "summary": "Error"}

def simulate_remove_outliers(df: pd.DataFrame, forecast_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes top 5% outlier weeks from the training dataset and reruns the forecast.
    
    Args:
        df: Original historical data.
        forecast_dict: existing forecast dictionary (to determine periods etc).
        
    Returns:
        A comparison dictionary.
    """
    try:
        data = df.copy()
        # Find 95th percentile
        threshold = data['sales'].quantile(0.95)
        
        # Remove top 5% outliers
        clean_data = data[data['sales'] <= threshold].copy()
        
        # Re-run simplified forecast
        # Try to extract periods from existing forecast if possible, else default to 8
        periods = len(forecast_dict.get('forecast', [])) if 'forecast' in forecast_dict else 8
        
        new_forecast = run_forecast(clean_data, periods=periods)
        
        return {
            "original_forecast": forecast_dict.get('forecast', []),
            "clean_forecast": new_forecast.get('forecast', []),
            "summary": "Removed top 5% outliers and regenerated forecast."
        }
    except Exception as e:
        import logging
        logging.error(f"Error simulating outliers removal: {e}")
        return {}

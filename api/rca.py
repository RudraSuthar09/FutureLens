import shap
import pandas as pd
import numpy as np
from typing import Any, List, Dict

def compute_shap(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes SHAP values using TreeExplainer on a LightGBM model, 
    and returns the top 5 features.
    
    Args:
        model: Trained LightGBM model.
        X_train: Training data features.
        X_test: Testing data features.
        
    Returns:
        List of dictionaries with feature, importance, and direction.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Determine direction (correlation of feature value with SHAP value)
        directions = []
        for i, col in enumerate(X_test.columns):
            correlation = np.corrcoef(X_test[col], shap_values[:, i])[0, 1]
            if pd.isna(correlation):
                directions.append("neutral")
            else:
                directions.append("positive" if correlation > 0 else "negative")
                
        # Combine into results
        results = []
        for col, importance, direction in zip(X_test.columns, mean_abs_shap, directions):
            results.append({
                "feature": col,
                "importance": float(importance),
                "direction": direction
            })
            
        # Sort and get top 5
        results.sort(key=lambda x: x["importance"], reverse=True)
        top_5 = results[:5]
        
        # Normalize importance to percentages for easier reporting
        total_importance = sum(x["importance"] for x in results)
        if total_importance > 0:
            for item in top_5:
                item["contribution_percent"] = (item["importance"] / total_importance) * 100
        else:
            for item in top_5:
                item["contribution_percent"] = 0.0
                
        return top_5
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return []

def explain_rca(shap_results: List[Dict[str, Any]]) -> str:
    """
    Converts SHAP results into a plain English explanation.
    
    Args:
        shap_results: List of dicts representing top feature contributions.
        
    Returns:
        str: English text explanation of root cause.
    """
    if not shap_results:
        return "Not enough data to determine root cause."
        
    top_feature = shap_results[0]
    feature_name = top_feature.get("feature", "unknown_factor")
    perc = top_feature.get("contribution_percent", 0.0)
    direction = top_feature.get("direction", "neutral")
    
    direction_text = "reduced" if direction == "positive" else "increased"
    
    # A generic sentence mimicking the example provided by the prompt.
    # e.g., "Sales drop mainly driven by reduced ad_spend (62% contribution)"
    return f"Variation mainly driven by {direction_text} {feature_name} ({perc:.0f}% contribution)."

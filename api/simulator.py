import pandas as pd
import numpy as np
from typing import List, Dict, Any


def simulate_scenario(
    base_forecast: List[float],
    change_percent: float,
    scenario_type: str = "growth",
    historical: List[float] = None,
) -> Dict[str, Any]:
    """
    Applies a scenario transformation to the base forecast.

    Scenario types:
    - "growth"         : multiply every point by (1 + change_percent/100)
    - "flat"           : replace all points with mean of last 4 historical values
    - "recent_trend"   : extrapolate the last 4-week slope of history forward
    - "remove_outliers": cap outlier periods at mean ± 1.5 * std of the forecast

    Args:
        base_forecast  : List of baseline future forecasted values.
        change_percent : Percentage adjustment (used by growth scenario).
        scenario_type  : One of the four named scenario types above.
        historical     : Full historical sales list (used by flat/trend scenarios).

    Returns:
        Dict with baseline, scenario, and a human-readable summary string.
    """
    try:
        hist = historical or []

        if scenario_type == "flat":
            # Replace forecast with the mean of the last 4 historical points
            anchor = float(np.mean(hist[-4:])) if len(hist) >= 4 else (float(np.mean(hist)) if hist else float(np.mean(base_forecast)))
            scenario_values = [anchor] * len(base_forecast)
            sign = ""
            summary = (
                f"Under a flat trend scenario, {anchor:.2f} is maintained "
                f"throughout the forecast period "
                f"(vs baseline range {min(base_forecast):.2f}–{max(base_forecast):.2f})."
            )

        elif scenario_type == "recent_trend":
            # Extrapolate the slope of the last 4 historical points
            if len(hist) >= 4:
                recent = hist[-4:]
                slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
            else:
                slope = 0.0
            scenario_values = []
            last_hist_val = float(hist[-1]) if hist else (float(base_forecast[0]) if base_forecast else 0.0)
            for step in range(len(base_forecast)):
                scenario_values.append(last_hist_val + slope * (step + 1))
            summary = (
                f"Under a recent-trend scenario (slope: {slope:+.2f}/period), "
                f"the forecast reaches {scenario_values[-1]:.2f} "
                f"(vs baseline {base_forecast[-1]:.2f})."
            )

        elif scenario_type == "remove_outliers":
            # Cap outlier periods at mean ± 1.5 * std
            fc_arr = np.array(base_forecast)
            mean_fc = float(np.mean(fc_arr))
            std_fc = float(np.std(fc_arr)) if len(fc_arr) > 1 else 0.0
            lo = mean_fc - 1.5 * std_fc
            hi = mean_fc + 1.5 * std_fc
            scenario_values = [float(np.clip(v, lo, hi)) for v in base_forecast]
            capped = sum(1 for a, b in zip(base_forecast, scenario_values) if a != b)
            summary = (
                f"Outliers removed: {capped} period(s) capped to "
                f"[{lo:.2f}, {hi:.2f}]. "
                f"Forecast now ranges {min(scenario_values):.2f}–{max(scenario_values):.2f}."
            )

        else:
            # Default: "growth" — simple percentage adjustment
            factor = 1 + (change_percent / 100.0)
            scenario_values = [float(v * factor) for v in base_forecast]
            sign = "+" if change_percent >= 0 else ""
            summary = (
                f"Under a {sign}{change_percent:.0f}% growth scenario, "
                f"forecast reaches {scenario_values[-1]:.2f} "
                f"(vs baseline {base_forecast[-1]:.2f}). "
                f"Range: {min(scenario_values):.2f}–{max(scenario_values):.2f}."
            )

        return {
            "baseline": base_forecast,
            "scenario": scenario_values,
            "summary": summary,
        }
    except Exception as e:
        import logging

        logging.error(f"Error simulating scenario: {e}")
        return {"baseline": base_forecast, "scenario": base_forecast, "summary": "Error during simulation."}

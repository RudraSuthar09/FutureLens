# api/tools.py
import logging
from api.simulator import simulate_scenario

logger = logging.getLogger(__name__)


def forecast_tool(card: dict, args: dict) -> dict:
    """Return next N periods from the pre-built forecast."""
    periods = int(args.get("periods", card.get("horizon", 4)))
    fc = card.get("forecast_values", [])
    lo = card.get("lower_values", [])
    hi = card.get("upper_values", [])
    dates = card.get("forecast_dates", [])
    target = card.get("target_col", "target")
    freq = card.get("freq_label", "period")

    periods = min(periods, len(fc))
    if not fc:
        return {"error": "No forecast data available."}

    period_data = []
    for i in range(periods):
        period_data.append({
            "period": i + 1,
            "date": dates[i] if i < len(dates) else f"Period {i+1}",
            "forecast": round(fc[i], 2),
            "lower": round(lo[i], 2) if i < len(lo) else None,
            "upper": round(hi[i], 2) if i < len(hi) else None,
        })

    last = card.get("last_historical_value", 0)
    overall_change = ((fc[periods-1] - last) / abs(last) * 100) if last else 0

    return {
        "tool": "forecast",
        "target": target,
        "periods": periods,
        "freq": freq,
        "overall_change_pct": round(overall_change, 1),
        "trend": card.get("trend_direction", "stable"),
        "period_data": period_data,
        "reliability": card.get("reliability_plain", ""),
        "one_liner": card.get("one_liner", ""),
    }


def scenario_tool(card: dict, forecast_data: dict, args: dict) -> dict:
    """Run a scenario simulation and return side-by-side comparison."""
    change_pct = float(args.get("change_percent", 10))
    scenario_type = args.get("scenario_type", "growth")
    target_col = args.get("target_col", card.get("target_col", "target"))

    base_forecast = forecast_data.get("forecast", [])
    historical = forecast_data.get("historical", [])

    if not base_forecast:
        return {"error": "No base forecast available for simulation."}

    sim = simulate_scenario(
        base_forecast,
        change_pct,
        scenario_type=scenario_type,
        historical=historical,
    )

    baseline_vals = sim.get("baseline", [])
    scenario_vals = sim.get("scenario", [])
    dates = forecast_data.get("dates", [])

    sum_base = sum(baseline_vals)
    sum_scen = sum(scenario_vals)
    diff_pct = ((sum_scen - sum_base) / sum_base * 100) if sum_base else 0

    # Cross-column impact if relevant column was mentioned
    cross_col_note = None
    corr_data = card.get("column_relationships", {}).get("correlations_with_target", {})
    if target_col != card.get("target_col") and target_col in corr_data:
        corr = corr_data[target_col]
        estimated_impact = corr * change_pct
        cross_col_note = {
            "column": target_col,
            "correlation": corr,
            "estimated_target_change_pct": round(estimated_impact, 1),
            "note": "Correlation-based estimate only — not causation."
        }

    return {
        "tool": "scenario",
        "scenario_type": scenario_type,
        "change_percent": change_pct,
        "target": card.get("target_col"),
        "baseline_total": round(sum_base, 2),
        "scenario_total": round(sum_scen, 2),
        "difference_percent": round(diff_pct, 1),
        "baseline_values": [round(v, 2) for v in baseline_vals],
        "scenario_values": [round(v, 2) for v in scenario_vals],
        "dates": dates,
        "cross_column_impact": cross_col_note,
        "summary": sim.get("summary", ""),
    }


def anomaly_tool(card: dict, anomalies_data: list, args: dict) -> dict:
    """Return structured anomaly analysis."""
    if not anomalies_data:
        return {
            "tool": "anomaly",
            "count": 0,
            "message": "No anomalies detected in this dataset.",
            "anomaly_plain": card.get("anomaly_plain", "No unusual patterns.")
        }

    high = [a for a in anomalies_data if a.get("severity") == "high"]
    medium = [a for a in anomalies_data if a.get("severity") == "medium"]
    recent = sorted(anomalies_data, key=lambda x: x.get("date", ""), reverse=True)[:3]

    return {
        "tool": "anomaly",
        "count": len(anomalies_data),
        "high_severity_count": len(high),
        "medium_severity_count": len(medium),
        "recent_anomalies": recent,
        "anomaly_plain": card.get("anomaly_plain", ""),
        "target": card.get("target_col", "target"),
    }


def group_forecast_tool(card: dict, group_forecasts: list, args: dict) -> dict:
    """Return forecast for a specific group or all groups ranked."""
    group_value = args.get("group_value", "").lower()

    if not group_forecasts:
        return {"error": "No group data available for this dataset."}

    if group_value:
        # Filter for specific group
        match = [g for g in group_forecasts if group_value in str(g.get("group", "")).lower()]
        if match:
            return {"tool": "group_forecast", "groups": match}
        return {
            "tool": "group_forecast",
            "message": f"Group '{group_value}' not found.",
            "available_groups": [g["group"] for g in group_forecasts]
        }

    # Return all groups ranked
    return {
        "tool": "group_forecast",
        "groups": group_forecasts,
        "top_group": group_forecasts[0] if group_forecasts else None,
    }


def recommendation_tool(card: dict, args: dict) -> dict:
    """Return optimization recommendations based on SHAP + correlation data."""
    question_context = args.get("question_context", "")
    target = card.get("target_col", "target")
    corr_data = card.get("column_relationships", {}).get("correlations_with_target", {})
    lag_data = card.get("column_relationships", {}).get("lag_correlations", {})
    margin = card.get("column_relationships", {}).get("profit_margin")

    recommendations = []

    # Sort by absolute correlation strength
    sorted_corr = sorted(corr_data.items(), key=lambda x: abs(x[1]), reverse=True)

    for col, corr in sorted_corr[:3]:
        direction = "increase" if corr > 0 else "decrease"
        strength = "strongly" if abs(corr) > 0.6 else ("moderately" if abs(corr) > 0.3 else "weakly")
        recommendations.append({
            "column": col,
            "correlation": corr,
            "insight": f"Increasing {col} {strength} tends to {direction} {target} (corr: {corr:+.2f}).",
            "actionable": abs(corr) > 0.3
        })

    # Lag-based recommendations
    for col, corr in lag_data.items():
        direction = "rise" if corr > 0 else "fall"
        recommendations.append({
            "column": col,
            "lag_insight": f"When {col} goes up this period, {target} tends to {direction} next period (lag-1 corr: {corr:+.2f}).",
            "type": "leading_indicator"
        })

    return {
        "tool": "recommendation",
        "target": target,
        "profit_margin": margin,
        "recommendations": recommendations,
        "top_driver": card.get("top_driver", "recent trend"),
        "top_driver_direction": card.get("top_driver_direction", "neutral"),
        "trend": card.get("trend_direction", "stable"),
        "one_liner": card.get("one_liner", ""),
    }


def compare_tool(card: dict, group_forecasts: list, args: dict) -> dict:
    """Compare two groups side by side."""
    group_a = args.get("group_a", "").lower()
    group_b = args.get("group_b", "").lower()

    if not group_forecasts:
        return {"error": "No group data available for comparison."}

    match_a = next((g for g in group_forecasts if group_a in str(g.get("group", "")).lower()), None)
    match_b = next((g for g in group_forecasts if group_b in str(g.get("group", "")).lower()), None)

    if not match_a or not match_b:
        available = [g["group"] for g in group_forecasts]
        return {
            "error": f"Could not find both groups for comparison.",
            "available_groups": available
        }

    winner = match_a if match_a["expected_change_percent"] > match_b["expected_change_percent"] else match_b

    return {
        "tool": "compare",
        "group_a": match_a,
        "group_b": match_b,
        "better_performing": winner["group"],
        "difference_pct": round(
            match_a["expected_change_percent"] - match_b["expected_change_percent"], 1
        ),
    }

# ADD THESE to api/tools.py — before execute_tool()

def categorical_analysis_tool(card: dict, forecast_data: dict, args: dict) -> dict:
    """
    Analyse a categorical column (region/segment/category/product) by profit,
    sales volume, growth, or any numeric metric.
    Works entirely from group_forecasts + raw summary stored in the card.
    """
    group_col = args.get("group_col", "").lower()
    metric = args.get("metric", "sales").lower()  # sales | profit | growth
    group_forecasts = forecast_data.get("group_forecasts") or []

    if not group_forecasts:
        return {
            "tool": "categorical_analysis",
            "error": "No group/segment data available for this dataset.",
            "suggestion": "This dataset may not have a categorical column like Region or Segment."
        }

    # Rank groups by expected_change_percent (growth) or last_hist (volume)
    if "profit" in metric or "earn" in metric:
        ranked = sorted(group_forecasts, key=lambda x: x.get("last_forecast", 0), reverse=True)
        rank_by = "projected value"
    elif "growth" in metric or "increase" in metric or "rise" in metric:
        ranked = sorted(group_forecasts, key=lambda x: x.get("expected_change_percent", 0), reverse=True)
        rank_by = "expected growth %"
    else:
        ranked = sorted(group_forecasts, key=lambda x: x.get("last_hist", 0), reverse=True)
        rank_by = "current volume"

    group_col_name = ranked[0].get("group_col", "Segment") if ranked else "Segment"

    return {
        "tool": "categorical_analysis",
        "group_column": group_col_name,
        "metric": metric,
        "rank_by": rank_by,
        "ranked_groups": [
            {
                "rank": i + 1,
                "group": g["group"],
                "current_value": round(g.get("last_hist", 0), 2),
                "forecast_value": round(g.get("last_forecast", 0), 2),
                "expected_change_pct": round(g.get("expected_change_percent", 0), 1),
            }
            for i, g in enumerate(ranked)
        ],
        "top_group": ranked[0]["group"] if ranked else "unknown",
        "bottom_group": ranked[-1]["group"] if ranked else "unknown",
        "total_groups": len(ranked),
    }


def correlation_deep_tool(card: dict, args: dict) -> dict:
    """
    Deep correlation analysis between any two columns.
    Answers: 'Does X affect Y?', 'What drives Z?', 'Which column matters most?'
    """
    col_a = args.get("col_a", "").lower()
    col_b = args.get("col_b", card.get("target_col", "target")).lower()

    corr_data = card.get("column_relationships", {}).get("correlations_with_target", {})
    lag_data = card.get("column_relationships", {}).get("lag_correlations", {})
    target = card.get("target_col", "target")

    # Find matching columns (case-insensitive)
    matched_corr = {}
    for col, val in corr_data.items():
        if not col_a or col_a in col.lower() or col.lower() in col_a:
            matched_corr[col] = val

    # If no specific col asked, return full ranking
    if not matched_corr:
        matched_corr = corr_data

    sorted_corr = sorted(matched_corr.items(), key=lambda x: abs(x[1]), reverse=True)

    insights = []
    for col, corr in sorted_corr[:5]:
        strength = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.3 else "weak")
        direction = "positive" if corr > 0 else "negative"
        insights.append({
            "column": col,
            "correlation": corr,
            "strength": strength,
            "direction": direction,
            "plain": f"{col} has a {strength} {direction} link with {target} ({corr:+.2f}). "
                     f"{'Increasing ' + col + ' tends to increase ' + target if corr > 0 else 'Increasing ' + col + ' tends to decrease ' + target}."
        })

    lag_insights = []
    for col, corr in lag_data.items():
        if not col_a or col_a in col.lower():
            lag_insights.append({
                "column": col,
                "lag_correlation": corr,
                "plain": f"Last period's {col} predicts this period's {target} (lag-1 corr: {corr:+.2f})."
            })

    return {
        "tool": "correlation_deep",
        "target": target,
        "query_column": col_a or "all",
        "insights": insights,
        "lag_insights": lag_insights,
        "top_driver": insights[0]["column"] if insights else card.get("top_driver", "recent trend"),
        "top_driver_plain": insights[0]["plain"] if insights else "",
    }


def column_stats_tool(card: dict, args: dict) -> dict:
    """
    Return stats for any specific column the user asks about.
    Answers: 'Tell me about Discount', 'What is the average price?', 'How is Quantity trending?'
    """
    col_query = args.get("column", "").lower()
    target = card.get("target_col", "target")
    other_cols = card.get("other_columns", {})

    # Find best match
    matched = {}
    for col, stats in other_cols.items():
        if col_query and (col_query in col.lower() or col.lower() in col_query):
            matched[col] = stats

    # If no match, return all columns summary
    if not matched:
        matched = other_cols

    corr_data = card.get("column_relationships", {}).get("correlations_with_target", {})

    results = []
    for col, stats in matched.items():
        entry = {
            "column": col,
            "current_value": stats.get("last"),
            "mean": stats.get("mean"),
            "recent_change_pct": stats.get("change_pct"),
            "trend": stats.get("trend"),
            "correlation_with_target": corr_data.get(col),
        }
        results.append(entry)

    return {
        "tool": "column_stats",
        "target": target,
        "queried_column": col_query or "all",
        "columns": results,
        "count": len(results),
    }


def profitability_tool(card: dict, forecast_data: dict, args: dict) -> dict:
    """
    Profitability-focused analysis: margin, profit drivers, profit by segment.
    Answers: 'Which region is most profitable?', 'What is my profit margin?',
             'How can I increase profit?'
    """
    target = card.get("target_col", "target")
    margin_data = card.get("column_relationships", {}).get("profit_margin")
    corr_data = card.get("column_relationships", {}).get("correlations_with_target", {})
    group_forecasts = forecast_data.get("group_forecasts") or []

    # Determine if target IS profit or if profit is a feature column
    target_lower = target.lower()
    is_profit_target = any(k in target_lower for k in ["profit", "income", "margin", "earning"])

    # Profit by group
    group_profit = None
    if group_forecasts:
        ranked = sorted(group_forecasts, key=lambda x: x.get("last_forecast", 0), reverse=True)
        group_profit = [
            {
                "group": g["group"],
                "group_col": g.get("group_col", "Segment"),
                "current": round(g.get("last_hist", 0), 2),
                "forecast": round(g.get("last_forecast", 0), 2),
                "growth_pct": round(g.get("expected_change_percent", 0), 1),
            }
            for g in ranked
        ]

    # Find profit-related columns in correlations
    profit_corr = {}
    for col, corr in corr_data.items():
        col_lower = col.lower()
        if any(k in col_lower for k in ["discount", "cost", "expense", "price", "quantity", "margin"]):
            profit_corr[col] = corr

    return {
        "tool": "profitability",
        "target": target,
        "is_profit_target": is_profit_target,
        "margin_data": margin_data,
        "group_profit_ranking": group_profit,
        "top_group": group_profit[0] if group_profit else None,
        "profit_related_correlations": profit_corr,
        "one_liner": card.get("one_liner", ""),
        "trend": card.get("trend_direction", "stable"),
    }

def execute_tool(tool_call: dict, card: dict, forecast_data: dict,
                 anomalies_data: list, group_forecasts: list) -> dict:
    """Route tool_call JSON to the correct tool function."""
    tool_name = tool_call.get("tool", "none")
    args = tool_call.get("args", {})

    if tool_name == "forecast":
        return forecast_tool(card, args)
    elif tool_name == "scenario":
        return scenario_tool(card, forecast_data, args)
    elif tool_name == "anomaly":
        return anomaly_tool(card, anomalies_data, args)
    elif tool_name == "group_forecast":
        return group_forecast_tool(card, group_forecasts, args)
    elif tool_name == "categorical_analysis":
        return categorical_analysis_tool(card, forecast_data, args)
    elif tool_name == "correlation_deep":
        return correlation_deep_tool(card, args)
    elif tool_name == "column_stats":
        return column_stats_tool(card, args)
    elif tool_name == "profitability":
        return profitability_tool(card, forecast_data, args)
    elif tool_name == "recommendation":
        return recommendation_tool(card, args)
    elif tool_name == "compare":
        return compare_tool(card, group_forecasts, args)
    else:
        return {
            "tool": "none",
            "one_liner": card.get("one_liner", ""),
            "anomaly_plain": card.get("anomaly_plain", ""),
            "target": card.get("target_col", "target"),
        }
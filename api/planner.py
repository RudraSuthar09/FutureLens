# api/planner.py
import os
import json
import logging
import re
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

PLANNER_SYSTEM = """You are a routing agent for a time-series forecasting app.
Given a user question, return ONLY a JSON object selecting the correct tool.

Tools:
- forecast: future predictions/values. Args: {"periods": int}
- scenario: "what if X changes by N%". Args: {"change_percent": float, "scenario_type": "growth"|"flat"|"recent_trend", "target_col": str}
- anomaly: unusual/sudden changes, spikes, drops, alerts. Args: {}
- group_forecast: forecast for a SPECIFIC named group/region/segment. Args: {"group_value": str}
- categorical_analysis: RANK or COMPARE regions/segments/categories by profit/sales/growth. Args: {"metric": "profit"|"sales"|"growth", "group_col": str}
- profitability: profit margin, most profitable region/segment, profit drivers. Args: {}
- correlation_deep: how columns relate to each other, what drives the target. Args: {"col_a": str}
- column_stats: stats about a specific column (trend, current value, average). Args: {"column": str}
- recommendation: optimization advice, best strategy, what to improve. Args: {"question_context": str}
- compare: compare exactly two named groups side by side. Args: {"group_a": str, "group_b": str}
- none: greeting or cannot determine. Args: {}

CRITICAL RULES:
- Use categorical_analysis (NOT group_forecast) when user asks to RANK or FIND BEST region/segment
- Use profitability when user asks about profit by region/segment
- Use group_forecast ONLY when user asks about ONE specific named group
- Return ONLY valid JSON, no explanation, no markdown

Examples:
{"tool": "categorical_analysis", "args": {"metric": "profit"}}
{"tool": "profitability", "args": {}}
{"tool": "forecast", "args": {"periods": 4}}
{"tool": "correlation_deep", "args": {"col_a": "discount"}}
{"tool": "column_stats", "args": {"column": "discount"}}
{"tool": "scenario", "args": {"change_percent": 10, "scenario_type": "growth", "target_col": "sales"}}"""


def plan(message: str, card: dict) -> dict:
    """
    Use Groq to classify user intent and return a tool call JSON.
    Falls back to keyword planner on any failure.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using keyword fallback planner.")
        return _fallback_planner(message, card)

    target = card.get("target_col", "target")
    freq = card.get("freq_label", "period")
    other_cols = list(card.get("other_columns", {}).keys())
    corr_cols = list(card.get("column_relationships", {})
                     .get("correlations_with_target", {}).keys())
    group_forecasts = card.get("group_forecasts_hint", [])

    context_hint = (
        f"Target: {target}. Frequency: {freq}. "
        f"Other columns: {', '.join(other_cols[:6]) if other_cols else 'none'}. "
        f"Correlated columns: {', '.join(corr_cols[:4]) if corr_cols else 'none'}."
    )

    user_content = f"Context: {context_hint}\n\nUser question: {message}"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        tool_call = json.loads(raw)
        logger.info(f"Planner result: {tool_call}")
        return tool_call

    except json.JSONDecodeError as e:
        logger.warning(f"Planner JSON parse failed: {e}. Using keyword fallback.")
        return _fallback_planner(message, card)
    except Exception as e:
        logger.warning(f"Groq planner error: {e}. Using keyword fallback.")
        return _fallback_planner(message, card)


def _fallback_planner(message: str, card: dict) -> dict:
    """Keyword-based fallback — handles ~90% of real queries."""
    msg = message.lower()
    target = card.get("target_col", "target")
    corr_cols = list(card.get("column_relationships", {})
                     .get("correlations_with_target", {}).keys())
    other_cols = list(card.get("other_columns", {}).keys())

    # Anomaly
    if any(k in msg for k in ["unusual", "spike", "drop", "anomaly", "sudden", "alert", "weird"]):
        return {"tool": "anomaly", "args": {}}

    # Profitability by region/segment
    if any(k in msg for k in ["profit", "earn", "margin", "income"]) and \
       any(k in msg for k in ["region", "segment", "category", "where", "which", "area", "east", "west", "south", "north", "central"]):
        return {"tool": "profitability", "args": {}}

    # Categorical ranking
    if any(k in msg for k in ["which region", "which segment", "which category", "which area",
                               "best region", "best segment", "rank", "most sales", "most profitable",
                               "focus", "should i focus", "where should"]):
        metric = "profit" if any(k in msg for k in ["profit", "earn", "margin"]) else \
                 "growth" if any(k in msg for k in ["grow", "increase", "rise"]) else "sales"
        return {"tool": "categorical_analysis", "args": {"metric": metric}}

    # Compare two groups
    if ("vs" in msg or "versus" in msg or "compare" in msg) and \
       not any(k in msg for k in ["forecast", "predict", "next"]):
        return {"tool": "compare", "args": {"group_a": "", "group_b": ""}}

    # Column stats
    for col in other_cols:
        if col.lower() in msg and any(k in msg for k in ["what is", "tell me", "show", "average", "how is", "trend"]):
            return {"tool": "column_stats", "args": {"column": col}}

    # Correlation / what drives
    if any(k in msg for k in ["what drives", "what causes", "why does", "affect", "impact", "relation", "link"]):
        col_a = ""
        for col in corr_cols + other_cols:
            if col.lower() in msg:
                col_a = col
                break
        return {"tool": "correlation_deep", "args": {"col_a": col_a}}

    # Scenario / what-if
    if any(k in msg for k in ["what if", "if i", "if we", "suppose", "scenario",
                               "increase by", "decrease by", "double", "triple"]):
        pct_match = re.search(r"(\d+)\s*%", msg)
        pct = float(pct_match.group(1)) if pct_match else 10.0
        col = target
        for c in corr_cols:
            if c.lower() in msg:
                col = c
                break
        return {"tool": "scenario", "args": {"change_percent": pct, "scenario_type": "growth", "target_col": col}}

    # Recommendation
    if any(k in msg for k in ["recommend", "optimal", "best", "strategy", "improve",
                               "maximize", "should i", "what should", "how to increase",
                               "what discount", "what price"]):
        return {"tool": "recommendation", "args": {"question_context": message}}

    # Forecast (time-based questions)
    if any(k in msg for k in ["next", "future", "predict", "forecast", "week", "month",
                               "will", "look like", "coming", "upcoming"]):
        n_match = re.search(r"(\d+)\s*(week|month|day|period)", msg)
        periods = 4
        if n_match:
            n = int(n_match.group(1))
            unit = n_match.group(2)
            ppw = card.get("periods_per_week", 1)
            ppm = card.get("periods_per_month", 4)
            periods = max(1, round(n * ppw)) if "week" in unit else \
                      max(1, round(n * ppm)) if "month" in unit else n
        return {"tool": "forecast", "args": {"periods": periods}}

    return {"tool": "none", "args": {}}
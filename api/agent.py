import os
import yaml
import json
import re
import google.generativeai as genai
from typing import List, Dict, Any

# Load API key
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def explain_anomaly(anomaly_date: str, anomaly_value: float, expected_value: float, top_features: str) -> str:
    """
    Returns simple English explanation for an anomaly.
    """
    return f"On {anomaly_date}, there was an anomaly. The value was {anomaly_value} but the expected value was {expected_value}. The top driving features were: {top_features}."

def recommend_action(anomaly_type: str, top_feature: str) -> str:
    """
    Loads rules.yaml, matches rule, returns action + impact.
    """
    try:
        with open('config/rules.yaml', 'r') as file:
            rules = yaml.safe_load(file).get('rules', [])

        for rule in rules:
            if rule['anomaly_type'] == anomaly_type and rule['top_feature'] == top_feature:
                return f"Action: {rule['action']} Impact: {rule['impact']}"

        return "No specific rule found. Consider general review of the specified feature."
    except Exception as e:
        return f"Error loading recommendations: {e}"

def simulate_scenario(base_forecast: str, change_percent: float) -> str:
    """
    Adjusts forecast by change_percent, returns comparison.
    Since base_forecast is a stringified list in the prompt to avoid complicated types,
    we'll do a simple string parsing or just return a text response.
    """
    try:
        forecast_list = json.loads(base_forecast)
        adjusted = [val * (1 + (change_percent / 100.0)) for val in forecast_list]
        return f"Original Forecast: {forecast_list}. Adjusted Forecast: {adjusted}."
    except Exception:
        return f"Forecast adjusted by {change_percent}% uniformly."

def generate_report(forecast_summary: str, anomalies: str, truth_score: float) -> str:
    """
    Returns full text summary of analysis.
    """
    return f"Report Summary:\nForecast: {forecast_summary}\nAnomalies: {anomalies}\nModel Truth Score: {truth_score:.1f}%."

# Tools dictionary for easy reference if needed
tools = [explain_anomaly, recommend_action, simulate_scenario, generate_report]

_DEFAULT_SYSTEM = """You are FutureLens, an AI forecasting co-pilot.

Behavior rules (important):
- Be concise and non-technical. Prefer 3–7 bullet points.
- Handle typos and minor misspellings automatically.
- Be transparent: use the forecast window and uncertainty band; do not overclaim.
- If asked "which X will expand/grow", interpret as "which group has the highest expected growth"
  using any provided group forecast summary. If not available, explain what grouping column is needed.
- If the user asks about columns that exist in the dataset profile, answer using that profile.
- If you don't have enough information, ask ONE clarifying question, not multiple.

Answer style:
- Start with the direct answer.
- Then give short evidence (dates, % change, top anomalies).
- End with 1–2 suggested next steps if relevant.
"""

def _looks_like_cannot_answer(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    triggers = [
        "i cannot answer",
        "i can't answer",
        "no information",
        "not enough information",
        "not provided dataset",
        "there is no information",
        "i don't have",
        "cannot determine",
    ]
    return any(x in t for x in triggers)

def _extract_group_forecasts_from_system(system_instruction: str) -> list[dict]:
    """
    Best-effort extraction: if main.py includes group forecast summary in the system prompt,
    we can parse it or at least detect the top lines.
    This is a lightweight hackathon fallback, not perfect parsing.
    """
    if not system_instruction:
        return []

    # If your main.py system prompt contains a JSON-ish blob, try to find it.
    # Otherwise, return empty.
    # (We keep this conservative to avoid hallucinating.)
    return []

def _fallback_answer_from_prompt(message: str, system_instruction: str) -> str | None:
    """
    If Gemini says 'cannot answer', we try to provide a helpful response based on
    group forecast hints already embedded in system_instruction.
    """
    if not system_instruction:
        return None

    msg = (message or "").lower()

    # Heuristic: region/category expansion questions
    if any(k in msg for k in ["region", "category", "segment", "product"]) and any(k in msg for k in ["expand", "grow", "increase", "highest", "top"]):
        # Try to find the "Group forecast available by" block that main.py adds
        m = re.search(r"Group forecast available by `([^`]+)`.*?:\n(.+)", system_instruction, flags=re.DOTALL)
        if m:
            group_col = m.group(1)
            lines_block = m.group(2).strip()
            top_lines = []
            for line in lines_block.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    top_lines.append(line)
                if len(top_lines) >= 5:
                    break
            if top_lines:
                return (
                    f"Based on the group forecast by **{group_col}**, the top expected growth is:\n"
                    + "\n".join(top_lines)
                    + "\n\nIf you want, tell me which specific time window you care about (next few weeks vs next 1–2 months)."
                )

        return (
            "I can answer that once we pick a grouping column (for example: Region, Category, Segment, Product). "
            "Your current forecast is for a single target metric over time. "
            "If you tell me the column name to group by, I can rank which group is expected to grow the most."
        )

    return None

def chat(message: str, session_context: List[Dict[str, str]] = None, system_instruction: str = None) -> str:
    """
    Sends message to Gemini with tool definitions and session context.
    Handles tool calls and returns final text response.
    Includes hardcoded fallback if API fails (for Demo Mode).
    """
    if "Demo Mode" in (message or "") or "week 67" in (message or ""):
        return "In Demo Mode, I detected an anomaly at week 67. The primary driver is a drop in ad_spend which contributed heavily to the sales decline. I recommend reviewing the localized drop in ad_spend to avoid further revenue loss."

    if not api_key or api_key == "your_gemini_api_key_here":
        return "I am currently running in Demo fallback mode. A drop in ad_spend is the root cause for the anomaly!"

    # Always ensure we have a strong system instruction
    final_system = system_instruction.strip() if system_instruction else ""
    if final_system:
        final_system = _DEFAULT_SYSTEM + "\n\n" + final_system
    else:
        final_system = _DEFAULT_SYSTEM

    try:
        # Initialize Gemini model with tools
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=tools,
            system_instruction=final_system,
        )

        # Construct history from session_context
        history = []
        if session_context:
            for msg in session_context:
                role = "user" if msg.get("role") == "user" else "model"
                history.append({"role": role, "parts": [msg.get("content", "")]})

        chat_session = model.start_chat(history=history, enable_automatic_function_calling=True)
        response = chat_session.send_message(message)
        text = response.text or ""

        # If Gemini refuses due to missing context, try a deterministic fallback
        if _looks_like_cannot_answer(text):
            fallback = _fallback_answer_from_prompt(message, final_system)
            if fallback:
                return fallback

        return text

    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).error(f"Gemini API Error: {e}")
        return "I was unable to reach the AI model at this moment. Please try again in a few seconds."
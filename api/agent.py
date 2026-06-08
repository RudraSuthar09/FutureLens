"""
api/agent.py — Groq Agent + Gemini Formatter

Architecture:
1. GROQ AGENT (Main Orchestrator)
   - Receives user question + all data
   - Analyzes and processes the data
   - Generates forecast, detects patterns, analyzes anomalies
   - Returns structured results in JSON

2. GEMINI FORMATTER (Output Formatter)
   - Receives structured results from Groq
   - Formats for human-readable presentation
   - Saves tokens by receiving pre-processed data
   - Handles styling and natural language

This splits the work efficiently:
- Groq: Heavy lifting (analysis, forecasting, data processing)
- Gemini: Light formatting (presentation only)
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from groq import Groq

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# GROQ AGENT — Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

GROQ_AGENT_SYSTEM = """You are FutureLens Agent, an expert data analyst.

Your job: Analyze user questions against available forecast and anomaly data.

Return ONLY valid JSON with this structure:
{
  "answer": "Direct answer to the question",
  "key_metrics": {"metric_name": value, ...},
  "insights": ["insight 1", "insight 2"],
  "next_step": "What they should do next",
  "confidence": "High|Medium|Low"
}

Guidelines:
- Answer must directly address the user question
- Extract and include key numbers from provided data
- Insights should be 2-3 important findings
- Only mention data you were provided
- Format numbers with 2 decimal places
- Never fabricate or guess values
"""


def _get_groq_client() -> Groq:
    """Get Groq client."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def groq_agent(
    user_message: str,
    forecast_data: Dict[str, Any] = None,
    anomalies_data: List[Dict] = None,
    group_forecasts: List[Dict] = None,
    card: Dict[str, Any] = None,
    chat_history: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Main Groq Agent - Orchestrates analysis and processing.
    
    Receives user question and all available data, processes it,
    and returns structured JSON results for Gemini to format.
    """
    if not GROQ_API_KEY:
        return {
            "answer": "Groq API not configured",
            "error": True,
        }
    
    try:
        client = _get_groq_client()
        card = card or {}
        target_col = card.get("target_col", "target")
        freq_label = card.get("freq_label", "period")
        
        # Build data summary for Groq
        forecast_summary = ""
        if forecast_data:
            hist = forecast_data.get("historical", [])
            fc = forecast_data.get("forecast", [])
            lo = forecast_data.get("lower", [])
            hi = forecast_data.get("upper", [])
            
            if hist and fc:
                last_val = float(hist[-1])
                first_fc = float(fc[0])
                last_fc = float(fc[-1])
                change = ((last_fc - last_val) / abs(last_val) * 100) if last_val else 0
                forecast_summary = f"""
FORECAST DATA:
- Historical: {len(hist)} data points, latest value = {last_val:.2f}
- Forecast: {len(fc)} periods ahead
- Next period forecast: {first_fc:.2f} [range: {lo[0]:.2f} to {hi[0]:.2f}]
- Final period forecast: {last_fc:.2f} [range: {lo[-1]:.2f} to {hi[-1]:.2f}]
- Overall change: {change:+.1f}% over forecast period
"""
        
        anomalies_summary = ""
        if anomalies_data:
            high_sev = len([a for a in anomalies_data if a.get("severity") == "high"])
            if anomalies_data:
                recent = anomalies_data[-1]
                anomalies_summary = f"""
ANOMALIES:
- Total detected: {len(anomalies_data)}
- High severity: {high_sev}
- Recent: {recent.get('date', 'unknown')} with {recent.get('deviation_percent', 0):.0f}% deviation
"""
        
        groups_summary = ""
        if group_forecasts and len(group_forecasts) > 0:
            sorted_groups = sorted(group_forecasts, key=lambda x: x.get("last_forecast", 0), reverse=True)[:3]
            groups_summary = "TOP GROUPS:\n"
            for g in sorted_groups:
                groups_summary += f"  - {g.get('group', 'N/A')}: forecast {g.get('last_forecast', 0):.2f}, change {g.get('expected_change_percent', 0):+.1f}%\n"
        
        # Build agent message
        agent_prompt = f"""User question: {user_message}

Target metric: {target_col}
Time frequency: {freq_label}

{forecast_summary}
{anomalies_summary}
{groups_summary}

Analyze this question against the data. Return structured JSON response."""
        
        # Build messages list
        messages = [
            {
                "role": "system",
                "content": GROQ_AGENT_SYSTEM,
            }
        ]
        
        # Add chat history
        if chat_history:
            messages.extend(chat_history)
            
        # Add current prompt
        messages.append({
            "role": "user",
            "content": agent_prompt,
        })
        
        # Call Groq
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )
        
        response_text = message.choices[0].message.content.strip()
        
        # Parse JSON from response
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text
            
            result = json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            result = {
                "answer": response_text[:500],
                "key_metrics": {},
                "insights": [],
                "next_step": "Ask for clarification",
                "confidence": "Medium",
            }
        
        result["source"] = "groq_agent"
        return result
        
    except Exception as e:
        logger.error(f"Groq agent error: {e}")
        return {
            "answer": f"Analysis error: {str(e)[:80]}",
            "error": True,
            "source": "groq_agent_error",
        }


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI FORMATTER — Output Formatting  
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_FORMATTER_SYSTEM = """You are a data communication specialist for FutureLens.

Your job: Convert technical analysis into business-friendly insights.

Rules for formatting:
- Answer in 2-4 sentences maximum
- Start with the key finding or number
- Use simple language (no jargon)
- Include confidence level if provided
- End with ONE suggested follow-up question
- Never fabricate or guess numbers
- Format: "Key finding. Supporting detail. Action. Question?"
"""


def format_for_user(
    groq_result: Dict[str, Any],
    user_message: str = "",
) -> str:
    """
    Format Groq's structured analysis for user presentation using Gemini.
    Converts technical JSON into friendly narrative.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return _format_fallback(groq_result)
    
    try:
        answer = groq_result.get("answer", "")
        insights = groq_result.get("insights", [])
        metrics = groq_result.get("key_metrics", {})
        next_step = groq_result.get("next_step", "")
        confidence = groq_result.get("confidence", "Medium")
        
        # Build formatting request
        format_prompt = f"""Format this analysis for a business user:

Question: {user_message}
Answer: {answer}
Key metrics: {json.dumps(metrics) if metrics else 'None'}
Insights: {', '.join(insights) if insights else 'None'}
Recommended action: {next_step}
Confidence: {confidence}

Output: 2-4 sentences that start with the key finding, include important numbers, 
suggest a next step, and end with ONE follow-up question."""
        
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            system_instruction=GEMINI_FORMATTER_SYSTEM,
        )
        
        response = model.generate_content(
            format_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.2,
            )
        )
        
        formatted = response.text.strip() if response and response.text else ""
        
        return formatted if formatted else _format_fallback(groq_result)
            
    except Exception as e:
        logger.warning(f"Gemini formatting failed: {e}")
        return _format_fallback(groq_result)


def _format_fallback(groq_result: Dict[str, Any]) -> str:
    """Fallback formatting when Gemini unavailable."""
    answer = groq_result.get("answer", "No analysis available")
    insights = groq_result.get("insights", [])
    next_step = groq_result.get("next_step", "")
    confidence = groq_result.get("confidence", "")
    
    formatted = answer
    if insights:
        formatted += f" Key insight: {insights[0]}."
    if next_step:
        formatted += f" Recommended: {next_step}."
    if confidence:
        formatted += f" Confidence: {confidence}."
    
    if formatted and formatted[-1] not in ".!?":
        formatted += "."
    
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def explain(
    user_message: str,
    tool_result: Dict[str, Any],
    card: Dict[str, Any] = None,
    chat_history: List[Dict] = None,
) -> str:
    """Legacy wrapper for old chat flow compatibility."""
    if isinstance(tool_result, dict):
        return _format_fallback(tool_result)
    return str(tool_result)


def chat(
    message: str,
    session_context: List = None,
    system_instruction: str = "",
    intelligence_card: Dict = None,
) -> str:
    """Legacy wrapper for backwards compatibility."""
    return f"Chat message received: {message[:40]}..."

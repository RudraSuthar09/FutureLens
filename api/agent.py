# api/agent.py — Gemini as EXPLAINER only
import os
import time
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)

# Compact explainer system prompt — Gemini only sees this + tool result + user question
EXPLAINER_SYSTEM = """You are FutureLens, a friendly AI forecasting assistant.
You will receive:
1. The user's original question
2. A JSON result computed by a Python backend tool

Your job: explain the JSON result in plain English. Rules:
- Lead with the KEY NUMBER or direct answer in sentence 1
- Give a complete answer.
- Use 4 to 8 sentences when needed.
- Keep answer within 4 to 5 short lines.
- Answer must be fully complete.
- Never stop mid-sentence.
- Use point-to-point style.
- Never end with words like: 'by the end of', 'increase to', 'because', 'from', 'vs'.
- If needed, shorten the answer but always complete it.
- Start with the direct answer or key number.
- Mention important drivers/reasons if available.
- If the question asks for recommendation, explain reasoning clearly.
- Never cut off mid-sentence.
- For anomaly questions: explain what changed, likely drivers only if present in JSON, and suggest one next step.
- Never invent causes. Only mention drivers explicitly present in the JSON result.
- No technical jargon (no "MAPE", "conformal", "SHAP", "correlation coefficient")
- If cross_column_impact exists, always mention it with: "Note: this is a statistical estimate based on correlation, not a guaranteed outcome."
- For scenario results with is_cross_sectional=true: explain using estimated_target_change_pct and estimated_new_value. Always add the correlation-based disclaimer.
- For scenario results with a note field: include the note in your explanation naturally.
- End with exactly ONE follow-up question the user might want to ask next.
- Never say "I don't know" if data is in the JSON."""


def _pick_model() -> str:
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                if "flash" in m.name:
                    return m.name
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception:
        pass
    return "models/gemini-1.5-flash"


MODEL_NAME = os.environ.get("GEMINI_MODEL") or _pick_model()


def explain(user_message: str, tool_result: dict, card: dict, chat_history=None) -> str:
    """
    Gemini explains a pre-computed tool result.
    Token cost: ~300 tokens in, ~150 tokens out.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return _safe_fallback(tool_result, card, user_message)

    tool_name = tool_result.get("tool", "none")
    result_str = json.dumps(tool_result, indent=None)

    # Hard limit: trim to most important fields if too long
    if len(result_str) > 2000:
        important_keys = ["tool", "target", "overall_change_pct", "trend", "period_data",
                          "difference_percent", "scenario_total", "baseline_total",
                          "count", "recommendations", "one_liner", "error",
                          "cross_column_impact", "better_performing", "difference_pct",
                          "estimated_target_change_pct", "estimated_new_value",
                          "baseline_value", "input_column", "correlation",
                          "is_cross_sectional", "note", "message", "suggestion"]
        trimmed = {k: tool_result[k] for k in important_keys if k in tool_result}
        result_str = json.dumps(trimmed, indent=None)

    history_text = ""
    if chat_history:
        history_lines = []
        for row in chat_history:
            role = row["role"].capitalize()
            content = row["content"]
            history_lines.append(f"{role}: {content}")
        history_text = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    user_content = (
        history_text +
        f"Current user question: {user_message}\n\n"
        f"Backend tool '{tool_name}' returned:\n{result_str}"
    )

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=EXPLAINER_SYSTEM
    )

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=1500,
        temperature=0.2,
        top_p=0.85,
    )

    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                [{"role": "user", "parts": [user_content]}],
                generation_config=generation_config
            )
            text = getattr(response, "text", "") or ""
            text = text.strip()

            bad_endings = (
                "by the end of", "increase to", "decrease to",
                "from", "vs", "because", "due to", "expected to"
            )
            if text.lower().endswith(bad_endings):
                text += " the forecast period."

            if text and text[-1] not in ".!?":
                text += "."

            return text

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in str(e) or "quota" in err_str:
                wait = 5 * (attempt + 1)
                logger.warning(f"Gemini rate limit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"Gemini error: {e}")
            break

    logger.error(f"Gemini explainer failed: {last_error}")
    return _safe_fallback(tool_result, card, user_message)


def _safe_fallback(tool_result: dict, card: dict, message: str) -> str:
    """Template-based fallback when Gemini is unavailable."""
    tool = tool_result.get("tool", "none")
    target = card.get("target_col", "metric")
    freq = card.get("freq_label", "period")

    # Handle cross-sectional "no forecast" gracefully
    if tool == "forecast" and tool_result.get("error") == "no_forecast":
        return (
            f"{tool_result.get('message', '')} "
            f"{tool_result.get('suggestion', '')}"
        )

    if tool == "forecast":
        periods = tool_result.get("period_data", [])
        chg = tool_result.get("overall_change_pct", 0)
        trend = tool_result.get("trend", "stable")
        if periods:
            p1 = periods[0]
            return (
                f"{target} is forecast at {p1['forecast']:.2f} next {freq} "
                f"(range {p1['lower']:.2f}–{p1['upper']:.2f}). "
                f"Overall {len(periods)}-{freq} trend: {chg:+.1f}% ({trend}). "
                f"{tool_result.get('reliability', '')} "
                f"Would you like to explore a scenario — e.g. what if {target} grows by 10%?"
            )

    if tool == "scenario":
        diff = tool_result.get("difference_percent", 0)
        scen_total = tool_result.get("scenario_total", 0)
        base_total = tool_result.get("baseline_total", 0)
        cross = tool_result.get("cross_column_impact")
        input_col = tool_result.get("input_column")

        # Cross-sectional correlation-based scenario
        if tool_result.get("is_cross_sectional") and input_col:
            corr = tool_result.get("correlation", 0)
            est_pct = tool_result.get("estimated_target_change_pct", diff)
            est_val = tool_result.get("estimated_new_value", scen_total)
            change_pct = tool_result.get("change_percent", 0)
            return (
                f"Based on the statistical link between {input_col} and {target} "
                f"(correlation: {corr:+.2f}), a {change_pct:.0f}% increase in {input_col} "
                f"is estimated to change {target} by {est_pct:+.1f}%, "
                f"bringing it to approximately {est_val:.2f} from {base_total:.2f}. "
                f"Note: this is a statistical estimate based on correlation, not a guaranteed outcome. "
                f"Would you like to explore another scenario?"
            )

        # Direct target scenario (cross-sectional, no correlation match)
        if tool_result.get("is_cross_sectional"):
            note = tool_result.get("note", "")
            return (
                f"{note} "
                f"Would you like to explore the correlation drivers of {target} instead?"
            )

        # Time-series scenario
        base_text = (
            f"Under this scenario, {target} is projected at {scen_total:.2f} "
            f"vs baseline {base_total:.2f} ({diff:+.1f}% difference). "
        )
        if cross:
            base_text += (
                f"Note: this is a statistical estimate based on a {cross['correlation']:+.2f} "
                f"correlation — not a guarantee. "
            )
        base_text += "Would you like to compare this with a flat or trend-based scenario?"
        return base_text

    if tool == "anomaly":
        msg = tool_result.get("anomaly_plain", "An unusual change was detected.")
        drivers = tool_result.get("possible_drivers", [])
        reply = msg
        if drivers:
            reply += " Potential drivers: " + ", ".join(drivers[:2]) + "."
        reply += " Suggested next step: investigate affected region/system logs and compare with recent trends."
        return reply

    if tool == "recommendation":
        recs = tool_result.get("recommendations", [])
        if recs:
            top = recs[0]
            return (
                f"{top.get('insight', '')} "
                f"Top forecast driver: {tool_result.get('top_driver', 'recent trend')} "
                f"({tool_result.get('top_driver_direction', 'neutral')} impact). "
                f"{tool_result.get('one_liner', '')} "
                f"Would you like a scenario simulation to test a specific change?"
            )

    # Generic fallback
    one_liner = card.get("one_liner", "")
    return (
        f"{one_liner} "
        f"Ask me: 'What will {target} look like next {freq}?', "
        f"'Are there any sudden changes?', or 'What if {target} increases by 10%?'"
    )


# Keep old chat() signature for any places still calling it — routes to new flow
def chat(message: str, session_context: list,
         system_instruction: str, intelligence_card: dict = None) -> str:
    """Legacy compatibility wrapper — prefer calling plan() + execute_tool() + explain() directly."""
    card = intelligence_card or {}
    tool_result = {"tool": "none", "one_liner": card.get("one_liner", ""),
                   "anomaly_plain": card.get("anomaly_plain", "")}
    return explain(message, tool_result, card)
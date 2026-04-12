# api/agent.py  — full file replacement

import os
import time
import logging
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _normalize_role(role: str) -> str:
    r = (role or "").lower()
    if r in {"assistant", "model", "ai", "bot"}:
        return "model"
    if r in {"user", "human"}:
        return "user"
    return "user"


def _pick_model() -> str:
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            if "flash" in m.name:
                return m.name
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            return m.name
    raise RuntimeError("No Gemini models available.")


MODEL_NAME = os.environ.get("GEMINI_MODEL") or _pick_model()


def _build_contents(session_context: list, message: str) -> list:
    """
    Build a role-alternating contents list safe for Gemini.
    Gemini requires strict user/model alternation.
    Drops any turn that would create consecutive same-role messages.
    """
    contents = []
    expected_role = "user"  # Gemini conversation must start with user

    for turn in session_context:
        role = _normalize_role(turn.get("role"))
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role != expected_role:
            # Skip turns that break alternation rather than crash
            continue
        contents.append({"role": role, "parts": [content]})
        expected_role = "model" if expected_role == "user" else "user"

    # Final user message — must be "user" role
    if expected_role != "user":
        # History ended on user turn — drop last history item and retry
        if contents:
            contents.pop()
    contents.append({"role": "user", "parts": [message]})
    return contents


def _safe_fallback(card: dict, message: str) -> str:
    """
    Graceful fallback using intelligence card data — never leaks system prompt.
    Produces a useful answer even when Gemini is unavailable.
    """
    msg = message.lower().strip()
    target = card.get("target_col", "the metric")
    freq = card.get("freq_label", "week")
    one_liner = card.get("one_liner", "")
    anomaly = card.get("anomaly_plain", "No anomalies detected.")
    reliability = card.get("reliability_plain", "")
    horizon = card.get("horizon", 4)

    # Forecast questions
    if any(k in msg for k in ["next", "future", "predict", "forecast", "week", "month", "look like", "will"]):
        fc_vals = card.get("forecast_values", [])
        lo_vals = card.get("lower_values", [])
        hi_vals = card.get("upper_values", [])
        if fc_vals:
            mid = fc_vals[min(3, len(fc_vals)-1)]
            lo  = lo_vals[min(3, len(lo_vals)-1)] if lo_vals else mid * 0.9
            hi  = hi_vals[min(3, len(hi_vals)-1)] if hi_vals else mid * 1.1
            return (
                f"{one_liner} "
                f"In {freq} {min(4, horizon)}: central estimate {mid:.2f} "
                f"(range {lo:.2f}–{hi:.2f}). "
                f"{reliability} "
                f"Try asking: 'Are there any sudden changes I should look at?'"
            )
        return f"{one_liner} Try asking about anomalies or scenarios."

    # Anomaly questions
    if any(k in msg for k in ["unusual", "spike", "drop", "anomaly", "sudden", "alert", "change", "wrong"]):
        return (
            f"{anomaly} "
            f"Overall trend: {card.get('trend_direction', 'stable')}. "
            f"Try asking: 'What will {target} look like next {freq}?'"
        )

    # Scenario / what-if
    if any(k in msg for k in ["what if", "scenario", "increase", "decrease", "suppose", "if i", "if we"]):
        last = card.get("last_historical_value", 0)
        est_10 = last * 1.10
        return (
            f"Under a +10% scenario, {target} would reach approximately {est_10:.2f} "
            f"(vs baseline {last:.2f}). "
            f"For cross-column effects, check the correlation data. "
            f"Try the scenario simulator for more precise modelling."
        )

    # Casual / greeting / other
    other_cols = list(card.get("other_columns", {}).keys())
    corr_cols = list(card.get("column_relationships", {}).get("correlations_with_target", {}).keys())
    suggest_col = corr_cols[0] if corr_cols else (other_cols[0] if other_cols else target)
    return (
        f"I'm FutureLens, your forecasting assistant. Here's your snapshot: {one_liner} "
        f"{anomaly} "
        f"Ask me: 'What will {target} look like next {freq}?', "
        f"'Are there any sudden changes?', or "
        f"'If {suggest_col} increases by 10%, how does that affect {target}?'"
    )


def chat(
    message: str,
    session_context: list,
    system_instruction: str,
    intelligence_card: dict = None,
) -> str:
    """
    Token-efficient Gemini chat with retry on rate-limit and safe fallback.
    - Retries up to 3 times with backoff on 429
    - Never leaks system prompt in error messages
    - Falls back to card-based answers when Gemini is unavailable
    """
    card = intelligence_card or {}

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return _safe_fallback(card, message)

    contents = _build_contents(session_context, message)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction
    )

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=800,
        temperature=0.2,
        top_p=0.8,
        top_k=20
    )

    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                contents=contents,
                generation_config=generation_config
            )
            return response.text

        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            # Rate limit — wait and retry
            if "429" in str(e) or "quota" in err_str or "rate" in err_str:
                wait = 5 * (attempt + 1)   # 5s, 10s, 15s
                logging.warning(f"Gemini rate limit hit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                continue

            # Invalid argument — usually a role ordering issue
            if "invalid" in err_str or "400" in str(e):
                logging.error(f"Gemini invalid request: {e}")
                # Drop history and retry with just the current message
                contents = [{"role": "user", "parts": [message]}]
                continue

            # Any other error — log and break immediately
            logging.error(f"Gemini error (attempt {attempt+1}): {e}")
            break

    # All retries exhausted — use card-based fallback, never leak system prompt
    logging.error(f"Gemini unavailable after retries. Last error: {last_error}")
    return _safe_fallback(card, message)
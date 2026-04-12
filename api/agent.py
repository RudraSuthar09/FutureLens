import os
import logging
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def _pick_model() -> str:
    # Prefer flash models that support generateContent
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            if "flash" in m.name:
                return m.name  # use full name: "models/..."
    # Fallback to first model that supports generateContent
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            return m.name
    raise RuntimeError("No Gemini models available for generateContent with this API key.")

MODEL_NAME = os.environ.get("GEMINI_MODEL") or _pick_model()

def chat(message: str,
         session_context: list,
         system_instruction: str) -> str:
    """Token-efficient Gemini chat.
    Uses pre-built system prompt.
    Max input: ~570 tokens.
    Max output: 600 tokens (adjustable)."""

    # Demo fallback if no API key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return _demo_fallback(message)

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction
        )

        # Build message history (already truncated by /chat endpoint before passing here)
        contents = []
        for turn in session_context:
            contents.append({
                "role": turn["role"],
                "parts": [turn["content"]]
            })
        contents.append({
            "role": "user",
            "parts": [message]
        })

        response = model.generate_content(
            contents=contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800,   # sufficient for 4 sentences + follow-up question
                temperature=0.2,
                top_p=0.8,
                top_k=20
            )
        )
        return response.text

    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return (
            "I had trouble processing that. "
            "Based on the data: "
            f"{system_instruction.split('FORECAST FOR')[1][:200] if 'FORECAST FOR' in system_instruction else 'please try again.'}"
        )

def _demo_fallback(message: str) -> str:
    """Intent-aware fallback when no API key."""
    msg = message.lower()
    if any(k in msg for k in
           ["next", "future", "predict",
            "forecast", "weeks", "look like"]):
        return (
            "Next 4 weeks: central estimate +6.2% "
            "growth. Lower bound: -2.1%. "
            "Upper bound: +12.4%. "
            "Seasonal pattern expected in Week 3. "
            "Top driver: recent trend has a positive "
            "impact. Try asking: "
            "'Are there any sudden changes?'"
        )
    elif any(k in msg for k in
             ["unusual", "spike", "drop",
              "anomaly", "sudden", "alert"]):
        return (
            "1 unusual point detected (high severity). "
            "Most recent: a 28% drop from expected. "
            "Likely driver: reduced momentum in "
            "recent periods. "
            "Suggested action: investigate the "
            "most recent data point more closely."
        )
    elif any(k in msg for k in
             ["what if", "scenario", "increase",
              "decrease", "suppose"]):
        return (
            "Under a +10% scenario, the metric is "
            "expected to reach approximately 11,000 "
            "(vs 10,000 baseline). "
            "Range: 10,450–11,550."
        )
    else:
        return (
            "Based on the demo data: next 4 weeks "
            "show +6.2% growth trend with 1 anomaly "
            "detected. Try asking: "
            "'What will sales look like next week?' "
            "or 'Are there any sudden changes?'"
        )
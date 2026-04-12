"""
api/column_picker.py
────────────────────
Uses Groq (llama-3.3-70b-versatile) to inspect a dataset profile and decide:
  - dataset_type  : "time_series" | "cross_sectional"
  - chart_type    : "horizontal_bar" | "bar" | "scatter" | "pie" | "donut" | "histogram"
  - x_col         : column name for the X axis (or category axis)
  - y_col         : column name for the Y axis (the main numeric measure)
  - color_col     : optional column name for colour-grouping (or None)
  - size_col      : optional column name for bubble size (or None)
  - chart_title   : short human-readable title
  - x_label       : axis label for X
  - y_label       : axis label for Y
  - top_n         : how many rows to show (default 20)
  - sort_order    : "desc" | "asc"
  - reason        : one-sentence explanation of the choice
"""

from __future__ import annotations

import json
import logging
import os
import re

from groq import Groq

logger = logging.getLogger(__name__)

# ── Groq client ──────────────────────────────────────────────────────────────
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


# ── Default fallback ──────────────────────────────────────────────────────────
_FALLBACK: dict = {
    "dataset_type": "cross_sectional",
    "chart_type":   "horizontal_bar",
    "x_col":        None,
    "y_col":        None,
    "color_col":    None,
    "size_col":     None,
    "chart_title":  "Dataset Overview",
    "x_label":      "",
    "y_label":      "",
    "top_n":        20,
    "sort_order":   "desc",
    "reason":       "Fallback: Groq unavailable or returned an unparseable response.",
}


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(profile: dict) -> str:
    columns      = profile.get("columns", [])
    dtypes       = profile.get("dtypes", {})
    num_summary  = profile.get("numeric_summary", {})
    cat_summary  = profile.get("categorical_summary", {})
    sample_rows  = profile.get("sample_rows", [])[:5]   # keep prompt small

    col_descriptions = []
    for c in columns:
        dtype = dtypes.get(c, "unknown")
        if c in num_summary:
            s = num_summary[c]
            col_descriptions.append(
                f"  {c!r} [numeric] — min={s['min']:.2g}, max={s['max']:.2g}, mean={s['mean']:.2g}"
            )
        elif c in cat_summary:
            top_vals = [item["value"] for item in cat_summary[c][:5]]
            col_descriptions.append(
                f"  {c!r} [categorical] — top values: {top_vals}"
            )
        else:
            col_descriptions.append(f"  {c!r} [{dtype}]")

    col_block    = "\n".join(col_descriptions)
    sample_block = json.dumps(sample_rows, default=str, indent=2) if sample_rows else "N/A"

    return f"""You are a data visualisation expert. Analyse this dataset profile and decide:

1. Is this a TIME-SERIES dataset (has a date/time column that progresses over time) or a CROSS-SECTIONAL dataset (snapshot, no meaningful time axis)?
2. What is the single best chart to show the most important insight?

## Dataset profile
Columns:
{col_block}

Sample rows (first 5):
{sample_block}

## Your task
Return ONLY a JSON object — no markdown, no explanation outside the JSON — with these keys:

{{
  "dataset_type":  "time_series" | "cross_sectional",
  "chart_type":    "horizontal_bar" | "bar" | "scatter" | "pie" | "donut" | "histogram",
  "x_col":         "<column name for X axis or category>",
  "y_col":         "<column name for the primary numeric measure>",
  "color_col":     "<column name for colour grouping, or null>",
  "size_col":      "<column name for bubble size (scatter only), or null>",
  "chart_title":   "<short title, max 8 words>",
  "x_label":       "<human-friendly X axis label>",
  "y_label":       "<human-friendly Y axis label>",
  "top_n":         <integer, how many rows to display, typically 10–20>,
  "sort_order":    "desc" | "asc",
  "reason":        "<one sentence: why this chart type and these columns>"
}}

Rules:
- For TIME-SERIES: x_col must be the date/time column; chart_type should almost always be "bar" or "scatter".
- For CROSS-SECTIONAL with a clear ranking (e.g. companies, products, countries): prefer "horizontal_bar".
- For distributions: use "histogram".
- For part-of-whole (≤8 categories): use "pie" or "donut".
- For two numeric variables: use "scatter".
- y_col must be the most meaningful numeric column (avoid IDs, row numbers, or counts that are just row indices).
- x_col for cross-sectional charts should be the most descriptive categorical column (names, labels).
- If no obvious categorical column exists, use the column with the most varied string values.
- Return null (not the string "null") for optional fields that don't apply.
"""


# ── JSON extractor (robust) ───────────────────────────────────────────────────
def _extract_json(text: str) -> dict:
    """Try multiple strategies to pull a JSON object out of `text`."""
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from Groq response: {text[:300]}")


# ── Public API ────────────────────────────────────────────────────────────────
def pick_columns_with_groq(dataset_profile: dict) -> dict:
    """
    Call Groq to decide the best chart configuration for a dataset.

    Parameters
    ----------
    dataset_profile : dict
        Output of ``build_dataset_profile(df)`` from main.py.

    Returns
    -------
    dict
        Keys: dataset_type, chart_type, x_col, y_col, color_col, size_col,
              chart_title, x_label, y_label, top_n, sort_order, reason.
    """
    try:
        client = _get_client()
        prompt = _build_prompt(dataset_profile)

        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 512,
        )

        raw_text = response.choices[0].message.content or ""
        logger.debug(f"Groq column_picker raw response: {raw_text[:500]}")

        result = _extract_json(raw_text)

        # ── Validate & fill defaults ──────────────────────────────────────────
        allowed_chart_types   = {"horizontal_bar", "bar", "scatter", "pie", "donut", "histogram"}
        allowed_dataset_types = {"time_series", "cross_sectional"}

        if result.get("dataset_type") not in allowed_dataset_types:
            result["dataset_type"] = "cross_sectional"

        if result.get("chart_type") not in allowed_chart_types:
            result["chart_type"] = "horizontal_bar"

        result.setdefault("color_col",   None)
        result.setdefault("size_col",    None)
        result.setdefault("chart_title", "Dataset Overview")
        result.setdefault("x_label",     result.get("x_col") or "")
        result.setdefault("y_label",     result.get("y_col") or "")
        result.setdefault("top_n",       20)
        result.setdefault("sort_order",  "desc")
        result.setdefault("reason",      "")

        # Convert JSON null strings to Python None
        for key in ("x_col", "y_col", "color_col", "size_col"):
            if result.get(key) in ("null", "None", ""):
                result[key] = None

        logger.info(
            f"column_picker result — type={result['dataset_type']}, "
            f"chart={result['chart_type']}, x={result['x_col']}, y={result['y_col']}, "
            f"color={result['color_col']}, reason={result['reason']}"
        )
        return result

    except EnvironmentError as e:
        logger.error(f"column_picker env error: {e}")
        return {**_FALLBACK, "reason": str(e)}

    except Exception as e:
        logger.error(f"column_picker failed: {e}", exc_info=True)
        return {**_FALLBACK, "reason": f"Groq call failed: {e}"}
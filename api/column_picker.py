# api/column_picker.py
import os
import json
import logging

logger = logging.getLogger(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

COLUMN_PICKER_SYSTEM = """You are a data visualization expert. 
Given column names, types, and sample data from any CSV file, 
decide the best way to visualize and analyze it.

Step 1 — Classify dataset type:
- "time_series": rows represent time progression (has date col OR sequential integer id/order/row)
- "cross_sectional": rows are entities (companies, people, products, countries) with attributes
- "distribution": single numeric column to show spread (histogram)
- "comparison": comparing categories across a metric

Step 2 — Pick the best chart:
- time_series → "forecast" (run ML forecast)
- cross_sectional with many rows → "horizontal_bar" (ranked by top metric)
- cross_sectional with few rows (<8) → "bar"
- two numeric columns → "scatter"
- one numeric + categories → "bar" or "horizontal_bar"
- proportions/shares → "pie" (only if <=8 categories)
- single numeric distribution → "histogram"
- multiple time series → "line"

Step 3 — Pick columns:
- x_col: label/category column (name, ticker, region, product) OR date for time series
- y_col: primary numeric metric (highest business value: revenue, price, profit, sales, market_cap)
- color_col: optional grouping column (sector, category, region) — null if not useful
- size_col: optional size for scatter plots — null if not needed

Return ONLY valid JSON, no explanation:
{
  "dataset_type": "time_series|cross_sectional|distribution|comparison",
  "chart_type": "forecast|bar|horizontal_bar|scatter|histogram|pie|line",
  "x_col": "column_name",
  "y_col": "column_name", 
  "color_col": null_or_column_name,
  "size_col": null_or_column_name,
  "top_n": 20,
  "sort_order": "desc",
  "chart_title": "descriptive title for this chart",
  "x_label": "human readable x axis label",
  "y_label": "human readable y axis label",
  "reason": "one sentence why this chart was chosen"
}

Critical rules:
- NEVER pick URL columns, ID columns with all unique values, or columns >50% null
- For time_series: x_col = the date column name
- For cross_sectional: x_col = best label (name/ticker/symbol/country), y_col = best numeric KPI  
- color_col only if it has 2-10 unique values and adds meaningful grouping
- top_n = how many rows to show (10-30 for bar charts, all for scatter/histogram)
- chart_title should describe what the chart shows, e.g. "S&P 500 Stocks Ranked by Market Cap"
"""


def pick_columns_with_groq(df_profile: dict) -> dict:
    """
    Ask Groq to classify any dataset and return a complete chart config.
    Works for any CSV — stocks, sales, surveys, IoT, demographics, anything.
    """
    if not GROQ_API_KEY:
        return _fallback_column_picker(df_profile)

    # Build compact but rich column summary
    col_lines = []
    numeric_stats = df_profile.get("numeric_summary", {})
    cat_stats = df_profile.get("categorical_summary", {})
    all_cols = df_profile.get("columns", [])

    for col, stats in numeric_stats.items():
        col_lines.append(
            f"  {col} [numeric]: min={stats['min']:.2f}, max={stats['max']:.2f}, "
            f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, count={stats['count']}"
        )
    for col, vals in cat_stats.items():
        top_vals = [v["value"] for v in vals[:5]]
        nunique = len(vals)
        col_lines.append(
            f"  {col} [categorical, {nunique} unique]: sample={top_vals}"
        )

    # Include sample rows — critical for Groq to understand row = entity vs row = time point
    sample_rows = df_profile.get("sample_rows", [])[:4]
    sample_str = ""
    if sample_rows:
        # Show only first 3 columns of each row to save tokens
        trimmed = [{k: v for i, (k, v) in enumerate(row.items()) if i < 6}
                   for row in sample_rows]
        sample_str = f"\nSample rows (first 4): {json.dumps(trimmed, default=str)[:800]}"

    user_content = (
        f"CSV with {len(all_cols)} columns, {numeric_stats.__len__()} numeric, "
        f"{cat_stats.__len__()} categorical.\n"
        f"Total rows: {df_profile.get('numeric_summary', {}).get(list(numeric_stats.keys())[0] if numeric_stats else '', {}).get('count', 'unknown')}\n"
        f"\nColumns:\n" + "\n".join(col_lines) +
        sample_str +
        "\n\nClassify this dataset and pick the best chart configuration."
    )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": COLUMN_PICKER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        result = json.loads(raw)
        logger.info(f"Groq chart config: {result}")

        # Case-insensitive validation of all column references
        col_map = {c.lower(): c for c in all_cols}
        for field in ["x_col", "y_col", "color_col", "size_col"]:
            val = (result.get(field) or "").strip().lower()
            if val and val in col_map:
                result[field] = col_map[val]   # fix case
            elif val:
                logger.warning(f"Groq picked invalid {field}='{result[field]}' — clearing")
                result[field] = None

        # Ensure required fields have defaults
        result.setdefault("dataset_type", "cross_sectional")
        result.setdefault("chart_type", "horizontal_bar")
        result.setdefault("top_n", 20)
        result.setdefault("sort_order", "desc")
        result.setdefault("chart_title", f"Analysis by {result.get('y_col', 'metric')}")
        result.setdefault("x_label", result.get("x_col", ""))
        result.setdefault("y_label", result.get("y_col", ""))

        # Safety: time_series must have a valid x_col that looks like a date/sequence
        if result["dataset_type"] == "time_series" and not result.get("x_col"):
            logger.warning("time_series but no x_col — downgrading to cross_sectional")
            result["dataset_type"] = "cross_sectional"
            result["chart_type"] = "horizontal_bar"

        logger.info(
            f"Final chart config: type={result['dataset_type']}, "
            f"chart={result['chart_type']}, x={result.get('x_col')}, "
            f"y={result.get('y_col')}, color={result.get('color_col')}"
        )
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Groq column picker failed: {e}. Using fallback.")
        return _fallback_column_picker(df_profile)


def _fallback_column_picker(df_profile: dict) -> dict:
    """
    Pure keyword fallback — no Groq. Handles ~80% of common datasets correctly.
    """
    numeric_cols = list(df_profile.get("numeric_summary", {}).keys())
    cat_cols = list(df_profile.get("categorical_summary", {}).keys())
    all_cols = df_profile.get("columns", [])
    stats = df_profile.get("numeric_summary", {})

    # ── Detect time axis ──
    time_keywords = ["date", "time", "datetime", "timestamp", "ds", "month", "year", "week"]
    seq_keywords = ["id", "order", "index", "seq", "row", "no", "num", "rank"]

    x_col = None
    dataset_type = "cross_sectional"
    chart_type = "horizontal_bar"

    # Check categorical cols for date-like names
    for col in cat_cols + all_cols:
        if any(k in col.lower() for k in time_keywords):
            x_col = col
            dataset_type = "time_series"
            chart_type = "forecast"
            break

    # Check numeric cols for sequential integer
    if not x_col:
        for col in numeric_cols:
            col_s = stats.get(col, {})
            if (any(k in col.lower() for k in seq_keywords)
                    and col_s.get("min", 999) <= 2
                    and col_s.get("std", 0) > 0):
                x_col = col
                dataset_type = "time_series"
                chart_type = "forecast"
                break

    # ── Pick y_col (target metric) ──
    target_keywords = ["price", "sales", "revenue", "profit", "amount", "value",
                       "income", "close", "cap", "earnings", "cost", "total", "count"]
    y_col = None
    for kw in target_keywords:
        for col in numeric_cols:
            if kw in col.lower():
                y_col = col
                break
        if y_col:
            break
    if not y_col and numeric_cols:
        # Pick highest std (most variable = most interesting)
        y_col = max(numeric_cols, key=lambda c: stats.get(c, {}).get("std", 0))

    # ── Pick x_col for cross-sectional ──
    if dataset_type == "cross_sectional":
        label_keywords = ["name", "ticker", "symbol", "company", "region",
                          "country", "product", "category", "segment", "sector", "city"]
        for kw in label_keywords:
            for col in cat_cols:
                if kw in col.lower():
                    x_col = col
                    break
            if x_col:
                break
        if not x_col and cat_cols:
            x_col = cat_cols[0]

    # ── Pick color_col ──
    color_keywords = ["sector", "category", "region", "segment", "type", "group", "industry"]
    color_col = None
    for kw in color_keywords:
        for col in cat_cols:
            if col != x_col and kw in col.lower():
                color_col = col
                break
        if color_col:
            break

    return {
        "dataset_type": dataset_type,
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "color_col": color_col,
        "size_col": None,
        "top_n": 20,
        "sort_order": "desc",
        "chart_title": f"{y_col or 'Metric'} by {x_col or 'Entity'}",
        "x_label": x_col or "",
        "y_label": y_col or "",
        "reason": "Fallback: keyword-based column selection."
    }
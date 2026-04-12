"""
api/forecast_query.py
---------------------
Groq detects forecast-related keywords in ANY user query
and returns a structured forecast_update dict.

This is COMPLETELY SEPARATE from the chat interface.
The flow is:
  1. User types anything in the app (chat or a dedicated query box)
  2. Groq reads the message and the intelligence card
  3. If forecast-related → returns a structured forecast_update
  4. app.py receives forecast_update and re-renders the Forecast tab chart
  5. A small toast/banner appears: "Forecast tab updated for: West region"

No LLM explanation is generated here — just structured routing.
"""

import os
import json
import logging
import time
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─── Keyword-based fast-path (no Groq call needed) ───────────────────────────

_FORECAST_KEYWORDS = [
    "forecast", "predict", "next", "future", "will", "look like",
    "next week", "next month", "coming weeks", "upcoming",
    "projection", "outlook", "trend",
]

_GROUP_KEYWORDS = {
    # region names
    "west": "Region", "east": "Region", "south": "Region",
    "central": "Region", "north": "Region",
    # category names
    "technology": "Category", "furniture": "Category",
    "office supplies": "Category", "office": "Category",
    # segment names
    "consumer": "Segment", "corporate": "Segment",
    "home office": "Segment",
    # generic
    "region": "Region", "category": "Category",
    "segment": "Segment", "product": "Product",
}

_TIME_UNITS = {
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "day": "day", "days": "day",
    "quarter": "quarter", "quarters": "quarter",
}


def _fast_parse(message: str, card: dict) -> dict | None:
    """
    Pure keyword scan — no Groq call.
    Returns a forecast_update dict or None if not forecast-related.
    """
    msg = message.lower().strip()

    # Is this even a forecast question?
    if not any(kw in msg for kw in _FORECAST_KEYWORDS):
        return None

    # Time horizon detection (e.g. "next 3 weeks")
    import re
    horizon_match = re.search(r'(\d+)\s*(week|weeks|month|months|day|days|quarter|quarters)', msg)
    if horizon_match:
        n    = int(horizon_match.group(1))
        unit = _TIME_UNITS.get(horizon_match.group(2), 'week')
        per_week  = card.get('periods_per_week',  1)
        per_month = card.get('periods_per_month', 4)
        multiplier = {
            'week': per_week, 'month': per_month,
            'day': card.get('periods_per_week', 1) * (1/7),
            'quarter': per_month * 3,
        }.get(unit, per_week)
        periods_requested = max(1, int(round(n * multiplier)))
        max_h = card.get('horizon', 8)
        return {
            'type':    'horizon',
            'periods': min(periods_requested, max_h),
            'label':   f'Next {n} {unit}{"s" if n > 1 else ""}',
            'source':  'keyword',
        }

    # Group/segment detection
    for kw, col in _GROUP_KEYWORDS.items():
        if kw in msg:
            return {
                'type':    'group',
                'keyword': kw,
                'column':  col,
                'label':   f'{col}: {kw.title()}',
                'source':  'keyword',
            }

    # Generic forecast question with no specific horizon or group
    return {
        'type':    'default',
        'periods': card.get('horizon', 8),
        'label':   f'Next {card.get("horizon", 8)} {card.get("freq_label", "week")}s',
        'source':  'keyword',
    }


def _groq_parse(message: str, card: dict) -> dict | None:
    """
    Groq LLM fallback for ambiguous queries.
    Returns the same forecast_update dict format as _fast_parse.
    Only called when keyword scan is inconclusive.
    """
    if not _groq_client:
        return None

    target    = card.get('target_col', 'metric')
    freq      = card.get('freq_label', 'week')
    groups    = [g['group'] for g in card.get('group_forecasts', [])] if 'group_forecasts' in card else []
    group_str = ', '.join(groups[:8]) if groups else 'none'

    system = (
        "You extract forecast intent from user queries. "
        "Reply ONLY with a JSON object — no prose.\n\n"
        f"Dataset: forecasting '{target}' at {freq}ly frequency.\n"
        f"Available groups: {group_str}.\n\n"
        "JSON schema:\n"
        '{"is_forecast": bool, '
        '"type": "horizon"|"group"|"default"|"none", '
        '"periods": int_or_null, '
        '"group_keyword": str_or_null, '
        '"label": str}'
    )

    for attempt in range(2):
        try:
            resp = _groq_client.chat.completions.create(
                model    = "llama-3.3-70b-versatile",
                messages = [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": message},
                ],
                max_tokens  = 80,
                temperature = 0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(raw)

            if not parsed.get('is_forecast'):
                return None

            t = parsed.get('type', 'default')
            if t == 'horizon' and parsed.get('periods'):
                return {
                    'type':    'horizon',
                    'periods': int(parsed['periods']),
                    'label':   parsed.get('label', f'Forecast'),
                    'source':  'groq',
                }
            if t == 'group' and parsed.get('group_keyword'):
                return {
                    'type':    'group',
                    'keyword': parsed['group_keyword'],
                    'label':   parsed.get('label', parsed['group_keyword'].title()),
                    'source':  'groq',
                }
            return {
                'type':    'default',
                'periods': card.get('horizon', 8),
                'label':   parsed.get('label', 'Forecast'),
                'source':  'groq',
            }

        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(3 * (attempt + 1))
                continue
            logger.warning(f"Groq forecast_query parse failed: {e}")
            return None

    return None


def detect_forecast_intent(message: str, card: dict) -> dict | None:
    """
    Main entry point.
    Returns a forecast_update dict if the message is forecast-related,
    or None if it has nothing to do with forecasting.

    forecast_update schema:
    {
        "type":    "horizon" | "group" | "default",
        "periods": int,           # only for type=horizon
        "keyword": str,           # only for type=group
        "column":  str,           # only for type=group
        "label":   str,           # human label for the notification banner
        "source":  "keyword" | "groq",
    }
    """
    # Fast path first (no API call, instant)
    result = _fast_parse(message, card)
    if result is not None:
        return result

    # Groq fallback for ambiguous phrasing
    if GROQ_API_KEY:
        result = _groq_parse(message, card)

    return result  # None if not forecast-related
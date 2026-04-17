"""
excel_filter.py — Natural-language → DataFrame filter.

extract_filter_criteria()  asks the LLM to parse the user's question into
                           structured filter rules.
apply_filters()            executes those rules on a DataFrame.
format_filter_report()     returns a human-readable summary.
"""

import json
import re

import pandas as pd


# ── LLM prompt ─────────────────────────────────────────────────────────────────

def _build_filter_prompt(question: str, columns: list[str], sample: list[dict]) -> str:
    return f"""You are a data analyst helping filter a spreadsheet.

Available columns: {columns}
Sample rows (first 3): {json.dumps(sample, indent=2, default=str)}

User request: "{question}"

Extract ALL filter conditions from the request.

Reply ONLY with valid JSON — no explanation, no markdown fences:
{{
  "filters": [
    {{
      "column": "<exact column name from the list above>",
      "type": "<month|year|date_range|contains|equals|gt|lt|gte|lte>",
      "value": <number, string, or ["YYYY-MM-DD","YYYY-MM-DD"] for date_range>
    }}
  ],
  "description": "<one sentence: what was filtered and why>"
}}

Filter type reference:
  month      — value is month number 1-12  (e.g. March → 3)
  year       — value is a 4-digit year
  date_range — value is ["YYYY-MM-DD", "YYYY-MM-DD"]
  contains   — case-insensitive substring match; value is a string
  equals     — exact match; value is a string or number
  gt/lt/gte/lte — numeric comparison; value is a number

Only include columns that exist in the list above.
If nothing to filter, return {{"filters": [], "description": "No filter applied"}}."""


# ── LLM extraction ─────────────────────────────────────────────────────────────

def extract_filter_criteria(
    question: str,
    columns: list[str],
    df_sample: list[dict],
    llm,
) -> dict:
    """
    Ask the LLM to parse the user's question into structured filter criteria.
    Returns a dict with keys 'filters' and 'description'.
    Falls back to an empty filter dict on any failure.
    """
    prompt = _build_filter_prompt(question, columns, df_sample)
    try:
        response = llm.invoke(prompt).content.strip()
        response = re.sub(r"```[a-z]*\n?", "", response).strip()
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as exc:  # noqa: BLE001
        return {"filters": [], "description": f"LLM filter extraction failed: {exc}"}
    return {"filters": [], "description": "Could not parse filter criteria"}


# ── Filter application ─────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """Apply structured filter criteria to a DataFrame. Skips invalid rules."""
    result = df.copy()

    for f in criteria.get("filters", []):
        col   = f.get("column")
        ftype = f.get("type")
        value = f.get("value")

        if col not in result.columns:
            continue

        try:
            if ftype == "month":
                dates  = pd.to_datetime(result[col], infer_datetime_format=True, errors="coerce")
                result = result[dates.dt.month == int(value)]

            elif ftype == "year":
                dates  = pd.to_datetime(result[col], infer_datetime_format=True, errors="coerce")
                result = result[dates.dt.year == int(value)]

            elif ftype == "date_range":
                dates  = pd.to_datetime(result[col], infer_datetime_format=True, errors="coerce")
                start  = pd.to_datetime(value[0])
                end    = pd.to_datetime(value[1])
                result = result[(dates >= start) & (dates <= end)]

            elif ftype == "contains":
                mask   = result[col].astype(str).str.contains(str(value), case=False, na=False)
                result = result[mask]

            elif ftype == "equals":
                result = result[result[col].astype(str).str.lower() == str(value).lower()]

            elif ftype == "gt":
                result = result[pd.to_numeric(result[col], errors="coerce") > float(value)]

            elif ftype == "lt":
                result = result[pd.to_numeric(result[col], errors="coerce") < float(value)]

            elif ftype == "gte":
                result = result[pd.to_numeric(result[col], errors="coerce") >= float(value)]

            elif ftype == "lte":
                result = result[pd.to_numeric(result[col], errors="coerce") <= float(value)]

        except Exception:  # noqa: BLE001
            continue  # skip bad rules silently

    return result


# ── Report ─────────────────────────────────────────────────────────────────────

def format_filter_report(
    criteria: dict,
    original_count: int,
    filtered_count: int,
    out_path: str,
) -> str:
    sep = "=" * 72
    lines = [
        sep,
        "  FILTER REPORT",
        f"  {criteria.get('description', '')}",
        sep,
        f"  Rows before filter : {original_count}",
        f"  Rows after  filter : {filtered_count}",
        f"  Saved to           : {out_path}",
        sep,
    ]
    for f in criteria.get("filters", []):
        lines.append(f"  Rule: [{f.get('type')}]  {f.get('column')}  →  {f.get('value')}")
    if criteria.get("filters"):
        lines.append(sep)
    return "\n".join(lines)

"""
excel_filter.py — Natural-language → DataFrame filter.

extract_filter_criteria()  asks the LLM to parse the user's question into
                           structured filter rules.
correct_criteria()         post-processing hard rules that fix common LLM
                           mistakes (e.g. month name → type:contains instead
                           of type:month).
apply_filters()            executes the corrected rules on a DataFrame.
format_filter_report()     returns a human-readable summary.
"""

import json
import re

import pandas as pd


# ── Month lookup (all 12 months, every common format) ─────────────────────────

# Maps any recognisable month token → month number 1-12
MONTH_MAP: dict[str, int] = {
    # English full
    "january": 1,  "february": 2,  "march": 3,    "april": 4,
    "may": 5,      "june": 6,      "july": 7,      "august": 8,
    "september": 9,"october": 10,  "november": 11, "december": 12,
    # English abbreviation
    "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
    "jun": 6,  "jul": 7,  "aug": 8,  "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Chinese
    "一月": 1,  "二月": 2,  "三月": 3,  "四月": 4,
    "五月": 5,  "六月": 6,  "七月": 7,  "八月": 8,
    "九月": 9,  "十月": 10, "十一月": 11,"十二月": 12,
    # Zero-padded numbers as strings ("03" → 3)
    "01": 1,  "02": 2,  "03": 3,  "04": 4,
    "05": 5,  "06": 6,  "07": 7,  "08": 8,
    "09": 9,  "10": 10, "11": 11, "12": 12,
    # Plain numbers as strings ("3" → 3)
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9,
}

# Columns whose name suggests they hold date/time values
_DATE_COL_RE = re.compile(r"date|time|日期|时间|dated", re.IGNORECASE)


def _parse_month(value: str) -> int | None:
    """Return 1-12 if value looks like a month, else None."""
    return MONTH_MAP.get(str(value).strip().lower())


def _is_date_column(col_name: str) -> bool:
    return bool(_DATE_COL_RE.search(col_name))


# ── LLM prompt ─────────────────────────────────────────────────────────────────

def _build_filter_prompt(question: str, columns: list[str], sample: list[dict]) -> str:
    return f"""You are a data analyst helping filter a spreadsheet.

Available columns: {columns}
Sample rows (first 3): {json.dumps(sample, indent=2, default=str)}

User request: "{question}"

Extract ALL filter conditions from the request and return ONLY valid JSON.
No explanation, no markdown fences.

{{
  "filters": [
    {{
      "column": "<exact column name from the list above>",
      "type": "<month|year|date_range|contains|equals|gt|lt|gte|lte>",
      "value": <number, string, or ["YYYY-MM-DD","YYYY-MM-DD"] for date_range>
    }}
  ],
  "description": "<one sentence summary>"
}}

Filter type rules — follow these EXACTLY:
  month      — use when user mentions a month name or number. value MUST be an integer 1-12.
               January=1, February=2, March=3, April=4, May=5, June=6,
               July=7, August=8, September=9, October=10, November=11, December=12
               一月=1, 二月=2, 三月=3, 四月=4, 五月=5, 六月=6,
               七月=7, 八月=8, 九月=9, 十月=10, 十一月=11, 十二月=12
  year       — use when user mentions a specific year. value is a 4-digit integer.
  date_range — use when user gives a date range. value is ["YYYY-MM-DD", "YYYY-MM-DD"].
  contains   — substring match for text columns (NOT for dates). value is a string.
  equals     — exact match. value is a string or number.
  gt/lt/gte/lte — numeric comparison. value is a number.

IMPORTANT: For date columns, NEVER use type "contains". Use "month", "year", or "date_range".

Examples:
  "show me March transactions"
  → {{"filters": [{{"column": "Transaction Date", "type": "month", "value": 3}}], "description": "Filter March transactions"}}

  "filter by 三月"
  → {{"filters": [{{"column": "Transaction Date", "type": "month", "value": 3}}], "description": "Filter March (month 3) transactions"}}

  "transactions over $500"
  → {{"filters": [{{"column": "Amount", "type": "gt", "value": 500}}], "description": "Filter transactions over $500"}}

  "Netflix payments in March"
  → {{"filters": [{{"column": "Transaction Date", "type": "month", "value": 3}}, {{"column": "Description", "type": "contains", "value": "Netflix"}}], "description": "Filter March Netflix payments"}}

Only include columns that exist in: {columns}
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


# ── Post-processing: hard-rule corrections ─────────────────────────────────────

def correct_criteria(criteria: dict, df: pd.DataFrame) -> dict:
    """
    Fix common LLM mistakes using deterministic hard rules.
    Runs AFTER extract_filter_criteria, BEFORE apply_filters.

    Rules applied:
      1. Date column + type=contains + value looks like a month
         → force type=month, value=int
      2. Date column + type=contains + value looks like a year (4 digits)
         → force type=year, value=int
      3. Any filter whose value is a month name/number string on a date column
         → normalise to type=month regardless of original type
    """
    date_cols = {col for col in df.columns if _is_date_column(col)}
    corrected_filters = []

    for f in criteria.get("filters", []):
        col   = f.get("column", "")
        ftype = f.get("type", "")
        value = str(f.get("value", ""))

        if col in date_cols:
            # Rule 1 & 3: value is a recognisable month → force type=month
            month_num = _parse_month(value)
            if month_num is not None and ftype != "month":
                print(f"  [correct_criteria] '{col}' type:{ftype!r} value:{value!r} "
                      f"→ corrected to type:month value:{month_num}")
                f = {**f, "type": "month", "value": month_num}

            # Rule 2: value looks like a 4-digit year → force type=year
            elif re.fullmatch(r"20\d{2}|19\d{2}", value) and ftype not in ("year", "date_range"):
                print(f"  [correct_criteria] '{col}' type:{ftype!r} value:{value!r} "
                      f"→ corrected to type:year value:{int(value)}")
                f = {**f, "type": "year", "value": int(value)}

        corrected_filters.append(f)

    return {**criteria, "filters": corrected_filters}


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
                dates  = pd.to_datetime(result[col], dayfirst=False, errors="coerce")
                result = result[dates.dt.month == int(value)]

            elif ftype == "year":
                dates  = pd.to_datetime(result[col], dayfirst=False, errors="coerce")
                result = result[dates.dt.year == int(value)]

            elif ftype == "date_range":
                dates  = pd.to_datetime(result[col], dayfirst=False, errors="coerce")
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
            continue

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

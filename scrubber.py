"""
scrubber.py — Sensitive information redactor for CSV / Excel bank statements.

Detection strategy (two layers, both run and results are merged):

  Layer 1 — LLM analysis (when an llm is supplied)
      Feed column names + a few sample rows to the LLM.
      It returns:
        • sensitive_columns  — columns to fully redact
        • sensitive_patterns — free-text descriptions of value-level patterns
                               the LLM spotted (e.g. "9-digit tax file numbers")
      The LLM decision supersedes the regex defaults for column classification.

  Layer 2 — Regex fallback (always active)
      Hard-coded patterns catch anything the LLM might miss and make the
      standalone CLI (no llm) fully functional.

Usage (standalone, regex-only):
    python scrubber.py /path/to/statement.csv

Usage (from agent, with LLM):
    process_file(path, llm=llm)
"""

import json
import os
import re
import sys
from pathlib import Path, PureWindowsPath, PurePosixPath

import pandas as pd


# ── Path normalisation ─────────────────────────────────────────────────────────

def _detect_and_normalize_path(raw: str) -> str:
    """
    Auto-detect Windows vs Linux/Mac path format and return a Path object
    usable on the current OS.

    Examples:
        "C:\\Users\\micha\\file.csv"   → Windows absolute
        "C:/Users/micha/file.csv"      → Windows absolute (mixed slashes)
        "/home/user/file.csv"          → Unix absolute
        "./data/file.csv"              → relative (either OS)
        "test_run_data/file.csv"       → relative (either OS)
    """
    raw = raw.strip().strip("\"'")

    # Windows absolute path: starts with a drive letter  C:\ or C:/
    if re.match(r"^[A-Za-z]:[/\\]", raw):
        return str(Path(PureWindowsPath(raw)))

    # Git Bash / MSYS2 Windows path: /c/Users/... → C:\Users\...
    if re.match(r"^/[A-Za-z]/", raw):
        drive = raw[1].upper()          # 'c' → 'C'
        rest = raw[2:].replace("/", "\\")  # /Users/micha/... → \Users\micha\...
        return str(Path(PureWindowsPath(f"{drive}:{rest}")))

    # Unix absolute path: starts with /
    if raw.startswith("/"):
        return str(Path(PurePosixPath(raw)))

    # Relative path — works on both OSes as-is
    return str(Path(raw))


# ── Regex fallback patterns ────────────────────────────────────────────────────

_SENSITIVE_COLUMN_RE = re.compile(
    r"account(?![\s_\-]?type)[\s_\-]?(no|num|number)?"
    r"|acct"
    r"|bsb"
    r"|sort[\s_\-]?code"
    r"|card[\s_\-]?(no|num|number)?"
    r"|credit[\s_\-]?card"
    r"|debit[\s_\-]?card"
    r"|iban"
    r"|swift"
    r"|routing"
    r"|ssn"
    r"|social[\s_\-]?security"
    r"|tax[\s_\-]?(file|id|number)"
    r"|tfn|abn|acn"
    r"|password|pin|cvv|cvc",
    re.IGNORECASE,
)

# Value-level patterns: (compiled_regex, human_label)
_VALUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "credit/debit card"),
    (re.compile(r"\b\d{3}[\s\-]\d{3}\b"),                            "BSB"),
    (re.compile(r"(?<!\d)\d{6,12}(?!\d)"),                           "account number"),
]

REDACT = "[REDACTED]"

# Columns that should never be redacted regardless of what the LLM says
_SAFE_COLUMN_RE = re.compile(
    r"^date$|^time$|^timestamp$|^datetime$"
    r"|^amount$|^balance$|^debit$|^credit$|^value$|^total$"
    r"|^currency$|^ccy$"
    r"|^description$|^desc$|^details$|^narrative$|^memo$|^note$|^remarks$"
    r"|^category$|^type$|^transaction[\s_]?type$",
    re.IGNORECASE,
)


def _is_safe_column(name: str) -> bool:
    return bool(_SAFE_COLUMN_RE.search(name.strip()))


# ── LLM-based column / pattern detection ──────────────────────────────────────

def _build_llm_prompt(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    # Send up to 5 sample rows, converting to plain dicts for readability
    sample = df.head(5).astype(str).to_dict(orient="records")

    return f"""You are a data-privacy expert auditing a bank statement file before it is shared.

Column names:
{columns}

Sample rows (first {len(sample)}):
{json.dumps(sample, indent=2)}

Task:
Identify columns that contain DIRECTLY IDENTIFYING sensitive data that must be redacted.

SENSITIVE — redact these:
- Account numbers, BSB / sort codes, IBAN, routing numbers
- Credit / debit card numbers
- PIN, password, CVV
- Government IDs: SSN, TFN, passport number
- Full person names (first + last name together)
- Phone numbers, email addresses, physical addresses

NOT SENSITIVE — do NOT redact these:
- Dates and timestamps
- Transaction amounts and balances
- General text descriptions of transactions (e.g. "Coffee", "Salary", "Netflix")
- Merchant / payee category labels
- Currency codes

Reply with ONLY valid JSON — no explanation, no markdown fences:
{{
  "sensitive_columns": ["<exact column name>", ...],
  "value_patterns_found": ["<plain-English description of sensitive values hidden inside non-obvious columns>", ...],
  "reasoning": "<one short sentence>"
}}

Only reference column names that exist in: {columns}
If nothing is sensitive, return empty lists."""


def _parse_llm_response(text: str) -> tuple[list[str], list[str]]:
    """Extract sensitive_columns and value_patterns_found from LLM JSON output."""
    # Strip markdown fences if present
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    # Find the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return [], []
    try:
        data = json.loads(m.group())
        cols = [str(c) for c in data.get("sensitive_columns", [])]
        patterns = [str(p) for p in data.get("value_patterns_found", [])]
        return cols, patterns
    except json.JSONDecodeError:
        return [], []


def detect_sensitive_with_llm(df: pd.DataFrame, llm) -> tuple[list[str], list[str]]:
    """
    Ask the LLM to analyse column names + sample data.

    Returns:
        sensitive_cols    — list of column names to fully redact
        value_pattern_notes — human-readable descriptions of value-level
                              patterns spotted (logged in the report)
    """
    prompt = _build_llm_prompt(df)
    try:
        response = llm.invoke(prompt).content.strip()
        return _parse_llm_response(response)
    except Exception as exc:  # noqa: BLE001
        # If the LLM call fails for any reason, fall back silently to regex
        return [], [f"(LLM detection failed: {exc} — regex fallback active)"]


# ── Core scrubbing ─────────────────────────────────────────────────────────────

def _regex_sensitive_column(name: str) -> bool:
    return bool(_SENSITIVE_COLUMN_RE.search(name.strip()))


def _redact_string(value: str) -> tuple[str, list[str]]:
    """Apply value-level regex patterns; return (new_value, [matched labels])."""
    reasons: list[str] = []
    v = value
    for pattern, label in _VALUE_PATTERNS:
        if pattern.search(v):
            v = pattern.sub(REDACT, v)
            reasons.append(label)
    return v, reasons


def scrub_dataframe(
    df: pd.DataFrame,
    llm_sensitive_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Redact sensitive data from df.

    llm_sensitive_cols — column names flagged by the LLM (may be None / empty).
                         Merged with regex-detected columns; union wins.

    Returns (scrubbed_df, changes) where changes is a list of dicts:
        row, column, original, redacted, reason
    """
    llm_cols_set = set(llm_sensitive_cols or [])

    # Convert to object dtype so string placeholders can replace numeric cells
    scrubbed = df.astype(object).copy()
    changes: list[dict] = []

    for col in df.columns:
        # Whitelist overrides everything — never redact safe columns
        if _is_safe_column(col):
            continue

        # Column is sensitive if LLM said so OR regex matches the name
        col_is_sensitive = col in llm_cols_set or _regex_sensitive_column(col)
        source = []
        if col in llm_cols_set:
            source.append("LLM")
        if _regex_sensitive_column(col):
            source.append("regex")
        col_reason = f"sensitive column [{'/'.join(source)}]" if col_is_sensitive else ""

        for pos, idx in enumerate(df.index):
            cell = df.at[idx, col]

            if pd.isna(cell):
                continue

            original_str = str(cell)

            if col_is_sensitive:
                new_val = REDACT
                reason = col_reason
            else:
                new_val, reasons = _redact_string(original_str)
                reason = ", ".join(reasons)

            if new_val != original_str:
                scrubbed.at[idx, col] = new_val
                changes.append(
                    {
                        "row": pos,          # always an int (0-based position)
                        "column": col,
                        "original": original_str,
                        "redacted": new_val,
                        "reason": reason,
                    }
                )

    return scrubbed, changes


# ── Diff display ───────────────────────────────────────────────────────────────

def format_diff(
    changes: list[dict],
    file_path: str,
    out_path: str,
    llm_notes: list[str] | None = None,
    detection_mode: str = "regex",
) -> str:
    sep = "=" * 72
    lines = [
        sep,
        "  REDACTION REPORT",
        f"  Detection : {detection_mode}",
        f"  Input     : {file_path}",
        f"  Output    : {out_path}",
    ]

    if llm_notes:
        for note in llm_notes:
            lines.append(f"  LLM note  : {note}")

    lines.append(sep)

    if not changes:
        lines.append("  No sensitive information detected. File saved unchanged.")
        lines.append(sep)
        return "\n".join(lines)

    lines.append(f"  {len(changes)} redaction(s) applied\n")
    lines.append(f"{'Row':<6}  {'Column':<28}  {'Original':>22}   {'Redacted':<12}  Reason")
    lines.append(f"{'-'*6}  {'-'*28}  {'-'*22}   {'-'*12}  {'-'*30}")

    for c in changes:
        orig = c["original"]
        display = orig if len(orig) <= 22 else orig[:19] + "..."
        lines.append(
            f"{str(c['row'] + 2):<6}  {c['column']:<28}  "
            f"{display!r:>22} → {c['redacted']:<12}  [{c['reason']}]"
        )

    lines.append(sep)
    return "\n".join(lines)


# ── Public entry point ─────────────────────────────────────────────────────────

def process_file(file_path: str, llm=None) -> str:
    """
    Read file_path, scrub sensitive data, save *_scrubbed copy, return report.

    llm — optional LangChain-compatible LLM instance.
          When supplied, the LLM analyses columns + sample rows first,
          then regex runs as an additional safety net.
          When None, regex-only mode is used (suitable for CLI use).
    """
    file_path = _detect_and_normalize_path(file_path)

    if not os.path.exists(file_path):
        return f"Error: file not found — {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)

    # If pandas used a non-default index (e.g. CSV header has N cols but data has
    # N+1 → first data column becomes the row index with repeated string values),
    # reset to a clean RangeIndex so df.at[idx, col] always returns a scalar.
    # If pandas auto-promoted the first data column to the row index
    # (happens when data rows have one more field than the header, e.g. trailing
    # commas), re-read with index_col=False so every data field maps to the
    # correct header column, then drop any trailing unnamed/empty columns.
    if not isinstance(df.index, pd.RangeIndex):
        if ext == ".csv":
            df = pd.read_csv(file_path, index_col=False)
        else:
            df = pd.read_excel(file_path, index_col=False)
        # Drop auto-generated "Unnamed: N" columns produced by trailing commas
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    else:
        return (
            f"Unsupported file type '{ext}'. "
            "Please provide a .csv, .xlsx, or .xls file."
        )

    # ── Detection ──────────────────────────────────────────────
    llm_cols: list[str] = []
    llm_notes: list[str] = []
    detection_mode = "regex only"

    if llm is not None:
        print("  [scrubber] Asking LLM to analyse columns and sample data…")
        llm_cols, llm_notes = detect_sensitive_with_llm(df, llm)
        detection_mode = "LLM + regex"
        if llm_cols:
            print(f"  [scrubber] LLM flagged columns: {llm_cols}")

    # ── Scrub ──────────────────────────────────────────────────
    scrubbed_df, changes = scrub_dataframe(df, llm_sensitive_cols=llm_cols)

    # ── Save ───────────────────────────────────────────────────
    base = os.path.splitext(file_path)[0]
    out_path = base + "_scrubbed" + ext
    if ext == ".csv":
        scrubbed_df.to_csv(out_path, index=False)
    else:
        scrubbed_df.to_excel(out_path, index=False, engine="openpyxl")

    return format_diff(changes, file_path, out_path, llm_notes, detection_mode)


# ── Standalone CLI (regex-only) ───────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrubber.py <path_to_csv_or_excel>")
        sys.exit(1)
    print(process_file(sys.argv[1]))

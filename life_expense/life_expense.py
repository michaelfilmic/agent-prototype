"""
life_expense.py — Personal expense pipeline.

Steps (applied to every input file in order):
  1. Scrub    — redact sensitive columns (account numbers, card numbers, etc.)
  2. Filter   — keep only rows whose date falls in the requested calendar month
  3. Normalise— rename amount / date / description columns to standard names
  4. Combine  — stack all files into one DataFrame aligned on "Amount"
  5. Report   — group by expense category, sort & compute % of total spend

Usage:
    python life_expense.py --month 3 file_a.csv file_b.csv
    python life_expense.py -month=3 file_a.csv file_b.csv   (single-dash also OK)

Output (written to <repo_root>/out_test/):
    life_expense_month_<N>_combined.csv   — every transaction after scrub + filter
    life_expense_month_<N>_summary.csv    — category breakdown with percentages
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

# ── Repo root on sys.path so excel_utils is importable ────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from excel_utils.scrubber import scrub_dataframe
from excel_utils.excel_filter import apply_filters

SEP      = "=" * 68
OUT_DIR  = Path(__file__).parent.parent / "out_test"

# ── Column auto-detection patterns ────────────────────────────────────────────

_DATE_RE   = re.compile(r"date|time|dated|trans", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"cad\$|usd\$|amount|value|debit|credit|total", re.IGNORECASE)
_DESC_RE   = re.compile(r"description\s*1|description$|category|merchant|payee|narrative", re.IGNORECASE)


def _detect_col(df: pd.DataFrame, pattern: re.Pattern) -> str | None:
    """Return first column name matching pattern, or None."""
    for col in df.columns:
        if pattern.search(col):
            return col
    return None


def _detect_amount_col(df: pd.DataFrame) -> str | None:
    """
    Pick the amount column with the most non-null numeric values.
    Prefers CAD$ > USD$ > anything else matching _AMOUNT_RE.
    """
    candidates = [c for c in df.columns if _AMOUNT_RE.search(c)]
    if not candidates:
        return None
    # Score each by count of non-null numeric cells
    def score(col: str) -> int:
        return pd.to_numeric(df[col], errors="coerce").notna().sum()
    return max(candidates, key=score)


# ── Step 1: Scrub ──────────────────────────────────────────────────────────────

def _scrub(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    scrubbed, changes = scrub_dataframe(df)
    print(f"  [scrub]  {label}: {len(changes)} cell(s) redacted")
    return scrubbed, len(changes)


# ── Step 2: Month filter ───────────────────────────────────────────────────────

def _filter_month(df: pd.DataFrame, month: int, label: str) -> pd.DataFrame:
    date_col = _detect_col(df, _DATE_RE)
    if not date_col:
        print(f"  [filter] {label}: no date column detected — skipping filter")
        return df

    criteria = {
        "filters": [{"column": date_col, "type": "month", "value": month}],
        "description": f"month={month}",
    }
    filtered = apply_filters(df, criteria)
    print(f"  [filter] {label}: {len(df)} -> {len(filtered)} rows  (month={month}, col='{date_col}')")
    return filtered


# ── Step 3: Normalise ──────────────────────────────────────────────────────────

STD_DATE   = "Date"
STD_AMOUNT = "Amount"
STD_DESC   = "Description"
STD_SOURCE = "Source File"


def _normalise(df: pd.DataFrame, source_name: str) -> pd.DataFrame | None:
    """
    Map detected columns to standard names and keep only relevant columns.
    Returns None if mandatory columns (amount) cannot be found.
    """
    amount_col = _detect_amount_col(df)
    date_col   = _detect_col(df, _DATE_RE)
    desc_col   = _detect_col(df, _DESC_RE)

    if not amount_col:
        print(f"  [normalise] {source_name}: no amount column found — skipping file")
        return None

    rename = {}
    keep   = []

    if date_col:
        rename[date_col] = STD_DATE
        keep.append(STD_DATE)

    if desc_col:
        rename[desc_col] = STD_DESC
        keep.append(STD_DESC)

    rename[amount_col] = STD_AMOUNT
    keep.append(STD_AMOUNT)

    out = df.rename(columns=rename)[keep].copy()
    out[STD_AMOUNT] = pd.to_numeric(out[STD_AMOUNT], errors="coerce")
    out[STD_SOURCE] = source_name

    if date_col:
        out[STD_DATE] = pd.to_datetime(out[STD_DATE], dayfirst=False, errors="coerce")

    print(f"  [normalise] {source_name}: mapped '{amount_col}' -> Amount"
          + (f", '{date_col}' -> Date" if date_col else "")
          + (f", '{desc_col}' -> Description" if desc_col else ""))
    return out


# ── Step 4: Combine ────────────────────────────────────────────────────────────

def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [STD_SOURCE]
    if STD_DATE in combined.columns:
        sort_cols.append(STD_DATE)
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


# ── Step 5: Expense report ─────────────────────────────────────────────────────

def _expense_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to expenses (negative Amount), group by Description,
    sum absolute value, sort descending, add Percentage column.
    """
    expenses = df[df[STD_AMOUNT] < 0].copy()
    expenses[STD_AMOUNT] = expenses[STD_AMOUNT].abs()

    if STD_DESC not in expenses.columns or expenses.empty:
        return pd.DataFrame(columns=[STD_DESC, "Total Spent", "Percentage"])

    grouped = (
        expenses.groupby(STD_DESC, as_index=False)[STD_AMOUNT]
        .sum()
        .rename(columns={STD_AMOUNT: "Total Spent"})
        .sort_values("Total Spent", ascending=False)
        .reset_index(drop=True)
    )

    total = grouped["Total Spent"].sum()
    grouped["Percentage"] = (grouped["Total Spent"] / total * 100).round(2)
    return grouped


# ── Read helper ────────────────────────────────────────────────────────────────

def _read(path: str) -> pd.DataFrame | None:
    ext = Path(path).suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path, index_col=False)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            print(f"  [read] Unsupported file type '{ext}': {path}")
            return None
    except Exception as exc:
        print(f"  [read] Failed to read {path}: {exc}")
        return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(month: int, input_files: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"  LIFE EXPENSE PIPELINE  —  Month: {month}")
    print(f"  Files : {len(input_files)}")
    print(SEP)

    normalised_frames: list[pd.DataFrame] = []
    total_scrubbed = 0
    total_filtered = 0

    for fpath in input_files:
        fpath = fpath.strip().strip("\"'")
        name  = Path(fpath).name

        if not os.path.exists(fpath):
            print(f"\n  [skip] File not found: {fpath}")
            continue

        print(f"\n  -- {name}")

        # Step 1 — scrub
        df = _read(fpath)
        if df is None:
            continue
        df, n_scrubbed = _scrub(df, name)
        total_scrubbed += n_scrubbed

        # Step 2 — month filter
        df = _filter_month(df, month, name)
        total_filtered += len(df)

        if df.empty:
            print(f"  [skip] No rows for month={month} in {name}")
            continue

        # Step 3 — normalise
        normed = _normalise(df, name)
        if normed is not None and not normed.empty:
            normalised_frames.append(normed)

    if not normalised_frames:
        print(f"\n  No data found for month={month} across all input files.")
        return

    # Step 4 — combine
    print(f"\n{SEP}")
    print("  STAGE 4 — Combining files")
    print(SEP)
    combined = _combine(normalised_frames)
    print(f"  Total rows combined : {len(combined)}")

    combined_path = OUT_DIR / f"life_expense_month_{month}_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved               : {combined_path}")

    # Step 5 — expense report
    print(f"\n{SEP}")
    print("  STAGE 5 — Expense Summary")
    print(SEP)
    summary = _expense_report(combined)

    if summary.empty:
        print("  No expenses found.")
    else:
        print(summary.to_string(index=False))

    summary_path = OUT_DIR / f"life_expense_month_{month}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n  Total cells scrubbed : {total_scrubbed}")
    print(f"  Total rows in output : {len(combined)}")
    print(f"  Expense categories   : {len(summary)}")
    print(f"  Saved summary        : {summary_path}")
    print(SEP)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _preprocess_argv(argv: list[str]) -> list[str]:
    """Convert -month=3 or -month 3 (single-dash) to --month 3 for argparse."""
    out = []
    for arg in argv:
        if re.match(r"^-month=", arg):
            out.extend(["--month", arg.split("=", 1)[1]])
        elif arg == "-month" or arg == "-m":
            out.append("--month")
        else:
            out.append(arg)
    return out


HELP_TEXT = """
life_expense.py  --  Personal expense pipeline
================================================

USAGE
    python life_expense.py --month <N> <file1> [file2 ...]

    --month / -month    Calendar month to process (1-12).
                        Both formats accepted:
                            --month 3
                            -month=3

    files               One or more CSV or XLSX bank-export files.
                        Separate multiple files with spaces (no commas).

SUPPORTED INPUT FORMATS
    The script auto-detects columns — no config needed.
    It looks for:

    Amount column   : CAD$, USD$, Amount, Value, Debit, Credit, Total
                      (picks whichever has the most non-null numeric values)

    Date column     : any column whose name contains "date", "time", or "trans"
                      Values can be MM/DD/YYYY, YYYY-MM-DD, DD/MM/YYYY, etc.

    Description col : "Description 1", "Description", "Category",
                      "Merchant", "Payee", or "Narrative"
                      (used as the expense category in the summary)

EXAMPLES
    # Single file, March
    python life_expense.py --month 3 statement.csv

    # Multiple files, single-dash month syntax
    python life_expense.py -month=3 chequing.csv savings.csv visa.xlsx

    # Absolute paths
    python life_expense.py --month 12 C:\\Downloads\\bank_a.csv C:\\Downloads\\bank_b.xlsx

OUTPUT  (always written to <repo_root>/out_test/)
    life_expense_month_<N>_combined.csv   all transactions after scrub + filter
    life_expense_month_<N>_summary.csv    expense categories sorted by % of spend

PIPELINE STEPS
    1. Scrub   -- sensitive columns (account numbers, card numbers, BSB, etc.)
                  are redacted to [REDACTED] before any data is written
    2. Filter  -- only rows whose date falls in the requested month are kept
    3. Normalise -- amount / date / description columns renamed to standard names
                    so files with different column names can be combined
    4. Combine -- all files stacked into one table, sorted by date
    5. Summary -- expenses (negative amounts) grouped by category,
                  sorted descending by total spend with % of total
"""


if __name__ == "__main__":
    # Show custom help if -h / --help / -help passed
    if any(a in sys.argv[1:] for a in ("-h", "--help", "-help", "help")):
        print(HELP_TEXT)
        sys.exit(0)

    args_in = _preprocess_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--month", "-month", type=int, required=True)
    parser.add_argument("files", nargs="+")

    try:
        args = parser.parse_args(args_in)
    except SystemExit:
        print(HELP_TEXT)
        sys.exit(1)

    run(args.month, args.files)

"""
expense_report.py — Generate a two-level expense summary from a combined CSV.

Uses up to three description columns to build the summary:
  Primary   — "Description"       (main category, e.g. "Dining")
  Secondary — "Sub-description"   (first sub-detail, e.g. "TIM HORTONS")
  Tertiary  — "Description 2"     (second sub-detail, used when secondary is absent)

Output (CSV):
    Category | Sub Description | Total Spent | % of Category | % of Total
    Rows are sorted by category total (desc), then sub-description total (desc).
    A TOTAL row is appended at the bottom.

Terminal output shows the same data in a readable grouped hierarchy.

Usage:
    python expense_report.py <combined.csv> [output.csv]

    If output.csv is omitted the summary is saved alongside the input as
    <stem>_summary.csv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

STD_AMOUNT = "Amount"
STD_DESC   = "Description"

# Column names to probe for secondary description, in priority order
_SUB_DESC_CANDIDATES = ["Sub-description", "Sub description", "sub_description",
                         "Description 2", "Merchant", "Payee", "Narrative"]

COL_CATEGORY    = "Category"
COL_SUB         = "Sub Description"
COL_SPENT       = "Total Spent"
COL_PCT_CAT     = "% of Category"
COL_PCT_TOTAL   = "% of Total"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_sub_desc_col(df: pd.DataFrame) -> str | None:
    """Return the first matching sub-description column present in df."""
    for name in _SUB_DESC_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _coalesce_sub(row: pd.Series, cols: list[str]) -> str:
    """Return the first non-empty value across the given columns."""
    for c in cols:
        val = str(row.get(c, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            return val
    return "(no detail)"


# ── Core ───────────────────────────────────────────────────────────────────────

def generate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a two-level expense summary.

    Parameters
    ----------
    df : DataFrame with at least an 'Amount' column; ideally also 'Description'
         and one of the sub-description columns listed in _SUB_DESC_CANDIDATES.

    Returns
    -------
    Flat DataFrame with columns:
        Category | Sub Description | Total Spent | % of Category | % of Total
    Includes a TOTAL row at the bottom.
    """
    if STD_AMOUNT not in df.columns:
        raise ValueError(
            f"Column '{STD_AMOUNT}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Respect Valid column if present: 0 = exclude, 1 (or empty) = include
    if "Valid" in df.columns:
        df = df[pd.to_numeric(df["Valid"], errors="coerce").fillna(1) == 1]

    expenses = df[df[STD_AMOUNT] < 0].copy()
    expenses[STD_AMOUNT] = expenses[STD_AMOUNT].abs()

    if expenses.empty:
        return pd.DataFrame(columns=[COL_CATEGORY, COL_SUB,
                                     COL_SPENT, COL_PCT_CAT, COL_PCT_TOTAL])

    # Fill missing primary description
    if STD_DESC not in expenses.columns:
        expenses[STD_DESC] = "Other"
    else:
        expenses[STD_DESC] = expenses[STD_DESC].fillna("Other").astype(str).str.strip()

    # Determine sub-description: coalesce sub-desc + description-2
    sub_cols = [c for c in _SUB_DESC_CANDIDATES if c in expenses.columns]
    if sub_cols:
        expenses[COL_SUB] = expenses.apply(lambda r: _coalesce_sub(r, sub_cols), axis=1)
    else:
        expenses[COL_SUB] = "(no detail)"

    # Group by category + sub
    grouped = (
        expenses.groupby([STD_DESC, COL_SUB], as_index=False)[STD_AMOUNT]
        .sum()
        .rename(columns={STD_AMOUNT: COL_SPENT, STD_DESC: COL_CATEGORY})
    )

    grand_total = grouped[COL_SPENT].sum()

    # Category-level totals for sorting and % of category
    cat_totals = grouped.groupby(COL_CATEGORY)[COL_SPENT].sum().rename("_cat_total")
    grouped = grouped.join(cat_totals, on=COL_CATEGORY)

    grouped[COL_PCT_CAT]   = (grouped[COL_SPENT] / grouped["_cat_total"] * 100).round(2)
    grouped[COL_PCT_TOTAL] = (grouped[COL_SPENT] / grand_total * 100).round(2)

    # Sort: by category total desc, then sub-description total desc
    grouped = (
        grouped
        .sort_values(["_cat_total", COL_SPENT], ascending=[False, False])
        .drop(columns=["_cat_total"])
        .reset_index(drop=True)
    )

    # Append TOTAL row
    total_row = pd.DataFrame([{
        COL_CATEGORY:  "TOTAL",
        COL_SUB:       "",
        COL_SPENT:     round(grand_total, 2),
        COL_PCT_CAT:   "",
        COL_PCT_TOTAL: 100.0,
    }])
    grouped = pd.concat([grouped, total_row], ignore_index=True)

    return grouped[[COL_CATEGORY, COL_SUB, COL_SPENT, COL_PCT_CAT, COL_PCT_TOTAL]]


# ── Terminal display ───────────────────────────────────────────────────────────

def format_report(summary: pd.DataFrame) -> str:
    """Render the summary as a readable grouped hierarchy for terminal output."""
    sep  = "=" * 68
    sep2 = "-" * 68
    lines = [sep, "  EXPENSE SUMMARY  (by category > sub-description)", sep]

    if summary.empty or (len(summary) == 1 and summary.iloc[0][COL_CATEGORY] == "TOTAL"):
        lines.append("  No expenses found.")
        lines.append(sep)
        return "\n".join(lines)

    rows = summary[summary[COL_CATEGORY] != "TOTAL"]
    total_row = summary[summary[COL_CATEGORY] == "TOTAL"]

    current_cat = None
    for _, row in rows.iterrows():
        if row[COL_CATEGORY] != current_cat:
            if current_cat is not None:
                lines.append("")
            current_cat = row[COL_CATEGORY]
            # Category header: name + % of total
            pct_total = rows[rows[COL_CATEGORY] == current_cat][COL_PCT_TOTAL].sum()
            cat_spent = rows[rows[COL_CATEGORY] == current_cat][COL_SPENT].sum()
            lines.append(f"  {current_cat:<28}  ${cat_spent:>10.2f}   ({pct_total:.2f}% of total)")
            lines.append(f"  {'  Sub Description':<28}  {'Spent':>10}   {'% cat':>6}  {'% total':>7}")
            lines.append(f"  {sep2[2:]}")

        lines.append(
            f"    {str(row[COL_SUB]):<26}  ${row[COL_SPENT]:>10.2f}"
            f"   {str(row[COL_PCT_CAT]):>5}%  {str(row[COL_PCT_TOTAL]):>6}%"
        )

    if not total_row.empty:
        t = total_row.iloc[0]
        lines.append(f"\n{sep}")
        lines.append(f"  {'TOTAL':<28}  ${t[COL_SPENT]:>10.2f}   (100%)")

    lines.append(sep)
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str | None = None) -> str:
    ext = Path(input_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    else:
        return f"Error: unsupported file type '{ext}'."

    summary = generate(df)

    if output_path is None:
        output_path = str(Path(input_path).with_name(
            Path(input_path).stem + "_summary.csv"
        ))

    summary.to_csv(output_path, index=False)

    return format_report(summary) + f"\n  Saved : {output_path}\n"


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(run(sys.argv[1], out))

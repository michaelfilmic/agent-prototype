"""
table_edit.py — DataFrame transformation utilities for position tables.

Functions:
    convert_market_value_to_position_percentage(df, value_col="Value")
        Filters to Summary rows only, computes each position's market value
        as a percentage of total gross exposure, and adds a
        "Position Percentage" column (rounded to 2 dp).
        Note: mixed-currency files are treated as-is (no FX conversion).
              Short positions (negative values) appear as negative percentages.

    table_filter_out(df)
        Keeps only Asset Category, Currency, Symbol, and Position Percentage.
        Raises ValueError if Position Percentage column is missing — call
        convert_market_value_to_position_percentage() first.

Usage (standalone CLI):
    python -m excel_utils.table_edit <input.csv> [output.csv]
"""

from __future__ import annotations

import os
import sys

import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

DISCRIMINATOR_COL  = "DataDiscriminator"
SUMMARY_VALUE      = "Summary"
POSITION_PCT_COL   = "Position Percentage"
KEEP_COLS          = ["Asset Category", "Currency", "Symbol", POSITION_PCT_COL]


# ── Function 1: market value → position percentage ─────────────────────────────

def convert_market_value_to_position_percentage(
    df: pd.DataFrame,
    value_col: str = "Value",
) -> pd.DataFrame:
    """
    Add a 'Position Percentage' column to df.

    Steps:
      1. Keep only rows where DataDiscriminator == 'Summary'
         (drops sub-total / grand-total rows).
      2. Coerce value_col to numeric (non-parseable cells → NaN, then 0).
      3. Compute gross exposure = sum of absolute values across all positions.
      4. Position Percentage = (value / gross_exposure) * 100, rounded to 2 dp.

    Parameters
    ----------
    df        : input DataFrame (from read_csv / read_excel)
    value_col : column name that holds market value (default "Value")

    Returns
    -------
    DataFrame with only Summary rows plus the new Position Percentage column.
    """
    if value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # ── 1. Filter to Summary rows only ────────────────────────────────────────
    if DISCRIMINATOR_COL in df.columns:
        result = df[df[DISCRIMINATOR_COL].astype(str).str.strip() == SUMMARY_VALUE].copy()
        if result.empty:
            raise ValueError(
                f"No rows with {DISCRIMINATOR_COL} == '{SUMMARY_VALUE}' found."
            )
    else:
        # No discriminator column — use all rows (plain table)
        result = df.copy()

    # ── 2. Coerce value column to numeric ─────────────────────────────────────
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce").fillna(0.0)

    # ── 3. Gross exposure (sum of absolute values) ────────────────────────────
    gross_exposure = result[value_col].abs().sum()
    if gross_exposure == 0:
        raise ValueError(
            f"Gross exposure is zero — all values in '{value_col}' are 0 or NaN."
        )

    # ── 4. Position Percentage ────────────────────────────────────────────────
    result[POSITION_PCT_COL] = (
        (result[value_col] / gross_exposure * 100)
        .round(2)
    )

    return result.reset_index(drop=True)


# ── Function 2: filter to key columns ─────────────────────────────────────────

def table_filter_out(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only: Asset Category, Currency, Symbol, Position Percentage.

    Call convert_market_value_to_position_percentage() before this function
    so that the Position Percentage column exists.

    Returns
    -------
    Filtered DataFrame with exactly the four columns above (in that order).
    """
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Run convert_market_value_to_position_percentage() first."
        )
    return (
        df[KEEP_COLS]
        .sort_values(POSITION_PCT_COL, ascending=False)
        .reset_index(drop=True)
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────

def process(input_path: str, output_path: str | None = None) -> str:
    """
    Run both transformations on a CSV/Excel file and save the result.

    Returns a short summary string.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    else:
        return f"Unsupported file type '{ext}'."

    df = convert_market_value_to_position_percentage(df)
    df = table_filter_out(df)

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_positions" + ext

    if ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False, engine="openpyxl")

    lines = [
        f"Input  : {input_path}",
        f"Output : {output_path}",
        f"Rows   : {len(df)}",
        "",
        df.to_string(index=False),
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m excel_utils.table_edit <input.csv> [output.csv]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(process(sys.argv[1], out))

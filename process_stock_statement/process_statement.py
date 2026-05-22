"""
process_statement.py — Two-stage pipeline for IB statement CSV files.

Stage 1 — table_extractor:
    Detects and extracts tables from the raw export.
    Produces one CSV per section, e.g.:
        {stem}_Statement.csv
        {stem}_Open_Positions.csv

Stage 2 — table_edit (Open Positions only):
    Converts market value to position percentage and filters to
    the four key columns (Asset Category, Currency, Symbol, Position %).
    Produces:
        {stem}_Open_Positions_positions.csv

Usage:
    python process_statement.py <input.csv>
    python process_statement.py <input.csv> --out-dir <folder>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add repo root to sys.path so excel_utils can be found regardless of where
# this script is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from excel_utils.table_extractor import extract_tables
from excel_utils.table_edit import convert_market_value_to_position_percentage, table_filter_out

import pandas as pd


SEP = "=" * 68
OPEN_POSITIONS_KEY = "Open_Positions"   # safe_name produced by table_extractor


def _find_open_positions_file(stem: str, out_dir: str) -> str | None:
    """Return the path of the Open Positions CSV if it exists."""
    candidate = os.path.join(out_dir, f"{stem}_{OPEN_POSITIONS_KEY}.csv")
    return candidate if os.path.exists(candidate) else None


def run(input_path: str, out_dir: str | None = None) -> None:
    input_path = input_path.strip().strip("\"'")

    if not os.path.exists(input_path):
        print(f"Error: file not found — {input_path}")
        sys.exit(1)

    stem    = Path(input_path).stem
    out_dir = out_dir or str(Path(input_path).parent)

    # ── Stage 1: table_extractor ──────────────────────────────────────────────
    print(SEP)
    print("  STAGE 1 — Table Extraction")
    print(SEP)
    report = extract_tables(input_path, out_dir)
    print(report)

    # ── Locate Open Positions output ──────────────────────────────────────────
    open_pos_path = _find_open_positions_file(stem, out_dir)
    if not open_pos_path:
        print(
            f"\nNo '{OPEN_POSITIONS_KEY}' table found in the extracted output.\n"
            f"Looked for: {os.path.join(out_dir, stem + '_' + OPEN_POSITIONS_KEY + '.csv')}\n"
            "Skipping Stage 2."
        )
        return

    # ── Stage 2: table_edit ───────────────────────────────────────────────────
    print(SEP)
    print("  STAGE 2 — Position Percentage & Column Filter")
    print(SEP)
    print(f"  Input : {open_pos_path}\n")

    df = pd.read_csv(open_pos_path)
    df = convert_market_value_to_position_percentage(df)
    df = table_filter_out(df)

    out_path = os.path.join(
        out_dir, f"{stem}_{OPEN_POSITIONS_KEY}_positions.csv"
    )
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\n  Saved : {out_path}")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tables from an IB statement CSV, then compute position percentages."
    )
    parser.add_argument("input", help="Path to the raw IB export (.csv)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output files (default: same folder as input)",
    )
    args = parser.parse_args()
    run(args.input, args.out_dir)

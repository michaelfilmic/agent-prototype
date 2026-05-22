"""
table_extractor.py — Detect and extract tables from structured CSV files.

Supports two modes (auto-detected):

  Mode A — Type-column format (e.g. Interactive Brokers exports)
      Column 0: section name  (e.g. "Open Positions", "Statement")
      Column 1: row type      (Header | Data | Total | Notes | ...)
      A section is a TABLE if it contains at least one Header row and one
      Data row.  Metadata-only sections (e.g. key-value "Statement" blocks)
      are discarded.
      Output: one CSV per detected table, named after the section.

  Mode B — Heuristic (plain CSV with mixed content)
      Rows are split into blocks at every blank line.
      A block is a TABLE if:
        • its first row looks like a header (≥ 50 % non-empty string cells)
        • it has at least one data row after the header
      Non-table blocks are dropped.
      Output: all tables written to a single cleaned CSV separated by a
      blank line.

Usage:
    python table_extractor.py <input.csv> [output_dir]

    output_dir defaults to the same directory as input_dir.
    If only one table is found the output file shares the input stem;
    if multiple tables are found they are suffixed _table_1, _table_2, …
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_bom(value: str) -> str:
    return value.lstrip("﻿")


def _looks_like_header(row: list[str]) -> bool:
    """True when the majority of non-empty cells are non-numeric strings."""
    non_empty = [c.strip() for c in row if c.strip()]
    if not non_empty:
        return False
    numeric = sum(1 for c in non_empty if _is_numeric(c))
    return (numeric / len(non_empty)) < 0.4


def _is_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _is_blank_row(row: list[str]) -> bool:
    return all(c.strip() == "" for c in row)


# ── Mode A: type-column format ─────────────────────────────────────────────────

def _detect_type_column_format(rows: list[list[str]]) -> bool:
    """
    Return True when ≥ 60 % of rows have a recognised type keyword in col 1.
    """
    type_keywords = {"header", "data", "total", "notes", "summary"}
    if len(rows) < 2:
        return False
    hits = sum(
        1 for r in rows
        if len(r) > 1 and r[1].strip().lower() in type_keywords
    )
    return hits / len(rows) >= 0.6


def _extract_type_column(rows: list[list[str]]) -> dict[str, list[list[str]]]:
    """
    Group rows by section (col 0).  For each section keep only header+data
    rows and rebuild a clean table (col 0 and col 1 are stripped).

    Returns {section_name: [header_row, data_row, …]}
    Only sections that are genuine tables (have both Header and Data rows)
    are included.
    """
    # group rows by section
    sections: dict[str, list[list[str]]] = {}
    for row in rows:
        if not row:
            continue
        section = _strip_bom(row[0]).strip()
        sections.setdefault(section, []).append(row)

    tables: dict[str, list[list[str]]] = {}

    for section, section_rows in sections.items():
        types_present = {r[1].strip().lower() for r in section_rows if len(r) > 1}
        # Must have both a header and data rows to qualify as a table
        if "header" not in types_present or "data" not in types_present:
            continue

        # Use the FIRST Header row as the column definition
        header_cols: list[str] | None = None
        table_rows: list[list[str]] = []

        for row in section_rows:
            if len(row) < 2:
                continue
            row_type = row[1].strip().lower()
            payload = row[2:]  # strip section + type columns

            if row_type == "header":
                if header_cols is None:
                    header_cols = [c.strip() for c in payload]
                    table_rows.append(header_cols)
                # subsequent Header rows with same shape are separators — skip
            elif row_type in ("data", "total"):
                # Pad / trim to match header width
                if header_cols is not None:
                    clean = [c.strip() for c in payload]
                    # align to header length
                    clean = (clean + [""] * len(header_cols))[: len(header_cols)]
                    table_rows.append(clean)

        if len(table_rows) > 1:  # at least header + one data row
            tables[section] = table_rows

    return tables


# ── Mode B: heuristic plain-CSV ───────────────────────────────────────────────

def _split_blocks(rows: list[list[str]]) -> list[list[list[str]]]:
    """Split rows into blocks at blank lines."""
    blocks: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in rows:
        if _is_blank_row(row):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(row)
    if current:
        blocks.append(current)
    return blocks


def _extract_heuristic(rows: list[list[str]]) -> list[list[list[str]]]:
    """
    Return a list of tables found via heuristic block detection.
    Each table is a list of rows (first row = header).
    """
    blocks = _split_blocks(rows)
    tables: list[list[list[str]]] = []

    for block in blocks:
        if len(block) < 2:
            continue  # single-row block — not a table
        if _looks_like_header(block[0]):
            tables.append(block)

    return tables


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def _write_csv(path: str, rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_tables(input_path: str, output_dir: str | None = None) -> str:
    """
    Detect and extract tables from input_path.
    Returns a human-readable report string.
    """
    if not os.path.exists(input_path):
        return f"Error: file not found — {input_path}"

    rows = _read_csv(input_path)
    if not rows:
        return "Error: file is empty."

    stem = Path(input_path).stem
    out_dir = output_dir or str(Path(input_path).parent)
    report_lines: list[str] = []
    sep = "=" * 68

    report_lines += [sep, "  TABLE EXTRACTOR REPORT", f"  Input : {input_path}", sep]

    # ── choose mode ──────────────────────────────────────────────────────────
    if _detect_type_column_format(rows):
        mode = "Type-column (IB-style)"
        tables_dict = _extract_type_column(rows)
        named_tables: list[tuple[str, list[list[str]]]] = list(tables_dict.items())
    else:
        mode = "Heuristic (plain CSV)"
        raw_tables = _extract_heuristic(rows)
        named_tables = [(f"{stem}_table_{i+1}", t) for i, t in enumerate(raw_tables)]

    report_lines.append(f"  Mode  : {mode}")
    report_lines.append(f"  Tables found: {len(named_tables)}\n")

    if not named_tables:
        report_lines.append("  No tables detected.")
        report_lines.append(sep)
        return "\n".join(report_lines)

    for name, table_rows in named_tables:
        safe_name = name.replace(" ", "_").replace("/", "-")
        if len(named_tables) == 1:
            out_path = os.path.join(out_dir, f"{stem}_extracted.csv")
        else:
            out_path = os.path.join(out_dir, f"{stem}_{safe_name}.csv")

        _write_csv(out_path, table_rows)

        header = table_rows[0] if table_rows else []
        data_rows = len(table_rows) - 1  # exclude header

        report_lines.append(f"  Table : {name!r}")
        report_lines.append(f"    Rows    : {data_rows} data row(s)")
        report_lines.append(f"    Columns : {len(header)}")
        report_lines.append(f"    Headers : {header}")
        report_lines.append(f"    Saved   : {out_path}\n")

    report_lines.append(sep)
    return "\n".join(report_lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <input.csv> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    out_directory = sys.argv[2] if len(sys.argv) > 2 else None
    print(extract_tables(input_file, out_directory))

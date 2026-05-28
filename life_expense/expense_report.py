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
STD_DESC   = "Description"       # mapped from Description 1

# Explicit column names for the three description levels
COL_CATEGORY    = "Category"        # <- standard category (mapped)
COL_DESC        = "Description"     # <- Description 2 (merchant)
COL_SUB         = "Sub Description" # <- Sub-description
COL_SPENT       = "Total Spent"
COL_PCT_CAT     = "% of Category"
COL_PCT_TOTAL   = "% of Total"

# Source column names in the combined CSV
_SRC_DESC2  = "Description 2"
_SRC_SUB    = "Sub-description"

# ── Standard category rules ────────────────────────────────────────────────────
# Each entry: (standard_category, [keyword, ...])
# Matching is case-insensitive against "<Description1> <Description2>" combined.
# First match wins; unmatched rows fall into "Other".

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Income & Deposits",    ["payroll", "salary", "deposit", "direct deposit",
                               "transfer cr", "funds transfer cr", "correction",
                               "rebate", "refund", "cashback", "interest payment",
                               "payment - thank you", "paiement - merci",
                               "monthly fee rebate", "misc payment"]),
    ("Food & Dining",        ["mcdonald", "tim horton", "wendy", "harvey", "a&w",
                               "hot pot", "eatery", "noodle", "doner", "chatime",
                               "bakery", "sushi", "szechuan", "freshii", "poke",
                               "restaurant", "dough", "coco fresh", "saint germain",
                               "aramark", "me va me", "choice of the orient",
                               "bread & cup", "one bowl", "kome", "dining",
                               "burger", "pizza", "cafe", "coffee", "subway",
                               "c-idp purchase", "pos purchase"]),
    ("Groceries",            ["walmart", "wal-mart", "winco food", "loblaws",
                               "metro", "whole foods", "food mart", "grocery",
                               "groceries", "supermarket", "sobeys", "costco",
                               "no frills", "food basics"]),
    ("Transport",            ["petro-canada", "petro canada", "esso", "shell",
                               "gas station", "presto", "transit", "uber",
                               "lyft", "taxi", "parking", "transport", "go train",
                               "ttc", "highway toll"]),
    ("Housing & Utilities",  ["rent", "enercare", "hydro", "enbridge", "utility",
                               "utilities", "mortgage", "property", "home service",
                               "electricity", "water bill", "utility bill"]),
    ("Health",               ["pharmacy", "shoppers", "healthcare", "medical",
                               "drug mart", "health", "dental", "clinic",
                               "hospital", "prescription"]),
    ("Sports & Leisure",     ["volleyball", "javelin", "gametime", "bpnsprts",
                               "entertainment", "cinema", "cineplex", "steam",
                               "sport", "gym", "fitness", "golf", "bowling",
                               "ticket", "concert", "theatre"]),
    ("Subscriptions & Bills",["rogers", "bell", "telus", "netflix", "spotify",
                               "amazon prime", "disney", "apple", "google",
                               "insurance", "monthly fee", "subscription",
                               "internet", "phone bill", "wireless", "cable"]),
    ("Finance & Transfers",  ["www tfr", "www trf", "e-transfer", "email trf",
                               "transfer", "loan pmt", "loan interest", "interest",
                               "payment", "bill payment", "misc payment",
                               "funds transfer", "service charge", "bank fee",
                               "withdrawal", "customer transfer", "monthly fee",
                               "csra"]),
]

OTHER_CATEGORY = "Other"


def _classify(raw_desc1: str, raw_desc2: str) -> str:
    """Map raw description values to a standard category using keyword matching."""
    text = f"{raw_desc1} {raw_desc2}".lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return OTHER_CATEGORY


def _clean(val) -> str:
    """Strip and return empty string for NaN/None."""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none") else s


# ── Core ───────────────────────────────────────────────────────────────────────

def generate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a three-level expense summary.

    Column mapping from combined CSV:
        Category        <- Description   (Description 1 — expense type)
        Description     <- Description 2 (merchant / second label)
        Sub Description <- Sub-description (lowest-level detail)

    Returns
    -------
    DataFrame with columns:
        Category | Description | Sub Description |
        Total Spent | % of Category | % of Total
    Includes a TOTAL row at the bottom.
    """
    if STD_AMOUNT not in df.columns:
        raise ValueError(
            f"Column '{STD_AMOUNT}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop the artificial TOTAL row written by _combine
    if COL_CATEGORY in df.columns:
        df = df[~df[COL_CATEGORY].astype(str).str.startswith("TOTAL")]
    if STD_DESC in df.columns:
        df = df[~df[STD_DESC].astype(str).str.startswith("TOTAL")]

    # Respect Valid column: 0 = exclude, 1 or empty = include
    if "Valid" in df.columns:
        df = df[pd.to_numeric(df["Valid"], errors="coerce").fillna(1) == 1]

    # Keep ALL rows — income, expenses, refunds, zero-amount rows excluded
    expenses = df[df[STD_AMOUNT] != 0].copy()

    if expenses.empty:
        return pd.DataFrame(columns=[COL_CATEGORY, COL_DESC, COL_SUB,
                                     COL_SPENT, COL_PCT_CAT, COL_PCT_TOTAL])

    # ── Map the three description levels ──────────────────────────────────────
    raw_desc1 = (
        expenses[STD_DESC].fillna("").astype(str).str.strip()
        if STD_DESC in expenses.columns else pd.Series("", index=expenses.index)
    )
    raw_desc2 = (
        expenses[_SRC_DESC2].apply(_clean)
        if _SRC_DESC2 in expenses.columns
        else pd.Series("", index=expenses.index)
    )

    # Classify into standard categories (< 10)
    expenses[COL_CATEGORY] = [
        _classify(d1, d2) for d1, d2 in zip(raw_desc1, raw_desc2)
    ]
    expenses[COL_DESC] = raw_desc2
    expenses[COL_SUB] = (
        expenses[_SRC_SUB].apply(_clean)
        if _SRC_SUB in expenses.columns else ""
    )

    # Group by all three levels
    group_cols = [COL_CATEGORY, COL_DESC, COL_SUB]
    grouped = (
        expenses.groupby(group_cols, as_index=False)[STD_AMOUNT]
        .sum()
        .rename(columns={STD_AMOUNT: COL_SPENT})
    )

    # Use absolute values for percentage calculation so income and expenses
    # are both shown as positive proportions of total activity
    grouped["_abs"] = grouped[COL_SPENT].abs()
    grand_abs = grouped["_abs"].sum()

    cat_abs = grouped.groupby(COL_CATEGORY)["_abs"].sum().rename("_cat_abs")
    grouped = grouped.join(cat_abs, on=COL_CATEGORY)

    grouped[COL_PCT_CAT]   = (grouped["_abs"] / grouped["_cat_abs"] * 100).round(2)
    grouped[COL_PCT_TOTAL] = (grouped["_abs"] / grand_abs * 100).round(2)

    grouped = (
        grouped
        .sort_values(["_cat_abs", "_abs"], ascending=[False, False])
        .drop(columns=["_abs", "_cat_abs"])
        .reset_index(drop=True)
    )

    net_total = grouped[COL_SPENT].sum()

    # Append TOTAL row
    total_row = pd.DataFrame([{
        COL_CATEGORY:  "TOTAL",
        COL_DESC:      "",
        COL_SUB:       "",
        COL_SPENT:     round(net_total, 2),
        COL_PCT_CAT:   "",
        COL_PCT_TOTAL: 100.0,
    }])
    grouped = pd.concat([grouped, total_row], ignore_index=True)

    return grouped[[COL_CATEGORY, COL_DESC, COL_SUB,
                    COL_SPENT, COL_PCT_CAT, COL_PCT_TOTAL]]


# ── Terminal display ───────────────────────────────────────────────────────────

def format_report(summary: pd.DataFrame) -> str:
    """Render the summary as a readable grouped hierarchy for terminal output."""
    sep  = "=" * 76
    sep2 = "-" * 74
    lines = [sep, "  EXPENSE SUMMARY  (Category > Description > Sub Description)", sep]

    if summary.empty or (len(summary) == 1 and summary.iloc[0][COL_CATEGORY] == "TOTAL"):
        lines.append("  No expenses found.")
        lines.append(sep)
        return "\n".join(lines)

    rows      = summary[summary[COL_CATEGORY] != "TOTAL"]
    total_row = summary[summary[COL_CATEGORY] == "TOTAL"]

    current_cat = None
    for _, row in rows.iterrows():
        cat = row[COL_CATEGORY]
        if cat != current_cat:
            if current_cat is not None:
                lines.append("")
            current_cat = cat
            cat_spent   = rows[rows[COL_CATEGORY] == cat][COL_SPENT].sum()
            pct_total   = rows[rows[COL_CATEGORY] == cat][COL_PCT_TOTAL].sum()
            lines.append(f"  {cat:<30}  ${cat_spent:>10.2f}   ({pct_total:.2f}% of total)")
            lines.append(f"    {'Description':<24}  {'Sub Description':<22}  {'Spent':>10}  {'%cat':>6}  {'%tot':>6}")
            lines.append(f"    {sep2}")

        desc = str(row[COL_DESC]) if row[COL_DESC] else "-"
        sub  = str(row[COL_SUB])  if row[COL_SUB]  else "-"
        lines.append(
            f"    {desc:<24}  {sub:<22}  ${row[COL_SPENT]:>10.2f}"
            f"  {str(row[COL_PCT_CAT]):>5}%  {str(row[COL_PCT_TOTAL]):>5}%"
        )

    if not total_row.empty:
        t = total_row.iloc[0]
        lines.append(f"\n{sep}")
        lines.append(f"  {'TOTAL':<30}  ${t[COL_SPENT]:>10.2f}   (100%)")

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

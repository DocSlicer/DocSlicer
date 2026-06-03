"""
Shared table utilities used across HTML, PDF, and DOCX pipelines.
"""

from __future__ import annotations

import re

import pandas as pd


_YEAR_PAT = re.compile(r"\b20\d{2}\b")
_DATE_PAT = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}\b",
    re.IGNORECASE,
)
_UNIT_PHRASES = frozenset(
    {
        "in thousands",
        "in millions",
        "in billions",
        "except per share",
        "per share",
        "percentage",
        "(%)",
        "year ended",
        "years ended",
        "month ended",
        "months ended",
        "quarter ended",
        "quarters ended",
        "three months",
        "six months",
        "nine months",
        "twelve months",
        "total",
        "actual",
        "adjusted",
        "number",
        "shares",
        "amount",
        "value",
    }
)
_NUMBER_RE = re.compile(r"^[\(\-]?\$?[\d,]+(\.\d+)?\)?$")

# Currency/unit indicator patterns — matches cells like "$m", "(€bn)", "k£", "(%)", etc.
# Scale abbreviations: m/mm (millions), b/bn (billions), tr/trn (trillions), k (thousands),
# '000 / 000 (thousands).  Word forms: millions, billions, thousands, trillions.
_CUR_SYM = r"[£€$¥₹₽₩₪₺¢]"
_CUR_CODE = r"(?:USD|EUR|GBP|JPY|CHF|AUD|CAD|CNY|HKD|SEK|NOK|DKK|NZD|ZAR|MXN|BRL|INR|RUB|KRW|TRY|THB|SGD|MYR)"
_SCALE = r"(?:trillions?|billions?|millions?|thousands?|trn?|bn?|mm?|k|'?0{3})"
_CURRENCY_UNIT_RE = re.compile(
    r"^\s*[\(\[]?\s*"
    r"(?:"
    # symbol/code  [scale]  [%]
    r"(?:" + _CUR_SYM + r"|" + _CUR_CODE + r")(?:\s*" + _SCALE + r")?\s*%?"
    r"|"
    # scale  [symbol/code]  [%]  — e.g. k$, m€
    r"(?:" + _SCALE + r")(?:\s*(?:" + _CUR_SYM + r"|" + _CUR_CODE + r"))?\s*%?"
    r"|"
    # bare percent
    r"%"
    r")"
    r"\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def _looks_numeric(text: str) -> bool:
    t = (text or "").strip()
    if t in {"—", "-", ""}:
        return False
    return bool(_NUMBER_RE.match(t)) and not _YEAR_PAT.search(t)


def _is_currency_unit_cell(text: str) -> bool:
    """True when a cell contains only a currency/scale/percent indicator."""
    t = (text or "").strip()
    return bool(t) and bool(_CURRENCY_UNIT_RE.match(t))


def detect_cell_roles(
    df: pd.DataFrame,
    with_row_label: bool = True,
) -> pd.DataFrame:
    """
    Add a ``role`` column (header | row_label | data) to a table-cells DataFrame.

    Required columns: row_start, col_start, rowspan, text
    Optional column:  th (bool) — True when the HTML cell was a <th> element;
                      used to promote entire rows to header.

    with_row_label: if True, col_start==0 in data rows becomes 'row_label'.
    """
    if df.empty:
        df = df.copy()
        df["role"] = pd.Series(dtype="object")
        return df

    df = df.copy()

    # --- 1. Collect header rows ---
    header_rows: set[int] = {0}

    # Rowspan cells in row 0 extend the header zone
    row0 = df[df["row_start"] == 0]
    for rs in row0["rowspan"]:
        for r in range(1, int(rs)):
            header_rows.add(r)

    # TH-tagged cells promote their entire row
    if "th" in df.columns:
        th_rows = set(df.loc[df["th"].fillna(False).astype(bool), "row_start"])
        header_rows |= th_rows

    # Heuristic: year / date / unit-phrase rows (up to row 5, stop at first numeric row)
    max_row = int(df["row_start"].max())
    for r in range(1, min(6, max_row + 1)):
        if r in header_rows:
            continue
        row_cells = df[df["row_start"] == r]
        if row_cells.empty:
            continue

        # Stop if any cell looks like numeric data
        if row_cells["text"].fillna("").apply(_looks_numeric).any():
            break

        # If the row-label column (col 0) has content this is a data row, not a header
        col0_cells = row_cells[row_cells["col_start"] == 0]
        col0_blank = col0_cells.empty or col0_cells["text"].fillna("").str.strip().eq("").all()
        if not col0_blank:
            break

        combined = " ".join(row_cells["text"].fillna("").str.lower())
        nonblank = row_cells[row_cells["text"].fillna("").str.strip() != ""]
        is_currency_unit_row = (
            not nonblank.empty
            and nonblank["text"].apply(_is_currency_unit_cell).any()
        )
        if (
            _YEAR_PAT.search(combined)
            or _DATE_PAT.search(combined)
            or any(p in combined for p in _UNIT_PHRASES)
            or is_currency_unit_row
        ):
            header_rows.add(r)
        else:
            # col0 is blank/absent — treat as continuation of the header zone
            header_rows.add(r)

    # --- 2. Assign roles ---
    df["role"] = "data"
    df.loc[df["row_start"].isin(header_rows), "role"] = "header"

    if with_row_label:
        is_data = ~df["row_start"].isin(header_rows)
        is_col0 = df["col_start"] == 0
        df.loc[is_data & is_col0, "role"] = "row_label"

    return df

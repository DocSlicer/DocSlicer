# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Shared table utilities used across HTML, PDF, and DOCX pipelines.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..text_utils import _CURRENCY_SYM_CLASS, numeric_value_mask

# ============================================================
# Config
# ============================================================

# Year mentions: 2025, FY2025, FY25, and estimate/forecast/actual suffixed 2027E
_YEAR_PAT = re.compile(r"\b(?:FY\s?\d{2}(?:\d{2})?|20\d{2})[EFA]?\b")
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
# Single compiled alternation of the unit phrases — a C-level substring scan
# replaces the per-row Python ``any(p in s for p in _UNIT_PHRASES)`` loop.
_UNIT_PHRASE_RE = re.compile("|".join(re.escape(p) for p in _UNIT_PHRASES))
# Currency/unit indicator patterns — matches cells like "$m", "(€bn)", "k£", "(%)", etc.
# Scale abbreviations: m/mm (millions), b/bn (billions), tr/trn (trillions), k (thousands),
# '000 / 000 (thousands).  Word forms: millions, billions, thousands, trillions.
_CUR_SYM = _CURRENCY_SYM_CLASS
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


def _table_key(df: pd.DataFrame) -> pd.Series:
    """
    Per-cell table id used as the outer grouping key so a whole multi-table
    cells frame is processed in one pass. Frames without ``table_id`` (a single
    table) collapse to one group.
    """
    if "table_id" in df.columns:
        return df["table_id"]
    return pd.Series(0, index=df.index)


# ============================================================
# Categorize a table row/line into numeric | text (Public function imported by pdf table builder)
# ============================================================


def _row_style(
    gkey: list[pd.Series | np.ndarray],
    populated: pd.Series,
    numeric_cell: pd.Series,
    eligible: pd.Series,
) -> pd.Series:
    """
    blank | numeric | text per group, from pre-computed cell masks. ``eligible``
    is ``populated`` with the row-label column removed. Index is the group key.
    """
    n_populated = populated.groupby(gkey).sum()
    n_eligible = eligible.groupby(gkey).sum()
    n_numeric = (numeric_cell & eligible).groupby(gkey).sum()
    frac_numeric = n_numeric / n_eligible.astype(float).replace(0.0, np.nan)
    return pd.Series(
        np.select(
            [n_populated.eq(0), frac_numeric.ge(0.5).fillna(False)],
            ["blank", "numeric"],
            default="text",
        ),
        index=n_populated.index,
    )


def assign_table_row_style(
    df: pd.DataFrame,
    row_col: str | None = None,
) -> pd.DataFrame:
    """
    Add a ``table_row_style`` column (blank | numeric | text) to a cells
    DataFrame, computed per row and broadcast to every cell of that row.

    Rows are keyed on ``row_start`` when present, else ``line_id`` (or an
    explicit ``row_col``), scoped by ``table_id`` when the frame carries one so
    multiple tables can share row numbers. The first column is ignored — it
    usually holds the row label, which is text even in numeric rows. "First
    column" means col_start == 0 when the df has grid positions, otherwise the
    first populated cell of the row in reading order (by x_left when present).

      numeric  >=50% of the remaining populated cells look like numeric
               values (numeric_value_mask: $100 / 1,234 / 12% / (567) /
               dash & NA placeholders), cells containing a 20xx year excluded
               so year header rows don't count
      blank    no populated cells at all
      text     everything else (including rows where only the first column
               is populated)
    """
    df = df.copy()
    if row_col is None:
        row_col = "row_start" if "row_start" in df.columns else "line_id"
    if df.empty or row_col not in df.columns or "text" not in df.columns:
        df["table_row_style"] = pd.Series(dtype="object")
        return df

    tkey = _table_key(df)
    rows = df[row_col]

    text = df["text"].fillna("").astype(str).str.strip()
    populated = text.ne("")
    year_cell = text.str.contains(_YEAR_PAT).fillna(False)
    numeric_cell = numeric_value_mask(text) & ~year_cell

    if "col_start" in df.columns:
        first_col = df["col_start"].eq(0)
    else:
        # First populated cell of each row in reading order stands in for col 0
        sub = df.loc[populated & rows.notna(), [row_col]].copy()
        sub["_t"] = tkey.loc[sub.index]
        if "x_left" in df.columns:
            sub["_x"] = df.loc[sub.index, "x_left"]
            sub = sub.sort_values(["_t", row_col, "_x"], kind="stable")
        first_col = pd.Series(False, index=df.index)
        first_col.loc[sub.groupby(["_t", row_col], sort=False).head(1).index] = True

    eligible = populated & ~first_col
    style = _row_style([tkey, rows], populated, numeric_cell, eligible)
    mi = pd.MultiIndex.from_arrays([tkey.to_numpy(), rows.to_numpy()])
    df["table_row_style"] = style.reindex(mi).to_numpy()
    return df


# ============================================================
# Identify table header rows
# ============================================================


# Per-row header-detection features added by assign_header_features, broadcast
# to every cell of the row so a table-cells CSV shows the decision inputs.
_ROW_FEATURE_COLS = [
    "table_row_style",
    "hdr_n_populated",
    "hdr_frac_numeric",
    "hdr_frac_bold",
    "hdr_frac_th",
    "hdr_has_year",
    "hdr_has_date",
    "hdr_is_currency_unit",
    "hdr_has_unit_phrase",
    "hdr_col0_blank",
    "hdr_in_row0_span",
    "hdr_has_colspan",
    "hdr_max_char_count",
    "hdr_has_ix",
]

# Decision columns added by detect_cell_roles on top of the features.
_ROW_DECISION_COLS = ["hdr_score", "hdr_decision"]


def _compute_row_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core of assign_header_features / detect_cell_roles. Enrich a (possibly
    multi-table) table-cells frame with the header-detection features in a
    single grouped pass and return ``(df, feat)`` where:

      df    the input with ``char_count`` and the _ROW_FEATURE_COLS added
            (row features broadcast to every cell of the row)
      feat  one row per (table_id, row_start), columns _ROW_FEATURE_COLS,
            sorted — the input score_header_rows/detect_cell_roles consume

    The whole frame is grouped once on ``(table_id, row_start)`` instead of a
    dozen separate groupby passes per table. See assign_header_features for the
    feature definitions. Required columns: row_start, col_start, rowspan, text.
    """
    df = df.copy()
    tkey = _table_key(df)
    rows = df["row_start"].astype(int)
    tkey_a = tkey.to_numpy()
    rows_a = rows.to_numpy()

    # --- 1. Per-cell signals (vectorized over the whole frame, once) ---
    text = df["text"].fillna("").astype(str)
    stripped = text.str.strip()
    char_count = text.str.len()

    populated = stripped.ne("")
    year_cell = stripped.str.contains(_YEAR_PAT).fillna(False)
    # Numeric-like values incl. %, $-prefixed, dash/NA placeholders; cells
    # containing a year are excluded so year header rows stay non-numeric.
    numeric_cell = numeric_value_mask(stripped) & ~year_cell
    date_cell = stripped.str.contains(_DATE_PAT).fillna(False)
    currency_unit_cell = populated & stripped.str.match(_CURRENCY_UNIT_RE).fillna(False)

    if "table_header_flag" in df.columns:
        th_cell = df["table_header_flag"].fillna(False).astype(bool)
    else:
        th_cell = pd.Series(False, index=df.index)
    if "is_bold" in df.columns:
        bold_cell = df["is_bold"].fillna(False).astype(bool)
    else:
        bold_cell = pd.Series(False, index=df.index)

    col0 = df["col_start"].eq(0)
    eligible = populated & ~col0  # table_row_style ignores the row-label column

    if "colspan" in df.columns:
        colspan_gt1 = df["colspan"].fillna(1).astype(int).gt(1)
    else:
        colspan_gt1 = pd.Series(False, index=df.index)
    if "ix" in df.columns:
        ix_nonblank = df["ix"].notna() & df["ix"].astype(str).str.strip().ne("")
    else:
        ix_nonblank = pd.Series(False, index=df.index)

    # --- 2. Fold cell signals to one record per (table_id, row_start) in one pass ---
    src = pd.DataFrame(
        {
            "_pop": populated.to_numpy(),
            "_num_pop": (numeric_cell & populated).to_numpy(),
            "_bold_pop": (bold_cell & populated).to_numpy(),
            "_th": th_cell.to_numpy(dtype=float),
            "_year": year_cell.to_numpy(),
            "_date": date_cell.to_numpy(),
            "_curunit": currency_unit_cell.to_numpy(),
            "_col0pop": (populated & col0).to_numpy(),
            "_char": char_count.to_numpy(),
            "_colspan": colspan_gt1.to_numpy(),
            "_ix": ix_nonblank.to_numpy(),
            "_elig": eligible.to_numpy(),
            "_num_elig": (numeric_cell & eligible).to_numpy(),
        }
    )
    feat = src.groupby([tkey_a, rows_a], sort=True).agg(
        hdr_n_populated=("_pop", "sum"),
        _num_pop=("_num_pop", "sum"),
        _bold_pop=("_bold_pop", "sum"),
        hdr_frac_th=("_th", "mean"),
        hdr_has_year=("_year", "any"),
        hdr_has_date=("_date", "any"),
        hdr_is_currency_unit=("_curunit", "any"),
        _col0pop=("_col0pop", "any"),
        hdr_max_char_count=("_char", "max"),
        hdr_has_colspan=("_colspan", "any"),
        hdr_has_ix=("_ix", "any"),
        _n_elig=("_elig", "sum"),
        _num_elig=("_num_elig", "sum"),
    )

    denom = feat["hdr_n_populated"].astype(float).replace(0.0, np.nan)
    feat["hdr_frac_numeric"] = (feat.pop("_num_pop") / denom).fillna(0.0)
    feat["hdr_frac_bold"] = (feat.pop("_bold_pop") / denom).fillna(0.0)
    feat["hdr_col0_blank"] = ~feat.pop("_col0pop")
    feat["hdr_n_populated"] = feat["hdr_n_populated"].astype(int)
    feat["hdr_max_char_count"] = feat["hdr_max_char_count"].astype(int)

    # table_row_style: numeric when >=50% of the non-first-column populated cells look numeric
    frac_elig = feat.pop("_num_elig") / feat.pop("_n_elig").astype(float).replace(0.0, np.nan)
    feat["table_row_style"] = np.select(
        [feat["hdr_n_populated"].eq(0), frac_elig.ge(0.5).fillna(False)],
        ["blank", "numeric"],
        default="text",
    )

    # Unit phrase: one C-level regex scan over each row's joined lowercase text
    joined = stripped.str.lower().groupby([tkey_a, rows_a], sort=True).agg(" ".join)
    feat["hdr_has_unit_phrase"] = (
        joined.reindex(feat.index).str.contains(_UNIT_PHRASE_RE).fillna(False)
    )

    # hdr_in_row0_span: rows covered by a rowspan reaching down from the table's
    # first row (per table).
    if "rowspan" in df.columns:
        rowspan = df["rowspan"].fillna(1).astype(int)
    else:
        rowspan = pd.Series(1, index=df.index)
    first_row_by_t = rows.groupby(tkey).min()
    is_first_row = rows.eq(tkey.map(first_row_by_t))
    row0_span_by_t = rowspan.where(is_first_row).groupby(tkey).max()

    feat_t = feat.index.get_level_values(0)
    feat_r = feat.index.get_level_values(1).to_numpy()
    fr = first_row_by_t.reindex(feat_t).to_numpy()
    span = row0_span_by_t.reindex(feat_t).fillna(1).to_numpy()
    feat["hdr_in_row0_span"] = (feat_r > fr) & (feat_r < fr + span)

    feat = feat[_ROW_FEATURE_COLS]

    # --- 3. Broadcast row features back to cells in one assign ---
    mi = pd.MultiIndex.from_arrays([tkey_a, rows_a])
    bcast = {col: feat[col].reindex(mi).to_numpy() for col in _ROW_FEATURE_COLS}
    bcast["char_count"] = char_count.to_numpy()
    df = df.assign(**bcast)
    return df, feat


def assign_header_features(df_table_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Add the header-detection feature columns to a table-cells DataFrame
    (one or many tables) — enrichment only, no role decision.

    Cell-level:
      char_count            length of the raw cell text, whitespace included

    Row-level (computed once per (table_id, row_start), broadcast back to every
    cell of that row — see _ROW_FEATURE_COLS):
      table_row_style       blank | numeric | text — see assign_table_row_style:
                            first column ignored, numeric when >=50% of the
                            remaining populated cells look numeric
      hdr_n_populated       number of non-blank cells in the row
      hdr_frac_numeric      fraction of populated cells that look numeric
      hdr_frac_bold         fraction of populated cells with is_bold
      hdr_frac_th           fraction of cells flagged table_header_flag
      hdr_has_year          any cell contains a year (2025 / FY25 / 2027E)
      hdr_has_date          any cell contains a "Mon DD, YYYY" style date
      hdr_is_currency_unit  any populated cell is only a currency/scale/% marker
      hdr_has_unit_phrase   row text contains a unit phrase ("in thousands", …)
      hdr_col0_blank        the row-label column (col_start 0) is blank/absent
      hdr_in_row0_span      row is covered by a rowspan reaching down from row 0
      hdr_has_colspan       any cell in the row spans multiple columns
      hdr_max_char_count    longest cell text in the row (whitespace included)
      hdr_has_ix            any cell has a non-blank ix value (html only)

    Required columns: row_start, col_start, rowspan, text
    Optional columns: table_id, table_header_flag (bool-ish), is_bold (bool-ish), ix
    """
    if df_table_cells.empty:
        df = df_table_cells.copy()
        df["char_count"] = pd.Series(dtype="int64")
        for col in _ROW_FEATURE_COLS:
            df[col] = pd.Series(dtype="object")
        return df
    df, _ = _compute_row_features(df_table_cells)
    return df


# Header-evidence score weights and penalties. Booleans contribute the full
# weight; negatives argue against a header row. Tune against table-cells CSVs.
_HDR_WEIGHTS = {
    "bold_minority": 3.0,      # row is bold in a table that is generally not bold
    "col0_blank": 3.0,         # hdr_col0_blank
    "numeric": -10.0,          # table_row_style == numeric — effectively a veto
    "first_row": 1.0,          # table's first row
    "second_row": 0.0,         # table's second row
    "in_row0_span": 1.0,       # hdr_in_row0_span
    "year_or_date": 2.0,       # hdr_has_year | hdr_has_date
    "currency_or_unit": 3.0,   # hdr_is_currency_unit | hdr_has_unit_phrase
    "has_ix": -2.0,            # hdr_has_ix (html-only index column)
    "has_colspan": 1.0,        # hdr_has_colspan — any cell spans columns
    "single_cell": -3.0,       # non-first row with only 1 populated cell
    "two_row_table": -1.0,     # flat penalty on every row of a 2-row table
}
# Long cells argue against a header row: penalty by the row's longest cell,
# first matching threshold wins (not cumulative).
_HDR_CHAR_PENALTIES = [(300, -5.0), (200, -2.0), (100, -1.0)]
# Depth argues against a header row: penalty by offset from the table's first
# row (offsets 0/1 get the first/second_row bonuses instead); offsets beyond
# the mapped ones get the floor penalty.
_HDR_ROW_OFFSET_PENALTIES = {2: -2.0, 3: -3.0, 4: -4.0}
_HDR_ROW_OFFSET_FLOOR = -8.0
# Minority gate: styling/content cues (bold, year/date, currency/unit) only
# signal a header when they are the exception — if more than this fraction of
# the table's rows share the cue it carries no information (e.g. a directory
# table where every data row contains a date).
_HDR_MINORITY_FRAC = 0.5
_HDR_THRESHOLD = 1.0


def score_header_rows(feat: pd.DataFrame) -> pd.Series:
    """
    Header-evidence score per row, from a row-level feature frame (one row per
    (table_id, row_start), columns _ROW_FEATURE_COLS — _compute_row_features
    builds it). Positive = header-like; compare against _HDR_THRESHOLD. Scoring
    is per-table: first-row/depth/minority/two-row terms are all scoped to each
    table_id. TH-flagged rows are auto-headers in the caller and are not part of
    the scoring model.
    """
    w = _HDR_WEIGHTS
    tid = feat.index.get_level_values(0)
    ridx = pd.Series(feat.index.get_level_values(1).astype(int), index=feat.index)
    first_row = ridx.groupby(tid).transform("min")
    n_rows = ridx.groupby(tid).transform("size")
    offset = ridx - first_row

    def _minority(cue: pd.Series, weight: float) -> pd.Series:
        # A cue only signals when it is the exception within its table: if most
        # rows share it (> _HDR_MINORITY_FRAC) it carries no header information —
        # e.g. bold in an all-bold table, dates in a table with a date column.
        share = cue.groupby(tid).transform("mean")
        return cue.astype(float) * np.where(share <= _HDR_MINORITY_FRAC, weight, 0.0)

    bold_signal = _minority(feat["hdr_frac_bold"].ge(0.5), w["bold_minority"])
    year_date_signal = _minority(
        feat["hdr_has_year"] | feat["hdr_has_date"], w["year_or_date"]
    )
    cur_unit_signal = _minority(
        feat["hdr_is_currency_unit"] | feat["hdr_has_unit_phrase"],
        w["currency_or_unit"],
    )

    max_chars = feat["hdr_max_char_count"].astype(float)
    char_penalty = pd.Series(
        np.select(
            [max_chars.gt(t) for t, _ in _HDR_CHAR_PENALTIES],
            [p for _, p in _HDR_CHAR_PENALTIES],
            default=0.0,
        ),
        index=feat.index,
    )

    depth_penalty = offset.map(_HDR_ROW_OFFSET_PENALTIES).fillna(0.0)
    depth_penalty[offset > max(_HDR_ROW_OFFSET_PENALTIES)] = _HDR_ROW_OFFSET_FLOOR

    score = (
        bold_signal
        + year_date_signal
        + cur_unit_signal
        + char_penalty
        + depth_penalty
        + w["col0_blank"] * feat["hdr_col0_blank"]
        + w["numeric"] * feat["table_row_style"].eq("numeric")
        + w["first_row"] * offset.eq(0)
        + w["second_row"] * offset.eq(1)
        + w["in_row0_span"] * feat["hdr_in_row0_span"]
        + w["has_ix"] * feat["hdr_has_ix"]
        + w["has_colspan"] * feat["hdr_has_colspan"]
        + w["single_cell"] * (offset.ne(0) & feat["hdr_n_populated"].eq(1))
        + w["two_row_table"] * n_rows.eq(2)
    )
    return score.astype(float)


def detect_cell_roles(
    df_table_cells: pd.DataFrame,
    with_row_label: bool = True,
) -> pd.DataFrame:
    """
    Add a ``table_cell_role`` column (header | row_label | data) to a table-cells
    DataFrame, which may hold one or many tables (grouped by ``table_id``), plus
    the feature columns from assign_header_features and two decision columns:

      hdr_score      per-row header-evidence score (score_header_rows)
      hdr_decision   why the row landed where it did:
                       th_promote      row has table_header_flag → header
                       single_row      1-row table without TH → never header
                       score:X         scored X, header iff inside the zone
                       outside_zone:X  scored >= threshold but after the zone cut

    Decision (evaluated per table):
      1. Any row with table_header_flag is a header, wherever it sits
         (w:tblHeader / <th> — the source format says so).
      2. A 1-row table is otherwise never a header.
      3. Remaining rows are scored (score_header_rows, _HDR_WEIGHTS) against
         _HDR_THRESHOLD; the header zone is the unbroken run of qualifying
         rows from the table's first row — nothing after the first failing
         row is header. A first row that fails means no header at all.

    with_row_label: if True, col_start==0 in non-header rows becomes
    'row_label'.
    """
    if df_table_cells.empty:
        df = df_table_cells.copy()
        df["char_count"] = pd.Series(dtype="int64")
        for col in _ROW_FEATURE_COLS:
            df[col] = pd.Series(dtype="object")
        for col in _ROW_DECISION_COLS:
            df[col] = pd.Series(dtype="object")
        df["table_cell_role"] = pd.Series(dtype="string")
        return df

    df, feat = _compute_row_features(df_table_cells)
    tkey = _table_key(df)
    rows = df["row_start"].astype(int)

    tid = feat.index.get_level_values(0)
    ridx = pd.Series(feat.index.get_level_values(1).astype(int), index=feat.index)
    n_rows = ridx.groupby(tid).transform("size")
    single = n_rows.eq(1)

    th_row = feat["hdr_frac_th"].gt(0)
    score = score_header_rows(feat)
    score_str = score.round(2).astype(str)

    # Header zone: unbroken run of qualifying rows from each table's first row.
    # feat is sorted by (table_id, row_start) so the per-table cumprod runs in
    # increasing row order.
    header_like = th_row | score.ge(_HDR_THRESHOLD)
    zone = header_like.groupby(tid).cumprod().astype(bool)
    header_row = pd.Series(
        np.where(single, th_row, zone | th_row), index=feat.index
    )

    decision = pd.Series(
        np.select(
            [single & th_row, single, th_row, zone, score.ge(_HDR_THRESHOLD)],
            [
                "th_promote",
                "single_row",
                "th_promote",
                "score:" + score_str,
                "outside_zone:" + score_str,
            ],
            default="score:" + score_str,
        ),
        index=feat.index,
    )

    # --- Broadcast per-row decisions back to cells ---
    mi = pd.MultiIndex.from_arrays([tkey.to_numpy(), rows.to_numpy()])
    df["hdr_score"] = score.reindex(mi).to_numpy()
    df["hdr_decision"] = decision.reindex(mi).to_numpy()

    is_header = pd.Series(header_row.reindex(mi).to_numpy(), index=df.index)
    df["table_cell_role"] = "data"
    df.loc[is_header, "table_cell_role"] = "header"
    if with_row_label:
        df.loc[~is_header & df["col_start"].eq(0), "table_cell_role"] = "row_label"
    df["table_cell_role"] = df["table_cell_role"].astype("string")

    return df

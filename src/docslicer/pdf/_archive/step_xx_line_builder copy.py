"""
step_08_line_scorer.py

Aggregate cells into lines and score each line for table-like features.

Public API:
    df_lines, df_cells = build_line_scores(df_cells, df_words)

Pipeline:
    Step 1: Aggregate cells → df_lines
        - Groups cells by line_id
        - Merges text, geometry, style, counts
        - Creates bracketed 'cells' column: "[cell1] [cell2] [cell3]"

    Step 2: Compute ratios
        - width_ratio, digit_ratio, capitalized_word_ratio  (from df_cells)
        - median_x0x1_gap, max_x0x1_gap, gap_ratio          (from df_words, vectorized)

    Step 3: Score table-likeness
        - Vectorized heuristic scoring → table_row_score per line

    Step 4: Merge score back to df_cells
        - Each cell inherits table_row_score from its parent line_id

Notes:
    df_lines is designed to be kept alive through the rest of the pipeline.
    Steps 09+ add layout columns (layout_id, col_start, row_start, layout_type)
    to df_cells via joins; df_lines content does not change after this step.
    step_11_line_merger can therefore use df_lines directly instead of
    re-aggregating from df_cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Columns that are identical for every cell in a line (set during line_id assignment).
# Aggregated with "first" — no value_counts() overhead.
_IDENTITY_COLS = [
    "page_number",
    "page_width",
    "page_height",
    "reading_column",
    "gutter_id_left",
    "gutter_id_right",
    "is_sentence_like",
    "sentence_score",
]

# Boolean flag columns — "max" means True if any cell has it.
_FLAG_COLS = [
    "has_link",
    "is_underlined",
    "has_vertical_line",
]

# Style columns — take the most common value across cells in the line.
_MODE_COLS = [
    "font_name",
    "font_family",
    "font_size",
    "font_weight",
    "non_stroking_color",
    "stroking_color",
    "text_orientation",
]

# Numeric count columns — summed across cells.
_SUM_COLS = [
    "char_count",
    "alpha_count",
    "digit_count",
    "uppercase_count",
    "word_count",
    "alpha_word_count",
    "capitalized_word_count",
]


# ============================================================
# HELPERS
# ============================================================

def _mode_or_first(series: pd.Series) -> Any:
    """Return the most prevalent value, or the first non-null if all unique."""
    if series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if not vc.empty:
        return vc.index[0]
    s2 = series.dropna()
    return s2.iloc[0] if not s2.empty else None


# ============================================================
# STEP 1: Aggregate cells → lines
# ============================================================

def _aggregate_cells_to_lines(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Group cells by line_id to produce one row per line.

    Geometry: union (min x_left, max x_right, min y_top, max y_bottom).
    Text:     cells joined with a space, left→right order.
    Cells:    bracketed "[cell1] [cell2] ..." representation.
    Counts:   summed.
    bold/italic ratios: char-count-weighted average across cells.
    Style:    most common value across cells (_mode_or_first).
    Flags:    True if any cell has the flag (max).
    """
    df = df_cells.sort_values(["line_id", "x_left", "y_top"], kind="mergesort")

    # Pre-compute weighted bold/italic char estimates for ratio recalculation.
    df = df.copy()
    df["_bold_char_est"]   = df["bold_ratio"].fillna(0.0)   * df["char_count"].fillna(0.0)
    df["_italic_char_est"] = df["italic_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)

    # Ensure flag columns are present (documents without links/underlines/etc).
    for col in _FLAG_COLS:
        if col not in df.columns:
            df[col] = False

    present = set(df.columns)

    agg_spec: dict[str, Any] = {
        # Geometry
        "x_left":   "min",
        "x_right":  "max",
        "y_top":    "min",
        "y_bottom": "max",
        # Weighted ratio helpers
        "_bold_char_est":   "sum",
        "_italic_char_est": "sum",
        # Cell count (will be renamed)
        "cell_id": "count",
        # Text
        "text": lambda s: " ".join(t for t in s.astype(str) if t.strip()),
        # Identity (same for every cell in a line)
        **{col: "first" for col in _IDENTITY_COLS if col in present},
        # Flags
        **{col: "max" for col in _FLAG_COLS},
        # Counts
        **{col: "sum" for col in _SUM_COLS if col in present},
        # Style
        **{col: _mode_or_first for col in _MODE_COLS if col in present},
    }

    grouped = (
        df.groupby("line_id", sort=True, observed=True)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"cell_id": "cell_count"})
    )

    # Derived geometry
    grouped["width"]  = grouped["x_right"] - grouped["x_left"]
    grouped["height"] = grouped["y_bottom"] - grouped["y_top"]

    # Recompute bold/italic ratios from weighted estimates
    total_chars = grouped["char_count"].replace(0, np.nan)
    grouped["bold_ratio"]   = (grouped["_bold_char_est"]   / total_chars).fillna(0.0)
    grouped["italic_ratio"] = (grouped["_italic_char_est"] / total_chars).fillna(0.0)
    grouped = grouped.drop(columns=["_bold_char_est", "_italic_char_est"])

    # Bracketed cells column — separate groupby to avoid lambda collision above.
    cells_col = (
        df.groupby("line_id", sort=False)["text"]
        .apply(lambda s: " ".join(f"[{t}]" for t in s.astype(str) if t.strip()))
        .rename("cells")
        .reset_index()
    )
    grouped = grouped.merge(cells_col, on="line_id", how="left")

    return grouped


# ============================================================
# STEP 2: Compute ratios
# ============================================================

def _compute_line_ratios(df_lines: pd.DataFrame, df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio features used by the table-row scorer.

    Cell-level ratios (derived from df_lines):
        width_ratio             line_width / page_width
        digit_ratio             digit_count / char_count
        capitalized_word_ratio  capitalized_word_count / word_count
        underlined_ratio        1.0 if is_underlined else 0.0

    Word-level gap stats (derived from df_words, fully vectorised):
        median_x0x1_gap     median positive gap between consecutive words
        max_x0x1_gap        maximum positive gap between consecutive words
        gap_ratio           max_gap / (median_gap + 1e-6)
    """
    df = df_lines.copy()

    df["width_ratio"] = (
        df["width"] / df["page_width"].replace(0, np.nan)
    ).fillna(0.0)

    df["digit_ratio"] = (
        df["digit_count"] / df["char_count"].replace(0, np.nan)
    ).fillna(0.0)

    df["capitalized_word_ratio"] = (
        df["capitalized_word_count"] / df["word_count"].replace(0, np.nan)
    ).fillna(0.0)

    df["underlined_ratio"] = df["is_underlined"].fillna(False).astype(float)

    # --- Word-gap stats (vectorised) ---
    if df_words.empty or "line_id" not in df_words.columns:
        df["median_x0x1_gap"] = 0.0
        df["max_x0x1_gap"]    = 0.0
        df["gap_ratio"]       = 0.0
        return df

    ws = (
        df_words[["line_id", "x_left", "x_right"]]
        .sort_values(["line_id", "x_left"])
        .copy()
    )

    # Gap to the next word within the same line (NaN for the last word of each line).
    ws["_next_x_left"] = ws.groupby("line_id", sort=False)["x_left"].shift(-1)
    ws["_gap"]         = ws["_next_x_left"] - ws["x_right"]

    positive = ws[ws["_gap"] > 0]

    if positive.empty:
        df["median_x0x1_gap"] = 0.0
        df["max_x0x1_gap"]    = 0.0
        df["gap_ratio"]       = 0.0
        return df

    gap_stats = (
        positive.groupby("line_id")["_gap"]
        .agg(median_x0x1_gap="median", max_x0x1_gap="max")
        .reset_index()
    )
    gap_stats["gap_ratio"] = (
        gap_stats["max_x0x1_gap"] / (gap_stats["median_x0x1_gap"] + 1e-6)
    )

    df = df.merge(gap_stats, on="line_id", how="left")
    for col in ("median_x0x1_gap", "max_x0x1_gap", "gap_ratio"):
        df[col] = df[col].fillna(0.0)

    return df


# ============================================================
# STEP 3: Score table-likeness
# ============================================================

def score_row(
    digit_ratio: float,
    underlined_ratio: float,
    width_ratio: float,
    has_vertical_line: bool,
    median_x0x1_gap: float,
    gap_ratio: float,
    capitalized_word_ratio: float,
    cell_count: int,
) -> float:
    """
    Heuristic score for how likely a single line belongs to a table row.

    Roughly in the range [-5, +15].  Exposed as a public scalar function
    so it can be called from tests or used for explanation/debugging.
    For batch scoring use _compute_table_row_scores() which is vectorised.
    """
    score = 0.0

    clusters = int(cell_count or 0)
    if clusters < 3:
        score -= 1.5
    elif clusters == 3:
        score += 0.3
    elif clusters == 4:
        score += 0.8
    else:
        score += 1.2

    score += float(np.clip((float(digit_ratio) - 0.2) * 3.0, -0.5, 1.5))
    score += min(float(underlined_ratio) * 1.5, 1.0)

    wr = float(width_ratio)
    if wr < 0.35:
        score -= 0.5
    elif wr >= 0.55:
        score += 0.3

    if bool(has_vertical_line):
        score += 6.0

    mg = float(median_x0x1_gap)
    if mg < 5.0:
        score -= 1.5
    elif mg < 10.0:
        score += float(np.interp(mg, [5.0, 10.0], [0.0, 1.0]))
    elif mg < 15.0:
        score += float(np.interp(mg, [10.0, 15.0], [1.0, 1.5]))
    else:
        score += 1.7

    gr = float(gap_ratio)
    if gr < 2.0:
        score -= 1.5
    elif gr < 5.0:
        score += float(np.interp(gr, [2.0, 5.0], [0.0, 1.0]))
    elif gr < 15.0:
        score += float(np.interp(gr, [5.0, 15.0], [1.0, 2.0]))
    else:
        score += 2.0

    ctr = float(capitalized_word_ratio)
    if ctr < 0.2:
        score -= 0.5
    elif ctr < 0.7:
        score += float(np.interp(ctr, [0.2, 0.7], [0.0, 1.0]))
    else:
        score += float(np.interp(min(ctr, 1.0), [0.7, 1.0], [1.0, 1.5]))

    return float(score)


def _compute_table_row_scores(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised batch version of score_row — avoids Python-level row iteration.
    Adds column: table_row_score (float).
    """
    if df_lines.empty:
        return df_lines.assign(table_row_score=pd.Series(dtype="float64"))

    df = df_lines.copy()

    cc  = df.get("cell_count",              pd.Series(0,     index=df.index)).fillna(0).to_numpy(dtype=float)
    dr  = df.get("digit_ratio",             pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    ur  = df.get("underlined_ratio",        pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    wr  = df.get("width_ratio",             pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    hvl = df.get("has_vertical_line",       pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=float)
    mg  = df.get("median_x0x1_gap",         pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    gr  = df.get("gap_ratio",               pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    ctr = df.get("capitalized_word_ratio",  pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()

    score = np.zeros(len(df), dtype=float)

    # Cell count
    score += np.where(cc < 3, -1.5,
             np.where(cc == 3, 0.3,
             np.where(cc == 4, 0.8, 1.2)))

    # Digit ratio
    score += np.clip((dr - 0.2) * 3.0, -0.5, 1.5)

    # Underlined ratio
    score += np.minimum(ur * 1.5, 1.0)

    # Width ratio
    score += np.where(wr < 0.35, -0.5, np.where(wr >= 0.55, 0.3, 0.0))

    # Vertical line
    score += hvl * 6.0

    # Median word gap
    score += np.where(mg < 5.0, -1.5,
             np.where(mg < 10.0, np.interp(mg, [5.0, 10.0], [0.0, 1.0]),
             np.where(mg < 15.0, np.interp(mg, [10.0, 15.0], [1.0, 1.5]),
             1.7)))

    # Gap ratio
    score += np.where(gr < 2.0, -1.5,
             np.where(gr < 5.0,  np.interp(gr, [2.0, 5.0],  [0.0, 1.0]),
             np.where(gr < 15.0, np.interp(gr, [5.0, 15.0], [1.0, 2.0]),
             2.0)))

    # Capitalized word ratio
    ctr_c = np.minimum(ctr, 1.0)
    score += np.where(ctr < 0.2, -0.5,
             np.where(ctr < 0.7, np.interp(ctr, [0.2, 0.7], [0.0, 1.0]),
             np.interp(ctr_c, [0.7, 1.0], [1.0, 1.5])))

    df["table_row_score"] = score
    return df


# ============================================================
# STEP 4: Merge score back to df_cells
# ============================================================

def _merge_score_to_cells(
    df_cells: pd.DataFrame,
    df_lines: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join table_row_score from df_lines onto df_cells via line_id."""
    score_map = df_lines.set_index("line_id")["table_row_score"].to_dict()
    df_out = df_cells.copy()
    df_out["table_row_score"] = df_out["line_id"].map(score_map).fillna(0.0)
    return df_out


# ============================================================
# PUBLIC API
# ============================================================

def build_lines(
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate cells into lines and score each line for table-like features.

    Parameters
    ----------
    df_cells : pd.DataFrame
        One row per cell. Output of step_07_cell_builder.build_cells().
        Must contain: line_id, cell_id, x_left, x_right, y_top, y_bottom,
        text, bold_ratio, italic_ratio, char_count, page_width.

    df_words : pd.DataFrame
        One row per word. The df_words_out returned by build_cells(),
        augmented with line_id. Used for word-level gap statistics.

    Returns
    -------
    df_lines : pd.DataFrame
        One row per line_id.  Contains aggregated content (text, geometry,
        style, counts) plus table_row_score and ratio features.
        Designed to be kept alive and reused by step_11_line_merger —
        content does not change after this step.

    df_cells : pd.DataFrame
        Input df_cells with table_row_score column added.
    """
    if df_cells.empty:
        return pd.DataFrame(), df_cells.copy()

    df_lines = _aggregate_cells_to_lines(df_cells)
    df_lines = _compute_line_ratios(df_lines, df_words)
    df_lines = _compute_table_row_scores(df_lines)
    df_cells = _merge_score_to_cells(df_cells, df_lines)

    return df_lines, df_cells

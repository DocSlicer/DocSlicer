"""
step_07_temp_line_builder.py

This script builds line-level aggregates from cells and scores them for table-like features.

Pipeline:
  Step 1: Build temp_lines_df
    - Groups cells by temp_line_id to create line-level aggregates
    - Merges cell text into line text
    - Creates bracketed 'cells' column: "[cell1] [cell2] [cell3]"
    - Aggregates geometry, style, counts, and flags
    
  Step 2: Compute Ratios
    - Uses word-level data to calculate gaps between consecutive words
    - Computes width_ratio, digit_ratio, capitalized_token_ratio
    - Calculates median_x0x1_gap, max_x0x1_gap, gap_ratio
    
  Step 3: Compute Table Row Score
    - Applies heuristic scoring function to identify table-like rows
    - Considers: digit ratio, underlined, width, vertical lines, gaps, etc.
    
  Step 4: Column Reordering
    - Reorders columns for consistency and readability
    - Places hierarchy columns first, then content, geometry, and computed features
    
  Step 5: Merge Score Back to Cells
    - Merges table_row_score from temp_lines back onto cells_df
    - Each cell inherits the score from its parent temp_line_id
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


# =====================
# Helpers
# =====================

def _mode_or_first(series: pd.Series) -> Any:
    """
    Fast-ish "most prevalent" helper:
      - if there is a mode, return the first mode
      - else fall back to first non-null
    """
    if series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if not vc.empty:
        return vc.index[0]
    # All NA
    return series.dropna().iloc[0] if series.dropna().size else None


# =====================
# STEP 1: Build temp_lines_df
# =====================

def _build_temp_lines_df(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Group cells by temp_line_id to create line-level aggregates.

    Rules:
      - Geometry:
          x_left  = min(x_left)
          x_right = max(x_right)
          y_top   = min(y_top)
          y_bottom= max(y_bottom)
          width   = x_right - x_left
          height  = y_bottom - y_top
      - Text: join cell texts with a space.
      - Cells: create bracketed representation of each cell's text.
      - Counts: sum the per-cell counts (char_count, alpha_count, etc.).
      - bold_ratio / italic_ratio:
          weighted by char_count, via per-cell bold_ratio * char_count.
      - font_name / colors: most prevalent (mode; fall back to first).
      - Flags: use max for boolean flags (has_link, is_underlined, has_vertical_line).
    """
    if df_cells.empty:
        return pd.DataFrame(
            columns=[
                "temp_line_id",
                "page_number",
                "horizontal_band_id",
                "text",
                "cells",
                "x_left",
                "x_right",
                "y_top",
                "y_bottom",
                "width",
                "height",
                "font_name",
                "font_family",
                "font_size",
                "text_orientation",
                "non_stroking_color",
                "stroking_color",
                "bold_ratio",
                "italic_ratio",
                "char_count",
                "alpha_count",
                "digit_count",
                "uppercase_count",
                "word_count",
                "alpha_word_count",
                "capitalized_word_count",
                "page_width",
                "page_height",
                "cell_count",
                "has_link",
                "is_underlined",
                "has_vertical_line",
                "block_role",
                "page_label",
                "line_gap",
                "page_median_gap",
                "page_gap_thresh",
            ]
        )

    df = df_cells.copy()

    # Ensure optional flag columns exist (PDFs without links/underlines/etc won't have these)
    for col in ["has_link", "is_underlined", "has_vertical_line"]:
        if col not in df.columns:
            df[col] = False

    # Ensure cells inside each temp_line are ordered left→right
    # (fallback on y_top just to break ties deterministically)
    df = df.sort_values(["temp_line_id", "x_left", "y_top"], kind="mergesort").reset_index(drop=True)

    # Helper: approximate bold/italic char counts for weighted ratios
    df["bold_char_est"] = df["bold_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    df["italic_char_est"] = df["italic_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)

    agg_spec: Dict[str, Any] = {
        "page_number": "first",
        "horizontal_band_id": "first",
        "page_width": "first",
        "page_height": "first",
        "block_role": _mode_or_first,
        "page_label": "first",
        "line_gap": "first",
        "page_median_gap": "first",
        "page_gap_thresh": "first",

        "x_left": "min",
        "x_right": "max",
        "y_top": "min",
        "y_bottom": "max",

        "char_count": "sum",
        "alpha_count": "sum",
        "digit_count": "sum",
        "uppercase_count": "sum",
        "word_count": "sum",
        "alpha_word_count": "sum",
        "capitalized_word_count": "sum",

        "bold_char_est": "sum",
        "italic_char_est": "sum",

        "font_name": _mode_or_first,
        "font_family": _mode_or_first,
        "font_size": _mode_or_first,
        "non_stroking_color": _mode_or_first,
        "stroking_color": _mode_or_first,
        "text_orientation": _mode_or_first,

        "has_link": "max",
        "is_underlined": "max",
        "has_vertical_line": "max",

        "cell_id": "count",  # Will be renamed to cell_count
        "text": lambda s: " ".join(t for t in s.astype(str) if t and str(t).strip()),
    }

    grouped = df.groupby("temp_line_id", sort=True, observed=True).agg(agg_spec).reset_index()

    # Rename cell_id count to cell_count
    grouped = grouped.rename(columns={"cell_id": "cell_count"})
    
    # Create cells column by re-aggregating text with brackets
    cells_text = df.groupby("temp_line_id", sort=False)["text"].apply(
        lambda s: " ".join(f"[{text}]" for text in s.astype(str) if text and str(text).strip())
    ).reset_index()
    cells_text = cells_text.rename(columns={"text": "cells"})
    
    # Merge cells column into grouped dataframe
    grouped = grouped.merge(cells_text, on="temp_line_id", how="left")

    # Recompute geometry
    grouped["width"] = grouped["x_right"] - grouped["x_left"]
    grouped["height"] = grouped["y_bottom"] - grouped["y_top"]

    # Recompute ratios from estimated bold/italic chars
    total_chars = grouped["char_count"].replace(0, np.nan)
    grouped["bold_ratio"] = (grouped["bold_char_est"] / total_chars).fillna(0.0)
    grouped["italic_ratio"] = (grouped["italic_char_est"] / total_chars).fillna(0.0)

    # Drop helper bold/italic estimates
    grouped = grouped.drop(columns=["bold_char_est", "italic_char_est"])

    return grouped


# =====================
# STEP 2: Compute Ratios
# =====================

def _compute_ratios(temp_lines_df: pd.DataFrame, words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional ratios and metrics for table row scoring.
    
    Uses word-level data to calculate gaps between consecutive words,
    which is more meaningful for table detection than cell-level gaps.
    
    Adds columns:
        - width_ratio: float (line_width / page_width)
        - median_x0x1_gap: float (median gap between consecutive words' x coords)
        - max_x0x1_gap: float (max gap between consecutive words' x coords)
        - gap_ratio: float (max_x0x1_gap / (median_x0x1_gap + 1e-6))
        - capitalized_word_ratio: float (capitalized_word_count / word_count)
        - digit_ratio: float (digit_count / char_count)
        - underlined_ratio: float (depends on is_underlined flag and char_count)
    """
    if temp_lines_df.empty:
        return temp_lines_df
    
    df = temp_lines_df.copy()
    
    # 1. Width ratio
    df["width_ratio"] = df["width"] / df["page_width"].replace(0, np.nan)
    df["width_ratio"] = df["width_ratio"].fillna(0.0)
    
    # 2. Capitalized word ratio
    df["capitalized_word_ratio"] = (
        df["capitalized_word_count"] / df["word_count"].replace(0, np.nan)
    ).fillna(0.0)
    
    # 3. Digit ratio
    df["digit_ratio"] = (
        df["digit_count"] / df["char_count"].replace(0, np.nan)
    ).fillna(0.0)
    
    # 4. Underlined ratio - approximate as binary flag for now
    # (could be improved by tracking underlined char counts)
    df["underlined_ratio"] = df["is_underlined"].fillna(0).astype(float)
    
    # 5. Compute gaps between WORDS within each temp_line
    gap_stats_list = []
    
    for temp_line_id in df["temp_line_id"].unique():
        # Get all words belonging to this temp_line
        line_words = words_df[words_df["temp_line_id"] == temp_line_id].copy()
        line_words = line_words.sort_values("x_left").reset_index(drop=True)
        
        if len(line_words) < 2:
            # Single word or empty - no gaps
            gap_stats_list.append({
                "temp_line_id": temp_line_id,
                "median_x0x1_gap": 0.0,
                "max_x0x1_gap": 0.0,
                "gap_ratio": 0.0,
            })
            continue
        
        # Calculate gaps between consecutive words
        gaps = []
        for i in range(1, len(line_words)):
            prev_x_right = line_words.loc[i-1, "x_right"]
            curr_x_left = line_words.loc[i, "x_left"]
            gap = curr_x_left - prev_x_right
            if gap > 0:
                gaps.append(gap)
        
        if gaps:
            median_gap = float(np.median(gaps))
            max_gap = float(np.max(gaps))
            gap_ratio = max_gap / (median_gap + 1e-6)
        else:
            median_gap = 0.0
            max_gap = 0.0
            gap_ratio = 0.0
        
        gap_stats_list.append({
            "temp_line_id": temp_line_id,
            "median_x0x1_gap": median_gap,
            "max_x0x1_gap": max_gap,
            "gap_ratio": gap_ratio,
        })
    
    # Merge gap stats back into temp_lines_df
    gap_stats_df = pd.DataFrame(gap_stats_list)
    df = df.merge(gap_stats_df, on="temp_line_id", how="left")
    
    # Fill any missing values
    df["median_x0x1_gap"] = df["median_x0x1_gap"].fillna(0.0)
    df["max_x0x1_gap"] = df["max_x0x1_gap"].fillna(0.0)
    df["gap_ratio"] = df["gap_ratio"].fillna(0.0)
    
    return df


# =====================
# STEP 3: Compute Table Row Score
# =====================

def compute_table_row_score(
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
    Heuristic score for how likely a line is part of a numeric table row.
    Rough scale: ~[-5, +15] (depending on params).
    """

    score = 0.0

    # ---------- 1) number of cells (replacing local_x_clusters) ----------
    clusters = int(cell_count or 0)

    if clusters < 3:
        score -= 1.5
    elif clusters == 3:
        score += 0.3
    elif clusters == 4:
        score += 0.8
    else:  # >= 5
        score += 1.2

    # ---------- 2) digit_ratio ----------
    dr = float(digit_ratio or 0.0)  # digits / chars
    # center at 0.2: ~no effect at 0.2, positive above, negative below
    contrib = (dr - 0.2) * 3.0
    contrib = max(min(contrib, 1.5), -0.5)  # clamp
    score += contrib

    # ---------- 3) underlined_ratio ----------
    ur = float(underlined_ratio or 0.0)
    score += min(ur * 1.5, 1.0)

    # ---------- 4) width_ratio ----------
    wr = float(width_ratio or 0.0)  # already precomputed as band_width / page_width
    if wr < 0.35:
        score -= 0.5
    elif wr >= 0.55:
        score += 0.3

    # ---------- 5) has_vertical_line flag ----------
    if bool(has_vertical_line):
        score += 6.0

    # ---------- 6) median_x0x1_gap ----------
    # median_x0x1_gap < 5:   -1.5
    # 5–10:                  0 → +1.0 (linear)
    # 10–15:                 +1.0 → +1.5 (linear)
    # >15:                   +1.7
    mg = float(median_x0x1_gap or 0.0)
    if mg < 5.0:
        score -= 1.5
    elif mg < 10.0:
        # map 5..10 → 0..1
        contrib = np.interp(mg, [5.0, 10.0], [0.0, 1.0])
        score += contrib
    elif mg < 15.0:
        # map 10..15 → 1.0..1.5
        contrib = np.interp(mg, [10.0, 15.0], [1.0, 1.5])
        score += contrib
    else:
        score += 1.7

    # ---------- 7) gap_ratio ----------
    # gap_ratio < 2:         -1.5
    # 2–5:                   0 → +1.0 (linear)
    # 5–15:                  +1.0 → +2.0 (linear)
    gr = float(gap_ratio or 0.0)
    if gr < 2.0:
        score -= 1.5
    elif gr < 5.0:
        contrib = np.interp(gr, [2.0, 5.0], [0.0, 1.0])
        score += contrib
    elif gr < 15.0:
        contrib = np.interp(gr, [5.0, 15.0], [1.0, 2.0])
        score += contrib
    else:
        score += 2.0

    # ---------- 8) capitalized_word_ratio ----------
    # <0.2:        -0.5
    # 0.2–0.7:     0 → +1.0
    # 0.7–1.0:     +1.0 → +1.5
    ctr = float(capitalized_word_ratio or 0.0)
    if ctr < 0.2:
        score -= 0.5
    elif ctr < 0.7:
        contrib = np.interp(ctr, [0.2, 0.7], [0.0, 1.0])
        score += contrib
    else:
        ctr_clamped = min(ctr, 1.0)
        contrib = np.interp(ctr_clamped, [0.7, 1.0], [1.0, 1.5])
        score += contrib

    return float(score)


def _compute_table_row_score(temp_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the table row scoring function to each line.
    
    Adds column:
        - table_row_score: float (heuristic score for table-like features)
    """
    if temp_lines_df.empty:
        temp_lines_df["table_row_score"] = pd.Series(dtype="float64")
        return temp_lines_df
    
    df = temp_lines_df.copy()
    
    # Apply the scoring function row by row
    df["table_row_score"] = df.apply(
        lambda row: compute_table_row_score(
            digit_ratio=row.get("digit_ratio", 0.0),
            underlined_ratio=row.get("underlined_ratio", 0.0),
            width_ratio=row.get("width_ratio", 0.0),
            has_vertical_line=row.get("has_vertical_line", False),
            median_x0x1_gap=row.get("median_x0x1_gap", 0.0),
            gap_ratio=row.get("gap_ratio", 0.0),
            capitalized_word_ratio=row.get("capitalized_word_ratio", 0.0),
            cell_count=row.get("cell_count", 0),
        ),
        axis=1
    )
    
    return df


# =====================
# STEP 4: Merge Score Back to Cells
# =====================

def _merge_score_to_cells(
    cells_df: pd.DataFrame,
    temp_lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge table_row_score from temp_lines back onto cells.
    
    Each cell gets the table_row_score of its parent temp_line.
    
    Args:
        cells_df: Original cell-level DataFrame
        temp_lines_df: Line-level DataFrame with table_row_score
    
    Returns:
        cells_df with table_row_score column added
    """
    if cells_df.empty or temp_lines_df.empty:
        cells_out = cells_df.copy()
        cells_out["table_row_score"] = pd.Series(dtype="float64")
        return cells_out
    
    # Extract just the temp_line_id and table_row_score columns
    score_mapping = temp_lines_df[["temp_line_id", "table_row_score"]].copy()
    
    # Merge onto cells_df
    cells_out = cells_df.merge(
        score_mapping,
        on="temp_line_id",
        how="left"
    )
    
    # Fill any missing scores with 0.0 (shouldn't happen if data is consistent)
    cells_out["table_row_score"] = cells_out["table_row_score"].fillna(0.0)
    
    return cells_out


# =====================
# Public API for orchestrator
# =====================

def build_temp_lines(
    cells_df: pd.DataFrame,
    words_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to build temp_lines from cells_df with table scoring.
    
    Args:
        cells_df: DataFrame with cell-level data including temp_line_id
        words_df: DataFrame with word-level data (needed for gap calculations)
    
    Returns:
        tuple: (temp_lines_df, cells_df_with_score)
            - temp_lines_df: DataFrame with line-level aggregates and table row scores
            - cells_df_with_score: Original cells_df with table_row_score merged from temp_lines
    """
    if cells_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. Build temp_lines_df by pivoting on temp_line_id and recomputing geometry
    temp_lines_df = _build_temp_lines_df(cells_df)

    # 2. Compute Ratios (needs original words_df for gap calculations)
    temp_lines_df = _compute_ratios(temp_lines_df, words_df)

    # 3. Compute Table Row Score
    temp_lines_df = _compute_table_row_score(temp_lines_df)

    # 4. Merge table_row_score back onto cells_df
    cells_df_with_score = _merge_score_to_cells(cells_df, temp_lines_df)

    return temp_lines_df, cells_df_with_score
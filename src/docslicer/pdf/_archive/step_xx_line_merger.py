"""
step_10_line_merger.py

Merge cells into lines based on layout type:
  - table: merge by row_start
  - text_singlecol: merge by temp_line_id
  - text_multicol: merge by temp_line_id within each col_start

Output ordering:
  - Lines are sorted: page_number → layout_id → layout-specific order
  - Layout-specific order preserves column reading order:
    * multicol: left column (col_start=0) before right column (col_start=1)
    * table: by row_start
    * singlecol: by temp_line_id
  - line_id is assigned sequentially (1, 2, 3...) in this order
  - Ensures correct reading order (no y-position interleaving in multi-column)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import re


# =====================
# Helpers
# =====================

def _remove_bracketed_text(text: str) -> str:
    """
    Remove text within brackets (parentheses, square brackets, curly braces).
    Used for uppercase detection to ignore mixed-case content in brackets.
    
    Example: "RECENT NOTABLE DEVELOPMENTS (Since August 5, 2025)" -> "RECENT NOTABLE DEVELOPMENTS "
    """
    if not text:
        return text
    # Remove text within parentheses, square brackets, and curly braces
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    return text


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


def _assign_line_grouping_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a 'line_group_key' column based on layout_type:
    
    - table: group by (page_number, layout_id, row_start)
    - text_singlecol: group by (page_number, layout_id, temp_line_id)
    - text_multicol: group by (page_number, layout_id, col_start, temp_line_id)
    
    Returns a copy of df with 'line_group_key' added.
    """
    df = df.copy()
    
    # Initialize line_group_key as empty string
    df["line_group_key"] = ""
    
    # Handle tables: group by row_start
    if "layout_type" in df.columns and "row_start" in df.columns:
        mask_table = df["layout_type"] == "table"
        df.loc[mask_table, "line_group_key"] = (
            df.loc[mask_table, "page_number"].astype(str) + "|" +
            df.loc[mask_table, "layout_id"].astype(str) + "|" +
            "table|" +
            df.loc[mask_table, "row_start"].astype(str)
        )
    
    # Handle text_singlecol: group by temp_line_id
    if "layout_type" in df.columns and "temp_line_id" in df.columns:
        mask_singlecol = df["layout_type"] == "text_singlecol"
        df.loc[mask_singlecol, "line_group_key"] = (
            df.loc[mask_singlecol, "page_number"].astype(str) + "|" +
            df.loc[mask_singlecol, "layout_id"].astype(str) + "|" +
            "singlecol|" +
            df.loc[mask_singlecol, "temp_line_id"].astype(str)
        )
    
    # Handle text_multicol: group by col_start + temp_line_id
    if "layout_type" in df.columns and "col_start" in df.columns and "temp_line_id" in df.columns:
        mask_multicol = df["layout_type"] == "text_multicol"
        df.loc[mask_multicol, "line_group_key"] = (
            df.loc[mask_multicol, "page_number"].astype(str) + "|" +
            df.loc[mask_multicol, "layout_id"].astype(str) + "|" +
            "multicol|" +
            df.loc[mask_multicol, "col_start"].astype(str) + "|" +
            df.loc[mask_multicol, "temp_line_id"].astype(str)
        )
    
    return df


# =====================
# Alignment Detection
# =====================

def _compute_alignment_features(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Compute layout_align and text_align for each line group.
    
    Returns a DataFrame with columns: line_group_key, layout_align, text_align
    that can be merged to the aggregated lines DataFrame.
    
    Logic:
      - layout_align: position of entire layout relative to page content area
        {left, center, right, full-width}
      - text_align: alignment of text within a column
        {left, center, right, justified}
      - Single-line layouts inherit layout_align as text_align
    
    Thresholds:
      - full-width: layout takes up > 90% of content width
      - left: left edge within 5% AND not right-aligned (prioritized over center)
      - right: right edge within 5% AND not left-aligned (prioritized over center)
      - center: center within 10% AND not touching either edge
      - justified: both edges align (std dev < 2pt for both x_left and x_right)
    
    Note:
      - Edge alignment is checked BEFORE center alignment
      - A line at the left edge that's 80% wide is "left", not "center"
    """
    if df_cells.empty or "line_group_key" not in df_cells.columns:
        return pd.DataFrame(columns=["line_group_key", "layout_align", "text_align"])
    
    results = []
    
    # Process each page separately
    for page_number, page_df in df_cells.groupby("page_number", sort=False):
        # Compute page content area (min/max x across all cells on page)
        page_x_min = page_df["x_left"].min()
        page_x_max = page_df["x_right"].max()
        page_content_width = page_x_max - page_x_min
        page_content_center = (page_x_min + page_x_max) / 2
        
        if page_content_width == 0:
            # Degenerate case: all cells at same x position
            page_content_width = 1.0
        
        # Process each layout on this page
        for layout_id, layout_df in page_df.groupby("layout_id", sort=False):
            # Layout bounds
            layout_x_min = layout_df["x_left"].min()
            layout_x_max = layout_df["x_right"].max()
            layout_width = layout_x_max - layout_x_min
            layout_center = (layout_x_min + layout_x_max) / 2
            
            # Determine layout_align
            width_ratio = layout_width / page_content_width
            left_offset = (layout_x_min - page_x_min) / page_content_width
            right_offset = (page_x_max - layout_x_max) / page_content_width
            center_offset = abs(layout_center - page_content_center) / page_content_width
            
            # Prioritize edge alignment over width or center
            # Check if aligned to left or right edge first
            left_aligned = left_offset < 0.05
            right_aligned = right_offset < 0.05
            
            if width_ratio > 0.90:
                # Full width: takes up almost entire content area
                layout_align = "full-width"
            elif left_aligned and not right_aligned:
                # Starts at left edge but doesn't reach right edge
                layout_align = "left"
            elif right_aligned and not left_aligned:
                # Ends at right edge but doesn't start at left edge
                layout_align = "right"
            elif center_offset < 0.10 and not left_aligned and not right_aligned:
                # Centered AND not touching either edge
                layout_align = "center"
            else:
                # Default: use whichever edge is closer
                if left_offset < right_offset:
                    layout_align = "left"
                else:
                    layout_align = "right"
            
            # Count lines in this layout (for single-line check)
            line_count = layout_df["line_group_key"].nunique()
            
            # Process each column within layout (or entire layout if no columns)
            # Group by col_start if it exists, otherwise treat as single column
            if "col_start" in layout_df.columns and layout_df["col_start"].notna().any():
                column_groups = layout_df.groupby("col_start", sort=False)
            else:
                # No col_start: treat entire layout as one column
                column_groups = [(None, layout_df)]
            
            for col_start, col_df in column_groups:
                # Determine text_align for this column
                if line_count == 1:
                    # Single-line layout: inherit layout_align
                    text_align = layout_align
                else:
                    # Multi-line: analyze alignment within column
                    col_line_groups = col_df.groupby("line_group_key", sort=False)
                    
                    # Collect x_left and x_right for each line
                    line_x_lefts = []
                    line_x_rights = []
                    for _, line_group in col_line_groups:
                        line_x_lefts.append(line_group["x_left"].min())
                        line_x_rights.append(line_group["x_right"].max())
                    
                    if len(line_x_lefts) < 2:
                        # Only one line in column: inherit layout_align
                        text_align = layout_align
                    else:
                        # Check alignment consistency
                        left_std = np.std(line_x_lefts)
                        right_std = np.std(line_x_rights)
                        
                        # Thresholds (in points)
                        aligned_threshold = 2.0  # < 2pt std = aligned
                        
                        left_aligned = left_std < aligned_threshold
                        right_aligned = right_std < aligned_threshold
                        
                        if left_aligned and right_aligned:
                            text_align = "justified"
                        elif left_aligned:
                            text_align = "left"
                        elif right_aligned:
                            text_align = "right"
                        else:
                            # Neither edge aligns: check which is more consistent
                            if left_std < right_std:
                                text_align = "left"
                            else:
                                text_align = "center"  # ragged on both sides
                
                # Assign to all line groups in this column
                for line_group_key in col_df["line_group_key"].unique():
                    results.append({
                        "line_group_key": line_group_key,
                        "layout_align": layout_align,
                        "text_align": text_align,
                    })
    
    result_df = pd.DataFrame(results)
    
    # Defensive check: ensure no duplicate line_group_keys
    # This shouldn't happen if logic is correct, but prevents data corruption if it does
    if not result_df.empty:
        dup_count = result_df["line_group_key"].duplicated().sum()
        if dup_count > 0:
            # Log warning - this indicates a bug in the alignment logic
            print(f"Warning: Found {dup_count} duplicate line_group_keys in alignment calculation. Deduplicating.")
            result_df = result_df.drop_duplicates(subset=["line_group_key"], keep="first")
    
    return result_df


# =====================
# STEP 1: Build lines_df
# =====================

def _build_lines_df(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Group cells into lines based on layout type.
    
    Grouping strategy:
      - table: group by row_start (each table row becomes a line)
      - text_singlecol: group by temp_line_id
      - text_multicol: group by col_start + temp_line_id (each column processed separately)

    Aggregation rules:
      - Geometry: min/max to get bounding box
      - Text: join cell texts with a space
      - Cells: create bracketed representation of each cell's text
      - Counts: sum the per-cell counts
      - bold_ratio / italic_ratio: weighted by char_count
      - font_name / colors: most prevalent (mode; fall back to first)
      - Flags: use max for boolean flags
    
    Line ID assignment:
      - Lines are sorted by document order: page_number → layout_id → within-layout order
      - Within-layout order preserves the grouping logic:
        * multicol: left column (col_start=0) before right column (col_start=1)
        * table: rows by row_start (top to bottom)
        * singlecol: lines by temp_line_id (top to bottom)
      - line_id is assigned sequentially (1, 2, 3...) in this sorted order
      - This ensures correct reading order, even if right column starts higher than left
    """
    if df_cells.empty:
        return pd.DataFrame(
            columns=[
                "line_id",
                "page_number",
                "layout_id",
                "layout_type",
                "block_role",
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
                "link_url",
                "link_dest",
                "link_type",
                "is_underlined",
                "has_vertical_line",
                "inside_rect_shape",
                "background_non_stroking_color",
                "background_stroking_color",
                "table_id",
                "table_type",
                "row_start",
                "col_start",
                "layout_align",
                "text_align",
                "is_bold",
                "is_italic",
                "is_uppercase",
                "font_size_ratio",
                "page_label",
                "page_label_type",
                "page_label_value",
            ]
        )

    # Assign line grouping key based on layout type
    df = _assign_line_grouping_key(df_cells)
    
    # Remove rows with no grouping key (shouldn't happen but be safe)
    df = df[df["line_group_key"] != ""].copy()
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "line_id",
                "page_number",
                "layout_id",
                "layout_type",
                "block_role",
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
                "link_url",
                "link_dest",
                "link_type",
                "is_underlined",
                "has_vertical_line",
                "inside_rect_shape",
                "background_non_stroking_color",
                "background_stroking_color",
                "table_id",
                "table_type",
                "row_start",
                "col_start",
                "layout_align",
                "text_align",
                "is_bold",
                "is_italic",
                "is_uppercase",
                "font_size_ratio",
                "page_label",
                "page_label_type",
                "page_label_value",
            ]
        )

    # Ensure optional flag/feature columns exist (PDFs without links/underlines/etc won't have these)
    optional_bool_cols = ["has_link", "is_underlined", "has_vertical_line", "inside_rect_shape"]
    for col in optional_bool_cols:
        if col not in df.columns:
            df[col] = False
    
    optional_str_cols = ["link_url", "link_dest", "link_type", "background_non_stroking_color", "background_stroking_color"]
    for col in optional_str_cols:
        if col not in df.columns:
            df[col] = None
    
    # Ensure col_start and row_start exist (used for table positioning and multicol tracking)
    if "col_start" not in df.columns:
        df["col_start"] = 0  # Default to 0 for text_singlecol
    if "row_start" not in df.columns:
        df["row_start"] = None

    # Sort cells within each line group left→right, top→bottom
    df = df.sort_values(
        ["line_group_key", "x_left", "y_top"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Helper: approximate bold/italic char counts for weighted ratios
    df["bold_char_est"] = df["bold_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    df["italic_char_est"] = df["italic_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)

    # Build aggregation spec
    agg_spec: Dict[str, Any] = {
        "page_number": "first",
        "layout_id": "first",
        "layout_type": "first",
        "block_role": "first",
        "page_width": "first",
        "page_height": "first",
        "page_label": "first",
        "page_label_type": "first",
        "page_label_value": "first",
        
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
        
        # Link details
        "link_url": _mode_or_first,
        "link_dest": _mode_or_first,
        "link_type": _mode_or_first,
        
        # Rectangle/background info
        "inside_rect_shape": "max",
        "background_non_stroking_color": _mode_or_first,
        "background_stroking_color": _mode_or_first,
        
        # Table info and column positioning
        "table_id": "first",
        "table_type": "first",
        "row_start": "first",  # For tables: which row this line represents
        "col_start": "min",    # For tables: leftmost cell's column; for multicol: column index (0=left, 1=right, etc)

        "cell_id": "count",  # Will be renamed to cell_count
        "text": lambda s: " ".join(t for t in s.astype(str) if t and str(t).strip()),
    }

    # Group and aggregate
    grouped = df.groupby("line_group_key", sort=True, observed=True).agg(agg_spec).reset_index()

    # Rename cell_id count to cell_count
    grouped = grouped.rename(columns={"cell_id": "cell_count"})
    
    # Create cells column by re-aggregating text with brackets
    cells_text = df.groupby("line_group_key", sort=False)["text"].apply(
        lambda s: " ".join(f"[{text}]" for text in s.astype(str) if text and str(text).strip())
    ).reset_index()
    cells_text = cells_text.rename(columns={"text": "cells"})
    
    # Merge cells column into grouped dataframe
    grouped = grouped.merge(cells_text, on="line_group_key", how="left")

    # Recompute geometry
    grouped["width"] = grouped["x_right"] - grouped["x_left"]
    grouped["height"] = grouped["y_bottom"] - grouped["y_top"]

    # Recompute ratios from estimated bold/italic chars
    total_chars = grouped["char_count"].replace(0, np.nan)
    grouped["bold_ratio"] = (grouped["bold_char_est"] / total_chars).fillna(0.0)
    grouped["italic_ratio"] = (grouped["italic_char_est"] / total_chars).fillna(0.0)

    # Drop helper columns (but keep line_group_key temporarily for ordering)
    grouped = grouped.drop(columns=["bold_char_est", "italic_char_est"])
    
    # ===== Calculated Properties =====
    
    # is_bold: True if bold_ratio > 0.75
    grouped["is_bold"] = grouped["bold_ratio"] > 0.75
    
    # is_italic: True if italic_ratio > 0.75
    grouped["is_italic"] = grouped["italic_ratio"] > 0.75
    
    # is_uppercase: True if uppercase_count / alpha_count > 0.90
    # Exclude text within brackets when calculating uppercase ratio
    def calc_uppercase_ratio(text):
        if not text or not isinstance(text, str):
            return 0.0
        cleaned = _remove_bracketed_text(text)
        alpha_chars = [c for c in cleaned if c.isalpha()]
        if not alpha_chars:
            return 0.0
        uppercase_chars = [c for c in alpha_chars if c.isupper()]
        return len(uppercase_chars) / len(alpha_chars)
    
    grouped["is_uppercase"] = grouped["text"].apply(lambda t: calc_uppercase_ratio(t) > 0.90)
    
    # font_size_ratio: ratio of this line's font_size vs document median
    # Calculate document median font_size (across all lines)
    doc_median_font_size = grouped["font_size"].median()
    if pd.notna(doc_median_font_size) and doc_median_font_size > 0:
        grouped["font_size_ratio"] = grouped["font_size"] / doc_median_font_size
    else:
        # Fallback if no valid median (all None or 0)
        grouped["font_size_ratio"] = 1.0
    
    # The groupby(sort=True) already sorted lines correctly within each layout:
    # - multicol: by col_start, then temp_line_id (left col, then right col)
    # - table: by row_start (top to bottom)
    # - singlecol: by temp_line_id (top to bottom)
    # We need to preserve this order while ensuring page_number and layout_id are sorted.
    
    # Add a temporary column to capture the order within each page+layout group
    grouped["_line_order_within_layout"] = grouped.groupby(
        ["page_number", "layout_id"], sort=False
    ).cumcount()
    
    # Sort by document order: page_number → layout_id → within-layout order
    # This preserves the column ordering from groupby while sorting pages and layouts
    grouped = grouped.sort_values(
        ["page_number", "layout_id", "_line_order_within_layout"],
        kind="mergesort"
    ).reset_index(drop=True)
    
    # Compute alignment features (from original cells, merge to lines)
    # Do this BEFORE dropping line_group_key so we can merge
    alignment_df = _compute_alignment_features(df)
    if not alignment_df.empty:
        grouped = grouped.merge(alignment_df, on="line_group_key", how="left")
    else:
        grouped["layout_align"] = None
        grouped["text_align"] = None
    
    # Drop temporary columns
    grouped = grouped.drop(columns=["line_group_key", "_line_order_within_layout"])
    
    # Assign line_id (sequential, now in correct document order)
    grouped.insert(0, "line_id", range(1, len(grouped) + 1))

    return grouped


# =====================
# Public API
# =====================

def merge_cells_to_lines(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cells into lines based on layout type.
    
    Layout type handling:
      - table: Each row_start becomes a line
      - text_singlecol: Each temp_line_id becomes a line
      - text_multicol: Each temp_line_id per col_start becomes a line
        (columns are stacked vertically in output)
    
    Output ordering:
      - Lines are returned in document order: page_number → layout_id → layout-specific order
      - Layout-specific ordering:
        * multicol: left column first, then right column (by col_start, then temp_line_id)
        * table: by row_start (top to bottom)
        * singlecol: by temp_line_id (top to bottom)
      - line_id increases sequentially (1, 2, 3...) through the document
      - Page 1 lines have lower line_ids than page 2 lines
      - Respects column order in multi-column layouts (no interleaving by y position)
    
    Args:
        cells_df: DataFrame with cell-level data
    
    Returns:
        DataFrame with line-level data, sorted in document order
    """
    return _build_lines_df(cells_df)
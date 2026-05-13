"""
reorder_columns.py

Centralized column ordering for all DataFrames in the PDF pipeline.

This module defines the "ground truth" column order for all DataFrames
exported by the PDF pipeline, enabling consistent CSV inspection without
per-script reordering logic.

Usage:
    from utils.reorder_columns import reorder_columns
    
    df_ordered = reorder_columns(df, dataframe_type="cells")
"""

from __future__ import annotations
import pandas as pd
from typing import List, Literal

# Type for specifying which DataFrame we're reordering
DataFrameType = Literal[
    "words",
    "boxes",
    "shapes", 
    "links",
    "cells",
    "temp_lines",
    "table_cells",
    "lines",
]


# =====================
# Master Column Order
# =====================
# This is the comprehensive "pick and mix" order that includes
# all columns from all DataFrames in the pipeline.
# 
# Columns are ordered by logical groups:
#   1. Document/Page identifiers
#   2. IDs and hierarchy
#   3. Content
#   4. Geometry
#   5. Style
#   6. Calculated features
#   7. Page context
#   8. Relationships (links, rects, underlines, vertical lines)
#   9. Table-specific fields
#   10. Layout-specific fields
#   11. Band/Line statistics
#   12. Scores
#   13. Other metadata

MASTER_COLUMN_ORDER: List[str] = [
    # ===== 1. Document/Page identifiers =====
    "page_number",
    "page_label",
    "document_region",
    "section_id", # Only for .docx
    
    # ===== 2. IDs and hierarchy =====
    
    # Main IDs
        # -- Raw data
    "word_id",
    "box_id",
    "image_id",
    "raw_shape_id",
    "shape_id",
    "link_id",
    "run_id",
        # -- Processed in HTML and PDF
    "gutter_id",
    "gutter_candidate_id",
    "cell_id",
    "line_id",
    "paragraph_id",
        # -- Processed in Shared
    "block_id",
    "chunk_index",
    "chunk_id",
    

    # Secondary IDs
    "structure_tag",
    "structure_tag_id",
    "temp_line_id",
    "horizontal_band_id",

    # Layout IDs
    "layout_id",
    "layout_type",
    "block_role",
    
    # Table IDs
    "table_id",
    "table_row_id",
    "table_cell_id",
    "table_header_flag",

    "row_start",
    "col_start",

    # Shape Core
    "raw_shape_type",
    "shape_type",
    "orientation",
    
    # Enhanced shape IDs
    "enhanced_shape_id",
    "enhanced_shape_id_underline",
    "enhanced_shape_id_container",

    # TEMP PAGE LABELS
    "page_label_raw",
    "page_label_candidate",
    "page_label_type",
    "page_label_value",
    "page_label_cell_sharing",
    "page_label_wrapper",
    "page_label_score",
    
    # ===== 3. Content =====
    "chunk_heading",
    "chunk_path",
    "text",
    "token_count",
    "embed_char_count",
    "active_heading_id",
    "block_count",
    "cells",
    "cell_ids",
    "temp_line_ids",

    # Hierarchy
    "heading_score",
    "hierarchy_marker",
    "hierarchy_type",
    "heading_type",
    "heading_id",
    "heading_fp_id",
    "heading_fingerprint",
    "heading_hash",
    "heading_weight_static",
    "heading_weight_dynamic",
    "heading_sequence",
    "heading_level",
    "parent_heading_id",

    
    # Cell count
    "cell_count",
    
    # ===== 4. Geometry =====
    "x_left",
    "x_right",
    "y_top",
    "y_bottom",
    "top_bucket",
    "width",
    "height",
    "area",

    # Gutter columns ### TEMPORARY ###
    "sliding_window_id",
    "sliding_window",
    "x_page_min",
    "x_page_max",
    "gutter_candidate_x_left",
    "gutter_candidate_x_right",
    
    # ===== 5. Style =====
    "font_name",
    "font_family",
    "font_size",
    "font_size_ratio",
    "font_weight",
    "text_align",
    "layout_align",
    "text_orientation",
    "non_stroking_color",
    "stroking_color",
    "linewidth",
    
    # Paint state (shapes)
    "fill",
    "stroke",
    "paint_op",

    # Merged IDs
    "word_ids",
    "cell_ids",
    
    # ===== 6. Calculated features =====
    # Counts
    "char_count",
    "alpha_count",
    "digit_count",
    "uppercase_count",
    "word_count",
    "alpha_token_count",
    "capitalized_token_count",
    
    # Ratios
    "bold_ratio",
    "italic_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "capitalized_token_ratio",
    "underlined_ratio",
    "inside_rect_fraction",
    
    # Boolean flags (derived from ratios/features)
    "is_bold",
    "is_italic",
    "is_uppercase",
    
    # Average metrics
    "avg_word_len",
    "avg_cell_count",
    
    # ===== 7. Page context =====
    "page_width",
    "page_height",
    
    # ===== 8. Relationships =====
    # Links
    "has_link",
    "link_url",
    "link_dest",
    "link_type",
    "ixbrl_id",
    "html_data_attrs",

    # Rectangles/shapes
    "is_underlined",
    "has_vertical_line",
    "inside_rect_shape",
    "background_non_stroking_color",
    "background_stroking_color",
    
    # HTML Provenance
    "dom_id",
    "dom_class",
    "wrapping_tag",
    "split_reason",
    "ancestor_ids",
    "ancestor_classes",
    "ancestor_tags",
    "ancestor_aria_roles",

    
    
    
    # ===== 9. Table-specific fields =====
    "cell_role",
    "row_index",
    "col_index",
    "rowspan",
    "colspan",
    
    # ===== 11. Band/Line statistics =====
    "line_gap",
    "line_gap_to_prev",
    "page_median_gap",
    "page_gap_thresh",
    
    # Column/gap structure
    "median_x0x1_gap",
    "max_x0x1_gap",
    "gap_ratio",
    "width_ratio",
    
    # ===== 12. Scores =====
    "table_row_score",
    "table_header_score",
    "table_title_score",
    "avg_table_row_score",
    "average_table_score",
]


# =====================
# Reorder Function
# =====================

def reorder_columns(
    df: pd.DataFrame,
    dataframe_type: DataFrameType | None = None,
) -> pd.DataFrame:
    """
    Reorder columns in a DataFrame based on the master column order.
    
    This function:
      1. Picks columns from MASTER_COLUMN_ORDER that exist in df
      2. Appends any other columns not in MASTER_COLUMN_ORDER
      3. Returns df with reordered columns
    
    Args:
        df: DataFrame to reorder
        dataframe_type: Optional type hint for documentation purposes
                       (not currently used for logic, but useful for
                       type checking and future customization)
    
    Returns:
        DataFrame with reordered columns
    
    Examples:
        >>> df_words_ordered = reorder_columns(df_words, "words")
        >>> df_cells_ordered = reorder_columns(df_cells, "cells")
    """
    if df.empty:
        return df
    
    # Pick columns that exist in both master order and dataframe
    existing_cols = [col for col in MASTER_COLUMN_ORDER if col in df.columns]
    
    # Find any columns not in master order (catch-all for new/unexpected columns)
    other_cols = [col for col in df.columns if col not in MASTER_COLUMN_ORDER]
    
    # Reorder: known columns first, then unknown columns
    return df[existing_cols + other_cols]

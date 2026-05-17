"""
DataFrame export utilities for debug and production outputs.

Provides standardized DataFrame preprocessing before export:
- Column reordering
- Float rounding
- Excel formula escaping
- Production column filtering
"""

from typing import List, Optional
import pandas as pd
import logging

_log = logging.getLogger(__name__)

from .reorder_columns import reorder_columns


# Columns to drop in production exports (internal/debug fields)
PRODUCTION_DROP_COLS = [
    # Debug/internal fields
    "block_count",
    "heading_fp_id",
    "heading_fingerprint",
    "font_name",
    "font_size_ratio",
    "stroking_color",
    "alpha_count",
    "digit_count",
    "uppercase_count",
    "alpha_token_count",
    "capitalized_token_count",
    "bold_ratio",
    "italic_ratio",
    "is_bold",
    "is_italic",
    "is_uppercase",
    "has_link",
    "link_dest",
    "inside_rect_shape",
    "background_stroking_color",
    "is_underlined",
    "has_vertical_line",
    "contains_table",
    "merged_chunk_id",
    "merge_mode",
    "merge_group_parent_heading_id",
    "merge_member_heading_ids",
    "active_heading_id",
    "word_count",
    "alpha_word_count",
    "capitalized_word_count",
    "font_family",
    "font_size",
    "font_weight",
    "font_name",
    "text_align",
    "underlined_ratio",
    "non_stroking_color",
    "table_row_id",
    "layout_id",
    "table_header_flag",
    # Metadata fields we don't want in UI
    "author_meta",
    "author_text",
    "title_meta",
    "title_text",
    "language_confidence",
    "language_source",
    "rendered_html",
    "has_mixed_page_sizes",
    "page_format",
    "page_count",
    "chars",
    "total_chars",
    "estimated_tokens",
    "is_scanned"
]


def export_debug(df: pd.DataFrame, drop_none: bool = False) -> pd.DataFrame:
    """
    Prepare DataFrame for debug export (CSV inspection).
    
    Operations:
    - Reorders columns by priority (identity, geometry, content, etc.)
    - Converts object columns with numeric values to float (handles pd.NA columns)
    - Rounds all float columns to 2 decimals (preserves ints)
    - Escapes Excel formula characters in text fields
    - Optionally drops None-only columns
    
    Args:
        df: Input DataFrame
        drop_none: If True, drops columns that are all None/NaN
    
    Returns:
        Processed DataFrame ready for export
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Drop None-only columns if requested
    if drop_none:
        df = df.dropna(axis=1, how='all')
    
    # Reorder columns for better readability
    df = reorder_columns(df)
    
    # Convert object columns containing numeric values to float
    # (handles columns initialized with pd.NA that later get numeric values)
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                # Try to convert to numeric, coerce errors to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # If all non-null values successfully converted, use numeric version
                if numeric_col.notna().sum() == df[col].notna().sum():
                    df[col] = numeric_col
        except (AttributeError, KeyError):
            # Skip columns that can't be accessed normally
            continue
    
    # Round all float columns to 2 decimals (keeps ints as ints).
    # Use positional indexing to handle DataFrames with duplicate column names.
    float_dtypes = {'float64', 'float32', 'Float64', 'Float32'}
    for i, dt in enumerate(df.dtypes):
        if str(dt) in float_dtypes:
            df.iloc[:, i] = df.iloc[:, i].round(2)

    # Escape Excel-style formulas in text column
    if "text" in df.columns:
        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .apply(lambda s: "'" + s if s.startswith(("+", "-", "=")) else s)
        )

    return df


def export_production(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    drop_none: bool = True
) -> pd.DataFrame:
    """
    Prepare DataFrame for production export (API/end users).
    
    Operations:
    - Drops internal/debug columns (customizable list)
    - Reorders remaining columns by priority
    - Converts object columns with numeric values to float (handles pd.NA columns)
    - Rounds all float columns to 2 decimals
    - Escapes Excel formula characters in text fields
    - Drops None-only columns by default
    
    Args:
        df: Input DataFrame
        drop_cols: Custom list of columns to drop (uses PRODUCTION_DROP_COLS if None)
        drop_none: If True, drops columns that are all None/NaN (default: True)
    
    Returns:
        Processed DataFrame ready for production export
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Drop production columns (internal/debug fields)
    if drop_cols is None:
        drop_cols = PRODUCTION_DROP_COLS
    
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Drop None-only columns if requested
    if drop_none:
        df = df.dropna(axis=1, how='all')
    
    # Reorder columns for better readability
    df = reorder_columns(df)
    
    # Convert object columns containing numeric values to float
    # (handles columns initialized with pd.NA that later get numeric values)
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                # Try to convert to numeric, coerce errors to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # If all non-null values successfully converted, use numeric version
                if numeric_col.notna().sum() == df[col].notna().sum():
                    df[col] = numeric_col
        except (AttributeError, KeyError):
            # Skip columns that can't be accessed normally
            continue
    
    # Round all float columns to 2 decimals (keeps ints as ints).
    # Use positional indexing to handle DataFrames with duplicate column names.
    float_dtypes = {'float64', 'float32', 'Float64', 'Float32'}
    for i, dt in enumerate(df.dtypes):
        if str(dt) in float_dtypes:
            df.iloc[:, i] = df.iloc[:, i].round(2)
    
    # Escape Excel-style formulas in text column
    if "text" in df.columns:
        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .apply(lambda s: "'" + s if s.startswith(("+", "-", "=")) else s)
        )
    
    return df

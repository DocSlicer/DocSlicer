"""
hierarchical_aggregator.py

Shared aggregation logic for hierarchical document structures (words → cells → lines → blocks → chunks).

Generic pattern:
  1. Input DF has a grouping ID column (e.g., "_cell_id", "_line_group_key", "_block_id")
  2. Aggregate lower-level features to higher level using consistent rules
  3. Recompute derived geometry (width, height)
  4. Recompute weighted ratios (bold_ratio, italic_ratio)
  5. Compute derived flags (is_bold, is_italic, is_uppercase, font_size_ratio)

This module extracts the boilerplate so cell/line/block builders only handle:
  - Decision logic (what constitutes a cell/line/block?)
  - Text merging strategy (space vs newline vs table markdown)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np

# TODO: Maybe add a param to skip certain columns a certain step does not need

# =============================================================================
# HELPERS
# =============================================================================

def _mode_or_first(series: pd.Series) -> Any:
    """
    Return the most prevalent value, or first non-null if all unique.
    
    This is used for aggregating categorical/string columns like font_name, color, etc.
    """
    if series is None or series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if not vc.empty:
        return vc.index[0]
    s2 = series.dropna()
    return s2.iloc[0] if not s2.empty else None


def _collect_unique_list(series: pd.Series) -> Optional[List[Any]]:
    """
    Collect all unique non-null values into a list.
    
    This is used for aggregating fields like ixbrl_id where we want to preserve
    all unique values across the group (e.g., ['f-5', 'f-6', 'f-7']).
    
    If values are already lists (from previous aggregation), they are flattened
    and then deduplicated.
    
    Returns:
        List of unique values, or None if series is empty/all null/only empty strings
    """
    if series is None or series.empty:
        return None
    
    all_values = []
    for val in series.dropna():
        if isinstance(val, list):
            # Flatten lists (e.g., ['f-7', 'f-8'] from previous aggregation)
            all_values.extend(val)
        elif val != "":
            # Single value
            all_values.append(val)
    
    if not all_values:
        return None
    
    # Deduplicate while preserving order
    seen = set()
    unique_vals = []
    for v in all_values:
        if v not in seen and v != "":
            seen.add(v)
            unique_vals.append(v)
    
    return unique_vals if unique_vals else None


def _merge_dicts(series: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Merge all dictionaries in the series into a single dictionary.
    
    This is used for aggregating fields like html_data_attrs where each row
    might have {"key1": "val1"} and we want to consolidate into one dict.
    Later values overwrite earlier ones for duplicate keys.
    
    Returns:
        Merged dictionary, or None if series is empty/all null
    """
    if series is None or series.empty:
        return None
    
    merged = {}
    for val in series.dropna():
        if isinstance(val, dict):
            merged.update(val)
        elif val:  # Handle string representations of dicts if needed
            # Could add json.loads here if string dicts are common
            pass
    
    return merged if merged else None


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure optional columns exist with default values.
    Modifies df in place and returns it for chaining.
    """
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


# =============================================================================
# STANDARD AGGREGATION SPEC BUILDERS
# =============================================================================

def build_standard_agg_spec(
    identity_cols: Optional[List[str]] = None,
    include_hierarchy: bool = True,
    include_geometry: bool = True,
    include_style: bool = True,
    include_counts: bool = True,
    include_metadata: bool = True,
    include_table: bool = False,
    include_html_provenance: bool = False,
    extra_first: Optional[List[str]] = None,
    extra_agg: Optional[Dict[str, Any]] = None,
    count_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standard aggregation spec for hierarchical aggregation.
    
    Args:
        identity_cols: Identity columns to keep (using "first"). 
                      Default: ["doc_name", "page_number", "layout_id", "layout_type", "block_role",
                                "document_region", "page_*"]
        include_geometry: Include x/y bounding box aggregation
        include_counts: Include char/alpha/digit/token count aggregation
        include_style: Include font/color style aggregation
        include_flags: Include boolean flags (link, underline, etc.)
        include_table: Include table-specific columns
        include_alignment: Include alignment columns
        include_hierarchy: Include heading hierarchy columns
        extra_first: Additional columns to aggregate with "first"
        extra_agg: Additional custom aggregations (merged into spec)
        count_col: Column to count (e.g., "cell_id" → will be renamed to "{count_col}_count")
    
    Returns:
        Dict suitable for df.groupby().agg(spec)
    """
    spec: Dict[str, Any] = {}
    
    # Identity columns (take first value in group)
    if identity_cols is None:
        identity_cols = [
            "page_number", # TODO: This may become a list if chunk spans multiple pages
            "layout_id",
            "layout_type",
            "block_role",
            "document_region",
            "page_width",
            "page_height",
            "page_format", # US Letter, A4, etc.
            "page_label", # TODO: This may become a list if chunk spans multiple pages
            "page_label_type",
            "page_label_value",
        ]
    
    for col in identity_cols:
        spec[col] = "first"
    
    if extra_first:
        for col in extra_first:
            spec[col] = "first"
    
    # Hierarchy (heading structure from step_04_hierarchy_assigner)
    if include_hierarchy:
        spec.update({
            "document_region": "first",
            "heading_id": "first",
            "parent_heading_id": "first",
            "heading_level": "first",
            "heading_type": "first",
            "heading_fp_id": "first",
            "heading_fingerprint": "first",
            "heading_hash": "first",
        })
    
    # Geometry (bounding box)
    if include_geometry:
        spec.update({
            "x_left": "min",
            "x_right": "max",
            "y_top": "min",
            "y_bottom": "max",
            "layout_align": _mode_or_first,
            "text_align": _mode_or_first,
        })

    # Style (most prevalent)
    if include_style:
        spec.update({
            "font_size": _mode_or_first,
            "font_weight": _mode_or_first,
            "font_name": _mode_or_first,
            "font_family": _mode_or_first,
            "text_orientation": _mode_or_first,
            "non_stroking_color": _mode_or_first,
            "stroking_color": _mode_or_first,
            "background_non_stroking_color": _mode_or_first,
            "background_stroking_color": _mode_or_first,
        })

        spec.update({
            "is_underlined": "max",
            "has_vertical_line": "max",
            "inside_rect_shape": "max",
            
        })
    
    # Counts (sum across group)
    if include_counts:
        spec.update({
            "char_count": "sum",
            "alpha_count": "sum",
            "digit_count": "sum",
            "uppercase_count": "sum",
            "word_count": "sum",
            "alpha_word_count": "sum",
            "capitalized_word_count": "sum",
        })
        
        # Weighted ratio numerators (for bold/italic recomputation)
        # These should be computed before aggregation as: ratio * char_count
        spec.update({
            "_bold_char_est": "sum",
            "_italic_char_est": "sum",
            "_underlined_char_est": "sum",
        })

        
    # Link details (best-effort)
    if include_metadata:
        spec.update({
            "has_link": "max",
            "link_url": _collect_unique_list, 
            "link_dest": _mode_or_first,
            "link_type": _mode_or_first,
            "ixbrl_id": _collect_unique_list,  # Consolidate all unique ixbrl IDs into list
            "html_data_attrs": _merge_dicts,   # Merge all data attributes into single dict
        })
    
    # Table-specific
    if include_table:
        spec.update({
            "table_id": "first",
            "table_row_id": "first", # Needed for Line Merger
            "table_header_flag": "first",
            "table_type": "first",
            "row_start": "first",
            "col_start": "first",
        })
    
    # HTML Provenance
    if include_html_provenance:
        spec.update({
            "structure_tag_id": "first",
            "structure_tag": "first",
            "wrapping_tag": "first",
            "split_reason": "first",
            "dom_id": "first",
            "dom_class": "first",
        })

        spec.update({ #NOTE: TBD if needed
            "ancestor_ids": "first",
            "ancestor_classes": "first",
            "ancestor_tags": "first",
            "ancestor_aria_roles": "first",
        })
    
    # Count column (e.g., "cell_id" for lines, "word_id" for cells)
    if count_col:
        spec[count_col] = "count"

    # Merge any custom aggregations
    if extra_agg:
        spec.update(extra_agg)
    
    return spec


# ================================================================================================================
# AGGREGATION ENGINE
# ================================================================================================================

def aggregate_hierarchical(
    df: pd.DataFrame,
    group_col: str,
    agg_spec: Dict[str, Any],
    rename_group_col: Optional[str] = None,
    rename_count_col: Optional[Dict[str, str]] = None,
    compute_derived: bool = True,
) -> pd.DataFrame:
    """
    Generic hierarchical aggregation with standard post-processing.
    
    Automatically handles:
      - Weighted ratio computation for bold/italic (creates helper columns before aggregation)
      - Filtering agg_spec to only existing columns (pick and mix)
      - Derived geometry (width, height) after aggregation
      - Recomputed weighted ratios (bold_ratio, italic_ratio) after aggregation
      - Derived flags (is_bold, is_italic, is_uppercase, font_size_ratio) after aggregation
    
    Args:
        df: Input dataframe with grouping column already assigned
        group_col: Column name to group by (e.g., "_cell_id", "_block_id")
        agg_spec: Aggregation specification (from build_standard_agg_spec or custom)
                  Note: Only columns that exist in df will be aggregated (pick and mix)
        rename_group_col: What to rename group_col to after reset_index (e.g., "cell_id", "block_id")
        rename_count_col: Dict mapping count columns to new names (e.g., {"cell_id": "cell_count"})
        compute_derived: Whether to compute derived columns (geometry, ratios, flags)
    
    Returns:
        Aggregated dataframe with derived columns added
    """
    df = df.copy()
    
    # -------------------------
    # PREPARATION: Compute weighted ratio numerators for correct aggregation
    # -------------------------
    if "bold_ratio" in df.columns and "char_count" in df.columns:
        df["_bold_char_est"] = df["bold_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    
    if "italic_ratio" in df.columns and "char_count" in df.columns:
        df["_italic_char_est"] = df["italic_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    
    if "underlined_ratio" in df.columns and "char_count" in df.columns:
        df["_underlined_char_est"] = df["underlined_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    
    # -------------------------
    # AGGREGATION: Filter spec to existing columns and aggregate
    # -------------------------
    # Filter agg_spec to only include columns that exist in df (pick and mix)
    # Exclude the group_col itself (it becomes the index and is restored by reset_index)
    filtered_spec = {col: agg_func for col, agg_func in agg_spec.items() 
                     if col in df.columns and col != group_col}
    
    # Aggregate
    grouped = df.groupby(group_col, sort=False, observed=True).agg(filtered_spec).reset_index()
    
    # Rename columns
    renames = {}
    if rename_group_col:
        renames[group_col] = rename_group_col
    if rename_count_col:
        renames.update(rename_count_col)
    
    if renames:
        grouped = grouped.rename(columns=renames)
    
    if not compute_derived:
        return grouped
    
    # -------------------------
    # DERIVED GEOMETRY
    # -------------------------
    if "x_left" in grouped.columns and "x_right" in grouped.columns:
        grouped["width"] = grouped["x_right"] - grouped["x_left"]
    
    if "y_top" in grouped.columns and "y_bottom" in grouped.columns:
        grouped["height"] = grouped["y_bottom"] - grouped["y_top"]
    
    # -------------------------
    # RECOMPUTE WEIGHTED RATIOS
    # -------------------------
    if "_bold_char_est" in grouped.columns and "char_count" in grouped.columns:
        total_chars = grouped["char_count"].replace(0, np.nan)
        grouped["bold_ratio"] = (grouped["_bold_char_est"] / total_chars).fillna(0.0)
        grouped = grouped.drop(columns=["_bold_char_est"])
    
    if "_italic_char_est" in grouped.columns and "char_count" in grouped.columns:
        total_chars = grouped["char_count"].replace(0, np.nan)
        grouped["italic_ratio"] = (grouped["_italic_char_est"] / total_chars).fillna(0.0)
        grouped = grouped.drop(columns=["_italic_char_est"])
    
    if "_underlined_char_est" in grouped.columns and "char_count" in grouped.columns:
        total_chars = grouped["char_count"].replace(0, np.nan)
        grouped["underlined_ratio"] = (grouped["_underlined_char_est"] / total_chars).fillna(0.0)
        grouped = grouped.drop(columns=["_underlined_char_est"])
    
    # -------------------------
    # DERIVED FLAGS
    # -------------------------
    if "bold_ratio" in grouped.columns:
        grouped["is_bold"] = grouped["bold_ratio"] > 0.75
    
    if "italic_ratio" in grouped.columns:
        grouped["is_italic"] = grouped["italic_ratio"] > 0.75
    
    if "underlined_ratio" in grouped.columns:
        grouped["is_underlined"] = grouped["underlined_ratio"] > 0.75

    #TODO: Add new logic for is_uppercase:
    
    if "uppercase_count" in grouped.columns and "alpha_count" in grouped.columns:
        alpha_count_safe = grouped["alpha_count"].replace(0, np.nan)
        uppercase_ratio = (grouped["uppercase_count"] / alpha_count_safe).fillna(0.0)
        grouped["is_uppercase"] = uppercase_ratio > 0.90
    
    # Font size ratio (vs document median)
    if "font_size" in grouped.columns:
        fs = pd.to_numeric(grouped["font_size"], errors="coerce")
        doc_median_fs = float(fs.dropna().median()) if fs.notna().any() else np.nan
        
        if np.isfinite(doc_median_fs) and doc_median_fs > 0:
            grouped["font_size_ratio"] = fs / doc_median_fs
        else:
            grouped["font_size_ratio"] = 1.0
    
    return grouped
# backend/app/services/parsing/_utils/line_merger.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# -----------------------
# CONFIG
# -----------------------

YAlignment = Literal["top", "center", "bottom"]


@dataclass(frozen=True)
class LineMergerConfig:
    # Do not merge rows with these block_roles (case-insensitive)
    blocked_block_roles: tuple[str, ...] = ("image", "hr")
    
    # Vertical tolerances (points)
    TOL_BASE: float = 5.0          # default tolerance on selected y key
    TOL_EXPANDED: float = 8.0      # expanded tolerance when overlap is good
    MIN_VERTICAL_OVERLAP: float = 4.0  # (pt) min overlap to use expanded tol


# -----------------------
# Public API
# -----------------------

def assign_line_id(
    df: pd.DataFrame,
    y_alignment: YAlignment = "center",
    config: LineMergerConfig = LineMergerConfig(),
) -> pd.DataFrame:
    """
    Simple one-pass line assignment:
    1. Process by page_number
    2. Check table_row_id matches -> same line
    3. Check y_alignment tolerance -> same line
    4. Otherwise -> new line
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["line_id"] = pd.Series(dtype="Int64")
        return out

    required = ["page_number", "y_top", "y_bottom"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"assign_line_id: missing required columns: {missing}")

    out = df.copy()
    out["line_id"] = 0  # Initialize all to 0

    # Calculate y_key based on alignment
    y_top = out["y_top"].astype("float64")
    y_bottom = out["y_bottom"].astype("float64")
    
    if y_alignment == "top":
        y_key = y_top
    elif y_alignment == "bottom":
        y_key = y_bottom
    else:
        y_key = (y_top + y_bottom) / 2.0

    has_table_row_id = "table_row_id" in out.columns
    has_block_role = "block_role" in out.columns
    line_counter = 1

    # Process each page
    for page_num in out["page_number"].unique():
        page_mask = out["page_number"] == page_num
        page_indices = out[page_mask].index
        
        if len(page_indices) == 0:
            continue
        
        # Initialize tracking variables
        current_line_id = None
        current_y = None
        current_top = None
        current_bottom = None
        current_table_row_id = None
        
        for idx in page_indices:
            row_y = float(y_key.loc[idx])
            row_top = float(out.at[idx, "y_top"])
            row_bottom = float(out.at[idx, "y_bottom"])
            
            if has_table_row_id:
                row_table_id = out.at[idx, "table_row_id"]
            else:
                row_table_id = None
            
            # Check if row has blocked block_role
            is_blocked = False
            if has_block_role:
                row_block_role = out.at[idx, "block_role"]
                if pd.notna(row_block_role):
                    role_str = str(row_block_role).strip().lower()
                    if role_str in config.blocked_block_roles:
                        is_blocked = True
            
            # Blocked rows get their own line_id
            if is_blocked:
                out.at[idx, "line_id"] = line_counter
                line_counter += 1
                continue
            
            # First row on page
            if current_line_id is None:
                current_line_id = line_counter
                out.at[idx, "line_id"] = current_line_id
                line_counter += 1
                
                current_y = row_y
                current_top = row_top
                current_bottom = row_bottom
                current_table_row_id = row_table_id
                continue
            
            # Check if should merge with current line
            merge = False
            
            # Check table_row_id match
            if has_table_row_id and pd.notna(row_table_id) and row_table_id == current_table_row_id:
                merge = True
            else:
                # Check y tolerance
                dy = abs(row_y - current_y)
                overlap = min(row_bottom, current_bottom) - max(row_top, current_top)
                
                if dy <= config.TOL_BASE:
                    merge = True
                elif dy <= config.TOL_EXPANDED and overlap >= config.MIN_VERTICAL_OVERLAP:
                    merge = True
            
            if merge:
                # Assign to current line
                out.at[idx, "line_id"] = current_line_id
                # Update bounding box
                current_top = min(current_top, row_top)
                current_bottom = max(current_bottom, row_bottom)
            else:
                # Start new line
                current_line_id = line_counter
                out.at[idx, "line_id"] = current_line_id
                line_counter += 1
                
                current_y = row_y
                current_top = row_top
                current_bottom = row_bottom
                current_table_row_id = row_table_id
    
    # Compute bucket columns
    _compute_buckets_inplace(out, y_alignment)
    
    return out


# -----------------------
# Internal helpers
# -----------------------

def _compute_buckets_inplace(out: pd.DataFrame, y_alignment: YAlignment) -> None:
    """
    Compute bucket columns based on line_id grouping:
    - top_bucket: min(y_top) rounded to int
    - bottom_bucket: min(y_bottom) rounded to int
    - center_bucket: true center of line bbox (min y_top, max y_bottom)
    """
    if "line_id" not in out.columns or (out["line_id"] == 0).all():
        # No valid line_ids, add empty bucket column
        if y_alignment == "top":
            out["top_bucket"] = pd.Series(dtype="Int64")
        elif y_alignment == "bottom":
            out["bottom_bucket"] = pd.Series(dtype="Int64")
        else:
            out["center_bucket"] = pd.Series(dtype="Int64")
        return
    
    # Group by line_id (exclude 0 if present)
    valid_mask = out["line_id"] > 0
    if not valid_mask.any():
        if y_alignment == "top":
            out["top_bucket"] = pd.Series(dtype="Int64")
        elif y_alignment == "bottom":
            out["bottom_bucket"] = pd.Series(dtype="Int64")
        else:
            out["center_bucket"] = pd.Series(dtype="Int64")
        return
    
    g = out[valid_mask].groupby("line_id")
    
    if y_alignment == "top":
        top_min = g["y_top"].min()
        top_bucket = top_min.round().astype("Int64")
        out["top_bucket"] = out["line_id"].map(top_bucket)
    
    elif y_alignment == "bottom":
        bottom_min = g["y_bottom"].min()
        bottom_bucket = bottom_min.round().astype("Int64")
        out["bottom_bucket"] = out["line_id"].map(bottom_bucket)
    
    else:
        # Center bucket: true center of merged line bbox
        top_min = g["y_top"].min()
        bottom_max = g["y_bottom"].max()
        center_true = (top_min.astype("float64") + bottom_max.astype("float64")) / 2.0
        center_bucket = center_true.round().astype("Int64")
        out["center_bucket"] = out["line_id"].map(center_bucket)

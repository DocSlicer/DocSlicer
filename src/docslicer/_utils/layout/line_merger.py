# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Group boxes and words into lines by vertical alignment (assign_line_id, same_line)."""

# backend/app/services/parsing/_utils/line_merger.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# -----------------------
# CONFIG
# -----------------------

YAlignment = Literal["top", "center", "bottom"]


@dataclass(frozen=True)
class LineMergerConfig:
    # Do not merge rows with these block_types (case-insensitive)
    blocked_block_types: tuple[str, ...] = ("image", "hr")

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

    # Extract all columns as numpy arrays up front — avoids per-row .at overhead
    y_top_arr    = out["y_top"].to_numpy(dtype=float)
    y_bottom_arr = out["y_bottom"].to_numpy(dtype=float)

    if y_alignment == "top":
        y_key_arr = y_top_arr
    elif y_alignment == "bottom":
        y_key_arr = y_bottom_arr
    else:
        y_key_arr = (y_top_arr + y_bottom_arr) / 2.0

    page_arr = out["page_number"].to_numpy()

    has_table_row_id = "table_row_id" in out.columns
    has_block_type   = "block_type"   in out.columns

    table_row_arr = out["table_row_id"].to_numpy() if has_table_row_id else None
    block_type_arr = out["block_type"].to_numpy()  if has_block_type   else None

    # Pre-compute blocked mask if needed
    if has_block_type:
        blocked_arr = np.zeros(len(out), dtype=bool)
        for i, role in enumerate(block_type_arr):
            if role is not None and not (isinstance(role, float) and np.isnan(role)):
                if str(role).strip().lower() in config.blocked_block_types:
                    blocked_arr[i] = True
    else:
        blocked_arr = None

    # Pre-compute table_row notna mask
    if has_table_row_id:
        table_notna_arr = np.array([
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in table_row_arr
        ], dtype=bool)
    else:
        table_notna_arr = None

    line_id_arr  = np.zeros(len(out), dtype=np.int64)
    line_counter = 1

    for page_num in out["page_number"].unique():
        page_pos = np.where(page_arr == page_num)[0]
        if len(page_pos) == 0:
            continue

        current_line_id      = None
        current_y            = None
        current_top          = None
        current_bottom       = None
        current_table_row_id = None

        for pos in page_pos:
            row_y      = y_key_arr[pos]
            row_top    = y_top_arr[pos]
            row_bottom = y_bottom_arr[pos]

            # Blocked rows get their own line_id
            if blocked_arr is not None and blocked_arr[pos]:
                line_id_arr[pos] = line_counter
                line_counter += 1
                continue

            row_table_id     = table_row_arr[pos]  if has_table_row_id else None
            row_table_notna  = table_notna_arr[pos] if has_table_row_id else False

            # First row on page
            if current_line_id is None:
                current_line_id      = line_counter
                line_id_arr[pos]     = current_line_id
                line_counter        += 1
                current_y            = row_y
                current_top          = row_top
                current_bottom       = row_bottom
                current_table_row_id = row_table_id
                continue

            # Check if should merge with current line
            merge = False
            if has_table_row_id and row_table_notna and row_table_id == current_table_row_id:
                merge = True
            else:
                dy      = abs(row_y - current_y)
                overlap = min(row_bottom, current_bottom) - max(row_top, current_top)
                if dy <= config.TOL_BASE:
                    merge = True
                elif dy <= config.TOL_EXPANDED and overlap >= config.MIN_VERTICAL_OVERLAP:
                    merge = True

            if merge:
                line_id_arr[pos] = current_line_id
                current_top      = min(current_top, row_top)
                current_bottom   = max(current_bottom, row_bottom)
            else:
                current_line_id      = line_counter
                line_id_arr[pos]     = current_line_id
                line_counter        += 1
                current_y            = row_y
                current_top          = row_top
                current_bottom       = row_bottom
                current_table_row_id = row_table_id

    # Single bulk assignment instead of per-row .at writes
    out["line_id"] = line_id_arr

    _compute_buckets_inplace(out, y_alignment)

    return out


# -----------------------
# Line comparison
# -----------------------

def same_line(
    y_top_a: float,
    y_bottom_a: float,
    y_top_b: float,
    y_bottom_b: float,
    config: LineMergerConfig = LineMergerConfig(),
) -> bool:
    """Return True if two bboxes belong on the same text line."""
    yc_a = (y_top_a + y_bottom_a) / 2.0
    yc_b = (y_top_b + y_bottom_b) / 2.0
    dy = abs(yc_a - yc_b)
    if dy <= config.TOL_BASE:
        return True
    overlap = min(y_bottom_a, y_bottom_b) - max(y_top_a, y_top_b)
    if dy <= config.TOL_EXPANDED and overlap >= config.MIN_VERTICAL_OVERLAP:
        return True
    return False


def same_line_pairwise(
    y_top_a: np.ndarray,
    y_bottom_a: np.ndarray,
    y_top_b: np.ndarray,
    y_bottom_b: np.ndarray,
    config: LineMergerConfig = LineMergerConfig(),
) -> np.ndarray:
    """Vectorized :func:`same_line`.

    Element-wise equivalent of ``same_line`` for equal-length arrays of bbox
    edges, returning a boolean mask. Use this instead of calling ``same_line``
    in a Python loop — it evaluates every pair in a handful of NumPy ops.

    All four inputs must be float arrays of the same length. ``y_top`` is the
    smaller (upper) edge and ``y_bottom`` the larger (lower) edge, matching the
    screen-space convention used across the pipeline (y increases downward).
    """
    y_top_a    = np.asarray(y_top_a,    dtype=float)
    y_bottom_a = np.asarray(y_bottom_a, dtype=float)
    y_top_b    = np.asarray(y_top_b,    dtype=float)
    y_bottom_b = np.asarray(y_bottom_b, dtype=float)

    yc_a = (y_top_a + y_bottom_a) * 0.5
    yc_b = (y_top_b + y_bottom_b) * 0.5
    dy   = np.abs(yc_a - yc_b)

    overlap = np.minimum(y_bottom_a, y_bottom_b) - np.maximum(y_top_a, y_top_b)

    return (dy <= config.TOL_BASE) | (
        (dy <= config.TOL_EXPANDED) & (overlap >= config.MIN_VERTICAL_OVERLAP)
    )


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
        if y_alignment == "top":
            out["top_bucket"] = pd.Series(dtype="Int64")
        elif y_alignment == "bottom":
            out["bottom_bucket"] = pd.Series(dtype="Int64")
        else:
            out["center_bucket"] = pd.Series(dtype="Int64")
        return

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
        top_min    = g["y_top"].min()
        bottom_max = g["y_bottom"].max()
        center_true   = (top_min.astype("float64") + bottom_max.astype("float64")) / 2.0
        center_bucket = center_true.round().astype("Int64")
        out["center_bucket"] = out["line_id"].map(center_bucket)

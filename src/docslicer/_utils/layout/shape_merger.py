# step_05_shape_merger.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Iterable

import numpy as np
import pandas as pd


# ==============================
# Config
# ==============================

_GAP_TOL_PX   = 1.5  # max y (or x) spread to group shapes into the same band
_CHAIN_TOL_PX = 1.5  # max gap between segments in a run to merge into one shape
LINE_HEIGHT_MAX_PX = 3  # max height (or width) to reclassify a rect/curve as a line
_PAGE_BG_COVERAGE = 0.80  # min fraction of page width AND height for a rect to count as page_background


# ==============================
# Types
# ==============================

ShapeType   = Literal["rect", "line", "curve", "unknown"]
ShapeRole   = Literal["page_background", "table_grid", "underline", "separator", "background_band", "other"]
Orientation = Literal["horizontal", "vertical", "unknown"]

# Drawing metadata copied onto merged records from the representative shape.
# Optional columns (PDF-only) are emitted as None when absent from the input.
_META_COLS = (
    "raw_shape_type", "linewidth", "fill", "stroke", "paint_op",
    "non_stroking_color", "stroking_color",
)


# ==============================
# Helpers
# ==============================

def _ensure_shape_columns(
    df: pd.DataFrame,
    *,
    step_name: str = "merge_shapes",
    required_cols: Iterable[str] | None = None,
) -> None:
    """Raise ValueError if any required columns are missing from df."""
    if required_cols is None:
        required_cols = [
            "page_number", "raw_shape_id", "raw_shape_type",
            "x_left", "y_top", "x_right", "y_bottom",
            "width", "height", "area",
            "non_stroking_color",
        ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{step_name}: missing required columns: {missing}")


def _add_raw_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a raw_orientation column:
      - 'square':     |width - height| <= 1 px
      - 'horizontal': width > height
      - 'vertical':   height > width
    """
    df = df.copy()
    df["raw_orientation"] = np.select(
        [
            (df["width"] - df["height"]).abs() <= 1.0,
            df["width"] > df["height"],
        ],
        ["square", "horizontal"],
        default="vertical",
    )
    return df


def _extract_page_arrays(page_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Pull everything the merge needs out of pandas once, as numpy arrays
    sorted by raw_shape_id. All later work indexes these positionally.
    """
    page_df = page_df.sort_values("raw_shape_id")
    arrays: Dict[str, Any] = {
        "ids":             page_df["raw_shape_id"].to_numpy(dtype=np.int64),
        "x_left":          page_df["x_left"].to_numpy(dtype=np.float64),
        "x_right":         page_df["x_right"].to_numpy(dtype=np.float64),
        "y_top":           page_df["y_top"].to_numpy(dtype=np.float64),
        "y_bottom":        page_df["y_bottom"].to_numpy(dtype=np.float64),
        "raw_orientation": page_df["raw_orientation"].to_numpy(),
    }
    for col in _META_COLS:
        arrays[col] = page_df[col].to_numpy(dtype=object) if col in page_df.columns else None
    # Page dimensions (optional — absent on the OCR path); constant per page.
    for col in ("page_width", "page_height"):
        arrays[col] = float(page_df[col].iloc[0]) if col in page_df.columns else None
    return arrays


def _meta(pa: Dict[str, Any], col: str, pos: int) -> Any:
    arr = pa[col]
    return arr[pos] if arr is not None else None


# ==============================
# Candidate Groups
# ==============================

def _band_groups(
    sel: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> List[np.ndarray]:
    """
    Seed-anchored banding: repeatedly take the first unused shape as seed and
    group every unused shape whose lo/hi edges are both within _GAP_TOL_PX of
    the seed's. Returns a list of position arrays (positions into the page
    arrays). sel holds the candidate positions; lo/hi are aligned with sel.
    """
    groups: List[np.ndarray] = []
    used = np.zeros(sel.size, dtype=bool)

    while True:
        remaining = np.flatnonzero(~used)
        if remaining.size == 0:
            break
        i = remaining[0]

        mask = (
            ~used
            & (np.abs(lo - lo[i]) <= _GAP_TOL_PX)
            & (np.abs(hi - hi[i]) <= _GAP_TOL_PX)
        )
        used |= mask
        groups.append(sel[mask])

    return groups


# ==============================
# Shape Record Builder
# ==============================

def _build_shape_record(
    pa: Dict[str, Any],
    run_positions: List[int],
    *,
    page_number: int,
    group_id: int,
    shape_id: int,
    orientation: Literal["horizontal", "vertical"],
    x_left: float,
    x_right: float,
    y_top: float,
    y_bottom: float,
) -> Dict[str, Any]:
    """
    Build a merged shape record from a run of positions into the page arrays.
    Geometry is the union bbox (already tracked by the caller); drawing
    metadata is taken from the shape with the lowest raw_shape_id.
    """
    # Page arrays are sorted by raw_shape_id, so min position = min id.
    rep_pos = min(run_positions)

    width  = x_right - x_left
    height = y_bottom - y_top

    raw_shape_type: ShapeType = pa["raw_shape_type"][rep_pos]
    shape_type: ShapeType = raw_shape_type
    is_thin = (
        (orientation == "horizontal" and height <= LINE_HEIGHT_MAX_PX)
        or (orientation == "vertical" and width <= LINE_HEIGHT_MAX_PX)
    )
    if shape_type in ("rect", "curve"):
        if is_thin:
            shape_type = "line"
    elif shape_type == "line" and not is_thin:
        # The raw extractor's path classifier can't always tell a genuine
        # thin line stroke from a box outline drawn as several disjoint
        # moveto+lineto edge segments in one path object (e.g. a page
        # border drawn as 4 separate strokes rather than one closed
        # rectangle subpath) -- both come back raw_shape_type "line". A
        # "line" whose bbox isn't thin in either dimension is geometrically
        # a box, not a line, so promote it to "rect" here.
        shape_type = "rect"

    linewidth = _meta(pa, "linewidth", rep_pos)
    fill      = _meta(pa, "fill", rep_pos)
    stroke    = _meta(pa, "stroke", rep_pos)
    paint_op  = _meta(pa, "paint_op", rep_pos)

    # A rect covering (almost) the whole page is a slide/page background, not content.
    shape_role: ShapeRole = "other"
    page_w = pa["page_width"]
    page_h = pa["page_height"]
    if (
        page_w is not None and page_h is not None
        and shape_type == "rect"
        and width  >= _PAGE_BG_COVERAGE * page_w
        and height >= _PAGE_BG_COVERAGE * page_h
    ):
        shape_role = "page_background"

    return {
        # Identity
        "page_number":        page_number,
        "shape_id":           shape_id,
        "raw_shape_ids":      [int(pa["ids"][p]) for p in run_positions],
        "candidate_group_id": group_id,
        # Geometry
        "x_left":   x_left,
        "x_right":  x_right,
        "y_top":    y_top,
        "y_bottom": y_bottom,
        "width":    width,
        "height":   height,
        "area":     width * height,
        # Drawing info (from representative shape)
        "raw_shape_type":     raw_shape_type,
        "linewidth":          float(linewidth) if linewidth is not None else None,
        "fill":               bool(fill)       if fill      is not None else None,
        "stroke":             bool(stroke)     if stroke    is not None else None,
        "paint_op":           str(paint_op)    if paint_op  is not None else None,
        "non_stroking_color": _meta(pa, "non_stroking_color", rep_pos),
        "stroking_color":     _meta(pa, "stroking_color", rep_pos),
        # Derived
        "shape_type":        shape_type,
        "shape_orientation": orientation,
        "table_id":          None,
        "shape_role":        shape_role,
        # Populated by later pipeline steps
        "has_intersection":      False,
        "intersection_count":    0,
        "intersecting_line_ids": [],
        "color_hex":             None,
        "color_label":           None,
        "page_width": page_w,
        "page_height": page_h,
    }


def _split_candidate_group(
    pa: Dict[str, Any],
    positions: np.ndarray,
    *,
    page_number: int,
    group_id: int,
    orientation: Literal["horizontal", "vertical"],
    start_id: int,
) -> List[Dict[str, Any]]:
    """
    Split a candidate group into one or more shape records by chaining
    segments within _CHAIN_TOL_PX of each other along the primary axis.
    positions are positions into the page arrays.
    """
    horizontal = orientation == "horizontal"
    sort_vals = pa["x_left"][positions] if horizontal else pa["y_top"][positions]
    pos_sorted = positions[np.argsort(sort_vals, kind="stable")]

    x_left_arr  = pa["x_left"][pos_sorted]
    x_right_arr = pa["x_right"][pos_sorted]
    y_top_arr   = pa["y_top"][pos_sorted]
    y_bot_arr   = pa["y_bottom"][pos_sorted]
    gap_ref_arr = x_left_arr if horizontal else y_top_arr

    records: List[Dict[str, Any]] = []
    current_positions: List[int] = []
    current_x0 = current_x1 = current_top = current_bottom = 0.0
    prev_gap_to: float | None = None
    next_id = start_id

    def _flush() -> None:
        nonlocal next_id
        records.append(_build_shape_record(
            pa, current_positions,
            page_number=page_number, group_id=group_id, shape_id=next_id,
            orientation=orientation,
            x_left=current_x0, x_right=current_x1,
            y_top=current_top, y_bottom=current_bottom,
        ))
        next_id += 1

    for i in range(len(pos_sorted)):
        pos       = int(pos_sorted[i])
        sx0       = x_left_arr[i]
        sx1       = x_right_arr[i]
        sy_top    = y_top_arr[i]
        sy_bottom = y_bot_arr[i]

        if prev_gap_to is not None and gap_ref_arr[i] - prev_gap_to <= _CHAIN_TOL_PX:
            current_positions.append(pos)
            if sx0 < current_x0: current_x0 = sx0
            if sx1 > current_x1: current_x1 = sx1
            if sy_top < current_top: current_top = sy_top
            if sy_bottom > current_bottom: current_bottom = sy_bottom
        else:
            if current_positions:
                _flush()
            current_positions = [pos]
            current_x0     = sx0
            current_x1     = sx1
            current_top    = sy_top
            current_bottom = sy_bottom

        # Track the trailing edge of the current run (not just the current shape)
        prev_gap_to = current_x1 if horizontal else current_bottom

    if current_positions:
        _flush()

    return records


# ==============================
# Merge Orchestrator
# ==============================

def _run_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge raw shapes into logical shape records, processing each page
    sequentially (horizontal then vertical) so IDs increment naturally.
    """
    df = _add_raw_orientation(df)

    all_shapes: List[Dict[str, Any]] = []
    next_group_id = 1
    next_shape_id = 1

    for page_number, page_df in df.groupby("page_number", sort=True):
        page_number = int(page_number)
        pa = _extract_page_arrays(page_df)
        page_shapes: List[Dict[str, Any]] = []

        # Horizontal pass (includes squares)
        h_sel = np.flatnonzero(np.isin(pa["raw_orientation"], ("horizontal", "square")))
        h_groups = _band_groups(h_sel, pa["y_top"][h_sel], pa["y_bottom"][h_sel])
        for positions in h_groups:
            shapes = _split_candidate_group(
                pa, positions,
                page_number=page_number, group_id=next_group_id,
                orientation="horizontal", start_id=next_shape_id,
            )
            next_group_id += 1
            page_shapes.extend(shapes)
            next_shape_id += len(shapes)

        # Vertical pass (includes square singletons from the horizontal pass)
        singleton_ids = [
            sid
            for s in page_shapes
            if len(s["raw_shape_ids"]) == 1
            for sid in s["raw_shape_ids"]
        ]
        v_mask = (pa["raw_orientation"] == "vertical") | (
            (pa["raw_orientation"] == "square") & np.isin(pa["ids"], singleton_ids)
        )
        v_sel = np.flatnonzero(v_mask)
        v_groups = _band_groups(v_sel, pa["x_left"][v_sel], pa["x_right"][v_sel])
        for positions in v_groups:
            shapes = _split_candidate_group(
                pa, positions,
                page_number=page_number, group_id=next_group_id,
                orientation="vertical", start_id=next_shape_id,
            )
            next_group_id += 1
            page_shapes.extend(shapes)
            next_shape_id += len(shapes)

        all_shapes.extend(page_shapes)

    return pd.DataFrame(all_shapes)


# ==============================
# Public API
# ==============================

def merge_shapes(
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge raw shapes from extract_shapes into logical shape records.

    Each logical shape may span multiple raw shapes (e.g. a dashed line
    rendered as many small rects is merged into one). Thin rects and curves
    are reclassified as lines based on their dimensions.

    Input columns (required):
        page_number, raw_shape_id, raw_shape_type,
        x_left, y_top, x_right, y_bottom, width, height, area,
        non_stroking_color

    Input columns (optional — PDF-only, pass through as None if absent):
        stroking_color, linewidth, fill, stroke, paint_op

    Input columns (optional — enable page_background detection):
        page_width, page_height

    Output columns (one row per logical shape):
        page_number, shape_id, raw_shape_ids, candidate_group_id,
        x_left, x_right, y_top, y_bottom, width, height, area,
        raw_shape_type, shape_type, shape_orientation,
        linewidth, fill, stroke, paint_op,
        non_stroking_color, stroking_color,
        table_id, shape_role,
        has_intersection, intersection_count, intersecting_line_ids,
        color_hex, color_label
    """
    if df_shapes.empty:
        return df_shapes.copy()

    _ensure_shape_columns(df_shapes)

    return _run_merge(df_shapes)

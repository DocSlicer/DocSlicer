# step_04_shape_merger.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, List, Set, Iterable

import numpy as np
import pandas as pd


# ==============================
# Config
# ==============================

_GAP_TOL_PX   = 1.5  # max y (or x) spread to group shapes into the same band
_CHAIN_TOL_PX = 1.5  # max gap between segments in a run to merge into one shape
LINE_HEIGHT_MAX_PX = 3  # max height (or width) to reclassify a rect/curve as a line


# ==============================
# Types
# ==============================

ShapeType     = Literal["rect", "line", "curve", "unknown"]
ShapeSemantic = Literal["table_grid", "underline", "separator", "background_band", "other"]
Orientation   = Literal["horizontal", "vertical", "unknown"]


@dataclass
class CandidateGroup:
    group_id: int
    page_number: int
    raw_shape_ids: List[int]
    group_orientation: Literal["horizontal", "vertical"]

    # Bounding box (union of all shapes in the group)
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

    @property
    def width(self) -> float:
        return self.x_right - self.x_left

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top

    @property
    def is_singleton(self) -> bool:
        return len(self.raw_shape_ids) == 1

    @classmethod
    def from_shapes_df(
        cls,
        *,
        group_id: int,
        page_number: int,
        shape_orientation: Literal["horizontal", "vertical"],
        shapes_df: pd.DataFrame,
    ) -> "CandidateGroup":
        """Build a CandidateGroup from a subset of the shapes DataFrame."""
        return cls(
            group_id=group_id,
            page_number=page_number,
            raw_shape_ids=shapes_df["raw_shape_id"].astype(int).tolist(),
            group_orientation=shape_orientation,
            x_left=float(shapes_df["x_left"].min()),
            x_right=float(shapes_df["x_right"].max()),
            y_top=float(shapes_df["y_top"].min()),
            y_bottom=float(shapes_df["y_bottom"].max()),
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


# ==============================
# Candidate Groups
# ==============================

def _make_horizontal_candidate_groups(
    df: pd.DataFrame,
    *,
    start_group_id: int = 1,
) -> Tuple[List[CandidateGroup], int]:
    """
    Group horizontal (and square) shapes that share the same y-band.
    Returns (groups, next_group_id).
    """
    groups: List[CandidateGroup] = []
    group_id = start_group_id

    horiz = df[df["raw_orientation"].isin(["horizontal", "square"])].copy()

    for page, page_df in horiz.groupby("page_number"):
        remaining = page_df.sort_values("raw_shape_id").copy()

        while len(remaining):
            anchor = remaining.iloc[0]
            mask = (
                (remaining["y_top"]    - anchor["y_top"]).abs()    <= _GAP_TOL_PX
            ) & (
                (remaining["y_bottom"] - anchor["y_bottom"]).abs() <= _GAP_TOL_PX
            )
            band_df = remaining[mask]
            group = CandidateGroup.from_shapes_df(
                group_id=group_id,
                page_number=page,
                shape_orientation="horizontal",
                shapes_df=band_df,
            )
            groups.append(group)
            group_id += 1
            remaining = remaining[~remaining["raw_shape_id"].isin(group.raw_shape_ids)]

    return groups, group_id


def _get_vertical_candidate_df(
    df: pd.DataFrame,
    horizontal_shapes: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build the DataFrame for vertical grouping:
      - all shapes with raw_orientation == 'vertical'
      - plus square shapes that ended up as singletons in the horizontal pass
    """
    square_ids: Set[int] = set(
        df.loc[df["raw_orientation"] == "square", "raw_shape_id"].astype(int)
    )
    singleton_ids: Set[int] = {
        sid
        for s in horizontal_shapes
        if len(s["raw_shape_ids"]) == 1
        for sid in s["raw_shape_ids"]
    }
    square_singletons = square_ids & singleton_ids

    return df[
        (df["raw_orientation"] == "vertical")
        | (df["raw_shape_id"].isin(square_singletons))
    ].copy()


def _make_vertical_candidate_groups(
    df: pd.DataFrame,
    horizontal_shapes: List[Dict[str, Any]],
    *,
    start_group_id: int = 1,
) -> Tuple[List[CandidateGroup], int]:
    """
    Group vertical shapes (and square singletons from the horizontal pass)
    that share the same x-band.
    Returns (groups, next_group_id).
    """
    vertical_df = _get_vertical_candidate_df(df, horizontal_shapes)

    groups: List[CandidateGroup] = []
    group_id = start_group_id

    for page, page_df in vertical_df.groupby("page_number"):
        remaining = page_df.sort_values("raw_shape_id").copy()

        while len(remaining):
            anchor = remaining.iloc[0]
            mask = (
                (remaining["x_left"]  - anchor["x_left"]).abs()  <= _GAP_TOL_PX
            ) & (
                (remaining["x_right"] - anchor["x_right"]).abs() <= _GAP_TOL_PX
            )
            band_df = remaining[mask]
            group = CandidateGroup.from_shapes_df(
                group_id=group_id,
                page_number=page,
                shape_orientation="vertical",
                shapes_df=band_df,
            )
            groups.append(group)
            group_id += 1
            remaining = remaining[~remaining["raw_shape_id"].isin(group.raw_shape_ids)]

    return groups, group_id


# ==============================
# Shape Record Builder
# ==============================

def _build_shape_record(
    df: pd.DataFrame,
    group: CandidateGroup,
    raw_shape_ids: List[int],
    shape_id: int,
) -> Dict[str, Any]:
    """
    Build a merged shape record from a run of raw shape IDs.
    Geometry is the union bbox; drawing metadata is taken from the first shape.
    """
    sub = df[df["raw_shape_id"].isin(raw_shape_ids)].sort_values("raw_shape_id")
    rep = sub.iloc[0]

    x_left   = float(sub["x_left"].min())
    x_right  = float(sub["x_right"].max())
    y_top    = float(sub["y_top"].min())
    y_bottom = float(sub["y_bottom"].max())
    width    = x_right - x_left
    height   = y_bottom - y_top

    shape_orientation: Orientation = (
        group.group_orientation
        if group.group_orientation in ("horizontal", "vertical")
        else "unknown"
    )

    raw_shape_type: ShapeType = rep["raw_shape_type"]
    shape_type: ShapeType = raw_shape_type
    if shape_type in ("rect", "curve"):
        if shape_orientation == "horizontal" and height <= LINE_HEIGHT_MAX_PX:
            shape_type = "line"
        elif shape_orientation == "vertical" and width <= LINE_HEIGHT_MAX_PX:
            shape_type = "line"

    linewidth = rep.get("linewidth")
    fill      = rep.get("fill")
    stroke    = rep.get("stroke")
    paint_op  = rep.get("paint_op")

    return {
        # Identity
        "page_number":        int(rep["page_number"]),
        "shape_id":           shape_id,
        "raw_shape_ids":      [int(sid) for sid in raw_shape_ids],
        "candidate_group_id": group.group_id,
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
        "non_stroking_color": rep.get("non_stroking_color"),
        "stroking_color":     rep.get("stroking_color"),
        # Derived
        "shape_type":        shape_type,
        "shape_orientation": shape_orientation,
        "table_id":          None,
        "shape_semantic":    "other",
        # Populated by later pipeline steps
        "has_intersection":      False,
        "intersection_count":    0,
        "intersecting_line_ids": [],
        "color_hex":             None,
        "color_label":           None,
    }


def _split_candidate_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int,
    sort_col: str,
    gap_ref_col: str,
    gap_to_col: str,
) -> List[Dict[str, Any]]:
    """
    Split a CandidateGroup into one or more shape records by chaining segments
    within _CHAIN_TOL_PX of each other along the primary axis.
    """
    sub = df[df["raw_shape_id"].isin(group.raw_shape_ids)].sort_values(sort_col)

    records: List[Dict[str, Any]] = []
    current_ids: List[int] = []
    current_x0 = current_x1 = current_top = current_bottom = None
    prev_gap_to: float | None = None
    next_id = start_id

    for _, row in sub.iterrows():
        sid       = int(row["raw_shape_id"])
        sx0       = float(row["x_left"])
        sx1       = float(row["x_right"])
        sy_top    = float(row["y_top"])
        sy_bottom = float(row["y_bottom"])
        gap_ref   = float(row[gap_ref_col])

        if prev_gap_to is None:
            current_ids    = [sid]
            current_x0     = sx0
            current_x1     = sx1
            current_top    = sy_top
            current_bottom = sy_bottom
        elif gap_ref - prev_gap_to <= _CHAIN_TOL_PX:
            current_ids.append(sid)
            current_x0     = min(current_x0, sx0)
            current_x1     = max(current_x1, sx1)
            current_top    = min(current_top, sy_top)
            current_bottom = max(current_bottom, sy_bottom)
        else:
            records.append(_build_shape_record(df, group, current_ids, next_id))
            next_id        += 1
            current_ids    = [sid]
            current_x0     = sx0
            current_x1     = sx1
            current_top    = sy_top
            current_bottom = sy_bottom

        # Track the trailing edge of the current run (not just the current shape)
        prev_gap_to = current_x1 if gap_to_col == "x_right" else current_bottom

    if current_ids:
        records.append(_build_shape_record(df, group, current_ids, next_id))

    return records


def _shapes_from_horizontal_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int = 1,
) -> List[Dict[str, Any]]:
    return _split_candidate_group(
        df, group, start_id=start_id,
        sort_col="x_left", gap_ref_col="x_left", gap_to_col="x_right",
    )


def _shapes_from_vertical_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int = 1,
) -> List[Dict[str, Any]]:
    return _split_candidate_group(
        df, group, start_id=start_id,
        sort_col="y_top", gap_ref_col="y_top", gap_to_col="y_bottom",
    )


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

    for page_number in sorted(df["page_number"].unique()):
        page_df = df[df["page_number"] == page_number].copy()

        # Horizontal pass
        h_groups, next_group_id = _make_horizontal_candidate_groups(
            page_df, start_group_id=next_group_id
        )
        page_shapes: List[Dict[str, Any]] = []
        for g in h_groups:
            shapes = _shapes_from_horizontal_group(page_df, g, start_id=next_shape_id)
            page_shapes.extend(shapes)
            next_shape_id += len(shapes)

        # Vertical pass (includes square singletons from horizontal pass)
        v_groups, next_group_id = _make_vertical_candidate_groups(
            page_df, page_shapes, start_group_id=next_group_id
        )
        for g in v_groups:
            shapes = _shapes_from_vertical_group(page_df, g, start_id=next_shape_id)
            page_shapes.extend(shapes)
            next_shape_id += len(shapes)

        all_shapes.extend(page_shapes)

    return pd.DataFrame(all_shapes)


# ==============================
# Public API
# ==============================

def merge_shapes(
    df_shapes: pd.DataFrame,
    *,
    merge_lines: bool = True,
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

    Output columns (one row per logical shape):
        page_number, shape_id, raw_shape_ids, candidate_group_id,
        x_left, x_right, y_top, y_bottom, width, height, area,
        raw_shape_type, shape_type, shape_orientation,
        linewidth, fill, stroke, paint_op,
        non_stroking_color, stroking_color,
        table_id, shape_semantic,
        has_intersection, intersection_count, intersecting_line_ids,
        color_hex, color_label
    """
    if df_shapes.empty:
        return df_shapes.copy()

    df = df_shapes.copy()
    _ensure_shape_columns(df)
    df["raw_shape_ids"] = df["raw_shape_id"].astype(int).map(lambda v: [v])

    if merge_lines:
        df = _run_merge(df)

    return df

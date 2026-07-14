# step_05_shape_processor.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Iterable, Tuple

import numpy as np
import pandas as pd


# ==============================
# Config
# ==============================

_GAP_TOL_PX   = 1.5  # max y (or x) spread to group shapes into the same band
_CHAIN_TOL_PX = 1.7  # max gap between segments in a run to merge into one shape
LINE_HEIGHT_MAX_PX = 4.2  # max height (or width) to reclassify a rect/curve as a line (a double line is usually 4pt)
_CURVE_SQUARE_TOL_PX = 1.0  # max |width - height| for a thin curve to still count as "square" (never a line)
_CURVE_TO_LINE_MIN_LONG_SIDE_PX = 8.0  # min long-side length for a thin curve to reclassify as a line
_PAGE_BG_COVERAGE = 0.80  # min fraction of page width AND height for a rect to count as page_background
_BAND_MAJOR_COVERAGE = 0.80  # min fraction of page width (or height) along the band's long axis
_BAND_MINOR_COVERAGE = 0.30  # min fraction of page height (or width) along the band's short axis
_GRID_SNAP_TOL_PX = 3.0  # max gap for a horizontal and vertical line to count as touching (grid detection)
_GRID_MIN_LINE_LEN_PX = 20.0  # min primary-axis length for a line to participate in a grid (rejects tiny fragments)

# Columns of the grid-cells DataFrame emitted alongside the shape records.
_GRID_CELL_COLS = (
    "grid_cell_id", "table_grid_id", "page_number",
    "row_start", "col_start", "rowspan", "colspan",
    "x_left", "y_top", "x_right", "y_bottom",
)


# ==============================
# Types
# ==============================

ShapeType   = Literal["rect", "line", "curve", "unknown"]
ShapeRole   = Literal[
    "page_background", "table_grid", "box",
    "table_rule", "underline", "strikethrough", "separator",
    "background_band", "other",
]
Orientation = Literal["horizontal", "vertical", "unknown"]

# Drawing metadata copied onto merged records from the representative shape.
# Optional columns (PDF-only) are emitted as None when absent from the input.
_META_COLS = (
    "raw_shape_type", "linewidth", "fill", "stroke", "paint_op",
    "non_stroking_color", "stroking_color",
)

# Struct-tree / content-stream provenance (attached per raw shape by
# step_03_shape_extractor when a struct_index is supplied). Copied verbatim
# from the representative shape — the raw shapes of a merged run come from the
# same marked-content item in practice. Absent columns pass through as None.
_STRUCT_COLS = (
    "mcid", "dfs_position",
    "struct_tag", "struct_raw_tag", "struct_tag_id",
    "struct_ancestors", "struct_raw_ancestors", "struct_ancestor_ids",
    "struct_scope", "struct_headers", "struct_col_span", "struct_row_span",
)


# ==============================
# Helpers
# ==============================

def _ensure_shape_columns(
    df: pd.DataFrame,
    *,
    step_name: str = "process_shapes",
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
    for col in _META_COLS + _STRUCT_COLS:
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
    if shape_type == "curve":
        is_square = abs(width - height) <= _CURVE_SQUARE_TOL_PX
        long_side = max(width, height)
        if is_thin and not is_square and long_side >= _CURVE_TO_LINE_MIN_LONG_SIDE_PX:
            shape_type = "line"
    elif shape_type == "rect":
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

    # shape_role is left as the default "other" here; all role classification
    # happens in the role-assignment phase, once every record's final geometry
    # is known (see _assign_shape_roles).
    page_w = pa["page_width"]
    page_h = pa["page_height"]

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
        # Struct-tree provenance (from representative shape)
        **{col: _meta(pa, col, rep_pos) for col in _STRUCT_COLS},
        # Derived
        "shape_type":        shape_type,
        "shape_orientation": orientation,
        "table_id":          None,
        "shape_role":        "other",
        "table_grid_id":     None,
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
# Role Assignment
# ==============================

def _assign_page_background_roles(df: pd.DataFrame) -> None:
    """
    Tag rects covering (almost) the whole page as 'page_background' (in place).

    A rect spanning >= _PAGE_BG_COVERAGE of both page dimensions is a slide/page
    backdrop, not content. No-op when page dimensions are absent (OCR path).
    """
    if "page_width" not in df.columns or "page_height" not in df.columns:
        return
    is_bg = (
        (df["shape_type"] == "rect")
        & df["page_width"].notna()
        & df["page_height"].notna()
        & (df["width"]  >= _PAGE_BG_COVERAGE * df["page_width"])
        & (df["height"] >= _PAGE_BG_COVERAGE * df["page_height"])
    )
    df.loc[is_bg, "shape_role"] = "page_background"


def _assign_background_band_roles(df: pd.DataFrame) -> None:
    """
    Tag rects covering a full-width or full-height band of the page as
    'background_band' (in place).

    A band spans >= _BAND_MAJOR_COVERAGE of the page along one dimension and
    >= _BAND_MINOR_COVERAGE along the other -- e.g. a horizontal stripe
    running edge-to-edge across the width but only partway down the height,
    or the vertical mirror. Runs after page_background so it only claims
    records still tagged the default 'other' (page_background already covers
    the case where both dimensions are >= _PAGE_BG_COVERAGE). No-op when page
    dimensions are absent (OCR path).
    """
    if "page_width" not in df.columns or "page_height" not in df.columns:
        return
    has_dims = df["page_width"].notna() & df["page_height"].notna()
    is_horizontal_band = (
        (df["width"]  >= _BAND_MAJOR_COVERAGE * df["page_width"])
        & (df["height"] >= _BAND_MINOR_COVERAGE * df["page_height"])
    )
    is_vertical_band = (
        (df["height"] >= _BAND_MAJOR_COVERAGE * df["page_height"])
        & (df["width"]  >= _BAND_MINOR_COVERAGE * df["page_width"])
    )
    is_band = (
        (df["shape_type"] == "rect")
        & (df["shape_role"] == "other")
        & has_dims
        & (is_horizontal_band | is_vertical_band)
    )
    df.loc[is_band, "shape_role"] = "background_band"


def _cluster_touching_lines(
    sel: np.ndarray,
    orient: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    y_top: np.ndarray,
    y_bottom: np.ndarray,
    tol: float,
) -> List[List[int]]:
    """
    Union-find `sel` (positions into the full line arrays) into connected
    clusters, where a horizontal and vertical line join the same cluster when
    they touch, i.e. the vertical crosses the horizontal's x-span and the
    horizontal crosses the vertical's y-span, each within `tol`. Returns
    clusters as lists of local indices into `sel`.
    """
    parent = list(range(sel.size))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    h_loc = [t for t in range(sel.size) if orient[sel[t]] == "horizontal"]
    v_loc = [t for t in range(sel.size) if orient[sel[t]] == "vertical"]

    for a in h_loc:
        ha = sel[a]
        for b in v_loc:
            vb = sel[b]
            if (
                x_left[ha] - tol <= cx[vb] <= x_right[ha] + tol
                and y_top[vb] - tol <= cy[ha] <= y_bottom[vb] + tol
            ):
                union(a, b)

    clusters: Dict[int, List[int]] = {}
    for t in range(sel.size):
        clusters.setdefault(find(t), []).append(t)
    return list(clusters.values())


def _assign_table_grid_roles(df: pd.DataFrame, start_id: int) -> int:
    """
    Detect table grids among the line shapes and tag their members with
    shape_role 'table_grid' plus a shared table_grid_id (in place). Returns the
    next free table_grid_id. Processes each page independently so ids are
    document-unique.

    Horizontal and vertical lines are snapped into connected clusters: a
    vertical and a horizontal join the same cluster when they touch, i.e. the
    vertical crosses the horizontal's x-span and the horizontal crosses the
    vertical's y-span, each within _GRID_SNAP_TOL_PX. A cluster is a real grid
    -- as opposed to a lone box/square -- only when it has at least two
    horizontal lines (a top and a bottom, plus any in between) and at least one
    interior vertical, i.e. a vertical that is not the left or right side of the
    cluster. Each qualifying cluster gets its own table_grid_id.
    """
    next_id = start_id

    # Positional numpy views; results are written back once at the end.
    role   = df["shape_role"].to_numpy(dtype=object, copy=True)
    grid   = df["table_grid_id"].to_numpy(dtype=object, copy=True)
    page   = df["page_number"].to_numpy()
    orient = df["shape_orientation"].to_numpy()
    stype  = df["shape_type"].to_numpy()
    x_left   = df["x_left"].to_numpy(dtype=np.float64)
    x_right  = df["x_right"].to_numpy(dtype=np.float64)
    y_top    = df["y_top"].to_numpy(dtype=np.float64)
    y_bottom = df["y_bottom"].to_numpy(dtype=np.float64)
    cx = 0.5 * (x_left + x_right)   # line center x (meaningful for verticals)
    cy = 0.5 * (y_top + y_bottom)   # line center y (meaningful for horizontals)

    # Primary-axis length: span along the line's own orientation. Lines shorter
    # than _GRID_MIN_LINE_LEN_PX are tiny fragments (e.g. curve stubs) that can
    # spuriously chain into a grid, so they are excluded from grid detection.
    is_h = orient == "horizontal"
    primary_len = np.where(is_h, x_right - x_left, y_bottom - y_top)
    long_enough = primary_len >= _GRID_MIN_LINE_LEN_PX

    tol = _GRID_SNAP_TOL_PX
    for pg in pd.unique(page):
        sel = np.flatnonzero(
            (page == pg)
            & (stype == "line")
            & np.isin(orient, ("horizontal", "vertical"))
            & (role == "other")
            & long_enough
        )
        if sel.size < 3:  # need >= 2 horizontals + >= 1 vertical
            continue

        clusters = _cluster_touching_lines(sel, orient, cx, cy, x_left, x_right, y_top, y_bottom, tol)

        for members in clusters:
            h_mem = [t for t in members if orient[sel[t]] == "horizontal"]
            v_mem = [t for t in members if orient[sel[t]] == "vertical"]
            if len(h_mem) < 2 or not v_mem:
                continue

            # Grid horizontal extent, taken from its horizontal lines.
            x_min = min(x_left[sel[t]]  for t in h_mem)
            x_max = max(x_right[sel[t]] for t in h_mem)
            has_interior_v = any(
                x_min + tol < cx[sel[t]] < x_max - tol
                for t in v_mem
            )
            if not has_interior_v:
                continue  # just a box/square, no interior column separator

            for t in members:
                role[sel[t]] = "table_grid"
                grid[sel[t]] = next_id
            next_id += 1

    df["shape_role"] = role
    df["table_grid_id"] = grid
    return next_id


def _assign_box_roles(df: pd.DataFrame) -> None:
    """
    Detect plain 4-line boxes among the line shapes not already claimed by
    table_grid, and tag their members with shape_role 'box' (in place).

    A box is a cluster of exactly 2 horizontal lines (top/bottom) and 2
    vertical lines (left/right) with no ruling lines in between -- unlike a
    table_grid, every horizontal must touch every vertical, i.e. all four
    corners are closed. This is what distinguishes a box from a table_grid
    (which requires at least one interior separator).

    Unlike table_grid detection, no minimum line length is enforced: box
    sides (e.g. a single small cell) are often shorter than
    _GRID_MIN_LINE_LEN_PX. The all-four-corners-closed check is a strong
    enough constraint on its own to reject spurious tiny fragments.
    """
    role   = df["shape_role"].to_numpy(dtype=object, copy=True)
    page   = df["page_number"].to_numpy()
    orient = df["shape_orientation"].to_numpy()
    stype  = df["shape_type"].to_numpy()
    x_left   = df["x_left"].to_numpy(dtype=np.float64)
    x_right  = df["x_right"].to_numpy(dtype=np.float64)
    y_top    = df["y_top"].to_numpy(dtype=np.float64)
    y_bottom = df["y_bottom"].to_numpy(dtype=np.float64)
    cx = 0.5 * (x_left + x_right)
    cy = 0.5 * (y_top + y_bottom)

    tol = _GRID_SNAP_TOL_PX
    for pg in pd.unique(page):
        sel = np.flatnonzero(
            (page == pg)
            & (stype == "line")
            & np.isin(orient, ("horizontal", "vertical"))
            & (role == "other")
        )
        if sel.size < 4:  # need exactly 2 horizontals + 2 verticals
            continue

        clusters = _cluster_touching_lines(sel, orient, cx, cy, x_left, x_right, y_top, y_bottom, tol)

        for members in clusters:
            if len(members) != 4:
                continue
            h_mem = [t for t in members if orient[sel[t]] == "horizontal"]
            v_mem = [t for t in members if orient[sel[t]] == "vertical"]
            if len(h_mem) != 2 or len(v_mem) != 2:
                continue

            # Every horizontal must touch every vertical -- all four corners closed.
            closed = all(
                x_left[sel[h]] - tol <= cx[sel[v]] <= x_right[sel[h]] + tol
                and y_top[sel[v]] - tol <= cy[sel[h]] <= y_bottom[sel[v]] + tol
                for h in h_mem
                for v in v_mem
            )
            if not closed:
                continue

            for t in members:
                role[sel[t]] = "box"

    df["shape_role"] = role


def _assign_shape_roles(df: pd.DataFrame) -> None:
    """
    Classify merged shape records into shape_roles (in place).

    Runs after merge so every classifier sees final geometry. Passes run in
    precedence order and each only claims records still tagged the default
    'other', so roles never fight over the same shape.
    """
    _assign_page_background_roles(df)
    _assign_background_band_roles(df)
    _assign_table_grid_roles(df, start_id=1)
    _assign_box_roles(df)


# ==============================
# Grid Cell Extraction
# ==============================

def _cluster_1d(values: np.ndarray, tol: float) -> np.ndarray:
    """
    Collapse 1-D coordinates into sorted cluster centers: values are sorted and
    consecutive ones within `tol` are averaged into a single representative.
    Used to turn the many ruling-line positions of a grid into its distinct
    row / column separator lines.
    """
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.sort(values.astype(np.float64))
    centers: List[float] = []
    members: List[float] = [float(order[0])]
    for v in order[1:]:
        if v - members[-1] <= tol:
            members.append(float(v))
        else:
            centers.append(float(np.mean(members)))
            members = [float(v)]
    centers.append(float(np.mean(members)))
    return np.array(centers, dtype=np.float64)


def _build_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct table cells (with row/col spans) from the ruling lines of each
    detected table grid. One row per logical cell.

    For each table_grid_id, the horizontal lines define row separators and the
    vertical lines define column separators (the grid's outer bbox is folded in
    so open-sided tables still get their border boundaries). Consecutive
    separators bound a matrix of atomic cells; two adjacent atomic cells belong
    to the same logical cell when the ruling line on their shared edge is
    *absent* -- tested by checking whether any line at that separator actually
    covers the shared edge's midpoint. Connected atomic cells are then collapsed
    into one record whose span is the extent of its members.

    Output columns: grid_cell_id (unique PK), table_grid_id, page_number,
    row_start, col_start, rowspan, colspan, x_left, y_top, x_right, y_bottom.
    Row/col indices are 0-based (row 0 is the top row).
    """
    cols = list(_GRID_CELL_COLS)
    if "shape_role" not in df.columns or "table_grid_id" not in df.columns:
        return pd.DataFrame(columns=cols)

    grid_df = df[df["shape_role"] == "table_grid"]
    if grid_df.empty:
        return pd.DataFrame(columns=cols)

    tol = _GRID_SNAP_TOL_PX
    records: List[Dict[str, Any]] = []

    for grid_id, g in grid_df.groupby("table_grid_id", sort=True):
        page_number = int(g["page_number"].iloc[0])
        orient = g["shape_orientation"].to_numpy()
        xl = g["x_left"].to_numpy(dtype=np.float64)
        xr = g["x_right"].to_numpy(dtype=np.float64)
        yt = g["y_top"].to_numpy(dtype=np.float64)
        yb = g["y_bottom"].to_numpy(dtype=np.float64)

        h = orient == "horizontal"
        v = orient == "vertical"
        # Horizontal lines -> row separators (center y); their x-span gates the
        # presence of a horizontal edge. Vertical lines -> column separators.
        h_cy = 0.5 * (yt[h] + yb[h])
        h_xl, h_xr = xl[h], xr[h]
        v_cx = 0.5 * (xl[v] + xr[v])
        v_yt, v_yb = yt[v], yb[v]

        # Fold in the grid's outer bbox so missing border lines still bound it.
        gx0, gx1 = float(xl.min()), float(xr.max())
        gy0, gy1 = float(yt.min()), float(yb.max())
        xs = _cluster_1d(np.concatenate([v_cx, (gx0, gx1)]), tol)
        ys = _cluster_1d(np.concatenate([h_cy, (gy0, gy1)]), tol)

        n_rows = ys.size - 1
        n_cols = xs.size - 1
        if n_rows < 1 or n_cols < 1:
            continue

        # Union-find over atomic cells, indexed r * n_cols + c.
        parent = list(range(n_rows * n_cols))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for r in range(n_rows):
            ymid = 0.5 * (ys[r] + ys[r + 1])
            for c in range(n_cols):
                xmid = 0.5 * (xs[c] + xs[c + 1])
                # Merge with right neighbour when the vertical separator at
                # xs[c+1] does not cover this row's midpoint.
                if c + 1 < n_cols:
                    present = bool(np.any(
                        (np.abs(v_cx - xs[c + 1]) <= tol)
                        & (v_yt - tol <= ymid) & (v_yb + tol >= ymid)
                    ))
                    if not present:
                        union(r * n_cols + c, r * n_cols + c + 1)
                # Merge with bottom neighbour when the horizontal separator at
                # ys[r+1] does not cover this column's midpoint.
                if r + 1 < n_rows:
                    present = bool(np.any(
                        (np.abs(h_cy - ys[r + 1]) <= tol)
                        & (h_xl - tol <= xmid) & (h_xr + tol >= xmid)
                    ))
                    if not present:
                        union(r * n_cols + c, (r + 1) * n_cols + c)

        # Collapse atomic cells into their connected components (bounding span).
        comp: Dict[int, Dict[str, int]] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                root = find(r * n_cols + c)
                b = comp.get(root)
                if b is None:
                    comp[root] = {"r0": r, "r1": r, "c0": c, "c1": c}
                else:
                    b["r0"] = min(b["r0"], r); b["r1"] = max(b["r1"], r)
                    b["c0"] = min(b["c0"], c); b["c1"] = max(b["c1"], c)

        for b in comp.values():
            records.append({
                "grid_cell_id":  None,  # assigned after sorting, below
                "table_grid_id": int(grid_id),
                "page_number":   page_number,
                "row_start":     b["r0"],
                "col_start":     b["c0"],
                "rowspan":       b["r1"] - b["r0"] + 1,
                "colspan":       b["c1"] - b["c0"] + 1,
                "x_left":        float(xs[b["c0"]]),
                "y_top":         float(ys[b["r0"]]),
                "x_right":       float(xs[b["c1"] + 1]),
                "y_bottom":      float(ys[b["r1"] + 1]),
            })

    out = pd.DataFrame(records, columns=cols)
    if not out.empty:
        out = out.sort_values(
            ["table_grid_id", "row_start", "col_start"]
        ).reset_index(drop=True)
        out["grid_cell_id"] = np.arange(1, len(out) + 1)
    return out


# ==============================
# Public API
# ==============================

def process_shapes(
    df_shapes: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn raw shapes from extract_shapes into classified logical shape records,
    and reconstruct the cell structure of any detected table grids.

    Phases:
      1. Merge  -- collapse raw shapes into logical records. Each logical shape
         may span multiple raw shapes (e.g. a dashed line rendered as many
         small rects is merged into one); thin rects and curves are
         reclassified as lines based on their dimensions.
      2. Role assignment -- classify each record's shape_role (page_background,
         table_grid, ...) from the merged geometry.
      3. Grid cells -- reconstruct table cells (with spans) from the ruling
         lines of each table grid.

    Returns a (df_shapes, df_grid_cells) tuple.

    Input columns (required):
        page_number, raw_shape_id, raw_shape_type,
        x_left, y_top, x_right, y_bottom, width, height, area,
        non_stroking_color

    Input columns (optional — PDF-only, pass through as None if absent):
        stroking_color, linewidth, fill, stroke, paint_op

    Input columns (optional — enable page_background detection):
        page_width, page_height

    Input columns (optional — struct-tree provenance, pass through as None if
    absent): mcid, dfs_position, struct_tag, struct_raw_tag, struct_tag_id,
        struct_ancestors, struct_raw_ancestors, struct_ancestor_ids,
        struct_scope, struct_headers, struct_col_span, struct_row_span

    df_shapes columns (one row per logical shape):
        page_number, shape_id, raw_shape_ids, candidate_group_id,
        x_left, x_right, y_top, y_bottom, width, height, area,
        raw_shape_type, shape_type, shape_orientation,
        linewidth, fill, stroke, paint_op,
        non_stroking_color, stroking_color,
        mcid, dfs_position, struct_* (provenance, from representative shape),
        table_id, shape_role, table_grid_id,
        has_intersection, intersection_count, intersecting_line_ids,
        color_hex, color_label

    df_grid_cells columns (one row per reconstructed table cell):
        grid_cell_id, table_grid_id, page_number, row_start, col_start,
        rowspan, colspan, x_left, y_top, x_right, y_bottom
    """
    if df_shapes.empty:
        return df_shapes.copy(), pd.DataFrame(columns=list(_GRID_CELL_COLS))

    _ensure_shape_columns(df_shapes)

    df = _run_merge(df_shapes)
    _assign_shape_roles(df)
    df_grid_cells = _build_grid_cells(df)
    return df, df_grid_cells

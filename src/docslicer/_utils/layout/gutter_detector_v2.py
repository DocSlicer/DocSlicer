"""
gutter_detector_v2.py

Stage 1 of the v2 gutter detector: maximal whitespace rectangles.

Ground-up replacement for the sliding-window / candidate-tracking machinery in
gutter_detector.py.  Instead of stitching per-row gaps into vertical chains and
policing them with interacting toggles, this module treats every word bbox and
shape bbox as an obstacle and enumerates the maximal empty rectangles on each
page directly, via Breuel's branch-and-bound search ("Two Geometric Algorithms
for Layout Analysis", 2002).

A column gutter is simply a tall empty rectangle, found natively in 2D:
  - horizontal rules stop a rectangle vertically (no clip/kill toggles),
  - vertical rules split a rectangle horizontally (table borders),
  - page margins come out as ordinary rectangles and can be filtered later.

This stage is intentionally dumb: it outputs every maximal whitespace
rectangle that clears the minimum size and could plausibly be a column
gutter — dropped at output are rectangles that are wider than tall
(horizontal bands), that touch the left/right page border (margins), or that
lack a touching obstacle (word, shape, or image) on both vertical edges
(ragged-edge / indent fragments bounded by the page border or another
whitespace rect).  Scoring / selection of true column gutters is a later
stage.

Used by the OCR pipeline only — the sole inputs are bboxes (no struct tree).

Public API:
    df_gutters = detect_gutters(df_words, df_shapes)

df_gutters columns:
    page_number, gutter_id,
    gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
    gutter_width, gutter_height, gutter_area

Coordinate convention (matches the rest of the pipeline):
    y increases downward; y_top < y_bottom.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
import pandas as pd

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

_MIN_RECT_WIDTH: float = 9.2     # pt - minimum whitespace rectangle width
_MIN_RECT_HEIGHT: float = 30.0   # pt - minimum whitespace rectangle height
_MAX_RECTS_PER_PAGE: int = 30    # stop enumerating after this many rectangles per page
_MAX_NODE_EXPANSIONS: int = 200_000  # hard safety cap on branch-and-bound nodes per page

_EPS: float = 1e-6  # strict-overlap epsilon: touching edges do not count as overlap
_PAGE_EDGE_EPS: float = 1.0  # pt - a rect within this distance of the left/right page edge counts as touching it
_SIDE_TEXT_TOL: float = 1.0  # pt - a word edge within this distance of a rect's vertical edge counts as touching it


# =======================================================================================================================
# Breuel branch-and-bound: maximal empty rectangles
# =======================================================================================================================

def _enumerate_max_rects(
    bound: tuple,
    obstacles: np.ndarray,
    min_w: float,
    min_h: float,
    max_rects: int,
) -> list:
    """
    Enumerate maximal empty rectangles inside `bound`, tallest first.

    bound     : (x0, y0, x1, y1) page rectangle
    obstacles : (N, 4) float array of (x0, y0, x1, y1) obstacle bboxes

    Branch and bound with a height-first priority (height, then area as
    tie-break).  Any empty rectangle inside a node is no taller and no larger
    than the node itself, so the node priority is a valid lexicographic upper
    bound and the first obstacle-free rectangle popped is the tallest — and,
    among equally tall ones, the widest — remaining empty rectangle.  This is
    what makes vertical gutters come out whole: a tall rectangle claims its
    full vertical run first, and wide horizontal whitespace bands break
    around it, rather than area-maximal bands slicing the gutter into stubs.
    Splitting a bound on a pivot obstacle yields the four sub-bounds to its
    left / right / above / below.  Accepted rectangles are handled lazily:
    a popped rectangle overlapping an already-accepted one is split on it as
    if it were an obstacle, so results never overlap each other.
    """
    ox0, oy0, ox1, oy1 = obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], obstacles[:, 3]
    ocx = (ox0 + ox1) * 0.5
    ocy = (oy0 + oy1) * 0.5

    tiebreak = itertools.count()
    heap: list = []
    seen: set = set()
    accepted: list = []

    def push(rect: tuple) -> None:
        x0, y0, x1, y1 = rect
        w, h = x1 - x0, y1 - y0
        if w < min_w or h < min_h:
            return
        # Exact-coordinate key: duplicate bounds from different split orders are
        # bitwise-identical (coords always come from the same obstacle edges).
        # Never round here — conflating two distinct rects whose edges differ by
        # less than the rounding step drops a whole branch of the search and
        # fragments gutters.
        if rect in seen:
            return
        seen.add(rect)
        heapq.heappush(heap, (-h, -(w * h), next(tiebreak), rect))

    def split(rect: tuple, pivot: tuple) -> None:
        rx0, ry0, rx1, ry1 = rect
        px0, py0, px1, py1 = pivot
        push((rx0, ry0, px0, ry1))  # left of pivot
        push((px1, ry0, rx1, ry1))  # right of pivot
        push((rx0, ry0, rx1, py0))  # above pivot
        push((rx0, py1, rx1, ry1))  # below pivot

    push(bound)
    expansions = 0

    while heap and len(accepted) < max_rects and expansions < _MAX_NODE_EXPANSIONS:
        expansions += 1
        _, _, _, rect = heapq.heappop(heap)
        x0, y0, x1, y1 = rect

        # Lazy exclusion of already-accepted rectangles: split on the first
        # accepted rectangle this one overlaps, then continue.
        hit_accepted = None
        for acc in accepted:
            if x0 < acc[2] - _EPS and x1 > acc[0] + _EPS and y0 < acc[3] - _EPS and y1 > acc[1] + _EPS:
                hit_accepted = acc
                break
        if hit_accepted is not None:
            split(rect, hit_accepted)
            continue

        hits = np.nonzero(
            (ox0 < x1 - _EPS) & (ox1 > x0 + _EPS) & (oy0 < y1 - _EPS) & (oy1 > y0 + _EPS)
        )[0]

        if hits.size == 0:
            accepted.append(rect)
            continue

        # Pivot on the obstacle whose center is closest to the rect center —
        # produces balanced splits (Breuel's heuristic).
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        d2 = (ocx[hits] - cx) ** 2 + (ocy[hits] - cy) ** 2
        p = int(hits[int(np.argmin(d2))])
        # Clamp the pivot to the rect so sub-bounds never exceed it.
        pivot = (
            max(float(ox0[p]), x0), max(float(oy0[p]), y0),
            min(float(ox1[p]), x1), min(float(oy1[p]), y1),
        )
        split(rect, pivot)

    return accepted


# =======================================================================================================================
# Obstacle collection
# =======================================================================================================================

_BBOX_COLS = ["x_left", "y_top", "x_right", "y_bottom"]


def _drop_invalid(arr: np.ndarray) -> np.ndarray:
    """Drop degenerate / NaN boxes (allow zero-thickness rule lines: pad below)."""
    if arr.size == 0:
        return arr.reshape(0, 4)
    valid = np.isfinite(arr).all(axis=1) & (arr[:, 2] >= arr[:, 0]) & (arr[:, 3] >= arr[:, 1])
    return arr[valid]


def _page_obstacles(df: pd.DataFrame | None, page_number) -> np.ndarray:
    """Extract (N, 4) obstacle bboxes for one page; empty array if nothing usable."""
    if df is None or df.empty or not set(_BBOX_COLS).issubset(df.columns):
        return np.empty((0, 4), dtype=np.float64)

    page_bg_color = None
    if "shape_role" in df.columns:
        bg_rows = df[
            (df["page_number"] == page_number) & (df["shape_role"] == "page_background")
        ]
        if not bg_rows.empty and "non_stroking_color" in df.columns:
            page_bg_color = str(bg_rows["non_stroking_color"].iloc[0]).lower()
        df = df[df["shape_role"] != "page_background"]

    # Fill-only shapes painted in the page's own blank color (no stroke, same
    # color as the page_background shape) render indistinguishably from the
    # empty page — e.g. a thin decorative seam rect used to patch a gap
    # between design elements.  They cannot visually delimit a gutter, so
    # they must not act as obstacles, same reasoning as the page_background
    # exclusion above, just not full-page-sized.
    if page_bg_color and {"fill", "stroke", "non_stroking_color"}.issubset(df.columns):
        invisible = (
            df["fill"].fillna(False).astype(bool)
            & ~df["stroke"].fillna(False).astype(bool)
            & (df["non_stroking_color"].astype(str).str.lower() == page_bg_color)
        )
        df = df[~invisible]

    page = df[df["page_number"] == page_number]
    if page.empty:
        return np.empty((0, 4), dtype=np.float64)
    return _drop_invalid(page[_BBOX_COLS].to_numpy(dtype=np.float64))


def _word_page_obstacles(page_words: pd.DataFrame) -> np.ndarray:
    """
    Build obstacle bboxes for one page's words, collapsing struct/table groups.

    Native-PDF-only feature: struct_group_id and table_id are absent from OCR
    words, so this degrades to one obstacle per word there (unchanged
    behavior).  When present on native-PDF words:
      - words sharing a table_id collapse into ONE obstacle per table — a
        table's internal row/column gaps are never gutters, so the whole
        table masks out as a single block;
      - remaining words sharing a struct_group_id collapse into one obstacle
        per struct group — this keeps the small gap between two words that
        belong to the same tagged inline run (e.g. a split date or a
        hyphenated span) from registering as its own whitespace rect;
      - any leftover words (no group membership) are kept as individual
        word-level obstacles, same as before.
    Table membership takes priority: a word in a table is never also
    collapsed via struct_group_id.
    """
    if not set(_BBOX_COLS).issubset(page_words.columns):
        return np.empty((0, 4), dtype=np.float64)

    has_table = "table_id" in page_words.columns
    has_struct = "struct_group_id" in page_words.columns
    if not has_table and not has_struct:
        return _drop_invalid(page_words[_BBOX_COLS].to_numpy(dtype=np.float64))

    remaining = page_words
    parts: list = []

    if has_table:
        table_mask = remaining["table_id"].notna()
        if table_mask.any():
            tbl = remaining[table_mask].groupby("table_id", sort=False).agg(
                x_left=("x_left", "min"), y_top=("y_top", "min"),
                x_right=("x_right", "max"), y_bottom=("y_bottom", "max"),
            )
            parts.append(tbl[_BBOX_COLS].to_numpy(dtype=np.float64))
            remaining = remaining[~table_mask]

    if has_struct:
        struct_mask = remaining["struct_group_id"].notna()
        if struct_mask.any():
            grp = remaining[struct_mask].groupby("struct_group_id", sort=False).agg(
                x_left=("x_left", "min"), y_top=("y_top", "min"),
                x_right=("x_right", "max"), y_bottom=("y_bottom", "max"),
            )
            parts.append(grp[_BBOX_COLS].to_numpy(dtype=np.float64))
            remaining = remaining[~struct_mask]

    parts.append(remaining[_BBOX_COLS].to_numpy(dtype=np.float64))
    arr = np.vstack(parts) if parts else np.empty((0, 4), dtype=np.float64)
    return _drop_invalid(arr)


# =======================================================================================================================
# Public API
# =======================================================================================================================

def detect_gutters(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Find maximal whitespace rectangles per page (raw, unclassified).

    Rectangles that are wider than tall or that touch the left/right page
    border are dropped — they are horizontal bands / margins, never gutters.

    df_words  : requires page_number, x_left, x_right, y_top, y_bottom.
                page_width / page_height are used for the page bound when
                present; otherwise the bound falls back to content extents.
    df_shapes : optional; rows are treated as obstacles via the same bbox
                columns (rule lines naturally terminate/split rectangles).
    df_images : optional; image bboxes become obstacles too (PDF pipeline
                only — the OCR pipeline has no image dataframe).

    If df_words carries table_id / struct_group_id (native-PDF pipeline
    only — OCR words have neither), words are pre-grouped into obstacles
    before the search: each table_id collapses to one obstacle for the whole
    table, and each remaining struct_group_id collapses to one obstacle for
    that inline run.  See _word_page_obstacles for why.

    Returns one row per whitespace rectangle:
        page_number, gutter_id,
        gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
        gutter_width, gutter_height, gutter_area
    """
    out_cols = [
        "page_number", "gutter_id",
        "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
        "gutter_width", "gutter_height", "gutter_area",
    ]
    if df_words is None or df_words.empty:
        return pd.DataFrame(columns=out_cols)

    missing = {"page_number", *_BBOX_COLS} - set(df_words.columns)
    if missing:
        raise ValueError(f"df_words missing required columns: {sorted(missing)}")

    # Vertical text (TTB/BTT: rotated captions, spine text, watermarks) is
    # invisible to gutter detection — it sits in margins/edges and would
    # block or fragment whitespace rectangles that are real column gutters.
    if "text_orientation" in df_words.columns:
        orient = df_words["text_orientation"].astype(str).str.upper().str.strip()
        df_words = df_words[~orient.isin(["TTB", "BTT"])]
        if df_words.empty:
            return pd.DataFrame(columns=out_cols)

    has_page_dims = {"page_width", "page_height"}.issubset(df_words.columns)

    records: list = []
    for page_number, page_words in df_words.groupby("page_number", sort=True):
        word_obs = _word_page_obstacles(page_words)
        shape_obs = _page_obstacles(df_shapes, page_number)
        image_obs = _page_obstacles(df_images, page_number)
        obstacles = np.vstack([word_obs, shape_obs, image_obs])
        if obstacles.shape[0] == 0:
            continue

        if has_page_dims:
            page_w = float(page_words["page_width"].iloc[0])
            page_h = float(page_words["page_height"].iloc[0])
        else:
            page_w = float(obstacles[:, 2].max())
            page_h = float(obstacles[:, 3].max())
        if page_w <= 0 or page_h <= 0:
            continue
        bound = (0.0, 0.0, page_w, page_h)

        # Clip obstacles to the page bound so splits never escape it.
        obstacles[:, 0] = np.clip(obstacles[:, 0], 0.0, page_w)
        obstacles[:, 2] = np.clip(obstacles[:, 2], 0.0, page_w)
        obstacles[:, 1] = np.clip(obstacles[:, 1], 0.0, page_h)
        obstacles[:, 3] = np.clip(obstacles[:, 3], 0.0, page_h)

        rects = _enumerate_max_rects(
            bound, obstacles,
            min_w=_MIN_RECT_WIDTH, min_h=_MIN_RECT_HEIGHT,
            max_rects=_MAX_RECTS_PER_PAGE,
        )
        ox_left, oy_top = obstacles[:, 0], obstacles[:, 1]
        ox_right, oy_bottom = obstacles[:, 2], obstacles[:, 3]

        for x0, y0, x1, y1 in rects:
            # Drop rectangles that cannot be column gutters: wider than tall
            # (horizontal whitespace bands) or touching the left/right page
            # border (margins).
            if (x1 - x0) >= (y1 - y0):
                continue
            if x0 <= _PAGE_EDGE_EPS or x1 >= page_w - _PAGE_EDGE_EPS:
                continue
            # Both-sides obstacle test: a rect is maximal, so each vertical
            # edge is defined by whatever stopped it — a word, a shape/image,
            # the page border, or an already-claimed whitespace rect.  A real
            # column gutter is delimited by actual content (words, a table's
            # horizontal rule grazing in from the side, a figure) on both
            # flanks; fragments have the page border or another whitespace
            # rect on at least one side.  Require >= 1 word/shape/image
            # obstacle touching each vertical edge with y-overlap.
            y_ov = (oy_top < y1 - _EPS) & (oy_bottom > y0 + _EPS)
            if not np.any(y_ov & (np.abs(ox_right - x0) <= _SIDE_TEXT_TOL)):
                continue
            if not np.any(y_ov & (np.abs(ox_left - x1) <= _SIDE_TEXT_TOL)):
                continue
            records.append({
                "page_number": page_number,
                "gutter_x_left": x0,
                "gutter_x_right": x1,
                "gutter_y_top": y0,
                "gutter_y_bottom": y1,
            })

    if not records:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame.from_records(records)
    df["gutter_width"] = df["gutter_x_right"] - df["gutter_x_left"]
    df["gutter_height"] = df["gutter_y_bottom"] - df["gutter_y_top"]
    df["gutter_area"] = df["gutter_width"] * df["gutter_height"]
    df = df.sort_values(
        ["page_number", "gutter_x_left", "gutter_y_top"], kind="mergesort"
    ).reset_index(drop=True)
    df["gutter_id"] = range(1, len(df) + 1)
    return df[out_cols]

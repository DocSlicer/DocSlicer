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
    df_gutters = detect_gutters(df_words, df_shapes)              # stage 1 + 2
    df_gutters = score_gutters(df_gutters, df_words, df_shapes)   # stage 2 alone

df_gutters columns:
    page_number, gutter_id,
    gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
    gutter_width, gutter_height, gutter_area
score_gutters adds:
    score_too_wide, score_markers_left, score_numeric_flank,
    score_images_flank, score_small_cluster, score_stacked,
    score_line_boxed, score_divider, score_height,
    gutter_score, gutter_keep

Coordinate convention (matches the rest of the pipeline):
    y increases downward; y_top < y_bottom.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
import pandas as pd

from docslicer._utils.text_utils import list_marker_mask, numeric_value_mask

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

# ---- Stage 2: scoring weights (signed contributions; 0 when the signal is off) ----
_SCORE_TOO_WIDE: float = -10.0           # wider than _WIDE_GUTTER_WIDTH_FRAC of the page (de facto exclusion)
_SCORE_WIDE_MEDIUM: float = -5.0         # width between _WIDE_GUTTER_MEDIUM_FRAC and _WIDE_GUTTER_WIDTH_FRAC
_SCORE_WIDE_SLIGHT: float = -2.0         # width between _WIDE_GUTTER_SLIGHT_FRAC and _WIDE_GUTTER_MEDIUM_FRAC
_SCORE_ONLY_MARKERS_LEFT: float = -5.0   # left flank is nothing but bullets / list markers
_SCORE_ONLY_NUMERIC_FLANK: float = -4.0  # left or right flank is nothing but numeric / currency / % / dash tokens
_SCORE_ONLY_IMAGES_FLANK: float = -2.0   # left or right flank holds images but no words (icon / figure column)
_SCORE_SMALL_CLUSTER: float = -3.0       # short gutter sharing its y-span with many other short gutters (indicates a table instead of a gutter)
_SCORE_STACKED_NEIGHBOR: float = -2.0    # another gutter in the same x-span within close y proximity
_SCORE_LINE_BOXED: float = -2.0          # a horizontal line touches the gutter's top AND bottom (table row)
_SCORE_DIVIDES_HALVES_THIRDS: float = 2.0  # x-center on the page's 1/2, 1/3 or 2/3 line (2- / 3-col split)
_SCORE_DIVIDES_QUARTERS: float = 1.0       # x-center on the page's 1/4 or 3/4 line (4-col split)
_SCORE_TALL_THIRD: float = 1.0           # taller than 1/3 of the page
_SCORE_TALL_HALF: float = 2.0            # taller than 1/2 of the page
_SCORE_TALL_TWO_THIRDS: float = 3.0      # taller than 2/3 of the page

# ---- Stage 2: scoring thresholds ----
_WIDE_GUTTER_WIDTH_FRAC: float = 0.10     # wider than this fraction of page width = whitespace region, not a gutter
_WIDE_GUTTER_MEDIUM_FRAC: float = 0.07    # 5-10% of page width: suspiciously wide, medium penalty
_WIDE_GUTTER_SLIGHT_FRAC: float = 0.03    # 3-5% of page width: slightly wide, small penalty
_SMALL_GUTTER_HEIGHT_FRAC: float = 0.25   # "short" gutter = height below this fraction of page height
                                          # (0.25 not 0.20: a table-body-height gutter on a table that fills
                                          #  ~1/5 of the page lands right at 0.20 and must still count)
_SMALL_CLUSTER_MIN_NEIGHBORS: int = 3     # short gutter needs >= this many y-overlapping short peers to be a cluster
_STACKED_NEIGHBOR_MAX_GAP: float = 50.0   # pt - max vertical gap for the stacked-neighbor penalty
_LINE_TOUCH_TOL: float = 2.0              # pt - max distance between a line and a gutter edge to count as touching
_DIVIDER_CENTER_TOL_FRAC: float = 0.04    # of page width - tolerance for the divider-alignment bonuses
_SCORE_KEEP_THRESHOLD: float = 1.0        # gutter_keep = gutter_score >= this (placeholder decision rule)


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
# Stage 1: candidate filter
# =======================================================================================================================

def _filter_gutter_rects(rects: list, obstacles: np.ndarray, page_w: float) -> list:
    """
    Drop whitespace rectangles that cannot be column gutters.  This is a hard
    exclusion step, as decisive as scoring: a rectangle dropped here is never
    a candidate downstream.

    Three tests; a rect must pass all of them:
      - taller than wide — wider-than-tall rects are horizontal whitespace
        bands, never gutters;
      - clear of the left/right page border — border-touching rects are
        margins;
      - both-sides obstacle test: a rect is maximal, so each vertical edge is
        defined by whatever stopped it — a word, a shape/image, the page
        border, or an already-claimed whitespace rect.  A real column gutter
        is delimited by actual content (words, a table's horizontal rule
        grazing in from the side, a figure) on both flanks; fragments have
        the page border or another whitespace rect on at least one side.
        Require >= 1 word/shape/image obstacle touching each vertical edge
        with y-overlap.

    Returns the surviving (x0, y0, x1, y1) tuples.
    """
    ox_left, oy_top = obstacles[:, 0], obstacles[:, 1]
    ox_right, oy_bottom = obstacles[:, 2], obstacles[:, 3]

    kept: list = []
    for x0, y0, x1, y1 in rects:
        if (x1 - x0) >= (y1 - y0):
            continue
        if x0 <= _PAGE_EDGE_EPS or x1 >= page_w - _PAGE_EDGE_EPS:
            continue
        y_ov = (oy_top < y1 - _EPS) & (oy_bottom > y0 + _EPS)
        if not np.any(y_ov & (np.abs(ox_right - x0) <= _SIDE_TEXT_TOL)):
            continue
        if not np.any(y_ov & (np.abs(ox_left - x1) <= _SIDE_TEXT_TOL)):
            continue
        kept.append((x0, y0, x1, y1))
    return kept


# =======================================================================================================================
# Stage 2: gutter scoring
# =======================================================================================================================

_SCORE_COLS = [

    "score_too_wide", "score_markers_left", "score_numeric_flank",
    "score_images_flank", "score_small_cluster", "score_stacked",
    "score_line_boxed", "score_divider", "score_height",
]


def score_gutters(
    df_gutters: pd.DataFrame,
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Stage 2: score each whitespace rectangle on how likely it is to be a real
    column gutter vs. noise (table-interior whitespace, bullet indents,
    fragment chains).

    Adds one signed column per signal (0.0 when the signal is off) plus the
    sum, so every rectangle's verdict is auditable in the debug CSV:

        score_too_wide     -10  wider than 10% of the page width — a whitespace
                                region between content blocks, not a column
                                gutter (de facto exclusion);
                            -5  5-10% of the page width (suspiciously wide);
                            -2  3-5% of the page width (slightly wide)
        score_markers_left  -5  left flank is nothing but bullet / list-marker
                                tokens (bullet indent, footnote markers)
        score_numeric_flank -4  left or right flank is nothing but numeric /
                                currency / percent / dash tokens (table value
                                columns)
        score_images_flank  -2  left or right flank holds at least one image
                                and no words — an icon / figure column, e.g.
                                warning glyphs beside a note block
                                (needs df_images; 0 when not passed)
        score_small_cluster -3  gutter is short (< 25% of page height) and
                                shares its y-span with >= 3 other short
                                gutters (table interior)
        score_stacked       -2  another gutter overlaps this one's x-span
                                within 50 pt vertically (fragmented chain)
        score_line_boxed    -2  a horizontal line crosses the gutter's full
                                x-span at both its top and bottom edge —
                                whitespace boxed into a table row
                                (needs df_shapes; 0 when not passed)
        score_divider       +2  x-center on the page's 1/2, 1/3 or 2/3 line;
                            +1  on the 1/4 or 3/4 line
        score_height        +1/+2/+3  taller than 1/3 / 1/2 / 2/3 of the page
        gutter_score            sum of the above
        gutter_keep             gutter_score >= _SCORE_KEEP_THRESHOLD
                                (placeholder decision rule)

    A gutter's *flank* is the horizontal strip beside it: from its vertical
    edge out to the nearest y-overlapping gutter on that side, or the page
    edge if there is none.  Only words / images that y-overlap the gutter
    count, and they must lie entirely inside the strip.

    Everything is computed per page with numpy broadcasting (words x gutters,
    gutters x gutters); word text masks are vectorized once over the whole
    dataframe up front — no per-word Python loops.
    """
    if df_gutters is None or df_gutters.empty:
        out = pd.DataFrame() if df_gutters is None else df_gutters.copy()
        for col in [*_SCORE_COLS, "gutter_score"]:
            out[col] = pd.Series(dtype=np.float64)
        out["gutter_keep"] = pd.Series(dtype=bool)
        return out

    # Same visibility rule as detection: vertical text never flanks a gutter.
    words = df_words
    if words is not None and not words.empty and "text_orientation" in words.columns:
        orient = words["text_orientation"].astype(str).str.upper().str.strip()
        words = words[~orient.isin(["TTB", "BTT"])]

    have_words = (
        words is not None and not words.empty
        and {"page_number", "text", *_BBOX_COLS}.issubset(words.columns)
    )
    if have_words:
        w_marker_all = list_marker_mask(words["text"]).to_numpy()
        w_numeric_all = numeric_value_mask(words["text"]).to_numpy()
        word_pages = {
            page: idx for page, idx in words.groupby("page_number", sort=False).indices.items()
        }
        w_bbox_all = words[_BBOX_COLS].to_numpy(dtype=np.float64)
        has_page_dims = {"page_width", "page_height"}.issubset(words.columns)
        if has_page_dims:
            page_w_all = words["page_width"].to_numpy(dtype=np.float64)
            page_h_all = words["page_height"].to_numpy(dtype=np.float64)
    else:
        word_pages = {}
        has_page_dims = False

    # Horizontal lines (table borders, rules, underlines).  Line-ness comes
    # from the shape classification, NOT thickness — the shape merger keeps
    # fused parallel border hairlines classified as "line".
    line_pages: dict = {}
    if (
        df_shapes is not None and not df_shapes.empty
        and {"page_number", *_BBOX_COLS}.issubset(df_shapes.columns)
    ):
        lines = df_shapes
        if "shape_orientation" in lines.columns:
            lines = lines[lines["shape_orientation"] == "horizontal"]
        else:
            lines = lines[
                (lines["x_right"] - lines["x_left"]) > (lines["y_bottom"] - lines["y_top"])
            ]
        if "shape_type" in lines.columns:
            lines = lines[lines["shape_type"] == "line"]
        l_bbox_all = lines[_BBOX_COLS].to_numpy(dtype=np.float64)
        line_pages = {
            page: l_bbox_all[idx]
            for page, idx in lines.groupby("page_number", sort=False).indices.items()
        }

    # Image bboxes for the images-only-flank signal (PDF pipeline only).
    image_pages: dict = {}
    if (
        df_images is not None and not df_images.empty
        and {"page_number", *_BBOX_COLS}.issubset(df_images.columns)
    ):
        i_bbox_all = df_images[_BBOX_COLS].to_numpy(dtype=np.float64)
        image_pages = {
            page: i_bbox_all[idx]
            for page, idx in df_images.groupby("page_number", sort=False).indices.items()
        }

    n_total = len(df_gutters)
    scores = {col: np.zeros(n_total, dtype=np.float64) for col in _SCORE_COLS}

    gx0_all = df_gutters["gutter_x_left"].to_numpy(dtype=np.float64)
    gx1_all = df_gutters["gutter_x_right"].to_numpy(dtype=np.float64)
    gy0_all = df_gutters["gutter_y_top"].to_numpy(dtype=np.float64)
    gy1_all = df_gutters["gutter_y_bottom"].to_numpy(dtype=np.float64)

    for page_number, page_idx in df_gutters.groupby("page_number", sort=False).indices.items():
        gx0, gx1 = gx0_all[page_idx], gx1_all[page_idx]
        gy0, gy1 = gy0_all[page_idx], gy1_all[page_idx]
        gh = gy1 - gy0
        n = len(page_idx)

        widx = word_pages.get(page_number)
        if widx is not None:
            wb = w_bbox_all[widx]
            wx0, wy0, wx1, wy1 = wb[:, 0], wb[:, 1], wb[:, 2], wb[:, 3]
            w_marker = w_marker_all[widx]
            w_numeric = w_numeric_all[widx]

        # Page dimensions: same source as detection, falling back to extents.
        if widx is not None and has_page_dims:
            page_w = float(page_w_all[widx[0]])
            page_h = float(page_h_all[widx[0]])
        elif widx is not None:
            page_w = float(max(wx1.max(), gx1.max()))
            page_h = float(max(wy1.max(), gy1.max()))
        else:
            page_w = float(gx1.max())
            page_h = float(gy1.max())
        if page_w <= 0 or page_h <= 0:
            continue

        # Width tiers: the wider a rect, the less it looks like a column
        # gutter (highest tier wins, not cumulative).
        w_frac = (gx1 - gx0) / page_w
        scores["score_too_wide"][page_idx] = np.select(
            [
                w_frac > _WIDE_GUTTER_WIDTH_FRAC,
                w_frac > _WIDE_GUTTER_MEDIUM_FRAC,
                w_frac > _WIDE_GUTTER_SLIGHT_FRAC,
            ],
            [_SCORE_TOO_WIDE, _SCORE_WIDE_MEDIUM, _SCORE_WIDE_SLIGHT],
            default=0.0,
        )

        # ---- gutter-vs-gutter geometry (n x n, n <= _MAX_RECTS_PER_PAGE) ----
        yov = (gy0[:, None] < gy1[None, :] - _EPS) & (gy1[:, None] > gy0[None, :] + _EPS)
        xov = (gx0[:, None] < gx1[None, :] - _EPS) & (gx1[:, None] > gx0[None, :] + _EPS)
        np.fill_diagonal(yov, False)
        np.fill_diagonal(xov, False)

        # Short-gutter cluster: table interiors shed many short whitespace
        # rects side by side in the same horizontal band.
        small = gh < _SMALL_GUTTER_HEIGHT_FRAC * page_h
        n_small_peers = (yov & small[None, :]).sum(axis=1)
        small_cluster = small & (n_small_peers >= _SMALL_CLUSTER_MIN_NEIGHBORS)
        scores["score_small_cluster"][page_idx] = np.where(small_cluster, _SCORE_SMALL_CLUSTER, 0.0)

        # Stacked neighbor: accepted rects never overlap, so an x-overlapping
        # pair is vertically disjoint; gap is the signed vertical distance.
        gap = np.maximum(gy0[None, :] - gy1[:, None], gy0[:, None] - gy1[None, :])
        stacked = (xov & (gap < _STACKED_NEIGHBOR_MAX_GAP)).any(axis=1)
        scores["score_stacked"][page_idx] = np.where(stacked, _SCORE_STACKED_NEIGHBOR, 0.0)

        # Line-boxed: a horizontal line crossing the gutter's full x-span at
        # both its top and bottom edge means the whitespace is boxed into a
        # table row, not running free between columns.  (n x L broadcast)
        page_lines = line_pages.get(page_number)
        if page_lines is not None:
            lx0, ly0, lx1, ly1 = (
                page_lines[:, 0], page_lines[:, 1], page_lines[:, 2], page_lines[:, 3]
            )
            crosses = (
                (lx0[None, :] <= gx0[:, None] + _LINE_TOUCH_TOL)
                & (lx1[None, :] >= gx1[:, None] - _LINE_TOUCH_TOL)
            )
            at_top = (
                (ly0[None, :] <= gy0[:, None] + _LINE_TOUCH_TOL)
                & (ly1[None, :] >= gy0[:, None] - _LINE_TOUCH_TOL)
            )
            at_bottom = (
                (ly0[None, :] <= gy1[:, None] + _LINE_TOUCH_TOL)
                & (ly1[None, :] >= gy1[:, None] - _LINE_TOUCH_TOL)
            )
            line_boxed = (crosses & at_top).any(axis=1) & (crosses & at_bottom).any(axis=1)
            scores["score_line_boxed"][page_idx] = np.where(
                line_boxed, _SCORE_LINE_BOXED, 0.0
            )

        # Divider alignment: gutters that split the page into 2/3 (or 4)
        # equal columns sit on the corresponding division lines.
        cx = (gx0 + gx1) * 0.5
        tol = _DIVIDER_CENTER_TOL_FRAC * page_w
        halves_thirds = np.array([page_w / 2.0, page_w / 3.0, 2.0 * page_w / 3.0])
        quarters = np.array([page_w / 4.0, 3.0 * page_w / 4.0])
        on_ht = (np.abs(cx[:, None] - halves_thirds[None, :]) <= tol).any(axis=1)
        on_q = (np.abs(cx[:, None] - quarters[None, :]) <= tol).any(axis=1)
        scores["score_divider"][page_idx] = np.where(
            on_ht, _SCORE_DIVIDES_HALVES_THIRDS, np.where(on_q, _SCORE_DIVIDES_QUARTERS, 0.0)
        )

        # Height tiers (highest tier wins, not cumulative).
        h_frac = gh / page_h
        scores["score_height"][page_idx] = np.select(
            [h_frac > 2.0 / 3.0, h_frac > 0.5, h_frac > 1.0 / 3.0],
            [_SCORE_TALL_TWO_THIRDS, _SCORE_TALL_HALF, _SCORE_TALL_THIRD],
            default=0.0,
        )

        # ---- flank contents (n x W / n x I broadcasts) ----
        # Flank boundary: nearest y-overlapping gutter on that side, else the
        # page edge.
        left_of = yov & (gx1[None, :] <= gx0[:, None] + _EPS)
        right_of = yov & (gx0[None, :] >= gx1[:, None] - _EPS)
        lb = np.where(left_of, np.broadcast_to(gx1[None, :], (n, n)), 0.0).max(axis=1)
        rb = np.where(right_of, np.broadcast_to(gx0[None, :], (n, n)), page_w).min(axis=1)

        page_images = image_pages.get(page_number)
        if page_images is not None:
            ix0, iy0, ix1, iy1 = (
                page_images[:, 0], page_images[:, 1], page_images[:, 2], page_images[:, 3]
            )
            i_yov = (iy0[None, :] < gy1[:, None] - _EPS) & (iy1[None, :] > gy0[:, None] + _EPS)
            left_img = (
                i_yov
                & (ix1[None, :] <= gx0[:, None] + _SIDE_TEXT_TOL)
                & (ix0[None, :] >= lb[:, None] - _SIDE_TEXT_TOL)
            ).any(axis=1)
            right_img = (
                i_yov
                & (ix0[None, :] >= gx1[:, None] - _SIDE_TEXT_TOL)
                & (ix1[None, :] <= rb[:, None] + _SIDE_TEXT_TOL)
            ).any(axis=1)
        else:
            left_img = np.zeros(n, dtype=bool)
            right_img = np.zeros(n, dtype=bool)

        if widx is not None:
            w_yov = (wy0[None, :] < gy1[:, None] - _EPS) & (wy1[None, :] > gy0[:, None] + _EPS)
            left_sel = (
                w_yov
                & (wx1[None, :] <= gx0[:, None] + _SIDE_TEXT_TOL)
                & (wx0[None, :] >= lb[:, None] - _SIDE_TEXT_TOL)
            )
            right_sel = (
                w_yov
                & (wx0[None, :] >= gx1[:, None] - _SIDE_TEXT_TOL)
                & (wx1[None, :] <= rb[:, None] + _SIDE_TEXT_TOL)
            )

            left_any = left_sel.any(axis=1)
            right_any = right_sel.any(axis=1)
            left_only_markers = left_any & ~(left_sel & ~w_marker[None, :]).any(axis=1)
            left_only_numeric = left_any & ~(left_sel & ~w_numeric[None, :]).any(axis=1)
            right_only_numeric = right_any & ~(right_sel & ~w_numeric[None, :]).any(axis=1)

            scores["score_markers_left"][page_idx] = np.where(
                left_only_markers, _SCORE_ONLY_MARKERS_LEFT, 0.0
            )
            # Marker-only flanks are already fully penalized above; don't hit the
            # same gutter twice when the markers happen to be numeric too ("1.").
            numeric_flank = (left_only_numeric & ~left_only_markers) | right_only_numeric
            scores["score_numeric_flank"][page_idx] = np.where(
                numeric_flank, _SCORE_ONLY_NUMERIC_FLANK, 0.0
            )
        else:
            left_any = np.zeros(n, dtype=bool)
            right_any = np.zeros(n, dtype=bool)

        # Images-only flank: at least one image and zero words on a side —
        # an icon / figure column (warning glyphs beside a note block), not
        # a text column boundary.
        images_flank = (left_img & ~left_any) | (right_img & ~right_any)
        scores["score_images_flank"][page_idx] = np.where(
            images_flank, _SCORE_ONLY_IMAGES_FLANK, 0.0
        )

    out = df_gutters.copy()
    for col in _SCORE_COLS:
        out[col] = scores[col]
    out["gutter_score"] = sum(scores[col] for col in _SCORE_COLS)
    out["gutter_keep"] = out["gutter_score"] >= _SCORE_KEEP_THRESHOLD
    return out


# =======================================================================================================================
# Public API
# =======================================================================================================================

def detect_gutters(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Find maximal whitespace rectangles per page, then score them (stage 1 + 2).

    Rectangles that are wider than tall or that touch the left/right page
    border are dropped — they are horizontal bands / margins, never gutters.

    df_words  : requires page_number, x_left, x_right, y_top, y_bottom.
                page_width / page_height are used for the page bound when
                present; otherwise the bound falls back to content extents.
    df_shapes : optional; rows are treated as obstacles via the same bbox
                columns (rule lines naturally terminate/split rectangles).
    df_images : optional; image bboxes become obstacles too (PDF pipeline
                only — the OCR pipeline has no image dataframe).
    debug     : when False (default), rows with gutter_keep == False are
                removed from the result; when True, every scored rectangle
                is returned so rejects can be audited in the debug CSV.

    If df_words carries table_id / struct_group_id (native-PDF pipeline
    only — OCR words have neither), words are pre-grouped into obstacles
    before the search: each table_id collapses to one obstacle for the whole
    table, and each remaining struct_group_id collapses to one obstacle for
    that inline run.  See _word_page_obstacles for why.

    Returns one row per whitespace rectangle:
        page_number, gutter_id,
        gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
        gutter_width, gutter_height, gutter_area,
        plus the score_* / gutter_score / gutter_keep columns from
        score_gutters (stage 2 runs on the result before returning).
    """
    out_cols = [
        "page_number", "gutter_id",
        "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
        "gutter_width", "gutter_height", "gutter_area",
    ]
    if df_words is None or df_words.empty:
        return score_gutters(pd.DataFrame(columns=out_cols), df_words)

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
            return score_gutters(pd.DataFrame(columns=out_cols), df_words)

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
        rects = _filter_gutter_rects(rects, obstacles, page_w)

        for x0, y0, x1, y1 in rects:
            records.append({
                "page_number": page_number,
                "gutter_x_left": x0,
                "gutter_x_right": x1,
                "gutter_y_top": y0,
                "gutter_y_bottom": y1,
            })

    if not records:
        return score_gutters(pd.DataFrame(columns=out_cols), df_words)

    df = pd.DataFrame.from_records(records)
    df["gutter_width"] = df["gutter_x_right"] - df["gutter_x_left"]
    df["gutter_height"] = df["gutter_y_bottom"] - df["gutter_y_top"]
    df["gutter_area"] = df["gutter_width"] * df["gutter_height"]
    df = df.sort_values(
        ["page_number", "gutter_x_left", "gutter_y_top"], kind="mergesort"
    ).reset_index(drop=True)
    df["gutter_id"] = range(1, len(df) + 1)
    df = df[out_cols]

    df_gutters = score_gutters(df, df_words, df_shapes, df_images)
    if not debug:
        df_gutters = df_gutters[df_gutters["gutter_keep"]].reset_index(drop=True)


    return df_gutters



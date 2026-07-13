"""
gutter_detector_v3.py

Ground-up rebuild of the gutter detector around inspectable intermediate
frames.  Pipeline (see also page_zones.py, which owns step 0):

    0. page zones            assign_page_zones: y_line_id + header/footer
                             bands (page_zones.py, run at orchestrator level)
    1. collect_obstacles     body-zone words (+ table / struct-group
                             collapse), shapes, images, table grids — one
                             labeled dataframe                    [this file]
    2. enumerate_rects       Breuel branch-and-bound within the per-page
                             body bound (compute_body_bounds over df_bands);
                             obstacles fully outside the bound are dropped,
                             partial ones clipped                 [this file]
    3. apply_hard_gates      named gate columns instead of silent drops;
                             flank edges labeled with the obstacle sources
                             that touch them                      [this file]
    4. build_flank_context   per gutter-side KPIs grouped by y_line_id;
                             debug mode materializes the flank text line by
                             line                                 [this file]
    5. scoring               declarative bands + expressions over the
                             flank-context frame                       (TODO)
    6. merge onto words      unchanged from v2                         (TODO)

Step 1 — collect_obstacles.  Everything that can stop or split a whitespace
rectangle, one row per obstacle, tagged with where it came from:

    word          an individual word bbox
    table         all words sharing a table_id, collapsed to one bbox — a
                  table's internal row/column gaps are never gutters
    struct_group  all remaining words sharing a struct_group_id, collapsed —
                  keeps the gap inside a tagged inline run (split date,
                  hyphenated span) from registering as whitespace
    shape         a drawn shape (rule line, rect, ...); the page_background
                  shape and fill-only shapes painted in the page's own
                  background color are excluded — they render invisibly and
                  cannot visually delimit a gutter
    image         an image bbox (off-page/degenerate handling is the bound
                  clip's job in step 2, where fully-outside obstacles are
                  dropped rather than clipped to slivers)
    grid          all ruling lines sharing a table_grid_id via df_grid_cells,
                  collapsed to one bbox per grid — same reasoning as table,
                  from shapes instead of words

Word handling mirrors detection visibility: vertical (TTB/BTT) text never
participates, and when page_zone is present only body-zone words do (header/
footer words are excluded from the reading area entirely, not clipped).
Table membership takes priority over struct_group membership.

The source tag is the debugging backbone: hard gates (step 3) can name the
obstacle kind that defined a rect's edge, and flank tests can distinguish
"no text beside this gutter" from "an image beside this gutter".

Public API:
    df_rects     = detect_gutters(df_words, df_shapes, df_images,
                                  df_grid_cells, df_bands, config,
                                  debug=False)                       # steps 0-3
    df_obstacles = collect_obstacles(df_words, df_shapes, df_images, df_grid_cells)
    df_bounds    = compute_body_bounds(df_words, df_bands)
    df_rects     = enumerate_rects(df_obstacles, df_bounds, config)
    df_rects     = apply_hard_gates(df_rects, df_obstacles, df_bounds, config)
    df_rects     = build_flank_context(df_rects, df_words, config, debug)

detect_gutters is the WIP end-to-end endpoint: today it returns the
hard-gated whitespace rectangles with their flank-context KPIs (debug=True
keeps the gate rejects and materializes the per-line flank text); as
stages 5-6 land, its output grows score columns and it will return
(df_words, df_gutters) like v2.

df_obstacles columns:
    page_number, obstacle_source, x_left, y_top, x_right, y_bottom
df_bounds columns (one row per page: the rect-enumeration area):
    page_number, x_min, y_min, x_max, y_max, page_width, page_height
df_rects columns (one row per maximal whitespace rectangle; v2 gutter
naming so the df_viewer debug tooling renders them as-is):
    page_number, gutter_id,
    gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
    gutter_width, gutter_height, gutter_area

Coordinate convention (matches the rest of the pipeline):
    y increases downward; y_top < y_bottom.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

# =======================================================================================================================
# CONFIG
# =======================================================================================================================


@dataclass(frozen=True)
class GutterConfig:
    min_rect_width: float = 9.2         # pt - minimum whitespace rectangle width
    min_rect_height: float = 30.0       # pt - minimum whitespace rectangle height
    max_rects_per_page: int = 30        # stop enumerating after this many rectangles per page
    max_node_expansions: int = 200_000  # hard safety cap on branch-and-bound nodes per page
    # ---- step 3: hard gates ----
    bound_edge_eps: float = 1.0         # pt - rect within this of the bound's left/right edge = margin
    side_touch_tol: float = 1.0         # pt - obstacle edge within this of a rect edge counts as touching
    # ---- step 4: flank context ----
    flank_y_pad: float = 4.0            # pt - shrink the gutter's y-span at both ends before flank word
                                        # tests (descenders/ascenders of the lines that stopped the
                                        # gutter graze its ends and must not count as flank text)


_EPS: float = 1e-6  # strict-overlap epsilon: touching edges do not count as overlap

_BBOX_COLS = ["x_left", "y_top", "x_right", "y_bottom"]
_OBSTACLE_COLS = ["page_number", "obstacle_source", *_BBOX_COLS]
_BOUND_COLS = ["page_number", "x_min", "y_min", "x_max", "y_max", "page_width", "page_height"]
# v2 gutter naming, kept for the df_viewer debug tooling: every whitespace
# rect is a gutter *candidate*, and the viewer keys on gutter_id + bbox.
_RECT_COLS = [
    "page_number", "gutter_id",
    "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
    "gutter_width", "gutter_height", "gutter_area",
]


# =======================================================================================================================
# Step 1: obstacle collection
# =======================================================================================================================

def _bbox_part(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """One obstacle row per input row, tagged with its source."""
    part = df[["page_number", *_BBOX_COLS]].copy()
    part[_BBOX_COLS] = part[_BBOX_COLS].astype(np.float64)
    part["obstacle_source"] = source
    return part[_OBSTACLE_COLS]


def _grouped_part(df: pd.DataFrame, key: str, source: str) -> pd.DataFrame:
    """One obstacle row per (page, key) group: the group's bounding bbox."""
    grp = (
        df.groupby(["page_number", key], sort=False)
        .agg(
            x_left=("x_left", "min"), y_top=("y_top", "min"),
            x_right=("x_right", "max"), y_bottom=("y_bottom", "max"),
        )
        .reset_index()
    )
    grp["obstacle_source"] = source
    return grp[_OBSTACLE_COLS]


def _word_obstacles(df_words: pd.DataFrame) -> list:
    """
    Word-derived obstacle parts: table / struct-group collapse first (native
    PDF only — OCR words carry neither column and degrade to one obstacle
    per word), leftover words individually.
    """
    words = df_words
    if "text_orientation" in words.columns:
        orient = words["text_orientation"].astype(str).str.upper().str.strip()
        words = words[~orient.isin(["TTB", "BTT"])]
    if "page_zone" in words.columns:
        words = words[words["page_zone"] == "body"]
    if words.empty:
        return []

    parts: list = []
    if "table_id" in words.columns:
        mask = words["table_id"].notna()
        if mask.any():
            parts.append(_grouped_part(words[mask], "table_id", "table"))
            words = words[~mask]
    if "struct_group_id" in words.columns:
        mask = words["struct_group_id"].notna()
        if mask.any():
            parts.append(_grouped_part(words[mask], "struct_group_id", "struct_group"))
            words = words[~mask]
    if not words.empty:
        parts.append(_bbox_part(words, "word"))
    return parts


def _shape_obstacles(df_shapes: pd.DataFrame) -> list:
    """
    Shape obstacles, minus what renders invisibly: the page_background shape
    itself, and fill-only shapes painted in that page's background color (a
    thin decorative seam rect patching a gap between design elements looks
    exactly like the empty page — it cannot visually delimit a gutter).
    """
    shapes = df_shapes
    if "shape_role" in shapes.columns:
        bg = shapes[shapes["shape_role"] == "page_background"]
        shapes = shapes[shapes["shape_role"] != "page_background"]
        if (
            not bg.empty and not shapes.empty
            and {"fill", "stroke", "non_stroking_color"}.issubset(shapes.columns)
        ):
            bg_color = (
                bg.groupby("page_number")["non_stroking_color"]
                .first().astype(str).str.lower()
            )
            page_bg = shapes["page_number"].map(bg_color)
            invisible = (
                page_bg.notna()
                & shapes["fill"].fillna(False).astype(bool)
                & ~shapes["stroke"].fillna(False).astype(bool)
                & (shapes["non_stroking_color"].astype(str).str.lower() == page_bg)
            )
            shapes = shapes[~invisible]
    if shapes.empty:
        return []
    return [_bbox_part(shapes, "shape")]


def collect_obstacles(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
    df_grid_cells: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Step 1: gather every gutter-stopping obstacle into one labeled frame.

    See the module docstring for the source taxonomy and the visibility
    rules per source.  Degenerate boxes (NaN / inverted edges) are dropped;
    zero-thickness boxes survive (rule lines).  Off-page boxes survive too —
    step 2 drops obstacles that miss the body bound and clips the rest.

    Returns a dataframe with columns
        page_number, obstacle_source, x_left, y_top, x_right, y_bottom
    sorted by (page_number, obstacle_source) with a fresh 0..n-1 index.
    """
    parts: list = []

    if df_words is not None and not df_words.empty:
        missing = {"page_number", *_BBOX_COLS} - set(df_words.columns)
        if missing:
            raise ValueError(f"df_words missing required columns: {sorted(missing)}")
        parts += _word_obstacles(df_words)

    if (
        df_shapes is not None and not df_shapes.empty
        and {"page_number", *_BBOX_COLS}.issubset(df_shapes.columns)
    ):
        parts += _shape_obstacles(df_shapes)

    if (
        df_images is not None and not df_images.empty
        and {"page_number", *_BBOX_COLS}.issubset(df_images.columns)
    ):
        parts.append(_bbox_part(df_images, "image"))

    if (
        df_grid_cells is not None and not df_grid_cells.empty
        and {"page_number", "table_grid_id", *_BBOX_COLS}.issubset(df_grid_cells.columns)
    ):
        parts.append(_grouped_part(df_grid_cells, "table_grid_id", "grid"))

    if not parts:
        return pd.DataFrame(columns=_OBSTACLE_COLS)

    out = pd.concat(parts, ignore_index=True)
    bbox = out[_BBOX_COLS].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(bbox).all(axis=1)
        & (bbox[:, 2] >= bbox[:, 0]) & (bbox[:, 3] >= bbox[:, 1])
    )
    out = out[valid]
    return out.sort_values(
        ["page_number", "obstacle_source"], kind="mergesort"
    ).reset_index(drop=True)


# =======================================================================================================================
# Breuel branch-and-bound: maximal empty rectangles (ported unchanged from v2)
# =======================================================================================================================

def _enumerate_max_rects(
    bound: tuple,
    obstacles: np.ndarray,
    min_w: float,
    min_h: float,
    max_rects: int,
    max_expansions: int,
) -> list:
    """
    Enumerate maximal empty rectangles inside `bound`, tallest first.

    bound     : (x0, y0, x1, y1) rectangle to search within
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

    while heap and len(accepted) < max_rects and expansions < max_expansions:
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
# Step 2: body bounds + rect enumeration
# =======================================================================================================================

def compute_body_bounds(
    df_words: pd.DataFrame,
    df_bands: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One enumeration bound per page: the full page, shrunk vertically to the
    body region where header/footer bands were detected (page_zones step 0).

    The band's INNER edges bound the body — band_y_bottom is the body's
    first text line's top edge, band_y_top the last one's bottom — so the
    band whitespace itself is excluded and rect/gutter heights come out
    body-relative.

    Page width/height come from page_width / page_height columns when
    present, else fall back to the page's word extents.  Pages without
    words get no bound (and therefore no rects).

    Returns a dataframe with columns
        page_number, x_min, y_min, x_max, y_max, page_width, page_height
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(columns=_BOUND_COLS)

    has_dims = {"page_width", "page_height"}.issubset(df_words.columns)
    agg = {"x_ext": ("x_right", "max"), "y_ext": ("y_bottom", "max")}
    if has_dims:
        agg["page_width"] = ("page_width", "first")
        agg["page_height"] = ("page_height", "first")
    pages = df_words.groupby("page_number", sort=True).agg(**agg).reset_index()
    if not has_dims:
        pages["page_width"] = pages["x_ext"]
        pages["page_height"] = pages["y_ext"]

    pages["x_min"] = 0.0
    pages["y_min"] = 0.0
    pages["x_max"] = pages["page_width"].astype(np.float64)
    pages["y_max"] = pages["page_height"].astype(np.float64)

    if df_bands is not None and not df_bands.empty:
        for role, col, target in (
            ("header", "band_y_bottom", "y_min"),
            ("footer", "band_y_top", "y_max"),
        ):
            edge = (
                df_bands[df_bands["band_role"] == role]
                .set_index("page_number")[col]
            )
            mapped = pages["page_number"].map(edge)
            pages[target] = mapped.fillna(pages[target]).astype(np.float64)

    pages = pages[(pages["x_max"] > pages["x_min"]) & (pages["y_max"] > pages["y_min"])]
    return pages[_BOUND_COLS].reset_index(drop=True)


def enumerate_rects(
    df_obstacles: pd.DataFrame,
    df_bounds: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 2: maximal whitespace rectangles per page, within the body bound.

    Obstacles that do not intersect their page's bound are DROPPED (header/
    footer words already never reach the obstacle frame, but off-page images
    and band-zone shapes land here) — v2 instead clipped them to the bound,
    pinning degenerate slivers to its edges.  Partially-inside obstacles are
    clipped so splits never escape the bound.

    Purely geometric: every rectangle clearing min_rect_width/height is
    returned.  Gutter-ness (hard gates, scoring) is later stages' job — the
    gutter_* column naming (see _RECT_COLS) marks candidacy, not a verdict.

    Returns a dataframe with columns
        page_number, gutter_id,
        gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
        gutter_width, gutter_height, gutter_area
    with gutter_id unique across the document.
    """
    if df_bounds is None or df_bounds.empty:
        return pd.DataFrame(columns=_RECT_COLS)

    obstacle_pages: dict = {}
    if df_obstacles is not None and not df_obstacles.empty:
        obs_bbox = df_obstacles[_BBOX_COLS].to_numpy(dtype=np.float64)
        obstacle_pages = {
            page: obs_bbox[idx]
            for page, idx in df_obstacles.groupby("page_number", sort=False).indices.items()
        }

    records: list = []
    for row in df_bounds.itertuples(index=False):
        bound = (float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max))
        obs = obstacle_pages.get(row.page_number)
        if obs is None or obs.shape[0] == 0:
            continue

        inside = (
            (obs[:, 2] > bound[0] + _EPS) & (obs[:, 0] < bound[2] - _EPS)
            & (obs[:, 3] > bound[1] + _EPS) & (obs[:, 1] < bound[3] - _EPS)
        )
        obs = obs[inside]
        if obs.shape[0] == 0:
            continue
        obs = obs.copy()
        obs[:, 0] = np.clip(obs[:, 0], bound[0], bound[2])
        obs[:, 2] = np.clip(obs[:, 2], bound[0], bound[2])
        obs[:, 1] = np.clip(obs[:, 1], bound[1], bound[3])
        obs[:, 3] = np.clip(obs[:, 3], bound[1], bound[3])

        rects = _enumerate_max_rects(
            bound, obs,
            min_w=config.min_rect_width, min_h=config.min_rect_height,
            max_rects=config.max_rects_per_page,
            max_expansions=config.max_node_expansions,
        )
        for x0, y0, x1, y1 in rects:
            records.append({
                "page_number": row.page_number,
                "gutter_x_left": x0, "gutter_y_top": y0,
                "gutter_x_right": x1, "gutter_y_bottom": y1,
            })

    if not records:
        return pd.DataFrame(columns=_RECT_COLS)

    out = pd.DataFrame.from_records(records)
    out["gutter_width"] = out["gutter_x_right"] - out["gutter_x_left"]
    out["gutter_height"] = out["gutter_y_bottom"] - out["gutter_y_top"]
    out["gutter_area"] = out["gutter_width"] * out["gutter_height"]
    out = out.sort_values(
        ["page_number", "gutter_x_left", "gutter_y_top"], kind="mergesort"
    ).reset_index(drop=True)
    out["gutter_id"] = range(1, len(out) + 1)
    return out[_RECT_COLS]


# =======================================================================================================================
# Step 3: hard gates
# =======================================================================================================================

_GATE_COLS = ["gate_wider_than_tall", "gate_bound_border", "gate_no_left_flank", "gate_no_right_flank"]


def apply_hard_gates(
    df_rects: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    df_bounds: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 3: flag whitespace rectangles that cannot be column gutters.  As
    decisive as v2's filter, but recorded instead of silently dropped — one
    boolean column per gate (True = the gate fired = the rect is out) plus
    gutter_keep as their negated OR, so every rejection is auditable.

        gate_wider_than_tall   wider-than-tall rects are horizontal
                               whitespace bands, never gutters
        gate_bound_border      touches the bound's left/right edge — margins
        gate_no_left_flank     no obstacle touches the left edge with
                               y-overlap: the edge was defined by the bound
                               or an earlier-accepted whitespace rect, i.e. a
                               ragged-edge / indent fragment
        gate_no_right_flank    same for the right edge
        gutter_keep            no gate fired (provisional: stage 5 scoring
                               will redefine this on the gate survivors)

    A rect is maximal, so each vertical edge is defined by whatever stopped
    it; a real column gutter has actual content on both flanks.  Which
    content is recorded too:

        flank_sources_left / flank_sources_right
            comma-joined sorted unique obstacle_source values touching that
            edge ('' when nothing does) — e.g. 'word', 'shape,word', 'image'
    """
    out = df_rects.copy() if df_rects is not None else pd.DataFrame(columns=_RECT_COLS)
    for col in _GATE_COLS:
        out[col] = False
    out["flank_sources_left"] = ""
    out["flank_sources_right"] = ""
    if out.empty:
        out["gutter_keep"] = pd.Series(dtype=bool)
        return out

    bounds = {
        row.page_number: (float(row.x_min), float(row.x_max))
        for row in df_bounds.itertuples(index=False)
    } if df_bounds is not None and not df_bounds.empty else {}

    obs_pages: dict = {}
    if df_obstacles is not None and not df_obstacles.empty:
        obs_bbox_all = df_obstacles[_BBOX_COLS].to_numpy(dtype=np.float64)
        obs_src_all = df_obstacles["obstacle_source"].to_numpy(dtype=object)
        obs_pages = {
            page: idx
            for page, idx in df_obstacles.groupby("page_number", sort=False).indices.items()
        }

    gates = {col: np.zeros(len(out), dtype=bool) for col in _GATE_COLS}
    src_left = np.full(len(out), "", dtype=object)
    src_right = np.full(len(out), "", dtype=object)

    rx0_all = out["gutter_x_left"].to_numpy(dtype=np.float64)
    rx1_all = out["gutter_x_right"].to_numpy(dtype=np.float64)
    ry0_all = out["gutter_y_top"].to_numpy(dtype=np.float64)
    ry1_all = out["gutter_y_bottom"].to_numpy(dtype=np.float64)

    for page_number, page_idx in out.groupby("page_number", sort=False).indices.items():
        rx0, rx1 = rx0_all[page_idx], rx1_all[page_idx]
        ry0, ry1 = ry0_all[page_idx], ry1_all[page_idx]

        gates["gate_wider_than_tall"][page_idx] = (rx1 - rx0) >= (ry1 - ry0)

        bx = bounds.get(page_number)
        if bx is not None:
            gates["gate_bound_border"][page_idx] = (
                (rx0 <= bx[0] + config.bound_edge_eps)
                | (rx1 >= bx[1] - config.bound_edge_eps)
            )

        oidx = obs_pages.get(page_number)
        if oidx is None:
            gates["gate_no_left_flank"][page_idx] = True
            gates["gate_no_right_flank"][page_idx] = True
            continue
        ob = obs_bbox_all[oidx]
        osrc = obs_src_all[oidx]
        ox0, oy0, ox1, oy1 = ob[:, 0], ob[:, 1], ob[:, 2], ob[:, 3]

        # rects x obstacles: y-overlap, then edge touch on each side
        y_ov = (oy0[None, :] < ry1[:, None] - _EPS) & (oy1[None, :] > ry0[:, None] + _EPS)
        touch_left = y_ov & (np.abs(ox1[None, :] - rx0[:, None]) <= config.side_touch_tol)
        touch_right = y_ov & (np.abs(ox0[None, :] - rx1[:, None]) <= config.side_touch_tol)

        gates["gate_no_left_flank"][page_idx] = ~touch_left.any(axis=1)
        gates["gate_no_right_flank"][page_idx] = ~touch_right.any(axis=1)

        for i, gi in enumerate(page_idx):
            if touch_left[i].any():
                src_left[gi] = ",".join(sorted(set(osrc[touch_left[i]])))
            if touch_right[i].any():
                src_right[gi] = ",".join(sorted(set(osrc[touch_right[i]])))

    for col in _GATE_COLS:
        out[col] = gates[col]
    out["flank_sources_left"] = src_left
    out["flank_sources_right"] = src_right
    out["gutter_keep"] = ~np.logical_or.reduce([gates[col] for col in _GATE_COLS])
    return out


# =======================================================================================================================
# Step 4: flank context
# =======================================================================================================================

_FLANK_KPI_COLS = [
    "flank_words_left", "flank_lines_left",
    "flank_marker_frac_left", "flank_numeric_frac_left",
    "flank_words_right", "flank_lines_right",
    "flank_marker_frac_right", "flank_numeric_frac_right",
]


def build_flank_context(
    df_rects: pd.DataFrame,
    df_words: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
    debug: bool = True,
) -> pd.DataFrame:
    """
    Step 4: what actually sits beside each gutter candidate, as visible
    per-side KPIs — the input scoring reads, replacing v2's buried
    broadcast tests.

    A word flanks a candidate when it
      - is a body-zone, horizontal word (same visibility as obstacles),
      - y-overlaps the candidate's span shrunk by flank_y_pad at both ends,
      - lies to that side (word edge within side_touch_tol of the candidate
        edge, or further out), and
      - is not screened: no KEPT candidate (gutter_keep == True) sits fully
        between the word and this candidate at the word's own y.  v2
        screened with every candidate, so a rect that scoring itself was
        about to reject could hide a bullet rail from its neighbor and
        silently flip the neighbor's flank signals; v3's gates are decisive
        and run first, so only gate survivors screen.

    Flank words are grouped into visual lines by y_line_id (page_zones
    step 0), giving line-level KPIs per side:

        flank_words_left/right          number of flank words
        flank_lines_left/right          number of distinct visual lines
        flank_marker_frac_left/right    fraction of flank words that are
                                        list markers (NaN when no words)
        flank_numeric_frac_left/right   fraction that are numeric/currency/
                                        percent tokens (NaN when no words)

    debug=True additionally materializes flank_text_left/right: the flank's
    lines top-down as '1: <text> | 2: <text> | ...' (word join is x-sorted;
    string building is debug-only so the hot path stays numeric).

    Returns df_rects with the KPI (and debug text) columns appended.
    """
    from docslicer._utils.text_utils import list_marker_mask, numeric_value_mask

    out = df_rects.copy() if df_rects is not None else pd.DataFrame()
    for col in _FLANK_KPI_COLS:
        out[col] = 0.0
    if debug:
        out["flank_text_left"] = ""
        out["flank_text_right"] = ""
    if out.empty or df_words is None or df_words.empty:
        return out

    words = df_words
    if "text_orientation" in words.columns:
        orient = words["text_orientation"].astype(str).str.upper().str.strip()
        words = words[~orient.isin(["TTB", "BTT"])]
    if "page_zone" in words.columns:
        words = words[words["page_zone"] == "body"]
    if words.empty:
        return out

    w_marker_all = list_marker_mask(words["text"]).to_numpy()
    w_numeric_all = numeric_value_mask(words["text"]).to_numpy()
    w_text_all = words["text"].astype(str).to_numpy(dtype=object)
    w_bbox_all = words[_BBOX_COLS].to_numpy(dtype=np.float64)
    if "y_line_id" in words.columns:
        # NA-safe: factorize gives each NA its own... no — NA maps to -1; make
        # unlabeled words unique lines so they never fuse into one fake line.
        w_line_all = words["y_line_id"].to_numpy(dtype=object)
        na = pd.isna(w_line_all)
        w_line_all[na] = [f"_na{i}" for i in np.nonzero(na)[0]]
    else:
        w_line_all = np.arange(len(words), dtype=object)
    word_pages = {
        page: idx for page, idx in words.groupby("page_number", sort=False).indices.items()
    }

    kpi = {col: np.zeros(len(out), dtype=np.float64) for col in _FLANK_KPI_COLS}
    kpi["flank_marker_frac_left"][:] = np.nan
    kpi["flank_numeric_frac_left"][:] = np.nan
    kpi["flank_marker_frac_right"][:] = np.nan
    kpi["flank_numeric_frac_right"][:] = np.nan
    txt_left = np.full(len(out), "", dtype=object)
    txt_right = np.full(len(out), "", dtype=object)

    rx0_all = out["gutter_x_left"].to_numpy(dtype=np.float64)
    rx1_all = out["gutter_x_right"].to_numpy(dtype=np.float64)
    ry0_all = out["gutter_y_top"].to_numpy(dtype=np.float64)
    ry1_all = out["gutter_y_bottom"].to_numpy(dtype=np.float64)
    keep_all = (
        out["gutter_keep"].to_numpy(dtype=bool)
        if "gutter_keep" in out.columns else np.ones(len(out), dtype=bool)
    )

    for page_number, page_idx in out.groupby("page_number", sort=False).indices.items():
        widx = word_pages.get(page_number)
        if widx is None:
            continue
        rx0, rx1 = rx0_all[page_idx], rx1_all[page_idx]
        ry0, ry1 = ry0_all[page_idx], ry1_all[page_idx]
        keep = keep_all[page_idx]
        wb = w_bbox_all[widx]
        wx0, wy0, wx1, wy1 = wb[:, 0], wb[:, 1], wb[:, 2], wb[:, 3]
        w_marker = w_marker_all[widx]
        w_numeric = w_numeric_all[widx]

        # padded y-overlap for flank membership; unpadded for side/screening
        fy0, fy1 = ry0 + config.flank_y_pad, ry1 - config.flank_y_pad
        w_yov = (wy0[None, :] < fy1[:, None] - _EPS) & (wy1[None, :] > fy0[:, None] + _EPS)
        g_w_yov = (ry0[:, None] < wy1[None, :] - _EPS) & (ry1[:, None] > wy0[None, :] + _EPS)
        w_left = g_w_yov & (wx1[None, :] <= rx0[:, None] + config.side_touch_tol)
        w_right = g_w_yov & (wx0[None, :] >= rx1[:, None] - config.side_touch_tol)

        # Screening by KEPT candidates only.  blocked[i, j] =
        # any_k(kept[k] & between[i, k] & w_side[k, j]) via boolean matmul.
        g_left_of = ((rx1[None, :] <= rx0[:, None] + _EPS) & keep[None, :]).astype(np.uint8)
        g_right_of = ((rx0[None, :] >= rx1[:, None] - _EPS) & keep[None, :]).astype(np.uint8)
        blocked_left = (g_left_of @ w_left.astype(np.uint8)).astype(bool)
        blocked_right = (g_right_of @ w_right.astype(np.uint8)).astype(bool)

        left_sel = w_yov & w_left & ~blocked_left
        right_sel = w_yov & w_right & ~blocked_right

        for sel, n_col, l_col, m_col, q_col, txt_arr in (
            (left_sel, "flank_words_left", "flank_lines_left",
             "flank_marker_frac_left", "flank_numeric_frac_left", txt_left),
            (right_sel, "flank_words_right", "flank_lines_right",
             "flank_marker_frac_right", "flank_numeric_frac_right", txt_right),
        ):
            n_words = sel.sum(axis=1)
            kpi[n_col][page_idx] = n_words
            kpi[m_col][page_idx] = np.where(
                n_words > 0, (sel & w_marker[None, :]).sum(axis=1) / np.maximum(n_words, 1), np.nan
            )
            kpi[q_col][page_idx] = np.where(
                n_words > 0, (sel & w_numeric[None, :]).sum(axis=1) / np.maximum(n_words, 1), np.nan
            )
            for i, gi in enumerate(page_idx):
                j = np.nonzero(sel[i])[0]
                if j.size == 0:
                    continue
                wj = widx[j]
                lines = pd.unique(w_line_all[wj])
                kpi[l_col][gi] = len(lines)
                if debug:
                    order = np.lexsort((wx0[j], wy0[j]))
                    parts, line_no, seen_line = [], 0, {}
                    for k in order:
                        lid = w_line_all[wj[k]]
                        if lid not in seen_line:
                            line_no += 1
                            seen_line[lid] = line_no
                            parts.append(f"{line_no}: {w_text_all[wj[k]]}")
                        else:
                            parts[seen_line[lid] - 1] += f" {w_text_all[wj[k]]}"
                    txt_arr[gi] = " | ".join(parts)

    for col in _FLANK_KPI_COLS:
        if col.startswith(("flank_words", "flank_lines")):
            out[col] = kpi[col].astype(np.int64)
        else:
            out[col] = kpi[col]
    if debug:
        out["flank_text_left"] = txt_left
        out["flank_text_right"] = txt_right
    return out


# =======================================================================================================================
# End-to-end endpoint (WIP: steps 0-4)
# =======================================================================================================================

def detect_gutters(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
    df_grid_cells: pd.DataFrame | None = None,
    df_bands: pd.DataFrame | None = None,
    config: GutterConfig = GutterConfig(),
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run the v3 pipeline as far as it is built and return the result of the
    last implemented stage — currently the hard-gated whitespace rectangles
    (steps 0-3).  Stages 4-6 will extend this into the scored/merged
    (df_words, df_gutters) return v2 had.

    Step 0 runs internally when the caller hasn't: if df_words lacks a
    page_zone column, assign_page_zones supplies both the zone tags and
    df_bands.  A caller that already tagged zones keeps authority over
    df_bands (None then means full-page bounds).

    debug=False drops rects any hard gate rejected; debug=True returns every
    enumerated rect with its gate columns so rejections are auditable in the
    viewer (gutter_keep marks the survivors).
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(columns=[*_RECT_COLS, *_GATE_COLS,
                                     "flank_sources_left", "flank_sources_right",
                                     "gutter_keep"])

    if "page_zone" not in df_words.columns:
        from .page_zones import assign_page_zones
        df_words, df_bands = assign_page_zones(df_words)

    df_obstacles = collect_obstacles(df_words, df_shapes, df_images, df_grid_cells)
    df_bounds = compute_body_bounds(df_words, df_bands)
    df_rects = enumerate_rects(df_obstacles, df_bounds, config)
    df_rects = apply_hard_gates(df_rects, df_obstacles, df_bounds, config)
    df_rects = build_flank_context(df_rects, df_words, config, debug=debug)
    if not debug:
        df_rects = df_rects[df_rects["gutter_keep"]].reset_index(drop=True)
    return df_rects

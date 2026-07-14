"""
gutter_detector_new.py

Production rebuild of the gutter detector. Pipeline:

    1. compute_page_bounds   header/footer band detection + one enumeration
                             bound (the body rect) per page — pure numpy,
                             no word tagging
    2. collect_obstacles     everything that can stop a whitespace rect,
                             one labeled row per obstacle
    3. enumerate_rects       maximal whitespace rectangles per page (Breuel
                             branch-and-bound in _enumerate_max_rects),
                             clipped to the body bound
    4. apply_hard_gates      named gate columns instead of silent drops;
                             flank edges labeled with the obstacle sources
                             that touch them
    5. build_flank_context   per gutter-side KPIs on the words beside each
                             HARD-GATE SURVIVOR (rejected rects are never
                             scored, so their flanks are never computed);
                             cheap counts first, per-line classification
                             only where the counts aren't already
                             exclusionary
    6. build_gutter_kpis     geometry KPIs (width/height vs. body bound,
                             divider alignment, neighbor/shape signals)
    7. score_gutters         declarative bands over every KPI column into
                             one signed gutter_score + final gutter_keep
    8. merge onto words                                              (TODO)

Step 1 — page bounds.  A running header/footer is separated from the body
by a full-page-width horizontal whitespace band, i.e. a gap in the page
text's y-projection.  So no line ids and no per-word zone tags: words are
grouped into visual lines in one vectorized pass (line_merger's same_line
rule applied to consecutive y-center-sorted words — chained rather than
anchored, which only differs inside dense runs where the count is already
past every threshold below), then each page's line records are walked from
the edge and the scan stops at the first gap of at least min_band_gap.
Disqualifiers (any one kills the band): more candidate lines than
max_header/footer_lines, zone envelope deeper than header/footer_zone_frac
of the page, fewer than min_body_lines lines on the body side.  The output
is only what step 3 consumes: the per-page body bbox (x_min..y_max) —
header/footer words need no tagging because obstacles outside the bound are
dropped during enumeration.  Vertical (TTB/BTT) words are ignored
throughout (a rotated margin caption must not fuse the projection across
half the page height).

Step 2 — collect_obstacles.  One row per obstacle, tagged with its source:

    word          an individual word bbox
    table         all words sharing a table_id, collapsed to one bbox — a
                  table's internal row/column gaps are never gutters
    struct_group  all remaining words sharing a struct_group_id, collapsed —
                  keeps the gap inside a tagged inline run (split date,
                  hyphenated span) from registering as whitespace
    shape         a drawn shape; the page_background shape and fill-only
                  shapes painted in the page's own background color are
                  excluded — they render invisibly and cannot visually
                  delimit a gutter
    image         an image bbox (off-page/degenerate handling is the bound
                  clip's job in step 3)
    grid          all ruling lines sharing a table_grid_id via
                  df_grid_cells, collapsed to one bbox per grid

The source tag is the debugging backbone: hard gates (step 4) name the
obstacle kind that defined a rect's edge, and later flank tests distinguish
"no text beside this gutter" from "an image beside this gutter".

Public API:
    df_rects     = detect_gutters(df_words, df_shapes, df_images,
                                  df_grid_cells, config, debug=False)
    df_bounds    = compute_page_bounds(df_words, config)
    df_obstacles = collect_obstacles(df_words, df_shapes, df_images,
                                     df_grid_cells, df_bounds)
    df_rects     = enumerate_rects(df_obstacles, df_bounds, config)
    df_rects     = apply_hard_gates(df_rects, df_obstacles, df_bounds, config)
    df_rects     = build_flank_context(df_rects, df_words, df_bounds, config, debug)
    df_rects     = build_gutter_kpis(df_rects, df_bounds, df_shapes, config)
    df_rects     = score_gutters(df_rects, config)

df_bounds columns (one row per page: the rect-enumeration area):
    page_number, x_min, y_min, x_max, y_max, page_width, page_height
df_obstacles columns:
    page_number, obstacle_source, x_left, y_top, x_right, y_bottom
df_rects columns (one row per maximal whitespace rectangle; gutter_*
naming marks candidacy, not a verdict — the df_viewer debug tooling keys
on gutter_id + bbox):
    page_number, gutter_id,
    gutter_x_left, gutter_x_right, gutter_y_top, gutter_y_bottom,
    gutter_width, gutter_height, gutter_area

Coordinate convention (matches the rest of the pipeline):
    y increases downward; y_top < y_bottom.
"""

from __future__ import annotations

import heapq
import itertools
import operator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .line_merger import assign_line_id, same_line_pairwise

# =======================================================================================================================
# CONFIG
# =======================================================================================================================


@dataclass(frozen=True)
class GutterConfig:
    # ---- step 1: page bounds (header/footer bands) ----
    header_zone_frac: float = 0.08      # top page fraction a header line must lie within
    footer_zone_frac: float = 0.08      # bottom page fraction a footer line must lie within
    min_band_gap: float = 10.0          # pt - min whitespace between zone and body
    max_header_lines: int = 2           # more candidate lines than this = no header
    max_footer_lines: int = 1           # more candidate lines than this = no footer
    min_body_lines: int = 3             # a band needs a real body on its other side
    # ---- step 3: rect enumeration ----
    min_rect_width: float = 9.2         # pt - minimum whitespace rectangle width
    min_rect_height: float = 30.0       # pt - minimum whitespace rectangle height
    max_rects_per_page: int = 30        # stop enumerating after this many rectangles per page
    max_node_expansions: int = 200_000  # hard safety cap on branch-and-bound nodes per page
    # ---- step 4: hard gates ----
    bound_edge_eps: float = 1.0         # pt - rect within this of the bound's left/right edge = margin
    side_touch_tol: float = 1.0         # pt - obstacle edge within this of a rect edge counts as touching
    # ---- step 5: flank context ----
    flank_y_pad: float = 4.0            # pt - shrink the gutter's y-span at both ends before flank word
                                        # tests (descenders/ascenders of the lines that stopped the
                                        # gutter graze its ends and must not count as flank text)
    # ---- step 6: geometry KPIs ----
    divider_center_tol_frac: float = 0.04   # of body width
    small_gutter_height_frac: float = 0.25  # "short" gutter threshold (of body height)
    small_cluster_min_neighbors: int = 3    # short gutter needs >= this many y-overlapping short peers
    stacked_neighbor_max_gap: float = 50.0  # pt - max vertical gap for the stacked-neighbor signal
    line_touch_tol: float = 5.0             # pt - max distance between a line and a gutter edge to count as touching
    # ---- step 7: scoring policy ----
    keep_threshold: float = 1.0             # gutter_keep = gutter_score >= this


# =======================================================================================================================
# SCORING CRITERIA
# =======================================================================================================================
#
# One signed accumulator per rect — negative pulls toward "not a real column
# gutter", positive toward "is one" — the single-score pattern in layouts.py's
# _LAYOUT_SCORE_BANDS / _apply_score_bands (first-match-wins per feature,
# tuning only ever touches these tables, never the step functions below).
# Categorical signals are a flat {value: points} map.

_COMPARATORS = {
    ">": operator.gt, ">=": operator.ge,
    "<": operator.lt, "<=": operator.le,
    "==": operator.eq,
}

# Rects with an empty or single-line flank are dead on arrival: -50 on any of
# the four bands below can never be recovered (see the invariant assert after
# the tables), which is what lets step 5 skip per-line classification for them.
_EXCLUSION_PTS = -50.0

# NOTE: bands are (comparator, threshold, points); first match (in list order)
# wins, so a value matching no band — including NaN, whose comparisons are all
# False — contributes 0.
_GUTTER_SCORE_BANDS: dict[str, tuple[tuple[str, float, float], ...]] = {
    # Width as a fraction of the body bound: wide => whitespace region between
    # blocks, not a column gutter.
    "width_frac": (
        (">", 0.10, -5.0), (">", 0.07, -3.0), (">", 0.04, -2.0),
    ),
    # Height as a fraction of the body bound: taller reads more like a real
    # column-spanning gutter.
    "height_frac": (
        (">", 2.0 / 3.0, +3.0), (">", 0.5, +2.0), (">", 1.0 / 3.0, +1.0),
    ),
    # Empty flank: no words at all on one side (icon/figure column, or
    # whitespace dangling beside content that already ended).
    "flank_words_left":  (("==", 0.0, _EXCLUSION_PTS),),
    "flank_words_right": (("==", 0.0, _EXCLUSION_PTS),),
    # A flank touching only ONE visual line is a fragment (a lone caption,
    # heading, or isolated word) rather than a real multi-line column run —
    # gated on the RAW line count regardless of table-ness.
    "flank_lines_left":  (("==", 1.0, _EXCLUSION_PTS),),
    "flank_lines_right": (("==", 1.0, _EXCLUSION_PTS),),
    # Many flanking lines is only real column-gutter evidence when those
    # lines are VERIFIED prose — gated on flank_text_lines_* (classify_line()
    # says "text"), not the raw line count: an "undetermined" fragment (too
    # short to score) is not evidence either way and must not earn credit.
    "flank_text_lines_left":  ((">=", 15.0, +5.0), (">=", 8.0, +3.0), (">=", 5.0, +2.0), ("==", 3.0, -1.0), ("==", 2.0, -3.0), ("==", 1.0, -3.0), ("==", 0.0, -5.0)),
    "flank_text_lines_right": ((">=", 15.0, +5.0), (">=", 8.0, +3.0), (">=", 5.0, +2.0), ("==", 3.0, -1.0), ("==", 2.0, -3.0), ("==", 1.0, -3.0), ("==", 0.0, -5.0)),
    # Left flank is (almost) nothing but bullets / list markers.
    "flank_marker_frac_left": ((">=", 0.80, -5.0),),
    # A flank that is nothing but numeric / currency / percent tokens reads
    # as a table value column, not a text column boundary.
    "flank_numeric_frac_left":  ((">", 0.90, -15.0), (">", 0.80, -12.0), (">", 0.70, -10.0), (">", 0.50, -3.0)),
    "flank_numeric_frac_right": ((">", 0.90, -15.0), (">", 0.80, -12.0), (">", 0.70, -10.0), (">", 0.50, -3.0)),
    # Flanking lines that classify_line() calls tabular: this gutter is more
    # likely an undetected table's internal split than a real column gutter.
    "flank_table_frac_left":  ((">=", 0.5, -4.0), (">=", 0.25, -2.0)),
    "flank_table_frac_right": ((">=", 0.5, -4.0), (">=", 0.25, -2.0)),
}
assert all(
    op in _COMPARATORS
    for rules in _GUTTER_SCORE_BANDS.values()
    for op, _, _ in rules
), "_GUTTER_SCORE_BANDS: unknown comparator"

# Categorical / boolean signals: column -> {value: points}.
_GUTTER_CATEGORICAL_BANDS: dict[str, dict] = {
    # x-center on the body's 1/2, 1/3, 2/3 (or 1/4, 3/4) line: a 2-/3-column
    # (or 4-column) split sits exactly there.
    "divider_align": {"half_third": 2.0, "quarter": 1.0},
    # Short gutter sharing its y-span with several other short gutters reads
    # as a table interior, not a column gutter.
    "neighbor_small_cluster": {True: -3.0},
    # Another gutter overlaps this one's x-span within a small vertical gap:
    # a fragmented chain rather than one clean gutter.
    "neighbor_stacked": {True: -3.0},
    # A horizontal line crosses the gutter's full width at both its top and
    # bottom edge: the whitespace is boxed into a table row.
    "shape_line_boxed": {True: -4.0},
}

# Fast-path invariant: one _EXCLUSION_PTS hit must be unrecoverable — the
# best possible score from every other signal, minus the exclusion, must
# still fall below keep_threshold.  Step 5 relies on this to skip per-line
# classification for rects whose flank word/line counts already exclude them.
_MAX_POSITIVE_SCORE = sum(
    max((pts for _, _, pts in rules if pts > 0), default=0.0)
    for col, rules in _GUTTER_SCORE_BANDS.items()
    if not col.startswith(("flank_words", "flank_lines"))
) + sum(
    max((pts for pts in mapping.values() if pts > 0), default=0.0)
    for mapping in _GUTTER_CATEGORICAL_BANDS.values()
)
assert _MAX_POSITIVE_SCORE + _EXCLUSION_PTS < GutterConfig.keep_threshold, (
    "scoring bands: a flank word/line exclusion is no longer decisive — "
    "step 5's fast path (skipping classification for excluded rects) is unsound"
)


_EPS: float = 1e-6  # strict-overlap epsilon: touching edges do not count as overlap

_BBOX_COLS = ["x_left", "y_top", "x_right", "y_bottom"]
_OBSTACLE_COLS = ["page_number", "obstacle_source", *_BBOX_COLS]
_BOUND_COLS = ["page_number", "x_min", "y_min", "x_max", "y_max", "page_width", "page_height"]
_RECT_COLS = [
    "page_number", "gutter_id",
    "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
    "gutter_width", "gutter_height", "gutter_area",
]


# =======================================================================================================================
# Shared word-visibility helpers
# =======================================================================================================================

def _horizontal_words(df_words: pd.DataFrame) -> pd.DataFrame:
    """Drop vertical (TTB/BTT) text — it never participates in detection."""
    if "text_orientation" not in df_words.columns:
        return df_words
    orient = df_words["text_orientation"].astype(str).str.upper().str.strip()
    return df_words[~orient.isin(["TTB", "BTT"])]


def _body_words(df_words: pd.DataFrame, df_bounds: pd.DataFrame | None) -> pd.DataFrame:
    """
    Restrict words to the body: horizontal words whose y-center lies inside
    their page's bound.  Replaces the old per-word page_zone tag — the bound
    already encodes where the header/footer bands end.  Words on pages
    without a bound survive (there detection has no zone opinion).
    """
    words = _horizontal_words(df_words)
    if df_bounds is None or df_bounds.empty or words.empty:
        return words
    bounds = df_bounds.set_index("page_number")
    lo = words["page_number"].map(bounds["y_min"]).to_numpy(dtype=np.float64)
    hi = words["page_number"].map(bounds["y_max"]).to_numpy(dtype=np.float64)
    yc = (words["y_top"].to_numpy(np.float64) + words["y_bottom"].to_numpy(np.float64)) * 0.5
    return words[np.isnan(lo) | ((yc >= lo) & (yc <= hi))]


def _visual_line_breaks(
    page: np.ndarray, y_top: np.ndarray, y_bottom: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized visual-line grouping shared by steps 1 and 5: line_merger's
    same_line rule between consecutive y-center-sorted words, page
    boundaries forcing a break.  Chained rather than anchored (each word is
    compared to its predecessor, not the line's first word), which only
    differs inside dense runs of near-touching lines — where every consumer
    here is already past its counting threshold.

    Returns (order, new_page, new_line): the sort permutation and, aligned
    to it, booleans marking the first row of each page / each visual line.
    """
    order = np.lexsort(((y_top + y_bottom) * 0.5, page))
    pt, yt, yb = page[order], y_top[order], y_bottom[order]
    n = len(order)
    new_page = np.empty(n, dtype=bool)
    new_page[0] = True
    new_page[1:] = pt[1:] != pt[:-1]
    new_line = np.empty(n, dtype=bool)
    new_line[0] = True
    new_line[1:] = new_page[1:] | ~same_line_pairwise(yt[:-1], yb[:-1], yt[1:], yb[1:])
    return order, new_page, new_line


def _visual_line_ids(df_words: pd.DataFrame) -> np.ndarray:
    """
    Visual-line ids (document-global) aligned to df_words' rows, via
    line_merger.assign_line_id on a (page, y_top, x_left)-sorted view — the
    exact grouping the v2 pipeline's y_line_id had.  Deliberately ANCHORED,
    not the chained _visual_line_breaks rule the band scan uses: chained
    grouping drifts across superscript/footnote-marker chains and fuses
    whole runs of dense lines, which deflates the flank line counts that
    scoring reads at fine granularity (the band scan only ever counts up to
    its small thresholds, where the two rules agree).  Only geometry columns
    are passed so table_row_id / block_type never leak into the grouping.
    """
    s = df_words[["page_number", "x_left", "y_top", "y_bottom"]].sort_values(
        ["page_number", "y_top", "x_left"], kind="mergesort"
    )
    s = assign_line_id(s)
    return s["line_id"].reindex(df_words.index).to_numpy(dtype=np.int64)


# =======================================================================================================================
# Step 1: page bounds (header/footer band detection)
# =======================================================================================================================

def _scan_band_edge(
    tops: np.ndarray,
    bottoms: np.ndarray,
    zone_limit: float,
    max_lines: int,
    min_body_lines: int,
    min_band_gap: float,
) -> float | None:
    """
    Walk one page's visual-line records top-down and return the body-side
    edge of the first qualifying whitespace band, or None.

    tops/bottoms are the page's line records sorted by top edge (a footer
    scan passes them pre-flipped).  The scan grows the zone's y-envelope and
    stops at the FIRST gap of at least min_band_gap — never a fixed
    edge-fraction slice, which would pull early body lines into the zone on
    pages whose body starts high; the edge fraction only caps how deep the
    zone may reach.  The first qualifying gap decides: if the body side then
    has fewer than min_body_lines lines, there is no band at all (a deeper
    gap would swallow body).
    """
    n = len(tops)
    env_bottom = -np.inf
    for k in range(min(max_lines, n - 1)):
        env_bottom = max(env_bottom, float(bottoms[k]))
        if env_bottom > zone_limit:
            return None
        if float(tops[k + 1]) - env_bottom < min_band_gap:
            continue
        if n - (k + 1) < min_body_lines:
            return None
        return float(tops[k + 1])
    return None


def compute_page_bounds(
    df_words: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 1: one rect-enumeration bound per page — the full page, shrunk
    vertically to the body region where a header/footer band is detected.

    The band's INNER edge bounds the body (the body's first/last interval
    edge), so the band whitespace itself is excluded and rect/gutter heights
    come out body-relative.  Page width/height come from page_width /
    page_height columns when present, else fall back to the page's word
    extents (which understates the true size, making edge-zone tests
    conservative — a band can only be missed, never invented, that way).
    Pages without horizontal words get no bound (and therefore no rects).

    Fully vectorized: visual lines for ALL pages come from one pass of
    line_merger's same_line rule over consecutive y-center-sorted words
    (page boundaries force a break), then per-line tops/bottoms fall out of
    two reduceats.  Only the band scan itself loops, over pages x at most
    max_header/footer_lines lines.

    Returns a dataframe with columns
        page_number, x_min, y_min, x_max, y_max, page_width, page_height
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(columns=_BOUND_COLS)
    missing = {"page_number", *_BBOX_COLS} - set(df_words.columns)
    if missing:
        raise ValueError(f"df_words missing required columns: {sorted(missing)}")

    words = _horizontal_words(df_words)
    if words.empty:
        return pd.DataFrame(columns=_BOUND_COLS)

    page = words["page_number"].to_numpy()
    y_top = words["y_top"].to_numpy(dtype=np.float64)
    y_bottom = words["y_bottom"].to_numpy(dtype=np.float64)
    x_right = words["x_right"].to_numpy(dtype=np.float64)

    # Visual lines for all pages in one vectorized pass; per-line tops /
    # bottoms then fall out of two reduceats.
    order, new_page, new_line = _visual_line_breaks(page, y_top, y_bottom)
    page, y_top, y_bottom, x_right = page[order], y_top[order], y_bottom[order], x_right[order]

    page_start = np.flatnonzero(new_page)          # first row of each page
    page_ids = page[page_start]

    has_dims = {"page_width", "page_height"}.issubset(words.columns)
    if has_dims:
        page_w = words["page_width"].to_numpy(dtype=np.float64)[order][page_start]
        page_h = words["page_height"].to_numpy(dtype=np.float64)[order][page_start]
    else:
        page_w = np.maximum.reduceat(x_right, page_start)
        page_h = np.maximum.reduceat(y_bottom, page_start)

    line_first = np.flatnonzero(new_line)          # first row of each line
    line_top = np.minimum.reduceat(y_top, line_first)
    line_bottom = np.maximum.reduceat(y_bottom, line_first)
    line_pstart = np.r_[np.flatnonzero(new_page[line_first]), len(line_first)]

    y_min = np.zeros(len(page_ids), dtype=np.float64)
    y_max = page_h.astype(np.float64).copy()
    for p in range(len(page_ids)):
        ph = float(page_h[p])
        if ph <= 0:
            continue
        s, e = line_pstart[p], line_pstart[p + 1]
        tops, bottoms = line_top[s:e], line_bottom[s:e]
        top_order = np.argsort(tops, kind="stable")

        edge = _scan_band_edge(
            tops[top_order], bottoms[top_order],
            zone_limit=config.header_zone_frac * ph,
            max_lines=config.max_header_lines,
            min_body_lines=config.min_body_lines,
            min_band_gap=config.min_band_gap,
        )
        if edge is not None:
            y_min[p] = edge

        # Footer: the same scan in flipped coordinates walks bottom-up.
        f_tops, f_bottoms = ph - bottoms, ph - tops
        f_order = np.argsort(f_tops, kind="stable")
        edge = _scan_band_edge(
            f_tops[f_order], f_bottoms[f_order],
            zone_limit=config.footer_zone_frac * ph,
            max_lines=config.max_footer_lines,
            min_body_lines=config.min_body_lines,
            min_band_gap=config.min_band_gap,
        )
        if edge is not None:
            y_max[p] = ph - edge

    out = pd.DataFrame({
        "page_number": page_ids,
        "x_min": 0.0, "y_min": y_min,
        "x_max": page_w.astype(np.float64), "y_max": y_max,
        "page_width": page_w.astype(np.float64),
        "page_height": page_h.astype(np.float64),
    })
    out = out[(out["x_max"] > out["x_min"]) & (out["y_max"] > out["y_min"])]
    return out[_BOUND_COLS].reset_index(drop=True)


# =======================================================================================================================
# Step 2: obstacle collection
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


def _word_obstacles(df_words: pd.DataFrame, df_bounds: pd.DataFrame | None) -> list:
    """
    Word-derived obstacle parts: table / struct-group collapse first (native
    PDF only — OCR words carry neither column and degrade to one obstacle
    per word), leftover words individually.

    Vertical (TTB/BTT) text never participates, and when df_bounds is given
    only words whose y-center lies inside their page's body bound do —
    header/footer words are excluded from the reading area BEFORE the
    table / struct-group collapse, so a group straddling a band edge cannot
    drag band content into a body obstacle.
    """
    words = _body_words(df_words, df_bounds)
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
    df_bounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Step 2: gather every gutter-stopping obstacle into one labeled frame.

    See the module docstring for the source taxonomy and the visibility
    rules per source.  Degenerate boxes (NaN / inverted edges) are dropped;
    zero-thickness boxes survive (rule lines).  Off-page boxes survive too —
    step 3 drops obstacles that miss the body bound and clips the rest.

    Returns a dataframe with columns
        page_number, obstacle_source, x_left, y_top, x_right, y_bottom
    sorted by (page_number, obstacle_source) with a fresh 0..n-1 index.
    """
    parts: list = []

    if df_words is not None and not df_words.empty:
        missing = {"page_number", *_BBOX_COLS} - set(df_words.columns)
        if missing:
            raise ValueError(f"df_words missing required columns: {sorted(missing)}")
        parts += _word_obstacles(df_words, df_bounds)

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
# Step 3: rect enumeration (Breuel branch-and-bound)
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


def enumerate_rects(
    df_obstacles: pd.DataFrame,
    df_bounds: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 3: maximal whitespace rectangles per page, within the body bound.

    Obstacles that do not intersect their page's bound are DROPPED (header/
    footer words already never reach the obstacle frame, but off-page images
    and band-zone shapes land here); partially-inside obstacles are clipped
    so splits never escape the bound.

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
# Step 4: hard gates
# =======================================================================================================================

_GATE_COLS = ["gate_wider_than_tall", "gate_bound_border", "gate_no_left_flank", "gate_no_right_flank"]


def apply_hard_gates(
    df_rects: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    df_bounds: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 4: flag whitespace rectangles that cannot be column gutters —
    recorded instead of silently dropped, one boolean column per gate
    (True = the gate fired = the rect is out) plus gutter_keep as their
    negated OR, so every rejection is auditable.

        gate_wider_than_tall   wider-than-tall rects are horizontal
                               whitespace bands, never gutters
        gate_bound_border      touches the bound's left/right edge — margins
        gate_no_left_flank     no obstacle touches the left edge with
                               y-overlap: the edge was defined by the bound
                               or an earlier-accepted whitespace rect, i.e. a
                               ragged-edge / indent fragment
        gate_no_right_flank    same for the right edge
        gutter_keep            no gate fired (provisional: scoring will
                               redefine this on the gate survivors)

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
# Step 5: flank context
# =======================================================================================================================

_FLANK_KPI_COLS = [
    "flank_words_left", "flank_lines_left",
    "flank_marker_frac_left", "flank_numeric_frac_left",
    "flank_table_score_left", "flank_table_frac_left",
    "flank_text_lines_left",
    "flank_words_right", "flank_lines_right",
    "flank_marker_frac_right", "flank_numeric_frac_right",
    "flank_table_score_right", "flank_table_frac_right",
    "flank_text_lines_right",
]


def build_flank_context(
    df_rects: pd.DataFrame,
    df_words: pd.DataFrame,
    df_bounds: pd.DataFrame | None = None,
    config: GutterConfig = GutterConfig(),
    debug: bool = False,
) -> pd.DataFrame:
    """
    Step 5: what actually sits beside each gutter candidate, as visible
    per-side KPIs — the input scoring reads.

    HARD-GATE SURVIVORS ONLY: a rect the gates rejected can never become a
    gutter no matter what flanks it, so its KPI columns stay at their
    defaults (0 counts / NaN fracs).  On top of that, per-line
    classification — the expensive part — runs only for rects whose flank
    word/line counts aren't already exclusionary: a side with 0 words or 1
    line scores _EXCLUSION_PTS, which the invariant assert above the band
    tables guarantees is decisive, so table/text KPIs stay NaN there too.

    A word flanks a candidate when it
      - is a body, horizontal word (same visibility as obstacles: inside
        the page's bound, not TTB/BTT),
      - y-overlaps the candidate's span shrunk by flank_y_pad at both ends,
      - lies to that side (word edge within side_touch_tol of the candidate
        edge, or further out), and
      - is not screened: no other surviving candidate sits fully between
        the word and this candidate at the word's own y (only survivors
        screen — a rect the gates rejected must not hide a bullet rail from
        its neighbor and silently flip the neighbor's flank signals).

    Flank words are grouped into visual lines (same _visual_line_breaks
    grouping step 1 uses), giving line-level KPIs per side:

        flank_words_left/right          number of flank words
        flank_lines_left/right          number of distinct visual lines
        flank_marker_frac_left/right    fraction of flank words that are
                                        list markers (NaN when no words)
        flank_numeric_frac_left/right   fraction that are numeric/currency/
                                        percent tokens (NaN when no words)
        flank_table_score_left/right    mean classify_line() score across the
                                        flank's line FRAGMENTS (NaN when no
                                        lines or fast-path skipped); a
                                        fragment is this side's words on one
                                        visual line, not the whole original
                                        line — the candidate gutter is
                                        exactly a claim that one visual line
                                        is really two independent runs, so
                                        re-merging them here would erase the
                                        signal.  Identical fragments recur
                                        across candidates (same paragraph
                                        line flanking several gutter splits)
                                        and are classified once via a cache
                                        keyed on the fragment's exact word
                                        set.
        flank_table_frac_left/right     fraction of those line fragments
                                        classify_line() labels "table"
        flank_text_lines_left/right     count of flanking line fragments
                                        classify_line() labels "text" —
                                        strictly positive evidence, so an
                                        "undetermined" fragment does NOT
                                        count as text any more than a
                                        "table" one does

    debug=True additionally materializes flank_text_left/right: the flank's
    lines top-down as '1: <text> | 2: <text> | ...' (built for every
    survivor, including fast-path exclusions — the string join is cheap;
    only classification is gated).

    Returns df_rects with the KPI (and debug text) columns appended.
    """
    from docslicer._utils.text_utils import list_marker_mask, numeric_value_mask
    from docslicer.pdf._utils.line_classification import classify_line, line_gap_stats

    out = df_rects.copy() if df_rects is not None else pd.DataFrame()
    for col in _FLANK_KPI_COLS:
        out[col] = 0.0
    if debug:
        out["flank_text_left"] = ""
        out["flank_text_right"] = ""
    if out.empty or df_words is None or df_words.empty:
        return out

    kept_all = (
        out["gutter_keep"].to_numpy(dtype=bool)
        if "gutter_keep" in out.columns else np.ones(len(out), dtype=bool)
    )

    words = _body_words(df_words, df_bounds)
    if words.empty or not kept_all.any():
        return out

    w_marker_all = list_marker_mask(words["text"]).to_numpy()
    w_numeric_all = numeric_value_mask(words["text"]).to_numpy()
    w_text_all = words["text"].astype(str).to_numpy(dtype=object)
    w_fontsize_all = (
        words["font_size"].to_numpy(dtype=np.float64)
        if "font_size" in words.columns else np.full(len(words), np.nan)
    )
    w_bbox_all = words[_BBOX_COLS].to_numpy(dtype=np.float64)
    w_line_all = _visual_line_ids(words)
    word_pages = {
        page: idx for page, idx in words.groupby("page_number", sort=False).indices.items()
    }

    kpi = {col: np.zeros(len(out), dtype=np.float64) for col in _FLANK_KPI_COLS}
    for col in _FLANK_KPI_COLS:
        if col.startswith(("flank_marker", "flank_numeric", "flank_table")):
            kpi[col][:] = np.nan
    txt_left = np.full(len(out), "", dtype=object)
    txt_right = np.full(len(out), "", dtype=object)
    # Fragment-level table classification cache: keyed on the exact sorted
    # tuple of global word indices in one flank line fragment — the same
    # fragment recurs across many candidates sharing a paragraph line and is
    # classified only once.
    frag_table_cache: dict = {}

    rx0_all = out["gutter_x_left"].to_numpy(dtype=np.float64)
    rx1_all = out["gutter_x_right"].to_numpy(dtype=np.float64)
    ry0_all = out["gutter_y_top"].to_numpy(dtype=np.float64)
    ry1_all = out["gutter_y_bottom"].to_numpy(dtype=np.float64)

    for page_number, page_rows in out.groupby("page_number", sort=False).indices.items():
        page_idx = page_rows[kept_all[page_rows]]  # survivors only
        if page_idx.size == 0:
            continue
        widx = word_pages.get(page_number)
        if widx is None:
            continue
        rx0, rx1 = rx0_all[page_idx], rx1_all[page_idx]
        ry0, ry1 = ry0_all[page_idx], ry1_all[page_idx]
        wb = w_bbox_all[widx]
        wx0, wy0, wx1, wy1 = wb[:, 0], wb[:, 1], wb[:, 2], wb[:, 3]
        w_marker = w_marker_all[widx]
        w_numeric = w_numeric_all[widx]
        w_fontsize = w_fontsize_all[widx]

        # padded y-overlap for flank membership; unpadded for side/screening
        fy0, fy1 = ry0 + config.flank_y_pad, ry1 - config.flank_y_pad
        w_yov = (wy0[None, :] < fy1[:, None] - _EPS) & (wy1[None, :] > fy0[:, None] + _EPS)
        g_w_yov = (ry0[:, None] < wy1[None, :] - _EPS) & (ry1[:, None] > wy0[None, :] + _EPS)
        w_left = g_w_yov & (wx1[None, :] <= rx0[:, None] + config.side_touch_tol)
        w_right = g_w_yov & (wx0[None, :] >= rx1[:, None] - config.side_touch_tol)

        # Screening by fellow survivors: blocked[i, j] =
        # any_k(between[i, k] & w_side[k, j]) via boolean matmul (page_idx
        # holds survivors only, so no keep mask is needed).
        g_left_of = (rx1[None, :] <= rx0[:, None] + _EPS).astype(np.uint8)
        g_right_of = (rx0[None, :] >= rx1[:, None] - _EPS).astype(np.uint8)
        blocked_left = (g_left_of @ w_left.astype(np.uint8)).astype(bool)
        blocked_right = (g_right_of @ w_right.astype(np.uint8)).astype(bool)

        left_sel = w_yov & w_left & ~blocked_left
        right_sel = w_yov & w_right & ~blocked_right

        # Pass 1 — cheap KPIs for every survivor: word counts, marker /
        # numeric fractions (vectorized), then per-rect line grouping for
        # the line counts and debug text.  Fragment groupings are kept for
        # pass 2.
        pending: list = []  # (gi, ts_col, tf_col, nt_col, j, line_pos)
        for sel, n_col, l_col, m_col, q_col, ts_col, tf_col, nt_col, txt_arr in (
            (left_sel, "flank_words_left", "flank_lines_left",
             "flank_marker_frac_left", "flank_numeric_frac_left",
             "flank_table_score_left", "flank_table_frac_left",
             "flank_text_lines_left", txt_left),
            (right_sel, "flank_words_right", "flank_lines_right",
             "flank_marker_frac_right", "flank_numeric_frac_right",
             "flank_table_score_right", "flank_table_frac_right",
             "flank_text_lines_right", txt_right),
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
                lids = w_line_all[widx[j]]

                # Group this side's words by line fragment, top-down.
                order = np.lexsort((wx0[j], wy0[j]))
                parts, line_no, seen_line, line_pos = [], 0, {}, {}
                for k in order:
                    lid = lids[k]
                    if lid not in seen_line:
                        line_no += 1
                        seen_line[lid] = line_no
                        line_pos[lid] = []
                        if debug:
                            parts.append(f"{line_no}: {w_text_all[widx[j[k]]]}")
                    else:
                        if debug:
                            parts[seen_line[lid] - 1] += f" {w_text_all[widx[j[k]]]}"
                    line_pos[lid].append(k)
                kpi[l_col][gi] = len(line_pos)
                if debug:
                    txt_arr[gi] = " | ".join(parts)
                pending.append((gi, ts_col, tf_col, nt_col, j, line_pos))

        # Pass 2 — fragment classification, fast path: skip rects whose
        # word/line counts already exclude them on either side (see the
        # _EXCLUSION_PTS invariant assert; their table/text KPIs stay NaN).
        excluded = {
            gi for gi in page_idx
            if kpi["flank_words_left"][gi] == 0 or kpi["flank_lines_left"][gi] <= 1
            or kpi["flank_words_right"][gi] == 0 or kpi["flank_lines_right"][gi] <= 1
        }
        for gi, ts_col, tf_col, nt_col, j, line_pos in pending:
            if gi in excluded:
                continue
            scores, table_flags, text_flags = [], [], []
            for positions in line_pos.values():
                pos = np.asarray(positions)
                local_idx = j[pos]
                global_idx = widx[local_idx]
                cache_key = tuple(sorted(global_idx.tolist()))
                cached = frag_table_cache.get(cache_key)
                if cached is None:
                    ord2 = np.argsort(wx0[local_idx])
                    gstats = line_gap_stats(
                        wx0[local_idx][ord2], wx1[local_idx][ord2], w_fontsize[local_idx][ord2]
                    )
                    texts = w_text_all[global_idx][ord2].tolist()
                    cached = classify_line(texts, gstats)
                    frag_table_cache[cache_key] = cached
                scores.append(cached[1])
                table_flags.append(cached[0] == "table")
                text_flags.append(cached[0] == "text")
            if scores:
                kpi[ts_col][gi] = float(np.mean(scores))
                kpi[tf_col][gi] = float(np.mean(table_flags))
                kpi[nt_col][gi] = float(sum(text_flags))

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
# Step 6: gutter-scoring KPIs (geometry only)
# =======================================================================================================================

def build_gutter_kpis(
    df_rects: pd.DataFrame,
    df_bounds: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 6: geometry-only KPIs feeding the scorer — everything score_gutters
    needs that isn't already on df_rects from steps 1-5.  Kept as its own
    stage so every signal feeding the scorer is its own inspectable column,
    not just baked into a sum.

        width_frac / height_frac    gutter_width / gutter_height as a
                                    fraction of the page's BODY bound —
                                    body, not raw page height, so a header/
                                    footer band never inflates how "tall" a
                                    gutter looks.
        divider_align               "half_third" | "quarter" | <NA> — the
                                    gutter's x-center sits on the body's
                                    1/2, 1/3, 2/3 (half_third) or 1/4, 3/4
                                    (quarter) division line, within
                                    divider_center_tol_frac of body width.
        neighbor_small_cluster      bool — this gutter is short (< a
                                    fraction of body height) and shares its
                                    y-span with >= N other short surviving
                                    gutters on the same page (table
                                    interiors shed many short whitespace
                                    rects side by side).
        neighbor_stacked            bool — another surviving gutter
                                    x-overlaps this one within a small
                                    vertical gap (a fragmented gutter chain).
        shape_line_boxed            bool — a horizontal line (df_shapes)
                                    crosses this gutter's full x-span at
                                    BOTH its top and bottom edge (whitespace
                                    boxed into a table row).

    The neighbor/shape signals are restricted to hard-gate survivors, both
    as subject and as peer: a rejected rect can never become a gutter, and
    it must not affect a surviving neighbor's cluster/stacked signal either.
    The plain geometry columns (width/height/divider) are cheap and are
    filled for every rect so rejections stay auditable in debug output.

    Returns df_rects with these six columns appended; NaN/False/<NA> when
    df_bounds has no entry for a rect's page, or df_shapes is omitted.
    """
    out = df_rects.copy() if df_rects is not None else pd.DataFrame()
    out["width_frac"] = np.nan
    out["height_frac"] = np.nan
    out["divider_align"] = pd.array([pd.NA] * len(out), dtype="string")
    out["neighbor_small_cluster"] = False
    out["neighbor_stacked"] = False
    out["shape_line_boxed"] = False
    if out.empty:
        return out

    gate_keep = (
        out["gutter_keep"].to_numpy(dtype=bool)
        if "gutter_keep" in out.columns else np.ones(len(out), dtype=bool)
    )

    bounds = {
        row.page_number: (float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max))
        for row in df_bounds.itertuples(index=False)
    } if df_bounds is not None and not df_bounds.empty else {}

    lines_all = None
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
        lines_all = lines

    gx0_all = out["gutter_x_left"].to_numpy(dtype=np.float64)
    gx1_all = out["gutter_x_right"].to_numpy(dtype=np.float64)
    gy0_all = out["gutter_y_top"].to_numpy(dtype=np.float64)
    gy1_all = out["gutter_y_bottom"].to_numpy(dtype=np.float64)
    gh_all = gy1_all - gy0_all

    width_frac = np.full(len(out), np.nan)
    height_frac = np.full(len(out), np.nan)
    divider_align = np.full(len(out), None, dtype=object)
    small_cluster = np.zeros(len(out), dtype=bool)
    stacked = np.zeros(len(out), dtype=bool)
    line_boxed = np.zeros(len(out), dtype=bool)

    for page_number, page_idx in out.groupby("page_number", sort=False).indices.items():
        bx = bounds.get(page_number)
        if bx is None:
            continue
        x_min, y_min, x_max, y_max = bx
        body_w = x_max - x_min
        body_h = y_max - y_min
        if body_w <= 0 or body_h <= 0:
            continue

        gx0, gx1 = gx0_all[page_idx], gx1_all[page_idx]
        gy0, gy1 = gy0_all[page_idx], gy1_all[page_idx]
        gh = gh_all[page_idx]
        keep = gate_keep[page_idx]

        width_frac[page_idx] = (gx1 - gx0) / body_w
        height_frac[page_idx] = gh / body_h

        cx = (gx0 + gx1) * 0.5
        tol = config.divider_center_tol_frac * body_w
        halves_thirds = x_min + np.array([body_w / 2.0, body_w / 3.0, 2.0 * body_w / 3.0])
        quarters = x_min + np.array([body_w / 4.0, 3.0 * body_w / 4.0])
        on_ht = (np.abs(cx[:, None] - halves_thirds[None, :]) <= tol).any(axis=1)
        on_q = (np.abs(cx[:, None] - quarters[None, :]) <= tol).any(axis=1)
        divider_align[page_idx] = np.where(on_ht, "half_third", np.where(on_q, "quarter", None))

        if keep.sum() == 0:
            continue
        kx0, kx1 = gx0[keep], gx1[keep]
        ky0, ky1 = gy0[keep], gy1[keep]
        kh = gh[keep]
        n = len(kx0)

        yov = (ky0[:, None] < ky1[None, :] - _EPS) & (ky1[:, None] > ky0[None, :] + _EPS)
        xov = (kx0[:, None] < kx1[None, :] - _EPS) & (kx1[:, None] > kx0[None, :] + _EPS)
        np.fill_diagonal(yov, False)
        np.fill_diagonal(xov, False)

        small = kh < config.small_gutter_height_frac * body_h
        n_small_peers = (yov & small[None, :]).sum(axis=1)
        k_small_cluster = small & (n_small_peers >= config.small_cluster_min_neighbors)

        gap = np.maximum(ky0[None, :] - ky1[:, None], ky0[:, None] - ky1[None, :])
        k_stacked = (xov & (gap < config.stacked_neighbor_max_gap)).any(axis=1)

        k_line_boxed = np.zeros(n, dtype=bool)
        if lines_all is not None:
            page_lines = lines_all[lines_all["page_number"] == page_number]
            if not page_lines.empty:
                lx0 = page_lines["x_left"].to_numpy(dtype=np.float64)
                lx1 = page_lines["x_right"].to_numpy(dtype=np.float64)
                ly0 = page_lines["y_top"].to_numpy(dtype=np.float64)
                ly1 = page_lines["y_bottom"].to_numpy(dtype=np.float64)
                crosses = (
                    (lx0[None, :] <= kx0[:, None] + config.line_touch_tol)
                    & (lx1[None, :] >= kx1[:, None] - config.line_touch_tol)
                )
                at_top = (
                    (ly0[None, :] <= ky0[:, None] + config.line_touch_tol)
                    & (ly1[None, :] >= ky0[:, None] - config.line_touch_tol)
                )
                at_bottom = (
                    (ly0[None, :] <= ky1[:, None] + config.line_touch_tol)
                    & (ly1[None, :] >= ky1[:, None] - config.line_touch_tol)
                )
                k_line_boxed = (crosses & at_top).any(axis=1) & (crosses & at_bottom).any(axis=1)

        keep_idx = page_idx[keep]
        small_cluster[keep_idx] = k_small_cluster
        stacked[keep_idx] = k_stacked
        line_boxed[keep_idx] = k_line_boxed

    out["width_frac"] = width_frac
    out["height_frac"] = height_frac
    out["divider_align"] = pd.array(divider_align, dtype="string")
    out["neighbor_small_cluster"] = small_cluster
    out["neighbor_stacked"] = stacked
    out["shape_line_boxed"] = line_boxed
    return out


# =======================================================================================================================
# Step 7: scoring
# =======================================================================================================================

def _apply_numeric_bands(values: np.ndarray, rules: tuple) -> np.ndarray:
    """
    First-match-wins band lookup over `values` (mirrors layouts.py's
    _apply_score_bands / np.select semantics): each element takes the points
    of the first rule it satisfies, and an element matching none — including
    NaN, whose comparisons are all False — scores 0.
    """
    score = np.zeros(len(values), dtype=float)
    assigned = np.zeros(len(values), dtype=bool)
    for op, thr, pts in rules:
        hit = _COMPARATORS[op](values, thr) & ~assigned
        score[hit] = pts
        assigned |= hit
    return score


def score_gutters(
    df_rects: pd.DataFrame,
    config: GutterConfig = GutterConfig(),
) -> pd.DataFrame:
    """
    Step 7: score every hard-gate survivor into gutter_score and fold it
    into the final gutter_keep verdict.

    Pure over the KPI columns steps 5-6 already computed — no geometry of
    its own.  gate_keep (the hard-gate survivor flag apply_hard_gates left
    on gutter_keep) still gates the final verdict: a rect the gates already
    rejected never becomes a gutter regardless of score.

    Adds:
        score_<col>    float — one column per _GUTTER_SCORE_BANDS /
                       _GUTTER_CATEGORICAL_BANDS entry, this KPI's own signed
                       contribution (0.0 when no band matched) — so a
                       surprising gutter_score is auditable band-by-band in
                       the debug CSV.
        gutter_score   float — signed sum of every score_<col>
        gutter_keep    bool  — overwritten: gate_keep AND gutter_score >=
                       config.keep_threshold
    """
    out = df_rects.copy() if df_rects is not None else pd.DataFrame()
    score_cols = [f"score_{col}" for col in (*_GUTTER_SCORE_BANDS, *_GUTTER_CATEGORICAL_BANDS)]
    for col in score_cols:
        out[col] = 0.0
    out["gutter_score"] = np.nan
    if out.empty:
        out["gutter_keep"] = pd.Series(dtype=bool)
        return out

    gate_keep = (
        out["gutter_keep"].to_numpy(dtype=bool)
        if "gutter_keep" in out.columns else np.ones(len(out), dtype=bool)
    )

    total = np.zeros(len(out), dtype=float)
    for col, rules in _GUTTER_SCORE_BANDS.items():
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        contrib = _apply_numeric_bands(v, rules)
        out[f"score_{col}"] = contrib
        total += contrib

    for col, mapping in _GUTTER_CATEGORICAL_BANDS.items():
        if col not in out.columns:
            continue
        col_vals = out[col]
        contrib = np.zeros(len(out), dtype=float)
        for value, pts in mapping.items():
            val_mask = (col_vals == value).fillna(False).to_numpy(dtype=bool)
            contrib[val_mask] += pts
        out[f"score_{col}"] = contrib
        total += contrib

    out["gutter_score"] = total
    out["gutter_keep"] = gate_keep & (total >= config.keep_threshold)
    return out


# =======================================================================================================================
# End-to-end endpoint (steps 1-7; step 8 merge-onto-words TODO)
# =======================================================================================================================

def detect_gutters(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_images: pd.DataFrame | None = None,
    df_grid_cells: pd.DataFrame | None = None,
    config: GutterConfig = GutterConfig(),
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run the rebuilt pipeline and return the scored whitespace rectangles
    (steps 1-7).  Step 8 (merge onto words) will extend this into the
    (df_words, df_gutters) return v2 had.

    debug=False drops rects any hard gate rejected OR that scored below
    config.keep_threshold; debug=True returns every enumerated rect with
    its gate + KPI + score columns so rejections are auditable in the
    viewer (gutter_keep marks the final survivors).
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(columns=[*_RECT_COLS, *_GATE_COLS,
                                     "flank_sources_left", "flank_sources_right",
                                     "gutter_score", "gutter_keep"])

    df_bounds = compute_page_bounds(df_words, config)
    df_obstacles = collect_obstacles(df_words, df_shapes, df_images, df_grid_cells, df_bounds)
    df_rects = enumerate_rects(df_obstacles, df_bounds, config)
    df_rects = apply_hard_gates(df_rects, df_obstacles, df_bounds, config)
    df_rects = build_flank_context(df_rects, df_words, df_bounds, config, debug=debug)
    df_rects = build_gutter_kpis(df_rects, df_bounds, df_shapes, config)
    df_rects = score_gutters(df_rects, config)
    if not debug:
        df_rects = df_rects[df_rects["gutter_keep"]].reset_index(drop=True)
    return df_rects

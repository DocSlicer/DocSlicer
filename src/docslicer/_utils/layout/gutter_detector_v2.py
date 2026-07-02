"""
gutter_detector_v2.py

A ground-up rewrite of gutter detection based on a *whitespace-cover* model
instead of the bottom-up sliding-window / candidate-tracking machinery in
``gutter_detector.py``.

Motivation
----------
The legacy detector stitches per-row gaps into vertical chains and then applies
~15 interacting binary toggles (kill / clip / density / edge-eject / content
rules) to decide which chains survive.  Each toggle is a *local, binary* decision
so tuning one example doc regresses another.

This module replaces that with two clean stages:

    Stage 1  (geometry)  find_maximal_whitespace_rectangles()
        Treat every word bounding box as an obstacle and find the globally
        maximal empty rectangles on the page via Breuel's branch-and-bound
        (Baird's maximal-empty-rectangle algorithm).  A column gutter is simply
        a *tall, narrow* maximal whitespace rectangle — found directly in 2D,
        with no window ids, overlap epsilons, or y-gap kills.

    Stage 2  (scoring)   score_and_select_gutters()
        Score each candidate rectangle with a handful of *continuous* features
        (both-side flank coverage, height ratio, width, ruled-line crossing,
        content class of flanking words) and keep the ones above a single
        threshold.  Scoring is monotone: nudging a weight moves every document
        in the same direction, so there is no whack-a-mole.  The full feature
        table is returned for inspection / weight fitting.

Public API
----------
    df_gutters, df_scores = detect_gutters_v2(df_words, df_shapes)
        df_gutters : one row per accepted gutter, columns match the legacy
                     df_gutters schema (page_number, gutter_id, gutter_x_left,
                     gutter_x_right, gutter_y_top, gutter_y_bottom,
                     gutter_width, gutter_height) so it is a drop-in.
        df_scores  : every candidate rectangle with all features + the final
                     score + accept flag (for debugging / tuning).

    df_words, df_candidates, df_gutters = detect_and_annotate_gutters_v2(
        df_words, df_shapes)
        Signature-compatible with the legacy detect_and_annotate_gutters().
        Reuses merge_gutters_onto_words() from the legacy module.

Coordinate convention (matches the rest of the pipeline)
    y increases downward:  y_top is the visual top, y_bottom the visual bottom,
    with y_top < y_bottom.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..text_utils import _CURRENCY_SYM_CLASS, is_list_marker

# =======================================================================================================================
# CONFIG   (few, meaningful knobs — all continuous or hard geometric limits)
# =======================================================================================================================

# --- Stage 1: whitespace-rectangle search --------------------------------------------------------------
_MIN_GUTTER_WIDTH: float = 8.0     # pt  — a rectangle narrower than this is not a gutter candidate
_MAX_GUTTER_WIDTH: float = 90.0    # pt  — wider than this is a page margin / empty area, not a column gutter
_MIN_GUTTER_HEIGHT: float = 45.0   # pt  — a rectangle shorter than this cannot be a column separator
_MAX_RECTS_PER_REGION: int = 400   # safety cap on B&B extraction per page
_WORD_X_PADDING: float = 0.0       # pt  — inflate word boxes horizontally before search (0 = raw)
_WORD_Y_PADDING: float = 0.0       # pt  — inflate word boxes vertically before search

# --- Stage 2: scoring ----------------------------------------------------------------------------------
_FLANK_BAND: float = 60.0          # pt  — how far left/right of a rect edge we look for flanking text
_MIN_BOTH_SIDE_COVERAGE: float = 0.55   # fraction of rect height that must have text on BOTH sides
_IDEAL_GUTTER_WIDTH: float = 18.0  # pt  — width that scores best (real inter-column gaps cluster here)
_LINE_CROSS_MIN_OVERLAP: float = 5.0    # pt  — horizontal line must overlap rect x-range by this to count
_SCORE_THRESHOLD: float = 0.0      # accept a candidate if score >= this
_SELECT_MIN_X_OVERLAP: float = 3.0 # pt  — two kept gutters closer than this in x (same band) => suppress weaker

# Scoring weights (positive = evidence for a gutter, negative = against).
# These are deliberately exposed so they can be fit with logistic regression on
# labelled data instead of hand-tuned.
_W = {
    "both_side_coverage": 3.0,   # the single strongest signal
    "height_ratio":       1.5,   # taller = more likely a true column separator
    "width_fit":          1.0,   # closeness to the ideal inter-column width
    "line_cross":        -2.5,   # a ruled line crossing mid-rect => probably a table
    "numeric_flank":     -1.5,   # both flanks look like numbers/dashes => table column, not prose gutter
    "marker_flank":      -1.0,   # a flank is entirely list/bullet markers
    "identical_flank":   -1.0,   # a flank repeats the same token every row
    "short_flank":       -0.5,   # a flank is entirely very short tokens
    "bias":              -0.8,   # global offset so borderline noise falls below threshold
}


# =======================================================================================================================
# Stage 1 — Maximal whitespace rectangles (Breuel / Baird branch-and-bound)
# =======================================================================================================================

@dataclass(order=True)
class _QItem:
    """Priority-queue item; ordered by -quality so heapq pops the largest first."""
    neg_quality: float
    seq: int = field(compare=True)
    bound: tuple = field(compare=False, default=None)   # (x0, y0, x1, y1)
    obstacles: np.ndarray = field(compare=False, default=None)  # (n,4) float array


def _rect_quality(x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Upper-bound quality of any empty sub-rectangle of (x0,y0,x1,y1).

    We use area, which is monotone under containment (a sub-rectangle can only
    have <= area).  Monotonicity is what makes the branch-and-bound correct: the
    quality of a node is an over-estimate of any rectangle discoverable beneath
    it, so once the best queued node falls below the acceptance floor we can stop.
    """
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def find_maximal_whitespace_rectangles(
    obstacles: np.ndarray,
    bound: tuple,
    min_width: float = _MIN_GUTTER_WIDTH,
    min_height: float = _MIN_GUTTER_HEIGHT,
    max_results: int = _MAX_RECTS_PER_REGION,
    greedy_cover: bool = True,
) -> list[tuple]:
    """
    Find maximal empty (whitespace) rectangles inside ``bound`` that avoid every
    box in ``obstacles``.

    Parameters
    ----------
    obstacles : (n, 4) float ndarray of [x0, y0, x1, y1] boxes (word bboxes).
    bound     : (x0, y0, x1, y1) search region.
    min_width, min_height : reject rectangles smaller than this (also used to
                prune the search: a node whose bound is too small in the
                *relevant* dimension is dropped).
    max_results : stop after this many maximal rectangles (safety cap).
    greedy_cover : if True, each accepted rectangle is added back as an obstacle
                so subsequent rectangles do not overlap it.  This yields a clean,
                low-redundancy set of separators instead of many near-duplicates.

    Returns
    -------
    list of (x0, y0, x1, y1) maximal empty rectangles, largest-area first.

    Algorithm (Breuel 2002, "Two Geometric Algorithms for Layout Analysis";
    Baird's maximal empty rectangle):
        Maintain a max-priority-queue of (region, obstacles-inside-region)
        ordered by the region's area upper bound.  Pop the best region.  Drop
        obstacles that no longer intersect it.  If none remain, the region is a
        maximal empty rectangle — emit it.  Otherwise choose a pivot obstacle
        and push the four maximal sub-regions to the left / right / above / below
        of the pivot (each excludes the pivot's slab in one dimension), then
        recurse.  This provably enumerates maximal empty rectangles in
        decreasing area order.
    """
    x0b, y0b, x1b, y1b = bound
    if x1b - x0b <= 0 or y1b - y0b <= 0:
        return []

    # Keep only obstacles that actually intersect the initial bound.
    if obstacles.size:
        inside = (
            (obstacles[:, 0] < x1b) & (obstacles[:, 2] > x0b) &
            (obstacles[:, 1] < y1b) & (obstacles[:, 3] > y0b)
        )
        obstacles = obstacles[inside]

    min_area = min_width * min_height
    results: list[tuple] = []
    extra_obstacles: list[tuple] = []  # accepted rects (greedy cover)

    seq = 0
    heap: list[_QItem] = []
    heapq.heappush(heap, _QItem(-_rect_quality(x0b, y0b, x1b, y1b), seq, (x0b, y0b, x1b, y1b), obstacles))

    while heap and len(results) < max_results:
        item = heapq.heappop(heap)
        q = -item.neg_quality
        if q < min_area:
            break  # best remaining node cannot beat the floor => done (monotonicity)

        rx0, ry0, rx1, ry1 = item.bound
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < min_width or rh < min_height:
            continue

        obs = item.obstacles

        # Fold in greedy-cover obstacles discovered since this node was queued.
        if greedy_cover and extra_obstacles:
            extra = np.asarray(extra_obstacles, dtype=float)
            obs = np.vstack([obs, extra]) if obs.size else extra

        # Retain only obstacles strictly overlapping this region's interior.
        if obs.size:
            keep = (
                (obs[:, 0] < rx1) & (obs[:, 2] > rx0) &
                (obs[:, 1] < ry1) & (obs[:, 3] > ry0)
            )
            obs = obs[keep]

        if obs.size == 0:
            # Region is empty => a maximal whitespace rectangle.
            rect = (rx0, ry0, rx1, ry1)
            results.append(rect)
            if greedy_cover:
                extra_obstacles.append(rect)
            continue

        # Pick a pivot: the obstacle whose centre is closest to the region centre.
        cx = (rx0 + rx1) * 0.5
        cy = (ry0 + ry1) * 0.5
        ocx = (obs[:, 0] + obs[:, 2]) * 0.5
        ocy = (obs[:, 1] + obs[:, 3]) * 0.5
        pivot_i = int(np.argmin((ocx - cx) ** 2 + (ocy - cy) ** 2))
        px0, py0, px1, py1 = obs[pivot_i]

        # Four maximal sub-regions that exclude the pivot (clamped to region).
        candidates = [
            (rx0, ry0, min(px0, rx1), ry1),   # left of pivot
            (max(px1, rx0), ry0, rx1, ry1),   # right of pivot
            (rx0, ry0, rx1, min(py0, ry1)),   # above pivot (smaller y)
            (rx0, max(py1, ry0), rx1, ry1),   # below pivot (larger y)
        ]
        for sx0, sy0, sx1, sy1 in candidates:
            sw, sh = sx1 - sx0, sy1 - sy0
            if sw < min_width or sh < min_height:
                continue
            area = sw * sh
            if area < min_area:
                continue
            seq += 1
            heapq.heappush(heap, _QItem(-area, seq, (sx0, sy0, sx1, sy1), obs))

    return results


# =======================================================================================================================
# Stage 2 — scoring helpers
# =======================================================================================================================

_NUMERIC_VALUE_RE = re.compile(
    rf'^{_CURRENCY_SYM_CLASS}?\(?\d[\d,\.]*\)?(?:{_CURRENCY_SYM_CLASS}|%)?$'
)
_DASH_TOKENS = {"-", "–", "—", "−"}


def _is_numeric_or_dash(text: object) -> bool:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return False
    t = str(text).strip()
    return bool(t) and (t in _DASH_TOKENS or bool(_NUMERIC_VALUE_RE.match(t)))


def _interval_union_length(intervals: list[tuple]) -> float:
    """Total length covered by a set of (lo, hi) intervals (union, no double count)."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


def _both_side_intervals(left_iv: list[tuple], right_iv: list[tuple]) -> list[tuple]:
    """Intersection (in y) of the union of left intervals with the union of right intervals."""
    def _merge(ivs):
        if not ivs:
            return []
        ivs = sorted(ivs)
        out = [list(ivs[0])]
        for lo, hi in ivs[1:]:
            if lo <= out[-1][1]:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        return out

    L = _merge(left_iv)
    R = _merge(right_iv)
    out = []
    i = j = 0
    while i < len(L) and j < len(R):
        lo = max(L[i][0], R[j][0])
        hi = min(L[i][1], R[j][1])
        if hi > lo:
            out.append((lo, hi))
        if L[i][1] < R[j][1]:
            i += 1
        else:
            j += 1
    return out


def _score_candidate(
    rect: tuple,
    words_page: dict,
    hlines_page: dict | None,
) -> dict:
    """
    Compute features + score for one candidate rectangle on one page.

    words_page : dict of numpy arrays {x_left, x_right, y_top, y_bottom, word_id}
    hlines_page: dict {x_left, x_right, y_top} of horizontal ruled lines, or None
    """
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0

    xl = words_page["x_left"]
    xr = words_page["x_right"]
    yt = words_page["y_top"]
    yb = words_page["y_bottom"]
    wid = words_page["word_id"]

    # Vertical overlap with the rect (word touches the rect's y-span at all)
    v_overlap = (yt < y1) & (yb > y0)

    # Left flankers: word sits just to the LEFT of the rect (its right edge is
    # within [x0 - band, x0]) and overlaps the rect vertically.
    left_mask = v_overlap & (xr <= x0 + 0.5) & (xr >= x0 - _FLANK_BAND)
    right_mask = v_overlap & (xl >= x1 - 0.5) & (xl <= x1 + _FLANK_BAND)

    def _clamped_intervals(mask):
        if not mask.any():
            return []
        lo = np.maximum(yt[mask], y0)
        hi = np.minimum(yb[mask], y1)
        return [(float(a), float(b)) for a, b in zip(lo, hi) if b > a]

    left_iv = _clamped_intervals(left_mask)
    right_iv = _clamped_intervals(right_mask)

    both_iv = _both_side_intervals(left_iv, right_iv)
    both_cover = _interval_union_length(both_iv) / h if h > 0 else 0.0

    # Height ratio relative to the page's text height
    page_text_h = words_page["_page_text_h"]
    height_ratio = h / page_text_h if page_text_h > 0 else 0.0
    height_ratio = min(height_ratio, 1.0)

    # Width fit: 1.0 at the ideal width, decaying to 0 towards the min/max limits.
    if w <= _IDEAL_GUTTER_WIDTH:
        width_fit = max(0.0, (w - _MIN_GUTTER_WIDTH) / max(1e-6, _IDEAL_GUTTER_WIDTH - _MIN_GUTTER_WIDTH))
    else:
        width_fit = max(0.0, (_MAX_GUTTER_WIDTH - w) / max(1e-6, _MAX_GUTTER_WIDTH - _IDEAL_GUTTER_WIDTH))

    # Ruled-line crossing: a horizontal line whose y is strictly inside the rect
    # and whose x-range overlaps the rect => evidence of a table row separator.
    line_cross = 0.0
    if hlines_page is not None:
        hy = hlines_page["y_top"]
        hxl = hlines_page["x_left"]
        hxr = hlines_page["x_right"]
        inside_y = (hy > y0 + 2.0) & (hy < y1 - 2.0)
        x_ov = np.minimum(hxr, x1) - np.maximum(hxl, x0)
        crossing = inside_y & (x_ov >= _LINE_CROSS_MIN_OVERLAP)
        if crossing.any():
            # Fraction of the rect height "swept" by lines (density of crossings)
            n_cross = int(crossing.sum())
            # Normalise: many crossings over the height => strongly table-like.
            line_cross = min(1.0, n_cross / max(1.0, h / 24.0))

    # Content class of flanking words (soft, symmetric): look at the words that
    # actually flank the rect and see whether a whole side is non-content.
    def _texts(mask):
        ids = wid[mask]
        texts = [words_page["_text_by_id"].get(i) for i in ids]
        return [str(t).strip() for t in texts if t is not None and str(t).strip()]

    left_texts = _texts(left_mask)
    right_texts = _texts(right_mask)

    def _all(pred, texts):
        return bool(texts) and all(pred(t) for t in texts)

    numeric_flank = 1.0 if (_all(_is_numeric_or_dash, left_texts) or _all(_is_numeric_or_dash, right_texts)) else 0.0
    marker_flank = 1.0 if (_all(is_list_marker, left_texts) or _all(is_list_marker, right_texts)) else 0.0
    short_flank = 1.0 if (_all(lambda t: len(t) < 7, left_texts) or _all(lambda t: len(t) < 7, right_texts)) else 0.0
    identical_flank = 1.0 if (
        (len(left_texts) > 1 and len(set(left_texts)) == 1) or
        (len(right_texts) > 1 and len(set(right_texts)) == 1)
    ) else 0.0

    features = {
        "both_side_coverage": both_cover,
        "height_ratio": height_ratio,
        "width_fit": width_fit,
        "line_cross": line_cross,
        "numeric_flank": numeric_flank,
        "marker_flank": marker_flank,
        "identical_flank": identical_flank,
        "short_flank": short_flank,
    }

    score = _W["bias"] + sum(_W[k] * v for k, v in features.items())

    # Hard gate: a real column gutter must have text on both sides for a decent
    # fraction of its height, regardless of the linear score.  Everything else
    # (margins, whitespace above/below tables, figure gaps) fails here.
    passes_gate = both_cover >= _MIN_BOTH_SIDE_COVERAGE

    out = {
        "gutter_x_left": x0,
        "gutter_x_right": x1,
        "gutter_y_top": y0,
        "gutter_y_bottom": y1,
        "gutter_width": w,
        "gutter_height": h,
        "n_left_flank": len(left_texts),
        "n_right_flank": len(right_texts),
        "score": score,
        "passes_gate": passes_gate,
    }
    out.update(features)
    return out


# =======================================================================================================================
# Stage 2 — main scoring + selection
# =======================================================================================================================

def _select_non_overlapping(cands: list[dict]) -> list[dict]:
    """
    Greedy selection: highest score first; drop a candidate that overlaps an
    already-kept one in BOTH x (>= _SELECT_MIN_X_OVERLAP) and y.  Keeps the
    strongest representative of each real separator.
    """
    kept: list[dict] = []
    for c in sorted(cands, key=lambda d: d["score"], reverse=True):
        clash = False
        for k in kept:
            x_ov = min(c["gutter_x_right"], k["gutter_x_right"]) - max(c["gutter_x_left"], k["gutter_x_left"])
            y_ov = min(c["gutter_y_bottom"], k["gutter_y_bottom"]) - max(c["gutter_y_top"], k["gutter_y_top"])
            if x_ov >= _SELECT_MIN_X_OVERLAP and y_ov > 0:
                clash = True
                break
        if not clash:
            kept.append(c)
    return kept


def score_and_select_gutters(
    candidates_by_page: dict,
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every candidate rectangle and select the accepted gutters per page.

    Returns (df_gutters, df_scores).
    """
    ltr = df_words
    if "text_orientation" in df_words.columns:
        ltr = df_words[df_words["text_orientation"] == "LTR"]

    # Per-page word arrays
    page_words: dict = {}
    for pn, grp in ltr.groupby("page_number", sort=False):
        text_by_id = {}
        if "word_id" in grp.columns and "text" in grp.columns:
            text_by_id = dict(zip(grp["word_id"].values, grp["text"].values))
        page_words[pn] = {
            "x_left": grp["x_left"].to_numpy(float),
            "x_right": grp["x_right"].to_numpy(float),
            "y_top": grp["y_top"].to_numpy(float),
            "y_bottom": grp["y_bottom"].to_numpy(float),
            "word_id": grp["word_id"].to_numpy() if "word_id" in grp.columns else np.arange(len(grp)),
            "_text_by_id": text_by_id,
            "_page_text_h": float(grp["y_bottom"].max() - grp["y_top"].min()) if len(grp) else 0.0,
        }

    # Per-page horizontal lines
    page_hlines: dict = {}
    if df_shapes is not None and not df_shapes.empty:
        req = {"shape_orientation", "shape_type", "page_number", "x_left", "x_right", "y_top"}
        if req.issubset(df_shapes.columns):
            hl = df_shapes[
                (df_shapes["shape_orientation"].astype(str).str.lower() == "horizontal") &
                (df_shapes["shape_type"].astype(str).str.lower() == "line")
            ]
            for pn, grp in hl.groupby("page_number", sort=False):
                page_hlines[pn] = {
                    "x_left": grp["x_left"].to_numpy(float),
                    "x_right": grp["x_right"].to_numpy(float),
                    "y_top": grp["y_top"].to_numpy(float),
                }

    all_scores: list[dict] = []
    accepted_rows: list[dict] = []
    gutter_id = 1

    for pn, rects in candidates_by_page.items():
        wp = page_words.get(pn)
        if wp is None:
            continue
        hp = page_hlines.get(pn)

        scored = [_score_candidate(r, wp, hp) for r in rects]
        for s in scored:
            s["page_number"] = pn
        all_scores.extend(scored)

        # Accept: passes the both-side gate AND score >= threshold.
        passing = [s for s in scored if s["passes_gate"] and s["score"] >= _SCORE_THRESHOLD]
        kept = _select_non_overlapping(passing)

        for k in kept:
            accepted_rows.append({
                "page_number": pn,
                "gutter_candidate_id": gutter_id,
                "gutter_id": gutter_id,
                "gutter_x_left": k["gutter_x_left"],
                "gutter_x_right": k["gutter_x_right"],
                "gutter_y_top": k["gutter_y_top"],
                "gutter_y_bottom": k["gutter_y_bottom"],
                "gutter_width": k["gutter_width"],
                "gutter_height": k["gutter_height"],
                "score": k["score"],
                "both_side_coverage": k["both_side_coverage"],
            })
            gutter_id += 1

    gutter_cols = [
        "page_number", "gutter_candidate_id", "gutter_id",
        "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
        "gutter_width", "gutter_height", "score", "both_side_coverage",
    ]
    df_gutters = pd.DataFrame(accepted_rows, columns=gutter_cols)

    score_cols = [
        "page_number",
        "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
        "gutter_width", "gutter_height",
        "both_side_coverage", "height_ratio", "width_fit", "line_cross",
        "numeric_flank", "marker_flank", "identical_flank", "short_flank",
        "n_left_flank", "n_right_flank", "passes_gate", "score",
    ]
    df_scores = pd.DataFrame(all_scores, columns=score_cols)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(
            ["page_number", "score"], ascending=[True, False]
        ).reset_index(drop=True)

    return df_gutters, df_scores


# =======================================================================================================================
# Stage 1 driver — per page
# =======================================================================================================================

def _build_candidates_by_page(df_words: pd.DataFrame) -> dict:
    """
    Run the whitespace-rectangle search per page and return
    {page_number: [ (x0,y0,x1,y1), ... ]} filtered to gutter-shaped rectangles.
    """
    ltr = df_words
    if "text_orientation" in df_words.columns:
        ltr = df_words[df_words["text_orientation"] == "LTR"]

    out: dict = {}
    for pn, grp in ltr.groupby("page_number", sort=False):
        if grp.empty:
            continue
        obs = np.column_stack([
            grp["x_left"].to_numpy(float) - _WORD_X_PADDING,
            grp["y_top"].to_numpy(float) - _WORD_Y_PADDING,
            grp["x_right"].to_numpy(float) + _WORD_X_PADDING,
            grp["y_bottom"].to_numpy(float) + _WORD_Y_PADDING,
        ])
        # Search bound = the page's text bounding box (so page margins are not
        # reported as gutters — only whitespace *between* text counts).
        bound = (
            float(grp["x_left"].min()),
            float(grp["y_top"].min()),
            float(grp["x_right"].max()),
            float(grp["y_bottom"].max()),
        )
        rects = find_maximal_whitespace_rectangles(obs, bound)
        # Keep only gutter-shaped rectangles (tall + within width band).
        rects = [
            r for r in rects
            if _MIN_GUTTER_WIDTH <= (r[2] - r[0]) <= _MAX_GUTTER_WIDTH
            and (r[3] - r[1]) >= _MIN_GUTTER_HEIGHT
        ]
        out[pn] = rects
    return out


# =======================================================================================================================
# Public API
# =======================================================================================================================

def detect_gutters_v2(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect column gutters via whitespace-cover + scoring.

    Returns
    -------
    df_gutters : one row per accepted gutter (legacy-compatible schema + score).
    df_scores  : every candidate rectangle with its features and score.
    """
    empty_g = pd.DataFrame(columns=[
        "page_number", "gutter_candidate_id", "gutter_id",
        "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom",
        "gutter_width", "gutter_height", "score", "both_side_coverage",
    ])
    empty_s = pd.DataFrame()

    if df_words is None or df_words.empty:
        return empty_g, empty_s

    required = {"page_number", "x_left", "x_right", "y_top", "y_bottom"}
    missing = required - set(df_words.columns)
    if missing:
        raise ValueError(f"df_words missing required columns: {sorted(missing)}")

    candidates_by_page = _build_candidates_by_page(df_words)
    df_gutters, df_scores = score_and_select_gutters(candidates_by_page, df_words, df_shapes)
    return df_gutters, df_scores


def detect_and_annotate_gutters_v2(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop-in replacement for legacy detect_and_annotate_gutters().

    Returns (df_words_annotated, df_scores, df_gutters).  (The middle element is
    the per-candidate score table, replacing the legacy candidate frame.)
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_gutters, df_scores = detect_gutters_v2(df_words, df_shapes)

    # Reuse the legacy word-annotation step so downstream ordering is unchanged.
    try:
        from .gutter_detector import merge_gutters_onto_words
        df_words_out = merge_gutters_onto_words(df_words, df_gutters)
    except Exception:
        df_words_out = df_words

    return df_words_out, df_scores, df_gutters

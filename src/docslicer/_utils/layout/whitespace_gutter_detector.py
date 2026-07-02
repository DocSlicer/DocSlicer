"""
whitespace_gutter_detector.py

Gutter detection via maximal whitespace rectangles (Breuel-style branch & bound).

Instead of stitching per-row gaps into vertical chains (see gutter_detector.py),
this module finds gutters top-down: word bounding boxes and line shapes are
treated as obstacles, and the globally tallest empty rectangles on each page are
enumerated directly with a branch-and-bound search.  A gutter is simply a tall,
narrow maximal whitespace rectangle that is flanked by real text content on both
sides.

Anti-pattern (table gaps) is handled with continuous features rather than
binary kill-switches:
  - per-line gap density   (table rows have many aligned gaps; body text has 1)
  - numeric/dash flank fraction  (financial columns are numbers and dashes)
  - repeated/short flank content
  - parallel-rectangle count in the same y-band

Line shapes from df_shapes are obstacles too, so whitespace rectangles stop
naturally at horizontal rules (section separators, table borders) and vertical
rules (table column borders) — no clip/kill toggles needed.

Public API:
    df_gutters, df_candidates = detect_gutters_whitespace(df_words, df_shapes)

df_gutters columns (compatible with gutter_detector.promote_* output):
    page_number, gutter_id,
    gutter_y_top, gutter_y_bottom, gutter_x_left, gutter_x_right,
    gutter_width, gutter_height
plus diagnostic feature columns (n_flank_both, median_gap_density, ...).

df_candidates contains every whitespace rectangle that was considered, with all
features and a reject_reason column — useful for tuning/debugging.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..text_utils import _CURRENCY_SYM_CLASS, is_list_marker

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

_MIN_GUTTER_WIDTH: float = 9.2    # pt - minimum whitespace rectangle width to qualify as a gutter
_MIN_GUTTER_HEIGHT: float = 40.0  # pt - minimum (trimmed) rectangle height
_MIN_FLANK_LINES_BOTH: int = 4    # min number of text lines with words on BOTH sides of the rect
_LINE_CLUSTER_TOL: float = 4.0    # pt - y_center distance for clustering words into text lines
_MAX_LINE_GAP_DENSITY: int = 3    # a text line with more than this many wide gaps looks like a table row
_MIN_LOW_DENSITY_LINES: int = 3   # min flanked lines that must come from low-density (non-table) rows
_MAX_NUMERIC_FLANK_FRAC: float = 0.80  # reject if >80% of a flank looks numeric/dash (financial column)
_MAX_IDENTICAL_FLANK_FRAC: float = 0.90  # reject if >90% of a flank is the same repeated token
_MAX_RECTS_PER_PAGE: int = 30     # stop b&b after this many accepted rectangles per page
_MAX_POPS_PER_PAGE: int = 30_000  # hard safety valve on branch&bound iterations per page
_MIN_LINE_SHAPE_LENGTH: float = 8.0   # pt - ignore line shapes shorter than this (specks)
_VLINES_AS_OBSTACLES: bool = True     # vertical line shapes block whitespace (stops table-border gaps)
_HLINES_AS_OBSTACLES: bool = True     # horizontal line shapes block whitespace (stops at section rules)
_COLLAPSE_STRUCT_GROUPS: bool = True  # union word boxes sharing struct_group_id within one text line
_EXPAND_EPS: float = 0.05             # pt - numeric slack when expanding rectangles to maximality
_ABSORB_X_EPS: float = 2.0            # pt - a candidate this close (in x) to a taller gutter that y-covers it is a ragged-edge sliver
_ABSORB_MIN_Y_COVER: float = 0.8      # fraction of a candidate's height that must be covered by the taller gutter to absorb it
_VLINE_ADJACENT_EPS: float = 6.0      # pt - a border vline within this distance of a rect flags it as a table gap
_VLINE_GRID_SNAP: float = 4.0         # pt - vline endpoint must be this close to an x-overlapping hline to count as border grid
_MAX_VLINE_COVER_FRAC: float = 0.6    # reject when a grid vline covers >= this fraction of the rect height
_TABLE_STRUCT_TAGS = {"TD", "TH", "TR", "TABLE", "THEAD", "TBODY", "TFOOT"}  # struct-tree tags that mark table content
_MAX_TABLE_TAG_FRAC: float = 0.8      # reject when BOTH flanks are >= this fraction table-tagged words
_CROSS_BAND_X_WINDOW: float = 100.0   # pt - neighborhood width on each side for the horizontal-band (cuttability) test
_MIN_CROSS_BAND_HEIGHT: float = 6.0   # pt - min height of an empty horizontal band to count (row spacing, not line leading)
_MAX_CROSS_BAND_RATIO: float = 0.6    # reject when crossing bands / flanked lines >= this (region is row-structured)
_MIN_CROSS_BANDS: int = 3             # ...and at least this many bands (avoid killing on 1-2 section breaks)
# --- table-region propagation: gaps rejected as table gaps poison nearby accepted candidates ---
_TABLE_EVIDENCE_REASONS = {
    "numeric_flank", "table_gap_density", "table_border_vline",
    "repeated_flank", "marker_flank", "short_flank", "table_row_bands", "table_struct_tag",
}
_REGION_X_WINDOW: float = 250.0       # pt - max x distance between candidates to belong to the same region
_REGION_Y_CHAIN: float = 60.0         # pt - max y distance to chain same-x candidates into one region (stacked row bands)
_REGION_IMMUNE_HEIGHT_FRAC: float = 0.5   # candidates >= this fraction of page text height are never region-rejected
_REGION_WEAK_HEIGHT_FRAC: float = 0.35    # below this fraction, a single piece of table evidence in the region rejects
_REGION_MIN_EVIDENCE: int = 2         # at/above _REGION_WEAK_HEIGHT_FRAC, this much evidence is needed


# =======================================================================================================================
# Flank content classifiers (same spirit as gutter_detector.reject_non_content_gutters)
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


# =======================================================================================================================
# Obstacle construction
# =======================================================================================================================

def _collapse_struct_groups(page_words: pd.DataFrame, line_ids: np.ndarray) -> np.ndarray:
    """
    Union word boxes that share a struct_group_id within the same text line, so
    intra-block gaps (dot leaders, tab stops inside one logical block) never
    open a whitespace rectangle.  Returns an (n, 4) obstacle array
    [x_left, y_top, x_right, y_bottom].
    """
    boxes = page_words[["x_left", "y_top", "x_right", "y_bottom"]].to_numpy(dtype=np.float64)

    if not _COLLAPSE_STRUCT_GROUPS or "struct_group_id" not in page_words.columns:
        return boxes

    sgid = page_words["struct_group_id"]
    has_group = sgid.notna().to_numpy()
    if not has_group.any():
        return boxes

    keys = pd.DataFrame({
        "sgid": sgid.to_numpy(),
        "line": line_ids,
    })
    grouped = (
        pd.DataFrame(boxes[has_group], columns=["x0", "y0", "x1", "y1"])
        .assign(sgid=keys.loc[has_group, "sgid"].values, line=keys.loc[has_group, "line"].values)
        .groupby(["sgid", "line"], sort=False)
        .agg(x0=("x0", "min"), y0=("y0", "min"), x1=("x1", "max"), y1=("y1", "max"))
        .to_numpy(dtype=np.float64)
    )
    return np.vstack([grouped, boxes[~has_group]]) if (~has_group).any() else grouped


def _line_shape_obstacles(page_shapes: pd.DataFrame | None) -> np.ndarray:
    """Line shapes (horizontal + vertical) as thin obstacle boxes."""
    if page_shapes is None or page_shapes.empty:
        return np.empty((0, 4), dtype=np.float64)

    required = {"shape_type", "shape_orientation", "x_left", "x_right", "y_top", "y_bottom"}
    if not required.issubset(page_shapes.columns):
        return np.empty((0, 4), dtype=np.float64)

    is_line = page_shapes["shape_type"].astype(str).str.lower() == "line"
    orient = page_shapes["shape_orientation"].astype(str).str.lower()

    keep = pd.Series(False, index=page_shapes.index)
    if _HLINES_AS_OBSTACLES:
        h = is_line & (orient == "horizontal")
        h &= (page_shapes["x_right"] - page_shapes["x_left"]) >= _MIN_LINE_SHAPE_LENGTH
        keep |= h
    if _VLINES_AS_OBSTACLES:
        v = is_line & (orient == "vertical")
        v &= (page_shapes["y_bottom"] - page_shapes["y_top"]) >= _MIN_LINE_SHAPE_LENGTH
        keep |= v

    if not keep.any():
        return np.empty((0, 4), dtype=np.float64)

    return page_shapes.loc[keep, ["x_left", "y_top", "x_right", "y_bottom"]].to_numpy(dtype=np.float64)


def _grid_vlines(page_shapes: pd.DataFrame | None) -> np.ndarray:
    """
    Vertical line shapes that belong to a table border grid: at least one
    endpoint sits on (within _VLINE_GRID_SNAP of) a horizontal line that
    x-overlaps it.  A whitespace rectangle flanked by such a line is a gap
    between cell text and a table border, not a column gutter.  Free-floating
    vertical rules (decorative column separators) do NOT qualify, so they
    cannot kill a real gutter.

    Returns an (n, 3) array of [x_center, y_top, y_bottom].
    """
    if page_shapes is None or page_shapes.empty:
        return np.empty((0, 3), dtype=np.float64)
    required = {"shape_type", "shape_orientation", "x_left", "x_right", "y_top", "y_bottom"}
    if not required.issubset(page_shapes.columns):
        return np.empty((0, 3), dtype=np.float64)

    is_line = page_shapes["shape_type"].astype(str).str.lower() == "line"
    orient = page_shapes["shape_orientation"].astype(str).str.lower()
    v = page_shapes[is_line & (orient == "vertical")]
    h = page_shapes[is_line & (orient == "horizontal")]
    if v.empty or h.empty:
        return np.empty((0, 3), dtype=np.float64)

    vx0 = v["x_left"].to_numpy(); vx1 = v["x_right"].to_numpy()
    vy0 = v["y_top"].to_numpy(); vy1 = v["y_bottom"].to_numpy()
    hx0 = h["x_left"].to_numpy(); hx1 = h["x_right"].to_numpy()
    hy = ((h["y_top"] + h["y_bottom"]) / 2.0).to_numpy()

    # (n_v, n_h): hline x-overlaps the vline AND one vline endpoint touches it
    x_ov = (hx0[None, :] <= vx1[:, None] + _VLINE_GRID_SNAP) & (hx1[None, :] >= vx0[:, None] - _VLINE_GRID_SNAP)
    y_touch = (np.abs(hy[None, :] - vy0[:, None]) <= _VLINE_GRID_SNAP) | \
              (np.abs(hy[None, :] - vy1[:, None]) <= _VLINE_GRID_SNAP)
    grid = (x_ov & y_touch).any(axis=1)

    if not grid.any():
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack([((vx0 + vx1) / 2.0)[grid], vy0[grid], vy1[grid]])


def _vline_cover_frac(rect_row: dict, grid_vl: np.ndarray) -> float:
    """Max fraction of the (trimmed) rect height covered by an adjacent grid vline."""
    if not len(grid_vl):
        return 0.0
    x0, x1 = rect_row["gutter_x_left"], rect_row["gutter_x_right"]
    y0, y1 = rect_row["gutter_y_top"], rect_row["gutter_y_bottom"]
    h = y1 - y0
    if h <= 0:
        return 0.0
    near = (grid_vl[:, 0] >= x0 - _VLINE_ADJACENT_EPS) & (grid_vl[:, 0] <= x1 + _VLINE_ADJACENT_EPS)
    if not near.any():
        return 0.0
    cover = np.minimum(grid_vl[near, 2], y1) - np.maximum(grid_vl[near, 1], y0)
    return float(max(0.0, cover.max()) / h)


# =======================================================================================================================
# Branch & bound: maximal whitespace rectangles
# =======================================================================================================================

def _expand_rect(
    rect: tuple, obstacles: np.ndarray, accepted: list, bound: tuple
) -> tuple:
    """
    Grow an empty rectangle to maximality (until it touches an obstacle, an
    already-accepted rectangle, or the page text bound) — two passes of
    up/down/left/right, since growing one axis can unlock the other.
    """
    x0, y0, x1, y1 = rect
    if accepted:
        obs = np.vstack([obstacles, np.array(accepted, dtype=np.float64)]) if len(obstacles) else np.array(accepted, dtype=np.float64)
    else:
        obs = obstacles
    bx0, by0, bx1, by1 = bound

    for _ in range(2):
        if len(obs):
            # grow up: obstacles overlapping in x with bottom edge above y0
            mx = (obs[:, 0] < x1 - _EXPAND_EPS) & (obs[:, 2] > x0 + _EXPAND_EPS)
            tops = obs[mx & (obs[:, 3] <= y0 + _EXPAND_EPS), 3]
            y0 = max(by0, tops.max()) if len(tops) else by0
            # grow down
            bots = obs[mx & (obs[:, 1] >= y1 - _EXPAND_EPS), 1]
            y1 = min(by1, bots.min()) if len(bots) else by1
            # grow left: obstacles overlapping in (new) y with right edge left of x0
            my = (obs[:, 1] < y1 - _EXPAND_EPS) & (obs[:, 3] > y0 + _EXPAND_EPS)
            lefts = obs[my & (obs[:, 2] <= x0 + _EXPAND_EPS), 2]
            x0 = max(bx0, lefts.max()) if len(lefts) else bx0
            # grow right
            rights = obs[my & (obs[:, 0] >= x1 - _EXPAND_EPS), 0]
            x1 = min(bx1, rights.min()) if len(rights) else bx1
        else:
            x0, y0, x1, y1 = bx0, by0, bx1, by1
    return (x0, y0, x1, y1)


def _maximal_whitespace_rects(
    obstacles: np.ndarray,
    bound: tuple,
    min_w: float = _MIN_GUTTER_WIDTH,
    min_h: float = _MIN_GUTTER_HEIGHT,
    max_results: int = _MAX_RECTS_PER_PAGE,
) -> list:
    """
    Enumerate maximal empty rectangles inside `bound`, tallest first, using
    Breuel's branch-and-bound whitespace cover algorithm.

    Quality = rectangle height (we want tall separators).  The height of a
    bounding rectangle is an upper bound on the height of any empty rectangle
    inside it, so a max-heap on height pops candidates in globally optimal
    order.  Accepted rectangles become obstacles for the rest of the search,
    yielding a greedy non-overlapping cover.

    Returns a list of (x0, y0, x1, y1) tuples.
    """
    accepted: list = []
    heap: list = []
    counter = 0

    def push(rect: tuple, idxs: np.ndarray) -> None:
        nonlocal counter
        x0, y0, x1, y1 = rect
        if (x1 - x0) < min_w or (y1 - y0) < min_h:
            return
        counter += 1
        heapq.heappush(heap, (-(y1 - y0), counter, rect, idxs))

    push(bound, np.arange(len(obstacles)))
    pops = 0

    while heap and len(accepted) < max_results and pops < _MAX_POPS_PER_PAGE:
        pops += 1
        _, _, rect, idxs = heapq.heappop(heap)
        x0, y0, x1, y1 = rect

        # Re-filter parent obstacle set to those strictly overlapping this rect
        if len(idxs):
            o = obstacles[idxs]
            m = (o[:, 0] < x1 - _EXPAND_EPS) & (o[:, 2] > x0 + _EXPAND_EPS) & \
                (o[:, 1] < y1 - _EXPAND_EPS) & (o[:, 3] > y0 + _EXPAND_EPS)
            idxs = idxs[m]

        # Already-accepted rectangles act as obstacles (non-overlap greedy cover)
        acc_overlap = [
            a for a in accepted
            if a[0] < x1 - _EXPAND_EPS and a[2] > x0 + _EXPAND_EPS
            and a[1] < y1 - _EXPAND_EPS and a[3] > y0 + _EXPAND_EPS
        ]

        if len(idxs) == 0 and not acc_overlap:
            # Empty rectangle → expand to maximality, then accept
            ex = _expand_rect(rect, obstacles, accepted, bound)
            if (ex[2] - ex[0]) >= min_w and (ex[3] - ex[1]) >= min_h:
                accepted.append(ex)
            continue

        # Pick pivot: the overlapping obstacle nearest the rect center
        pool = obstacles[idxs] if len(idxs) else np.array(acc_overlap, dtype=np.float64)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        d = ((pool[:, 0] + pool[:, 2]) / 2.0 - cx) ** 2 + ((pool[:, 1] + pool[:, 3]) / 2.0 - cy) ** 2
        p = pool[int(np.argmin(d))]

        # Split into the four sub-rectangles around the pivot
        push((x0, y0, p[0], y1), idxs)   # left of pivot
        push((p[2], y0, x1, y1), idxs)   # right of pivot
        push((x0, y0, x1, p[1]), idxs)   # above pivot
        push((x0, p[3], x1, y1), idxs)   # below pivot

    return accepted


# =======================================================================================================================
# Text-line structure (for flank evidence + trimming)
# =======================================================================================================================

def _cluster_lines(page_words: pd.DataFrame) -> np.ndarray:
    """
    Cluster words into text lines by y_center proximity.  Returns an int array
    (line id per word, aligned with page_words row order).
    """
    yc = ((page_words["y_top"] + page_words["y_bottom"]) / 2.0).to_numpy(dtype=np.float64)
    order = np.argsort(yc, kind="stable")
    line_ids = np.zeros(len(yc), dtype=np.int64)
    lid = 0
    prev = None
    for i in order:
        if prev is not None and (yc[i] - prev) > _LINE_CLUSTER_TOL:
            lid += 1
        line_ids[i] = lid
        prev = yc[i]
    return line_ids


class _PageLines:
    """Precomputed per-line arrays for fast flank queries."""

    def __init__(self, page_words: pd.DataFrame, line_ids: np.ndarray):
        self.x_left = page_words["x_left"].to_numpy(dtype=np.float64)
        self.x_right = page_words["x_right"].to_numpy(dtype=np.float64)
        self.y_top = page_words["y_top"].to_numpy(dtype=np.float64)
        self.y_bottom = page_words["y_bottom"].to_numpy(dtype=np.float64)
        self.text = page_words["text"].astype(str).to_numpy() if "text" in page_words.columns else np.full(len(page_words), "", dtype=object)
        if "struct_tag" in page_words.columns:
            self.is_table_tag = (
                page_words["struct_tag"].astype(str).str.upper().isin(_TABLE_STRUCT_TAGS).to_numpy()
            )
            self.has_struct = page_words["struct_tag"].notna().to_numpy()
        else:
            self.is_table_tag = np.zeros(len(page_words), dtype=bool)
            self.has_struct = np.zeros(len(page_words), dtype=bool)
        self.line_ids = line_ids

        self.lines: list = []  # (y_center, word_indices sorted by x_left)
        for lid in np.unique(line_ids):
            idx = np.nonzero(line_ids == lid)[0]
            idx = idx[np.argsort(self.x_left[idx], kind="stable")]
            ycen = float(np.mean((self.y_top[idx] + self.y_bottom[idx]) / 2.0))
            # gap density: number of wide gaps between consecutive words in this line
            gaps = self.x_left[idx][1:] - self.x_right[idx][:-1]
            gap_count = int(np.sum(gaps > _MIN_GUTTER_WIDTH))
            self.lines.append((ycen, idx, gap_count))
        self.lines.sort(key=lambda t: t[0])
        self.line_ycenters = np.array([t[0] for t in self.lines], dtype=np.float64)


def _rect_features(rect: tuple, pl: _PageLines) -> dict:
    """
    Trim a whitespace rectangle to the y-span of text lines flanked on both
    sides, and compute classification features.
    """
    x0, y0, x1, y1 = rect
    lo = int(np.searchsorted(pl.line_ycenters, y0, side="left"))
    hi = int(np.searchsorted(pl.line_ycenters, y1, side="right"))

    n_span = 0
    n_left = 0
    n_right = 0
    both_lines: list = []  # (line index, nearest_left word idx, nearest_right word idx, gap_count)

    for li in range(lo, hi):
        _, idx, gap_count = pl.lines[li]
        n_span += 1
        # nearest word fully to the left of the rect
        left_mask = pl.x_right[idx] <= x0 + 0.5
        right_mask = pl.x_left[idx] >= x1 - 0.5
        has_l = bool(left_mask.any())
        has_r = bool(right_mask.any())
        n_left += has_l
        n_right += has_r
        if has_l and has_r:
            wl = idx[left_mask][int(np.argmax(pl.x_right[idx[left_mask]]))]
            wr = idx[right_mask][int(np.argmin(pl.x_left[idx[right_mask]]))]
            both_lines.append((li, wl, wr, gap_count))

    feat = {
        "n_lines_span": n_span,
        "n_flank_left": n_left,
        "n_flank_right": n_right,
        "n_flank_both": len(both_lines),
    }

    if not both_lines:
        feat.update(
            trim_y_top=y0, trim_y_bottom=y1,
            median_gap_density=np.nan, n_low_density_lines=0,
            numeric_frac_left=np.nan, numeric_frac_right=np.nan,
            identical_frac_left=np.nan, identical_frac_right=np.nan,
            all_short_left=False, all_short_right=False,
            all_marker_left=False, all_marker_right=False,
            table_tag_frac_left=np.nan, table_tag_frac_right=np.nan,
            n_cross_bands=0, cross_band_ratio=0.0,
        )
        return feat

    # Trim vertical span to first/last flanked line (use flanking word boxes)
    first, last = both_lines[0], both_lines[-1]
    feat["trim_y_top"] = float(min(pl.y_top[first[1]], pl.y_top[first[2]]))
    feat["trim_y_bottom"] = float(max(pl.y_bottom[last[1]], pl.y_bottom[last[2]]))

    gap_counts = np.array([b[3] for b in both_lines])
    feat["median_gap_density"] = float(np.median(gap_counts))
    feat["n_low_density_lines"] = int(np.sum(gap_counts <= _MAX_LINE_GAP_DENSITY))

    for side, wcol in (("left", 1), ("right", 2)):
        texts = [str(pl.text[b[wcol]]).strip() for b in both_lines]
        n = len(texts)
        feat[f"numeric_frac_{side}"] = sum(_is_numeric_or_dash(t) for t in texts) / n
        feat[f"identical_frac_{side}"] = Counter(texts).most_common(1)[0][1] / n
        feat[f"all_short_{side}"] = all(len(t) < 7 for t in texts)
        feat[f"all_marker_{side}"] = all(is_list_marker(t) for t in texts)
        # Struct-tree table evidence: fraction of flanking words tagged TD/TH/...
        # NaN when the flank has no struct info (untagged PDF) → rule is skipped.
        widx = np.array([b[wcol] for b in both_lines])
        tagged = pl.has_struct[widx]
        feat[f"table_tag_frac_{side}"] = (
            float(pl.is_table_tag[widx][tagged].mean()) if tagged.any() else np.nan
        )

    # Horizontal cuttability: empty y-bands crossing the rect AND its flanking
    # text neighborhoods.  Row-structured regions (tables) have a band between
    # every row; independent text columns interleave and leave no shared bands.
    ty0, ty1 = feat["trim_y_top"], feat["trim_y_bottom"]
    nb = (
        (pl.x_right > x0 - _CROSS_BAND_X_WINDOW) & (pl.x_left < x1 + _CROSS_BAND_X_WINDOW)
        & (pl.y_bottom > ty0) & (pl.y_top < ty1)
    )
    n_bands = 0
    if nb.any():
        ys = np.column_stack([pl.y_top[nb], pl.y_bottom[nb]])
        ys = ys[np.argsort(ys[:, 0], kind="stable")]
        cursor = ys[0, 1]
        for wy0, wy1 in ys[1:]:
            if wy0 - cursor >= _MIN_CROSS_BAND_HEIGHT:
                n_bands += 1
            cursor = max(cursor, wy1)
    feat["n_cross_bands"] = n_bands
    feat["cross_band_ratio"] = n_bands / max(len(both_lines) - 1, 1)

    return feat


# =======================================================================================================================
# Classification
# =======================================================================================================================

def _reject_reason(row: pd.Series) -> str | None:
    """Transparent accept/reject rule.  Returns None when the rect is a gutter."""
    if row["gutter_width"] < _MIN_GUTTER_WIDTH:
        return "too_narrow"
    if row["gutter_height"] < _MIN_GUTTER_HEIGHT:
        return "too_short"
    if row["n_flank_both"] < _MIN_FLANK_LINES_BOTH:
        return "insufficient_flank"
    if row["median_gap_density"] > _MAX_LINE_GAP_DENSITY:
        return "table_gap_density"
    if row["n_low_density_lines"] < _MIN_LOW_DENSITY_LINES:
        return "table_gap_density"
    if row["numeric_frac_left"] >= _MAX_NUMERIC_FLANK_FRAC or row["numeric_frac_right"] >= _MAX_NUMERIC_FLANK_FRAC:
        return "numeric_flank"
    if row["identical_frac_left"] >= _MAX_IDENTICAL_FLANK_FRAC or row["identical_frac_right"] >= _MAX_IDENTICAL_FLANK_FRAC:
        return "repeated_flank"
    if row["all_marker_left"] or row["all_marker_right"]:
        return "marker_flank"
    if row["all_short_left"] or row["all_short_right"]:
        return "short_flank"
    if row["vline_cover_frac"] >= _MAX_VLINE_COVER_FRAC:
        return "table_border_vline"
    ttl, ttr = row["table_tag_frac_left"], row["table_tag_frac_right"]
    if pd.notna(ttl) and pd.notna(ttr) and min(ttl, ttr) >= _MAX_TABLE_TAG_FRAC:
        return "table_struct_tag"
    if row["n_cross_bands"] >= _MIN_CROSS_BANDS and row["cross_band_ratio"] >= _MAX_CROSS_BAND_RATIO:
        return "table_row_bands"
    return None


# =======================================================================================================================
# Public API
# =======================================================================================================================

def detect_gutters_whitespace(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect column gutters via maximal whitespace rectangles.

    Parameters
    ----------
    df_words : word-level DataFrame (needs page_number, x_left, x_right, y_top,
               y_bottom, text_orientation; text and struct_group_id are used
               when present).
    df_shapes : shape DataFrame (line shapes become whitespace obstacles).

    Returns
    -------
    (df_gutters, df_candidates)
      df_gutters    — accepted gutters, one row each, with gutter_id.
      df_candidates — every whitespace rectangle considered, with features and
                      reject_reason (None = accepted).  For debugging/tuning.
    """
    empty_g = pd.DataFrame(columns=[
        "page_number", "gutter_id",
        "gutter_y_top", "gutter_y_bottom", "gutter_x_left", "gutter_x_right",
        "gutter_width", "gutter_height",
    ])
    if df_words is None or df_words.empty:
        return empty_g, pd.DataFrame()

    if "text_orientation" in df_words.columns:
        words = df_words[df_words["text_orientation"].astype(str).str.upper().str.strip() == "LTR"].copy()
    else:
        words = df_words.copy()
    if words.empty:
        return empty_g, pd.DataFrame()

    shapes_by_page: dict = {}
    if df_shapes is not None and not df_shapes.empty and "page_number" in df_shapes.columns:
        shapes_by_page = {pn: g for pn, g in df_shapes.groupby("page_number", sort=False)}

    candidate_rows: list = []

    for page_num, pw in words.groupby("page_number", sort=True):
        pw = pw.reset_index(drop=True)
        line_ids = _cluster_lines(pw)
        pl = _PageLines(pw, line_ids)

        page_shapes = shapes_by_page.get(page_num)
        word_obs = _collapse_struct_groups(pw, line_ids)
        line_obs = _line_shape_obstacles(page_shapes)
        obstacles = np.vstack([word_obs, line_obs]) if len(line_obs) else word_obs
        grid_vl = _grid_vlines(page_shapes)

        # Search bound = text region of the page (margins are not gutters)
        bound = (
            float(pw["x_left"].min()), float(pw["y_top"].min()),
            float(pw["x_right"].max()), float(pw["y_bottom"].max()),
        )
        if bound[2] - bound[0] < 2 * _MIN_GUTTER_WIDTH or bound[3] - bound[1] < _MIN_GUTTER_HEIGHT:
            continue

        rects = _maximal_whitespace_rects(obstacles, bound)

        for (x0, y0, x1, y1) in rects:
            feat = _rect_features((x0, y0, x1, y1), pl)
            row = {
                "page_number": page_num,
                "gutter_x_left": x0,
                "gutter_x_right": x1,
                "gutter_y_top": feat.pop("trim_y_top"),
                "gutter_y_bottom": feat.pop("trim_y_bottom"),
                "raw_y_top": y0,
                "raw_y_bottom": y1,
                **feat,
            }
            row["gutter_width"] = row["gutter_x_right"] - row["gutter_x_left"]
            row["gutter_height"] = row["gutter_y_bottom"] - row["gutter_y_top"]
            row["vline_cover_frac"] = _vline_cover_frac(row, grid_vl)
            row["page_text_height"] = bound[3] - bound[1]
            candidate_rows.append(row)

    if not candidate_rows:
        return empty_g, pd.DataFrame()

    df_candidates = pd.DataFrame(candidate_rows)
    df_candidates["reject_reason"] = df_candidates.apply(_reject_reason, axis=1)

    # Table-region propagation: gaps already rejected as table gaps mark their
    # neighborhood as a table region.  Accepted candidates in the same region
    # (connected by proximity) are rejected too — unless they are tall enough
    # to be a page-level column separator (a real gutter can run alongside a
    # table that lives inside one column).
    df_candidates["height_frac"] = df_candidates["gutter_height"] / df_candidates["page_text_height"]
    for pn, grp in df_candidates.groupby("page_number", sort=False):
        members = grp[grp["reject_reason"].isna() | grp["reject_reason"].isin(_TABLE_EVIDENCE_REASONS)]
        n = len(members)
        if n < 2:
            continue
        x0 = members["gutter_x_left"].to_numpy(); x1 = members["gutter_x_right"].to_numpy()
        y0 = members["gutter_y_top"].to_numpy(); y1 = members["gutter_y_bottom"].to_numpy()
        x_gap = np.maximum(x0[:, None] - x1[None, :], x0[None, :] - x1[:, None])
        y_gap = np.maximum(y0[:, None] - y1[None, :], y0[None, :] - y1[:, None])
        adj = ((y_gap < 0) & (x_gap <= _REGION_X_WINDOW)) | ((x_gap < 0) & (y_gap <= _REGION_Y_CHAIN))

        # Connected components via BFS
        comp = np.full(n, -1)
        cid = 0
        for i in range(n):
            if comp[i] >= 0:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                j = stack.pop()
                for k in np.nonzero(adj[j])[0]:
                    if comp[k] < 0:
                        comp[k] = cid
                        stack.append(k)
            cid += 1

        is_evidence = members["reject_reason"].isin(_TABLE_EVIDENCE_REASONS).to_numpy()
        hf = members["height_frac"].to_numpy()
        for c in range(cid):
            m = comp == c
            n_ev = int(is_evidence[m].sum())
            if n_ev == 0:
                continue
            kill = m & ~is_evidence & (hf < _REGION_IMMUNE_HEIGHT_FRAC) & (
                (n_ev >= _REGION_MIN_EVIDENCE) | (hf < _REGION_WEAK_HEIGHT_FRAC)
            )
            df_candidates.loc[members.index[kill], "reject_reason"] = "table_region"

    # Absorb ragged-edge slivers: a candidate that sits directly beside a taller
    # accepted gutter (ragged-right of the left column, hanging indents of the
    # right column) is whitespace *belonging to* that gutter, not a separator.
    ok_idx = df_candidates.index[df_candidates["reject_reason"].isna()]
    for pn, grp in df_candidates.loc[ok_idx].groupby("page_number", sort=False):
        order = grp.sort_values(["gutter_height", "gutter_width"], ascending=False)
        kept: list = []
        for idx, c in order.iterrows():
            absorbed = False
            for k in kept:
                x_gap = max(k["gutter_x_left"] - c["gutter_x_right"],
                            c["gutter_x_left"] - k["gutter_x_right"])
                y_cover = (min(k["gutter_y_bottom"], c["gutter_y_bottom"])
                           - max(k["gutter_y_top"], c["gutter_y_top"]))
                if x_gap <= _ABSORB_X_EPS and y_cover >= _ABSORB_MIN_Y_COVER * c["gutter_height"]:
                    df_candidates.at[idx, "reject_reason"] = "absorbed_adjacent"
                    absorbed = True
                    break
            if not absorbed:
                kept.append(c)

    # Parallel-rectangle feature: candidates passing all other checks that share
    # a y-band on the same page (3-4 col layouts are fine; big groups = grids)
    df_candidates["n_parallel"] = 0
    ok = df_candidates["reject_reason"].isna()
    for pn, grp in df_candidates[ok].groupby("page_number", sort=False):
        y0 = grp["gutter_y_top"].to_numpy()
        y1 = grp["gutter_y_bottom"].to_numpy()
        h = y1 - y0
        ov = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
        min_h = np.minimum(h[:, None], h[None, :])
        parallel = (ov > 0.5 * min_h).sum(axis=1) - 1
        df_candidates.loc[grp.index, "n_parallel"] = parallel

    df_gutters = df_candidates[df_candidates["reject_reason"].isna()].copy()
    df_gutters = df_gutters.sort_values(
        ["page_number", "gutter_x_left", "gutter_y_top"], kind="mergesort"
    ).reset_index(drop=True)
    df_gutters["gutter_id"] = range(1, len(df_gutters) + 1)

    front = [
        "page_number", "gutter_id",
        "gutter_y_top", "gutter_y_bottom", "gutter_x_left", "gutter_x_right",
        "gutter_width", "gutter_height",
    ]
    df_gutters = df_gutters[front + [c for c in df_gutters.columns if c not in front]]
    df_gutters = df_gutters.drop(columns=["reject_reason"])

    return df_gutters, df_candidates

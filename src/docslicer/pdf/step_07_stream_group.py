"""
step_07_reading_order.py

# ==============================================================================
# NATIVE PDF STREAMING-ORDER ASSIGNMENT
# ==============================================================================
#
# Goal
# ----
# 1. Derive an approximate reading order for native PDFs from the PDF content
#    stream (`text_object_id`) instead of relying primarily on geometric
#    heuristics such as XY-cut.
#
# 2. Partition consecutive text objects into `stream_group_id`s representing
#    logical reading segments (typically lines or contiguous reading runs).
#
# 3. Leverage tagged-PDF metadata (`struct_group_id`, `table_id`, `textbox_id`)
#    when available to improve grouping accuracy.
#
# 4. Assign `line_id`s to all words based on the resulting streaming groups.
#
# Background
# ----------
# This module is only used for native PDFs.
#
# PDFium exposes text objects in the order they appear in a page's content
# stream (`text_object_id`, numbered 1..N per page). This native PDF streaming
# order generally follows the intended reading order much more closely than
# geometric layout heuristics, although occasional outliers exist (e.g. page
# labels, headers/footers, or other content emitted earlier or later in the
# content stream).
#
# Scanned PDFs are handled by the OCR pipeline, which derives reading order from
# the page image using layout analysis (e.g. gutter detection), and therefore do
# not use this algorithm.
#
# A later pipeline stage may reposition isolated streaming groups to produce the
# final human reading order.
# ==============================================================================

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.layout.line_merger import same_line_pairwise

""""
flowchart TD

    A([Start: Compare obj x with obj x+1])

    A --> T{Tagged PDF relationship?}

    T -->|Same struct_group_id| H[Continue current streaming group]
    T -->|Same table_id| H
    T -->|Different textbox_id| G[Increment group_id += 1]
    T -->|No tagged relationship| B{Same line?}

    B -->|Yes| O
    B -->|No| D{Y center decreases?}

    D -->|Yes| G
    D -->|No| E{Gap larger than jump threshold?}

    E -->|Yes| G
    E -->|No| F{Shifted left outside current group x-range?}

    F -->|Yes| G
    F -->|No| O

    O{Objects in between?}

    O -->|Yes| G
    O -->|No| H

    G --> J([Next object pair])
    H --> J
    J --> A

# TODO: This script currently doesn't work for RTL arabic text -> make left | right a param (requires regression testing)
"""


# ================================================================================
# STREAM GROUP DETECTION  (text_object_id path only)
# ================================================================================

_STREAM_GROUP_Y_JUMP: float = 50.0  # pt — forward jump that starts a new group
_Y_DECREASE_TOL: float = 5.0        # pt — upward jump must exceed this to count

# objects_between: two objects are "far apart" (and thus worth scanning between)
# when the gap on either axis reaches this many points; a third object counts as
# "in between" when more than this fraction of its area lies inside the pair's
# collective bbox.
_OBJECTS_BETWEEN_GAP: float = 15.0
_OBJECTS_BETWEEN_AREA_FRAC: float = 0.5

# shifted_left: obj_b lies entirely to the LEFT of the current streaming group's
# accumulated x-range — its right edge is at/left of the group's x_min (within
# this much overlap slack) → it has "shifted left" out of the group's x-range and
# begins a new group. An object that overlaps the group's x-span in any
# meaningful way instead widens the group and does NOT count as shifted.
_SHIFT_LEFT_TOL: float = 2.0


# ================================================================================
# OBJECT-LEVEL COLLAPSE
# ================================================================================
#
# The pairwise algorithm reasons about *text objects*, not words. A single
# PDFium text object (one content-stream Tj run) usually spans several words,
# so we first collapse ``df_words`` down to one row per ``(page_number,
# text_object_id)``. The object bbox is the union of its word bboxes; tagged
# fields (struct_group_id / table_id / textbox_id) are constant within a text
# object (they are resolved per ``(page, text_object_id)`` upstream), so we take
# the first value.
#
# Words with a null ``text_object_id`` (rare — PDFium could not attribute them
# to a content-stream object) have no known stream position. Each is given its
# own synthetic object (so they never merge with unrelated words) with a large
# id (>= _SYNTH_OBJ_BASE) that parks them, in their original row order, at the
# END of the page's stream — the least-disruptive spot, since it makes them
# trailing objects rather than injecting a bogus leading pair.
#   NOTE: where null-id words truly belong is a genuine open question — see the
#   caveats returned to the caller.
_SYNTH_OBJ_BASE: int = 1_000_000

# Columns carried through to the object level, if present on df_words.
_OBJ_TAG_COLS: tuple[str, ...] = (
    "struct_group_id",
    "table_id",
    "table_row_id",
    "textbox_id",
    "block_type",
)


def _object_key(df_words: pd.DataFrame) -> np.ndarray:
    """Return a null-safe int64 object key aligned to ``df_words`` rows.

    The key is the row's ``text_object_id`` where present; null ids get a unique
    synthetic id (``>= _SYNTH_OBJ_BASE``, in row order) so groupby keeps them
    separate instead of dropping or merging them, and so they sort to the end of
    their page. The result is deterministic for a given row order, so the same
    key can be recomputed later to merge object-level results back onto words.
    """
    obj_ids = df_words["text_object_id"].to_numpy(dtype=object)
    null_mask = df_words["text_object_id"].isna().to_numpy()
    if null_mask.any():
        obj_ids = obj_ids.copy()
        obj_ids[null_mask] = _SYNTH_OBJ_BASE + np.arange(null_mask.sum(), dtype=np.int64)
    return np.asarray(obj_ids, dtype=np.int64)


def _collapse_to_objects(df_words: pd.DataFrame) -> pd.DataFrame:
    """Collapse word rows to one row per ``(page_number, text_object_id)``.

    Returns a frame sorted by ``(page_number, text_object_id)`` — i.e. content
    stream order within each page — with a union bbox and the object's tagged
    fields. The bbox columns are ``x_left, y_top, x_right, y_bottom``. The
    ``text_object_id`` column holds the null-safe key from :func:`_object_key`.
    """
    required = ["page_number", "text_object_id", "x_left", "y_top", "x_right", "y_bottom"]
    missing = [c for c in required if c not in df_words.columns]
    if missing:
        raise KeyError(f"_collapse_to_objects: missing required columns: {missing}")

    df = df_words.copy()
    df["text_object_id"] = _object_key(df_words)

    agg: dict[str, tuple[str, str]] = {
        "x_left":   ("x_left",   "min"),
        "y_top":    ("y_top",    "min"),
        "x_right":  ("x_right",  "max"),
        "y_bottom": ("y_bottom", "max"),
    }
    for col in _OBJ_TAG_COLS:
        if col in df.columns:
            agg[col] = (col, "first")

    df_objs = (
        df.groupby(["page_number", "text_object_id"], sort=True, dropna=False)
          .agg(**agg)
          .reset_index()
    )
    return df_objs


# ================================================================================
# PAIRWISE FEATURES
# ================================================================================
#
# Each row of the returned frame is an adjacent pair (obj_a, obj_b) where obj_b
# is the immediately following text object *on the same page*. Features are
# fully vectorized. The last object of every page has no successor and so
# produces no pair row.
#
# NOT computed here (requires streaming state):
#   - shifted_left    : depends on the cumulative x-span of the *current*
#                       streaming group, so it can only be evaluated while
#                       streaming groups are being built, not as a static
#                       pairwise column.

def build_pair_features(df_words: pd.DataFrame) -> pd.DataFrame:
    """Compute the vectorized pairwise feature table for reading-order grouping.

    One row per adjacent same-page text-object pair, with columns:

        page_number   the page both objects live on
        obj_a, obj_b  text_object_id of the earlier / later object
        same_line     obj_b sits on the same text line as obj_a
        y_decreases   obj_b's y-center is above obj_a's (jumped back up the page)
        large_gap     forward y-center jump exceeds the group-break threshold
        same_struct   both objects share a (non-null) struct_group_id
        same_table    both objects share a (non-null) table_id
        new_textbox   the two objects have different textbox_id values
        objects_between  a third object sits inside the pair's collective bbox

    ``shifted_left`` is intentionally omitted (see the module notes above).
    """
    df_objs = _collapse_to_objects(df_words)
    arr = _pairwise_arrays(df_objs)

    # Pairs are (row i, row i+1). A pair is valid only when both rows are on the
    # same page, so we drop the boundary rows where the page changes.
    page = df_objs["page_number"].to_numpy()
    same_page = page[:-1] == page[1:]

    obj_id = df_objs["text_object_id"].to_numpy()
    feats = pd.DataFrame({
        "page_number": page[:-1][same_page],
        "obj_a":       obj_id[:-1][same_page],
        "obj_b":       obj_id[1:][same_page],
        **{name: col[same_page] for name, col in arr.items()},
    })
    return feats


# The vectorized pair features, in the order the group-walk consults them.
_VECTOR_FEATURE_COLS: tuple[str, ...] = (
    "same_line",
    "y_decreases",
    "large_gap",
    "same_struct",
    "same_table",
    "new_textbox",
    "objects_between",
)


def _pairwise_arrays(df_objs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute the vectorized pair features over *every* consecutive row pair.

    Returns one boolean array per feature in ``_VECTOR_FEATURE_COLS``, each of
    length ``len(df_objs) - 1`` and indexed by gap ``i`` (between row ``i`` and
    row ``i + 1``). Cross-page gaps are included here (callers mask them out as
    needed); ``objects_between`` already forces them ``False``.
    """
    n = len(df_objs)
    if n < 2:
        return {name: np.zeros(0, dtype=bool) for name in _VECTOR_FEATURE_COLS}

    a = df_objs.iloc[:-1].reset_index(drop=True)
    b = df_objs.iloc[1:].reset_index(drop=True)

    yt_a, yb_a = a["y_top"].to_numpy(float), a["y_bottom"].to_numpy(float)
    yt_b, yb_b = b["y_top"].to_numpy(float), b["y_bottom"].to_numpy(float)

    # Forward vertical delta of the center: positive = obj_b is further down.
    dy_center = (yt_b + yb_b) * 0.5 - (yt_a + yb_a) * 0.5

    return {
        "same_line":       same_line_pairwise(yt_a, yb_a, yt_b, yb_b),
        "y_decreases":     dy_center < -_Y_DECREASE_TOL,
        "large_gap":       dy_center > _STREAM_GROUP_Y_JUMP,
        "same_struct":     _same_nonnull(a, b, "struct_group_id"),
        "same_table":      _same_nonnull(a, b, "table_id"),
        "new_textbox":     _differs(a, b, "textbox_id"),
        "objects_between": _compute_objects_between(df_objs),
    }


def _compute_objects_between(df_objs: pd.DataFrame) -> np.ndarray:
    """Flag adjacent object pairs that have a third object sitting between them.

    Operates on the object-level frame (sorted by ``(page_number,
    text_object_id)``) and returns a boolean array of length ``len(df_objs) - 1``
    where entry ``i`` describes the gap between row ``i`` and row ``i + 1``.
    Cross-page gaps are always ``False``.

    A pair qualifies when BOTH:
      1. the two objects are "far apart" — the horizontal gap
         (``b.x_left - a.x_right``) or the vertical gap
         (``b.y_top - a.y_bottom``) is at least ``_OBJECTS_BETWEEN_GAP`` pt; and
      2. some *other* object on the page has more than
         ``_OBJECTS_BETWEEN_AREA_FRAC`` of its own area inside the pair's
         collective (union) bbox.

    Within each page this is a vectorized ``(pairs x objects)`` broadcast, so the
    transient cost is O(M²) in the object count M of the busiest page.
    """
    n = len(df_objs)
    out = np.zeros(max(n - 1, 0), dtype=bool)
    if n < 3:
        return out  # need at least one object besides a pair

    page = df_objs["page_number"].to_numpy()
    xl = df_objs["x_left"].to_numpy(float)
    yt = df_objs["y_top"].to_numpy(float)
    xr = df_objs["x_right"].to_numpy(float)
    yb = df_objs["y_bottom"].to_numpy(float)

    # Contiguous per-page blocks (rows are already grouped by page after the sort).
    change = np.flatnonzero(page[1:] != page[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [n]))

    for s, e in zip(starts, ends):
        m = e - s
        if m < 3:
            continue  # only the two pair members exist — nothing can be between

        pxl, pyt, pxr, pyb = xl[s:e], yt[s:e], xr[s:e], yb[s:e]

        # Collective bbox of each consecutive pair (p = 0 .. m-2).
        cxl = np.minimum(pxl[:-1], pxl[1:])
        cyt = np.minimum(pyt[:-1], pyt[1:])
        cxr = np.maximum(pxr[:-1], pxr[1:])
        cyb = np.maximum(pyb[:-1], pyb[1:])

        # Precondition: far apart on either axis.
        far = (
            (pxl[1:] - pxr[:-1] >= _OBJECTS_BETWEEN_GAP)
            | (pyt[1:] - pyb[:-1] >= _OBJECTS_BETWEEN_GAP)
        )

        # Intersection area of every object (columns) with every pair bbox (rows).
        inter_w = np.clip(
            np.minimum(cxr[:, None], pxr[None, :]) - np.maximum(cxl[:, None], pxl[None, :]),
            0.0, None,
        )
        inter_h = np.clip(
            np.minimum(cyb[:, None], pyb[None, :]) - np.maximum(cyt[:, None], pyt[None, :]),
            0.0, None,
        )
        inter_area = inter_w * inter_h

        obj_area = (pxr - pxl) * (pyb - pyt)
        obj_area_safe = np.where(obj_area > 0.0, obj_area, np.inf)  # zero-area → frac 0
        inside = inter_area / obj_area_safe[None, :] > _OBJECTS_BETWEEN_AREA_FRAC

        # A pair's two own members must not count as "between".
        rows = np.arange(m - 1)
        inside[rows, rows] = False       # obj_a (local index p)
        inside[rows, rows + 1] = False   # obj_b (local index p+1)

        out[s : s + (m - 1)] = far & inside.any(axis=1)

    return out


def _same_nonnull(a: pd.DataFrame, b: pd.DataFrame, col: str) -> np.ndarray:
    """True where a[col] == b[col] and neither side is null.

    Missing column → all False (the relationship cannot hold without the data).
    """
    if col not in a.columns or col not in b.columns:
        return np.zeros(len(a), dtype=bool)
    va, vb = a[col], b[col]
    return (va.notna() & vb.notna() & (va == vb)).to_numpy()


def _differs(a: pd.DataFrame, b: pd.DataFrame, col: str) -> np.ndarray:
    """True where a[col] != b[col], treating two nulls as equal (not a change).

    A transition null→value or value→null counts as a difference. Missing
    column → all False.
    """
    if col not in a.columns or col not in b.columns:
        return np.zeros(len(a), dtype=bool)
    va, vb = a[col], b[col]
    both_null = va.isna() & vb.isna()
    return (~both_null & (va != vb)).to_numpy()


# ================================================================================
# STREAMING-GROUP WALK
# ================================================================================
#
# The final step walks objects in content-stream order, deciding for each
# adjacent pair whether obj_b CONTINUES the current streaming group or STARTS a
# new one, per the flowchart at the top of this module. It is an O(objects)
# Python loop rather than a vectorized op because ``shifted_left`` depends on the
# group's *accumulated* x-span, which only exists once earlier continue/split
# decisions are known.
#
# The current group keeps a running x-span [x_min, x_max]: every object that
# continues the group widens it (union of x-edges). obj_b has "shifted left" when
# its left edge sits left of the group's accumulated x_min — e.g. a group grown
# to [100, 350] followed by an object at [20, 80] starts a new group.

# Per-word output columns added by assign_reading_order (all transition-into,
# keyed to obj_b) plus the running-state shifted_left.
_PAIR_FEATURE_COLS: tuple[str, ...] = (*_VECTOR_FEATURE_COLS, "shifted_left")


def _decide_new_group(arr: dict[str, np.ndarray], i: int, shifted_left: bool) -> bool:
    """Apply the flowchart to pair ``i``. True = start a new group; False = continue.

    ``arr`` holds the vectorized pair features; ``shifted_left`` is supplied by
    the caller because it depends on the current group's running x-span.
    """
    # Tagged-PDF relationship (highest priority, in flowchart order).
    if arr["same_struct"][i]:
        return False          # same struct_group_id → continue
    if arr["same_table"][i]:
        return False          # same table_id → continue
    if arr["new_textbox"][i]:
        return True           # different textbox_id → new group

    # No tagged relationship → geometry.
    if arr["same_line"][i]:
        return bool(arr["objects_between"][i])   # same line → only split if something's between
    if arr["y_decreases"][i]:
        return True
    if arr["large_gap"][i]:
        return True
    if shifted_left:
        return True
    return bool(arr["objects_between"][i])


def _walk_streaming_groups(
    df_objs: pd.DataFrame, arr: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Assign a stream_group_id to every object row and compute shifted_left.

    Returns ``(group_id, shifted_left)``, both length ``len(df_objs)``:
      - ``group_id``: int64, monotonically increasing; each page boundary starts
        a fresh group (there is no pair across pages).
      - ``shifted_left``: object array, ``None`` at every page-start object (no
        predecessor), else a Python bool.
    """
    n = len(df_objs)
    group_id = np.zeros(n, dtype=np.int64)
    shifted_left = np.empty(n, dtype=object)
    shifted_left[:] = None
    if n == 0:
        return group_id, shifted_left

    page = df_objs["page_number"].to_numpy()
    xl = df_objs["x_left"].to_numpy(float)
    xr = df_objs["x_right"].to_numpy(float)

    gid = 0
    cur_xmin = cur_xmax = 0.0
    for j in range(n):
        if j == 0 or page[j] != page[j - 1]:
            # First object on the page → new group, seed the running x-span.
            gid += 1
            cur_xmin, cur_xmax = xl[j], xr[j]
            group_id[j] = gid
            continue

        i = j - 1  # pair (obj i, obj j) → feature index i
        # Shifted left only if obj_b sits entirely left of the group's x-range
        # (its right edge is at/left of x_min); any real x-overlap would just
        # widen the group, so it does not count.
        sl = bool(xr[j] <= cur_xmin + _SHIFT_LEFT_TOL)
        shifted_left[j] = sl

        if _decide_new_group(arr, i, sl):
            gid += 1
            cur_xmin, cur_xmax = xl[j], xr[j]
        else:
            cur_xmin = min(cur_xmin, xl[j])
            cur_xmax = max(cur_xmax, xr[j])
        group_id[j] = gid

    return group_id, shifted_left


def assign_stream_group_id(df_words: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Assign ``stream_group_id`` to every word and attach its pair features.

    Collapses words to text objects, runs the vectorized pair analysis and the
    streaming-group walk, then broadcasts the per-object results back to words on
    ``(page_number, text_object_id)``. Added columns:

        stream_group_id  Int64 — the reading-order segment the word belongs to
        shifted_left        boolean — obj_b started left of the group's x-span
        <pair features>     boolean — same_line, y_decreases, large_gap,
                            same_struct, same_table, new_textbox, objects_between

    Every feature/shifted_left value describes the transition *into* the word's
    text object; the first object on each page has no predecessor, so those words
    get ``<NA>`` (a handy page-start marker). ``stream_group_id`` is always set.

    If ``debug`` is False (the default), the intermediate pair-feature columns
    (``shifted_left`` and ``<pair features>``) are omitted and only
    ``stream_group_id`` is added.
    """
    if df_words is None or df_words.empty:
        out = df_words.copy() if df_words is not None else pd.DataFrame()
        out["stream_group_id"] = pd.Series(dtype="Int64")
        if debug:
            for col in _PAIR_FEATURE_COLS:
                out[col] = pd.Series(dtype="boolean")
        return out

    df_objs = _collapse_to_objects(df_words)
    arr = _pairwise_arrays(df_objs)
    group_id, shifted_left = _walk_streaming_groups(df_objs, arr)

    # Build a per-object result keyed by the null-safe object key. Pair features
    # (length n-1, indexed by gap i) attach to obj_b (object i+1); page-start
    # objects have no incoming pair and stay None.
    n = len(df_objs)
    page = df_objs["page_number"].to_numpy()
    page_start = np.ones(n, dtype=bool)
    page_start[1:] = page[1:] != page[:-1]

    result = pd.DataFrame({
        "page_number": page,
        "__obj_key": df_objs["text_object_id"].to_numpy(),
        "stream_group_id": group_id,
        "shifted_left": shifted_left,
    })
    for name in _VECTOR_FEATURE_COLS:
        col = np.empty(n, dtype=object)
        col[:] = None
        col[1:] = arr[name]          # gap i → object i+1
        col[page_start] = None       # drop cross-page / first-object values
        result[name] = col

    if not debug:
        result = result[["page_number", "__obj_key", "stream_group_id"]]

    words = df_words.copy()
    words["__obj_key"] = _object_key(df_words)
    out = words.merge(result, on=["page_number", "__obj_key"], how="left").drop(columns="__obj_key")

    out["stream_group_id"] = out["stream_group_id"].astype("Int64")
    if debug:
        for col in _PAIR_FEATURE_COLS:
            out[col] = out[col].astype("boolean")

    return out #df_words
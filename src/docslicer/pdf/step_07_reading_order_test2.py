"""
step_07_reading_order.py

Assign line_id to every word in df_words.

Strategy:
  1. Check whether text_object_id is populated (PDF byte-stream order, most accurate).
  2. If YES  → sort horizontal words by (page_number, text_object_id), call assign_line_id.
  3. If NO   → run gutter detection (if not already done), sort via gutter-aware heuristic,
               call assign_line_id.
  4. Vertical words (TTB / BTT) are processed separately after horizontal words and receive
     line_ids offset above the highest horizontal line_id so they sort to the end of each page.

Public API:
    df_words = assign_reading_order(df_words, df_shapes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.layout.line_merger import assign_line_id, same_line
from .._utils.layout.reading_order import _sort_by_gutters, assign_vertical_line_ids


# ================================================================================
# STREAM GROUP DETECTION  (text_object_id path only)
# ================================================================================

_STREAM_GROUP_Y_JUMP: float = 50.0  # pt — forward jump that starts a new group
_MIN_BETWEEN_GAP:    float = 10.0   # pt — minimum gap before checking for interlopers


def _corridor_has_interloper(
    gap_x_left: float,
    gap_x_right: float,
    gap_y_top: float,
    gap_y_bottom: float,
    page_x_left: np.ndarray,
    page_x_right: np.ndarray,
    page_y_top: np.ndarray,
    page_y_bottom: np.ndarray,
    exclude_mask: np.ndarray,
) -> bool:
    """Return True if any non-excluded object overlaps the given corridor rect."""
    if gap_x_left >= gap_x_right or gap_y_top >= gap_y_bottom:
        return False
    return bool(
        (
            ~exclude_mask
            & (page_x_left  < gap_x_right)
            & (page_x_right > gap_x_left)
            & (page_y_top    < gap_y_bottom)
            & (page_y_bottom > gap_y_top)
        ).any()
    )


def _object_in_between(
    ax_left: float, ax_right: float, ay_top: float, ay_bottom: float,
    bx_left: float, bx_right: float, by_top: float, by_bottom: float,
    page_x_left: np.ndarray, page_x_right: np.ndarray,
    page_y_top: np.ndarray, page_y_bottom: np.ndarray,
    exclude_mask: np.ndarray,
    direction: str,
) -> bool:
    """
    Return True if any non-excluded page object sits in the gap corridor between A and B.

    horizontal: x corridor = space between inner x edges; y span = combined y range.
    vertical:   y corridor = space between inner y edges; x span = combined x range.

    Returns False immediately when the two objects touch or overlap on the relevant axis
    (no gap → nothing can be between them).
    """
    if direction == "horizontal":
        gap_x_left   = min(ax_right, bx_right)
        gap_x_right  = max(ax_left,  bx_left)
        if gap_x_right - gap_x_left <= _MIN_BETWEEN_GAP:
            return False
        gap_y_top    = min(ay_top,   by_top)
        gap_y_bottom = max(ay_bottom, by_bottom)
    else:
        gap_x_left   = min(ax_left,   bx_left)
        gap_x_right  = max(ax_right,  bx_right)
        gap_y_top    = min(ay_bottom, by_bottom)
        gap_y_bottom = max(ay_top,    by_top)
        if gap_y_bottom - gap_y_top <= _MIN_BETWEEN_GAP:
            return False
    return _corridor_has_interloper(
        gap_x_left, gap_x_right, gap_y_top, gap_y_bottom,
        page_x_left, page_x_right, page_y_top, page_y_bottom,
        exclude_mask,
    )


def _assign_stream_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stream_group_id to words using text_object_id as the atomic unit.

    Within each page, text objects are visited in PDF stream order
    (text_object_id ascending).  A new stream group starts when consecutive
    objects A → B show any of:

      Same line (same_line returns True):
        • Another object sits in the horizontal corridor between A and B.

      Different lines:
        • B's y_center is lower than A's (stream jumped back up the page).
        • B's y_center jumped forward by more than _STREAM_GROUP_Y_JUMP.
        • B sits entirely to the LEFT of the running group x range.
        • Another object sits in the vertical corridor between A and B.

    The group x range widens with every object added.
    group_id is globally unique across the document (cumulative across pages).
    """
    df = df.copy()

    # Vectorised bbox aggregation: one row per (page, text_object_id)
    obj_stats = (
        df.groupby(["page_number", "text_object_id"], sort=False)
        .agg(
            y_top   =("y_top",    "min"),
            y_bottom=("y_bottom", "max"),
            x_left  =("x_left",  "min"),
            x_right =("x_right", "max"),
        )
        .reset_index()
    )
    obj_stats["y_center"] = (obj_stats["y_top"] + obj_stats["y_bottom"]) / 2.0
    obj_stats = (
        obj_stats
        .sort_values(["page_number", "text_object_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Pre-build per-page numpy arrays for vectorised corridor checks
    page_arrays: dict = {}
    for page_num, pg in obj_stats.groupby("page_number", sort=False):
        page_arrays[page_num] = {
            "x_left":         pg["x_left"].to_numpy(dtype=float),
            "x_right":        pg["x_right"].to_numpy(dtype=float),
            "y_top":          pg["y_top"].to_numpy(dtype=float),
            "y_bottom":       pg["y_bottom"].to_numpy(dtype=float),
            "text_object_id": pg["text_object_id"].to_numpy(),
        }

    group_ids:      list[int] = []
    group_triggers: list[str] = []
    current_group   = 0
    current_trigger = "first_row"
    group_x_left    = 0.0
    group_x_right   = 0.0

    prev_page:          object       = None
    prev_yc:            float | None = None
    prev_y_top:         float | None = None
    prev_y_bottom:      float | None = None
    prev_x_left:        float | None = None
    prev_x_right:       float | None = None
    prev_text_obj_id:   object       = None

    for row in obj_stats.itertuples(index=False):
        new_group = False
        trigger   = ""

        if prev_page is None or row.page_number != prev_page:
            new_group = True
            trigger   = "new_page" if prev_page is not None else "first_row"
        else:
            arrs = page_arrays[row.page_number]
            tids = arrs["text_object_id"]
            exclude_mask = (tids == row.text_object_id) | (tids == prev_text_obj_id)

            if same_line(prev_y_top, prev_y_bottom, row.y_top, row.y_bottom):  # type: ignore[arg-type]
                if _object_in_between(
                    prev_x_left, prev_x_right, prev_y_top, prev_y_bottom,  # type: ignore[arg-type]
                    row.x_left, row.x_right, row.y_top, row.y_bottom,
                    arrs["x_left"], arrs["x_right"], arrs["y_top"], arrs["y_bottom"],
                    exclude_mask, "horizontal",
                ):
                    new_group = True
                    trigger   = "obj_between_horiz"
            else:
                delta = row.y_center - prev_yc  # type: ignore[operator]
                if delta < 0:
                    new_group = True
                    trigger   = "y_backward"
                elif delta > _STREAM_GROUP_Y_JUMP:
                    new_group = True
                    trigger   = "y_gap"
                elif row.x_right < group_x_left:
                    new_group = True
                    trigger   = "x_left"
                elif _object_in_between(
                    prev_x_left, prev_x_right, prev_y_top, prev_y_bottom,  # type: ignore[arg-type]
                    row.x_left, row.x_right, row.y_top, row.y_bottom,
                    arrs["x_left"], arrs["x_right"], arrs["y_top"], arrs["y_bottom"],
                    exclude_mask, "vertical",
                ):
                    new_group = True
                    trigger   = "obj_between_vert"

        if new_group:
            current_group   += 1
            current_trigger  = trigger
            group_x_left     = row.x_left
            group_x_right    = row.x_right
        else:
            group_x_left  = min(group_x_left,  row.x_left)
            group_x_right = max(group_x_right, row.x_right)

        group_ids.append(current_group)
        group_triggers.append(current_trigger)

        prev_yc          = row.y_center
        prev_y_top       = row.y_top
        prev_y_bottom    = row.y_bottom
        prev_x_left      = row.x_left
        prev_x_right     = row.x_right
        prev_page        = row.page_number
        prev_text_obj_id = row.text_object_id

    obj_stats["stream_group_id"]      = group_ids
    obj_stats["stream_group_trigger"] = group_triggers
    obj_stats["line_y_center"]        = obj_stats["y_center"].round(2)

    # Map back to words via (page_number, text_object_id)
    mapping = obj_stats.set_index(["page_number", "text_object_id"])[
        ["stream_group_id", "stream_group_trigger", "line_y_center"]
    ]
    idx = pd.MultiIndex.from_arrays([df["page_number"], df["text_object_id"]])
    df["stream_group_id"]      = mapping["stream_group_id"].reindex(idx).values
    df["stream_group_trigger"] = mapping["stream_group_trigger"].reindex(idx).values
    df["line_y_center"]        = mapping["line_y_center"].reindex(idx).values

    return df


# ================================================================================
# STREAM GROUP RESHUFFLING  (text_object_id path only)
# ================================================================================

def _reshuffle_stream_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder stream groups using spatial layout analysis — no heuristic KPIs.

    Algorithm (per page):
    1. Compute total BBOX for each stream group.
    2. Greedily merge groups with y-overlap into horizontal buckets (transitive:
       if A∩B and B∩C then A, B, C share a bucket even if A∩C = ∅).
    3. Sort horizontal buckets top-to-bottom by their y-center.
    4. Within each horizontal bucket, greedily merge groups with x-overlap into
       vertical buckets, then sort those left-to-right by x-center.
    5. Within each vertical bucket, sort groups top-to-bottom by y-center.

    Vertical-text groups are appended after all horizontal groups in their
    original stream order.
    """
    # ── BBOX per stream group ─────────────────────────────────────────────────
    grp_bbox = (
        df.groupby(["page_number", "stream_group_id"])
        .agg(
            x_left       =("x_left",   "min"),
            x_right      =("x_right",  "max"),
            y_top        =("y_top",    "min"),
            y_bottom     =("y_bottom", "max"),
            first_line_id=("line_id",  "min"),
        )
        .reset_index()
    )
    grp_bbox["y_center"] = (grp_bbox["y_top"] + grp_bbox["y_bottom"]) / 2
    grp_bbox["x_center"] = (grp_bbox["x_left"] + grp_bbox["x_right"]) / 2

    # ── Mark vertical-text groups (keep in original order at end of page) ─────
    if "text_orientation" in df.columns:
        vert_groups: set = set(
            df.loc[df["text_orientation"].isin(["TTB", "BTT"]), "stream_group_id"].unique()
        )
    else:
        vert_groups = set()

    # ── Per-page spatial sort ─────────────────────────────────────────────────
    new_order_rows: list[dict] = []

    for _, pg in grp_bbox.groupby("page_number"):
        horiz_pg = pg[~pg["stream_group_id"].isin(vert_groups)].sort_values("y_top").reset_index(drop=True)
        vert_pg  = pg[pg["stream_group_id"].isin(vert_groups)].sort_values("first_line_id")

        # Step 2: horizontal buckets via greedy y-overlap interval merge
        h_buckets: list[list] = []
        cur_bucket: list = []
        cur_max_y = float("-inf")

        for row in horiz_pg.itertuples(index=False):
            if row.y_top < cur_max_y:
                cur_bucket.append(row)
                cur_max_y = max(cur_max_y, row.y_bottom)
            else:
                if cur_bucket:
                    h_buckets.append(cur_bucket)
                cur_bucket = [row]
                cur_max_y = row.y_bottom
        if cur_bucket:
            h_buckets.append(cur_bucket)

        # Step 3: sort h_buckets top-to-bottom by bucket y_center
        h_buckets.sort(key=lambda b: sum(r.y_center for r in b) / len(b))

        rank = 0
        for h_bucket_idx, h_bucket in enumerate(h_buckets):
            # Step 4: vertical buckets via greedy x-overlap interval merge
            h_sorted = sorted(h_bucket, key=lambda r: r.x_left)
            v_buckets: list[list] = []
            cur_v: list = []
            cur_max_x = float("-inf")

            for row in h_sorted:
                if row.x_left < cur_max_x:
                    cur_v.append(row)
                    cur_max_x = max(cur_max_x, row.x_right)
                else:
                    if cur_v:
                        v_buckets.append(cur_v)
                    cur_v = [row]
                    cur_max_x = row.x_right
            if cur_v:
                v_buckets.append(cur_v)

            # Sort v_buckets left-to-right by bucket x_center
            v_buckets.sort(key=lambda b: sum(r.x_center for r in b) / len(b))

            for v_bucket in v_buckets:
                # Step 5: within each vertical bucket, sort top-to-bottom
                for row in sorted(v_bucket, key=lambda r: r.y_center):
                    new_order_rows.append({"stream_group_id": int(row.stream_group_id), "new_group_rank": rank, "h_bucket_id": h_bucket_idx})
                    rank += 1

        # Vertical groups at end in original stream order
        for row in vert_pg.itertuples(index=False):
            new_order_rows.append({"stream_group_id": int(row.stream_group_id), "new_group_rank": rank, "h_bucket_id": -1})
            rank += 1

    new_order_df = pd.DataFrame(new_order_rows)

    # ── One row per distinct (page, group, line) — preserves stream order within groups ──
    per_line = df[["page_number", "stream_group_id", "line_id"]].drop_duplicates()
    per_line = per_line.assign(
        within_group_rank=(
            per_line.groupby(["page_number", "stream_group_id"])["line_id"]
            .rank(method="dense")
            .astype(int)
        )
    )

    # ── Assign final sequential line_id ──────────────────────────────────────
    per_line = (
        per_line
        .merge(new_order_df, on="stream_group_id", how="left")
        .sort_values(["page_number", "new_group_rank", "within_group_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    per_line["new_line_id"] = range(1, len(per_line) + 1)

    line_id_map  = dict(zip(per_line["line_id"], per_line["new_line_id"]))
    h_bucket_map = dict(zip(per_line["line_id"], per_line["h_bucket_id"]))
    out = df.copy()
    out["h_bucket_id"] = out["line_id"].map(h_bucket_map)
    out["line_id"]     = out["line_id"].map(line_id_map)
    return out


# ================================================================================
# PUBLIC API
# ================================================================================

def assign_reading_order(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add line_id to every word in df_words.

    Processes page by page so vertical words on each page receive IDs
    immediately after that page's horizontal words, keeping the global
    line_id sequence logically contiguous across the document.

    Parameters
    ----------
    df_words  : word-level DataFrame (output of step_01 through step_06).
    df_shapes : shape DataFrame, only needed if gutter detection has not yet run.

    Returns
    -------
    df_words with line_id (and center_bucket) populated.
    """
    if df_words is None or df_words.empty:
        out = (df_words.copy() if df_words is not None else pd.DataFrame())
        out["line_id"] = pd.Series(dtype="Int64")
        return out

    df = df_words.copy()

    # Determine whether PDF stream order is available
    has_text_object_id = (
        "text_object_id" in df.columns
        and df["text_object_id"].notna().any()
    )

    # Ensure gutter columns are present for the fallback path (do once, up front)
    if not has_text_object_id:
        gutter_cols_present = (
            "gutter_id_right" in df.columns
            and df["gutter_id_right"].notna().any()
        )
        if not gutter_cols_present:
            from .._utils.layout.gutter_detector import detect_and_annotate_gutters
            df, _, _ = detect_and_annotate_gutters(
                df, df_shapes if df_shapes is not None else pd.DataFrame()
            )

    # Split horizontal vs vertical
    if "text_orientation" in df.columns:
        vert_mask = df["text_orientation"].isin(["TTB", "BTT"])
    else:
        vert_mask = pd.Series(False, index=df.index)

    # ── Process page by page so vertical IDs slot right after horizontal on same page ──
    pages = sorted(df["page_number"].unique())
    horiz_pages: list[pd.DataFrame] = []
    vert_pages:  list[pd.DataFrame] = []
    running_line = 0

    for page_num in pages:
        page_mask = df["page_number"] == page_num
        df_h = df[page_mask & ~vert_mask].copy()
        df_v = df[page_mask & vert_mask].copy()

        # -- Horizontal --
        if not df_h.empty:
            if has_text_object_id:
                df_h = df_h.sort_values("text_object_id", kind="mergesort").reset_index(drop=True)
            else:
                df_h = _sort_by_gutters(df_h)

            df_h = assign_line_id(df_h)
            df_h["line_id"] = df_h["line_id"] + running_line
            running_line = int(df_h["line_id"].max())

        horiz_pages.append(df_h)

        # -- Vertical (offset above this page's horizontal ceiling) --
        if not df_v.empty:
            df_v = assign_vertical_line_ids(df_v, line_id_offset=running_line)
            running_line = int(df_v["line_id"].max())

        vert_pages.append(df_v)

    # ── Recombine ───────────────────────────────────────────────────────────────
    all_parts = horiz_pages + vert_pages
    non_empty = [p for p in all_parts if not p.empty]
    if not non_empty:
        df["line_id"] = pd.Series(dtype="Int64")
        return df

    result = (
        pd.concat(non_empty, ignore_index=True)
        .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
        .reset_index(drop=True)
    )

    if has_text_object_id:
        result = _assign_stream_groups(result)
        result = _reshuffle_stream_groups(result)

    # TODO: slide reading order
    # When page_format is a slide format (SLIDE_16_9, SLIDE_4_3, US_LETTER_LANDSCAPE etc.),
    # the current stream-group reshuffling is not sufficient. Slide content is laid out as a
    # collection of independent text boxes rather than flowing text, so stream order is
    # essentially arbitrary. A slide-aware pass should:
    #   1. Detect slide pages via page_format metadata (available on df_words or passed in).
    #   2. Classify the slide layout: a few columns read top-to-bottom, a few rows read
    #      left-to-right, or a mixed grid (e.g. the IPO workstreams 6-column table or the
    #      AZ 2×2 stats boxes). Layout type could be inferred from the spatial distribution
    #      of stream_group bounding boxes on the page.
    #   3. For column-dominant layouts: sort stream_groups by (x_band, y_center) so groups
    #      in the leftmost x-band come first, top to bottom, then the next band, etc.
    #   4. For row-dominant layouts: sort by (y_band, x_center).
    #   5. For mixed/grid layouts: assign (row_band, col_band) to each group and sort by
    #      (row_band, col_band).
    #   Bands can be derived from clustering the group centroids (e.g. simple gap-based
    #   1D clustering on x or y, similar to how gutter_detector finds column gaps).
    #   This pass should run instead of (not in addition to) the current reshuffle for
    #   slide pages, since the stream-group backbone assumption breaks down entirely.
    #   Note: pptx/step_06_line_builder.py has some early logic for slide reading order
    #   that is not optimal yet — the slide layout detection here could potentially be
    #   extracted into a shared utility and reused there.

    return result

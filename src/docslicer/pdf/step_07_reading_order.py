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

import pandas as pd

from .._utils.line_merger import assign_line_id


# ================================================================================
# GUTTER-BASED SORT  (fallback for PDFs without text_object_id)
# ================================================================================

def _sort_by_gutters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort words into correct reading order for mixed single/multi-column pages.

    For singlecol pages a plain (y_top, x_left) sort is sufficient.
    For multicol pages, words in the same gutter zone must be grouped together
    before sorting by y so col-2 text doesn't end up below trailing singlecol text.

    zone_y per word:
      - Singlecol words                   → zone_y = y_top  (their own)
      - Col-1 multicol words              → zone_y = min(y_top) of their gutter zone
      - Col-2+ words                      → zone_y inherited from col-1 via gutter chain
    """
    if df.empty:
        return df

    has_right = "gutter_id_right" in df.columns
    has_left  = "gutter_id_left"  in df.columns
    has_rc    = "reading_column"  in df.columns

    if not has_right or df["gutter_id_right"].isna().all():
        return df.sort_values(
            ["page_number", "y_top", "x_left"], kind="mergesort"
        ).reset_index(drop=True)

    df = df.copy()
    rc = df["reading_column"].fillna(1).astype(int) if has_rc else pd.Series(1, index=df.index)

    df["_zone_y"] = df["y_top"].astype(float)
    df["_rc"]     = rc

    for page in df["page_number"].unique():
        pm = df["page_number"] == page

        col1_gutter_mask = pm & df["gutter_id_right"].notna() & (rc == 1)
        if not col1_gutter_mask.any():
            continue

        zone_y: dict = (
            df.loc[col1_gutter_mask]
            .groupby("gutter_id_right")["y_top"]
            .min()
            .to_dict()
        )

        # Propagate zone_y through 3+ column chains
        if has_left:
            chain_mask = pm & df["gutter_id_left"].notna() & df["gutter_id_right"].notna()
            if chain_mask.any():
                chain_pairs = (
                    df.loc[chain_mask, ["gutter_id_left", "gutter_id_right"]]
                    .drop_duplicates()
                )
                chain_pairs = chain_pairs[chain_pairs["gutter_id_left"].isin(zone_y)]
                if not chain_pairs.empty:
                    chain_pairs = chain_pairs.copy()
                    chain_pairs["_inherited"] = chain_pairs["gutter_id_left"].map(zone_y)
                    min_inherited = chain_pairs.groupby("gutter_id_right")["_inherited"].min()
                    for G_right, inherited in min_inherited.items():
                        zone_y[G_right] = min(inherited, zone_y.get(G_right, inherited))

        for G, zy in zone_y.items():
            right_mask = pm & (df["gutter_id_right"] == G)
            if right_mask.any():
                df.loc[right_mask, "_zone_y"] = zy

            if has_left:
                left_mask = pm & (df["gutter_id_left"] == G)
                if left_mask.any():
                    df.loc[left_mask, "_zone_y"] = zy

    result = (
        df.sort_values(
            ["page_number", "_zone_y", "_rc", "y_top", "x_left"],
            kind="mergesort",
        )
        .drop(columns=["_zone_y", "_rc"])
        .reset_index(drop=True)
    )
    return result


# ================================================================================
# VERTICAL TEXT  (TTB / BTT)
# ================================================================================

def _assign_vertical_line_ids(df_vert: pd.DataFrame, line_id_offset: int) -> pd.DataFrame:
    """
    Assign line_ids to vertical (TTB/BTT) words using a coordinate-swap trick.

    Words sharing the same x-band (a column of rotated text) become a "line".
    Swap x↔y so assign_line_id groups by x-proximity, then swap back.
    All IDs are offset above the horizontal line_id ceiling.
    """
    if df_vert.empty:
        df_vert = df_vert.copy()
        df_vert["line_id"] = pd.Series(dtype="Int64")
        return df_vert

    df = df_vert.copy()

    orig_xl = df["x_left"].to_numpy(dtype=float)
    orig_xr = df["x_right"].to_numpy(dtype=float)
    orig_yt = df["y_top"].to_numpy(dtype=float)
    orig_yb = df["y_bottom"].to_numpy(dtype=float)

    df["y_top"]    = orig_xl
    df["y_bottom"] = orig_xr
    df["x_left"]   = orig_yt
    df["x_right"]  = orig_yb

    # BTT: negate x so ascending sort yields bottom→top order
    df["_x_left_pre_neg"]  = df["x_left"]
    df["_x_right_pre_neg"] = df["x_right"]
    if "text_orientation" in df.columns:
        btt_mask = df["text_orientation"] == "BTT"
        if btt_mask.any():
            btt_xl = df.loc[btt_mask, "x_left"].to_numpy(dtype=float)
            btt_xr = df.loc[btt_mask, "x_right"].to_numpy(dtype=float)
            df.loc[btt_mask, "x_left"]  = -btt_xr
            df.loc[btt_mask, "x_right"] = -btt_xl

    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)
    df = assign_line_id(df)

    # Restore coordinates
    df["x_left"]  = df["_x_left_pre_neg"]
    df["x_right"] = df["_x_right_pre_neg"]
    df.drop(columns=["_x_left_pre_neg", "_x_right_pre_neg"], inplace=True)

    cur_xl = df["x_left"].to_numpy(dtype=float)
    cur_xr = df["x_right"].to_numpy(dtype=float)
    cur_yt = df["y_top"].to_numpy(dtype=float)
    cur_yb = df["y_bottom"].to_numpy(dtype=float)
    df["x_left"]   = cur_yt
    df["x_right"]  = cur_yb
    df["y_top"]    = cur_xl
    df["y_bottom"] = cur_xr

    df["line_id"] = df["line_id"] + line_id_offset
    return df


# ================================================================================
# STREAM GROUP DETECTION  (text_object_id path only)
# ================================================================================

_STREAM_GROUP_Y_JUMP: float = 200.0  # pt — forward jump that starts a new group


def _assign_stream_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stream_group_id to words after text_object_id-based line assignment.

    Within each page, line_ids are in PDF stream order.  Most of the time that
    matches top-to-bottom reading order, but artifacts (page numbers, running
    titles, footnotes) can appear anywhere in the stream.

    A new group starts when moving from one line_id to the next (in stream order):
      - y_center decreases  (stream jumped back up the page)
      - y_center increases by more than _STREAM_GROUP_Y_JUMP  (large spatial gap)

    group_id is globally unique across the document (cumulative across pages).
    """
    df = df.copy()

    # Per-word y center, then mean per (page, line_id)
    word_yc = (df["y_top"] + df["y_bottom"]) / 2.0
    line_yc = (
        word_yc.groupby([df["page_number"], df["line_id"]])
        .mean()
        .rename("y_center")
        .reset_index()
    )

    # Sort by (page, line_id) — line_id already encodes stream order within each page
    line_yc = line_yc.sort_values(["page_number", "line_id"], kind="mergesort").reset_index(drop=True)

    # Detect breaks vectorized
    prev_yc   = line_yc["y_center"].shift(1)
    prev_page = line_yc["page_number"].shift(1)
    delta     = line_yc["y_center"] - prev_yc

    is_break = (
        line_yc["page_number"].ne(prev_page)   # new page
        | prev_yc.isna()                        # first row overall
        | delta.lt(0)                           # y went backwards
        | delta.gt(_STREAM_GROUP_Y_JUMP)        # large forward gap
    )

    line_yc["stream_group_id"] = is_break.cumsum().astype("int64")
    line_yc["line_y_center"]   = line_yc["y_center"].round(2)

    # Map back to words
    mapping = line_yc.set_index(["page_number", "line_id"])[["stream_group_id", "line_y_center"]]
    idx = pd.MultiIndex.from_arrays([df["page_number"], df["line_id"]])
    df["stream_group_id"] = mapping["stream_group_id"].reindex(idx).values
    df["line_y_center"]   = mapping["line_y_center"].reindex(idx).values

    return df


# ================================================================================
# STREAM GROUP RESHUFFLING  (text_object_id path only)
# ================================================================================

_RESHUFFLE_MAX_LINES = 3      # groups with ≤ this many distinct line_ids are candidates
_RESHUFFLE_MIN_JUMP  = 100.0  # pt — min absolute y-jump from prior group to trigger reshuffle


def _reshuffle_stream_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reposition small stream groups that make large jumps (page numbers, running
    headers/footers) into their correct spatial position among the backbone of
    large groups (columns, body text) which stay in stream order.

    Rules
    -----
    - Candidate: ≤ _RESHUFFLE_MAX_LINES distinct line_ids AND jump from prior
      group's last line y_center > _RESHUFFLE_MIN_JUMP pt (forward or backward).
    - Candidates are inserted into the keep-group backbone ordered by their first
      line's y_center.  Tiebreaker: smaller line_id first.
    - After reshuffling, line_id is reindexed sequentially across the document.
    """
    # ── One row per distinct line_id in each group ────────────────────────────
    per_line = (
        df.groupby(["page_number", "stream_group_id", "line_id"])["line_y_center"]
        .first()
        .reset_index()
        .sort_values(["page_number", "stream_group_id", "line_id"])
        .reset_index(drop=True)
    )

    # ── Mark groups that contain any vertical word (never reshuffle those) ────
    if "text_orientation" in df.columns:
        vert_groups = (
            df.loc[df["text_orientation"].isin(["TTB", "BTT"]), "stream_group_id"]
            .unique()
        )
        vert_group_set: set = set(vert_groups)
    else:
        vert_group_set = set()

    # ── Per-group stats ───────────────────────────────────────────────────────
    grp = (
        per_line
        .groupby(["page_number", "stream_group_id"], sort=True)
        .agg(
            n_lines      =("line_id",       "nunique"),
            first_line_y =("line_y_center", "first"),
            last_line_y  =("line_y_center", "last"),
            first_line_id=("line_id",       "min"),
        )
        .reset_index()
        .sort_values(["page_number", "stream_group_id"])
        .reset_index(drop=True)
    )

    grp["prev_last_y"] = grp.groupby("page_number")["last_line_y"].shift(1)
    grp["jump"]        = grp["first_line_y"] - grp["prev_last_y"]
    grp["is_reshuffle"] = (
        (grp["n_lines"] <= _RESHUFFLE_MAX_LINES)
        & (grp["jump"].abs() > _RESHUFFLE_MIN_JUMP)
        & grp["prev_last_y"].notna()   # never reshuffle the first group on a page
        & ~grp["stream_group_id"].isin(vert_group_set)   # never reshuffle vertical text groups
    )

    # ── Per page: merge candidates into backbone by first_line_y ──────────────
    new_order_rows: list[dict] = []
    for _, pg in grp.groupby("page_number"):
        pg = pg.reset_index(drop=True)

        keep_list = pg.loc[~pg["is_reshuffle"], ["stream_group_id", "first_line_y"]].values.tolist()
        resh_list = (
            pg[pg["is_reshuffle"]]
            .sort_values(["first_line_y", "first_line_id"])
            [["stream_group_id", "first_line_y"]]
            .values.tolist()
        )

        final: list[int] = []
        ri = 0
        for k_sgid, k_fy in keep_list:
            while ri < len(resh_list) and resh_list[ri][1] <= k_fy:
                final.append(int(resh_list[ri][0]))
                ri += 1
            final.append(int(k_sgid))
        while ri < len(resh_list):
            final.append(int(resh_list[ri][0]))
            ri += 1

        for rank, sgid in enumerate(final):
            new_order_rows.append({"stream_group_id": sgid, "new_group_rank": rank})  # noqa: PERF401

    new_order_df = pd.DataFrame(new_order_rows)

    # ── Within-group line rank (preserves stream order inside each group) ─────
    per_line["within_group_rank"] = (
        per_line.groupby(["page_number", "stream_group_id"])["line_id"]
        .rank(method="dense")
        .astype(int)
    )

    # ── Sort all lines into final order and assign new sequential line_id ──────
    per_line = (
        per_line
        .merge(new_order_df, on="stream_group_id", how="left")
        .sort_values(
            ["page_number", "new_group_rank", "within_group_rank"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    per_line["new_line_id"] = range(1, len(per_line) + 1)

    line_id_map = dict(zip(per_line["line_id"], per_line["new_line_id"]))
    out = df.copy()
    out["line_id"] = out["line_id"].map(line_id_map)
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
            from ._utils.gutter_detector import detect_and_annotate_gutters
            df, _, _ = detect_and_annotate_gutters(df, df_shapes or pd.DataFrame())

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
            df_v = _assign_vertical_line_ids(df_v, line_id_offset=running_line)
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

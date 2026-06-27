"""
step_07_gutter_detector.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..._utils.text_utils import _CURRENCY_SYM_CLASS, is_list_marker

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

_Y_TOP_SLIDING_WINDOW: float = 5.0  # pt
_MIN_GAP_WIDTH: float = 9.2  # pt
_MIN_PAGE_MIN_GAP: float = 100.0  # pt
_TEXT_PADDING: float = 0.0            # pt
_MIN_GUTTER_CANDIDATE_OVERLAP: float = 3.0  # pt - require min 3pt overlap to maintain an existing gutter_candidate_id (prevent destroying a series on accidental contact)
_MIN_GUTTER_LINE_KILL_OVERLAP: float = 5.0  # pt - horizontal line must overlap a gutter by at least this much to kill it (prevent marginal line touches from terminating a gutter)
_GUTTER_LINE_Y_PADDING: float = 7.0  # pt - vertical padding applied above/below a promoted gutter when checking for intersecting horizontal lines
_MIN_GUTTER_HEIGHT: float = 50.0  # pt
_MIN_INTERNAL_GAPS: int = 3 # how many internal gaps does a gutter_candidate_id need to have to be a gutter
_MAX_INTERNAL_GAP_DENSITY: int = 3 # those _MIN_INTERNAL_GAPS need to come from gutter_candidate_id with <= 4 internal gaps, otherwise if those gaps only exist within high density areas, its a table
_MAX_GUTTER_WINDOW_Y_GAP: float = 40.0  # pt - if the y distance between two consecutive sliding windows exceeds this, kill all active gutters (large vertical gap = new layout region)
_EDGE_WINDOW_Y_GAP_FACTOR: float = 1.4  # if the first/last window of a gutter_candidate_id is page_left/page_right and its y-distance to the adjacent window exceeds this multiple of the median gap, eject it
_GUTTER_X_SNAP_EPS: float = 0.5        # pt - epsilon for floating-point x-coordinate comparisons at gutter edges
_EXPAND_X_EPS: float = 0.5            # pt - tolerance for x-edge comparisons during expansion/merge
_MERGE_MIN_X_INTERSECTION: float = 5.0  # pt - minimum x-range intersection for two gutter segments to be mergeable



# =======================================================================================================================
# Helper Functions
# =======================================================================================================================

# ------------------------------
# Add sliding windows
# ------------------------------

def add_sliding_windows(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - sliding_window: lowest y_top of the current bucket (per page)
      - sliding_window_id: 1..N counter across the entire document (unique globally)

    Bucket rule (per page, sorted by y_top ascending):
      Start a bucket at the first row's y_top (bucket_start).
      Keep adding rows until a row with y_top > bucket_start + _Y_TOP_SLIDING_WINDOW.
      That row starts the next bucket (with its own bucket_start).

    Vectorized implementation using groupby + shift + cumsum.
    """
    if words_df is None or words_df.empty:
        return words_df

    df = words_df.copy()

    required = {"page_number", "y_top"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Stable sort so "start from 1st row (lowest y_top)" is well-defined per page
    # Using word_id as a tie-breaker if present.
    sort_cols = ["page_number", "y_top"]
    if "word_id" in df.columns:
        sort_cols.append("word_id")

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Per-page bucket assignment using bucket_start tracking.
    # A pairwise shift comparison (y[i] - y[i-1]) is WRONG here because it
    # compares to the previous row rather than to the bucket start — rows like
    # 651, 655, 659 with window=5 would never split (each gap=4) even though
    # 659 > 651+5.  The correct rule requires remembering bucket_start.
    def _new_bucket_flags(y_series: pd.Series) -> pd.Series:
        y = y_series.to_numpy(dtype=np.float64)
        flags = np.zeros(len(y), dtype=bool)
        if len(y):
            flags[0] = True
            bucket_start = y[0]
            for i in range(1, len(y)):
                if y[i] > bucket_start + _Y_TOP_SLIDING_WINDOW:
                    flags[i] = True
                    bucket_start = y[i]
        return pd.Series(flags, index=y_series.index, dtype=bool)

    is_new_bucket = df.groupby("page_number", sort=False, group_keys=False).apply(
        lambda g: _new_bucket_flags(g["y_top"])
    )

    # Global cumulative bucket id (1-based), unique across entire document
    df["sliding_window_id"] = is_new_bucket.cumsum().astype("int64")

    # "sliding_window" is the lowest y_top of the bucket (i.e., min y_top within bucket)
    df["sliding_window"] = df.groupby(
        ["page_number", "sliding_window_id"], sort=False
    )["y_top"].transform("min")

    return df


# ------------------------------
# Add page x bounds
# ------------------------------

def add_page_x_bounds(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - x_page_min: per-page min x_left (computed on a filtered subset)
      - x_page_max: per-page max x_right (computed on a filtered subset)

    Rules:
      - Only use rows where text_orientation == "LTR"
      - Exclude the first and last sliding_window_id per page
      - Bounds are: min(x_left), max(x_right) over the remaining rows
      - Broadcast bounds back to all rows in that page

    Notes:
      - Requires columns: page_number, x_left, x_right, sliding_window_id, text_orientation
      - If a page has no eligible rows after filtering, bounds remain NA for that page.
    """
    if words_df is None or words_df.empty:
        return words_df

    df = words_df.copy()

    required = {"page_number", "x_left", "x_right", "sliding_window_id", "text_orientation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Per-page first/last sliding window id
    sw_min = df.groupby("page_number", sort=False)["sliding_window_id"].transform("min")
    sw_max = df.groupby("page_number", sort=False)["sliding_window_id"].transform("max")

    eligible = (
        (df["text_orientation"] == "LTR") &
        (df["sliding_window_id"] != sw_min) &
        (df["sliding_window_id"] != sw_max)
    )

    # Compute bounds on eligible subset
    bounds = (
        df.loc[eligible, ["page_number", "x_left", "x_right"]]
          .groupby("page_number", sort=False)
          .agg(x_page_min=("x_left", "min"), x_page_max=("x_right", "max"))
          .reset_index()
    )

    # Merge and broadcast to all rows on the page
    df = df.merge(bounds, on="page_number", how="left")

    return df


# =======================================================================================================================
# Struct Group Collapse
# =======================================================================================================================

def _collapse_words_by_struct_group(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (page, sliding_window) bucket, collapse words that share a
    struct_group_id into a single row with a union bbox.  Words without a
    struct_group_id are kept as individual rows.

    The collapsed row keeps a representative word_id (first by df order) so
    that reject_non_content_gutters can still resolve text for gap-boundary words.
    """
    if "struct_group_id" not in df_words.columns:
        return df_words

    has_group = df_words["struct_group_id"].notna()
    if not has_group.any():
        return df_words

    keys = ["page_number", "sliding_window_id", "struct_group_id"]

    agg_spec: dict = {
        "sliding_window": ("sliding_window", "first"),
        "x_left": ("x_left", "min"),
        "x_right": ("x_right", "max"),
        "text_orientation": ("text_orientation", "first"),
        "x_page_min": ("x_page_min", "first"),
        "x_page_max": ("x_page_max", "first"),
    }
    if "word_id" in df_words.columns:
        agg_spec["word_id"] = ("word_id", "first")
    if "y_top" in df_words.columns:
        agg_spec["y_top"] = ("y_top", "min")
    if "y_bottom" in df_words.columns:
        agg_spec["y_bottom"] = ("y_bottom", "max")

    collapsed = (
        df_words[has_group]
        .groupby(keys, sort=False)
        .agg(**agg_spec)
        .reset_index()
    )

    return pd.concat([collapsed, df_words[~has_group]], ignore_index=True)


# =======================================================================================================================
# Gutter Extraction Functions
# =======================================================================================================================

# ------------------------------
# Build gutter candidate dataframe
# ------------------------------

def build_gutter_candidate_df(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a normalized gutter-candidate dataframe:
      one row per (page_number, sliding_window, candidate)

    Requires columns on words_df:
      - page_number, sliding_window_id, sliding_window
      - x_left, x_right
      - x_page_min, x_page_max

    Produces gutters_df with columns:
      - page_number
      - sliding_window_id
      - sliding_window
      - gutter_node_candidate_id  (int, 1..N unique across entire df)
      - candidate_type            ("page_left" | "page_right" | "internal_gap")
      - gutter_x_left
      - gutter_x_right
      - gutter_width
      - left_word_id              (nullable; for internal gaps = prev word)
      - right_word_id             (nullable; for internal gaps = next word)
      - left_x_right              (nullable; for internal gaps = prev x_right)
      - right_x_left              (nullable; for internal gaps = next x_left)
      - padding
      - min_gap_width
      - min_page_min_gap
      - internal_gap_density      (int, count of internal_gap rows per sliding_window_id)

    Notes:
      - For page-left/right candidates, word_id columns are NA (by design).
      - Internal gaps are computed by sorting words within each window by x_left
        (tie-break on word_id if present).
    """
    
    if words_df is None or words_df.empty:
        return pd.DataFrame(
            columns=[
                "page_number", "sliding_window_id", "sliding_window",
                "gutter_node_candidate_id", "candidate_type",
                "gutter_x_left", "gutter_x_right", "gutter_width",
                "left_word_id", "right_word_id",
                "left_x_right", "right_x_left",
                "padding", "min_gap_width", "min_page_min_gap",
                "internal_gap_density",
            ]
        )

    # ---- FILTER FIRST ----
    df = words_df.copy()
    df = df[df["text_orientation"].astype(str).str.upper().str.strip() == "LTR"].copy()

    required = {
        "page_number",
        "sliding_window_id",
        "sliding_window",
        "x_left",
        "x_right",
        "x_page_min",
        "x_page_max",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keys = ["page_number", "sliding_window_id", "sliding_window"]

    # -------------------------
    # (1) + (2) Page-edge candidates (vectorized)
    # -------------------------
    g = df.groupby(keys, sort=False)
    win_min_x_left = g["x_left"].min()
    win_max_x_right = g["x_right"].max()

    # Take x_page_min/x_page_max from the first row in each group (should be constant per page anyway)
    win_x_page_min = g["x_page_min"].first()
    win_x_page_max = g["x_page_max"].first()

    left_ok = win_min_x_left > (win_x_page_min + _MIN_PAGE_MIN_GAP)
    right_ok = win_max_x_right < (win_x_page_max - _MIN_PAGE_MIN_GAP)

    left_df = pd.DataFrame({
        "page_number": win_min_x_left.index.get_level_values(0),
        "sliding_window_id": win_min_x_left.index.get_level_values(1),
        "sliding_window": win_min_x_left.index.get_level_values(2),
        "candidate_type": "page_left",
        "gutter_x_left": win_x_page_min,
        "gutter_x_right": win_min_x_left - _TEXT_PADDING,
        "left_word_id": pd.NA,
        "right_word_id": pd.NA,
        "left_x_right": pd.NA,
        "right_x_left": pd.NA,
    }).reset_index(drop=True)
    left_df = left_df[left_ok.values].copy()

    right_df = pd.DataFrame({
        "page_number": win_max_x_right.index.get_level_values(0),
        "sliding_window_id": win_max_x_right.index.get_level_values(1),
        "sliding_window": win_max_x_right.index.get_level_values(2),
        "candidate_type": "page_right",
        "gutter_x_left": win_max_x_right + _TEXT_PADDING,
        "gutter_x_right": win_x_page_max,
        "left_word_id": pd.NA,
        "right_word_id": pd.NA,
        "left_x_right": pd.NA,
        "right_x_left": pd.NA,
    }).reset_index(drop=True)
    right_df = right_df[right_ok.values].copy()

    # Ensure sane intervals (can happen with tiny padding edge cases)
    if not left_df.empty:
        left_df = left_df[left_df["gutter_x_right"] > left_df["gutter_x_left"]]
    if not right_df.empty:
        right_df = right_df[right_df["gutter_x_right"] > right_df["gutter_x_left"]]

    # -------------------------
    # (3) Internal gap candidates (needs ordering within window)
    # -------------------------
    sort_cols = ["page_number", "sliding_window_id", "sliding_window", "x_left"]
    if "word_id" in df.columns:
        sort_cols.append("word_id")

    s = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    prev_x_right = s.groupby(keys, sort=False)["x_right"].shift(1)
    prev_word_id = s.groupby(keys, sort=False)["word_id"].shift(1) if "word_id" in s.columns else pd.Series(pd.NA, index=s.index)

    gap = s["x_left"] - prev_x_right
    gap_ok = gap > _MIN_GAP_WIDTH

    if "word_id" in s.columns:
        right_word_id = s["word_id"]
    else:
        right_word_id = pd.Series(pd.NA, index=s.index)

    internal_df = pd.DataFrame({
        "page_number": s["page_number"],
        "sliding_window_id": s["sliding_window_id"],
        "sliding_window": s["sliding_window"],
        "candidate_type": "internal_gap",
        "gutter_x_left": prev_x_right + _TEXT_PADDING,
        "gutter_x_right": s["x_left"] - _TEXT_PADDING,
        "left_word_id": prev_word_id,
        "right_word_id": right_word_id,
        "left_x_right": prev_x_right,
        "right_x_left": s["x_left"],
    })

    internal_df = internal_df[prev_x_right.notna() & gap_ok].copy()
    if not internal_df.empty:
        internal_df = internal_df[internal_df["gutter_x_right"] > internal_df["gutter_x_left"]]

    # -------------------------
    # Combine + add ids + widths + params
    # -------------------------
    gutters = pd.concat([left_df, right_df, internal_df], ignore_index=True)

    if gutters.empty:
        gutters = gutters.assign(
            gutter_node_candidate_id=pd.Series(dtype="int64"),
            gutter_width=pd.Series(dtype="float64"),
            padding=pd.Series(dtype="float64"),
            min_gap_width=pd.Series(dtype="float64"),
            min_page_min_gap=pd.Series(dtype="float64"),
            internal_gap_density=pd.Series(dtype="int64"),
        )
        # enforce column order
        return gutters[
            [
                "page_number", "sliding_window_id", "sliding_window",
                "gutter_node_candidate_id", "candidate_type",
                "gutter_x_left", "gutter_x_right", "gutter_width",
                "left_word_id", "right_word_id",
                "left_x_right", "right_x_left",
                "padding", "min_gap_width", "min_page_min_gap",
                "internal_gap_density",
            ]
        ]

    gutters["gutter_width"] = gutters["gutter_x_right"] - gutters["gutter_x_left"]

    gutters["padding"] = _TEXT_PADDING
    gutters["min_gap_width"] = _MIN_GAP_WIDTH
    gutters["min_page_min_gap"] = _MIN_PAGE_MIN_GAP

    gutters = gutters.sort_values(
        ["page_number", "sliding_window_id", "sliding_window", "gutter_x_left", "gutter_x_right", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Assign unique sequential IDs across the entire dataframe
    gutters["gutter_node_candidate_id"] = range(1, len(gutters) + 1)

    # Calculate internal_gap_density per (page_number, sliding_window_id)
    # This counts how many "internal_gap" candidates exist in each sliding window
    _density = (
        gutters[gutters["candidate_type"] == "internal_gap"]
        .groupby(["page_number", "sliding_window_id"], sort=False)
        .size()
        .rename("internal_gap_density")
        .reset_index()
    )
    gutters = gutters.merge(_density, on=["page_number", "sliding_window_id"], how="left")
    gutters["internal_gap_density"] = gutters["internal_gap_density"].fillna(0).astype(int)

    return gutters[
        [
            "page_number", "sliding_window_id", "sliding_window",
            "gutter_node_candidate_id", "candidate_type",
            "gutter_x_left", "gutter_x_right", "gutter_width",
            "left_word_id", "right_word_id",
            "left_x_right", "right_x_left",
            "padding", "min_gap_width", "min_page_min_gap",
            "internal_gap_density",
        ]
    ]


# ------------------------------
# Cluster gutter candidates into persistent gutters - writes to df_gutter_candidates
# ------------------------------

def cluster_gutter_candidates(
    gutters_df: pd.DataFrame, 
    shapes_df: pd.DataFrame = None,
    words_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Cluster gutter node candidates into persistent gutter_candidate_ids.
    
    Per page_number, process sliding windows from top to bottom:
    - For each candidate in the current window, check overlap against ALL ongoing gutters
      from the previous window
    - A candidate can belong to multiple gutters (stored as lists)
    - Each overlapping gutter gets its shape updated to the intersection
    - Gutters are terminated when:
      * There's a complete gap in sliding_window_id (e.g., windows 125→130 with no 126-129)
      * They have no overlapping candidates in a window (when the window exists)
      * Their width becomes < _MIN_GAP_WIDTH
      * A horizontal line exists between current and next sliding window within the gutter's x bounds
    
    Input columns required on gutters_df:
      - page_number
      - sliding_window_id
      - sliding_window (y value)
      - gutter_node_candidate_id
      - gutter_x_left
      - gutter_x_right
      - gutter_width
    
    Input columns required on shapes_df (optional):
      - page_number
      - shape_orientation
      - shape_type
      - x_left, x_right
      - y_top
      
    Input columns required on words_df (optional):
      - page_number
      - sliding_window_id
      - sliding_window (y value)
    
    Output columns added:
      - gutter_candidate_id: list of gutter IDs this candidate belongs to
      - gutter_candidate_shape: list of shapes for each gutter (parallel to IDs)
    
    Returns:
        DataFrame with gutter_candidate_id and gutter_candidate_shape columns added (as lists).
    """
    if gutters_df is None or gutters_df.empty:
        return gutters_df.assign(
            gutter_candidate_id=pd.Series(dtype="object"),
            gutter_candidate_shape=pd.Series(dtype="object"),
        )
    
    required = {
        "page_number",
        "sliding_window_id",
        "gutter_node_candidate_id",
        "gutter_x_left",
        "gutter_x_right",
        "gutter_width",
    }
    missing = required - set(gutters_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = gutters_df.copy()
    df = df.sort_values(
        ["page_number", "sliding_window_id", "gutter_x_left", "gutter_x_right"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Pre-extract numpy arrays — avoids repeated DataFrame column access in the hot loop
    arr_x_left = df["gutter_x_left"].values
    arr_x_right = df["gutter_x_right"].values

    # Build window-id → y-value map without iterrows
    if "sliding_window" in df.columns:
        sliding_window_y_map = dict(zip(
            df["sliding_window_id"].values, df["sliding_window"].values
        ))
    else:
        sliding_window_y_map = {}

    # Pre-group rows by (page, window) so inner loops never do boolean DataFrame masking
    page_win_map: dict = {}
    for (page_num, win_id), idxs in sorted(
        df.groupby(["page_number", "sliding_window_id"], sort=True).groups.items()
    ):
        page_win_map.setdefault(page_num, []).append((win_id, idxs.to_numpy()))

    # Pre-group horizontal lines by page for O(1) per-window lookup
    page_h_lines: dict = {}
    if shapes_df is not None and not shapes_df.empty:
        required_cols = {"shape_orientation", "shape_type", "page_number", "x_left", "x_right", "y_top"}
        if required_cols.issubset(shapes_df.columns):
            h_lines = shapes_df[
                (shapes_df["shape_orientation"].astype(str).str.lower() == "horizontal") &
                (shapes_df["shape_type"].astype(str).str.lower() == "line")
            ]
            for pn, grp in h_lines.groupby("page_number", sort=False):
                page_h_lines[pn] = (
                    grp["x_left"].values,
                    grp["x_right"].values,
                    grp["y_top"].values,
                )

    n_rows = len(df)
    gutter_candidate_ids: list = [None] * n_rows
    gutter_candidate_shapes: list = [None] * n_rows
    next_gutter_id = 1

    for page_num, win_list in page_win_map.items():
        active_gutters: dict = {}  # {gutter_id: (x_left, x_right)}
        prev_window_id = None
        n_wins = len(win_list)

        for win_pos, (window_id, win_idxs) in enumerate(win_list):
            # Kill all active gutters on a window-ID gap or a large y jump
            if prev_window_id is not None:
                if window_id != prev_window_id + 1:
                    active_gutters.clear()
                else:
                    prev_y = sliding_window_y_map.get(prev_window_id)
                    curr_y = sliding_window_y_map.get(window_id)
                    if (
                        prev_y is not None and curr_y is not None
                        and (curr_y - prev_y) > _MAX_GUTTER_WINDOW_Y_GAP
                    ):
                        active_gutters.clear()

            node_lefts = arr_x_left[win_idxs]
            node_rights = arr_x_right[win_idxs]

            # Gutter-centric best-path matching:
            # Each active gutter picks the single node with the largest overlap
            # (>= _MIN_GUTTER_CANDIDATE_OVERLAP), preventing marginal touches from
            # stealing a gutter_candidate_id away from a better-matching node.
            # A node can still be claimed by multiple gutters (if multiple gutters
            # each independently chose it as their best match).
            gutter_best: dict = {}
            if active_gutters:
                gids_list = list(active_gutters.keys())
                g_lefts = np.array([active_gutters[g][0] for g in gids_list])
                g_rights = np.array([active_gutters[g][1] for g in gids_list])

                # Vectorized overlap matrix: shape (n_gutters, n_nodes)
                int_lefts = np.maximum(g_lefts[:, None], node_lefts[None, :])
                int_rights = np.minimum(g_rights[:, None], node_rights[None, :])
                overlaps = int_rights - int_lefts
                valid = overlaps >= _MIN_GUTTER_CANDIDATE_OVERLAP

                best_node_local = np.argmax(np.where(valid, overlaps, -np.inf), axis=1)
                has_match = valid[np.arange(len(gids_list)), best_node_local]

                for i, gid in enumerate(gids_list):
                    if has_match[i]:
                        ni = int(best_node_local[i])
                        gutter_best[gid] = (
                            int(win_idxs[ni]),
                            float(int_lefts[i, ni]),
                            float(int_rights[i, ni]),
                        )

            # Invert: node df-index → list of (gutter_id, int_left, int_right)
            node_claimed_by: dict = {}
            for gid, (df_idx, il, ir) in gutter_best.items():
                node_claimed_by.setdefault(df_idx, []).append((gid, il, ir))

            matched_gutters: dict = {}
            for local_i, df_idx in enumerate(win_idxs):
                df_idx = int(df_idx)
                node_x_left = float(node_lefts[local_i])
                node_x_right = float(node_rights[local_i])
                claims = node_claimed_by.get(df_idx, [])
                if claims:
                    out_ids = [g for g, _, _ in claims]
                    out_shapes = [f"[{l:.2f}, {r:.2f}]" for _, l, r in claims]
                    for gid, il, ir in claims:
                        matched_gutters[gid] = (il, ir)
                else:
                    new_id = next_gutter_id
                    next_gutter_id += 1
                    out_ids = [new_id]
                    out_shapes = [f"[{node_x_left:.2f}, {node_x_right:.2f}]"]
                    matched_gutters[new_id] = (node_x_left, node_x_right)
                gutter_candidate_ids[df_idx] = out_ids
                gutter_candidate_shapes[df_idx] = out_shapes

            # Drop gutters that became too narrow
            active_gutters = {
                gid: bounds
                for gid, bounds in matched_gutters.items()
                if bounds[1] - bounds[0] >= _MIN_GAP_WIDTH
            }

            # Vectorized horizontal-line kill check for the gap before the next window
            if active_gutters and win_pos < n_wins - 1 and page_num in page_h_lines:
                next_win_id = win_list[win_pos + 1][0]
                current_y = sliding_window_y_map.get(window_id)
                next_y = sliding_window_y_map.get(next_win_id)
                if current_y is not None and next_y is not None:
                    lx, rx, ly = page_h_lines[page_num]
                    between = (ly > current_y) & (ly < next_y)
                    if between.any():
                        lx_b, rx_b = lx[between], rx[between]
                        gids_list = list(active_gutters.keys())
                        g_lefts = np.array([active_gutters[g][0] for g in gids_list])
                        g_rights = np.array([active_gutters[g][1] for g in gids_list])
                        # (n_gutters, n_lines) overlap
                        ol = np.maximum(g_lefts[:, None], lx_b[None, :])
                        or_ = np.minimum(g_rights[:, None], rx_b[None, :])
                        kill = (or_ - ol >= _MIN_GUTTER_LINE_KILL_OVERLAP).any(axis=1)
                        for i, gid in enumerate(gids_list):
                            if kill[i]:
                                del active_gutters[gid]

            prev_window_id = window_id

    df["gutter_candidate_id"] = gutter_candidate_ids
    df["gutter_candidate_shape"] = gutter_candidate_shapes
    return df


# ------------------------------
# Eject outlier edge windows from gutter candidate groups
# ------------------------------

def eject_outlier_edge_windows(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each gutter_candidate_id, check whether the very first and/or last
    sliding window in the group is a page_left or page_right candidate whose
    y-distance to its neighbour exceeds _EDGE_WINDOW_Y_GAP_FACTOR × the median
    y-gap of that group.  If so, remove that gutter_candidate_id from the row's
    list (the window stays in the dataframe — it just loses that ID).

    Rows that end up with an empty gutter_candidate_id list are dropped.

    This catches headers/footers that span only part of the page width and
    briefly maintain a gutter_candidate_id, even though they are not part of
    the multi-column structure proper.
    """
    if candidates_df is None or candidates_df.empty:
        return candidates_df

    df = candidates_df.copy().reset_index(drop=True)
    df["_orig_idx"] = df.index

    # Explode and pre-sort once — avoids per-group sort_values inside the loop
    exploded = (
        df[["_orig_idx", "gutter_candidate_id", "sliding_window", "candidate_type"]]
        .explode("gutter_candidate_id")
        .dropna(subset=["gutter_candidate_id"])
        .sort_values(["gutter_candidate_id", "sliding_window"])
        .reset_index(drop=True)
    )

    orig_idxs = exploded["_orig_idx"].values
    gids = exploded["gutter_candidate_id"].values
    y_vals_all = exploded["sliding_window"].values.astype(float)
    ctypes = exploded["candidate_type"].values

    # np.unique on the pre-sorted gids gives group boundaries without per-group sorting
    unique_gids, group_starts = np.unique(gids, return_index=True)
    group_ends = np.append(group_starts[1:], len(gids))

    to_eject_by_row: dict = {}  # {orig_idx: set of gutter_ids to remove}

    for i, gid in enumerate(unique_gids):
        s, e = int(group_starts[i]), int(group_ends[i])
        if e - s < 3:
            continue  # need at least 2 gaps for a meaningful median

        y = y_vals_all[s:e]
        median_gap = float(np.median(y[1:] - y[:-1]))
        if median_gap <= 0:
            continue

        threshold = _EDGE_WINDOW_Y_GAP_FACTOR * median_gap

        if ctypes[s] in ("page_left", "page_right") and (y[1] - y[0]) > threshold:
            to_eject_by_row.setdefault(int(orig_idxs[s]), set()).add(gid)

        if ctypes[e - 1] in ("page_left", "page_right") and (y[-1] - y[-2]) > threshold:
            to_eject_by_row.setdefault(int(orig_idxs[e - 1]), set()).add(gid)

    if not to_eject_by_row:
        return candidates_df

    # Update only the affected rows (typically far fewer than total row count)
    df = df.drop(columns=["_orig_idx"])
    rows_to_drop = []
    for orig_idx, gids_to_remove in to_eject_by_row.items():
        old_ids = df.at[orig_idx, "gutter_candidate_id"]
        old_shapes = df.at[orig_idx, "gutter_candidate_shape"]
        new_ids = [g for g in old_ids if g not in gids_to_remove]
        new_shapes = [sh for g, sh in zip(old_ids, old_shapes) if g not in gids_to_remove]
        df.at[orig_idx, "gutter_candidate_id"] = new_ids
        df.at[orig_idx, "gutter_candidate_shape"] = new_shapes
        if not new_ids:
            rows_to_drop.append(orig_idx)

    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
    return df


# ------------------------------
# Promote gutter candidates to gutters - writes to df_gutters
# ------------------------------

def promote_gutter_candidates_to_gutters(gutters_df: pd.DataFrame, words_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Filter gutter candidates and produce a gutter-level dataframe.
    
    A gutter_candidate_id is promoted to a gutter if it meets ALL criteria:
    1. Has at least _MIN_INTERNAL_GAPS (3) occurrences
    2. Spans at least _MIN_GUTTER_HEIGHT (50pt) vertically (based on sliding_window range)
    3. Contains at least _MIN_INTERNAL_GAPS internal_gap type candidates
    4. Has at least _MIN_INTERNAL_GAPS internal gaps in low-density regions
       (sliding windows with internal_gap_density <= _MAX_INTERNAL_GAP_DENSITY)
    5. The median internal_gap_density of all internal_gap candidates is <= _MAX_INTERNAL_GAP_DENSITY
       (prevents false positives where a few low-density gaps exist in an otherwise high-density region)
    6. Final gutter width is at least _MIN_GAP_WIDTH (12pt)
    
    Note: Gutters can extend into high-density regions (e.g., tables), but must have
    sufficient internal gaps in low-density regions to be considered valid.
    
    Input columns required on gutters_df:
      - page_number
      - gutter_candidate_id (list of IDs)
      - gutter_candidate_shape (list of shapes, parallel to IDs)
      - sliding_window
      - candidate_type
      - internal_gap_density
    
    Input columns required on words_df (optional):
      - page_number
      - sliding_window
      - y_bottom
    
    Returns:
        One row per valid gutter with columns:
          - page_number
          - gutter_candidate_id
          - gutter_y_top (first sliding_window)
          - gutter_y_bottom (max y_bottom of words in last sliding_window if words_df provided, 
                            otherwise last sliding_window)
          - gutter_x_left (from final gutter_candidate_shape)
          - gutter_x_right (from final gutter_candidate_shape)
          - gutter_width
          - gutter_height
    """
    if gutters_df is None or gutters_df.empty:
        return pd.DataFrame(
            columns=[
                "page_number", "gutter_candidate_id",
                "gutter_y_top", "gutter_y_bottom",
                "gutter_x_left", "gutter_x_right",
                "gutter_width", "gutter_height",
            ]
        )
    
    required = {
        "page_number",
        "gutter_candidate_id",
        "sliding_window",
        "candidate_type",
        "internal_gap_density",
        "gutter_candidate_shape",
    }
    missing = required - set(gutters_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = gutters_df.copy()
    
    # Explode list columns so each gutter_candidate_id gets its own row
    df = df.explode(["gutter_candidate_id", "gutter_candidate_shape"])
    df = df.dropna(subset=["gutter_candidate_id"]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "page_number", "gutter_candidate_id",
                "gutter_y_top", "gutter_y_bottom",
                "gutter_x_left", "gutter_x_right",
                "gutter_width", "gutter_height",
            ]
        )
    
    # Group by gutter_candidate_id to calculate metrics
    grouped = df.groupby("gutter_candidate_id", sort=False)
    
    # Criterion 1: Count occurrences per gutter_candidate_id
    occurrence_counts = grouped.size()
    valid_by_count = occurrence_counts >= _MIN_INTERNAL_GAPS
    
    # Criterion 2: Calculate vertical span (sliding_window range)
    sliding_window_min = grouped["sliding_window"].min()
    sliding_window_max = grouped["sliding_window"].max()
    vertical_span = sliding_window_max - sliding_window_min
    valid_by_height = vertical_span >= _MIN_GUTTER_HEIGHT
    
    # Criterion 3: Count internal_gap type candidates
    internal_gap_counts = (
        df[df["candidate_type"] == "internal_gap"]
        .groupby("gutter_candidate_id", sort=False)
        .size()
    )
    # Fill missing with 0 (gutters with no internal gaps)
    internal_gap_counts = internal_gap_counts.reindex(
        occurrence_counts.index, fill_value=0
    )
    valid_by_internal_gaps = internal_gap_counts >= _MIN_INTERNAL_GAPS
    
    # Criterion 4: Count internal gaps in low-density regions
    low_density_mask = (
        (df["candidate_type"] == "internal_gap") &
        (df["internal_gap_density"] <= _MAX_INTERNAL_GAP_DENSITY)
    )
    low_density_internal_gap_counts = (
        df[low_density_mask]
        .groupby("gutter_candidate_id", sort=False)
        .size()
    )
    # Fill missing with 0
    low_density_internal_gap_counts = low_density_internal_gap_counts.reindex(
        occurrence_counts.index, fill_value=0
    )
    valid_by_low_density_gaps = low_density_internal_gap_counts >= _MIN_INTERNAL_GAPS
    
    # Criterion 5: Median internal_gap_density for internal_gap candidates must be low
    # This prevents false positives where a gutter_candidate_id has a few low-density gaps
    # but the overall median density is high (indicating it's likely a table)
    internal_gap_median_density = (
        df[df["candidate_type"] == "internal_gap"]
        .groupby("gutter_candidate_id", sort=False)["internal_gap_density"]
        .median()
    )
    # Fill missing with inf (gutters with no internal gaps will fail this criterion)
    internal_gap_median_density = internal_gap_median_density.reindex(
        occurrence_counts.index, fill_value=float('inf')
    )
    valid_by_median_density = internal_gap_median_density <= _MAX_INTERNAL_GAP_DENSITY
    
    # Combine all criteria
    valid_gutter_ids = (
        valid_by_count &
        valid_by_height &
        valid_by_internal_gaps &
        valid_by_low_density_gaps &
        valid_by_median_density
    )
    
    # Get the set of valid gutter_candidate_ids
    valid_ids = valid_gutter_ids[valid_gutter_ids].index.tolist()
    
    if not valid_ids:
        # No valid gutters
        return pd.DataFrame(
            columns=[
                "page_number", "gutter_candidate_id",
                "gutter_y_top", "gutter_y_bottom",
                "gutter_x_left", "gutter_x_right",
                "gutter_width", "gutter_height",
            ]
        )
    
    # Filter to only valid gutters
    valid_df = df[df["gutter_candidate_id"].isin(valid_ids)].copy()
    
    # Aggregate to gutter level (one row per gutter_candidate_id)
    gutter_level = valid_df.groupby(["page_number", "gutter_candidate_id"], sort=False).agg(
        gutter_y_top=("sliding_window", "min"),
        gutter_y_bottom_sliding=("sliding_window", "max"),  # Temporarily rename to sliding
        final_shape=("gutter_candidate_shape", "last"),  # Get the most recent shape
    ).reset_index()
    
    # Calculate gutter_y_bottom using max y_bottom of words in last sliding window
    if (
        words_df is not None
        and not words_df.empty
        and "y_bottom" in words_df.columns
        and "sliding_window" in words_df.columns
    ):
        _window_max_y = (
            words_df.groupby(["page_number", "sliding_window"], sort=False)["y_bottom"]
            .max()
            .reset_index()
            .rename(columns={"sliding_window": "gutter_y_bottom_sliding", "y_bottom": "gutter_y_bottom"})
        )
        gutter_level = gutter_level.merge(
            _window_max_y, on=["page_number", "gutter_y_bottom_sliding"], how="left"
        )
        gutter_level["gutter_y_bottom"] = gutter_level["gutter_y_bottom"].fillna(
            gutter_level["gutter_y_bottom_sliding"]
        )
    else:
        gutter_level["gutter_y_bottom"] = gutter_level["gutter_y_bottom_sliding"]
    
    # Drop the temporary sliding window column
    gutter_level = gutter_level.drop(columns=["gutter_y_bottom_sliding"])
    
    # Parse the final shape "[x_left, x_right]" to get x bounds — vectorized string ops
    _parts = gutter_level["final_shape"].str.strip("[]").str.split(",", expand=True)
    gutter_level["gutter_x_left"] = pd.to_numeric(_parts[0].str.strip(), errors="coerce")
    gutter_level["gutter_x_right"] = pd.to_numeric(_parts[1].str.strip(), errors="coerce")
    
    # Drop the temporary final_shape column
    gutter_level = gutter_level.drop(columns=["final_shape"])
    
    # Calculate width and height
    gutter_level["gutter_width"] = gutter_level["gutter_x_right"] - gutter_level["gutter_x_left"]
    gutter_level["gutter_height"] = gutter_level["gutter_y_bottom"] - gutter_level["gutter_y_top"]
    
    # Criterion 6: Filter out gutters that are too narrow
    # Final width must be at least _MIN_GAP_WIDTH
    gutter_level = gutter_level[gutter_level["gutter_width"] >= _MIN_GAP_WIDTH].copy()
    
    return gutter_level


# ------------------------------
# Reject gutters whose left or right side fails content checks
# ------------------------------

# Numeric value: optional currency prefix, number body (123 / 1,23 / 1.23 / (123)), optional % or currency suffix
_NUMERIC_VALUE_RE = re.compile(
    rf'^{_CURRENCY_SYM_CLASS}?\(?\d[\d,\.]*\)?(?:{_CURRENCY_SYM_CLASS}|%)?$'
)
_DASH_TOKENS = {"-", "–", "—", "−"}


def _is_numeric_or_dash(text: object) -> bool:
    """True if text is a numeric value (possibly with currency/percent) or a dash."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return False
    t = str(text).strip()
    return bool(t) and (t in _DASH_TOKENS or bool(_NUMERIC_VALUE_RE.match(t)))


def reject_non_content_gutters(
    gutters_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    words_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove promoted gutters where the internal-gap left or right words uniformly
    look like non-content columns.  A gutter is rejected if ANY of these hold for
    either the left or the right side:

    1. All words are list/bullet markers  (e.g. "1.", "(a)", "•").
    2. All words are short  (< 7 characters each).
    3. All words are numeric values or dashes  (123, 1,23, 1.23, (123), $5, 12%,
       or dash variants).
    4. All words are identical  (e.g. the same date/label repeated on every row).

    Requires candidates_df to have: gutter_candidate_id (list), candidate_type,
    left_word_id, right_word_id.  Requires words_df to have: word_id, text.
    """
    if gutters_df is None or gutters_df.empty:
        return gutters_df
    if (
        candidates_df is None or candidates_df.empty
        or words_df is None or words_df.empty
        or "word_id" not in words_df.columns
        or "text" not in words_df.columns
    ):
        return gutters_df

    word_text = words_df.drop_duplicates("word_id").set_index("word_id")["text"]

    internal_gaps = candidates_df[candidates_df["candidate_type"] == "internal_gap"]
    if internal_gaps.empty:
        return gutters_df

    promoted_ids = set(gutters_df["gutter_candidate_id"])

    # Explode list column so each gutter_candidate_id gets its own row
    exploded = internal_gaps.explode("gutter_candidate_id")
    exploded = exploded[exploded["gutter_candidate_id"].isin(promoted_ids)].copy()
    if exploded.empty:
        return gutters_df

    exploded["_left_text"]  = exploded["left_word_id"].map(word_text)
    exploded["_right_text"] = exploded["right_word_id"].map(word_text)

    def _side_is_bad(grp: pd.DataFrame, col: str) -> bool:
        """Return True if the side (col) looks like a non-content column."""
        texts = grp[col].dropna()
        if texts.empty:
            return False
        strs = texts.apply(lambda t: str(t).strip())
        # 1. All list/bullet markers
        if strs.apply(is_list_marker).all():
            return True
        # 2. All short (< 7 chars)
        if strs.apply(len).lt(7).all():
            return True
        # 3. All numeric values or dashes
        if strs.apply(_is_numeric_or_dash).all():
            return True
        # 4. All identical
        if strs.nunique() == 1:
            return True
        return False

    bad_ids: set = set()
    for gutter_id, grp in exploded.groupby("gutter_candidate_id"):
        if _side_is_bad(grp, "_left_text") or _side_is_bad(grp, "_right_text"):
            bad_ids.add(gutter_id)

    if not bad_ids:
        return gutters_df

    return gutters_df[~gutters_df["gutter_candidate_id"].isin(bad_ids)].copy()


# ------------------------------
# Annotate gutters with intersecting horizontal lines
# ------------------------------

def filter_gutters_by_horizontal_lines(
    gutters_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    min_x_overlap: float = 10.0,
    y_padding: float = _GUTTER_LINE_Y_PADDING,
) -> pd.DataFrame:
    """
    Annotate gutters with shape_ids of intersecting horizontal lines.
    
    For each gutter, finds horizontal lines that:
    1. Have y_top within the gutter's vertical span expanded by y_padding:
       [gutter_y_top - y_padding, gutter_y_bottom + y_padding]
    2. Have x range [x_left, x_right] that overlaps with gutter's x range by >= min_x_overlap
    
    The y_padding allows detection of horizontal lines just above or below the gutter,
    which can disqualify gutter candidates.
    
    Example: If a gutter spans [644, 731] with y_padding=10, horizontal lines with
    y values in [634, 741] will be considered as intersecting.
    
    Adds a column 'intersecting_horizontal_line_ids' containing a list of shape_ids.
    
    Args:
        gutters_df: DataFrame with gutter information
        shapes_df: DataFrame with shape information
        min_x_overlap: Minimum horizontal overlap (in points) required for intersection
        y_padding: Vertical padding (in points) to expand gutter range top and bottom
    
    Input columns required on gutters_df:
      - gutter_candidate_id
      - page_number
      - gutter_y_top
      - gutter_y_bottom
      - gutter_x_left
      - gutter_x_right
    
    Input columns required on shapes_df:
      - page_number
      - shape_id
      - shape_orientation
      - shape_type
      - x_left, x_right
      - y_top
    
    Returns:
        DataFrame with added 'intersecting_horizontal_line_ids' column (list of shape_ids).
    """
    if gutters_df is None or gutters_df.empty:
        return gutters_df.assign(intersecting_horizontal_line_ids=pd.Series(dtype="object"))
    
    df = gutters_df.copy()
    
    if shapes_df is None or shapes_df.empty:
        # No shapes to check, add empty list column
        df["intersecting_horizontal_line_ids"] = [[] for _ in range(len(df))]
        return df
    
    # Filter shapes to only horizontal lines
    if "shape_orientation" not in shapes_df.columns or "shape_type" not in shapes_df.columns:
        # Missing required columns, add empty list column
        df["intersecting_horizontal_line_ids"] = [[] for _ in range(len(df))]
        return df
    
    if "shape_id" not in shapes_df.columns:
        # Missing shape_id, add empty list column
        df["intersecting_horizontal_line_ids"] = [[] for _ in range(len(df))]
        return df
    
    horizontal_lines = shapes_df[
        (shapes_df["shape_orientation"].astype(str).str.lower() == "horizontal") &
        (shapes_df["shape_type"].astype(str).str.lower() == "line")
    ].copy()
    
    if horizontal_lines.empty:
        # No horizontal lines to check, add empty list column
        df["intersecting_horizontal_line_ids"] = [[] for _ in range(len(df))]
        return df
    
    # Pre-group lines by page to avoid repeated full-df boolean masks in the loop
    page_lines_map = {
        pn: grp for pn, grp in horizontal_lines.groupby("page_number", sort=False)
    }

    # For each gutter, collect intersecting shape_ids (vectorized inner check)
    intersecting_ids_list = []
    for _, gutter in df.iterrows():
        page_lines = page_lines_map.get(gutter["page_number"])
        if page_lines is None or page_lines.empty:
            intersecting_ids_list.append([])
            continue

        g_y_min = gutter["gutter_y_top"] - y_padding
        g_y_max = gutter["gutter_y_bottom"] + y_padding
        g_x_left = gutter["gutter_x_left"]
        g_x_right = gutter["gutter_x_right"]

        in_y = (page_lines["y_top"].values >= g_y_min) & (page_lines["y_top"].values <= g_y_max)
        if not in_y.any():
            intersecting_ids_list.append([])
            continue

        x_overlap = (
            np.minimum(page_lines["x_right"].values, g_x_right)
            - np.maximum(page_lines["x_left"].values, g_x_left)
        )
        hit = in_y & (x_overlap >= min_x_overlap)
        intersecting_ids_list.append(page_lines["shape_id"].values[hit].tolist())
    
    # Add the column with intersecting shape_ids
    df["intersecting_horizontal_line_ids"] = intersecting_ids_list
    
    return df


def filter_and_assign_gutter_ids(gutters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter gutters to keep only those with no intersecting horizontal lines,
    then assign a unique gutter_id to each surviving gutter.
    
    Input columns required:
      - intersecting_horizontal_line_ids (list of shape_ids)
      - All other columns from gutters_df are preserved
    
    Output columns added:
      - gutter_id: unique sequential ID for each valid gutter (1, 2, 3, ...)
    
    Returns:
        DataFrame with only gutters that have no intersecting horizontal lines,
        with a new gutter_id column added.
    """
    if gutters_df is None or gutters_df.empty:
        return gutters_df.assign(gutter_id=pd.Series(dtype="int64"))
    
    df = gutters_df.copy()
    
    # Check if required column exists
    if "intersecting_horizontal_line_ids" not in df.columns:
        # If column doesn't exist, treat all gutters as valid and assign IDs
        df["gutter_id"] = range(1, len(df) + 1)
        return df
    
    # Filter to keep only gutters with no intersecting horizontal lines
    # Check if the list is empty for each gutter
    df = df[df["intersecting_horizontal_line_ids"].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)].copy()
    
    if df.empty:
        # No gutters survived
        df["gutter_id"] = pd.Series(dtype="int64")
        return df
    
    # Assign sequential gutter_id to surviving gutters
    df["gutter_id"] = range(1, len(df) + 1)
    
    return df


# =======================================================================================================================
# Debug / Diagnosis
# =======================================================================================================================

def diagnose_gutter_candidate(
    candidate_id: int,
    candidates_df: pd.DataFrame,
    words_df: pd.DataFrame = None,
    shapes_df: pd.DataFrame = None,
) -> dict:
    """
    Explain why a specific gutter_candidate_id did not survive to the final gutters df.

    Traces through every filter stage in the same order as detect_and_annotate_gutters
    and reports the first failing criterion.

    Args:
        candidate_id:   The gutter_candidate_id to diagnose.
        candidates_df:  The df_gutter_candidates returned by detect_and_annotate_gutters
                        (after clustering, before promotion).  Must still contain the raw
                        list-valued gutter_candidate_id column.
        words_df:       The df_words returned by detect_and_annotate_gutters (optional;
                        needed for the content-rejection check).
        shapes_df:      The original shapes dataframe (optional; needed for the
                        horizontal-line check).

    Returns:
        dict with keys:
          'found'  – bool: candidate_id appears in candidates_df at all
          'stage'  – str:  pipeline stage that rejected it, or 'survived' if it passed all
          'reason' – str:  human-readable explanation
          'detail' – dict: raw metric values for every criterion checked
    """
    # ── Explode candidates so each gutter_candidate_id is its own row ──────
    exploded = (
        candidates_df
        .explode(["gutter_candidate_id", "gutter_candidate_shape"])
        .dropna(subset=["gutter_candidate_id"])
        .reset_index(drop=True)
    )
    rows = exploded[exploded["gutter_candidate_id"] == candidate_id].copy()

    detail: dict = {}

    if rows.empty:
        return {
            "found": False,
            "stage": "cluster_gutter_candidates",
            "reason": (
                f"candidate_id {candidate_id} does not appear in candidates_df at all. "
                "It was likely killed during clustering (window-ID gap, y-gap > "
                f"{_MAX_GUTTER_WINDOW_Y_GAP}pt, width < {_MIN_GAP_WIDTH}pt, or a "
                "horizontal line crossed it)."
            ),
            "detail": detail,
        }

    detail["n_rows"] = len(rows)
    detail["sliding_windows"] = sorted(rows["sliding_window"].unique().tolist())

    # ── Stage 3 (promote): criterion 1 — occurrence count ──────────────────
    n = len(rows)
    detail["occurrence_count"] = n
    detail["min_occurrences_required"] = _MIN_INTERNAL_GAPS
    if n < _MIN_INTERNAL_GAPS:
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Too few occurrences: {n} < {_MIN_INTERNAL_GAPS} required. "
                "The gutter_candidate_id appeared in too few sliding windows."
            ),
            "detail": detail,
        }

    # ── Stage 3: criterion 2 — vertical span ───────────────────────────────
    y_min = float(rows["sliding_window"].min())
    y_max = float(rows["sliding_window"].max())
    span = y_max - y_min
    detail["vertical_span_pt"] = round(span, 2)
    detail["min_height_required"] = _MIN_GUTTER_HEIGHT
    if span < _MIN_GUTTER_HEIGHT:
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Vertical span too small: {span:.1f}pt < {_MIN_GUTTER_HEIGHT}pt required "
                f"(sliding_window range {y_min:.1f}–{y_max:.1f})."
            ),
            "detail": detail,
        }

    # ── Stage 3: criterion 3 — internal gap count ──────────────────────────
    n_internal = int((rows["candidate_type"] == "internal_gap").sum())
    detail["internal_gap_count"] = n_internal
    detail["min_internal_gaps_required"] = _MIN_INTERNAL_GAPS
    if n_internal < _MIN_INTERNAL_GAPS:
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Too few internal_gap candidates: {n_internal} < {_MIN_INTERNAL_GAPS} required. "
                "Most occurrences are page_left/page_right edge candidates."
            ),
            "detail": detail,
        }

    # ── Stage 3: criterion 4 — low-density internal gaps ───────────────────
    low_density_mask = (
        (rows["candidate_type"] == "internal_gap") &
        (rows["internal_gap_density"] <= _MAX_INTERNAL_GAP_DENSITY)
    )
    n_low_density = int(low_density_mask.sum())
    detail["low_density_internal_gap_count"] = n_low_density
    detail["max_gap_density_threshold"] = _MAX_INTERNAL_GAP_DENSITY
    if n_low_density < _MIN_INTERNAL_GAPS:
        densities = rows.loc[rows["candidate_type"] == "internal_gap", "internal_gap_density"].tolist()
        detail["internal_gap_densities"] = densities
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Too few internal gaps in low-density windows: {n_low_density} < "
                f"{_MIN_INTERNAL_GAPS} required (density threshold <= {_MAX_INTERNAL_GAP_DENSITY}). "
                f"Internal gap densities seen: {densities}. "
                "This usually means the gap only appears inside tables or dense grid regions."
            ),
            "detail": detail,
        }

    # ── Stage 3: criterion 5 — median density ──────────────────────────────
    internal_rows = rows[rows["candidate_type"] == "internal_gap"]
    median_density = float(internal_rows["internal_gap_density"].median())
    detail["median_internal_gap_density"] = round(median_density, 2)
    if median_density > _MAX_INTERNAL_GAP_DENSITY:
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Median internal_gap_density too high: {median_density:.1f} > "
                f"{_MAX_INTERNAL_GAP_DENSITY} allowed. "
                "The gap predominantly occurs inside high-density regions (likely a table)."
            ),
            "detail": detail,
        }

    # ── Stage 3: criterion 6 — final width ─────────────────────────────────
    last_shape = rows.sort_values("sliding_window").iloc[-1]["gutter_candidate_shape"]
    try:
        parts = str(last_shape).strip("[]").split(",")
        final_x_left = float(parts[0].strip())
        final_x_right = float(parts[1].strip())
        final_width = final_x_right - final_x_left
    except Exception:
        final_width = None
    detail["final_shape"] = last_shape
    detail["final_width_pt"] = round(final_width, 2) if final_width is not None else None
    detail["min_width_required"] = _MIN_GAP_WIDTH
    if final_width is not None and final_width < _MIN_GAP_WIDTH:
        return {
            "found": True,
            "stage": "promote_gutter_candidates_to_gutters",
            "reason": (
                f"Final gutter width too narrow: {final_width:.2f}pt < {_MIN_GAP_WIDTH}pt. "
                "The gutter was progressively squeezed to below the minimum width."
            ),
            "detail": detail,
        }

    # ── Stage 4: content rejection ──────────────────────────────────────────
    if words_df is not None and not words_df.empty and "word_id" in words_df.columns and "text" in words_df.columns:
        word_text = words_df.drop_duplicates("word_id").set_index("word_id")["text"]
        internal_gap_rows = rows[rows["candidate_type"] == "internal_gap"].copy()
        if not internal_gap_rows.empty:
            internal_gap_rows["_left_text"]  = internal_gap_rows["left_word_id"].map(word_text)
            internal_gap_rows["_right_text"] = internal_gap_rows["right_word_id"].map(word_text)

            def _side_diagnosis(col: str) -> tuple[bool, str]:
                texts = internal_gap_rows[col].dropna()
                if texts.empty:
                    return False, "no texts"
                strs = texts.apply(lambda t: str(t).strip())
                if strs.apply(is_list_marker).all():
                    return True, f"all list/bullet markers: {strs.tolist()}"
                if strs.apply(len).lt(7).all():
                    return True, f"all short (<7 chars): {strs.tolist()}"
                if strs.apply(_is_numeric_or_dash).all():
                    return True, f"all numeric/dash: {strs.tolist()}"
                if strs.nunique() == 1:
                    return True, f"all identical: '{strs.iloc[0]}'"
                return False, "ok"

            left_bad, left_why = _side_diagnosis("_left_text")
            right_bad, right_why = _side_diagnosis("_right_text")
            detail["content_check_left"] = left_why
            detail["content_check_right"] = right_why

            if left_bad or right_bad:
                side = "left" if left_bad else "right"
                why = left_why if left_bad else right_why
                return {
                    "found": True,
                    "stage": "reject_non_content_gutters",
                    "reason": (
                        f"Rejected because the {side}-side words look like non-content: {why}."
                    ),
                    "detail": detail,
                }

    # ── Stage 5: horizontal line intersection ───────────────────────────────
    if shapes_df is not None and not shapes_df.empty:
        # Reconstruct approximate gutter bounds from the rows we have
        if final_width is not None:
            page_number = int(rows["page_number"].iloc[0])
            g_y_top = y_min
            g_y_bottom = y_max
            g_x_left = final_x_left
            g_x_right = final_x_right
            req_cols = {"shape_orientation", "shape_type", "page_number", "x_left", "x_right", "y_top"}
            if req_cols.issubset(shapes_df.columns):
                h_lines = shapes_df[
                    (shapes_df["page_number"] == page_number) &
                    (shapes_df["shape_orientation"].astype(str).str.lower() == "horizontal") &
                    (shapes_df["shape_type"].astype(str).str.lower() == "line")
                ]
                if not h_lines.empty:
                    padded_y_min = g_y_top - _GUTTER_LINE_Y_PADDING
                    padded_y_max = g_y_bottom + _GUTTER_LINE_Y_PADDING
                    in_y = (h_lines["y_top"] >= padded_y_min) & (h_lines["y_top"] <= padded_y_max)
                    if in_y.any():
                        x_overlap = (
                            np.minimum(h_lines.loc[in_y, "x_right"], g_x_right) -
                            np.maximum(h_lines.loc[in_y, "x_left"], g_x_left)
                        )
                        hit = x_overlap >= 10.0
                        if hit.any():
                            hit_ids = h_lines.loc[in_y].loc[hit.values, "shape_id"].tolist() if "shape_id" in h_lines.columns else "unknown"
                            detail["intersecting_line_ids"] = hit_ids
                            return {
                                "found": True,
                                "stage": "filter_and_assign_gutter_ids",
                                "reason": (
                                    f"Removed because a horizontal line crosses it "
                                    f"(shape_ids: {hit_ids})."
                                ),
                                "detail": detail,
                            }

    return {
        "found": True,
        "stage": "survived",
        "reason": "Passed all filter stages — this candidate_id should appear in the final gutters df.",
        "detail": detail,
    }


def audit_gutter_candidates(
    candidates_df: pd.DataFrame,
    words_df: pd.DataFrame = None,
    shapes_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Return one row per gutter_candidate_id with the stage and reason it was rejected
    (or 'survived' if it passed all filters).

    Uses the actual pipeline functions (promote, content-reject, h-line filter) to
    determine pass/fail — avoids reimplementing their logic and guarantees the audit
    matches what detect_and_annotate_gutters actually does.

    Columns returned:
      - gutter_candidate_id
      - page_number
      - stage          rejection stage ('promote' | 'content_rejection' | 'horizontal_line' | 'survived')
      - reason         human-readable explanation of the first failing criterion
      - occurrence_count
      - vertical_span_pt
      - internal_gap_count
      - low_density_internal_gap_count
      - median_internal_gap_density
      - final_width_pt
      - final_shape
    """
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=[
            "gutter_candidate_id", "page_number", "stage", "reason",
            "occurrence_count", "vertical_span_pt", "internal_gap_count",
            "low_density_internal_gap_count", "median_internal_gap_density",
            "final_width_pt", "final_shape",
        ])

    # ── Explode to get per-candidate metrics ─────────────────────────────────
    exploded = (
        candidates_df
        .explode(["gutter_candidate_id", "gutter_candidate_shape"])
        .dropna(subset=["gutter_candidate_id"])
        .reset_index(drop=True)
    )

    grouped      = exploded.groupby("gutter_candidate_id", sort=False)
    page_numbers = grouped["page_number"].first()
    sliding_min  = grouped["sliding_window"].min()
    sliding_max  = grouped["sliding_window"].max()

    internal_rows = exploded[exploded["candidate_type"] == "internal_gap"]

    occurrence_counts   = grouped.size()
    vertical_spans      = (sliding_max - sliding_min).round(2)
    internal_gap_counts = internal_rows.groupby("gutter_candidate_id", sort=False).size().reindex(occurrence_counts.index, fill_value=0)
    low_density_counts  = (
        internal_rows[internal_rows["internal_gap_density"] <= _MAX_INTERNAL_GAP_DENSITY]
        .groupby("gutter_candidate_id", sort=False).size()
        .reindex(occurrence_counts.index, fill_value=0)
    )
    median_densities = (
        internal_rows.groupby("gutter_candidate_id", sort=False)["internal_gap_density"]
        .median()
        .reindex(occurrence_counts.index, fill_value=float("inf"))
        .round(2)
    )
    last_shapes = (
        exploded.sort_values("sliding_window")
        .groupby("gutter_candidate_id", sort=False)["gutter_candidate_shape"]
        .last()
    )

    def _parse_width(shape):
        try:
            parts = str(shape).strip("[]").split(",")
            return round(float(parts[1].strip()) - float(parts[0].strip()), 2)
        except Exception:
            return None

    final_widths = last_shapes.map(_parse_width)

    metrics = pd.DataFrame({
        "page_number":                    page_numbers,
        "occurrence_count":               occurrence_counts,
        "vertical_span_pt":               vertical_spans,
        "internal_gap_count":             internal_gap_counts,
        "low_density_internal_gap_count": low_density_counts,
        "median_internal_gap_density":    median_densities,
        "final_shape":                    last_shapes,
        "final_width_pt":                 final_widths,
    })

    # ── Run the real pipeline functions to get the surviving sets ────────────
    promoted_df         = promote_gutter_candidates_to_gutters(candidates_df, words_df)
    promoted_ids        = set(promoted_df["gutter_candidate_id"]) if not promoted_df.empty else set()

    content_df          = reject_non_content_gutters(promoted_df, candidates_df, words_df)
    content_ids         = set(content_df["gutter_candidate_id"]) if not content_df.empty else set()

    hline_annotated_df  = filter_gutters_by_horizontal_lines(content_df, shapes_df)
    final_df            = filter_and_assign_gutter_ids(hline_annotated_df)
    final_ids           = set(final_df["gutter_candidate_id"]) if not final_df.empty else set()

    # ── Assign stage from set membership ────────────────────────────────────
    def _stage(cid):
        if cid in final_ids:
            return "survived", ""
        if cid in content_ids:
            # killed by h-line filter
            if not hline_annotated_df.empty and "intersecting_horizontal_line_ids" in hline_annotated_df.columns:
                row = hline_annotated_df[hline_annotated_df["gutter_candidate_id"] == cid]
                if not row.empty:
                    ids = row.iloc[0]["intersecting_horizontal_line_ids"]
                    return "horizontal_line", f"crossed by horizontal line(s): {ids}"
            return "horizontal_line", "crossed by a horizontal line"
        if cid in promoted_ids:
            return "content_rejection", "left or right side words are non-content (markers / short / numeric / identical)"
        # Failed promote — compute the specific criterion from metrics
        m = metrics.loc[cid]
        if m["occurrence_count"] < _MIN_INTERNAL_GAPS:
            return "promote", f"occurrence_count {m['occurrence_count']} < {_MIN_INTERNAL_GAPS}"
        if m["vertical_span_pt"] < _MIN_GUTTER_HEIGHT:
            return "promote", f"vertical_span {m['vertical_span_pt']:.1f}pt < {_MIN_GUTTER_HEIGHT}pt"
        if m["internal_gap_count"] < _MIN_INTERNAL_GAPS:
            return "promote", f"internal_gap_count {m['internal_gap_count']} < {_MIN_INTERNAL_GAPS}"
        if m["low_density_internal_gap_count"] < _MIN_INTERNAL_GAPS:
            return "promote", f"low_density_internal_gap_count {m['low_density_internal_gap_count']} < {_MIN_INTERNAL_GAPS} (density threshold <= {_MAX_INTERNAL_GAP_DENSITY})"
        if m["median_internal_gap_density"] > _MAX_INTERNAL_GAP_DENSITY:
            return "promote", f"median_internal_gap_density {m['median_internal_gap_density']:.1f} > {_MAX_INTERNAL_GAP_DENSITY}"
        w = m["final_width_pt"]
        if w is not None and w < _MIN_GAP_WIDTH:
            return "promote", f"final_width {w:.2f}pt < {_MIN_GAP_WIDTH}pt"
        return "promote", "rejected by promote (criterion unclear from metrics — check promote function directly)"

    stages_col  = []
    reasons_col = []
    for cid in metrics.index:
        s, r = _stage(cid)
        stages_col.append(s)
        reasons_col.append(r)

    metrics["stage"]  = stages_col
    metrics["reason"] = reasons_col

    return metrics.reset_index().rename(columns={"gutter_candidate_id": "gutter_candidate_id"})[[
        "gutter_candidate_id", "page_number", "stage", "reason",
        "occurrence_count", "vertical_span_pt", "internal_gap_count",
        "low_density_internal_gap_count", "median_internal_gap_density",
        "final_width_pt", "final_shape",
    ]].sort_values("gutter_candidate_id").reset_index(drop=True)


# =======================================================================================================================
# Expand and Merge Gutters
# =======================================================================================================================

def expand_and_merge_gutters(
    gutters_df: pd.DataFrame,
    words_df: pd.DataFrame,
    shapes_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Expand each promoted gutter vertically as far as possible, then merge
    adjacent segments whose x-ranges intersect and whose vertical gap is clear.

    Expansion (per gutter):
      - Upward:   new y_top  = y_bottom of the nearest word that x-overlaps
                  the gutter and lies above its current top (0 if none).
      - Downward: new y_bottom = y_top of the nearest word that x-overlaps
                  the gutter and lies below its current bottom (unchanged if none).
      Horizontal lines that x-overlap additionally cap the expansion.

    Merge (per page, after expansion):
      Gutters are sorted by y_top. Adjacent pairs whose x-ranges intersect by
      >= _MERGE_MIN_X_INTERSECTION and whose vertical gap contains no word that
      x-overlaps the intersection are merged: the result takes the x-intersection
      and the union of the y-extents.

    Reassigns gutter_id values sequentially.
    """
    if gutters_df is None or gutters_df.empty:
        return gutters_df
    if words_df is None or words_df.empty:
        return gutters_df

    ltr = words_df["text_orientation"] == "LTR" if "text_orientation" in words_df.columns else pd.Series(True, index=words_df.index)
    w = words_df.loc[ltr, ["page_number", "x_left", "x_right", "y_top", "y_bottom"]].copy()

    page_words = {pn: grp for pn, grp in w.groupby("page_number", sort=False)}

    page_hlines: dict = {}
    if shapes_df is not None and not shapes_df.empty:
        req = {"shape_orientation", "shape_type", "page_number", "x_left", "x_right", "y_top"}
        if req.issubset(shapes_df.columns):
            hl = shapes_df[
                (shapes_df["shape_orientation"].astype(str).str.lower() == "horizontal") &
                (shapes_df["shape_type"].astype(str).str.lower() == "line")
            ][["page_number", "x_left", "x_right", "y_top"]]
            page_hlines = {pn: grp for pn, grp in hl.groupby("page_number", sort=False)}

    result_rows: list = []

    for page_num, page_gutters in gutters_df.groupby("page_number", sort=False):
        pw = page_words.get(page_num)
        ph = page_hlines.get(page_num)

        gutters = page_gutters.to_dict("records")

        # ── Phase 1: Expand each gutter vertically ────────────────────────
        page_yb_max = float(pw["y_bottom"].max()) if pw is not None else None

        for g in gutters:
            xl, xr = g["gutter_x_left"], g["gutter_x_right"]
            yt_orig = g["gutter_y_top"]
            yb_orig = g["gutter_y_bottom"]
            yt = yt_orig
            yb = yb_orig

            if pw is not None:
                px  = pw["x_left"].values
                pr  = pw["x_right"].values
                pyt = pw["y_top"].values
                pyb = pw["y_bottom"].values
                x_hit = (px < xr - _EXPAND_X_EPS) & (pr > xl + _EXPAND_X_EPS)

                above = x_hit & (pyb <= yt_orig + _EXPAND_X_EPS)
                yt = float(pyb[above].max()) if above.any() else 0.0

                # Extend downward to the first x-overlapping word below; if none,
                # fall through to the page's text bottom so short gutters grow down.
                below = x_hit & (pyt >= yb_orig - _EXPAND_X_EPS)
                yb = float(pyt[below].min()) if below.any() else page_yb_max

            if ph is not None:
                hx  = ph["x_left"].values
                hr  = ph["x_right"].values
                hy  = ph["y_top"].values
                x_hit_h = (hx < xr - _EXPAND_X_EPS) & (hr > xl + _EXPAND_X_EPS)

                # Lines crossed during upward expansion: hy in [yt, yt_orig).
                # Stop at the last one (highest y = closest to original top).
                if yt < yt_orig:
                    crossed_top = x_hit_h & (hy >= yt) & (hy < yt_orig)
                    if crossed_top.any():
                        yt = float(hy[crossed_top].max())

                # Lines crossed during downward expansion: hy in (yb_orig, yb].
                # Stop at the first one (lowest y = closest to original bottom).
                if yb > yb_orig:
                    crossed_bot = x_hit_h & (hy > yb_orig) & (hy <= yb)
                    if crossed_bot.any():
                        yb = float(hy[crossed_bot].min())

            g["gutter_y_top"]    = yt
            g["gutter_y_bottom"] = yb

        # ── Phase 2: Merge compatible adjacent gutters ────────────────────
        gutters.sort(key=lambda g: g["gutter_y_top"])

        merged: list = []
        for g in gutters:
            if not merged:
                merged.append(dict(g))
                continue

            prev = merged[-1]
            ix_left  = max(prev["gutter_x_left"],  g["gutter_x_left"])
            ix_right = min(prev["gutter_x_right"], g["gutter_x_right"])

            if ix_right - ix_left < _MERGE_MIN_X_INTERSECTION:
                merged.append(dict(g))
                continue

            # Gap between prev bottom and g top — check if it is clear
            gap_yt = prev["gutter_y_bottom"]
            gap_yb = g["gutter_y_top"]
            can_merge = True

            if gap_yb > gap_yt and pw is not None:
                px  = pw["x_left"].values
                pr  = pw["x_right"].values
                pyt = pw["y_top"].values
                pyb = pw["y_bottom"].values
                in_gap = (
                    (pyt < gap_yb - _EXPAND_X_EPS) &
                    (pyb > gap_yt + _EXPAND_X_EPS) &
                    (px  < ix_right - _EXPAND_X_EPS) &
                    (pr  > ix_left  + _EXPAND_X_EPS)
                )
                if in_gap.any():
                    can_merge = False

            if can_merge:
                prev["gutter_x_left"]   = ix_left
                prev["gutter_x_right"]  = ix_right
                prev["gutter_y_bottom"] = max(prev["gutter_y_bottom"], g["gutter_y_bottom"])
            else:
                merged.append(dict(g))

        for g in merged:
            g["gutter_width"]  = g["gutter_x_right"]  - g["gutter_x_left"]
            g["gutter_height"] = g["gutter_y_bottom"] - g["gutter_y_top"]

        result_rows.extend(merged)

    if not result_rows:
        return gutters_df.iloc[:0].copy()

    result = pd.DataFrame(result_rows).reset_index(drop=True)
    result["gutter_id"] = range(1, len(result) + 1)
    return result


# =======================================================================================================================
# Merge Gutters back onto Words DataFrame
# =======================================================================================================================

def merge_gutters_onto_words(df_words: pd.DataFrame, df_gutters: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate each word with the gutters that border it and its reading column.

    Adds columns:
      - gutter_id_left:   gutter_id of the nearest gutter whose x_right <= word x_left
                          (with vertical overlap), or pd.NA if none.
      - gutter_id_right:  gutter_id of the nearest gutter whose x_left >= word x_right
                          (with vertical overlap), or pd.NA if none.
      - reading_column:   1-based column index; equals 1 + number of gutters fully to the
                          left of the word (with vertical overlap).

    Vertical overlap condition:
        gutter_y_top < word_y_bottom  AND  gutter_y_bottom > word_y_top
    """
    df_words = df_words.copy()
    df_words["gutter_id_left"] = pd.NA
    df_words["gutter_id_right"] = pd.NA
    df_words["reading_column"] = 1

    if df_gutters is None or df_gutters.empty:
        return df_words

    required = {"page_number", "gutter_id", "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom"}
    missing = required - set(df_gutters.columns)
    if missing:
        raise ValueError(f"df_gutters missing columns: {sorted(missing)}")

    gutter_cols = df_gutters[["page_number", "gutter_id", "gutter_x_left", "gutter_x_right", "gutter_y_top", "gutter_y_bottom"]].copy()

    for page, page_gutters in gutter_cols.groupby("page_number"):
        word_mask = df_words["page_number"] == page
        if not word_mask.any():
            continue

        # Cross-join words × page gutters, then filter by vertical overlap
        w = df_words.loc[word_mask, ["x_left", "x_right", "y_top", "y_bottom"]].copy()
        w["_widx"] = w.index
        w["_key"] = 1
        g = page_gutters.copy()
        g["_key"] = 1

        cross = w.merge(g, on="_key").drop(columns="_key")

        # Vertical overlap
        cross = cross[
            (cross["gutter_y_top"] < cross["y_bottom"]) &
            (cross["gutter_y_bottom"] > cross["y_top"])
        ]

        if cross.empty:
            continue

        # --- gutter_id_left: gutter whose right edge is at or left of the word ---
        # Small epsilon guards against floating-point display equality failing (e.g.
        # gutter_x_right=264.5799... vs x_left=264.5800... comparing as unequal)
        left = cross[cross["gutter_x_right"] <= cross["x_left"] + _GUTTER_X_SNAP_EPS].copy()
        if not left.empty:
            # nearest = largest gutter_x_right per word
            best_left = (
                left.sort_values("gutter_x_right")
                    .groupby("_widx", sort=False)
                    .last()[["gutter_id"]]
            )
            df_words.loc[best_left.index, "gutter_id_left"] = best_left["gutter_id"].values

            # reading_column = 1 + count of gutters to the left
            left_count = left.groupby("_widx").size()
            df_words.loc[left_count.index, "reading_column"] = left_count.values + 1

        # --- gutter_id_right: gutter whose left edge is at or right of the word ---
        right = cross[cross["gutter_x_left"] >= cross["x_right"] - _GUTTER_X_SNAP_EPS].copy()
        if not right.empty:
            # nearest = smallest gutter_x_left per word
            best_right = (
                right.sort_values("gutter_x_left")
                     .groupby("_widx", sort=False)
                     .first()[["gutter_id"]]
            )
            df_words.loc[best_right.index, "gutter_id_right"] = best_right["gutter_id"].values

    return df_words


# =======================================================================================================================
# Public API
# =======================================================================================================================

def detect_and_annotate_gutters(df_words: pd.DataFrame, df_shapes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detect vertical gutters in the document.
    A gutter is a vertical area without text or shapes that separates (part of) a page into two or more columns.
    """
    _empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if df_words is None or df_words.empty:
        return _empty

    if "text_orientation" not in df_words.columns:
        raise ValueError("df_words must contain a 'text_orientation' column")

    # Run gutter detection on LTR words only; non-LTR words are merged back at the end
    ltr_mask = df_words["text_orientation"] == "LTR"
    df_non_ltr = df_words[~ltr_mask].copy()
    df_words = df_words[ltr_mask].copy()
    if df_words.empty:
        return _empty

    # 1) Add sliding windows
    df_words = add_sliding_windows(df_words)

    # 2) Add page x bounds
    df_words = add_page_x_bounds(df_words)

    # 3) Build gutter candidates (collapsed by struct_group_id so that gaps
    #    within a single logical content block never become gutter candidates)
    df_gutter_candidates = build_gutter_candidate_df(_collapse_words_by_struct_group(df_words))

    # 4) Cluster candidates into persistent gutter tracks
    df_gutter_candidates = cluster_gutter_candidates(df_gutter_candidates, df_shapes, df_words)

    # 5) Eject page_left/page_right edge windows that are outliers in y-distance
    df_gutter_candidates = eject_outlier_edge_windows(df_gutter_candidates)

    # 6) Promote gutter candidates to actual gutters
    df_gutters = promote_gutter_candidates_to_gutters(df_gutter_candidates, df_words)

    # 7) Reject gutters whose left or right side looks like non-content (markers, short, numeric, or repeated)
    df_gutters = reject_non_content_gutters(df_gutters, df_gutter_candidates, df_words)

    # 8) Annotate gutters with intersecting horizontal line shape_ids
    df_gutters = filter_gutters_by_horizontal_lines(df_gutters, df_shapes)

    # 9) Filter out gutters crossed by a horizontal line and assign final gutter_id
    df_gutters = filter_and_assign_gutter_ids(df_gutters)

    # 10) Expand each gutter vertically and merge adjacent segments with a clear gap
    #df_gutters = expand_and_merge_gutters(df_gutters, df_words, df_shapes)

    # 11) Merge gutters onto words
    df_words = merge_gutters_onto_words(df_words, df_gutters)

    df_words = pd.concat([df_words, df_non_ltr])

    return df_words, df_gutter_candidates, df_gutters
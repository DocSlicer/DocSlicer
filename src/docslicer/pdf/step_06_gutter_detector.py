"""
step_05_gutter_extractor.py
"""

from __future__ import annotations

import re

import pandas as pd

from .._utils.text_utils import is_list_marker

# Numeric value: optional currency prefix, number body (123 / 1,23 / 1.23 / (123)), optional % or currency suffix
_NUMERIC_VALUE_RE = re.compile(
    r'^[\$€£¥₹₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾]?\(?\d[\d,\.]*\)?[\$€£¥₹₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾%]?$'
)
_DASH_TOKENS = {"-", "–", "—", "−"}


def _is_numeric_or_dash(text: object) -> bool:
    """True if text is a numeric value (possibly with currency/percent) or a dash."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return False
    t = str(text).strip()
    return bool(t) and (t in _DASH_TOKENS or bool(_NUMERIC_VALUE_RE.match(t)))

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

_Y_TOP_SLIDING_WINDOW: float = 5.0  # pt
_MIN_GAP_WIDTH: float = 10.0  # pt
_MIN_PAGE_MIN_GAP: float = 100.0  # pt
_TEXT_PADDING: float = 0.0            # pt
_MIN_GUTTER_CANDIDATE_OVERLAP: float = 3.0  # pt - require min 3pt overlap to maintain an existing gutter_candidate_id (prevent destroying a series on accidental contact)
_MIN_GUTTER_LINE_KILL_OVERLAP: float = 5.0  # pt - horizontal line must overlap a gutter by at least this much to kill it (prevent marginal line touches from terminating a gutter)
_GUTTER_LINE_Y_PADDING: float = 7.0  # pt - vertical padding applied above/below a promoted gutter when checking for intersecting horizontal lines
_MIN_GUTTER_HEIGHT: float = 50.0  # pt
_MIN_INTERNAL_GAPS: int = 3 # how many internal gaps does a gutter_candidate_id need to have to be a gutter
_MAX_INTERNAL_GAP_DENSITY: int = 3 # those _MIN_INTERNAL_GAPS need to come from gutter_candidate_id with <= 4 internal gaps, otherwise if those gaps only exist within high density areas, its a table
_MAX_GUTTER_WINDOW_Y_GAP: float = 30.0  # pt - if the y distance between two consecutive sliding windows exceeds this, kill all active gutters (large vertical gap = new layout region)
_EDGE_WINDOW_Y_GAP_FACTOR: float = 1.4  # if the first/last window of a gutter_candidate_id is page_left/page_right and its y-distance to the adjacent window exceeds this multiple of the median gap, eject it



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
        flags = pd.Series(False, index=y_series.index)
        bucket_start = None
        for idx, y in y_series.items():
            if bucket_start is None or y > bucket_start + _Y_TOP_SLIDING_WINDOW:
                bucket_start = y
                flags[idx] = True
        return flags

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
      - Only use rows where text_orientation == "LRT"
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
    internal_gap_counts = (
        gutters[gutters["candidate_type"] == "internal_gap"]
        .groupby(["page_number", "sliding_window_id"], sort=False)
        .size()
    )
    gutters["internal_gap_density"] = (
        gutters.set_index(["page_number", "sliding_window_id"])
        .index.map(lambda idx: internal_gap_counts.get(idx, 0))
    )

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
    
    # Sort by page_number, then sliding_window_id, then by x position
    df = df.sort_values(
        ["page_number", "sliding_window_id", "gutter_x_left", "gutter_x_right"],
        kind="mergesort"
    ).reset_index(drop=True)
    
    # Initialize output columns (lists)
    gutter_candidate_ids = []
    gutter_candidate_shapes = []
    
    # Track active gutters per page
    # Key: gutter_id, Value: (x_left, x_right)
    next_gutter_id = 1
    
    # Prepare horizontal lines data for checking between windows (if available)
    horizontal_lines = None
    if shapes_df is not None and not shapes_df.empty:
        if all(col in shapes_df.columns for col in ["shape_orientation", "shape_type", "page_number", "x_left", "x_right", "y_top"]):
            horizontal_lines = shapes_df[
                (shapes_df["shape_orientation"].astype(str).str.lower() == "horizontal") &
                (shapes_df["shape_type"].astype(str).str.lower() == "line")
            ].copy()
    
    # Get sliding window y values mapping (window_id -> y value) if available
    sliding_window_y_map = {}
    if "sliding_window" in df.columns:
        for _, row in df[["sliding_window_id", "sliding_window"]].drop_duplicates().iterrows():
            sliding_window_y_map[row["sliding_window_id"]] = row["sliding_window"]
    
    # Process page by page
    for page_num in df["page_number"].unique():
        page_mask = df["page_number"] == page_num
        page_df = df[page_mask].copy()
        
        # Track active gutters for this page
        active_gutters = {}  # {gutter_id: (x_left, x_right)}
        prev_window_id = None
        
        # Get sorted window IDs for this page
        window_ids = sorted(page_df["sliding_window_id"].unique())
        
        # Process each sliding window in order
        for window_idx, window_id in enumerate(window_ids):
            # Kill all active gutters on a gap in window IDs or a large y jump
            if prev_window_id is not None:
                if window_id != prev_window_id + 1:
                    active_gutters.clear()
                else:
                    prev_y = sliding_window_y_map.get(prev_window_id)
                    curr_y = sliding_window_y_map.get(window_id)
                    if prev_y is not None and curr_y is not None and (curr_y - prev_y) > _MAX_GUTTER_WINDOW_Y_GAP:
                        active_gutters.clear()
            
            window_mask = page_df["sliding_window_id"] == window_id
            window_rows = page_df[window_mask]
            
            # Gutter-centric best-path matching:
            # Each active gutter picks the single node with the largest overlap
            # (>= _MIN_GUTTER_CANDIDATE_OVERLAP), preventing marginal touches from
            # stealing a gutter_candidate_id away from a better-matching node.
            # A node can still be claimed by multiple gutters (if multiple gutters
            # each independently chose it as their best match).

            # Step 1: for each active gutter, find its best node
            gutter_best = {}  # {gutter_id: (node_idx, int_left, int_right)}
            for gutter_id, (g_left, g_right) in active_gutters.items():
                best_idx = None
                best_overlap = 0.0
                best_int_left = best_int_right = None
                for idx, row in window_rows.iterrows():
                    int_left = max(row["gutter_x_left"], g_left)
                    int_right = min(row["gutter_x_right"], g_right)
                    overlap = int_right - int_left
                    if overlap >= _MIN_GUTTER_CANDIDATE_OVERLAP and overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = idx
                        best_int_left, best_int_right = int_left, int_right
                if best_idx is not None:
                    gutter_best[gutter_id] = (best_idx, best_int_left, best_int_right)

            # Step 2: invert — for each node, collect the gutters that chose it
            node_claimed_by = {}  # {node_idx: [(gutter_id, int_left, int_right)]}
            for gutter_id, (node_idx, int_left, int_right) in gutter_best.items():
                node_claimed_by.setdefault(node_idx, []).append((gutter_id, int_left, int_right))

            # Step 3: build per-node output; unclaimed nodes start a new gutter
            matched_gutters = {}
            for idx, row in window_rows.iterrows():
                node_x_left = row["gutter_x_left"]
                node_x_right = row["gutter_x_right"]
                claims = node_claimed_by.get(idx, [])
                if claims:
                    candidate_gutter_ids = [g for g, _, _ in claims]
                    candidate_gutter_shapes = [f"[{l:.2f}, {r:.2f}]" for _, l, r in claims]
                    for gutter_id, int_left, int_right in claims:
                        matched_gutters[gutter_id] = (int_left, int_right)
                else:
                    new_id = next_gutter_id
                    next_gutter_id += 1
                    candidate_gutter_ids = [new_id]
                    candidate_gutter_shapes = [f"[{node_x_left:.2f}, {node_x_right:.2f}]"]
                    matched_gutters[new_id] = (node_x_left, node_x_right)
                gutter_candidate_ids.append(candidate_gutter_ids)
                gutter_candidate_shapes.append(candidate_gutter_shapes)
            
            # Update active_gutters: keep only matched gutters, remove too-narrow ones
            new_active_gutters = {}
            for gutter_id, (new_left, new_right) in matched_gutters.items():
                width = new_right - new_left
                if width >= _MIN_GAP_WIDTH:
                    new_active_gutters[gutter_id] = (new_left, new_right)
                # else: gutter dies (too narrow)
            
            active_gutters = new_active_gutters
            
            # Check for horizontal lines between current and next sliding window
            # If a horizontal line crosses a gutter's x bounds, kill that gutter
            if horizontal_lines is not None and not horizontal_lines.empty and window_idx < len(window_ids) - 1:
                next_window_id = window_ids[window_idx + 1]
                
                # Get y values for current and next window
                current_y = sliding_window_y_map.get(window_id)
                next_y = sliding_window_y_map.get(next_window_id)
                
                if current_y is not None and next_y is not None:
                    # Get horizontal lines on this page between current and next window
                    page_lines = horizontal_lines[
                        (horizontal_lines["page_number"] == page_num) &
                        (horizontal_lines["y_top"] > current_y) &
                        (horizontal_lines["y_top"] < next_y)
                    ]
                    
                    if not page_lines.empty:
                        # Check each active gutter against these lines
                        gutters_to_kill = []
                        for gutter_id, (g_left, g_right) in active_gutters.items():
                            for _, line in page_lines.iterrows():
                                line_x_left = line["x_left"]
                                line_x_right = line["x_right"]
                                
                                # Check if line overlaps with gutter's x bounds by at least
                                # _MIN_GUTTER_CANDIDATE_OVERLAP (marginal touches do not kill)
                                overlap_left = max(line_x_left, g_left)
                                overlap_right = min(line_x_right, g_right)

                                if overlap_right - overlap_left >= _MIN_GUTTER_LINE_KILL_OVERLAP:
                                    # Horizontal line crosses this gutter - kill it
                                    gutters_to_kill.append(gutter_id)
                                    break
                        
                        # Remove gutters killed by horizontal lines
                        for gutter_id in gutters_to_kill:
                            if gutter_id in active_gutters:
                                del active_gutters[gutter_id]
            
            # Update for next iteration
            prev_window_id = window_id
    
    # Add the new columns to the dataframe
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

    import numpy as np

    df = candidates_df.copy().reset_index(drop=True)
    df["_orig_idx"] = df.index

    # Explode to one row per scalar gutter_candidate_id for analysis
    exploded = (
        df[["_orig_idx", "gutter_candidate_id", "sliding_window", "candidate_type"]]
        .explode("gutter_candidate_id")
        .dropna(subset=["gutter_candidate_id"])
    )

    to_eject: set[tuple] = set()  # (orig_idx, gutter_candidate_id)

    for gid, grp in exploded.groupby("gutter_candidate_id", sort=False):
        grp = grp.sort_values("sliding_window").reset_index(drop=True)
        n = len(grp)
        if n < 3:
            continue  # need at least 2 gaps for a meaningful median

        y_vals = grp["sliding_window"].to_numpy(dtype=float)
        gaps = y_vals[1:] - y_vals[:-1]
        median_gap = float(np.median(gaps))
        if median_gap <= 0:
            continue

        threshold = _EDGE_WINDOW_Y_GAP_FACTOR * median_gap

        # First window
        if grp.iloc[0]["candidate_type"] in ("page_left", "page_right"):
            if (y_vals[1] - y_vals[0]) > threshold:
                to_eject.add((int(grp.iloc[0]["_orig_idx"]), gid))

        # Last window
        if grp.iloc[-1]["candidate_type"] in ("page_left", "page_right"):
            if (y_vals[-1] - y_vals[-2]) > threshold:
                to_eject.add((int(grp.iloc[-1]["_orig_idx"]), gid))

    if not to_eject:
        return candidates_df

    # Remove ejected (orig_idx, gutter_id) pairs from the list columns
    def _filter_row(row):
        orig_idx = row["_orig_idx"]
        ids = row["gutter_candidate_id"]
        shapes = row["gutter_candidate_shape"]
        new_ids, new_shapes = [], []
        for gid, gshape in zip(ids, shapes):
            if (orig_idx, gid) not in to_eject:
                new_ids.append(gid)
                new_shapes.append(gshape)
        return new_ids, new_shapes

    filtered = df.apply(_filter_row, axis=1)
    df["gutter_candidate_id"] = [r[0] for r in filtered]
    df["gutter_candidate_shape"] = [r[1] for r in filtered]
    df = df.drop(columns=["_orig_idx"])

    # Drop rows whose ID list became empty
    df = df[df["gutter_candidate_id"].apply(len) > 0].reset_index(drop=True)
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
    
    # Explode the list columns to create one row per gutter_candidate_id
    # Each row had lists of gutter_candidate_id and gutter_candidate_shape
    # We need to expand them so each ID gets its own row
    exploded_rows = []
    
    for idx, row in df.iterrows():
        gutter_ids = row["gutter_candidate_id"]
        gutter_shapes = row["gutter_candidate_shape"]
        
        # Handle cases where it might not be a list
        if not isinstance(gutter_ids, list):
            gutter_ids = [gutter_ids]
        if not isinstance(gutter_shapes, list):
            gutter_shapes = [gutter_shapes]
        
        # Create a row for each gutter_id
        for gid, gshape in zip(gutter_ids, gutter_shapes):
            new_row = row.copy()
            new_row["gutter_candidate_id"] = gid
            new_row["gutter_candidate_shape"] = gshape
            exploded_rows.append(new_row)
    
    if not exploded_rows:
        return pd.DataFrame(
            columns=[
                "page_number", "gutter_candidate_id",
                "gutter_y_top", "gutter_y_bottom",
                "gutter_x_left", "gutter_x_right",
                "gutter_width", "gutter_height",
            ]
        )
    
    df = pd.DataFrame(exploded_rows).reset_index(drop=True)
    
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
    
    # Helper to parse gutter_candidate_shape
    def parse_shape(shape_str):
        """Extract x_left and x_right from shape string like '[76.69, 114.25]'"""
        try:
            values = shape_str.strip('[]').split(',')
            return float(values[0].strip()), float(values[1].strip())
        except:
            return None, None
    
    # Aggregate to gutter level (one row per gutter_candidate_id)
    gutter_level = valid_df.groupby(["page_number", "gutter_candidate_id"], sort=False).agg(
        gutter_y_top=("sliding_window", "min"),
        gutter_y_bottom_sliding=("sliding_window", "max"),  # Temporarily rename to sliding
        final_shape=("gutter_candidate_shape", "last"),  # Get the most recent shape
    ).reset_index()
    
    # Calculate gutter_y_bottom using max y_bottom of words in last sliding window
    if words_df is not None and not words_df.empty and "y_bottom" in words_df.columns:
        # For each gutter, find the max y_bottom of words in the last sliding window
        gutter_bottoms = []
        for _, gutter in gutter_level.iterrows():
            page_num = gutter["page_number"]
            last_sliding_window = gutter["gutter_y_bottom_sliding"]
            
            # Find words in this page and sliding window
            words_in_window = words_df[
                (words_df["page_number"] == page_num) &
                (words_df["sliding_window"] == last_sliding_window)
            ]
            
            if not words_in_window.empty and "y_bottom" in words_in_window.columns:
                max_y_bottom = words_in_window["y_bottom"].max()
                gutter_bottoms.append(max_y_bottom)
            else:
                # Fallback to sliding_window value if no words found
                gutter_bottoms.append(last_sliding_window)
        
        gutter_level["gutter_y_bottom"] = gutter_bottoms
    else:
        # Fallback to sliding_window value if words_df not provided
        gutter_level["gutter_y_bottom"] = gutter_level["gutter_y_bottom_sliding"]
    
    # Drop the temporary sliding window column
    gutter_level = gutter_level.drop(columns=["gutter_y_bottom_sliding"])
    
    # Parse the final shape to get x bounds
    gutter_level[["gutter_x_left", "gutter_x_right"]] = gutter_level["final_shape"].apply(
        lambda s: pd.Series(parse_shape(s))
    )
    
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
    2. All words are short  (< 10 characters each).
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
    
    # For each gutter, collect intersecting shape_ids
    intersecting_ids_list = []
    
    for _, gutter in df.iterrows():
        page_num = gutter["page_number"]
        g_y_min = gutter["gutter_y_top"]
        g_y_max = gutter["gutter_y_bottom"]
        g_x_left = gutter["gutter_x_left"]
        g_x_right = gutter["gutter_x_right"]
        
        # Apply y_padding to expand the vertical range
        g_y_min_padded = g_y_min - y_padding
        g_y_max_padded = g_y_max + y_padding
        
        # Get horizontal lines on the same page
        page_lines = horizontal_lines[horizontal_lines["page_number"] == page_num]
        
        intersecting_shape_ids = []
        
        for _, line in page_lines.iterrows():
            line_y = line["y_top"]
            line_x_left = line["x_left"]
            line_x_right = line["x_right"]
            
            # Check if line's y is within gutter's vertical span (with padding)
            if g_y_min_padded <= line_y <= g_y_max_padded:
                # Check if line's x range overlaps with gutter's x range
                overlap_x_left = max(line_x_left, g_x_left)
                overlap_x_right = min(line_x_right, g_x_right)
                overlap_width = overlap_x_right - overlap_x_left
                
                # Require at least 10pt of overlap
                if overlap_width >= min_x_overlap:
                    # Significant intersection detected!
                    intersecting_shape_ids.append(line["shape_id"])
        
        intersecting_ids_list.append(intersecting_shape_ids)
    
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
        _EPS = 0.5
        left = cross[cross["gutter_x_right"] <= cross["x_left"] + _EPS].copy()
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
        right = cross[cross["gutter_x_left"] >= cross["x_right"] - _EPS].copy()
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

def detect_and_annotate_gutters(df_words: pd.DataFrame, df_shapes: pd.DataFrame) -> pd.DataFrame:
    """
    Detect vertical gutters in the document.
    A gutter is a vertical area without text or shapes that separates (part of) a page into two or more columns.
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame()

    # Exclude line numbers — they are not part of the text layout and must not
    # influence gutter detection
    if "line_number_flag" in df_words.columns:
        df_words = df_words[df_words["line_number_flag"] != True].copy()
        if df_words.empty:
            return pd.DataFrame()

    # Filter to only LTR (left-to-right) text orientation
    if "text_orientation" in df_words.columns:
        df_words = df_words[df_words["text_orientation"] == "LTR"].copy()
        if df_words.empty:
            return pd.DataFrame()

    # 1) Add sliding windows
    df_words = add_sliding_windows(df_words)

    # 2) Add page x bounds
    df_words = add_page_x_bounds(df_words)

    # 3) Add gutter candidates
    df_gutter_candidates = build_gutter_candidate_df(df_words)
    
    # 3.5) Cluster gutter candidates into persistent gutters
    df_gutter_candidates = cluster_gutter_candidates(df_gutter_candidates, df_shapes, df_words)

    # 3.6) Eject page_left/page_right edge windows that are outliers in y-distance
    df_gutter_candidates = eject_outlier_edge_windows(df_gutter_candidates)

    # 4) Promote gutter candidates to actual gutters
    df_gutters = promote_gutter_candidates_to_gutters(df_gutter_candidates, df_words)

    # 4.1) Reject gutters whose left or right side looks like non-content (markers, short, numeric, or repeated)
    df_gutters = reject_non_content_gutters(df_gutters, df_gutter_candidates, df_words)

    # 4.5) Annotate gutters with intersecting horizontal line shape_ids
    df_gutters = filter_gutters_by_horizontal_lines(df_gutters, df_shapes)
    
    # 4.6) Filter gutters with no intersecting lines and assign gutter_id
    df_gutters = filter_and_assign_gutter_ids(df_gutters)

    # 5) Merge gutters onto words
    df_words = merge_gutters_onto_words(df_words, df_gutters)

    return df_words, df_gutter_candidates, df_gutters
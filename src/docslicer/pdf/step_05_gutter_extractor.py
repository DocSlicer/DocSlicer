"""
step_05_gutter_extractor.py
"""

from __future__ import annotations

import pandas as pd

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

_Y_TOP_SLIDING_WINDOW: float = 5.0  # pt
_MIN_GAP_WIDTH: float = 12.0  # pt
_MIN_PAGE_MIN_GAP: float = 100.0  # pt
_TEXT_PADDING: float = 0.0            # pt
_MIN_GUTTER_HEIGHT: float = 50.0  # pt
_MIN_INTERNAL_GAPS: int = 3 # how many internal gaps does a gutter_candidate_id need to have to be a gutter
_MAX_INTERNAL_GAP_DENSITY: int = 3 # those _MIN_INTERNAL_GAPS need to come from gutter_candidate_id with <= 4 internal gaps, otherwise if those gaps only exist within high density areas, its a table



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

    # Per-page previous y_top
    prev_y = df.groupby("page_number", sort=False)["y_top"].shift(1)

    # New bucket starts when:
    #  - first row on page (prev_y is NaN), OR
    #  - y_top jumps by more than the window from the immediately previous row
    # This matches the "accumulate until y_top > initial + window" rule in a single pass.
    is_new_bucket = prev_y.isna() | ((df["y_top"] - prev_y) > _Y_TOP_SLIDING_WINDOW)

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
            # Check if there's a gap in sliding_window_id (missing windows in the data)
            if prev_window_id is not None and window_id != prev_window_id + 1:
                # Complete gap in data - kill all active gutters
                active_gutters.clear()
            
            window_mask = page_df["sliding_window_id"] == window_id
            window_rows = page_df[window_mask]
            
            # Track which gutters get matched in this window (with their updated shapes)
            # Key: gutter_id, Value: (new_x_left, new_x_right)
            matched_gutters = {}
            
            for idx, row in window_rows.iterrows():
                node_x_left = row["gutter_x_left"]
                node_x_right = row["gutter_x_right"]
                
                # Find ALL active gutters that overlap with this candidate
                overlapping_gutters = []
                
                for gutter_id, (g_left, g_right) in active_gutters.items():
                    # Calculate intersection
                    intersection_left = max(node_x_left, g_left)
                    intersection_right = min(node_x_right, g_right)
                    intersection_width = intersection_right - intersection_left
                    
                    # Check if there's overlap
                    if intersection_width > 0:
                        overlapping_gutters.append((gutter_id, intersection_left, intersection_right))
                
                if overlapping_gutters:
                    # This candidate maintains multiple gutters
                    candidate_gutter_ids = []
                    candidate_gutter_shapes = []
                    
                    for gutter_id, new_left, new_right in overlapping_gutters:
                        candidate_gutter_ids.append(gutter_id)
                        candidate_gutter_shapes.append(f"[{new_left:.2f}, {new_right:.2f}]")
                        
                        # Update the matched gutters tracking
                        # If this gutter was already matched by another candidate, keep the intersection
                        if gutter_id in matched_gutters:
                            prev_left, prev_right = matched_gutters[gutter_id]
                            # Take the intersection of the intersections
                            final_left = max(prev_left, new_left)
                            final_right = min(prev_right, new_right)
                            matched_gutters[gutter_id] = (final_left, final_right)
                        else:
                            matched_gutters[gutter_id] = (new_left, new_right)
                    
                    gutter_candidate_ids.append(candidate_gutter_ids)
                    gutter_candidate_shapes.append(candidate_gutter_shapes)
                else:
                    # Start a new gutter
                    gutter_id = next_gutter_id
                    next_gutter_id += 1
                    
                    gutter_candidate_ids.append([gutter_id])
                    gutter_candidate_shapes.append([f"[{node_x_left:.2f}, {node_x_right:.2f}]"])
                    
                    matched_gutters[gutter_id] = (node_x_left, node_x_right)
            
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
                                
                                # Check if line overlaps with gutter's x bounds
                                overlap_left = max(line_x_left, g_left)
                                overlap_right = min(line_x_right, g_right)
                                
                                if overlap_right > overlap_left:
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
# Annotate gutters with intersecting horizontal lines
# ------------------------------

def filter_gutters_by_horizontal_lines(
    gutters_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    min_x_overlap: float = 10.0,
    y_padding: float = 10.0,
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

# ------------------------------
# Build word groups in sliding windows (not needed, just for debug purposes)
# ------------------------------

def build_word_groups_in_sliding_windows(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups words into "word groups" within each (page_number, sliding_window_id).

    Rule:
      - Sort within (page_number, sliding_window_id) by x_left (tie-break: word_id if present)
      - Start a new group whenever the horizontal gap between consecutive words exceeds _MIN_GAP_WIDTH:
            gap = next.x_left - prev.x_right
            new_group if gap > _MIN_GAP_WIDTH

    Output:
      1 row per word-group with aggregated geometry + text:
        - page_number, sliding_window_id, sliding_window
        - word_group_id (1..N per window)
        - group_x_left (min x_left), group_x_right (max x_right)
        - group_y_top (min y_top), group_y_bottom (max y_bottom)
        - group_width, group_height
        - word_count
        - word_ids (list)  [optional but useful]
        - text (joined with single spaces)

    Notes:
      - This is analogous to "line grouping" but constrained to your sliding_window buckets.
      - Requires: page_number, sliding_window_id, x_left, x_right
      - If text/word_id/y_* missing, it will still group and output what it can.
    """
    if words_df is None or words_df.empty:
        return pd.DataFrame()

    df = words_df.copy()

    required = {"page_number", "sliding_window_id", "x_left", "x_right"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keys = ["page_number", "sliding_window_id"]

    # sort for deterministic neighbor gaps
    sort_cols = ["page_number", "sliding_window_id", "x_left"]
    if "word_id" in df.columns:
        sort_cols.append("word_id")

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    prev_x_right = df.groupby(keys, sort=False)["x_right"].shift(1)
    is_new_group = prev_x_right.isna() | ((df["x_left"] - prev_x_right) > _MIN_GAP_WIDTH)

    df["word_group_id"] = (
        is_new_group.groupby([df["page_number"], df["sliding_window_id"]], sort=False)
                    .cumsum()
                    .astype("int64")
    )

    agg = {
        "x_left": ("x_left", "min"),
        "x_right": ("x_right", "max"),
    }

    if "y_top" in df.columns:
        agg["y_top"] = ("y_top", "min")
    if "y_bottom" in df.columns:
        agg["y_bottom"] = ("y_bottom", "max")

    # Carry sliding_window along if present (constant within window)
    if "sliding_window" in df.columns:
        agg["sliding_window"] = ("sliding_window", "first")

    # Keep page bounds if present (constant per page)
    if "x_page_min" in df.columns:
        agg["x_page_min"] = ("x_page_min", "first")
    if "x_page_max" in df.columns:
        agg["x_page_max"] = ("x_page_max", "first")

    # Text aggregation
    if "text" in df.columns:
        agg["text"] = ("text", lambda s: " ".join(map(str, s.tolist())))
    # Word ids aggregation (helpful for debugging / later joins)
    if "word_id" in df.columns:
        agg["word_ids"] = ("word_id", lambda s: s.tolist())

    agg["word_count"] = ("x_left", "size")

    out = (
        df.groupby(["page_number", "sliding_window_id", "word_group_id"], sort=False)
          .agg(**agg)
          .reset_index()
    )

    # rename to clearer group geometry names
    out = out.rename(columns={
        "x_left": "group_x_left",
        "x_right": "group_x_right",
        "y_top": "group_y_top" if "y_top" in out.columns else "y_top",
        "y_bottom": "group_y_bottom" if "y_bottom" in out.columns else "y_bottom",
    })

    out["group_width"] = out["group_x_right"] - out["group_x_left"]
    if "group_y_top" in out.columns and "group_y_bottom" in out.columns:
        out["group_height"] = out["group_y_bottom"] - out["group_y_top"]
    else:
        out["group_height"] = pd.NA

    # Nice column order
    preferred = [
        "page_number",
        "sliding_window_id",
        "sliding_window" if "sliding_window" in out.columns else None,
        "word_group_id",
        "group_x_left",
        "group_x_right",
        "group_width",
        "group_y_top" if "group_y_top" in out.columns else None,
        "group_y_bottom" if "group_y_bottom" in out.columns else None,
        "group_height" if "group_height" in out.columns else None,
        "word_count",
        "text" if "text" in out.columns else None,
        "word_ids" if "word_ids" in out.columns else None,
        "x_page_min" if "x_page_min" in out.columns else None,
        "x_page_max" if "x_page_max" in out.columns else None,
    ]
    preferred = [c for c in preferred if c is not None and c in out.columns]
    remaining = [c for c in out.columns if c not in preferred]
    return out[preferred + remaining]

# ------------------------------
# Merge gutters onto words
# ------------------------------






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

    # 4) Promote gutter candidates to actual gutters
    df_gutters = promote_gutter_candidates_to_gutters(df_gutter_candidates, df_words)

    # 4.5) Annotate gutters with intersecting horizontal line shape_ids
    df_gutters = filter_gutters_by_horizontal_lines(df_gutters, df_shapes)
    
    # 4.6) Filter gutters with no intersecting lines and assign gutter_id
    df_gutters = filter_and_assign_gutter_ids(df_gutters)

    # 5) Group words together to validate gutter candidates
    # df_word_groups = build_word_groups_in_sliding_windows(df_words) -- not needed, just for debug purposes

    # 5) Merge gutters onto words
    #df_words = merge_gutters_onto_words(df_words, df_gutters)

    return df_words, df_gutter_candidates, df_gutters
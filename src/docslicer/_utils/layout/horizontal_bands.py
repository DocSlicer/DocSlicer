"""
Horizontal band assignment.

Public API:
    df = assign_horizontal_bands(df, merge_by_vertical_lines=False)

Accepts any DataFrame that has a `line_id` column — words, cells, or lines.
When multiple rows share the same line_id the per-line geometry (y_top, y_bottom)
is derived vectorized before band assignment and the results are joined back.

Pipeline (internal, always line-level):
    1. Derive one row per line_id: min(y_top), max(y_bottom), first page_number,
       first block_type, first text_orientation.
    2. Sort by (page_number, line_id) — line_id already encodes reading order.
    3. Compute per-line vertical gap.  A large negative gap means the reading
       order jumped to a new column; it opens a new band and resets the
       prev_y_bottom reference.
    4. Compute adaptive gap threshold per page (median of positive gaps ×
       interpolated multiplier).
    5. Assign horizontal_band_id (1-based, monotonically increasing).
    6. Optionally merge bands by shared vertical grid-line IDs (union-find).
    7. Join horizontal_band_id, line_gap, median_gap, page_gap_thresh back
       onto the original input df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

_MIN_PAGE_GAP_THRESH: float = 3.5


# ============================================================
# HELPERS
# ============================================================

def _interpolate_gap_multiplier(
    median_gap: float,
    low_gap:    float = 3.0,
    high_gap:   float = 10.0,
    mult_at_low:  float = 1.60,
    mult_at_high: float = 1.10,
) -> float:
    """
    Adaptive multiplier for the band-gap threshold.

    Tighter line spacing → higher multiplier (more sensitive detection).
    Looser line spacing  → lower multiplier (avoids over-splitting).

        median_gap ≤ low_gap  → mult_at_low
        median_gap ≥ high_gap → mult_at_high
        in between            → linear interpolation
    """
    if median_gap <= low_gap:
        return mult_at_low
    if median_gap >= high_gap:
        return mult_at_high
    t = (median_gap - low_gap) / (high_gap - low_gap)
    return mult_at_low + t * (mult_at_high - mult_at_low)


# ============================================================
# STEP 1: derive one row per line_id
# ============================================================

def _to_line_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per line_id.

    For each line_id takes:
        y_top    → min across rows  (topmost edge of the line)
        y_bottom → max across rows  (bottommost edge)
        page_number, block_type, text_orientation, shape_id_vertical_grid_line
                 → first value (they are constant within a line_id)
    """
    agg: dict[str, object] = {
        "y_top":    ("y_top",    "min"),
        "y_bottom": ("y_bottom", "max"),
    }
    for col in ("page_number", "block_type", "text_orientation"):
        if col in df.columns:
            agg[col] = (col, "first")

    line_df = df.groupby("line_id", sort=False).agg(**agg).reset_index()

    if "shape_id_vertical_grid_line" in df.columns:
        vlines = (
            df.groupby("line_id", sort=False)["shape_id_vertical_grid_line"]
            .apply(lambda s: list({v for cell in s if isinstance(cell, list) for v in cell}
                                  or ({s.dropna().iloc[0]} if not s.isna().all() else set())))
            .reset_index()
        )
        line_df = line_df.merge(vlines, on="line_id", how="left")

    return line_df


# ============================================================
# STEP 2–5: gap computation + band assignment (one page at a time)
# ============================================================

def _process_page(
    ltr_df:      pd.DataFrame,
    band_counter: int,
) -> tuple[dict[object, float], dict[object, int], int, float, float]:
    """
    Compute line_gap and horizontal_band_id for the LTR lines of one page.

    Parameters
    ----------
    ltr_df : already sorted by line_id, non-TTB/BTT lines only
    band_counter : running global band counter

    Returns
    -------
    gaps       : {line_id → gap}
    band_ids   : {line_id → band_id}
    band_counter : updated counter
    median_gap, threshold
    """
    line_ids  = ltr_df["line_id"].to_numpy()
    y_top_arr = ltr_df["y_top"].to_numpy(dtype=float)
    y_bot_arr = ltr_df["y_bottom"].to_numpy(dtype=float)
    bt_arr    = (
        ltr_df["block_type"].astype(str).str.lower().to_numpy()
        if "block_type" in ltr_df.columns
        else np.full(len(ltr_df), "", dtype=object)
    )
    n = len(line_ids)

    # ── Phase 1: compute gaps ────────────────────────────────────────────────
    raw_gaps   = np.zeros(n, dtype=float)
    page_gaps: list[float] = []
    prev_yb: float | None  = None

    for i in range(n):
        gap = (float(y_top_arr[i]) - prev_yb) if prev_yb is not None else 0.0
        raw_gaps[i] = gap
        page_gaps.append(gap if gap > 0 else 2.0)
        prev_yb = float(y_bot_arr[i])

    # ── Phase 2: adaptive threshold ─────────────────────────────────────────
    if page_gaps:
        median_gap = float(np.median(page_gaps))
        threshold  = max(_MIN_PAGE_GAP_THRESH, _interpolate_gap_multiplier(median_gap) * median_gap)
    else:
        median_gap = 0.0
        threshold  = float("inf")

    # ── Phase 3: assign band IDs ─────────────────────────────────────────────
    band_arr  = np.zeros(n, dtype=int)
    cur_band  = 0
    prev_yb   = None

    for i in range(n):
        gap = raw_gaps[i]
        yb  = float(y_bot_arr[i])

        if bt_arr[i] == "page_label":
            band_counter += 1
            band_arr[i]   = band_counter
            cur_band       = 0
            prev_yb        = yb
            continue

        is_col_jump = (gap < -threshold) and (prev_yb is not None)
        if cur_band == 0 or gap > threshold or is_col_jump:
            band_counter += 1
            cur_band       = band_counter

        band_arr[i] = cur_band
        prev_yb = yb

    gaps_map     = dict(zip(line_ids, raw_gaps))
    band_ids_map = dict(zip(line_ids, band_arr))
    return gaps_map, band_ids_map, band_counter, median_gap, threshold


# ============================================================
# STEP 6: merge bands by shared vertical lines (optional)
# ============================================================

def _merge_bands_by_shared_vertical_lines(
    line_df: pd.DataFrame,
    min_shared_lines: int = 1,
) -> pd.DataFrame:
    """
    For each page, merge bands that share ≥ min_shared_lines vertical grid-line IDs.
    Uses union-find; merged bands adopt the smallest band_id.
    Returns line_df unchanged if shape_id_vertical_grid_line is absent.
    """
    if "shape_id_vertical_grid_line" not in line_df.columns:
        return line_df

    df = line_df.copy()

    for _, page_df in df.groupby("page_number", sort=False):
        page_idx = page_df.index

        band_to_vlines: dict[int, set] = {}
        for idx in page_idx:
            bid = df.at[idx, "horizontal_band_id"]
            if pd.isna(bid) or bid < 0:
                continue
            bid    = int(bid)
            vlines = df.at[idx, "shape_id_vertical_grid_line"]
            if vlines is None or (isinstance(vlines, float) and pd.isna(vlines)):
                continue
            if not isinstance(vlines, list):
                vlines = [vlines]
            band_to_vlines.setdefault(bid, set()).update(vlines)

        bids = sorted(band_to_vlines)
        if len(bids) < 2:
            continue

        parent = {b: b for b in bids}

        def find(b: int) -> int:
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                b1, b2 = bids[i], bids[j]
                if len(band_to_vlines[b1] & band_to_vlines[b2]) >= min_shared_lines:
                    r1, r2 = find(b1), find(b2)
                    if r1 != r2:
                        parent[max(r1, r2)] = min(r1, r2)

        for idx in page_idx:
            bid = df.at[idx, "horizontal_band_id"]
            if pd.isna(bid) or bid < 0:
                continue
            bid = int(bid)
            if bid in parent:
                df.at[idx, "horizontal_band_id"] = find(bid)

    return df


# ============================================================
# PUBLIC API
# ============================================================

def assign_horizontal_bands(
    df: pd.DataFrame,
    merge_by_vertical_lines: bool = False,
    min_shared_lines: int = 1,
) -> pd.DataFrame:
    """
    Assign horizontal_band_id to every row of df.

    Accepts any DataFrame that has a `line_id` column.  When multiple rows share
    the same line_id (e.g. words or cells), per-line geometry is derived
    vectorized (min y_top, max y_bottom) before band assignment, and results are
    joined back onto the original rows.

    Assumptions
    -----------
    - line_id encodes reading order (monotonically increasing in the order lines
      should be read).  Column jumps appear as large negative y gaps and
      automatically open a new band.
    - page_number is present and identifies page boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Words, cells, or lines.  Required: line_id, page_number, y_top, y_bottom.
        Optional: block_type, text_orientation, shape_id_vertical_grid_line.
    merge_by_vertical_lines : bool
        Merge bands sharing vertical grid-line IDs (union-find).  Use True for
        PDF (reliable line detection).  Leave False for OCR — line detection is
        too noisy and risks collapsing an entire page into one band.
    min_shared_lines : int
        Minimum shared vertical line IDs required to merge two bands.
        Only relevant when merge_by_vertical_lines=True.

    Returns
    -------
    df with added columns (joined by line_id):
        horizontal_band_id  int    1-based, monotonically increasing
        line_gap            float  vertical gap above this line (pt)
        median_gap          float  per-page median of positive line gaps
        page_gap_thresh     float  adaptive gap threshold used for this page
    """
    if df.empty:
        return df.assign(
            horizontal_band_id=0,
            line_gap=np.nan,
            median_gap=0.0,
            page_gap_thresh=0.0,
        )

    # ── Step 1: one row per line ─────────────────────────────────────────────
    line_df = _to_line_df(df)

    if "page_number" not in line_df.columns:
        raise ValueError("df must contain a 'page_number' column")

    # ── Step 2–5: per-page band assignment ───────────────────────────────────
    line_df["line_gap"]           = np.nan
    line_df["median_gap"]         = 0.0
    line_df["horizontal_band_id"] = 0
    line_df["page_gap_thresh"]    = 0.0

    vert_mask = (
        line_df["text_orientation"].isin(["TTB", "BTT"])
        if "text_orientation" in line_df.columns
        else pd.Series(False, index=line_df.index)
    )

    band_counter = 0

    for _, page_df in line_df.groupby("page_number", sort=True):
        page_vert_mask = vert_mask.loc[page_df.index]
        vert_idx       = page_df.index[page_vert_mask].tolist()
        ltr_df         = page_df[~page_vert_mask].sort_values("line_id", kind="mergesort")

        if ltr_df.empty:
            median_gap = 0.0
            threshold  = float("inf")
        else:
            gaps_map, band_map, band_counter, median_gap, threshold = _process_page(
                ltr_df, band_counter
            )
            ltr_idx = ltr_df.index.tolist()
            line_df.loc[ltr_idx, "line_gap"]           = [gaps_map[lid] for lid in ltr_df["line_id"]]
            line_df.loc[ltr_idx, "median_gap"]         = median_gap
            line_df.loc[ltr_idx, "horizontal_band_id"] = [band_map[lid] for lid in ltr_df["line_id"]]
            line_df.loc[ltr_idx, "page_gap_thresh"]    = threshold

        for idx in (line_df.loc[vert_idx].sort_values("line_id").index if vert_idx else []):
            band_counter += 1
            line_df.at[idx, "median_gap"]         = median_gap
            line_df.at[idx, "page_gap_thresh"]    = threshold
            line_df.at[idx, "horizontal_band_id"] = band_counter

    # ── Step 6: optional merge by vertical lines ─────────────────────────────
    if merge_by_vertical_lines:
        line_df = _merge_bands_by_shared_vertical_lines(line_df, min_shared_lines=min_shared_lines)

    # ── Step 7: join back onto input df ─────────────────────────────────────
    band_cols = line_df.set_index("line_id")[
        ["horizontal_band_id", "line_gap", "median_gap", "page_gap_thresh"]
    ]
    out = df.copy()
    for col in band_cols.columns:
        out[col] = df["line_id"].map(band_cols[col])

    return out

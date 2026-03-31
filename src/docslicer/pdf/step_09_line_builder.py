"""
step_09_line_builder.py

Build lines from cells and assign horizontal band IDs.

Public API:
    df_lines, df_cells = build_lines(df_cells)

Pipeline:
    Step 1: Aggregate cells → df_lines
        - Groups cells by line_id
        - Merges text, geometry, style, counts
        - Aggregates shape_id_vertical_line (union per line)
        - Creates bracketed 'cells' column: "[cell1] [cell2] [cell3]"

    Step 2: Assign horizontal bands
        - Groups lines into horizontal bands based on vertical gaps
        - Gutter-aware: right-column lines compute their zone-entry gap
          vs the last singlecol content before the gutter started, not vs
          the end of the parallel left column
        - Adds: line_gap, horizontal_band_id, page_gap_thresh to df_lines

    Step 3: Merge bands by shared vertical lines
        - Bands sharing at least one vertical line ID are merged
        - Uses union-find; merged bands take the smallest band_id

    Step 4: Propagate horizontal_band_id to df_cells via line_id join
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from docslicer._utils.hierarchical_aggregator import (
    aggregate_hierarchical,
    build_standard_agg_spec,
    _collect_unique_list,
)


# ============================================================
# CONFIG
# ============================================================

# Minimum gap threshold (pts) applied after interpolation — prevents over-splitting
# on tightly packed text where the computed threshold would be very small.
_MIN_PAGE_GAP_THRESH: float = 3.5

# Boolean flag columns — "max" means True if any cell has it.
_FLAG_COLS = [
    "has_link",
    "is_underlined",
    "has_vertical_line",
]



# ============================================================
# HELPERS
# ============================================================

def _remove_bracketed_text(text: str) -> str:
    """
    Remove text within brackets (parentheses, square brackets, curly braces).
    Used for uppercase detection to ignore mixed-case content in brackets.

    Example: "RECENT NOTABLE DEVELOPMENTS (Since August 5, 2025)" -> "RECENT NOTABLE DEVELOPMENTS "
    """
    if not text:
        return text
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    return text


def _calc_uppercase_ratio(text: str) -> float:
    """Return fraction of alphabetic characters that are uppercase, ignoring bracketed text."""
    if not text or not isinstance(text, str):
        return 0.0
    cleaned = _remove_bracketed_text(text)
    alpha = [c for c in cleaned if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)



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


def _na(val) -> bool:
    """True if val is None or a float NaN."""
    if val is None:
        return True
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


# ============================================================
# STEP 1: Aggregate cells → lines
# ============================================================

def _aggregate_cells_to_lines(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Group cells by line_id to produce one row per line.

    Geometry: union (min x_left, max x_right, min y_top, max y_bottom).
    Text:     cells joined with a space, left→right order.
    Cells:    bracketed "[cell1] [cell2] ..." representation.
    Counts:   summed.
    bold/italic ratios: char-count-weighted average across cells.
    Style:    most common value across cells.
    Flags:    True if any cell has the flag (max).
    shape_id_vertical_line: union of all vertical line IDs across cells.
    """
    df = df_cells.sort_values(["line_id", "x_left", "y_top"], kind="mergesort").copy()

    # Ensure flag columns are present (documents without links/underlines/etc).
    for col in _FLAG_COLS:
        if col not in df.columns:
            df[col] = False

    agg_spec = build_standard_agg_spec(
        include_hierarchy=False,
        include_html_provenance=False,
        include_table=False,
        extra_agg={
            "text": lambda s: " ".join(t for t in s.astype(str) if t.strip()),
        },
        count_col="cell_id",
    )
    if "shape_id_vertical_line" in df.columns:
        agg_spec["shape_id_vertical_line"] = _collect_unique_list

    grouped = aggregate_hierarchical(
        df,
        group_col="line_id",
        agg_spec=agg_spec,
        rename_count_col={"cell_id": "cell_count"},
    )

    # Override is_uppercase: use bracketed-text-aware method for accuracy.
    grouped["is_uppercase"] = grouped["text"].apply(lambda t: _calc_uppercase_ratio(t) > 0.90)

    # Bracketed cells column — separate groupby to avoid lambda collision above.
    cells_col = (
        df.groupby("line_id", sort=False)["text"]
        .apply(lambda s: " ".join(f"[{t}]" for t in s.astype(str) if t.strip()))
        .rename("cells")
        .reset_index()
    )
    grouped = grouped.merge(cells_col, on="line_id", how="left")

    return grouped


# ============================================================
# STEP 2: Assign horizontal bands
# ============================================================

def _assign_horizontal_bands(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Assign horizontal_band_id to lines based on vertical gaps.

    Adds columns
    ------------
        line_gap            float  vertical gap above this line (pt)
        median_gap          float  per-page median of line_gaps (negatives clamped to 2.0)
        page_gap_thresh     float  median_gap × _interpolate_gap_multiplier(median_gap)
        horizontal_band_id  int    1-based, increments across the document

    line_gap is computed with full gutter awareness:
        singlecol / left-column lines
            gap = y_top − y_bottom of the previous line (natural order)
        right-column (non-left) FIRST line in a zone
            gap = y_top − y_bottom of the last line before the zone started
            (avoids the massive negative gap that would arise from comparing
             against the end of the parallel left column)
        right-column continuation lines
            gap = y_top − y_bottom of the previous line in the same column

    All line_gaps are included in the page median. Negative gaps (overlapping
    lines in tightly-packed PDFs) are clamped to 2.0 before computing the
    median, preventing a handful of large gaps from dominating and producing
    an enormous threshold.
    """
    if df_lines.empty:
        return df_lines.assign(
            line_gap=np.nan,
            median_gap=0.0,
            horizontal_band_id=0,
            page_gap_thresh=0.0,
        )

    has_gl = "gutter_id_left"  in df_lines.columns
    has_gr = "gutter_id_right" in df_lines.columns
    has_rc = "reading_column"  in df_lines.columns

    df = df_lines.copy()
    df["line_gap"]           = np.nan
    df["median_gap"]         = 0.0
    df["horizontal_band_id"] = 0
    df["page_gap_thresh"]    = 0.0

    band_counter = 0  # increments globally across pages

    for _, page_df in df.groupby("page_number", sort=True):

        # Sort lines top-to-bottom for this page.
        sorted_idx = page_df.sort_values("y_top").index.tolist()
        n = len(sorted_idx)

        # Pre-extract arrays — avoids per-row .loc overhead inside the loops.
        y_top_arr    = df.loc[sorted_idx, "y_top"].to_numpy(dtype=float)
        y_bottom_arr = df.loc[sorted_idx, "y_bottom"].to_numpy(dtype=float)

        if has_rc:
            rc_arr = df.loc[sorted_idx, "reading_column"].fillna(1).astype(int).to_numpy()
        else:
            rc_arr = np.ones(n, dtype=int)

        # Build gutter_id array (None = singlecol).
        gutter_id_arr: list = []
        for idx in sorted_idx:
            row = df.loc[idx]
            gl = row["gutter_id_left"]  if has_gl else None
            gr = row["gutter_id_right"] if has_gr else None
            gl = None if _na(gl) else gl
            gr = None if _na(gr) else gr
            gutter_id_arr.append(gl if gl is not None else gr)

        # ----------------------------------------------------------------
        # Phase 1 — compute line_gap for each line
        # ----------------------------------------------------------------
        gaps = np.zeros(n, dtype=float)
        page_gaps: list[float] = []  # all positive gaps → page median

        prev_all_y_bottom:  float | None = None   # high-water: every line
        current_gutter_id:  object       = None
        pre_zone_y_bottom:  float | None = None   # snapshot when zone started
        per_col_y_bottom:   dict         = {}     # {(gutter_id, rc): y_bottom}

        for i in range(n):
            gid      = gutter_id_arr[i]
            col_key  = (gid, int(rc_arr[i]))
            yt       = float(y_top_arr[i])
            yb       = float(y_bottom_arr[i])

            if gid is None:
                # ── Singlecol line ──────────────────────────────────────
                gap = (yt - prev_all_y_bottom) if prev_all_y_bottom is not None else 0.0
                gaps[i] = gap
                current_gutter_id = None  # exit any active zone

            else:
                # ── Gutter line ─────────────────────────────────────────
                if gid != current_gutter_id:
                    # Entering a new zone: snapshot the current high-water.
                    current_gutter_id = gid
                    pre_zone_y_bottom = prev_all_y_bottom

                if col_key not in per_col_y_bottom:
                    # First line of this column in the zone.
                    # Gap vs the content that existed before the zone started
                    # (not vs the end of the parallel column).
                    gap = (yt - pre_zone_y_bottom) if pre_zone_y_bottom is not None else 0.0
                    gaps[i] = gap
                else:
                    # Continuation within the same column.
                    gap = yt - per_col_y_bottom[col_key]
                    gaps[i] = gap

                per_col_y_bottom[col_key] = yb

            page_gaps.append(gap if gap > 0 else 2.0)

            # Update the global high-water mark (used for next gap reference).
            prev_all_y_bottom = (
                yb if prev_all_y_bottom is None else max(prev_all_y_bottom, yb)
            )

        # ----------------------------------------------------------------
        # Phase 2 — compute adaptive gap threshold for this page
        # ----------------------------------------------------------------
        if page_gaps:
            median_gap = float(np.median(page_gaps))
            threshold  = max(_MIN_PAGE_GAP_THRESH, _interpolate_gap_multiplier(median_gap) * median_gap)
        else:
            median_gap = 0.0
            threshold  = float("inf")  # single line or empty page → one band

        # ----------------------------------------------------------------
        # Phase 3 — assign horizontal_band_id
        #
        # Rules:
        #   Singlecol    gap > threshold  → new band
        #   Gutter, first column of zone  → new band if gap > threshold
        #   Gutter, second+ column        → join the band started for that zone
        #   Gutter, within-column cont.   → new band if gap > threshold
        # ----------------------------------------------------------------
        band_ids = np.zeros(n, dtype=int)

        current_band      = 0
        current_gutter_id = None
        zone_band:  dict  = {}   # gutter_id → band_id the zone started on
        per_col_seen: set = set()

        for i in range(n):
            gid     = gutter_id_arr[i]
            col_key = (gid, int(rc_arr[i]))

            if gid is None:
                # Singlecol: start new band on gap > threshold or first line.
                if current_band == 0 or gaps[i] > threshold:
                    band_counter += 1
                    current_band  = band_counter
                current_gutter_id = None

            else:
                if gid != current_gutter_id:
                    current_gutter_id = gid

                if col_key not in per_col_seen:
                    per_col_seen.add(col_key)
                    if gid not in zone_band:
                        # First column entering this zone.
                        if current_band == 0 or gaps[i] > threshold:
                            band_counter += 1
                            current_band  = band_counter
                        zone_band[gid] = current_band
                    else:
                        # Second+ column: join the zone's existing band.
                        current_band = zone_band[gid]
                else:
                    # Within-column continuation: normal gap check.
                    if gaps[i] > threshold:
                        band_counter += 1
                        current_band      = band_counter
                        zone_band[gid]    = current_band  # zone's band moves forward

            band_ids[i] = current_band

        # ----------------------------------------------------------------
        # Write results back to df (bulk assignment)
        # ----------------------------------------------------------------
        df.loc[sorted_idx, "line_gap"]           = gaps
        df.loc[sorted_idx, "median_gap"]         = median_gap
        df.loc[sorted_idx, "horizontal_band_id"] = band_ids
        df.loc[sorted_idx, "page_gap_thresh"]    = threshold

    return df


# ============================================================
# STEP 3: Merge bands by shared vertical lines
# ============================================================

def _merge_bands_by_shared_vertical_lines(
    df_lines: pd.DataFrame,
    min_shared_lines: int = 1,
) -> pd.DataFrame:
    """
    Merge bands that share vertical lines.

    For each page, finds bands that share at least `min_shared_lines`
    vertical line IDs and merges them to use the same band_id (the smallest).

    Requires column: shape_id_vertical_line (list[int] | None per line),
    horizontal_band_id, page_number.  If the column is absent, returns df
    unchanged.
    """
    if df_lines.empty or "shape_id_vertical_line" not in df_lines.columns:
        return df_lines

    df = df_lines.copy()

    for page_num, _ in df.groupby("page_number", sort=False):
        page_mask    = df["page_number"] == page_num
        page_indices = df.index[page_mask]

        # Build band → set of vertical line IDs
        band_to_vlines: dict[int, set] = {}
        for idx in page_indices:
            band_id = df.at[idx, "horizontal_band_id"]
            if pd.isna(band_id) or band_id < 0:
                continue
            band_id = int(band_id)
            vlines  = df.at[idx, "shape_id_vertical_line"]
            if vlines is None or (isinstance(vlines, float) and pd.isna(vlines)):
                continue
            if not isinstance(vlines, list):
                vlines = [vlines]
            band_to_vlines.setdefault(band_id, set()).update(vlines)

        band_ids = sorted(band_to_vlines)
        if len(band_ids) < 2:
            continue

        # Union-find
        parent = {bid: bid for bid in band_ids}

        def find_root(bid: int) -> int:
            if parent[bid] != bid:
                parent[bid] = find_root(parent[bid])
            return parent[bid]

        def union(bid1: int, bid2: int) -> None:
            r1, r2 = find_root(bid1), find_root(bid2)
            if r1 != r2:
                if r1 < r2:
                    parent[r2] = r1
                else:
                    parent[r1] = r2

        for i in range(len(band_ids)):
            for j in range(i + 1, len(band_ids)):
                b1, b2 = band_ids[i], band_ids[j]
                if len(band_to_vlines[b1] & band_to_vlines[b2]) >= min_shared_lines:
                    union(b1, b2)

        # Apply merges
        for idx in page_indices:
            band_id = df.at[idx, "horizontal_band_id"]
            if pd.isna(band_id) or band_id < 0:
                continue
            band_id = int(band_id)
            if band_id in parent:
                df.at[idx, "horizontal_band_id"] = find_root(band_id)

    return df


# ============================================================
# PUBLIC API
# ============================================================

def build_lines(
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lines from cells and assign horizontal band IDs.

    Parameters
    ----------
    df_cells : pd.DataFrame
        One row per cell. Output of step_07_cell_builder.build_cells().
        Must contain: line_id, cell_id, x_left, x_right, y_top, y_bottom,
        text, bold_ratio, italic_ratio, char_count, page_width.

    Returns
    -------
    df_lines : pd.DataFrame
        One row per line_id.  Contains aggregated content (text, geometry,
        style, counts) plus line_gap, page_gap_thresh, and horizontal_band_id.

    df_cells : pd.DataFrame
        Input df_cells with horizontal_band_id column added.
    """
    if df_cells.empty:
        return pd.DataFrame(), df_cells.copy()

    df_lines = _aggregate_cells_to_lines(df_cells)
    df_lines = _assign_horizontal_bands(df_lines)
    df_lines = _merge_bands_by_shared_vertical_lines(df_lines)

    # Propagate horizontal_band_id to cells via line_id join.
    band_map = df_lines.set_index("line_id")["horizontal_band_id"].to_dict()
    df_cells = df_cells.copy()
    df_cells["horizontal_band_id"] = df_cells["line_id"].map(band_map)

    return df_lines, df_cells

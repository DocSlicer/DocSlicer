"""
step_11_table_builder.py

Classify bands into text or table, and for tables derive the full layout.

Public API:
    df_lines, df_cells, df_table_cells = build_tables(df_lines, df_cells, df_words=None)

Pipeline (northstar — steps marked [TODO] are not yet implemented):
    Step 1: Infer column grid                                         [DONE]
        Input:  df_cells with horizontal_band_id, x_left, x_right, line_id
        Output: col_start, col_end, colspan, band_total_cols added to df_cells
        - Within each (page_number, horizontal_band_id), infers a column grid
          by seeding from the densest line and walking up/down

    Step 2: Classify horizontal bands as table or text                [DONE]
        Input:  df_cells with col_start, colspan, band_total_cols, horizontal_band_id
        Output: layout_type ("table"|"text"), layout_id, band_table_score
                added to df_cells and df_lines
        - Bands are scored on grid quality signals (column reuse, atomic cell
          ratio, gap patterns, prose penalties) and thresholded at score ≥ 3.0
        - layout_id is currently set equal to horizontal_band_id (no merging yet)

    Step 3: Eject header rows from table bands                        [DONE
        Input:  df_cells with layout_type, col_start, colspan, band_total_cols
        Output: ejected lines reclassified as layout_type="text"
        - A single-cell, left-aligned line at the top of a table band is often
          a section header that was pulled into the band; it should be split off
          and reclassified as text before further table processing

    Step 4: Merge consecutive table bands → layout_id                 [DONE]
        Input:  df_cells with layout_type, band_total_cols, col_start/end, y_top/bottom
        Output: layout_id updated so adjacent table bands with matching column
                structure and small vertical gap share the same layout_id
        - Consecutive table bands on the same page with equal band_total_cols,
          aligned column boundaries, and vertical gap ≤ ~8 pt are merged
        - Text bands always keep their own layout_id

    Step 5: Build table structure — rows, rowspan, cell roles          [TODO]
        Input:  df_cells with layout_type="table", layout_id, col_start, col_end, colspan
        Output: df_table_cells with one row per assembled table cell containing:
                table_cell_id, layout_id, table_id, row_start, col_start,
                rowspan, colspan, text, text_raw_lines, cell_ids, line_ids, role
        - Iterates line_ids in order within each (layout_id, page_number),
          accumulating cells into rows; flushes a row when column coverage
          is complete or an underline boundary is crossed
        - Rowspan: columns missing from a row extend the previous row's cell
        - Roles assigned: header, row_label, value_numeric, value_text, footnote
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.table_utils import detect_cell_roles

# ============================================================
# CONFIG
# ============================================================

_MAX_VERTICAL_GAP = 8.0   # max gap (pt) between bands for layout merging


# ============================================================
# STEP 1: Infer column grid
# ============================================================

def _assign_column_layout(
    df_cells: pd.DataFrame,
    group_col: str = "horizontal_band_id",
) -> pd.DataFrame:
    """
    Infer a column grid within each (page_number, group_col) group and assign
    col_start, col_end, colspan, band_total_cols to every cell.

    group_col is normally "horizontal_band_id" (initial pass) but is set to
    "layout_id" when re-running after tables have been merged (step 4).

    Algorithm (per group):
        1. Seed columns from the line with the most cells.
        2. Walk lines below the seed (DOWN), then above (UP).
           - 0 overlaps → insert new column.
           - 1 overlap  → extend that column's x range.
           - ≥2 overlaps → colspan (no column mutation).
        3. SPLIT detection: if a single column receives ≥2 disjoint same-line
           cells that each hit only that column, split it into subcolumns.
    """
    # Columns are stored as two parallel Python lists (faster than list-of-dicts).
    def _find_hits(x_left, x_right, col_lefts, col_rights):
        hits = []
        for i in range(len(col_lefts)):
            if x_left < col_rights[i] and col_lefts[i] < x_right:
                hits.append(i)
        return hits

    def _process_cell(x_left, x_right, col_lefts, col_rights):
        hits = _find_hits(x_left, x_right, col_lefts, col_rights)
        if not hits:
            pos = len(col_lefts)
            for i, cl in enumerate(col_lefts):
                if x_right <= cl:
                    pos = i
                    break
            col_lefts.insert(pos, x_left)
            col_rights.insert(pos, x_right)
            return [pos]
        if len(hits) == 1:
            i = hits[0]
            col_lefts[i]  = min(x_left,  col_lefts[i])
            col_rights[i] = max(x_right, col_rights[i])
        return hits

    def _maybe_split(xl_arr, xr_arr, col_lefts, col_rights):
        # xl_arr / xr_arr are numpy float arrays — no itertuples overhead.
        sole: dict[int, list[tuple[float, float]]] = {}
        for xl, xr in zip(xl_arr, xr_arr):
            hits = _find_hits(xl, xr, col_lefts, col_rights)
            if len(hits) == 1:
                sole.setdefault(hits[0], []).append((xl, xr))
        splits = []
        for col_idx, intervals in sole.items():
            if len(intervals) < 2:
                continue
            ivs = sorted(intervals)
            segs: list[tuple[float, float]] = []
            cl, cr = ivs[0]
            for a, b in ivs[1:]:
                if cl < b and a < cr:  # ranges_overlap
                    cr = max(cr, b)
                else:
                    segs.append((cl, cr))
                    cl, cr = a, b
            segs.append((cl, cr))
            if len(segs) > 1:
                splits.append((col_idx, segs))
        for col_idx, segs in sorted(splits, key=lambda t: t[0], reverse=True):
            col_lefts[col_idx:col_idx + 1]  = [s for s, _ in segs]
            col_rights[col_idx:col_idx + 1] = [e for _, e in segs]

    result = df_cells.copy()
    # Defaults are correct for single-cell bands; multi-cell bands override below.
    result["col_start"]       = 0
    result["col_end"]         = 0
    result["colspan"]         = 1
    result["band_total_cols"] = 1

    # Max cells per line within each band — only bands with >1 need grid inference.
    # Use cell_count propagated from line_builder if available; fall back to groupby.
    if "cell_count" in result.columns:
        band_max = result.groupby(["page_number", group_col])["cell_count"].max()
    else:
        line_sizes = result.groupby("line_id")["cell_id"].transform("count")
        result["_tmp_line_size"] = line_sizes
        band_max = result.groupby(["page_number", group_col])["_tmp_line_size"].max()
        result = result.drop(columns=["_tmp_line_size"])

    multi_cell_bands = set(band_max[band_max > 1].index.tolist())

    for (page, band), band_df in result.groupby(["page_number", group_col]):
        if (page, band) not in multi_cell_bands:
            continue  # already initialised with correct single-cell values

        if "cell_count" in band_df.columns:
            line_cell_counts = band_df.groupby("line_id")["cell_count"].first()
        else:
            line_cell_counts = band_df.groupby("line_id").size()
        seed_line_id     = line_cell_counts.idxmax()

        # Pre-group by line_id once; store sorted numpy arrays to avoid
        # repeated DataFrame filtering and itertuples overhead.
        line_groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for lid, grp in band_df.groupby("line_id"):
            grp_sorted = grp.sort_values("x_left")
            line_groups[lid] = (
                grp_sorted["x_left"].to_numpy(dtype=float),
                grp_sorted["x_right"].to_numpy(dtype=float),
            )

        seed_xl, seed_xr = line_groups[seed_line_id]
        col_lefts:  list[float] = list(seed_xl)
        col_rights: list[float] = list(seed_xr)

        all_line_ids = sorted(band_df["line_id"].unique())
        seed_idx     = all_line_ids.index(seed_line_id)
        lines_below  = all_line_ids[seed_idx + 1:]
        lines_above  = list(reversed(all_line_ids[:seed_idx]))

        for line_id in lines_below + lines_above:
            xl_arr, xr_arr = line_groups[line_id]
            _maybe_split(xl_arr, xr_arr, col_lefts, col_rights)
            for xl, xr in zip(xl_arr, xr_arr):
                _process_cell(xl, xr, col_lefts, col_rights)

        total_cols = len(col_lefts)

        # Vectorised final assignment: broadcast (n_cells, 1) vs (1, n_cols).
        # This replaces the per-cell result.loc[mask] loop which was O(n²).
        band_xl = band_df["x_left"].to_numpy(dtype=float)
        band_xr = band_df["x_right"].to_numpy(dtype=float)

        if total_cols > 0:
            cl_np = np.array(col_lefts)
            cr_np = np.array(col_rights)
            overlap = (band_xl[:, None] < cr_np[None, :]) & \
                      (cl_np[None, :] < band_xr[:, None])   # (n_cells, n_cols)
            has_hit = overlap.any(axis=1)
            cs_arr  = np.where(has_hit, overlap.argmax(axis=1), total_cols)
            ce_arr  = np.where(has_hit,
                               total_cols - 1 - np.fliplr(overlap).argmax(axis=1),
                               total_cols)
        else:
            cs_arr = np.full(len(band_df), total_cols, dtype=np.intp)
            ce_arr = np.full(len(band_df), total_cols, dtype=np.intp)

        # Assign to result in one shot using the original DataFrame index.
        result.loc[band_df.index, "col_start"]       = cs_arr
        result.loc[band_df.index, "col_end"]         = ce_arr
        result.loc[band_df.index, "colspan"]         = ce_arr - cs_arr + 1
        result.loc[band_df.index, "band_total_cols"] = total_cols

    return result


# ============================================================
# STEP 2: Classify Horizontal Bands (into Table or Text)
# ============================================================

def _compute_line_grid_metrics(
    df_cells: pd.DataFrame,
    df_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Precompute all line-level metrics used by the band classifier.

    This replaces the old nested band→line loops. Most work is now dataframe-wide
    groupby/NumPy arithmetic, with only a small string aggregation for row
    patterns.
    """
    if df_cells.empty:
        return pd.DataFrame()

    df = df_cells.sort_values(
        ["horizontal_band_id", "line_id", "x_left"],
        kind="mergesort",
    ).copy()

    text = df["text"].fillna("").astype(str)
    df["_cell_word_count"] = text.str.count(r"\S+").astype("int32")
    df["_digit_count"] = text.str.count(r"\d").astype("int32")
    df["_char_count"] = text.str.len().clip(lower=1).astype("int32")

    same_line = df["line_id"].eq(df["line_id"].shift())
    prev_x_right = df["x_right"].shift()
    df["_gap"] = np.where(
        same_line,
        df["x_left"].to_numpy(dtype=float) - prev_x_right.to_numpy(dtype=float),
        np.nan,
    )
    df.loc[df["_gap"] <= 0, "_gap"] = np.nan

    line_gb = df.groupby("line_id", sort=False)
    metrics = line_gb.agg(
        horizontal_band_id=("horizontal_band_id", "first"),
        cell_count=("cell_id", "size"),
        x_left=("x_left", "min"),
        x_right=("x_right", "max"),
        total_cols=("band_total_cols", "max"),
        max_colspan=("colspan", "max"),
        digit_count=("_digit_count", "sum"),
        char_count=("_char_count", "sum"),
        median_cell_words=("_cell_word_count", "median"),
        max_gap=("_gap", "max"),
        median_gap=("_gap", "median"),
    )

    gap_std = line_gb["_gap"].std(ddof=0).rename("gap_std")
    gap_count = line_gb["_gap"].count().rename("gap_count")
    metrics = metrics.join([gap_std, gap_count])

    metrics[["max_gap", "median_gap", "gap_std"]] = (
        metrics[["max_gap", "median_gap", "gap_std"]].fillna(0.0)
    )
    metrics["gap_cv"] = np.where(
        metrics["gap_count"].to_numpy(dtype=float) > 1,
        metrics["gap_std"].to_numpy(dtype=float)
        / (metrics["median_gap"].to_numpy(dtype=float) + 1e-6),
        0.0,
    )

    metrics["line_width"] = metrics["x_right"] - metrics["x_left"]
    metrics["digit_ratio"] = (
        metrics["digit_count"] / metrics["char_count"].replace(0, np.nan)
    ).fillna(0.0)

    if "page_width" in df_lines.columns:
        page_width = df_lines.set_index("line_id")["page_width"]
        metrics["page_width"] = metrics.index.map(page_width).astype(float)
        metrics["width_ratio"] = (
            metrics["line_width"] / metrics["page_width"].replace(0, np.nan)
        ).fillna(0.0)
    elif "width" in df_lines.columns:
        line_width = df_lines.set_index("line_id")["width"]
        metrics["source_width"] = metrics.index.map(line_width).astype(float)
        metrics["width_ratio"] = (
            metrics["source_width"] / metrics["line_width"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        metrics["width_ratio"] = 0.0

    metrics["has_large_gap"] = metrics["max_gap"] >= 25.0
    metrics["is_multi_cell"] = metrics["cell_count"] >= 2
    metrics["is_strong_multi_cell"] = metrics["cell_count"] >= 3
    metrics["is_full_span_line"] = (
        (metrics["cell_count"] == 1)
        & (metrics["max_colspan"] >= metrics["total_cols"])
    )
    metrics["is_prose_candidate"] = metrics["cell_count"] >= 3
    metrics["is_justified_prose_like"] = (
        metrics["is_prose_candidate"]
        & (metrics["digit_ratio"] < 0.15)
        & (metrics["median_cell_words"] <= 2.0)
        & (metrics["width_ratio"] >= 0.25)
        & (metrics["max_gap"] <= 30.0)
        & (metrics["gap_cv"] <= 0.75)
    )

    if "table_row_score" in df.columns:
        metrics["table_row_score"] = line_gb["table_row_score"].first()
    else:
        metrics["table_row_score"] = 0.0

    token_df = df[["line_id", "col_start", "col_end"]].copy()
    token_df["_pattern_token"] = (
        token_df["col_start"].astype("int32").astype(str)
        + ":"
        + token_df["col_end"].astype("int32").astype(str)
    )
    metrics["row_pattern"] = token_df.groupby("line_id", sort=False)["_pattern_token"].agg("|".join)

    return metrics


def _compute_band_grid_metrics(
    df_cells: pd.DataFrame,
    line_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate precomputed line metrics into one KPI row per horizontal band."""
    if df_cells.empty or line_metrics.empty:
        return pd.DataFrame()

    band_gb = line_metrics.groupby("horizontal_band_id", sort=False)
    band = band_gb.agg(
        total_lines=("cell_count", "size"),
        max_cell_count=("cell_count", "max"),
        total_cols=("total_cols", "max"),
        multi_cell_lines=("is_multi_cell", "sum"),
        strong_multi_cell_lines=("is_strong_multi_cell", "sum"),
        large_gap_lines=("has_large_gap", "sum"),
        prose_candidate_lines=("is_prose_candidate", "sum"),
        justified_prose_lines=("is_justified_prose_like", "sum"),
        full_span_lines=("is_full_span_line", "sum"),
        mean_table_row_score=("table_row_score", "mean"),
    )

    cell_gb = df_cells.groupby("horizontal_band_id", sort=False)
    cell_counts = cell_gb.size().rename("total_cells")
    atomic_counts = (
        df_cells["colspan"].astype(int).eq(1)
        .groupby(df_cells["horizontal_band_id"], sort=False)
        .sum()
        .rename("atomic_cells")
    )
    band = band.join([cell_counts, atomic_counts]).fillna({
        "total_cells": 0,
        "atomic_cells": 0,
    })

    atomic = df_cells[df_cells["colspan"].astype(int).eq(1)]
    if atomic.empty:
        reused_cols = pd.Series(0, index=band.index, name="reused_cols")
    else:
        col_line_counts = (
            atomic.groupby(["horizontal_band_id", "col_start"], sort=False)["line_id"]
            .nunique()
        )
        reused_cols = (
            col_line_counts.ge(2)
            .groupby(level=0, sort=False)
            .sum()
            .rename("reused_cols")
        )
    band = band.join(reused_cols).fillna({"reused_cols": 0})

    pattern_counts = (
        line_metrics.groupby(["horizontal_band_id", "row_pattern"], sort=False)
        .size()
    )
    row_pattern_max = (
        pattern_counts.groupby(level=0, sort=False)
        .max()
        .rename("row_pattern_max")
    )
    band = band.join(row_pattern_max).fillna({"row_pattern_max": 0})

    band["multi_cell_line_ratio"] = (
        band["multi_cell_lines"] / band["total_lines"].replace(0, np.nan)
    ).fillna(0.0)
    band["strong_multi_cell_line_ratio"] = (
        band["strong_multi_cell_lines"] / band["total_lines"].replace(0, np.nan)
    ).fillna(0.0)
    band["atomic_cell_ratio"] = (
        band["atomic_cells"] / band["total_cells"].replace(0, np.nan)
    ).fillna(0.0)
    band["column_reuse"] = (
        band["reused_cols"] / band["total_cols"].replace(0, np.nan)
    ).fillna(0.0)
    band["full_span_line_ratio"] = (
        band["full_span_lines"] / band["total_lines"].replace(0, np.nan)
    ).fillna(0.0)
    band["row_pattern_reuse"] = (
        band["row_pattern_max"] / band["total_lines"].replace(0, np.nan)
    ).fillna(0.0)
    band["large_gap_ratio"] = (
        band["large_gap_lines"] / band["multi_cell_lines"].replace(0, np.nan)
    ).fillna(0.0)
    band["justified_prose_ratio"] = (
        band["justified_prose_lines"]
        / band["prose_candidate_lines"].replace(0, np.nan)
    ).fillna(0.0)

    return band


def _classify_bands(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify every horizontal_band_id as "table" or "text" using the inferred grid.

    The key distinction is whether the grid is reused across lines. Justified
    prose can create many cells, but it usually creates weak columns, one-off row
    patterns, short word fragments, and many full-row paragraph-like splits.
    """
    result_cells = _assign_column_layout(df_cells)

    result_cells["layout_id"] = result_cells["horizontal_band_id"]
    result_cells["layout_type"] = "text"
    if "block_type" not in result_cells.columns:
        result_cells["block_type"] = pd.NA

    result_lines = df_lines.copy()
    result_lines["layout_id"] = result_lines["horizontal_band_id"]
    result_lines["layout_type"] = "text"

    if result_cells.empty:
        return result_lines, result_cells

    line_metrics = _compute_line_grid_metrics(result_cells, result_lines)
    band_metrics = _compute_band_grid_metrics(result_cells, line_metrics)

    if band_metrics.empty:
        return result_lines, result_cells

    eligible = (
        (band_metrics["max_cell_count"] > 1)
        & (band_metrics["multi_cell_lines"] >= 2)
        & (band_metrics["total_cols"] > 1)
    )

    score = pd.Series(0.0, index=band_metrics.index)
    score += np.where(band_metrics["total_cols"] >= 3, 2.0, 0.0)
    score += np.where(band_metrics["column_reuse"] >= 0.45, 2.0, 0.0)
    score += np.where(band_metrics["atomic_cell_ratio"] >= 0.55, 1.5, 0.0)
    score += np.where(band_metrics["multi_cell_line_ratio"] >= 0.25, 1.5, 0.0)
    score += np.where(band_metrics["strong_multi_cell_line_ratio"] >= 0.15, 1.0, 0.0)
    score += np.where(band_metrics["large_gap_ratio"] >= 0.40, 1.0, 0.0)
    score += np.where(band_metrics["row_pattern_reuse"] >= 0.35, 1.0, 0.0)
    score += np.where(band_metrics["mean_table_row_score"] >= 1.5, 1.0, 0.0)

    score -= np.where(
        (band_metrics["full_span_line_ratio"] >= 0.60)
        & (band_metrics["atomic_cell_ratio"] < 0.45),
        2.0,
        0.0,
    )

    strong_prose_penalty = (
        (band_metrics["justified_prose_ratio"] >= 0.60)
        & (band_metrics["large_gap_ratio"] < 0.25)
        & (band_metrics["mean_table_row_score"] < 1.5)
    )
    weak_prose_penalty = (
        (band_metrics["justified_prose_ratio"] >= 0.60)
        & ~strong_prose_penalty
    )
    score -= np.where(strong_prose_penalty, 8.0, 0.0)
    score -= np.where(weak_prose_penalty, 3.0, 0.0)
    score = score.where(eligible, 0.0)

    band_types = pd.Series(
        np.where(eligible & (score >= 3.0), "table", "text"),
        index=band_metrics.index,
    )
    band_scores = score

    result_cells["layout_type"] = result_cells["horizontal_band_id"].map(band_types).fillna("text")
    result_cells["band_table_score"] = result_cells["horizontal_band_id"].map(band_scores).fillna(0.0)
    result_cells.loc[result_cells["layout_type"] == "table", "block_type"] = "table"

    result_lines["layout_type"] = result_lines["horizontal_band_id"].map(band_types).fillna("text")
    result_lines["band_table_score"] = result_lines["horizontal_band_id"].map(band_scores).fillna(0.0)

    return result_lines, result_cells






# ============================================================
# STEP 3: Eject single-cell header rows from table bands
# ============================================================

def _eject_single_cell_table_headers(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each band classified as 'table', if the first line_id in that band
    contains exactly 1 cell at col_start=0 (a left-aligned full-width header),
    reclassify that line as layout_type='text' with its own new layout_id.

    The remaining lines in the band keep their layout_id and layout_type='table'.
    """
    result_cells = df_cells.copy()
    result_lines = df_lines.copy()

    new_lid = int(result_cells["layout_id"].max()) + 1

    table_cells = result_cells[result_cells["layout_type"] == "table"]
    for _, band_df in table_cells.groupby(["page_number", "horizontal_band_id"]):
        first_line_id = int(band_df["line_id"].min())
        first_line = band_df[band_df["line_id"] == first_line_id]

        if len(first_line) == 1 and int(first_line.iloc[0]["col_start"]) == 0:
            cell_mask = result_cells["line_id"] == first_line_id
            result_cells.loc[cell_mask, "layout_type"] = "text"
            result_cells.loc[cell_mask, "layout_id"]   = new_lid

            line_mask = result_lines["line_id"] == first_line_id
            result_lines.loc[line_mask, "layout_type"] = "text"
            result_lines.loc[line_mask, "layout_id"]   = new_lid

            new_lid += 1

    return result_lines, result_cells


# ============================================================
# STEP 4: Merge adjacent table bands → shared layout_id
# ============================================================

def _merge_adjacent_tables(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    On each page, merge consecutive table layout_ids that have no intervening
    text layout between them (regardless of column count).  The smallest
    layout_id in each merge group becomes the canonical id.

    After this step, call _reassign_merged_table_grids to re-infer the column
    grid for each merged layout using all its cells together.
    """
    result_cells = df_cells.copy()
    result_lines = df_lines.copy()

    for page_num, page_cells in result_cells.groupby("page_number"):
        # Build per-layout-id bounds and type, sorted by top edge.
        bounds = (
            page_cells.groupby("layout_id")
            .agg(
                layout_type  = ("layout_type", "first"),
                y_top_min    = ("y_top",        "min"),
                y_bottom_max = ("y_bottom",      "max"),
            )
            .sort_values("y_top_min")
            .reset_index()
        )

        table_bounds = bounds[bounds["layout_type"] == "table"].reset_index(drop=True)
        if len(table_bounds) < 2:
            continue

        # Union-find over table layout_ids on this page.
        lids = table_bounds["layout_id"].tolist()
        parent = {lid: lid for lid in lids}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x, y):
            rx, ry = _find(x), _find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        for i in range(len(table_bounds) - 1):
            lid1 = table_bounds.loc[i,     "layout_id"]
            lid2 = table_bounds.loc[i + 1, "layout_id"]
            gap_top    = table_bounds.loc[i,     "y_bottom_max"]
            gap_bottom = table_bounds.loc[i + 1, "y_top_min"]

            # Check whether any non-table layout occupies the vertical gap.
            text_between = bounds[
                (bounds["layout_type"] != "table")
                & (bounds["y_top_min"]    < gap_bottom)
                & (bounds["y_bottom_max"] > gap_top)
            ]
            if text_between.empty:
                _union(lid1, lid2)

        # Apply merges: remap every layout_id to its root.
        lid_map = {lid: _find(lid) for lid in lids}
        changed = {lid: root for lid, root in lid_map.items() if lid != root}
        if not changed:
            continue

        page_mask_c = result_cells["page_number"] == page_num
        page_mask_l = result_lines["page_number"] == page_num

        for old_lid, new_lid in changed.items():
            result_cells.loc[page_mask_c & (result_cells["layout_id"] == old_lid), "layout_id"] = new_lid
            result_lines.loc[page_mask_l & (result_lines["layout_id"] == old_lid), "layout_id"] = new_lid

    return result_lines, result_cells


def _reassign_merged_table_grids(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    For every table layout_id that now spans more than one horizontal_band_id
    (i.e. was created by merging), re-run column grid inference using the
    combined cells so col_start, col_end, colspan, band_total_cols are correct
    for the merged layout as a whole.
    """
    table_cells = df_cells[df_cells["layout_type"] == "table"]
    if table_cells.empty:
        return df_cells

    bands_per_layout = (
        table_cells.groupby("layout_id")["horizontal_band_id"].nunique()
    )
    merged_lids = set(bands_per_layout[bands_per_layout > 1].index.tolist())
    if not merged_lids:
        return df_cells

    result = df_cells.copy()
    merged_mask = result["layout_id"].isin(merged_lids) & (result["layout_type"] == "table")
    updated = _assign_column_layout(result[merged_mask].copy(), group_col="layout_id")

    grid_cols = ["col_start", "col_end", "colspan", "band_total_cols"]
    result.loc[updated.index, grid_cols] = updated[grid_cols]
    return result

# ============================================================
# LAYOUT ID REINDEX
# ============================================================

def _reindex_layout_id_and_add_table_id(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reassign layout_ids to sequential integers (1, 2, 3, …) ordered by
    (page_number, min line_id) so the final output has monotonically increasing
    layout_ids across the document regardless of how ejection/merging shifted them.

    line_id is assigned upstream in stable reading order, so do not reinterpret
    layout ordering from y_top/y_bottom coordinates here.
    """
    if df_cells.empty:
        return df_lines, df_cells

    layout_order = (
        df_cells.groupby("layout_id")
        .agg(
            page_number=("page_number", "first"),
            line_id_min=("line_id", "min"),
            layout_type=("layout_type", "first"),
        )
        .sort_values(["page_number", "line_id_min"])
        .reset_index()
    )
    old_to_new = {
        int(row["layout_id"]): i + 1
        for i, row in layout_order.iterrows()
    }

    table_counter = 0
    old_to_table_id: dict[int, int | None] = {}
    for _, row in layout_order.iterrows():
        if row["layout_type"] == "table":
            table_counter += 1
            old_to_table_id[int(row["layout_id"])] = table_counter
        else:
            old_to_table_id[int(row["layout_id"])] = None

    df_cells = df_cells.copy()
    df_cells["layout_id"] = df_cells["layout_id"].map(old_to_new)
    df_cells["table_id"] = df_cells["layout_id"].map(
        {new: old_to_table_id[old] for old, new in old_to_new.items()}
    )

    df_lines = df_lines.copy()
    if "layout_id" in df_lines.columns:
        df_lines["layout_id"] = df_lines["layout_id"].map(old_to_new)
        df_lines["table_id"] = df_lines["layout_id"].map(
            {new: old_to_table_id[old] for old, new in old_to_new.items()}
        )

    return df_lines, df_cells


def _sync_table_block_type(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep block_type aligned with the final table classification.

    The table classifier may reclassify/eject/merge layouts after the initial
    layout_type assignment, so write block_type from the final layout_type state.
    """
    result_lines = df_lines.copy()
    result_cells = df_cells.copy()

    for df in (result_lines, result_cells):
        if "block_type" not in df.columns:
            df["block_type"] = pd.NA
        if "layout_type" not in df.columns:
            continue

        layout_type = df["layout_type"].astype("string").str.strip().str.lower()
        table_mask = layout_type.eq("table").fillna(False)
        stale_table_mask = df["block_type"].astype("string").eq("table").fillna(False)
        df.loc[table_mask, "block_type"] = "table"
        df.loc[~table_mask & stale_table_mask, "block_type"] = pd.NA

    return result_lines, result_cells


# ============================================================
# STEP 5: Build table structure — rows, rowspan, cell roles
# ============================================================


# ============================================================
# STEP 5: Expand horizontal grid-line assignments for table cells
# ============================================================

def _expand_table_grid_lines(
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """
    For cells with layout_type='table', assign shape_id_horizontal_grid_line to
    the correct boundary shape using column-aware logic:

    For each horizontal shape S and each column (col_start) it x-overlaps, only
    the LAST cell in that column above S (i.e. the cell with the maximum y_bottom
    that is still ≤ S.y) receives the assignment.  If multiple shapes qualify for
    the same cell, the nearest one (minimum shape y) wins.

    This replaces the step-8 proximity assignment (≤ 10 pt) and correctly handles
    cells whose text sits far above the row's bottom border (e.g. a sparse label
    in a tall row where adjacent columns have many lines of wrapping text).

    Non-table cells are not touched.  Cells in the last table row (no shape below
    them) have their assignment cleared to NA / False.
    """
    if "layout_type" not in df_cells.columns:
        return df_cells

    result = df_cells.copy()
    table_mask = result["layout_type"] == "table"
    if not table_mask.any() or df_shapes.empty:
        return result

    shape_mask = pd.Series(True, index=df_shapes.index)
    if "shape_type" in df_shapes.columns:
        shape_mask &= df_shapes["shape_type"] == "line"
    if "shape_orientation" in df_shapes.columns:
        shape_mask &= df_shapes["shape_orientation"] == "horizontal"
    h_shapes = df_shapes[shape_mask].copy()
    if h_shapes.empty:
        return result

    has_page = "page_number" in h_shapes.columns

    for key, grp in result[table_mask].groupby(
        ["layout_id", "page_number"], sort=True
    ):
        page_num = key[1]
        page_shapes = (
            h_shapes[h_shapes["page_number"] == page_num]
            if has_page else h_shapes
        )
        if page_shapes.empty:
            continue

        tbl_xl = float(grp["x_left"].min())
        tbl_xr = float(grp["x_right"].max())
        tbl_yt = float(grp["y_top"].min())

        # Pre-filter: shapes that x-overlap the table and start at or below its top edge.
        # No tolerance — shapes outside the table's x/y range are never candidates.
        cand = page_shapes[
            (page_shapes["x_right"] > tbl_xl) &
            (page_shapes["x_left"]  < tbl_xr) &
            (page_shapes["y_top"]   >= tbl_yt)
        ]
        if cand.empty:
            continue

        cell_xl      = grp["x_left"].to_numpy(dtype=float)
        cell_xr      = grp["x_right"].to_numpy(dtype=float)
        cell_yb      = grp["y_bottom"].to_numpy(dtype=float)
        cell_cs      = grp["col_start"].to_numpy()
        cell_indices = grp.index.to_numpy()

        sh_xl  = cand["x_left"].to_numpy(dtype=float)
        sh_xr  = cand["x_right"].to_numpy(dtype=float)
        sh_y   = ((cand["y_top"] + cand["y_bottom"]) / 2).to_numpy(dtype=float)
        sh_ids = cand["shape_id"].to_numpy()

        # assignments[global_idx] = (shape_id, shape_y) — nearest shape wins
        assignments: dict[int, tuple[int, float]] = {}

        for si in range(len(sh_y)):
            sy  = sh_y[si]
            sxl = sh_xl[si]
            sxr = sh_xr[si]
            sid = int(sh_ids[si])

            # Cells that x-overlap this shape and sit at or above it
            x_ok     = (cell_xl < sxr) & (sxl < cell_xr)
            y_ok     = cell_yb <= sy
            cand_arr = np.where(x_ok & y_ok)[0]
            if cand_arr.size == 0:
                continue

            # Per col_start: keep only the cell with the maximum y_bottom
            # (the last cell in that column before this shape).
            cand_cs_arr = cell_cs[cand_arr]
            cand_yb_arr = cell_yb[cand_arr]
            for cs in np.unique(cand_cs_arr):
                in_cs      = cand_cs_arr == cs
                group_arr  = cand_arr[in_cs]
                best_local = group_arr[cand_yb_arr[in_cs].argmax()]
                global_idx = int(cell_indices[best_local])

                existing = assignments.get(global_idx)
                if existing is None or sy < existing[1]:
                    assignments[global_idx] = (sid, sy)

        for global_idx, (sid, _) in assignments.items():
            if pd.isna(result.at[global_idx, "shape_id_horizontal_grid_line"]):
                result.at[global_idx, "shape_id_horizontal_grid_line"] = sid
                result.at[global_idx, "has_horizontal_grid_line"] = True

    return result






# ============================================================
# STEP 5: Per-column slot-based table cell assembly
# ============================================================

def _compute_grid_line_last_map(df_tables: pd.DataFrame) -> dict:
    """
    For each (layout_id, page_number, shape_id_horizontal_grid_line),
    return the maximum line_id that carries that shape.
    This is the line that triggers the flush for that shape.
    """
    col = "shape_id_horizontal_grid_line"
    if col not in df_tables.columns:
        return {}
    valid = df_tables.dropna(subset=[col])
    if valid.empty:
        return {}
    last_df = (
        valid.groupby(["layout_id", "page_number", col])["line_id"]
        .max()
        .reset_index()
    )
    return {
        (int(r["layout_id"]), int(r["page_number"]), int(r[col])): int(r["line_id"])
        for _, r in last_df.iterrows()
    }


def _flush_slot(
    records: list,
    tcell_counter: int,
    slot: dict,
    cs: int,
    new_row_idx: int,
    layout_id: int,
    page_number: int,
    table_id: int,
) -> int:
    text_raw_lines: list[str] = []
    cell_ids: list[int] = []
    line_ids: list[int] = []
    seen_line_ids: set[int] = set()

    for row in sorted(slot["cells"], key=lambda r: (r.line_id, r.col_start)):
        lid = int(row.line_id)
        if lid not in seen_line_ids:
            seen_line_ids.add(lid)
            line_ids.append(lid)
        cell_ids.append(int(row.cell_id))

    # Build per-line text in line_id order
    from itertools import groupby as _groupby
    for lid, grp in _groupby(
        sorted(slot["cells"], key=lambda r: (r.line_id, r.col_start)),
        key=lambda r: r.line_id,
    ):
        parts = [str(r.text or "").strip() for r in grp if str(r.text or "").strip()]
        if parts:
            text_raw_lines.append(" ".join(parts))

    records.append({
        "table_cell_id":  tcell_counter,
        "page_number":    page_number,
        "layout_id":      layout_id,
        "table_id":       table_id,
        "row_start":      slot["row_start"],
        "col_start":      cs,
        "rowspan":        new_row_idx - slot["row_start"],
        "colspan":        slot["colspan"],
        "text":           " ".join(text_raw_lines),
        "text_raw_lines": text_raw_lines,
        "cell_ids":       cell_ids,
        "line_ids":       line_ids,
    })
    return tcell_counter + 1


def _build_table_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble one record per logical table cell using per-column slot flushing.

    Each column (col_start) maintains an open slot that accumulates PDF lines.
    A slot flushes when:
      1. The current line is the last instance of that column's
         shape_id_horizontal_grid_line (partial or complete flush depending on
         which columns share the shape).
      2. All columns are covered by a single line_id and no grid-line trigger
         fired (complete-flush fallback for borderless tables).

    global_row_idx increments once per flush event regardless of how many
    columns participate, so surviving slots accumulate the correct rowspan.
    """
    _EMPTY_COLS = [
        "table_cell_id", "page_number", "layout_id", "table_id",
        "row_start", "col_start", "rowspan", "colspan",
        "text", "text_raw_lines", "cell_ids", "line_ids",
    ]

    df_tables = (
        df_cells[df_cells["layout_type"] == "table"].copy()
        if "layout_type" in df_cells.columns
        else df_cells.copy()
    )
    if df_tables.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    if "shape_id_horizontal_grid_line" not in df_tables.columns:
        df_tables["shape_id_horizontal_grid_line"] = pd.NA

    grid_line_last_map = _compute_grid_line_last_map(df_tables)

    records: list[dict] = []
    tcell_counter = 1

    for (layout_id, page_number), df_seg in df_tables.groupby(
        ["layout_id", "page_number"], sort=True
    ):
        if df_seg.empty:
            continue

        df_seg       = df_seg.sort_values(["line_id", "col_start", "cell_id"])
        band_total_cols = int(df_seg["band_total_cols"].max())
        all_cols     = set(range(band_total_cols))
        table_id     = int(df_seg["table_id"].iloc[0]) if "table_id" in df_seg.columns else 0

        # Precompute: for each shape_id in this table, the set of col_starts
        # that carry it.  When a shape fires, all those columns flush together.
        shape_to_col_starts: dict[int, set[int]] = {}
        for row in df_seg.itertuples(index=False):
            gid_raw = getattr(row, "shape_id_horizontal_grid_line", None)
            if gid_raw is not None and not pd.isna(gid_raw):
                shape_to_col_starts.setdefault(int(gid_raw), set()).add(int(row.col_start))

        slot_open: dict[int, dict] = {}
        global_row_idx = 0

        for line_id in sorted(df_seg["line_id"].dropna().unique().tolist()):
            df_line = df_seg[df_seg["line_id"] == line_id].sort_values("col_start")

            flush_col_starts: set[int] = set()
            covered: set[int] = set()

            for row in df_line.itertuples(index=False):
                cs = int(row.col_start)
                ce = int(row.col_end)

                # Accumulate into slot
                if cs not in slot_open:
                    slot_open[cs] = {
                        "col_end":  ce,
                        "colspan":  ce - cs + 1,
                        "row_start": global_row_idx,
                        "cells":    [],
                    }
                slot_open[cs]["cells"].append(row)

                # Track column coverage for complete-flush fallback
                for c in range(cs, ce + 1):
                    covered.add(c)

                # Grid-line flush trigger: flush ALL col_starts that share this shape
                gid_raw = getattr(row, "shape_id_horizontal_grid_line", None)
                if gid_raw is not None and not pd.isna(gid_raw):
                    gid = int(gid_raw)
                    key = (int(layout_id), int(page_number), gid)
                    if grid_line_last_map.get(key) == int(line_id):
                        flush_col_starts |= shape_to_col_starts.get(gid, set())

            # Complete-flush fallback: all columns covered, no grid-line fired
            if not flush_col_starts and covered >= all_cols:
                flush_col_starts = set(slot_open.keys())

            if flush_col_starts:
                new_row_idx = global_row_idx + 1
                for cs in sorted(flush_col_starts):
                    slot = slot_open.pop(cs, None)
                    if slot is None:
                        continue
                    tcell_counter = _flush_slot(
                        records, tcell_counter, slot, cs, new_row_idx,
                        int(layout_id), int(page_number), table_id,
                    )
                global_row_idx = new_row_idx

        # End-of-table: flush all remaining open slots
        if slot_open:
            new_row_idx = global_row_idx + 1
            for cs in sorted(slot_open.keys()):
                tcell_counter = _flush_slot(
                    records, tcell_counter, slot_open[cs], cs, new_row_idx,
                    int(layout_id), int(page_number), table_id,
                )

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.DataFrame.from_records(records)


# ============================================================
# PUBLIC API
# ============================================================

def build_tables(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify bands into text or table, and for tables derive the full layout.

    Parameters
    ----------
    df_lines : pd.DataFrame
        One row per line_id.  Output of step_09_line_builder.build_lines().
        Must contain: line_id, cell_count, char_count, digit_count, word_count,
        capitalized_word_count, has_vertical_line, is_underlined, page_width, width,
        horizontal_band_id.

    df_cells : pd.DataFrame
        One row per cell.  Output of step_09_line_builder.build_lines()
        (with horizontal_band_id added).
        Must contain: cell_id, line_id, page_number, horizontal_band_id,
        x_left, x_right, y_top, y_bottom, text.

    df_shapes : pd.DataFrame | None
        Shape records from the PDF extractor (step_06 output).  Used to expand
        horizontal grid-line assignments for table cells beyond the 10-pt cap
        applied in the cell builder.  Pass None to skip this step.

    Returns
    -------
    df_lines : pd.DataFrame
        With added columns: layout_id, layout_type, band_table_score.

    df_cells : pd.DataFrame
        With added columns: col_start, col_end, colspan, band_total_cols,
        layout_id, layout_type, block_type, band_table_score.

    df_table_cells : pd.DataFrame
        One row per assembled table cell.  Contains: table_cell_id, page_number,
        layout_id, table_id, row_start, col_start, rowspan, colspan, text,
        text_raw_lines, cell_ids, line_ids, role.
    """
    if df_cells.empty:
        empty_tcells = pd.DataFrame(columns=[
            "table_cell_id", "page_number", "layout_id", "table_id",
            "row_start", "col_start", "rowspan", "colspan",
            "text", "text_raw_lines", "cell_ids", "line_ids", "role",
        ])
        return df_lines.copy(), df_cells.copy(), empty_tcells

    # Step 1/2 — infer grid, then classify each horizontal band.
    df_lines, df_cells = _classify_bands(df_lines, df_cells)

    # Step 3 — eject single-cell left-aligned headers from table bands.
    df_lines, df_cells = _eject_single_cell_table_headers(df_lines, df_cells)

    # Step 4 — merge adjacent table bands with no intervening text.
    df_lines, df_cells = _merge_adjacent_tables(df_lines, df_cells)

    # Step 4b — re-run column grid inference for merged layouts.
    df_cells = _reassign_merged_table_grids(df_cells)

    

    # Reindex layout_ids to be sequential ordered by stable upstream line_id.
    df_lines, df_cells = _reindex_layout_id_and_add_table_id(df_lines, df_cells)

    ## REMOVE THIS SYNC IN THE FINAL SCRIPT

    # Keep block_type in sync with final layout_type on both lines and cells.
    df_lines, df_cells = _sync_table_block_type(df_lines, df_cells)

    # Step 4c — expand horizontal grid-line assignments for table cells. THIS IS BELOW THE REINDEXING !!! - START OF A NEW METHOD WHERE WE CONVERT LINE INTO TABLE
    # STOP ADDING IT ON TOP OF REINDEXING !!!
    if df_shapes is not None and not df_shapes.empty:
        df_cells = _expand_table_grid_lines(df_cells, df_shapes)

    # Step 5 — assemble table rows and assign cell roles.
    df_table_cells = _build_table_cells(df_cells)
    if not df_table_cells.empty:
        df_table_cells = detect_cell_roles(df_table_cells, with_row_label=True)

    return df_lines, df_cells, df_table_cells

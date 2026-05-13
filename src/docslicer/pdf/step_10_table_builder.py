"""
step_10_table_builder.py

Classify bands into text or table, and for tables derive the full layout.

Public API:
    df_lines, df_cells, df_table_cells = build_tables(df_lines, df_cells, df_words=None)

Pipeline:
    Step 1: Infer column grid
        - Within each (page_number, horizontal_band_id), infers a column grid
        - Adds col_start, col_end, colspan, band_total_cols to df_cells

    Step 2: Classify horizontal bands
        - Single-cell bands are text
        - Multi-cell bands are scored from inferred grid quality

Notes:
    - temp_line_id (old pipeline) is replaced by line_id throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

_MAX_VERTICAL_GAP = 8.0   # max gap (pt) between bands for layout merging


# ============================================================
# STEP 1: Infer column grid
# ============================================================

def _assign_column_layout(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Infer a column grid within each (page_number, horizontal_band_id) band
    and assign col_start, col_end, colspan, band_total_cols to every cell.

    Algorithm (per band):
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
    result["col_start"]       = -1
    result["col_end"]         = -1
    result["colspan"]         = -1
    result["band_total_cols"] = -1

    for _, band_df in result.groupby(["page_number", "horizontal_band_id"]):
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
    if "block_role" not in result_cells.columns:
        result_cells["block_role"] = pd.NA

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
    result_cells.loc[result_cells["layout_type"] == "table", "block_role"] = "table"

    result_lines["layout_type"] = result_lines["horizontal_band_id"].map(band_types).fillna("text")
    result_lines["band_table_score"] = result_lines["horizontal_band_id"].map(band_scores).fillna(0.0)

    return result_lines, result_cells






# ============================================================
# PUBLIC API
# ============================================================

def build_tables(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame | None = None,
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

    df_words : pd.DataFrame | None
        Reserved for later table-cell assembly/scoring stages.

    Returns
    -------
    df_lines : pd.DataFrame
        With added columns: layout_id, layout_type, band_table_score.

    df_cells : pd.DataFrame
        With added columns: col_start, col_end, colspan, band_total_cols,
        layout_id, layout_type, block_role, band_table_score.

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

    # Table-cell assembly will come after the revised band classifier.
    empty_tcells = pd.DataFrame(columns=[
        "table_cell_id", "page_number", "layout_id", "table_id",
        "row_start", "col_start", "rowspan", "colspan",
        "text", "text_raw_lines", "cell_ids", "line_ids", "role",
    ])

    return df_lines, df_cells#, empty_tcells

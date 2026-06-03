"""
step_10_table_builder.py

Classify bands into text or table, and for tables derive the full layout.

Public API:
    df_lines, df_cells, df_table_cells = build_tables(df_lines, df_cells, df_words=None)

Pipeline:
    Step 1: Score lines
        - Computes ratio features (digit_ratio, width_ratio, gap stats, …)
        - Adds table_row_score per line, propagated to df_cells

    Step 2: Assign column layout
        - Within each (page_number, horizontal_band_id), infers a column grid
        - Adds col_start, col_end, colspan, band_total_cols to df_cells

    Step 3: Merge bands → layout_id
        - Consecutive bands with matching column structure and small vertical gap
          are merged into a single layout_id

    Step 4: Classify layouts
        - Computes average_table_score per layout_id
        - Classifies each layout_id as "text" or "table"
        - band_total_cols == 1 is always text; multi-col uses heuristic scoring
          (justified text can produce multiple cells per line without being a table)

    Step 5: Build table structure
        - Assembles table cells with row_start, rowspan, colspan
        - Assigns cell roles: header, row_label, value_numeric, value_text, footnote
        - Classifies table sub-type: standard, matrix, narrative

Notes:
    - temp_line_id (old pipeline) is replaced by line_id throughout.
    - df_words is optional; if absent, word-gap scoring features default to 0.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

_MAX_VERTICAL_GAP = 8.0   # max gap (pt) between bands for layout merging


# ============================================================
# STEP 1: Score lines
# ============================================================

def _compute_line_ratios(df_lines: pd.DataFrame, df_words: pd.DataFrame | None) -> pd.DataFrame:
    """
    Add ratio features used by the table-row scorer.

    Cell-level ratios (derived from df_lines):
        width_ratio             line_width / page_width
        digit_ratio             digit_count / char_count
        capitalized_word_ratio  capitalized_word_count / word_count
        underlined_ratio        1.0 if is_underlined else 0.0

    Word-level gap stats (derived from df_words with line_id; fully vectorised):
        median_x0x1_gap     median positive gap between consecutive words
        max_x0x1_gap        maximum positive gap between consecutive words
        gap_ratio           max_gap / (median_gap + 1e-6)
    """
    df = df_lines.copy()

    df["width_ratio"] = (
        df["width"] / df["page_width"].replace(0, np.nan)
    ).fillna(0.0)

    df["digit_ratio"] = (
        df["digit_count"] / df["char_count"].replace(0, np.nan)
    ).fillna(0.0)

    df["capitalized_word_ratio"] = (
        df["capitalized_word_count"] / df["word_count"].replace(0, np.nan)
    ).fillna(0.0)

    df["underlined_ratio"] = df["is_underlined"].fillna(False).astype(float)

    # --- Word-gap stats (vectorised) ---
    if (
        df_words is None
        or df_words.empty
        or "line_id" not in df_words.columns
    ):
        df["median_x0x1_gap"] = 0.0
        df["max_x0x1_gap"]    = 0.0
        df["gap_ratio"]       = 0.0
        return df

    ws = (
        df_words[["line_id", "x_left", "x_right"]]
        .sort_values(["line_id", "x_left"])
        .copy()
    )

    ws["_next_x_left"] = ws.groupby("line_id", sort=False)["x_left"].shift(-1)
    ws["_gap"]         = ws["_next_x_left"] - ws["x_right"]

    positive = ws[ws["_gap"] > 0]

    if positive.empty:
        df["median_x0x1_gap"] = 0.0
        df["max_x0x1_gap"]    = 0.0
        df["gap_ratio"]       = 0.0
        return df

    gap_stats = (
        positive.groupby("line_id")["_gap"]
        .agg(median_x0x1_gap="median", max_x0x1_gap="max")
        .reset_index()
    )
    gap_stats["gap_ratio"] = (
        gap_stats["max_x0x1_gap"] / (gap_stats["median_x0x1_gap"] + 1e-6)
    )

    df = df.merge(gap_stats, on="line_id", how="left")
    for col in ("median_x0x1_gap", "max_x0x1_gap", "gap_ratio"):
        df[col] = df[col].fillna(0.0)

    return df


def _compute_table_row_scores(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised table-row scoring.  Adds column: table_row_score (float).

    Inputs expected on df_lines:
        cell_count, digit_ratio, underlined_ratio, width_ratio,
        has_vertical_line, median_x0x1_gap, gap_ratio, capitalized_word_ratio
    """
    if df_lines.empty:
        return df_lines.assign(table_row_score=pd.Series(dtype="float64"))

    df = df_lines.copy()

    cc  = df.get("cell_count",             pd.Series(0,     index=df.index)).fillna(0).to_numpy(dtype=float)
    dr  = df.get("digit_ratio",            pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    ur  = df.get("underlined_ratio",       pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    wr  = df.get("width_ratio",            pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    hvl = df.get("has_vertical_line",      pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=float)
    mg  = df.get("median_x0x1_gap",        pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    gr  = df.get("gap_ratio",              pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()
    ctr = df.get("capitalized_word_ratio", pd.Series(0.0,   index=df.index)).fillna(0.0).to_numpy()

    score = np.zeros(len(df), dtype=float)

    score += np.where(cc < 3, -1.5,
             np.where(cc == 3, 0.3,
             np.where(cc == 4, 0.8, 1.2)))

    score += np.clip((dr - 0.2) * 3.0, -0.5, 1.5)
    score += np.minimum(ur * 1.5, 1.0)
    score += np.where(wr < 0.35, -0.5, np.where(wr >= 0.55, 0.3, 0.0))
    score += hvl * 6.0

    score += np.where(mg < 5.0, -1.5,
             np.where(mg < 10.0, np.interp(mg, [5.0, 10.0], [0.0, 1.0]),
             np.where(mg < 15.0, np.interp(mg, [10.0, 15.0], [1.0, 1.5]),
             1.7)))

    score += np.where(gr < 2.0, -1.5,
             np.where(gr < 5.0,  np.interp(gr, [2.0, 5.0],  [0.0, 1.0]),
             np.where(gr < 15.0, np.interp(gr, [5.0, 15.0], [1.0, 2.0]),
             2.0)))

    ctr_c = np.minimum(ctr, 1.0)
    score += np.where(ctr < 0.2, -0.5,
             np.where(ctr < 0.7, np.interp(ctr, [0.2, 0.7], [0.0, 1.0]),
             np.interp(ctr_c, [0.7, 1.0], [1.0, 1.5])))

    df["table_row_score"] = score
    return df


def _score_lines(
    df_lines: pd.DataFrame,
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute table_row_score on df_lines and propagate it to df_cells.

    Returns updated (df_lines, df_cells).
    """
    df_lines = _compute_line_ratios(df_lines, df_words)
    df_lines = _compute_table_row_scores(df_lines)

    score_map = df_lines.set_index("line_id")["table_row_score"].to_dict()
    df_cells = df_cells.copy()
    df_cells["table_row_score"] = df_cells["line_id"].map(score_map).fillna(0.0)

    return df_lines, df_cells


# ============================================================
# STEP 2: Assign column layout
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
# STEP 3: Merge bands → layout_id
# ============================================================

def _find_mergeable_bands(
    df_cells: pd.DataFrame,
    max_vertical_gap: float = _MAX_VERTICAL_GAP,
) -> dict[int, int]:
    """
    Return a mapping {horizontal_band_id: layout_id}.

    Consecutive bands on the same page are merged into the same layout_id when:
      - both have band_total_cols > 1
      - they have equal band_total_cols
      - their column boundary positions align exactly
      - their vertical gap is ≤ max_vertical_gap
    Single-column bands always get their own layout_id.
    """
    def _col_positions(band_cells):
        pos: set[int] = set()
        for _, cell in band_cells.iterrows():
            pos.add(int(cell["col_start"]))
            pos.add(int(cell["col_end"]) + 1)
        return sorted(pos)

    band_to_layout: dict[int, int] = {}
    layout_counter = 0

    for _, page_df in df_cells.groupby("page_number"):
        bands = sorted(page_df["horizontal_band_id"].unique())
        current_merge_group = None

        for i, band_id in enumerate(bands):
            band_cells   = page_df[page_df["horizontal_band_id"] == band_id]
            total_cols   = int(band_cells["band_total_cols"].iloc[0])

            if total_cols == 1:
                layout_counter += 1
                band_to_layout[band_id] = layout_counter
                current_merge_group = None
                continue

            can_merge = False
            if i > 0 and current_merge_group is not None:
                prev_id    = bands[i - 1]
                prev_cells = page_df[page_df["horizontal_band_id"] == prev_id]
                prev_total = int(prev_cells["band_total_cols"].iloc[0])

                if (
                    prev_total > 1
                    and band_id == prev_id + 1
                    and total_cols == prev_total
                    and _col_positions(band_cells) == _col_positions(prev_cells)
                    and (band_cells["y_top"].min() - prev_cells["y_bottom"].max()) <= max_vertical_gap
                ):
                    can_merge = True

            if can_merge:
                band_to_layout[band_id] = current_merge_group
            else:
                layout_counter += 1
                band_to_layout[band_id] = layout_counter
                current_merge_group = layout_counter

    return band_to_layout


# ============================================================
# STEP 4: Classify layouts
# ============================================================

def _add_average_table_score(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average table_row_score per layout_id (one score per line_id)
    and add it as average_table_score on df_cells.
    """
    line_scores = (
        df_cells.groupby(["layout_id", "line_id"])["table_row_score"]
        .first()
        .reset_index()
    )
    layout_avg = (
        line_scores.groupby("layout_id")["table_row_score"]
        .mean()
        .reset_index()
        .rename(columns={"table_row_score": "average_table_score"})
    )
    return df_cells.merge(layout_avg, on="layout_id", how="left")


def _classify_layout_types(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each layout_id as "text" or "table".

    band_total_cols == 1  → always "text"
    band_total_cols  > 1  → scored heuristically:
        score  0 → "text"   (multi-col layout that looks like prose)
        score  1 → "text"   (borderline; justified text can produce multi-cell lines)
        score ≥ 2 → "table"

    Scoring (additive):
        +1  if 2 ≤ band_total_cols ≤ 6
        +2  if band_total_cols ≥ 7
        +1  if distinct underline IDs ≥ 2
        +0/+1/+2 based on average_table_score thresholds (< 1 / 1–2 / ≥ 2)
    """
    result = df_cells.copy()
    layout_types: dict[int, str] = {}

    for layout_id, layout_df in result.groupby("layout_id"):
        total_cols = int(layout_df["band_total_cols"].iloc[0])

        if total_cols == 1:
            layout_types[layout_id] = "text"
            continue

        score = 0

        if 2 <= total_cols <= 6:
            score += 1
        elif total_cols >= 7:
            score += 2

        if "shape_id_underline" in layout_df.columns:
            if layout_df["shape_id_underline"].nunique() >= 2:
                score += 1

        if "average_table_score" in layout_df.columns:
            avg = float(layout_df["average_table_score"].iloc[0])
            if 1 <= avg < 2:
                score += 1
            elif avg >= 2:
                score += 2

        layout_types[layout_id] = "table" if score >= 2 else "text"

    result["layout_type"] = result["layout_id"].map(layout_types)

    if "block_type" not in result.columns:
        result["block_type"] = pd.NA
    result.loc[result["layout_type"] == "table", "block_type"] = "table"

    return result


# ============================================================
# STEP 5: Build table structure
# ============================================================


# --- helpers (underline-aware row building) ---

def _compute_underline_last(df: pd.DataFrame) -> dict[tuple, int]:
    df_u = df.dropna(subset=["shape_id_underline"]).copy()
    if df_u.empty:
        return {}
    df_u["underline_id"] = df_u["shape_id_underline"].astype(int)
    last_df = (
        df_u.groupby(["layout_id", "page_number", "underline_id"])["line_id"]
        .max()
        .reset_index()
        .rename(columns={"line_id": "last_line_id"})
    )
    return {
        (int(r.layout_id), int(r.page_number), int(r.underline_id)): int(r.last_line_id)
        for r in last_df.itertuples(index=False)
    }


def _compute_covered_cols(group_df: pd.DataFrame) -> set[int]:
    covered: set[int] = set()
    for row in group_df.itertuples(index=False):
        for c in range(int(row.col_start), int(row.col_end) + 1):
            covered.add(c)
    return covered


def _get_completion_threshold(band_total_cols: int, row_index: int) -> int:
    """
    Return how many columns must be covered for this row to count as 'complete',
    given the total number of columns in the band and the row index (1-based).

    Scheme (from your table):

        band_total_cols   min_normal   min_top   num_top_rows
        ----------------------------------------------------
        2                 2            1         1
        3                 3            2         1
        4                 4            3         2
        5                 4            3         2
        6                 4            3         3
        7                 4            3         3
        8                 4            3         3

    For >8 cols we just treat it like 8 (plateau).
    """
    if band_total_cols <= 0:
        return 0
    if band_total_cols == 1:
        return 1
    elif band_total_cols == 2:
        min_normal, min_top, n_top = 2, 1, 1
    elif band_total_cols == 3:
        min_normal, min_top, n_top = 3, 2, 1
    elif band_total_cols == 4:
        min_normal, min_top, n_top = 4, 3, 2
    elif band_total_cols == 5:
        min_normal, min_top, n_top = 4, 3, 2
    elif 6 <= band_total_cols <= 8:
        min_normal, min_top, n_top = 4, 3, 3
    else:
        min_normal, min_top, n_top = 4, 3, 3
    return min_top if row_index <= n_top else min_normal


def _is_complete_row(covered_cols: set[int], band_total_cols: int, row_index: int) -> bool:
    if band_total_cols <= 0:
        return False
    return len(covered_cols) >= _get_completion_threshold(band_total_cols, row_index)


def _has_last_underline(
    line_underline_ids: list[int],
    layout_id: int,
    page_number: int,
    line_id: int,
    underline_last_map: dict,
) -> bool:
    for u in line_underline_ids:
        last = underline_last_map.get((layout_id, page_number, u))
        if last is not None and int(line_id) == int(last):
            return True
    return False


def _flush_group_to_row(
    group_df: pd.DataFrame,
    layout_id: int,
    page_number: int,
    table_id: int,
    row_index: int,
    band_total_cols: int,
    covered_cols: set[int],
    records: list,
    table_cell_id_counter: int,
    cell_meta: dict,
    cell_record_idx: dict,
    row_to_cell_ids: dict,
    open_rowspan: dict,
    underline_row_anchor: dict,
    flush_reason: str,
) -> tuple[int, dict, dict]:
    # Extend rowspans for missing columns.
    for col in range(1, band_total_cols + 1):
        key_col = (layout_id, page_number, col)
        if col not in covered_cols:
            prev = open_rowspan.get(key_col)
            if prev is not None:
                records[cell_record_idx[prev]]["rowspan"] += 1

    row_key = (layout_id, page_number, row_index)
    row_to_cell_ids.setdefault(row_key, [])

    sorted_group = group_df.sort_values(["col_start", "col_end", "line_id", "cell_id"])
    for (col_start, col_end), sub in sorted_group.groupby(["col_start", "col_end"], sort=True):
        col_start = int(col_start)
        col_end   = int(col_end)
        colspan   = col_end - col_start + 1

        texts = [str(t or "").strip() for t in sub["text"] if str(t or "").strip()]
        merged_text = " ".join(texts)

        text_raw_lines: list[str] = []
        for _, sub_line in sub.groupby("line_id", sort=True):
            parts = [str(t or "").strip() for t in sub_line["text"] if str(t or "").strip()]
            if parts:
                text_raw_lines.append(" ".join(parts))

        tcell_id = table_cell_id_counter
        table_cell_id_counter += 1

        record = {
            "table_cell_id": tcell_id,
            "page_number":   page_number,
            "layout_id":     layout_id,
            "table_id":      table_id,
            "row_start":     row_index,
            "col_start":     col_start,
            "rowspan":       1,
            "colspan":       colspan,
            "text":          merged_text,
            "text_raw_lines": text_raw_lines,
            "cell_ids":      sub["cell_id"].tolist(),
            "line_ids":      sub["line_id"].dropna().drop_duplicates().astype(int).tolist(),
            "flush_reason":  flush_reason,
        }
        records.append(record)
        rec_idx = len(records) - 1
        cell_record_idx[tcell_id] = rec_idx
        cell_meta[tcell_id] = {
            "layout_id": layout_id,
            "page_number": page_number,
            "row_index": row_index,
            "col_start": col_start,
            "colspan": colspan,
        }
        row_to_cell_ids[row_key].append(tcell_id)
        for col in range(col_start, col_end + 1):
            open_rowspan[(layout_id, page_number, col)] = tcell_id

    # Anchor underline IDs on this row.
    for u in (
        group_df["shape_id_underline"].dropna().astype(int).unique().tolist()
        if "shape_id_underline" in group_df.columns else []
    ):
        underline_row_anchor.setdefault((layout_id, page_number, u), row_index)

    return table_cell_id_counter, open_rowspan, underline_row_anchor


def _try_attach_to_existing_row(
    df_line: pd.DataFrame,
    layout_id: int,
    page_number: int,
    records: list,
    cell_meta: dict,
    cell_record_idx: dict,
    row_to_cell_ids: dict,
    underline_row_anchor: dict,
) -> tuple[bool, set[int]]:
    if "shape_id_underline" not in df_line.columns:
        return False, set()
    underline_ids = (
        df_line["shape_id_underline"].dropna().astype(int).unique().tolist()
    )
    if not underline_ids:
        return False, set()

    attached_any = False
    anchor_rows: set[int] = set()

    for u in underline_ids:
        key_u = (layout_id, page_number, u)
        if key_u not in underline_row_anchor:
            continue
        row_index_anchor = underline_row_anchor[key_u]
        anchor_rows.add(row_index_anchor)
        row_key = (layout_id, page_number, row_index_anchor)
        dest_ids = row_to_cell_ids.get(row_key, [])
        if not dest_ids:
            continue

        df_line_u = df_line[df_line["shape_id_underline"].astype("Int64") == u]
        for _, cell in df_line_u.iterrows():
            col_start = int(cell["col_start"])
            chosen_id = next(
                (cid for cid in dest_ids
                 if cell_meta[cid]["col_start"] <= col_start
                 <= cell_meta[cid]["col_start"] + cell_meta[cid]["colspan"] - 1),
                max(dest_ids, key=lambda cid: int(cell_meta[cid]["col_start"])),
            )
            rec = records[cell_record_idx[chosen_id]]
            txt = str(cell["text"] or "").strip()
            if txt:
                rec["text"] = (rec["text"] + " " + txt).strip() if rec["text"] else txt
            rec["cell_ids"].append(cell["cell_id"])
            rec["line_ids"].append(int(cell["line_id"]))
            attached_any = True

    return attached_any, anchor_rows


def _attach_pending_to_rows(
    group_df: pd.DataFrame,
    layout_id: int,
    page_number: int,
    anchor_rows: set[int],
    records: list,
    cell_meta: dict,
    cell_record_idx: dict,
    row_to_cell_ids: dict,
) -> None:
    if group_df.empty or not anchor_rows:
        return
    for row_index_anchor in sorted(anchor_rows):
        row_key  = (layout_id, page_number, row_index_anchor)
        dest_ids = row_to_cell_ids.get(row_key, [])
        if not dest_ids:
            continue
        for _, cell in group_df.sort_values(["line_id", "col_start", "cell_id"]).iterrows():
            col_start = int(cell["col_start"])
            chosen_id = next(
                (cid for cid in dest_ids
                 if cell_meta[cid]["col_start"] <= col_start
                 <= cell_meta[cid]["col_start"] + cell_meta[cid]["colspan"] - 1),
                max(dest_ids, key=lambda cid: int(cell_meta[cid]["col_start"])),
            )
            rec = records[cell_record_idx[chosen_id]]
            txt = str(cell["text"] or "").strip()
            if txt:
                rec["text"] = (rec["text"] + " " + txt).strip() if rec["text"] else txt
            rec["cell_ids"].append(cell["cell_id"])
            rec["line_ids"].append(int(cell["line_id"]))


def _build_table_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble table cell records from cells classified as layout_type == "table".

    Groups cells by (layout_id, page_number), then iterates over line_id order,
    accumulating cells into rows based on column coverage and underline cues.
    """
    _EMPTY_COLS = [
        "table_cell_id", "page_number", "layout_id", "table_id",
        "row_start", "col_start", "rowspan", "colspan",
        "text", "text_raw_lines", "cell_ids", "line_ids",
    ]

    if "layout_type" in df_cells.columns:
        df_tables = df_cells[df_cells["layout_type"] == "table"].copy()
    else:
        df_tables = df_cells.copy()

    if df_tables.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    if "shape_id_underline" not in df_tables.columns:
        df_tables["shape_id_underline"] = np.nan

    underline_last_map = _compute_underline_last(df_tables)

    records: list[dict] = []
    table_cell_id_counter = 1
    table_id_counter      = 1

    cell_meta:           dict = {}
    cell_record_idx:     dict = {}
    row_to_cell_ids:     dict = {}
    underline_row_anchor: dict = {}
    open_rowspan:        dict = {}

    for (layout_id, page_number), df_seg in df_tables.groupby(
        ["layout_id", "page_number"], sort=True
    ):
        if df_seg.empty:
            continue

        df_seg   = df_seg.sort_values(["line_id", "col_start", "cell_id"])
        table_id = table_id_counter
        table_id_counter += 1

        row_index = 1
        open_group_line_ids: list[int] = []

        for line_id in sorted(df_seg["line_id"].dropna().drop_duplicates().tolist()):
            df_line = df_seg[df_seg["line_id"] == line_id]

            # Try to attach to an already-flushed underlined row.
            attached, anchor_rows = _try_attach_to_existing_row(
                df_line=df_line,
                layout_id=int(layout_id),
                page_number=int(page_number),
                records=records,
                cell_meta=cell_meta,
                cell_record_idx=cell_record_idx,
                row_to_cell_ids=row_to_cell_ids,
                underline_row_anchor=underline_row_anchor,
            )
            if attached:
                if open_group_line_ids:
                    pending = df_seg[df_seg["line_id"].isin(open_group_line_ids)]
                    _attach_pending_to_rows(
                        group_df=pending,
                        layout_id=int(layout_id),
                        page_number=int(page_number),
                        anchor_rows=anchor_rows,
                        records=records,
                        cell_meta=cell_meta,
                        cell_record_idx=cell_record_idx,
                        row_to_cell_ids=row_to_cell_ids,
                    )
                    open_group_line_ids = []
                continue

            open_group_line_ids.append(int(line_id))
            group_df = df_seg[df_seg["line_id"].isin(open_group_line_ids)]

            band_total_cols = int(group_df["band_total_cols"].max())
            covered_cols    = _compute_covered_cols(group_df)

            is_complete = _is_complete_row(covered_cols, band_total_cols, row_index)

            line_underline_ids = (
                df_line["shape_id_underline"].dropna().astype(int).unique().tolist()
                if "shape_id_underline" in df_line.columns else []
            )
            has_last_u = _has_last_underline(
                line_underline_ids=line_underline_ids,
                layout_id=int(layout_id),
                page_number=int(page_number),
                line_id=int(line_id),
                underline_last_map=underline_last_map,
            )

            flush_reason = (
                "complete_row" if is_complete
                else "last_underline" if has_last_u
                else None
            )

            if flush_reason is not None:
                (table_cell_id_counter, open_rowspan, underline_row_anchor) = _flush_group_to_row(
                    group_df=group_df,
                    layout_id=int(layout_id),
                    page_number=int(page_number),
                    table_id=table_id,
                    row_index=row_index,
                    band_total_cols=band_total_cols,
                    covered_cols=covered_cols,
                    records=records,
                    table_cell_id_counter=table_cell_id_counter,
                    cell_meta=cell_meta,
                    cell_record_idx=cell_record_idx,
                    row_to_cell_ids=row_to_cell_ids,
                    open_rowspan=open_rowspan,
                    underline_row_anchor=underline_row_anchor,
                    flush_reason=flush_reason,
                )
                row_index += 1
                open_group_line_ids = []

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLS)

    return pd.DataFrame.from_records(records)


# --- cell roles ---

def _looks_like_numberish(text: str) -> bool:
    if not text or not text.strip():
        return False
    clean = (text.strip()
             .replace(",", "").replace("$", "").replace("€", "").replace("£", "")
             .replace("(", "").replace(")", "").replace("%", "").replace(" ", ""))
    if not clean:
        return False
    try:
        float(clean)
        return True
    except ValueError:
        pass
    return sum(c.isdigit() for c in text) >= 3


def _assign_cell_roles(df_table_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assign role to each table cell:
        header       first row(s) and any rowspan-extended header rows
        row_label    col_start == 0 on data rows
        footnote     sole bottom-row cell starting at col_start == 0 with colspan ≥ 2
        value_numeric / value_text  all remaining data cells
    """
    if df_table_cells.empty:
        df_table_cells["role"] = pd.Series(dtype=str)
        return df_table_cells

    df = df_table_cells.copy()
    df["role"] = None

    year_re = re.compile(r"\b20\d{2}\b")
    date_re = re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}\b|"
        r"\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}\b",
        re.IGNORECASE,
    )
    unit_phrases = {
        "in thousands", "in millions", "in billions",
        "except per share", "per share", "percentage", "(%)",
        "year ended", "years ended", "months ended", "month ended",
        "quarters ended", "quarter ended", "total", "actual", "adjusted",
        "number", "shares", "amount", "value",
    }

    for (table_id, page_number, layout_id), tbl in df.groupby(
        ["table_id", "page_number", "layout_id"], sort=False
    ):
        rows = sorted(tbl["row_start"].unique())
        if not rows:
            continue
        max_row = max(rows)

        # Header detection
        header_rows: set[int] = {rows[0]}
        for _, cell in tbl[tbl["row_start"] == rows[0]].iterrows():
            for offset in range(1, int(cell["rowspan"])):
                spanned = rows[0] + offset
                if spanned in rows:
                    header_rows.add(spanned)

        for r in rows[1:min(6, len(rows))]:
            if r in header_rows:
                continue
            row_cells = tbl[tbl["row_start"] == r]
            texts_lower = [str(c["text"] or "").strip().lower() for _, c in row_cells.iterrows()]
            combined = " ".join(texts_lower)
            if any(
                _looks_like_numberish(t) and not year_re.search(t)
                for t in texts_lower if t
            ):
                break
            if year_re.search(combined) or date_re.search(combined):
                header_rows.add(r)
                continue
            if any(phrase in combined for phrase in unit_phrases):
                header_rows.add(r)
                continue
            break

        # Footnote detection (sole bottom-row cell at col 0 with colspan ≥ 2)
        bottom_cells = tbl[tbl["row_start"] == max_row]
        is_footnote_row = (
            len(bottom_cells) == 1
            and int(bottom_cells.iloc[0]["col_start"]) == 0
            and int(bottom_cells.iloc[0]["colspan"]) >= 2
        )

        for idx, cell in tbl.iterrows():
            r = int(cell["row_start"])
            if r in header_rows:
                df.at[idx, "role"] = "header"
            elif is_footnote_row and r == max_row:
                df.at[idx, "role"] = "footnote"
            elif int(cell["col_start"]) == 0:
                df.at[idx, "role"] = "row_label"
            else:
                df.at[idx, "role"] = (
                    "value_numeric"
                    if _looks_like_numberish(str(cell["text"] or ""))
                    else "value_text"
                )

    return df


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
        One row per word, with line_id (from step_07_cell_builder output).
        Used for word-gap features in the table-row scorer.
        If None or missing line_id, gap features default to 0.

    Returns
    -------
    df_lines : pd.DataFrame
        With added columns: table_row_score, ratio features,
        layout_id, layout_type.

    df_cells : pd.DataFrame
        With added columns: table_row_score, col_start, col_end, colspan,
        band_total_cols, layout_id, average_table_score, layout_type, block_type.

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

    # Step 1 — score
    df_lines, df_cells = _score_lines(df_lines, df_cells, df_words)

    # Step 2 — column grid
    df_cells = _assign_column_layout(df_cells)

    # Step 3 — merge bands → layout_id
    band_mapping = _find_mergeable_bands(df_cells)
    df_cells["layout_id"] = df_cells["horizontal_band_id"].map(band_mapping)

    # Step 4 — classify
    df_cells = _add_average_table_score(df_cells)
    df_cells = _classify_layout_types(df_cells)

    # Propagate layout_id and layout_type to df_lines.
    line_layout = (
        df_cells.groupby("line_id")[["layout_id", "layout_type"]]
        .first()
        .reset_index()
    )
    df_lines = df_lines.merge(line_layout, on="line_id", how="left")
    df_lines.loc[df_lines["layout_type"] == "table", "block_type"] = "table"

    # Step 5 — build table cells
    df_table_cells = _build_table_cells(df_cells)

    if not df_table_cells.empty:
        df_table_cells = _assign_cell_roles(df_table_cells)

        # Propagate table_id to df_lines via line_ids (explode list → line_id mapping).
        line_table_map = (
            df_table_cells[["table_id", "line_ids"]]
            .explode("line_ids")
            .rename(columns={"line_ids": "line_id"})
            .dropna(subset=["line_id"])
            .drop_duplicates("line_id")
        )
        line_table_map["line_id"] = line_table_map["line_id"].astype(df_lines["line_id"].dtype)
        df_lines = df_lines.merge(line_table_map, on="line_id", how="left")

        cell_table_map = (
            df_table_cells[["table_id", "cell_ids"]]
            .explode("cell_ids")
            .rename(columns={"cell_ids": "cell_id"})
            .dropna(subset=["cell_id"])
            .drop_duplicates("cell_id")
        )
        cell_table_map["cell_id"] = cell_table_map["cell_id"].astype(df_cells["cell_id"].dtype)
        df_cells = df_cells.merge(cell_table_map, on="cell_id", how="left")
    else:
        df_table_cells["role"] = pd.Series(dtype=str)
        df_lines["table_id"] = pd.NA
        df_cells["table_id"] = pd.NA

    return df_lines, df_cells, df_table_cells

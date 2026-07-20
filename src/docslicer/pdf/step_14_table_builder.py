"""
step_14_table_builder.py

"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from .._utils.df_aggregation.registry_aggregator import aggregate_to
from .._utils.df_aggregation.text_merge import merge_text_within_line
from .._utils.df_schemas import TABLE_CELLS_COLS, conform_table_cells
from .._utils.table.table_normalize import normalize_columns_by_table
from .._utils.table.table_header import detect_cell_roles
from .._utils.text_utils import numeric_value_mask

# ============================================================
# CONFIG
# ============================================================



# ============================================================
# Prepare DF
# ============================================================

# Struct-tree table columns (step_06) get a struct_ prefix so they sit
# alongside the grid_ namespace below instead of colliding with the final
# table_id / table_row_id / table_cell_id this module assigns.
_STRUCT_RENAME = {
    "table_id":      "struct_table_id",
    "table_row_id":  "struct_table_row_id",
    "table_cell_id": "struct_table_cell_id",
}

# df_grid_cells columns looked up by grid_cell_id and merged onto df_cells
# with a grid_ prefix (table_grid_id / grid_cell_id themselves are already
# correctly named and are left alone).
_GRID_CELL_LAYOUT_COLS = ("row_start", "col_start", "rowspan", "colspan")


def _merge_grid_cell_layout(
    df_cells: pd.DataFrame,
    df_grid_cells: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rename the struct-tree table columns to the struct_ namespace, then look
    up each cell's grid_cell_id in df_grid_cells and merge its row/col span
    geometry (row_start, col_start, rowspan, colspan) onto df_cells with a
    grid_ prefix. Cells with no grid_cell_id (no detected ruling grid) get NA.
    """
    result = df_cells.rename(columns=_STRUCT_RENAME)

    grid_cols = [f"grid_{c}" for c in _GRID_CELL_LAYOUT_COLS]
    if (df_grid_cells is None or df_grid_cells.empty
            or "grid_cell_id" not in result.columns):
        for col in grid_cols:
            result[col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        return result

    lookup = df_grid_cells[["grid_cell_id", *_GRID_CELL_LAYOUT_COLS]].rename(
        columns=dict(zip(_GRID_CELL_LAYOUT_COLS, grid_cols))
    )
    return result.merge(lookup, on="grid_cell_id", how="left")


def _assign_table_ids(
    df_cells: pd.DataFrame,
    layout_col: str = "layout_id",
    type_col: str = "layout_type",
) -> pd.DataFrame:
    """
    Assign the final table_id: dense, 1-based, one id per distinct layout_id
    whose layout_type is "table" (struct_table_id is untrustworthy on its own
    -- a document can be struct-tagged for some tables and not others, e.g. a
    presentation's native tables vs. pasted-in Excel tables -- so table
    identity is decided fresh here from the reconstructed layout, not the
    struct tree). layout_id is already 1-based and monotonically increasing
    in reading order, so sorting its distinct "table" values reproduces that
    order. Rows outside a table layout get NA.
    """
    result = df_cells.copy()
    result["table_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    if (result.empty or layout_col not in result.columns
            or type_col not in result.columns):
        return result

    is_table = result[type_col] == "table"
    if not is_table.any():
        return result

    table_layout_ids = np.sort(result.loc[is_table, layout_col].unique())
    id_map = {lid: i + 1 for i, lid in enumerate(table_layout_ids)}
    result.loc[is_table, "table_id"] = (
        result.loc[is_table, layout_col].map(id_map).astype("Int64")
    )
    return result


# ============================================================
# Infer column grid
# ============================================================

def _assign_column_layout(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Infer a column grid within each table (grouped by table_id) and assign
    col_start, col_end, colspan, band_total_cols to its cells. One pass per
    table_id; cells outside a table layout (table_id is NA) are skipped and
    get NA in all four columns.

    Algorithm (per table):
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
    n = len(result)

    if "table_id" not in result.columns or not result["table_id"].notna().any():
        for col in ("col_start", "col_end", "colspan", "band_total_cols"):
            result[col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        return result

    in_table = result["table_id"].notna().to_numpy()
    sub_pos  = np.flatnonzero(in_table)

    # Output accumulators (NaN -> NA outside tables); defaults are correct for
    # single-cell tables, multi-cell tables override below.
    cs_full  = np.full(n, np.nan)
    ce_full  = np.full(n, np.nan)
    btc_full = np.full(n, np.nan)
    cs_full[sub_pos]  = 0
    ce_full[sub_pos]  = 0
    btc_full[sub_pos] = 1

    # One global sort by (table_id, line_id, x_left) replaces the per-line
    # sort_values and the nested groupby iteration: everything below works on
    # contiguous numpy slices of these sorted arrays.
    sub = result.iloc[sub_pos]
    tid_arr  = sub["table_id"].astype("int64").to_numpy()
    line_arr = sub["line_id"].to_numpy()
    xl_all   = sub["x_left"].to_numpy(dtype=float)
    xr_all   = sub["x_right"].to_numpy(dtype=float)

    order = np.lexsort((xl_all, line_arr, tid_arr))
    st, sl   = tid_arr[order], line_arr[order]
    sxl, sxr = xl_all[order], xr_all[order]
    spos     = sub_pos[order]                    # positions in result, sorted

    # Only tables with >1 cell on some line need grid inference. Line sizes
    # are counted globally (per line_id over all in-table cells), matching the
    # previous transform-based check.
    _, line_inv, line_counts = np.unique(line_arr, return_inverse=True,
                                         return_counts=True)
    cell_line_size = line_counts[line_inv][order]

    table_bounds = np.concatenate(
        ([0], np.flatnonzero(np.diff(st)) + 1, [len(st)]))
    for ti in range(len(table_bounds) - 1):
        a, z = table_bounds[ti], table_bounds[ti + 1]
        if cell_line_size[a:z].max() <= 1:
            continue  # already initialised with correct single-cell values

        # Line slices within [a, z): sl is sorted, x_left sorted within line.
        line_starts = np.concatenate(
            ([a], a + np.flatnonzero(np.diff(sl[a:z])) + 1, [z]))
        n_lines = len(line_starts) - 1
        line_sizes = np.diff(line_starts)

        # Seed line: most cells; ties go to the lowest line_id (first max).
        seed_idx = int(np.argmax(line_sizes))
        s0, s1 = line_starts[seed_idx], line_starts[seed_idx + 1]
        col_lefts:  list[float] = list(sxl[s0:s1])
        col_rights: list[float] = list(sxr[s0:s1])

        # Walk lines below the seed (DOWN), then above (UP).
        for li in (*range(seed_idx + 1, n_lines),
                   *range(seed_idx - 1, -1, -1)):
            l0, l1 = line_starts[li], line_starts[li + 1]
            xl_arr, xr_arr = sxl[l0:l1], sxr[l0:l1]
            _maybe_split(xl_arr, xr_arr, col_lefts, col_rights)
            for xl, xr in zip(xl_arr, xr_arr):
                _process_cell(xl, xr, col_lefts, col_rights)

        total_cols = len(col_lefts)

        # Vectorised final assignment: broadcast (n_cells, 1) vs (1, n_cols).
        band_xl, band_xr = sxl[a:z], sxr[a:z]
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
            cs_arr = np.full(z - a, total_cols, dtype=np.intp)
            ce_arr = np.full(z - a, total_cols, dtype=np.intp)

        cs_full[spos[a:z]]  = cs_arr
        ce_full[spos[a:z]]  = ce_arr
        btc_full[spos[a:z]] = total_cols

    result["col_start"]       = pd.array(cs_full, dtype="Int64")
    result["col_end"]         = pd.array(ce_full, dtype="Int64")
    result["colspan"]         = pd.array(ce_full - cs_full + 1, dtype="Int64")
    result["band_total_cols"] = pd.array(btc_full, dtype="Int64")
    return result


# ============================================================
# Reconcile colspan across sources
# ============================================================

# The three per-cell column-span estimates, reconciled by max below. Each is
# unreliable on its own and in a different direction, so the largest span any
# source claims wins:
#   colspan         inferred here from x-overlap against the table's column grid;
#                   a near-miss (a header that only just fails to overlap the
#                   column below it) undercounts -- e.g. a "% Change" header that
#                   barely misses the "CER" column reads as span 1.
#   grid_colspan    from the ruling-line grid (df_grid_cells); present only where
#                   an actual grid was detected.
#   struct_col_span from the PDF struct tree; highly inaccurate but occasionally
#                   the only source that catches a merged header.
_COLSPAN_SOURCES = ("colspan", "grid_colspan", "struct_col_span")


def _reconcile_colspan(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Fold the available column-span estimates into the final ``colspan`` by
    taking the per-cell max across _COLSPAN_SOURCES (NA-aware: a source missing
    for a cell doesn't drag the max down; a cell with no source at all stays
    NA), then clamping that claim to the room the cell actually has on its line
    (see _clamp_colspan_to_line). Sources absent from the frame entirely (e.g.
    struct_col_span before the struct extraction lands) are simply skipped, so
    this degrades to max(colspan, grid_colspan) today.

    ``col_end`` is rewritten to follow the reconciled span so the three column
    fields stay consistent (col_end == col_start + colspan - 1).

    The padding of top-row header cells over unoccupied column positions is a
    separate second pass (_pad_header_colspans) that runs after the numeric
    line flags are available.
    """
    result = df_cells.copy()
    present = [c for c in _COLSPAN_SOURCES if c in result.columns]
    if not present:
        return result

    stacked = np.vstack([
        pd.to_numeric(result[c], errors="coerce").to_numpy(float) for c in present
    ])
    with warnings.catch_warnings():          # all-NaN column -> NaN, not a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        merged = np.nanmax(stacked, axis=0)

    merged = _clamp_colspan_to_line(result, merged)
    result["colspan"] = pd.array(merged, dtype="Int64")

    if "col_start" in result.columns:
        # NaN in either operand propagates, so cells with no layout stay NA.
        result["col_end"] = pd.array(
            pd.to_numeric(result["col_start"], errors="coerce").to_numpy(float)
            + merged - 1,
            dtype="Int64",
        )
    return result


def _clamp_colspan_to_line(df_cells: pd.DataFrame, merged: np.ndarray) -> np.ndarray:
    """
    Shrink each cell's claimed span back to what fits on its line.

    grid_colspan / struct_col_span are per *grid* cell, not per text cell: when
    two text cells share one grid cell (or the struct tree over-merges), every
    one of them inherits the full merged span and the naive max hands them each
    the same, impossibly wide claim. E.g. a 4-column band whose line holds a
    cell over columns [0,1] and another over column [2], both mapped to a
    grid cell of span 4: the max makes both span 4, for 8 columns on a 4-column
    row.

    The room a cell actually has is bounded by its neighbours: starting at its
    col_start it can run right only up to the next cell's col_start (or the end
    of the band, for the line's last cell). So the example resolves to span 2
    for the first cell (blocked by the cell at column 2) and span 2 for the
    second (free to the band edge). The layout-inferred colspan is a floor --
    clamping only ever gives back the span x-overlap already established.

    Cells with no table / no column layout, and lines whose geometry is missing,
    pass through untouched.
    """
    needed = {"table_id", "line_id", "col_start", "band_total_cols"}
    if not needed.issubset(df_cells.columns):
        return merged

    col_start = pd.to_numeric(df_cells["col_start"], errors="coerce")
    band      = pd.to_numeric(df_cells["band_total_cols"], errors="coerce")
    usable = (df_cells["table_id"].notna() & col_start.notna() & band.notna()
              & pd.notna(merged)).to_numpy()
    if not usable.any():
        return merged

    # Carry positions explicitly -- df_cells' index is not guaranteed unique.
    sub = pd.DataFrame({
        "pos":       np.flatnonzero(usable),
        "table_id":  df_cells["table_id"].to_numpy()[usable],
        "line_id":   df_cells["line_id"].to_numpy()[usable],
        "col_start": col_start.to_numpy(float)[usable],
        "band":      band.to_numpy(float)[usable],
    }).sort_values(["table_id", "line_id", "col_start"])

    # Exclusive right bound: the next cell's col_start, or the band edge.
    nxt   = sub.groupby(["table_id", "line_id"], sort=False)["col_start"].shift(-1)
    limit = nxt.fillna(sub["band"])
    room  = (limit - sub["col_start"]).clip(lower=1).to_numpy(float)

    out   = merged.copy()
    floor = pd.to_numeric(df_cells["colspan"], errors="coerce").to_numpy(float)
    pos   = sub["pos"].to_numpy()
    out[pos] = np.fmax(np.minimum(out[pos], room), floor[pos])
    return out


# ============================================================
# Pad header colspans (second reconciliation pass)
# ============================================================

# Only the first few lines of a table are header candidates for gap padding.
_PAD_MAX_HEADER_LINES = 3


def _pad_header_colspans(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Second colspan pass: widen top-row header cells over the column positions
    they visually govern but never claimed.

    _assign_column_layout spans a cell over exactly the columns its x-range
    grazes, so a short header ("2025") lands on one column even when it heads
    a group of three, and most PDFs have no grid_colspan / struct_col_span to
    widen it in _reconcile_colspan. The result is top rows with unoccupied
    column positions.

    Header candidates are the first _PAD_MAX_HEADER_LINES lines of each
    table, cut greedily at the first line that is numeric (is_numeric_line)
    or occupies only column 0 (a subheading): if line 2 is numeric, line 3 is
    out too even if it looks fine.

    Each run of unoccupied positions on a header line sits between a *left
    anchor* (the header cell ending just before the run; unusable when its
    col_start is 0 -- label cells are never widened) and a *right anchor*
    (the header cell starting just after; absent at the band edge). Every
    position in the run is resolved against a *reference cell*: the first
    cell on a later line that occupies the position (skipping col_start == 0
    cells, whose geometry belongs to the label column). The position goes to
    whichever anchor is horizontally closer to the reference cell, ties
    going left, using the anchors' original x geometry throughout so one
    extension can't drag the next position along.

    Because both anchors must stay contiguous, the run is split once: the
    left anchor's col_end grows over the positions up to the last
    left-preferring reference, the right anchor's col_start slides back to
    the first right-preferring one. Positions with no reference anywhere
    below stay unfilled unless a claimed position beyond them pulls the
    anchor across. Column 0 is never filled.

    Header lines are processed top-down, and every processed line deposits
    its cell edges (col_start and col_end + 1, post-padding) as *group
    boundaries* that later lines must respect: an anchor is never widened
    across a boundary from a line above, even when raw distance prefers it.
    This prevents staggered headers -- e.g. a row-2 "Change" cell under the
    "US" group grabbing the "$m" column that row 1 already assigned to
    "Emerging Markets"; the blocked position falls to the other anchor when
    that side can take it without crossing a boundary of its own.
    """
    result = df_cells.copy()
    needed = {"table_id", "line_id", "col_start", "col_end", "colspan",
              "band_total_cols", "x_left", "x_right", "is_numeric_line"}
    if not needed.issubset(result.columns) or not result["table_id"].notna().any():
        return result

    col_start = pd.to_numeric(result["col_start"], errors="coerce").to_numpy(float)
    col_end   = pd.to_numeric(result["col_end"],   errors="coerce").to_numpy(float)
    band      = pd.to_numeric(result["band_total_cols"], errors="coerce").to_numpy(float)
    x_left    = pd.to_numeric(result["x_left"],  errors="coerce").to_numpy(float)
    x_right   = pd.to_numeric(result["x_right"], errors="coerce").to_numpy(float)
    line_ids  = result["line_id"].to_numpy()
    numeric_line = result["is_numeric_line"].fillna(False).to_numpy(bool)

    usable = (result["table_id"].notna().to_numpy()
              & ~np.isnan(col_start) & ~np.isnan(col_end) & ~np.isnan(band))
    if not usable.any():
        return result

    changed: list[int] = []
    pos_all = np.flatnonzero(usable)
    for _, grp in pd.Series(pos_all).groupby(result["table_id"].to_numpy()[pos_all]):
        tpos = grp.to_numpy()
        b = int(band[tpos[0]])
        if b <= 1:
            continue

        tline = line_ids[tpos]
        uniq_lines = np.unique(tline)

        # Per line: cell positions sorted by col_start, label column excluded
        # (reference cells starting at col 0 carry label-column geometry).
        ref_cells: dict = {}
        for lid in uniq_lines:
            lpos = tpos[tline == lid]
            lpos = lpos[col_start[lpos] >= 1]
            ref_cells[lid] = lpos[np.argsort(col_start[lpos])]

        def _find_ref(c, after_lid):
            for lid in uniq_lines[uniq_lines > after_lid]:
                for p in ref_cells[lid]:
                    if col_start[p] <= c <= col_end[p]:
                        if np.isnan(x_left[p]) or np.isnan(x_right[p]):
                            return None
                        return x_left[p], x_right[p]
            return None

        # Group boundaries deposited by header lines already processed:
        # cell edges (col_start / col_end + 1) that lower lines must not
        # be widened across.
        edges: set[int] = set()

        for lid in uniq_lines[:_PAD_MAX_HEADER_LINES]:
            lpos = tpos[tline == lid]
            if numeric_line[lpos[0]] or col_start[lpos].max() == 0:
                break  # greedy cut: everything below this line is out too

            lpos = lpos[np.argsort(col_start[lpos])]
            cs_l, ce_l = col_start[lpos], col_end[lpos]
            occupied = np.zeros(b, dtype=bool)
            for s, e in zip(cs_l, ce_l):
                occupied[int(s):int(e) + 1] = True
            missing = np.flatnonzero(~occupied[1:]) + 1

            # Contiguous runs of missing positions.
            breaks = np.flatnonzero(np.diff(missing) > 1) + 1
            for run in (np.split(missing, breaks) if missing.size else ()):
                a, z = int(run[0]), int(run[-1])
                li = np.flatnonzero(ce_l == a - 1)
                ri = np.flatnonzero(cs_l == z + 1)
                left_ok  = li.size > 0 and cs_l[li[0]] != 0
                right_ok = ri.size > 0
                if not left_ok and not right_ok:
                    continue
                lx_right = x_right[lpos[li[0]]] if left_ok else np.nan
                rx_left  = x_left[lpos[ri[0]]]  if right_ok else np.nan

                # How far each anchor may reach without crossing a boundary
                # from a line above. The left anchor [cs, ..] may grow its
                # col_end up to k_max; the right anchor [.., ce] may slide
                # its col_start down to q_min.
                if left_ok:
                    left_cs = int(col_start[lpos[li[0]]])
                    k_max = min((e for e in edges if e > left_cs), default=z + 1) - 1
                else:
                    k_max = a - 1
                if right_ok:
                    right_ce = int(col_end[lpos[ri[0]]])
                    q_min = max((e for e in edges if e <= right_ce), default=a)
                else:
                    q_min = z + 1

                k, q = a - 1, z + 1
                left_open = left_ok
                for c in run:
                    c = int(c)
                    ref = _find_ref(c, lid)
                    if ref is None:
                        continue
                    can_left  = left_open and c <= k_max
                    can_right = right_ok and c >= q_min
                    if not can_left and not can_right:
                        left_open = False  # contiguity: left can't reach past c
                        continue
                    d_left  = ref[0] - lx_right if can_left else np.inf
                    d_right = rx_left - ref[1] if can_right else np.inf
                    if d_left <= d_right:
                        k = c
                    else:
                        q = c
                        break
                if left_ok and k >= a:
                    p = int(lpos[li[0]])
                    col_end[p] = k
                    changed.append(p)
                if right_ok and q <= z:
                    p = int(lpos[ri[0]])
                    col_start[p] = q
                    changed.append(p)

            # This line's (post-padding) cell edges bound the lines below.
            edges.update(int(s) for s in col_start[lpos])
            edges.update(int(e) + 1 for e in col_end[lpos])

    if not changed:
        return result

    idx = np.array(sorted(set(changed)))
    for col, arr in (("col_start", col_start), ("col_end", col_end),
                     ("colspan", col_end - col_start + 1)):
        loc = result.columns.get_loc(col)
        result.iloc[idx, loc] = arr[idx].astype(np.int64)
    return result


# ============================================================
# Row completeness
# ============================================================

# For a table row (one line_id) the occupied column count is the sum of its
# cells' colspans. A row counts as "complete" when that sum reaches a threshold
# that depends on the table's column count and the row's 1-based position: the
# first `num_top_rows` rows use the looser `min_top` (headers are often sparse /
# have merged, span-under-counted cells), the rest use `min_normal`.
#
#     band_total_cols   min_normal   min_top   num_top_rows
#     ---------------------------------------------------
#     2                 2            1         1
#     3                 3            2         1
#     4                 4            3         1
#     5                 4            3         2
#     6                 5            3         3
#     7                 6            3         3
#     8-10              band-2       3         3
#     11+               band-3       3         3
#
# min_normal follows one rule for every width: slack = min(3, (band-2)//3),
# min_normal = band - slack (reproduces the whole column above, incl. the ranges;
# band=10 resolves to band-2). min_top / num_top_rows plateau at 3 from band 6-7
# on and stay flat for all larger tables.
_TOP_THRESHOLDS: dict[int, tuple[int, int]] = {  # band -> (min_top, num_top_rows)
    2: (1, 1), 3: (2, 1), 4: (3, 1), 5: (3, 2), 6: (3, 3), 7: (3, 3),
}


def _row_completeness_thresholds(band_total_cols) -> tuple[int, int, int]:
    """(min_normal, min_top, num_top_rows) for a table this many columns wide."""
    b = int(band_total_cols)
    slack = max(0, min(3, (b - 2) // 3))
    min_normal = max(1, b - slack)
    min_top, num_top_rows = _TOP_THRESHOLDS.get(b, (3, 3))
    min_top = min(min_top, min_normal)          # never require more of a top row
    return min_normal, min_top, num_top_rows


def _assign_row_completeness(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``row_complete`` (nullable boolean): whether its line
    occupies enough column positions to count as a full table row. Occupancy is
    the sum of the line's cell colspans (so it must run AFTER _reconcile_colspan);
    the threshold comes from _row_completeness_thresholds, with the first
    `num_top_rows` rows of each table (ordered by line_id) using the looser
    `min_top`. Cells outside a table (table_id is NA) get NA.
    """
    result = df_cells.copy()
    needed = {"table_id", "band_total_cols", "colspan", "line_id"}
    if not needed.issubset(result.columns) or not result["table_id"].notna().any():
        result["row_complete"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        return result

    in_table = result["table_id"].notna()
    sub = result.loc[in_table, ["table_id", "line_id", "colspan", "band_total_cols"]]

    lines = (
        sub.groupby(["table_id", "line_id"], sort=True)
        .agg(occupied=("colspan", "sum"), band_total_cols=("band_total_cols", "max"))
        .reset_index()
        .sort_values(["table_id", "line_id"])
    )
    lines["row_idx"] = lines.groupby("table_id").cumcount() + 1

    th = lines["band_total_cols"].map(_row_completeness_thresholds)
    lines[["min_normal", "min_top", "num_top"]] = pd.DataFrame(
        th.tolist(), index=lines.index
    )
    threshold = np.where(
        lines["row_idx"].to_numpy() <= lines["num_top"].to_numpy(),
        lines["min_top"].to_numpy(),
        lines["min_normal"].to_numpy(),
    )
    lines["is_complete"] = lines["occupied"].astype("int64").to_numpy() >= threshold

    # A line_id lives in exactly one table, so map completeness back by line_id;
    # non-table lines are absent from the map and land as boolean NA.
    cmap = dict(zip(lines["line_id"], lines["is_complete"]))
    result["row_complete"] = result["line_id"].map(cmap).astype("boolean")
    return result


# ============================================================
# Last row above a table rule
# ============================================================

def _assign_last_tr_below(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``is_last_tr_below`` (nullable boolean): whether its
    line is the last (highest line_id, i.e. last in reading order) among all
    lines that share the same ``shape_id_tr_below`` -- the rule sitting directly
    below this line rather than merely somewhere below it.

    A single table rule backs every row above it, so several consecutive lines
    carry the same shape_id_tr_below; only the bottom-most of them is the row
    immediately above the rule. Example: rule 2803 has lines 17001 and 17002
    above it -> 17001 False, 17002 True. Cells with no shape_id_tr_below get NA.
    """
    result = df_cells.copy()
    if {"shape_id_tr_below", "line_id"}.issubset(result.columns):
        has = result["shape_id_tr_below"].notna()
    else:
        has = pd.Series(False, index=result.index)
    if not has.any():
        result["is_last_tr_below"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        return result

    sub = result.loc[has]
    max_line = sub.groupby("shape_id_tr_below")["line_id"].transform("max")
    is_last = (sub["line_id"] == max_line).reindex(result.index)
    result["is_last_tr_below"] = is_last.astype("boolean")
    return result


# ============================================================
# Subheading rows
# ============================================================

# Trailing inline script tokens ([^...] superscript / [_...] subscript, e.g. a
# "[^(a)]" footnote ref), one or more, with optional surrounding whitespace.
# Stripped from the tail before the ":" test so a footnoted label still counts.
_TRAILING_SCRIPT_RE = re.compile(r"(?:\s*\[[\^_][^\]]*\])+\s*$")


def _assign_is_subheading(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``is_subheading`` (bool): a full-width label row that
    introduces a group of table rows rather than carrying data. True when the
    line is a single cell (cell_count == 1) starting in the first column
    (col_start == 0) whose text ends in ":" or is bold -- e.g. "Market and per
    common share data" or "Race/Ethnicity:".

    col_start is NA outside a table, so this naturally fires only within a table
    layout (the big document title above a table is a single bold cell too, but
    has no col_start and is therefore never flagged).
    """
    result = df_cells.copy()
    if "col_start" not in result.columns or "line_id" not in result.columns:
        result["is_subheading"] = False
        return result

    single = result.groupby("line_id")["cell_id"].transform("count").eq(1)
    col0 = result["col_start"].eq(0).fillna(False)

    if "text" in result.columns:
        # Strip trailing footnote markers -- superscript / subscript script
        # tokens ([^...] / [_...], possibly repeated) that apply_inline_markup
        # appends with no space -- so "Race/Ethnicity:[^(a)]" still reads as
        # ending in ":".
        cleaned = (result["text"].fillna("").astype(str)
                   .str.replace(_TRAILING_SCRIPT_RE, "", regex=True)
                   .str.rstrip())
        ends_colon = cleaned.str.endswith(":")
    else:
        ends_colon = pd.Series(False, index=result.index)
    if "is_bold" in result.columns:
        is_bold = result["is_bold"].fillna(False).astype(bool)
    else:
        is_bold = pd.Series(False, index=result.index)

    result["is_subheading"] = (single & col0 & (ends_colon | is_bold)).to_numpy()
    return result


# ============================================================
# Numeric-like rows
# ============================================================

def _assign_is_numeric_cell(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``is_numeric_cell`` (bool): a single vectorized
    numeric_value_mask pass over the whole ``text`` column -- numbers in
    various formats ($ 802,873 / 10% / 1.29 / (100)), standalone $ or %, and
    dash / NA-style placeholders (- / NA / N/A / n.a.). Downstream row/line
    aggregations (e.g. _assign_is_numeric_line) reuse this column instead of
    recomputing the mask.
    """
    result = df_cells.copy()
    if "text" not in result.columns:
        result["is_numeric_cell"] = False
        return result

    result["is_numeric_cell"] = numeric_value_mask(result["text"]).to_numpy()
    return result


def _assign_is_numeric_line(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``is_numeric_line`` (bool, per line): whether its line
    looks like a numeric table data row -- among the line's populated cells,
    excluding the row-label column (col_start == 0), at least 50% are numeric.
    "Numeric" is is_numeric_cell (numeric_value_mask): numbers in various
    formats ($ 802,873 / 10% / 1.29 / (100)), standalone $, %, or *, and
    dash / NA-style placeholders (- / NA / N/A / n.a.), including those paired
    with a currency symbol or "%" (e.g. "— %").
    """
    result = df_cells.copy()
    if "text" not in result.columns or "line_id" not in result.columns:
        result["is_numeric_line"] = False
        return result

    stripped = result["text"].fillna("").astype(str).str.strip()
    populated = stripped.ne("")
    if "is_numeric_cell" in result.columns:
        numeric = result["is_numeric_cell"].fillna(False).astype(bool)
    else:
        numeric = numeric_value_mask(result["text"])

    if "col_start" in result.columns:
        col0 = result["col_start"].eq(0).fillna(False)
    else:
        col0 = pd.Series(False, index=result.index)
    eligible = populated & ~col0

    line_id = result["line_id"]
    n_eligible = eligible.groupby(line_id).sum()
    n_numeric = (numeric & eligible).groupby(line_id).sum()
    frac_numeric = n_numeric / n_eligible.astype(float).replace(0.0, np.nan)
    ok = frac_numeric.ge(0.5).fillna(False)

    result["is_numeric_line"] = (
        line_id.map(ok).fillna(False).astype(bool)
    )
    return result


# ============================================================
# Grid trustworthiness
# ============================================================

# A grid cell holding this many numeric cells has swallowed several real data
# rows: one logical cell carries one value, so a stack of numbers inside a
# single ruled box means the ruling is too sparse for the content there.
_UNTRUSTWORTHY_GRID_CELL_MIN_NUMERIC = 3


def _assign_grid_trust(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with the two trust columns, both nullable boolean and both
    NA wherever the ruling grid didn't place the cell (no grid_cell_id -- a
    grid-less table, or a cell outside the ruled area):

      ``grid_cell_trustworthy`` -- per grid_cell_id. A grid cell is NOT
        trustworthy when it contains >= _UNTRUSTWORTHY_GRID_CELL_MIN_NUMERIC
        cells with is_numeric_cell True. A trustworthy grid cell holds one
        logical value: either text (possibly wrapped over several lines, e.g.
        "Comirnaty / (COVID-19 / Vaccine, / mRNA)") or a single number. A box
        holding a stack of numbers (481 / 427 / 0.13 / 0.06) is several data
        rows the sparse ruling failed to separate.

      ``grid_row_trustworthy`` -- per (table_grid_id, grid_row_start). True
        when every grid cell in that band is trustworthy and no cell in the
        band lies on a numeric line (is_numeric_line): one over-packed box
        makes the whole band's row structure unusable, and a numeric data
        line inside a band means the ruling may have merged data rows.

    Trust is scored per grid cell rather than per table because it varies
    within one table: a header line can be tightly ruled (trustworthy) while
    the body below it has only sparse rules that merge many data rows into each
    band (untrustworthy).
    """
    result = df_cells.copy()
    result["grid_cell_trustworthy"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["grid_row_trustworthy"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    need = {"grid_cell_id", "is_numeric_cell", "table_grid_id", "grid_row_start"}
    if not need.issubset(result.columns):
        return result

    placed = result["grid_cell_id"].notna()
    if not placed.any():
        return result

    numeric = result["is_numeric_cell"].fillna(False).astype(bool)
    sub = result.loc[placed]

    numeric_per_grid_cell = numeric.loc[placed].groupby(sub["grid_cell_id"]).sum()
    cell_trust = numeric_per_grid_cell < _UNTRUSTWORTHY_GRID_CELL_MIN_NUMERIC
    cell_trust_cells = sub["grid_cell_id"].map(cell_trust)

    # A band is only as good as its worst grid cell, so min() over the band.
    row_trust = cell_trust_cells.groupby(
        [sub["table_grid_id"], sub["grid_row_start"]], dropna=False
    ).min()
    if "is_numeric_line" in sub.columns:
        band_has_numeric_line = (
            sub["is_numeric_line"].fillna(False).astype(bool).groupby(
                [sub["table_grid_id"], sub["grid_row_start"]], dropna=False
            ).max()
        )
        row_trust &= ~band_has_numeric_line
    row_trust_cells = pd.MultiIndex.from_arrays(
        [sub["table_grid_id"], sub["grid_row_start"]]
    ).map(row_trust)

    result.loc[placed, "grid_cell_trustworthy"] = pd.array(
        cell_trust_cells.to_numpy(), dtype="boolean")
    result.loc[placed, "grid_row_trustworthy"] = pd.array(
        np.asarray(row_trust_cells), dtype="boolean")
    return result



# ============================================================
# Row layout (row_start / rowspan)
# ============================================================

def _line_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (table_id, line_id) across ALL tables at once, sorted by
    (table_id, line_id) -- reading order within each table -- with the
    per-line signals row assignment needs. One global groupby instead of one
    per table: the per-call fixed cost of groupby().agg() dominates on
    documents with hundreds of small tables. groupby().first() skips NA, so
    struct_row is the line's first non-NA struct_table_row_id.

    ``grid_id`` / ``band`` locate the line in the ruling grid, both by max
    across the line's cells. Max, not first: a cell spanning several bands
    carries the grid_row_start of the *top* band it covers, so a line sitting
    lower in that span sees a too-small value from it. Every other cell on the
    line starts at the line's own band, so the max is that band. A line that
    falls entirely inside spanning cells reports their top band and so reads as
    a continuation of it, which is what it is.
    """
    agg: dict[str, tuple[str, str]] = {}
    if "struct_table_row_id" in grp.columns:
        agg["struct_row"] = ("struct_table_row_id", "first")
    if "row_complete" in grp.columns:
        agg["row_complete"] = ("row_complete", "max")
    if "is_subheading" in grp.columns:
        agg["is_subheading"] = ("is_subheading", "any")
    if "is_last_tr_below" in grp.columns:
        agg["is_last_tr"] = ("is_last_tr_below", "max")
    if "is_numeric_line" in grp.columns:
        agg["is_numeric_line"] = ("is_numeric_line", "any")
    if "table_grid_id" in grp.columns:
        agg["grid_id"] = ("table_grid_id", "max")
    if "grid_row_start" in grp.columns:
        agg["band"] = ("grid_row_start", "max")

    lines = grp.groupby(["table_id", "line_id"], sort=True).agg(**agg)
    for col, default in (("struct_row", pd.NA), ("row_complete", False),
                         ("is_subheading", False), ("is_last_tr", False),
                         ("is_numeric_line", False), ("grid_id", pd.NA),
                         ("band", pd.NA)):
        if col not in lines.columns:
            lines[col] = default
    return lines


def _band_trust_maps(grp: pd.DataFrame) -> dict[object, dict[tuple[float, float], bool]]:
    """
    Per-table band trust, computed for ALL tables in one groupby:
    table_id -> {(grid_id, band): grid_row_trustworthy} for every band the
    ruling grid placed *within that table's cells*. grid_row_trustworthy is
    constant within a band by construction (assigned per (table_grid_id,
    grid_row_start) in _assign_grid_trust), so this is just a lookup keyed the
    way the line walk asks for it. The maps must stay per table: one grid can
    straddle two detected tables, and a band a table never touches must read
    as untrusted for it even if the neighbouring table trusts it. Bands the
    grid never placed are absent -> treated as untrusted.
    """
    if not {"table_id", "table_grid_id", "grid_row_start",
            "grid_row_trustworthy"}.issubset(grp.columns):
        return {}
    g = pd.to_numeric(grp["table_grid_id"], errors="coerce")
    b = pd.to_numeric(grp["grid_row_start"], errors="coerce")
    placed = g.notna() & b.notna()
    if not placed.any():
        return {}
    trust = grp["grid_row_trustworthy"].fillna(False).astype(bool)
    by_band = (
        pd.DataFrame({"tid": grp["table_id"][placed],
                      "g": g[placed], "b": b[placed], "t": trust[placed]})
        .groupby(["tid", "g", "b"])["t"].all()
    )
    maps: dict[object, dict[tuple[float, float], bool]] = {}
    for (tid, gg, bb), v in by_band.items():
        maps.setdefault(tid, {})[(float(gg), float(bb))] = bool(v)
    return maps


def _grid_span(
    grid_id: float,
    band: float,
    span: float,
    trust_by_band: dict[tuple[float, float], bool],
) -> float:
    """
    The rowspan a cell's ruling geometry is worth, or NaN when the grid can't
    be believed for it. Every trusted band collapses to exactly one logical row
    (that's what trusting it means), so a cell spanning trusted bands spans that
    many rows and grid_rowspan carries over unchanged. If any band it covers is
    untrusted, that band explodes into however many rows the scratch walk finds
    inside it and the grid's span no longer counts rows at all -> NaN, and the
    caller falls back to 1.
    """
    if pd.isna(grid_id) or pd.isna(band) or pd.isna(span) or span <= 1:
        return np.nan
    if all(trust_by_band.get((float(grid_id), float(b)), False)
           for b in range(int(band), int(band) + int(span))):
        return float(span)
    return np.nan


def _assign_row_layout(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``row_start`` (0-based logical row within the table, shared by all
    cells of a line) and ``rowspan`` per cell.

    STRUCT (tagged table) takes precedence: walk the table's lines in line_id
    order; row_start increments whenever the line's struct_table_row_id differs
    from the previous line's (a line with no struct_table_row_id counts as its
    own new row).

    Otherwise walk the lines in line_id order and decide per line whether it
    opens a new row, because trust is per band, not per table. A line opens a
    new row when:

      - it enters a different grid band than the line above it (including
        entering or leaving the ruled area altogether) -- a band boundary is a
        row boundary whether or not the band is trusted; or
      - its band is NOT trusted (or the grid never placed it) and the SCRATCH
        rule fires: the line is a complete row (row_complete), a subheading
        (is_subheading), or a numeric-like data row (is_numeric_line), or the
        line above sat directly on a table rule (is_last_tr_below).

    Inside a TRUSTED band the scratch rule is ignored, so every line of the
    band folds into the one row the ruling says it is -- that's a header whose
    columns wrap over four lines becoming one row. Inside an UNTRUSTED band the
    ruling is just a box drawn around many real rows, so the scratch rule splits
    it line by line. A table ruled tightly at the header and sparsely below --
    the common financial-statement layout -- therefore gets both: one folded
    header row, then a row per data line.

    This is one walk, not a grid/scratch fork: a fully-trusted grid is the case
    where every band folds (one row per band, exactly the ruling's structure),
    and a grid-less table is the case where no band exists and the scratch rule
    decides every line.

    rowspan is struct_row_span where > 1, else the ruling grid's grid_rowspan
    where every band the cell covers is trusted (see _grid_span), else 1.

    Cells outside a table layout get NA in both columns.
    """
    result = df_cells.copy()
    n = len(result)

    if "table_id" not in result.columns or not result["table_id"].notna().any():
        result["row_start"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["rowspan"]   = pd.Series(pd.NA, index=result.index, dtype="Int64")
        return result

    def _col_as_float(name: str) -> np.ndarray:
        if name in result.columns:
            return pd.to_numeric(result[name], errors="coerce").to_numpy(float)
        return np.full(n, np.nan)

    struct_span    = _col_as_float("struct_row_span")
    grid_row_start = _col_as_float("grid_row_start")
    grid_rowspan   = _col_as_float("grid_rowspan")
    grid_id_num    = _col_as_float("table_grid_id")

    in_table = result["table_id"].notna().to_numpy()
    sub_pos  = np.flatnonzero(in_table)          # positions of in-table cells
    sub      = result.iloc[sub_pos]

    # One global line summary and one global (but per-table-keyed) band-trust
    # pass (see the helpers' docstrings), then a cheap numpy walk per table
    # instead of per-table pandas groupbys.
    lines_all  = _line_summary(sub)
    trust_maps = _band_trust_maps(sub)

    g = grid_id_num[sub_pos]
    b = grid_row_start[sub_pos]
    s = grid_rowspan[sub_pos]
    sub_tid = sub["table_id"].to_numpy()

    # The ruling grid's own claim on each cell, kept only where every band
    # the cell covers is trusted; NaN otherwise (see _grid_span). Only cells
    # with span > 1 can produce a value, so only those are checked.
    span_from_grid = np.full(len(sub_pos), np.nan)
    for i in np.flatnonzero(~np.isnan(g) & ~np.isnan(b) & (s > 1)):
        span_from_grid[i] = _grid_span(
            g[i], b[i], s[i], trust_maps.get(sub_tid[i], {}))

    sub_line  = sub["line_id"].to_numpy()
    row_start_sub = np.full(len(sub_pos), np.nan)

    lines_by_tid = {
        tid: frame.droplevel(0)
        for tid, frame in lines_all.groupby(level=0, sort=False)
    }
    for tid, cpos in sub.groupby("table_id", sort=False).indices.items():
        lines = lines_by_tid[tid]
        line_index = lines.index.to_numpy()      # sorted line_ids of this table
        trust_by_band = trust_maps.get(tid, {})

        if lines["struct_row"].notna().any():
            # New logical row whenever struct_table_row_id changes between
            # consecutive lines. NA lines (untagged, e.g. an inserted
            # subheading) get a unique sentinel so each one is its own row.
            keys = lines["struct_row"].astype(object)
            keys = keys.where(keys.notna(), -pd.Series(lines.index, index=lines.index))
            row_line = ((keys != keys.shift()).cumsum() - 1).to_numpy(float)
            row_start_sub[cpos] = row_line[
                np.searchsorted(line_index, sub_line[cpos])]
        else:
            # Band key per line; the grid-less / unplaced lines share the -1
            # sentinel so a run of them reads as one band and the scratch rule
            # (not a spurious band change) decides where their rows break.
            grid_vals = lines["grid_id"].astype(float).fillna(-1.0).to_numpy()
            band_vals = lines["band"].astype(float).fillna(-1.0).to_numpy()
            band_changed = np.ones(len(lines), dtype=bool)
            band_changed[1:] = ((grid_vals[1:] != grid_vals[:-1])
                                | (band_vals[1:] != band_vals[:-1]))
            band_trusted = np.fromiter(
                (trust_by_band.get((gv, bv), False)
                 for gv, bv in zip(grid_vals, band_vals)),
                dtype=bool, count=len(lines))

            scratch = (
                lines["row_complete"].fillna(False).to_numpy(bool)
                | lines["is_subheading"].fillna(False).to_numpy(bool)
                | lines["is_numeric_line"].fillna(False).to_numpy(bool)
            )
            is_last_prev = np.empty(len(lines), dtype=bool)
            is_last_prev[0] = False
            is_last_prev[1:] = lines["is_last_tr"].fillna(False).to_numpy(bool)[:-1]
            scratch |= is_last_prev

            increment = band_changed | (~band_trusted & scratch)
            increment[0] = False                # first line is always row 0
            row_line = np.cumsum(increment).astype(float)

            # A trusted band folds to exactly one row, so the first line's row
            # (in line order) is that row.
            band_row: dict[tuple[float, float], float] = {}
            for i in np.flatnonzero(band_trusted):
                key = (grid_vals[i], band_vals[i])
                if key not in band_row:
                    band_row[key] = row_line[i]

            # Where a trusted band placed the cell, its row comes from ITS OWN
            # band rather than from the line its text landed on. The two differ
            # for a cell whose text doesn't sit at the top of its box: a tall
            # vertically-centred row label is grouped into a line belonging to a
            # band further down, and keying it by that line would flush it late,
            # to the row its text sits next to instead of the row it opens.
            # Cells the grid never placed, or placed in an untrusted band, are
            # absent from band_row -> they keep their line's row.
            line_row = row_line[np.searchsorted(line_index, sub_line[cpos])]
            cg = np.where(np.isnan(g[cpos]), -1.0, g[cpos])
            cb = np.where(np.isnan(b[cpos]), -1.0, b[cpos])
            row_start_sub[cpos] = [
                band_row.get((cg[j], cb[j]), line_row[j])
                for j in range(len(cpos))
            ]

    # rowspan: struct_row_span where > 1, else the trusted grid span, else 1.
    ss = struct_span[sub_pos]
    rowspan_sub = np.where(ss > 1, ss,
                           np.where(np.isnan(span_from_grid), 1.0, span_from_grid))

    row_start_full = np.full(n, np.nan)
    rowspan_full   = np.full(n, np.nan)
    row_start_full[sub_pos] = row_start_sub
    rowspan_full[sub_pos]   = rowspan_sub
    result["row_start"] = pd.array(row_start_full, dtype="Int64")
    result["rowspan"]   = pd.array(rowspan_full, dtype="Int64")
    return result


# ============================================================
# Final table_cell_id
# ============================================================

def _assign_table_cell_ids(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the final ``table_cell_id``: dense, 1-based, global across the
    document (mirroring how the DOCX pipeline counts cells), in reading order
    -- by table_id, then row_start, then col_start. It identifies the LOGICAL
    table cell, so physical cells sharing (table_id, row_start, col_start) --
    e.g. a wrapped label whose continuation lines were folded into the same
    logical row -- share one id. Cells outside a table, or without a resolved
    row_start, get NA.
    """
    result = df_cells.copy()
    result["table_cell_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    key_cols = ["table_id", "row_start", "col_start"]
    if not set(key_cols).issubset(result.columns):
        return result
    mask = result["table_id"].notna() & result["row_start"].notna()
    if not mask.any():
        return result

    ordered = result.loc[mask, key_cols].sort_values(key_cols, kind="stable")
    ids = ordered.groupby(key_cols, sort=False, dropna=False).ngroup() + 1
    result.loc[ids.index, "table_cell_id"] = pd.array(
        ids.to_numpy(), dtype="Int64")
    return result


# ============================================================
# Table cells frame
# ============================================================

def _build_table_cells_df(df_cells: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Fold df_cells down to one row per logical table cell (table_cell_id):
    fragment texts join with a newline in reading order (line_id, then x_left)
    via merge_text_within_line, rowspan/colspan take the max across fragments
    (the widest claim wins, same policy as _reconcile_colspan), everything
    else is constant within a cell and takes the first value.
    table_header_flag/bold_ratio (and the is_bold recomputed from it) are
    re-aggregated via aggregate_to. Rows are ordered by table_cell_id, which
    is itself reading order.
    """
    if "table_cell_id" not in df_cells.columns or not df_cells["table_cell_id"].notna().any():
        return pd.DataFrame(columns=TABLE_CELLS_COLS)

    sub = df_cells.loc[df_cells["table_cell_id"].notna()].copy()
    sort_cols = [c for c in ("table_cell_id", "line_id", "x_left") if c in sub.columns]
    sub = sub.sort_values(sort_cols, kind="stable")

    agg: dict[str, tuple[str, object]] = {}
    for col in ("page_number", "page_label", "layout_id", "table_id",
                "row_start", "col_start"):
        if col in sub.columns:
            agg[col] = (col, "first")
    for col in ("rowspan", "colspan"):
        if col in sub.columns:
            agg[col] = (col, "max")

    out = (
        sub.groupby("table_cell_id", sort=True)
        .agg(**agg)
        .reset_index()
    )
    out["text"] = out["table_cell_id"].map(
        merge_text_within_line(sub["text"], sub["table_cell_id"], sep="\n")
    )

    style_cols = [c for c in ("table_header_flag", "bold_ratio", "char_count")
                  if c in sub.columns]
    if style_cols:
        style = aggregate_to(sub[["table_cell_id", *style_cols]],
                              by="table_cell_id")
        out = out.merge(style.drop(columns=["char_count"], errors="ignore"),
                         on="table_cell_id", how="left")

    if not out.empty and "table_id" in out.columns:
        # Drop blank/paren spacer columns and merge %, (b), $ cells into their
        # value neighbor (shared with the HTML pipeline). Cells can be dropped
        # here, so table_cell_id is re-densified in reading order afterwards.
        out = normalize_columns_by_table(out)
        out = out.sort_values(["table_id", "row_start", "col_start"],
                              kind="stable").reset_index(drop=True)
        out["table_cell_id"] = pd.array(range(1, len(out) + 1), dtype="Int64")

        # detect_cell_roles processes every table in one vectorized pass,
        # grouping internally on (table_id, row_start).
        out = detect_cell_roles(out)
    else:
        out["table_cell_role"] = pd.Series(pd.NA, index=out.index, dtype="string")

    return out #conform_table_cells(out, debug=debug)


# ============================================================
# PUBLIC API
# ============================================================

def build_tables(
    df_cells: pd.DataFrame,
    df_grid_cells: pd.DataFrame,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_cells = _merge_grid_cell_layout(df_cells, df_grid_cells)
    df_cells = _assign_table_ids(df_cells)
    df_cells = _assign_column_layout(df_cells)
    df_cells = _reconcile_colspan(df_cells)
    df_cells = _assign_is_numeric_cell(df_cells)
    df_cells = _assign_is_numeric_line(df_cells)
    df_cells = _pad_header_colspans(df_cells)
    df_cells = _assign_row_completeness(df_cells)
    df_cells = _assign_last_tr_below(df_cells)
    df_cells = _assign_is_subheading(df_cells)
    df_cells = _assign_grid_trust(df_cells)
    df_cells = _assign_row_layout(df_cells)
    df_cells = _assign_table_cell_ids(df_cells)

    df_table_cells = _build_table_cells_df(df_cells, debug=debug)

    return df_cells, df_table_cells

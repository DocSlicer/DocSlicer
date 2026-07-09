"""
step_14_table_builder.py

"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from .._utils.table_utils import detect_cell_roles
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
            result[col] = pd.array([pd.NA] * len(result), dtype="Int64")
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
    result["table_id"] = pd.array([pd.NA] * len(result), dtype="Int64")

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
    for col in ("col_start", "col_end", "colspan", "band_total_cols"):
        result[col] = pd.array([pd.NA] * len(result), dtype="Int64")

    if "table_id" not in result.columns:
        return result
    in_table = result["table_id"].notna()
    if not in_table.any():
        return result

    # Defaults are correct for single-cell tables; multi-cell tables override below.
    table_idx = result.index[in_table]
    result.loc[table_idx, "col_start"]       = 0
    result.loc[table_idx, "col_end"]         = 0
    result.loc[table_idx, "colspan"]         = 1
    result.loc[table_idx, "band_total_cols"] = 1

    # Only tables with >1 cell on some line need grid inference.
    line_sizes = result.loc[table_idx].groupby("line_id", sort=False)["cell_id"].transform("count")
    table_max  = line_sizes.groupby(result.loc[table_idx, "table_id"]).transform("max")
    multi_cell_tables = set(result.loc[table_idx, "table_id"][table_max > 1].unique())

    for table_id, band_df in result.loc[table_idx].groupby("table_id", sort=False):
        if table_id not in multi_cell_tables:
            continue  # already initialised with correct single-cell values

        line_cell_counts = band_df.groupby("line_id").size()
        seed_line_id     = line_cell_counts.idxmax()

        # Pre-group by line_id once; store sorted numpy arrays to avoid
        # repeated DataFrame filtering and itertuples overhead.
        line_groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for lid, grp in band_df.groupby("line_id", sort=False):
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
    NA). Sources absent from the frame entirely (e.g. struct_col_span before the
    struct extraction lands) are simply skipped, so this degrades to
    max(colspan, grid_colspan) today.

    NOTE (future): this only *widens* to the largest claimed span. It does not
    yet look at unoccupied column positions to pad top-row header cells over the
    gaps they should cover -- that padding pass would slot in here, after the max.
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
    result["colspan"] = pd.array(merged, dtype="Int64")
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
        result["row_complete"] = pd.array([pd.NA] * len(result), dtype="boolean")
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
        result["is_last_tr_below"] = pd.array([pd.NA] * len(result), dtype="boolean")
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

def _assign_is_numeric_like(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``is_numeric_like`` (bool, per line): whether its line
    looks like a table data row -- at least 3 populated cells, the first
    populated cell (by x_left) is text (not a numeric value), and at least 2 of
    the remaining cells are numeric. "Numeric" is numeric_value_mask: numbers
    in various formats ($ 802,873 / 10% / 1.29 / (100)), standalone $ or %,
    and dash / NA-style placeholders (- / NA / N/A / n.a.).
    """
    result = df_cells.copy()
    if "text" not in result.columns or "line_id" not in result.columns:
        result["is_numeric_like"] = False
        return result

    stripped = result["text"].fillna("").astype(str).str.strip()
    populated = stripped.ne("")
    numeric = numeric_value_mask(result["text"])

    sub = result.loc[populated, ["line_id"]].copy()
    sub["_num"] = numeric.loc[sub.index]
    if "x_left" in result.columns:
        sub["_x"] = result.loc[sub.index, "x_left"]
        sub = sub.sort_values(["line_id", "_x"], kind="stable")

    stats = sub.groupby("line_id")["_num"].agg(
        n_populated="size", n_numeric="sum", first_numeric="first"
    )
    ok = (
        (stats["n_populated"] >= 3)
        & ~stats["first_numeric"].astype(bool)
        & (stats["n_numeric"] >= 2)
    )
    result["is_numeric_like"] = (
        result["line_id"].map(ok).fillna(False).astype(bool)
    )
    return result


# ============================================================
# Grid trustworthiness
# ============================================================

# A grid_row_start holding this many complete rows means several table rows
# collapsed into one detected grid band -- the ruling grid is too coarse for the
# content, so it's not trustworthy as the row structure.
_SPARSE_GRID_ROW_MIN_COMPLETE = 3


def _assign_grid_trustworthy(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every cell with ``grid_trustworthy`` (nullable boolean): whether the
    detected ruling grid can be trusted as the table's row/column structure.
    Only meaningful for tables that actually have a detected grid; cells in a
    grid-less table (or outside any table) get NA.

    Per table_id:
      1. If the table's gridded rows don't all share one table_grid_id (the
         layout spans more than one detected grid), it's not trustworthy.
         Rows with no table_grid_id (subheadings, footnotes) are ignored here.
      2. Otherwise, trustworthy only when there is at least one qualifying data
         band and no band is over-packed: if any grid_row_start holds
         >= _SPARSE_GRID_ROW_MIN_COMPLETE distinct complete lines (row_complete)
         the grid is too sparse (many real rows in one band) -> not trustworthy;
         and if NO qualifying band exists at all (the grid never demonstrably
         places a complete data row below the top band) -> also not trustworthy.
         Excluded from the count: single-cell lines (cell_count == 1, e.g. a
         subheading), and the top band (grid_row_start == 0), whose header rows
         clear only the looser min_top threshold and would trip a false sparse.

    E.g. a fully-ruled table (one grid row per data row) is trustworthy; a table
    with far more rows than ruling lines packs several complete rows into a grid
    band and is not.
    """
    result = df_cells.copy()
    result["grid_trustworthy"] = pd.array([pd.NA] * len(result), dtype="boolean")

    need = {"table_id", "table_grid_id", "grid_row_start", "row_complete", "line_id"}
    if not need.issubset(result.columns) or not result["table_id"].notna().any():
        return result

    in_table = result["table_id"].notna()
    for _, grp in result.loc[in_table].groupby("table_id", sort=False):
        grid_ids = grp["table_grid_id"].dropna().unique()
        if len(grid_ids) == 0:
            continue  # no detected grid -> not applicable, leave NA
        if len(grid_ids) > 1:
            trust = False
        else:
            # Count only complete, multi-cell lines below the top band: a
            # single-cell line (cell_count == 1, e.g. a subheading) must not
            # inflate a band, and grid_row_start 0 rows pass only the looser
            # min_top threshold so they'd falsely read as over-packed.
            cells_per_line = grp.groupby("line_id")["cell_id"].transform("count")
            keep = (
                grp["row_complete"].fillna(False).astype(bool)
                & (cells_per_line > 1)
                & grp["grid_row_start"].ne(0).fillna(False).astype(bool)
            )
            per_row = grp[keep].groupby("grid_row_start")["line_id"].nunique()
            # Need at least one qualifying data band with an acceptable count:
            # a grid that never demonstrably places a complete data row below the
            # top band (per_row empty) isn't trusted either.
            trust = bool(len(per_row) > 0
                         and (per_row < _SPARSE_GRID_ROW_MIN_COMPLETE).all())
        result.loc[grp.index, "grid_trustworthy"] = trust
    return result



# ============================================================
# Row layout (row_start / rowspan)
# ============================================================

def _line_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per line_id of a single table, sorted by line_id (reading order),
    with the per-line signals row assignment needs. groupby().first() skips NA,
    so struct_row is the line's first non-NA struct_table_row_id.
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
    if "is_numeric_like" in grp.columns:
        agg["is_numeric_like"] = ("is_numeric_like", "any")

    lines = grp.groupby("line_id").agg(**agg).sort_index()
    for col, default in (("struct_row", pd.NA), ("row_complete", False),
                         ("is_subheading", False), ("is_last_tr", False),
                         ("is_numeric_like", False)):
        if col not in lines.columns:
            lines[col] = default
    return lines


def _assign_row_layout(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``row_start`` (0-based logical row within the table, shared by all
    cells of a line) and ``rowspan`` per cell. Three strategies, chosen per
    table_id:

    1. STRUCT (tagged table) -- takes precedence even over a trustworthy grid.
       Walk the table's lines in line_id order; row_start increments whenever
       the line's struct_table_row_id differs from the previous line's (a line
       with no struct_table_row_id counts as its own new row). rowspan is
       struct_row_span where > 1; where struct claims 1 but a trustworthy grid
       says grid_rowspan > 1, the grid wins; else 1.

    2. GRID (grid_trustworthy is True, no struct tagging) -- copy the detected
       ruling grid's grid_row_start / grid_rowspan per cell. Cells the grid
       didn't capture (no grid_cell_id, e.g. a footnote line outside the ruled
       area) keep NA row_start.

    3. SCRATCH (no struct, no trusted grid) -- first line gets row_start 0;
       each next line increments when the line itself is a complete row
       (row_complete), a subheading (is_subheading), or a numeric-like data
       row (is_numeric_like), or when the previous line sat directly above a
       table rule (is_last_tr_below) -- otherwise it continues the previous
       logical row (wrapped / multiline cells). rowspan is 1 everywhere.

    Cells outside a table layout get NA in both columns.
    """
    result = df_cells.copy()
    result["row_start"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["rowspan"]   = pd.array([pd.NA] * len(result), dtype="Int64")

    if "table_id" not in result.columns or not result["table_id"].notna().any():
        return result

    struct_span = (
        pd.to_numeric(result["struct_row_span"], errors="coerce")
        if "struct_row_span" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    grid_row_start = (
        pd.to_numeric(result["grid_row_start"], errors="coerce")
        if "grid_row_start" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    grid_rowspan = (
        pd.to_numeric(result["grid_rowspan"], errors="coerce")
        if "grid_rowspan" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    grid_trust = (
        result["grid_trustworthy"].fillna(False).astype(bool)
        if "grid_trustworthy" in result.columns
        else pd.Series(False, index=result.index)
    )

    in_table = result["table_id"].notna()
    for _, grp in result.loc[in_table].groupby("table_id", sort=False):
        lines = _line_summary(grp)
        has_struct = lines["struct_row"].notna().any()
        trusted_grid = bool(grid_trust.loc[grp.index].any())

        if has_struct:
            # New logical row whenever struct_table_row_id changes between
            # consecutive lines. NA lines (untagged, e.g. an inserted
            # subheading) get a unique sentinel so each one is its own row.
            keys = lines["struct_row"].astype(object)
            keys = keys.where(keys.notna(), -pd.Series(lines.index, index=lines.index))
            row_start_by_line = (keys != keys.shift()).cumsum() - 1

            span = struct_span.loc[grp.index].where(struct_span.loc[grp.index] > 1)
            if trusted_grid:
                # Struct says span 1 (or nothing) but the trusted ruling grid
                # detected a taller cell -> take the grid's rowspan.
                span = span.fillna(grid_rowspan.loc[grp.index].where(
                    grid_rowspan.loc[grp.index] > 1))
            rowspan_cells = span.fillna(1)

        elif trusted_grid:
            row_start_cells = grid_row_start.loc[grp.index]
            # rowspan defaults to 1 only where the grid placed the cell at
            # all; a cell outside the ruled area stays NA in both columns.
            span_cells = grid_rowspan.loc[grp.index].fillna(1.0).where(
                row_start_cells.notna())
            result.loc[grp.index, "row_start"] = pd.array(
                row_start_cells.to_numpy(), dtype="Int64")
            result.loc[grp.index, "rowspan"] = pd.array(
                span_cells.to_numpy(), dtype="Int64")
            continue

        else:
            increment = (
                lines["row_complete"].fillna(False).astype(bool)
                | lines["is_subheading"].fillna(False).astype(bool)
                | lines["is_numeric_like"].fillna(False).astype(bool)
                | lines["is_last_tr"].fillna(False).astype(bool).shift(fill_value=False)
            )
            increment.iloc[0] = False           # first line is always row 0
            row_start_by_line = increment.cumsum()
            rowspan_cells = pd.Series(1, index=grp.index)

        result.loc[grp.index, "row_start"] = pd.array(
            grp["line_id"].map(row_start_by_line).to_numpy(), dtype="Int64")
        result.loc[grp.index, "rowspan"] = pd.array(
            rowspan_cells.to_numpy(), dtype="Int64")

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
    result["table_cell_id"] = pd.array([pd.NA] * len(result), dtype="Int64")

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

# df_table_cells output columns, in order. table_cell_role is reserved for
# detect_cell_roles and left NA for now.
_TABLE_CELLS_COLS = [
    "page_number", "page_label", "layout_id", "table_id", "table_cell_id",
    "text", "row_start", "col_start", "rowspan", "colspan", "table_cell_role",
]


def _build_table_cells_df(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Fold df_cells down to one row per logical table cell (table_cell_id):
    fragment texts join with a newline in reading order (line_id, then x_left),
    rowspan/colspan take the max across fragments (the widest claim wins, same
    policy as _reconcile_colspan), everything else is constant within a cell
    and takes the first value. Rows are ordered by table_cell_id, which is
    itself reading order.
    """
    if "table_cell_id" not in df_cells.columns or not df_cells["table_cell_id"].notna().any():
        return pd.DataFrame(columns=_TABLE_CELLS_COLS)

    sub = df_cells.loc[df_cells["table_cell_id"].notna()].copy()
    sort_cols = [c for c in ("table_cell_id", "line_id", "x_left") if c in sub.columns]
    sub = sub.sort_values(sort_cols, kind="stable")

    agg: dict[str, tuple[str, object]] = {
        "text": ("text", lambda s: "\n".join(s.dropna().astype(str))),
    }
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
    out["table_cell_role"] = pd.array([pd.NA] * len(out), dtype="string")

    for col in _TABLE_CELLS_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[_TABLE_CELLS_COLS]


# ============================================================
# PUBLIC API
# ============================================================

def build_tables(
    df_cells: pd.DataFrame,
    df_grid_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_cells = _merge_grid_cell_layout(df_cells, df_grid_cells)
    df_cells = _assign_table_ids(df_cells)
    df_cells = _assign_column_layout(df_cells)
    df_cells = _reconcile_colspan(df_cells)
    df_cells = _assign_row_completeness(df_cells)
    df_cells = _assign_last_tr_below(df_cells)
    df_cells = _assign_is_subheading(df_cells)
    df_cells = _assign_is_numeric_like(df_cells)
    df_cells = _assign_grid_trustworthy(df_cells)
    df_cells = _assign_row_layout(df_cells)
    df_cells = _assign_table_cell_ids(df_cells)

    df_table_cells = _build_table_cells_df(df_cells)

    return df_cells, df_table_cells

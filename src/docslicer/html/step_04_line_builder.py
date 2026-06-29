# step_04_line_merger.py

import re
import pandas as pd

from docslicer._utils.layout.line_merger import assign_line_id, LineMergerConfig
from docslicer._utils.df_aggregation.hierarchical_aggregator import (
    aggregate_hierarchical,
    build_standard_agg_spec,
)


# =================================
# PRIVATE FUNCTIONS
# =================================

def _create_line_text(df: pd.DataFrame) -> dict:
    """
    Create text for each line by sorting boxes by x_left.

    Non-table lines are joined with spaces. Table lines are joined with pipes so
    downstream shared stages see a row-shaped representation.
    
    Returns:
        Dict mapping line_id -> text string
    """
    if "line_id" not in df.columns or "text" not in df.columns:
        return {}

    # Operate on the full DataFrame at once instead of per-group
    cols = ["line_id", "x_left", "text"]
    if "table_id" in df.columns:
        cols.append("table_id")
    working = df[cols].copy()
    working = working[working["text"].notna()]
    working["text"] = working["text"].astype(str).str.strip()
    working = working[working["text"] != ""]

    working = working.sort_values(["line_id", "x_left"])

    if "table_id" not in working.columns:
        return working.groupby("line_id", sort=False)["text"].agg(" ".join).to_dict()

    text_map = {}
    for line_id, group in working.groupby("line_id", sort=False):
        table_id = group["table_id"].iloc[0]
        has_table = pd.notna(table_id) and str(table_id).strip() != ""
        sep = " | " if has_table else " "
        text_map[line_id] = sep.join(group["text"].tolist())
    return text_map


_STARTS_WITH_NUMBER_OR_PARENS = re.compile(r'^(\d|\([a-zA-Z0-9]+\)|[•◦▪▸·‣⁃●○►▶◆◇□■])')


def _remove_single_row_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process single-row tables: merge or remove them.

    Consecutive runs of >=5 single-row tables with equal table_row_cell_count where
    <=10% of rows start with a digit or a parenthesised token like (1)/(i)/(a) are
    merged into one multi-row table with sequential table_row_ids. All other
    single-row tables are removed (table_id/table_row_id cleared).

    Remaining tables are reindexed sequentially and marked block_type="table".
    """
    if df is None or df.empty:
        return df

    if "table_id" not in df.columns or "table_row_id" not in df.columns:
        return df

    df = df.copy()
    df["original_table_id"] = df["table_id"]
    df["original_table_row_id"] = df["table_row_id"]

    # --- 1. Identify single-row tables ---
    tables_with_rows = (
        df[df["table_id"].notna()]
        .groupby("table_id")["table_row_id"]
        .nunique()
    )
    single_row_ids = set(tables_with_rows[tables_with_rows == 1].index)
    is_single = df["table_id"].isin(single_row_ids)

    # --- 2. Build consecutive-run groups (breaks on non-single or cell-count change) ---
    has_cell_count = "table_row_cell_count" in df.columns
    cc_vals = (
        df["table_row_cell_count"].fillna(-1).astype(str).values
        if has_cell_count
        else ["same"] * len(df)
    )
    is_single_arr = is_single.values
    group_arr = [None] * len(df)
    current_group = 0

    for i in range(len(df)):
        if not is_single_arr[i]:
            continue
        if i == 0 or not is_single_arr[i - 1] or cc_vals[i] != cc_vals[i - 1]:
            current_group += 1
        group_arr[i] = current_group

    df["_srg"] = group_arr

    # --- 3. Decide merge vs remove per group ---
    merge_groups: set = set()
    remove_groups: set = set()

    for gid, grp in df[df["_srg"].notna()].groupby("_srg"):
        if len(grp) >= 5:
            texts = grp["text"].fillna("").astype(str)
            n_flagged = texts.apply(
                lambda t: bool(_STARTS_WITH_NUMBER_OR_PARENS.match(t.strip()))
            ).sum()
            if n_flagged / len(texts) <= 0.10:
                merge_groups.add(gid)
                continue
        remove_groups.add(gid)

    # --- 4. Remove ---
    to_remove = df["_srg"].isin(remove_groups)
    df.loc[to_remove, "table_id"] = None
    df.loc[to_remove, "table_row_id"] = None
    if "text" in df.columns:
        df.loc[to_remove, "text"] = df.loc[to_remove, "text"].str.replace(" | ", " ", regex=False)

    # --- 5. Merge ---
    if merge_groups:
        existing_max = df["table_id"].max()
        next_tid = int(existing_max) + 1 if pd.notna(existing_max) else 1

        for gid in sorted(merge_groups):
            mask = df["_srg"] == gid
            indices = df[mask].index
            df.loc[mask, "table_id"] = next_tid
            for row_num, idx in enumerate(indices, start=1):
                df.loc[idx, "table_row_id"] = row_num
            next_tid += 1

    df = df.drop(columns=["_srg"])

    # --- 6. Reindex table_ids by first appearance ---
    if df["table_id"].notna().any():
        seen: dict = {}
        for tid in df.loc[df["table_id"].notna(), "table_id"]:
            if tid not in seen:
                seen[tid] = len(seen) + 1
        df["table_id"] = df["table_id"].map(lambda x: seen.get(x) if pd.notna(x) else None)

    # --- 7. Mark block_type = "table" ---
    if "block_type" not in df.columns:
        df["block_type"] = None
    has_table = df["table_id"].notna()
    no_existing_type = df["block_type"].isna()
    df.loc[has_table & no_existing_type, "block_type"] = "table"

    return df


def _add_layout_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add layout_id column based on page changes, table groups, and line boundaries.
    
    Rules:
    1. Start at 1 for the very first line
    2. Increment when page_number changes
    3. Rows with same table_id get same layout_id (increment at start/end of table groups)
    4. For non-table lines, increment for every new line (each row gets its own layout_id)
    
    Args:
        df: DataFrame with lines
        
    Returns:
        DataFrame with layout_id column added
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    layout_ids = []
    current_layout_id = 1
    prev_page = None
    prev_table_id = None
    
    for _, row in df.iterrows():
        page = row.get("page_number")
        table_id = row.get("table_id")
        
        # Rule 2: Increment when page_number changes
        if prev_page is not None and page != prev_page:
            current_layout_id += 1
            prev_table_id = None
        
        has_table = pd.notna(table_id)
        
        if has_table:
            # Rule 3: Handle table groups
            if table_id != prev_table_id:
                # Entering a new table group
                if prev_page is not None:  # Not the first row
                    current_layout_id += 1
                prev_table_id = table_id
        else:
            # Non-table line
            if prev_table_id is not None:
                # Exiting a table group
                current_layout_id += 1
                prev_table_id = None
            else:
                # Rule 4: Every new non-table line gets a new layout_id
                if prev_page is not None:
                    current_layout_id += 1
        
        layout_ids.append(current_layout_id)
        prev_page = page
    
    df["layout_id"] = layout_ids
    return df


# =========================
# Public API
# =========================
def merge_boxes_to_lines(
    boxes_df: pd.DataFrame,
    remove_single_row_tables: bool = True,
    merge_by_coordinates: bool = True,
) -> pd.DataFrame:
    """
    Merge boxes into lines:
    1. Assign line_id using line_merger
    2. Sort text within each line by x_left
    3. Aggregate using hierarchical_aggregator
    4. Optionally remove single-row tables and reindex
    5. Add layout_id

    Args:
        boxes_df: DataFrame with box-level data
        remove_single_row_tables: If True, remove table_id/table_row_id for tables with only 1 row
        merge_by_coordinates: When False, skip y-tolerance merging and give every non-table box
            its own line. Use for statically extracted boxes where y_top/y_bottom are all 0.
    """
    if boxes_df is None or boxes_df.empty:
        return boxes_df

    # Step 1: Assign line_id
    # When coordinates are absent (static extraction), y_top/y_bottom are all 0, which
    # causes every box to merge into one line via the tolerance check. Fix: synthetically
    # space y values so non-table boxes are always far enough apart to avoid merging.
    # Table cells still merge correctly via table_row_id (checked before coordinates).
    if not merge_by_coordinates:
        # With y_top=0 everywhere, assign_line_id's tolerance check merges everything.
        # Give every logical row a unique y so only table_row_id-based merging applies.
        # - Non-table boxes: unique y per box (each becomes its own line)
        # - Table cells: same y for cells sharing a table_row_id, unique per row group
        #   (coordinate check is skipped because table_row_id match fires first, but
        #    cells from *different* rows would also have dy=0 and spuriously merge)
        _STRIDE = LineMergerConfig().TOL_EXPANDED + 1  # > any merge tolerance
        boxes_df = boxes_df.copy()
        has_row_id = "table_row_id" in boxes_df.columns
        is_table_box = boxes_df["table_row_id"].notna() if has_row_id else pd.Series(False, index=boxes_df.index)

        # Assign a rank to each unique (table_id, table_row_id) pair in document order
        if has_row_id and is_table_box.any():
            row_groups = (
                boxes_df[is_table_box]
                .groupby(["table_id", "table_row_id"], sort=False)
                .ngroup()
            )

        sequential_y = pd.Series(range(len(boxes_df)), index=boxes_df.index, dtype=float) * _STRIDE
        synthetic_y = sequential_y.copy()

        if has_row_id and is_table_box.any():
            # All cells in the same TR share the same y (their row group rank × stride)
            synthetic_y[is_table_box] = row_groups.values * _STRIDE

        boxes_df["y_top"] = synthetic_y
        boxes_df["y_bottom"] = synthetic_y

    boxes_with_lines = assign_line_id(boxes_df, y_alignment="top")
    
    # Step 2: Create text for each line (sorted by x_left, joined with spaces)
    line_text_map = _create_line_text(boxes_with_lines)
    
    # Step 3: Build aggregation spec
    agg_spec = build_standard_agg_spec(
        include_hierarchy=True,
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,
        include_html_provenance=True,
        count_col="box_id",
    )
    
    # Step 4: Aggregate
    lines_df = aggregate_hierarchical(
        df=boxes_with_lines,
        group_col="line_id",
        agg_spec=agg_spec,
        rename_group_col="line_id",
        rename_count_col={"box_id": "box_count"},
        compute_derived=True,
    )
    
    # Step 5: Add the text column
    lines_df["text"] = lines_df["line_id"].map(line_text_map)

    # Step 6: Remove single-row tables if requested
    if remove_single_row_tables:
        lines_df = _remove_single_row_tables(lines_df)

    # Step 7: Add layout_id column
    lines_df = _add_layout_id(lines_df)
    
    return lines_df

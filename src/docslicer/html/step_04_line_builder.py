# step_04_line_merger.py

import pandas as pd

from docslicer._utils.line_merger import assign_line_id, LineMergerConfig
from docslicer._utils.hierarchical_aggregator import (
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


def _remove_single_row_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove table_id and table_row_id for tables with only 1 row, then reindex remaining tables.
    Also adds block_type = "table" for remaining tables (preserving existing values).
    
    Args:
        df: DataFrame with table_id and table_row_id columns
        
    Returns:
        DataFrame with single-row tables removed and remaining tables reindexed
    """
    if df is None or df.empty:
        return df
    
    if "table_id" not in df.columns or "table_row_id" not in df.columns:
        return df
    
    df = df.copy()
    
    # Identify tables with only 1 unique table_row_id
    tables_with_rows = (
        df[df["table_id"].notna()]
        .groupby("table_id")["table_row_id"]
        .nunique()
    )
    single_row_tables = set(tables_with_rows[tables_with_rows == 1].index)
    
    # Remove table_id and table_row_id for single-row tables
    is_single_row_table = df["table_id"].isin(single_row_tables)
    df.loc[is_single_row_table, "table_id"] = None
    df.loc[is_single_row_table, "table_row_id"] = None
    
    # Reindex remaining table_ids (1-based sequential)
    remaining_tables = df[df["table_id"].notna()]["table_id"].unique()
    if len(remaining_tables):
        # Sort to maintain order
        remaining_tables = sorted(remaining_tables)
        table_id_map = {old_id: new_id for new_id, old_id in enumerate(remaining_tables, start=1)}
        df["table_id"] = df["table_id"].map(lambda x: table_id_map.get(x) if pd.notna(x) else None)

    # Add block_type = "table" for remaining tables (preserve existing values)
    if "block_type" not in df.columns:
        df["block_type"] = None
    has_table = df["table_id"].notna()
    no_existing_role = df["block_type"].isna()
    df.loc[has_table & no_existing_role, "block_type"] = "table"
    
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
    """
    if boxes_df is None or boxes_df.empty:
        return boxes_df
    
    # Step 1: Assign line_id
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

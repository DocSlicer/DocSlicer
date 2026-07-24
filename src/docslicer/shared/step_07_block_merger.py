# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
step_07_block_merger.py

Merge lines into logical blocks.

Architecture:
  1. _assign_block_ids()  - block_id = layout_id, split/merged on heading_id boundaries
  2. aggregate_to()       - shared registry-driven aggregation (registry_aggregator)
  3. _join_text()         - Text merging strategy (space vs newline)

Table formatters:  _format_table_markdown / _format_table_jsonl / _format_table_melted
Chart formatters:  _format_chart_markdown / _format_chart_melted / _format_chart_jsonl
"""

from __future__ import annotations

import json

import pandas as pd

from .._utils.df_aggregation.registry_aggregator import aggregate_to
from .._utils.df_aggregation.text_merge import join_lines
from .._utils.text_utils import bullet_line_mask

# =======================================================================================================================
# STEP 1: ASSIGN BLOCK IDs
# =======================================================================================================================

# =================================
# Block ID Assignment
# =================================

def _assign_block_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign _block_id to each line: one block per layout_id, with heading-aware
    corrections.

    Every upstream pipeline assigns layout_id such that each layout usually
    corresponds to exactly one logical block (headings, paragraphs, columns,
    tables, etc. each get their own layout), so the block defaults to the layout.
    Two heading-driven corrections override that default:
      * distinct headings sharing one layout are split apart (heading_id change), and
      * one multi-row heading spanning several layouts is merged (heading_id equal).

    Lines are sorted into document order (layout_id, line_id) so the downstream
    text join sees them in reading order.

    Args:
        df: Lines dataframe (must carry "layout_id" and "line_id")

    Returns:
        Same df with "_block_id" column added (== layout_id)
    """
    df = df.sort_values(["layout_id", "line_id"], kind="mergesort").reset_index(drop=True)

    # A layout is normally one block, so a layout boundary starts a new block.
    prev_layout = df["layout_id"].shift(1)
    new_block = df["layout_id"].ne(prev_layout)

    # Two exceptions, both driven by heading_id:
    #
    #   * SPLIT WITHIN a layout: distinct numbered headings can share a single
    #     layout (e.g. "1.1 Purpose …" and "1.2 Applicable References:" flowed
    #     into one layout). They must NOT collapse into one block/HierarchyNode,
    #     so force a new block whenever the (non-null) heading_id changes between
    #     consecutive lines.
    #   * MERGE ACROSS layouts: one multi-row heading can span several layouts
    #     while sharing one heading_id (e.g. "TITLE I" + "SUBJECT MATTER AND
    #     SCOPE" on consecutive lines). Those rows must land in the SAME block,
    #     so suppress the layout-boundary split when the heading_id is unchanged.
    if "heading_id" in df.columns and len(df) > 1:
        hid = df["heading_id"]
        prev_hid = hid.shift(1)
        both_known = hid.notna() & prev_hid.notna()
        same_heading = both_known & hid.eq(prev_hid)
        diff_heading = both_known & hid.ne(prev_hid)
        new_block = (new_block & ~same_heading) | diff_heading

    if len(df):
        new_block.iloc[0] = True

    df["_block_id"] = new_block.cumsum().astype(int)

    return df


# =======================================================================================================================
# TABLE BLOCK GENERATORS
# =======================================================================================================================

# =================================
# Markdown Format
# =================================

def _format_table_markdown(table_df: pd.DataFrame) -> str:
    """
    Format table as markdown with pipes.
    
    Rules:
      - Separate columns with pipes
      - Draw header line underneath last row with role="header"
      - Empty columns: add blank cells
      - Col/row spans: duplicate values
    
    Args:
        table_df: Table cells (must have col_start, row_start, text, role, colspan, rowspan)

    Returns:
        Markdown formatted table string
    """
    # Ensure required columns exist
    if "row_start" not in table_df.columns:
        # Try to infer from temp_line_ids
        if "temp_line_ids" in table_df.columns:
            table_df["row_start"] = table_df["temp_line_ids"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
            )
        else:
            return "[Table: missing row information]"
    
    # Get grid dimensions
    max_row = int(table_df["row_start"].max())
    max_col = int(table_df["col_start"].max())
    
    # Add colspan/rowspan handling
    if "colspan" not in table_df.columns:
        table_df["colspan"] = 1
    if "rowspan" not in table_df.columns:
        table_df["rowspan"] = 1
    
    # Build grid: (row, col) -> cell text
    grid = {}
    last_header_row = -1

    # Pull columns as arrays once — iterrows() rebuilds a Series per row, which
    # dominated this function's runtime on wide tables (see _format_table_jsonl,
    # which uses the same to_numpy + zip pattern).
    rows_arr = table_df["row_start"].to_numpy(dtype=int)
    cols_arr = table_df["col_start"].to_numpy(dtype=int)
    texts_arr = table_df["text"].fillna("").astype(str).str.strip().to_numpy()
    colspans_arr = table_df["colspan"].fillna(1).to_numpy(dtype=int)
    rowspans_arr = table_df["rowspan"].fillna(1).to_numpy(dtype=int)
    if "table_cell_role" in table_df.columns:
        roles_arr = table_df["table_cell_role"].fillna("").to_numpy()
    else:
        roles_arr = [""] * len(table_df)

    for row, col, text, colspan, rowspan, role in zip(
        rows_arr, cols_arr, texts_arr, colspans_arr, rowspans_arr, roles_arr
    ):
        row = int(row)
        col = int(col)

        # Track last header row
        if role == "header":
            last_header_row = max(last_header_row, row)

        # Fill spans by duplicating value
        for r in range(row, row + int(rowspan)):
            for c in range(col, col + int(colspan)):
                grid[(r, c)] = text
    
    # Build markdown lines
    lines = []
    rows = sorted(set(r for r, c in grid.keys()))
    
    for row_idx, row in enumerate(rows):
        cols = []
        for col in range(0, max_col + 1):
            cell_text = grid.get((row, col), "")  # Empty cell if missing
            cols.append(cell_text)
        
        lines.append("| " + " | ".join(cols) + " |")
        
        # Add header separator after last header row
        if row == last_header_row and last_header_row >= 0:
            separator = "|" + "|".join(["---"] * (max_col + 1)) + "|"
            lines.append(separator)
    
    return "\n".join(lines)


# =================================
# JSONL Format
# =================================

def _format_table_jsonl(table_df: pd.DataFrame) -> str:
    """
    Format table as JSONL with one line per data row.

    Each row is a JSON object with headers as keys.
    Handles colspan: headers with colspan > 1 apply to multiple value columns

    Args:
        table_df: Table cells

    Returns:
        JSONL formatted string (one JSON object per line)
    """
    if "colspan" not in table_df.columns:
        table_df = table_df.copy()
        table_df["colspan"] = 1

    headers_df = table_df[table_df["table_cell_role"] == "header"]
    data_df = table_df[table_df["table_cell_role"] != "header"].copy()

    if headers_df.empty:
        return "[Table: no headers found]"

    # Build header keys using numpy arrays (avoids iterrows overhead)
    header_rows_sorted = sorted(headers_df["row_start"].unique())
    col_starts = headers_df["col_start"].to_numpy(dtype=int)
    colspans = headers_df["colspan"].to_numpy(dtype=int)
    rows_arr = headers_df["row_start"].to_numpy(dtype=int)
    texts_arr = headers_df["text"].fillna("").astype(str).str.strip().to_numpy()

    header_map: dict[int, dict[int, str]] = {}
    for col_start, colspan, row, text in zip(col_starts, colspans, rows_arr, texts_arr):
        for offset in range(colspan):
            col = col_start + offset
            if col not in header_map:
                header_map[col] = {}
            header_map[col][row] = text

    header_keys: dict[int, str] = {}
    for col, row_texts in header_map.items():
        parts = [row_texts[r] for r in header_rows_sorted if r in row_texts and row_texts[r]]
        header_keys[col] = "_".join(parts) if parts else f"col_{col}"

    # Pre-compute stripped text once
    data_df["_text"] = data_df["text"].fillna("").astype(str).str.strip()

    json_lines = []

    # groupby replaces repeated boolean filtering (data_df[data_df["row_start"] == row])
    for row, row_data in data_df.groupby("row_start", sort=True):
        value_cells = row_data[row_data["table_cell_role"] == "data"]

        if value_cells["_text"].eq("").all():
            label_cells = row_data[row_data["col_start"] == 0]
            label = label_cells.iloc[0]["_text"] if not label_cells.empty else ""
            if label:
                json_lines.append(json.dumps({"Metric": label}, ensure_ascii=False))
            continue

        row_obj = {}
        cols_arr = row_data["col_start"].to_numpy(dtype=int)
        vals_arr = row_data["_text"].to_numpy()
        for col, value in zip(cols_arr, vals_arr):
            header = header_keys.get(col, f"col_{col}")
            if col == 0 and header == "col_0":
                header = "Metric"
            row_obj[header] = value

        json_lines.append(json.dumps(row_obj, ensure_ascii=False))

    return "\n".join(json_lines)


# =================================
# Melted Format
# =================================

def _format_table_melted(table_df: pd.DataFrame) -> str:
    """
    Format table as fully melted with one fact per row.

    Each fact is represented as: row_label | header_path | value
    Handles colspan: headers with colspan > 1 apply to multiple value columns

    Args:
        table_df: Table cells

    Returns:
        Melted format string
    """
    if "colspan" not in table_df.columns:
        table_df = table_df.copy()
        table_df["colspan"] = 1

    headers_df = table_df[table_df["table_cell_role"] == "header"]
    data_df = table_df[table_df["table_cell_role"] != "header"].copy()

    # Build header paths using numpy arrays (avoids iterrows overhead)
    header_rows_sorted = sorted(headers_df["row_start"].unique())
    col_starts = headers_df["col_start"].to_numpy(dtype=int)
    colspans = headers_df["colspan"].to_numpy(dtype=int)
    rows_arr = headers_df["row_start"].to_numpy(dtype=int)
    texts_arr = headers_df["text"].fillna("").astype(str).str.strip().to_numpy()

    header_map: dict[int, dict[int, str]] = {}
    for col_start, colspan, row, text in zip(col_starts, colspans, rows_arr, texts_arr):
        for offset in range(colspan):
            col = col_start + offset
            if col not in header_map:
                header_map[col] = {}
            header_map[col][row] = text

    header_paths: dict[int, str] = {}
    for col, row_texts in header_map.items():
        parts = [row_texts[r] for r in header_rows_sorted if r in row_texts and row_texts[r]]
        header_paths[col] = " > ".join(parts) if parts else f"col_{col}"

    # Pre-compute stripped text once to avoid repeated per-cell string ops
    data_df["_text"] = data_df["text"].fillna("").astype(str).str.strip()

    melted_lines = []

    # groupby replaces repeated boolean filtering (data_df[data_df["row_start"] == row])
    for row, row_data in data_df.groupby("row_start", sort=True):
        row_label_rows = row_data[row_data["table_cell_role"] == "row_label"]
        row_label = row_label_rows.iloc[0]["_text"] if not row_label_rows.empty else f"row_{row}"

        value_cells = row_data[row_data["table_cell_role"] == "data"]

        if value_cells["_text"].eq("").all():
            if row_label.strip():
                melted_lines.append(row_label)
            continue

        cols_arr = value_cells["col_start"].to_numpy(dtype=int)
        vals_arr = value_cells["_text"].to_numpy()
        for col, value in zip(cols_arr, vals_arr):
            header = header_paths.get(col, f"col_{col}")
            melted_lines.append(f"{row_label} | {header} | {value}")

    return "\n".join(melted_lines)


# =================================
# Main Table Block Orchestrator
# =================================

def _generate_table_block(
    table_id: str,
    df_lines: pd.DataFrame,
    table_df: pd.DataFrame,
    representation: str = "markdown",
) -> str:
    """
    Generate appropriate representation of a table block.

    Formats:
      - "markdown": Pipe-separated markdown table
      - "jsonl": One JSON line per row with headers
      - "melted": One fact per row (fully melted)

    Args:
        table_id: Unique identifier for the table (only used for the fallback)
        df_lines: Lines belonging to this table block
        table_df: Table cells ALREADY scoped to this table (the caller pre-slices
            table_cells_df by table_id once — see _join_text). None/empty falls
            back to a plain line join.
        representation: Format to use for table output

    Returns:
        Formatted table text
    """
    if table_df is None or table_df.empty:
        return join_lines(df_lines["text"])

    # The formatters add/mutate columns in place — copy once so the shared slice
    # held in _join_text's group dict is never mutated.
    table_df = table_df.copy()

    if representation == "jsonl":
        return _format_table_jsonl(table_df)
    elif representation == "melted":
        return _format_table_melted(table_df)
    else:
        return _format_table_markdown(table_df)


# =======================================================================================================================
# CHART FORMATTERS
# =======================================================================================================================

def _chart_display_value(row: "pd.Series") -> str:
    """Return the display value string from a chart data row, or empty string if missing/NaN."""
    value = row.get("value", "")
    return str(value).strip() if value is not None and str(value).strip() not in ("", "nan") else ""


def _format_chart_markdown(chart_df: pd.DataFrame) -> str:
    """
    Format chart data as a markdown table with categories as rows and series as columns.

    Args:
        chart_df: Chart points dataframe for a single chart

    Returns:
        Markdown formatted table string
    """
    chart_type = str(chart_df["chart_type"].iloc[0]).strip() if "chart_type" in chart_df.columns else ""
    is_pie = chart_type in {"doughnutChart", "pieChart"}

    sorted_df = chart_df.sort_values(["series_index", "point_index"])
    series_names = [s for s in dict.fromkeys(sorted_df["series_name"].fillna("").astype(str).str.strip()) if s]
    categories = [c for c in dict.fromkeys(sorted_df["category"].fillna("").astype(str).str.strip()) if c]

    # Build value lookup: (series, category) -> display value
    lookup: dict[tuple[str, str], str] = {}
    for _, row in sorted_df.iterrows():
        s = str(row.get("series_name", "")).strip()
        c = str(row.get("category", "")).strip()
        v = _chart_display_value(row)
        if is_pie:
            pct = row.get("percent", "")
            pct_str = str(pct).strip() if pct is not None and str(pct).strip() not in ("", "nan") else ""
            if pct_str and pct_str != v:
                v = f"{v} ({pct_str})"
        lookup[(s, c)] = v

    header = "| Category | " + " | ".join(series_names) + " |"
    separator = "| --- |" + " --- |" * len(series_names)
    rows = []
    for cat in categories:
        cells = [lookup.get((s, cat), "") for s in series_names]
        rows.append(f"| {cat} | " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


def _format_chart_melted(chart_df: pd.DataFrame) -> str:
    """
    Format chart data as melted rows: one "series | category | value" line per data point.

    Args:
        chart_df: Chart points dataframe for a single chart

    Returns:
        Melted format string
    """
    chart_type = str(chart_df["chart_type"].iloc[0]).strip() if "chart_type" in chart_df.columns else ""
    is_pie = chart_type in {"doughnutChart", "pieChart"}

    lines = []
    for _, row in chart_df.sort_values(["series_index", "point_index"]).iterrows():
        series = str(row.get("series_name", "")).strip()
        category = str(row.get("category", "")).strip()
        value = _chart_display_value(row)

        if is_pie:
            pct = row.get("percent", "")
            pct_str = str(pct).strip() if pct is not None and str(pct).strip() not in ("", "nan") else ""
            if pct_str and pct_str != value:
                value = f"{value} ({pct_str})"

        lines.append(f"{series} | {category} | {value}")

    return "\n".join(lines)


def _format_chart_jsonl(chart_df: pd.DataFrame) -> str:
    """
    Format chart data as JSONL with one JSON object per series, keyed by category.

    Args:
        chart_df: Chart points dataframe for a single chart

    Returns:
        JSONL formatted string (one JSON object per line)
    """
    chart_type = str(chart_df["chart_type"].iloc[0]).strip() if "chart_type" in chart_df.columns else ""
    is_pie = chart_type in {"doughnutChart", "pieChart"}

    json_lines = []
    sorted_df = chart_df.copy()
    sorted_df["series_name"] = sorted_df["series_name"].fillna("").astype(str).str.strip()
    sorted_df = sorted_df.sort_values(["series_index", "point_index"])
    for series_name, group in sorted_df.groupby("series_name", sort=False):
        row_obj = {"Series": str(series_name).strip()}
        for _, row in group.iterrows():
            category = str(row.get("category", "")).strip()
            value = _chart_display_value(row)
            if is_pie:
                pct = row.get("percent", "")
                pct_str = str(pct).strip() if pct is not None and str(pct).strip() not in ("", "nan") else ""
                if pct_str and pct_str != value:
                    value = f"{value} ({pct_str})"
            row_obj[category] = value
        json_lines.append(json.dumps(row_obj, ensure_ascii=False))

    return "\n".join(json_lines)


def _generate_chart_block(
    chart_id: str,
    df_lines: pd.DataFrame,
    chart_points_df: pd.DataFrame,
    representation: str = "jsonl",
) -> str:
    """
    Generate appropriate representation of a chart block.

    Formats:
      - "markdown": Category-vs-series markdown table
      - "melted": One fact per row (series | category | value)
      - "jsonl": One JSON object per series (default)

    Args:
        chart_id: Unique identifier for the chart
        df_lines: Lines belonging to this chart block
        chart_points_df: Full chart points dataframe (filtered to this chart)
        representation: Format to use for chart output

    Returns:
        Formatted chart text
    """
    if chart_points_df is None or chart_points_df.empty:
        return join_lines(df_lines["text"])

    chart_df = chart_points_df[chart_points_df["chart_id"] == chart_id].copy()
    if chart_df.empty:
        return join_lines(df_lines["text"])

    parts = []

    # Chart title if present
    if "chart_title" in chart_df.columns:
        raw = chart_df["chart_title"].iloc[0]
        title = str(raw).strip() if raw is not None else ""
        if title and title.lower() not in ("nan", "none", ""):
            parts.append(title)

    if representation == "markdown":
        parts.append(_format_chart_markdown(chart_df))
    elif representation == "melted":
        parts.append(_format_chart_melted(chart_df))
    else:
        parts.append(_format_chart_jsonl(chart_df))

    return "\n".join(parts)


# =======================================================================================================================
# STEP 2: TEXT JOINER
# =======================================================================================================================

# =================================
# Helper Functions
# =================================

def _compute_line_separator(df_with_block_ids: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the join separator for each line based on content and context.
    
    Note: Lines with table_id are skipped as they use table rendering logic.
    
    Separator rules (applied per line):
      1. If block_type is "toc" or "exhibits" (and no table_id): newline
      2. If text starts with a bullet token: newline
      3. If hierarchy_marker is non-blank: newline
      4. Default: space
    
    Args:
        df_with_block_ids: Lines df with _block_id column
    
    Returns:
        Same df with "_join_sep" column added
    """
    # Initialize with default separator (space)
    df_with_block_ids["_join_sep"] = " "

    # Skip lines that belong to tables (handled by table rendering)
    has_table_id = pd.Series(False, index=df_with_block_ids.index)
    if "table_id" in df_with_block_ids.columns:
        has_table_id = df_with_block_ids["table_id"].notna()

    # Rule 1: TOC and exhibits always use newlines (if not part of a table)
    if "block_type" in df_with_block_ids.columns:
        is_toc_or_exhibits = df_with_block_ids["block_type"].isin(["toc", "exhibits"])
        df_with_block_ids.loc[is_toc_or_exhibits & ~has_table_id, "_join_sep"] = "\n"

    # Rule 2: Lines starting with bullet tokens use newlines (if not part of a table)
    is_bullet_start = bullet_line_mask(df_with_block_ids["text"])
    df_with_block_ids.loc[is_bullet_start & ~has_table_id, "_join_sep"] = "\n"

    # Rule 3: Lines with hierarchy markers use newlines (if not part of a table)
    if "hierarchy_marker" in df_with_block_ids.columns:
        has_hm = df_with_block_ids["hierarchy_marker"].fillna("").astype(str).str.strip().ne("")
        df_with_block_ids.loc[has_hm & ~has_table_id, "_join_sep"] = "\n"
    
    return df_with_block_ids


# =================================
# Main Text Joiner
# =================================

def _join_text(
    df_with_block_ids: pd.DataFrame,
    blocks_df: pd.DataFrame,
    table_cells_df: pd.DataFrame = None,
    table_representation: str = "markdown",
    chart_points_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Join line texts into block text with strategy-specific formatting.

    Text joining strategies:
      1. Blocks with table_id (table, toc, exhibits): use table rendering
      2. Blocks with chart_id: use chart rendering
      3. toc/exhibits without table_id, bullet lines, hierarchy markers: join with newlines
      4. Default: join with spaces

    Args:
        df_with_block_ids: Lines df with _block_id column
        blocks_df: Aggregated blocks df (without text column)
        table_cells_df: Full table cells dataframe (for table formatting)
        table_representation: Format for table/chart output ("markdown", "jsonl", "melted")
        chart_points_df: Chart points dataframe (for chart formatting)

    Returns:
        blocks_df with "text" column added
    """
    # -------------------------------------------------------------------------
    # STEP 1: COMPUTE LINE SEPARATORS
    # -------------------------------------------------------------------------
    df_with_block_ids = _compute_line_separator(df_with_block_ids)

    # -------------------------------------------------------------------------
    # STEP 2: IDENTIFY SPECIAL BLOCKS (table / chart rendering)
    # Use the first row of each block to determine its type and IDs.
    # -------------------------------------------------------------------------
    first = df_with_block_ids.groupby("_block_id", sort=False).first()

    _bt  = first["block_type"].fillna("") if "block_type" in first.columns else pd.Series("", index=first.index)
    _has_table = first["table_id"].notna() if "table_id" in first.columns else pd.Series(False, index=first.index)
    _has_chart = first["chart_id"].notna() if "chart_id" in first.columns else pd.Series(False, index=first.index)

    is_special_block = (
        (_bt == "table") |
        (_bt.isin(["toc", "exhibits"]) & _has_table) |
        (_bt == "chart") |
        _has_chart
    )
    special_block_ids = set(first.index[is_special_block])

    # -------------------------------------------------------------------------
    # STEP 3: VECTORIZED TEXT JOINING FOR REGULAR BLOCKS
    # Avoids per-group Python calls entirely for blocks that just join text.
    # -------------------------------------------------------------------------
    is_text_line = ~df_with_block_ids["_block_id"].isin(special_block_ids)
    text_lines = df_with_block_ids[is_text_line]

    result_parts: list[pd.DataFrame] = []

    if not text_lines.empty:
        text_vals = text_lines["text"].fillna("").astype(str).str.strip()
        is_nonempty = text_vals.ne("")

        nonempty = text_lines[is_nonempty].copy()
        nonempty["_text"] = text_vals[is_nonempty].values
        # Rank non-empty lines within each block (0 = first line, gets no sep prefix)
        nonempty["_rank"] = nonempty.groupby("_block_id", sort=False).cumcount()
        nonempty["_part"] = nonempty["_join_sep"].fillna(" ").astype(str) + nonempty["_text"]
        rank0 = nonempty["_rank"] == 0
        nonempty.loc[rank0, "_part"] = nonempty.loc[rank0, "_text"]

        joined = (
            nonempty.groupby("_block_id", sort=False)["_part"]
            .apply("".join)
            .reset_index()
            .rename(columns={"_block_id": "block_id", "_part": "text"})
        )
        result_parts.append(joined)

        # Blocks where every line was empty → emit empty string
        missing_ids = set(text_lines["_block_id"].unique()) - set(joined["block_id"])
        if missing_ids:
            result_parts.append(pd.DataFrame({"block_id": list(missing_ids), "text": ""}))

    # -------------------------------------------------------------------------
    # STEP 4: groupby.apply FOR TABLE / CHART BLOCKS (small subset)
    # -------------------------------------------------------------------------
    special_lines = df_with_block_ids[~is_text_line]

    if not special_lines.empty:
        # Pre-slice the cells / points by their id ONCE. Without this each block
        # re-scanned the full table_cells_df (O(n_blocks × all_cells)) — the
        # bottleneck on table-heavy docs (hundreds of tables). A single groupby
        # turns each per-block lookup into an O(1) dict fetch of its own slice.
        table_groups: dict = {}
        if table_cells_df is not None and not table_cells_df.empty and "table_id" in table_cells_df.columns:
            table_groups = dict(tuple(table_cells_df.groupby("table_id", sort=False)))
        chart_groups: dict = {}
        if chart_points_df is not None and not chart_points_df.empty and "chart_id" in chart_points_df.columns:
            chart_groups = dict(tuple(chart_points_df.groupby("chart_id", sort=False)))

        def _build_special_text(lines: pd.DataFrame) -> str:
            if lines.empty:
                return ""
            block_type = lines["block_type"].iloc[0] if "block_type" in lines.columns else None
            table_id  = lines["table_id"].iloc[0]  if "table_id"  in lines.columns else None
            chart_id  = lines["chart_id"].iloc[0]  if "chart_id"  in lines.columns else None
            has_table = table_id is not None and str(table_id).strip() not in ("", "nan", "None")
            has_chart = chart_id is not None and str(chart_id).strip() not in ("", "nan", "None")

            if block_type == "table" or (block_type in ["toc", "exhibits"] and has_table):
                # Pass this table's pre-sliced cells; a "table" block with no
                # table_id keeps the legacy whole-frame fallback.
                table_df = table_groups.get(table_id) if has_table else table_cells_df
                return _generate_table_block(table_id, lines, table_df, table_representation)
            if block_type == "chart" or has_chart:
                return _generate_chart_block(chart_id, lines, chart_groups.get(chart_id), table_representation)
            return join_lines(lines["text"])

        special_joined = (
            special_lines.groupby("_block_id", sort=False, observed=True)
            .apply(_build_special_text, include_groups=False)
            .reset_index()
            .rename(columns={"_block_id": "block_id", 0: "text"})
        )
        result_parts.append(special_joined)

    # -------------------------------------------------------------------------
    # STEP 5: MERGE TEXT INTO BLOCKS
    # -------------------------------------------------------------------------
    all_results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame(columns=["block_id", "text"])
    blocks_df = blocks_df.merge(all_results, on="block_id", how="left")

    return blocks_df


# =======================================================================================================================
# Public API
# =======================================================================================================================

def merge_blocks(
    lines_df: pd.DataFrame,
    table_cells_df: pd.DataFrame = None,
    chart_points_df: pd.DataFrame = None,
    table_representation: str = "markdown",
) -> pd.DataFrame:
    """
    Merge lines into logical blocks.

    Process:
      1. Validate inputs and prepare for aggregation
      2. Assign block IDs (decision logic)
      3. Aggregate to block level (shared boilerplate)
      4. Join text with smart separators (tables/charts use table_representation)
      5. Compute embed_char_count (includes whitespace for embedding estimation)
      6. Sort by document order

    Args:
        lines_df: Lines-level DataFrame
        table_cells_df: Table cells DataFrame (for table formatting)
        table_representation: Format for table/chart output ("markdown", "jsonl", "melted")
        chart_points_df: Chart points DataFrame (for chart formatting)

    Returns:
        Blocks-level DataFrame with embed_char_count column added
    """
    # -------------------------
    # STEP 1: VALIDATE
    # -------------------------
    if lines_df is None or lines_df.empty:
        return pd.DataFrame()

    # -------------------------
    # STEP 2: ASSIGN BLOCK IDs
    # -------------------------
    df = _assign_block_ids(lines_df.copy())

    # -------------------------
    # STEP 3: AGGREGATE
    # -------------------------
    # Registry-driven: each column's roll-up rule lives in COLUMN_REGISTRY, so
    # the block level picks up new line columns automatically. layout_id (== the
    # group key) is preserved by the registry's "first" rule for the STEP 6 sort.
    blocks_df = aggregate_to(
        df,
        by="_block_id",
        rename_by="block_id",
        derived=True,
    )

    # -------------------------
    # STEP 4: JOIN TEXT
    # -------------------------
    blocks_df = _join_text(df, blocks_df, table_cells_df, table_representation, chart_points_df)
    
    # -------------------------
    # STEP 5: COMPUTE EMBED CHAR COUNT
    # -------------------------
    # embed_char_count includes all characters (including whitespace) for embedding estimation
    # This differs from char_count which only counts net characters
    if "text" in blocks_df.columns:
        blocks_df["embed_char_count"] = blocks_df["text"].fillna("").astype(str).str.len()
    else:
        blocks_df["embed_char_count"] = 0
    
    # -------------------------
    # STEP 6: SORT
    # -------------------------
    blocks_df = blocks_df.sort_values(
        ["layout_id", "block_id"], 
        kind="mergesort"
    ).reset_index(drop=True)
    
    return blocks_df
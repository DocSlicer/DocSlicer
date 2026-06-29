"""
step_07_block_merger.py

Merge lines into logical blocks.

Architecture:
  1. _assign_block_ids()      - Decision logic: what constitutes a new block?
  2. hierarchical_aggregator  - Shared aggregation boilerplate
  3. _join_text()             - Text merging strategy (space vs newline)

Table formatters:  _format_table_markdown / _format_table_jsonl / _format_table_melted
Chart formatters:  _format_chart_markdown / _format_chart_melted / _format_chart_jsonl
"""

from __future__ import annotations

import json

import pandas as pd

from .._utils.df_aggregation.hierarchical_aggregator import (
    build_standard_agg_spec,
    aggregate_hierarchical,
)

# =======================================================================================================================
# STEP 1: ASSIGN BLOCK IDs (DECISION LOGIC)
# =======================================================================================================================

# =================================
# Config
# =================================

_LONG_LAYOUT_MIN_LINES = 10  # Min lines per (layout_id, col_start) to consider indent splitting
_INDENT_INCREASE_PTS = 10.0  # Min x_left increase (points) to trigger indent split

_BULLET_TOKENS = {
    "•", "", "·", "∙", "◦", "▪", "–", "-", "—", "*", "●", "○", "◆", "■", "►", "➤", "➢", "‣", "⁃",
}

# =================================
# Main Block Decision Engine
# =================================

def _assign_block_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign _block_id to each line based on block splitting rules.
    
    Block splitting strategy:
      1. ALWAYS split when:
         - block_type changes (unless both rows share the same non-null heading_id)
         - layout_id changes
         - page_number changes (safety)
         - col_start changes (text_multicol only - each column becomes separate block)
         - heading_id changes (for consecutive heading lines - different headings become separate blocks)
         NOTE: rows sharing the same heading_id are never split by block_type or style changes.
      
      2. CONDITIONALLY split for paragraph blocks in text layouts:
         a) Style property changes WITHIN same layout_id:
            - Applies to: block_type == "paragraph" in text_singlecol/text_multicol layouts
            - Properties tracked: font_size_ratio, non_stroking_color, is_bold, is_italic
            - Trigger: Any style property changes
         
         b) Indentation increases:
            - Applies to: block_type == "paragraph" in text_singlecol/text_multicol layouts
            - Group by: layout_id + col_start (each column treated separately)
            - Eligibility: group must have >10 lines
            - Trigger: x_left increases by ≥10pt (ONLY increases, not decreases)
            - Behavior: TRY to split (only if indent increases exist)
    
    Args:
        df: Lines dataframe
    
    Returns:
        Same df with "_block_id" column added
    """
    # -------------------------------------------------------------------------
    # 1: SORT BY DOCUMENT ORDER
    # -------------------------------------------------------------------------
    df = df.sort_values(["layout_id", "line_id"], kind="mergesort").reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # 2: IDENTIFY INDENT-SPLIT ELIGIBLE LINES
    # -------------------------------------------------------------------------
    # Only paragraph blocks in text_singlecol/text_multicol layouts can be split by indentation.
    # Group by (layout_id, col_start) to treat each column separately.
    # Only consider groups with >10 lines.
    
    # Initialize: no lines eligible for indent splitting
    eligible_for_indent_split = pd.Series(False, index=df.index)
    
    # Check if required columns exist
    if "layout_type" in df.columns and "col_start" in df.columns and "block_type" in df.columns:
        # Filter to text layouts with paragraph role only
        is_text_layout = df["layout_type"].isin(["text_singlecol", "text_multicol"])
        is_paragraph = df["block_type"] == "paragraph"
        
        if is_text_layout.any() and is_paragraph.any():
            # Create a grouping key: layout_id + col_start
            # This ensures we count lines per column within each layout
            df["_layout_col_group"] = (
                df["layout_id"].astype(str) + "|" + 
                df["col_start"].astype(str)
            )
            
            # Count lines per (layout_id, col_start) group
            group_line_counts = df.groupby("_layout_col_group", sort=False)["line_id"].transform("size")
            is_long_group = group_line_counts > _LONG_LAYOUT_MIN_LINES
            
            # Lines are eligible if: text layout + paragraph role + long group
            eligible_for_indent_split = is_text_layout & is_paragraph & is_long_group
            
            # Clean up temporary column
            df = df.drop(columns=["_layout_col_group"])
    
    # -------------------------------------------------------------------------
    # 3: DETECT INDENTATION INCREASES
    # -------------------------------------------------------------------------
    # For eligible lines, detect when x_left increases by ≥10pt from previous line
    # (within same layout_id + col_start)
    
    indent_increase = pd.Series(False, index=df.index)
    
    if eligible_for_indent_split.any():
        prev_layout = df["layout_id"].shift(1)
        prev_col = df["col_start"].shift(1) if "col_start" in df.columns else pd.Series(0, index=df.index)
        prev_x = df["x_left"].shift(1)
        
        # Check if current line is in same (layout_id, col_start) group as previous
        same_group = (
            prev_layout.eq(df["layout_id"]) & 
            prev_col.eq(df["col_start"])
        )
        
        # Calculate x_left delta
        x_delta = df["x_left"] - prev_x
        
        # Indent increase = same group + eligible + x_left increases by ≥10pt
        indent_increase = (
            same_group & 
            eligible_for_indent_split & 
            x_delta.ge(_INDENT_INCREASE_PTS)
        )
    
    # -------------------------------------------------------------------------
    # 4: COMPUTE NEW BLOCK TRIGGERS
    # -------------------------------------------------------------------------
    # Combine all conditions that trigger a new block
    
    prev_block_type = df["block_type"].shift(1)
    prev_page = df["page_number"].shift(1)
    prev_layout = df["layout_id"].shift(1)
    
    # For text_multicol: also split when col_start changes (column switch)
    col_start_change = pd.Series(False, index=df.index)
    if "layout_type" in df.columns and "col_start" in df.columns:
        is_multicol = df["layout_type"] == "text_multicol"
        prev_col = df["col_start"].shift(1)
        same_layout = prev_layout.eq(df["layout_id"])
        
        # Split when col_start changes within same layout
        col_start_change = is_multicol & same_layout & df["col_start"].ne(prev_col)
    
    # Within same layout: split when style fingerprint changes (only for paragraph blocks in text layouts)
    # Create a fingerprint from: font_size_ratio, non_stroking_color, is_bold, is_italic
    # Only applies to paragraph blocks in text_singlecol/text_multicol layouts for safety
    style_change = pd.Series(False, index=df.index)
    same_layout = prev_layout.eq(df["layout_id"])
    
    # Only apply to paragraph blocks in text layouts
    is_eligible = pd.Series(False, index=df.index)
    if "block_type" in df.columns and "layout_type" in df.columns:
        is_paragraph = df["block_type"] == "paragraph"
        is_text_layout = df["layout_type"].isin(["text_singlecol", "text_multicol"])
        is_eligible = is_paragraph & is_text_layout
    
    if same_layout.any() and is_eligible.any():
        # Build style fingerprint tuple for each line
        style_cols = ["font_size_ratio", "non_stroking_color", "is_bold", "is_italic"]
        available_style_cols = [col for col in style_cols if col in df.columns]
        
        if available_style_cols:
            # Create fingerprint as tuple of values
            df["_style_fp"] = df[available_style_cols].apply(tuple, axis=1)
            prev_style_fp = df["_style_fp"].shift(1)
            
            # Split when fingerprint changes within same layout (only for eligible lines)
            style_change = same_layout & is_eligible & df["_style_fp"].ne(prev_style_fp)
            
            # Clean up temporary column
            df = df.drop(columns=["_style_fp"])
    
    # Split when heading_id changes between consecutive heading lines
    heading_id_change = pd.Series(False, index=df.index)
    if "heading_id" in df.columns and "block_type" in df.columns:
        is_heading = df["block_type"] == "heading"
        prev_is_heading = prev_block_type == "heading"
        prev_heading_id = df["heading_id"].shift(1)
        
        # Split when both current and previous are headings, but heading_id changes
        heading_id_change = (
            is_heading & 
            prev_is_heading & 
            df["heading_id"].ne(prev_heading_id)
        )
    
    # Rows that share the same non-null heading_id must stay in the same block —
    # suppress block_type and style splits for them (page/layout boundaries still apply)
    same_heading_group = pd.Series(False, index=df.index)
    if "heading_id" in df.columns:
        prev_heading_id_gen = df["heading_id"].shift(1)
        hid_notna = df["heading_id"].notna() & prev_heading_id_gen.notna()
        same_heading_group = hid_notna & df["heading_id"].eq(prev_heading_id_gen)

    is_new_block = (
        (df["block_type"].ne(prev_block_type) |  # Role change
        df["layout_id"].ne(prev_layout) |        # Layout change
        df["page_number"].ne(prev_page) |        # Page change
        col_start_change |                        # Column change (text_multicol only)
        indent_increase |                         # Indentation increase (conditional)
        style_change |                            # Style property change (within same layout)
        heading_id_change)                        # Heading ID change (consecutive headings only)
        & ~same_heading_group                     # Never split rows that share the same heading_id
    )
    
    # First line always starts a new block
    is_new_block.iloc[0] = True
    
    # -------------------------------------------------------------------------
    # 5: ASSIGN SEQUENTIAL BLOCK IDs
    # -------------------------------------------------------------------------
    df["_block_id"] = is_new_block.cumsum().astype(int)
    
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
    
    for _, cell in table_df.iterrows():
        row = int(cell["row_start"])
        col = int(cell["col_start"])
        text = str(cell.get("text", "")).strip()
        colspan = int(cell.get("colspan", 1))
        rowspan = int(cell.get("rowspan", 1))
        role = cell.get("role", "")
        
        # Track last header row
        if role == "header":
            last_header_row = max(last_header_row, row)
        
        # Fill spans by duplicating value
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
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

    headers_df = table_df[table_df["role"] == "header"]
    data_df = table_df[table_df["role"] != "header"].copy()

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
        value_cells = row_data[row_data["role"] == "data"]

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

    headers_df = table_df[table_df["role"] == "header"]
    data_df = table_df[table_df["role"] != "header"].copy()

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
        row_label_rows = row_data[row_data["role"] == "row_label"]
        row_label = row_label_rows.iloc[0]["_text"] if not row_label_rows.empty else f"row_{row}"

        value_cells = row_data[row_data["role"] == "data"]

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
    table_cells_df: pd.DataFrame,
    representation: str = "markdown",
) -> str:
    """
    Generate appropriate representation of a table block.

    Formats:
      - "markdown": Pipe-separated markdown table
      - "jsonl": One JSON line per row with headers
      - "melted": One fact per row (fully melted)

    Args:
        table_id: Unique identifier for the table
        df_lines: Lines belonging to this table block
        table_cells_df: Full table cells dataframe (filtered to this table)
        representation: Format to use for table output

    Returns:
        Formatted table text
    """
    if table_cells_df is None or table_cells_df.empty:
        return _build_text_from_lines(df_lines)

    table_df = table_cells_df[table_cells_df["table_id"] == table_id].copy() if table_id else table_cells_df.copy()

    if table_df.empty:
        return _build_text_from_lines(df_lines)

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
        return _build_text_from_lines(df_lines)

    chart_df = chart_points_df[chart_points_df["chart_id"] == chart_id].copy()
    if chart_df.empty:
        return _build_text_from_lines(df_lines)

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

def _starts_with_any_token(s: str, tokens: set[str]) -> bool:
    """Check if string starts with any token from the set."""
    if not s:
        return False
    s2 = s.lstrip()
    if not s2:
        return False
    if s2[0] in tokens:
        return True
    for t in tokens:
        if s2.startswith(t + " "):
            return True
    return False


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
        has_table_id = df_with_block_ids["table_id"].fillna("").astype(str).str.strip().ne("")

    # Rule 1: TOC and exhibits always use newlines (if not part of a table)
    if "block_type" in df_with_block_ids.columns:
        is_toc_or_exhibits = df_with_block_ids["block_type"].isin(["toc", "exhibits"])
        df_with_block_ids.loc[is_toc_or_exhibits & ~has_table_id, "_join_sep"] = "\n"

    # Rule 2: Lines starting with bullet tokens use newlines (if not part of a table)
    text_s = df_with_block_ids["text"].fillna("").astype(str)
    is_bullet_start = text_s.map(lambda x: _starts_with_any_token(x, _BULLET_TOKENS))
    df_with_block_ids.loc[is_bullet_start & ~has_table_id, "_join_sep"] = "\n"

    # Rule 3: Lines with hierarchy markers use newlines (if not part of a table)
    if "hierarchy_marker" in df_with_block_ids.columns:
        has_hm = df_with_block_ids["hierarchy_marker"].fillna("").astype(str).str.strip().ne("")
        df_with_block_ids.loc[has_hm & ~has_table_id, "_join_sep"] = "\n"
    
    return df_with_block_ids


def _build_text_from_lines(lines: pd.DataFrame) -> str:
    """
    Build block text by joining line texts with their respective separators.

    Args:
        lines: DataFrame rows for ONE block (in document order)

    Returns:
        Joined text string
    """
    texts = lines["text"].fillna("").astype(str).tolist()
    seps = lines["_join_sep"].fillna(" ").astype(str).tolist()

    out_parts: list[str] = []
    for i, t in enumerate(texts):
        t2 = t.strip()
        if not t2:
            continue
        if not out_parts:
            out_parts.append(t2)
        else:
            out_parts.append(seps[i] + t2)

    return "".join(out_parts)


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
    _tid = first["table_id"].fillna("").astype(str).str.strip() if "table_id" in first.columns else pd.Series("", index=first.index)
    _cid = first["chart_id"].fillna("").astype(str).str.strip() if "chart_id" in first.columns else pd.Series("", index=first.index)

    _has_table = ~_tid.isin({"", "nan", "None"})
    _has_chart = ~_cid.isin({"", "nan", "None"})

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
        def _build_special_text(lines: pd.DataFrame) -> str:
            if lines.empty:
                return ""
            block_type = lines["block_type"].iloc[0] if "block_type" in lines.columns else None
            table_id  = lines["table_id"].iloc[0]  if "table_id"  in lines.columns else None
            chart_id  = lines["chart_id"].iloc[0]  if "chart_id"  in lines.columns else None
            has_table = table_id is not None and str(table_id).strip() not in ("", "nan", "None")
            has_chart = chart_id is not None and str(chart_id).strip() not in ("", "nan", "None")

            if block_type == "table" or (block_type in ["toc", "exhibits"] and has_table):
                return _generate_table_block(table_id, lines, table_cells_df, table_representation)
            if block_type == "chart" or has_chart:
                return _generate_chart_block(chart_id, lines, chart_points_df, table_representation)
            return _build_text_from_lines(lines)

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
    agg_spec = build_standard_agg_spec(
        include_hierarchy=True,
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,
    )
    
    blocks_df = aggregate_hierarchical(
        df,
        group_col="_block_id",
        agg_spec=agg_spec,
        rename_group_col="block_id",
        compute_derived=True,
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
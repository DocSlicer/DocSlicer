"""
step_05_block_merger.py

Merge lines into logical blocks.

Architecture:
  1. _assign_block_ids() - Decision logic: what constitutes a new block?
  2. hierarchical_aggregator - Shared aggregation boilerplate
  3. _join_text() - Text merging strategy (space vs newline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.hierarchical_aggregator import (
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
         - block_type changes
         - layout_id changes
         - page_number changes (safety)
         - col_start changes (text_multicol only - each column becomes separate block)
         - heading_id changes (for consecutive heading lines - different headings become separate blocks)
      
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
    
    is_new_block = (
        df["block_type"].ne(prev_block_type) |  # Role change
        df["layout_id"].ne(prev_layout) |        # Layout change
        df["page_number"].ne(prev_page) |        # Page change
        col_start_change |                        # Column change (text_multicol only)
        indent_increase |                         # Indentation increase (conditional)
        style_change |                            # Style property change (within same layout)
        heading_id_change                         # Heading ID change (consecutive headings only)
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
    import json
    
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
    import json
    
    # Ensure row_start exists
    if "row_start" not in table_df.columns:
        if "temp_line_ids" in table_df.columns:
            table_df["row_start"] = table_df["temp_line_ids"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
            )
        else:
            return "[Table: missing row information]"
    
    # Ensure colspan exists
    if "colspan" not in table_df.columns:
        table_df["colspan"] = 1
    
    # Separate headers and data
    headers_df = table_df[table_df["role"] == "header"].copy()
    data_df = table_df[table_df["role"] != "header"].copy()
    
    if headers_df.empty:
        return "[Table: no headers found]"
    
    # Build header map considering colspan
    # Map: col_index -> list of header texts (one per header row)
    header_map = {}  # col_index -> [header_row_0_text, header_row_1_text, ...]
    
    # Get unique header rows
    header_rows = sorted(headers_df["row_start"].unique())
    
    # Process each header cell
    for _, header_cell in headers_df.iterrows():
        col_start = int(header_cell["col_start"])
        colspan = int(header_cell.get("colspan", 1))
        row = int(header_cell["row_start"])
        text = str(header_cell.get("text", "")).strip()
        
        # Apply this header to all columns it spans
        for col_offset in range(colspan):
            col = col_start + col_offset
            
            # Initialize this column's header list if needed
            if col not in header_map:
                header_map[col] = {}
            
            # Store header text for this row level
            header_map[col][row] = text
    
    # Build header keys (combine multi-row headers with underscore)
    header_keys = {}
    for col, row_texts in header_map.items():
        # Build key from all header rows in order
        key_parts = []
        for row in header_rows:
            if row in row_texts and row_texts[row]:
                key_parts.append(row_texts[row])
        
        header_keys[col] = "_".join(key_parts) if key_parts else f"col_{col}"
    
    # Build JSON objects per data row
    json_lines = []
    data_rows = sorted(data_df["row_start"].unique())
    
    for row in data_rows:
        row_data = data_df[data_df["row_start"] == row]
        row_obj = {}
        
        for _, cell in row_data.iterrows():
            col = int(cell["col_start"])
            # Get header
            header = header_keys.get(col, f"col_{col}")
            
            # Replace generic "col_0" with "Metric" for column 0
            if col == 0 and header == "col_0":
                header = "Metric"
            
            value = str(cell.get("text", "")).strip()
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
    import json
    
    # Ensure row_start exists
    if "row_start" not in table_df.columns:
        if "temp_line_ids" in table_df.columns:
            table_df["row_start"] = table_df["temp_line_ids"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
            )
        else:
            return "[Table: missing row information]"
    
    # Ensure colspan exists
    if "colspan" not in table_df.columns:
        table_df["colspan"] = 1
    
    # Separate headers, row labels, and values
    headers_df = table_df[table_df["role"] == "header"].copy()
    
    # Build header map considering colspan
    # Map: col_index -> list of header texts (one per header row)
    # For headers with colspan > 1, replicate across all covered columns
    header_map = {}  # col_index -> [header_row_0_text, header_row_1_text, ...]
    
    # Get unique header rows
    header_rows = sorted(headers_df["row_start"].unique())
    
    # Process each header cell
    for _, header_cell in headers_df.iterrows():
        col_start = int(header_cell["col_start"])
        colspan = int(header_cell.get("colspan", 1))
        row = int(header_cell["row_start"])
        text = str(header_cell.get("text", "")).strip()
        
        # Apply this header to all columns it spans
        for col_offset in range(colspan):
            col = col_start + col_offset
            
            # Initialize this column's header list if needed
            if col not in header_map:
                header_map[col] = {}
            
            # Store header text for this row level
            header_map[col][row] = text
    
    # Build header paths (combine multi-row headers)
    header_paths = {}
    for col, row_texts in header_map.items():
        # Build path from all header rows in order
        path_parts = []
        for row in header_rows:
            if row in row_texts and row_texts[row]:
                path_parts.append(row_texts[row])
        
        header_paths[col] = " > ".join(path_parts) if path_parts else f"col_{col}"
    
    # Process data rows
    data_df = table_df[table_df["role"] != "header"].copy()
    melted_lines = []
    
    for row in sorted(data_df["row_start"].unique()):
        row_data = data_df[data_df["row_start"] == row]
        
        # Get row label (first column with role="row_label")
        row_label_cells = row_data[row_data["role"] == "row_label"]
        row_label = row_label_cells.iloc[0]["text"] if not row_label_cells.empty else f"row_{row}"
        
        # Get value cells
        value_cells = row_data[row_data["role"].str.startswith("value", na=False)]
        
        for _, cell in value_cells.iterrows():
            col = int(cell["col_start"])
            header = header_paths.get(col, f"col_{col}")
            value = str(cell.get("text", "")).strip()
            
            # Format: row_label | header_path | value
            melted_lines.append(f"{row_label} | {header} | {value}")
    
    return "\n".join(melted_lines)


# =================================
# Narrated Format ### TODO ###
# =================================

def _format_table_narrated(table_df: pd.DataFrame) -> str:
    """
    Format table as natural language narration.
    
    TODO: Implement intelligent narration based on table structure and content.
    
    Args:
        table_df: Table cells
    
    Returns:
        Narrated description of table
    """
    # Placeholder: return markdown for now
    return _format_table_markdown(table_df)


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
      - "narrated": Natural language description [TODO]

    Args:
        table_id: Unique identifier for the table
        df_lines: Lines belonging to this table block
        table_cells_df: Full table cells dataframe (filtered to this table)
        representation: Format to use for table output

    Returns:
        Formatted table text
    """
    if table_cells_df is None or table_cells_df.empty:
        # Fallback if no table cells data available
        return _build_text_from_lines(df_lines)
    
    # Filter to cells belonging to this table
    table_df = table_cells_df[table_cells_df["table_id"] == table_id].copy() if table_id else table_cells_df.copy()
    
    if table_df.empty:
        return _build_text_from_lines(df_lines)
    
    # Route to appropriate formatter
    if representation == "markdown":
        return _format_table_markdown(table_df)
    elif representation == "jsonl":
        return _format_table_jsonl(table_df)
    elif representation == "melted":
        return _format_table_melted(table_df)
    elif representation == "narrated":
        return _format_table_narrated(table_df)
    else:
        # Unknown format: fallback to markdown
        return _format_table_markdown(table_df)


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
    has_table_id = False
    if "table_id" in df_with_block_ids.columns:
        table_id_s = df_with_block_ids["table_id"].astype("string")
        has_table_id = table_id_s.notna() & table_id_s.str.strip().ne("")
    
    # Rule 1: TOC and exhibits always use newlines (if not part of a table)
    if "block_type" in df_with_block_ids.columns:
        is_toc_or_exhibits = df_with_block_ids["block_type"].isin(["toc", "exhibits"])
        is_toc_or_exhibits_non_table = is_toc_or_exhibits & ~has_table_id
        df_with_block_ids.loc[is_toc_or_exhibits_non_table, "_join_sep"] = "\n"
    
    # Rule 2: Lines starting with bullet tokens use newlines (if not part of a table)
    text_s = df_with_block_ids["text"].astype("string").fillna("")
    is_bullet_start = text_s.map(lambda x: _starts_with_any_token(str(x), _BULLET_TOKENS))
    is_bullet_start_non_table = is_bullet_start & ~has_table_id
    df_with_block_ids.loc[is_bullet_start_non_table, "_join_sep"] = "\n"
    
    # Rule 3: Lines with hierarchy markers use newlines (if not part of a table)
    if "hierarchy_marker" in df_with_block_ids.columns:
        hm_s = df_with_block_ids["hierarchy_marker"].astype("string")
        has_hm = hm_s.notna() & hm_s.str.strip().ne("")
        has_hm_non_table = has_hm & ~has_table_id
        df_with_block_ids.loc[has_hm_non_table, "_join_sep"] = "\n"
    
    return df_with_block_ids


def _build_text_from_lines(lines: pd.DataFrame) -> str:
    """
    Build block text by joining line texts with their respective separators.
    
    Args:
        lines: DataFrame rows for ONE block (in document order)
    
    Returns:
        Joined text string
    """
    texts = lines["text"].astype("string").fillna("").tolist()
    seps = lines["_join_sep"].astype("string").fillna(" ").tolist()
    
    out_parts: list[str] = []
    for i, t in enumerate(texts):
        t2 = str(t).strip()
        if not t2:
            continue
        if not out_parts:
            # First line: no separator
            out_parts.append(t2)
        else:
            # Subsequent lines: use separator
            out_parts.append(str(seps[i]) + t2)
    
    return "".join(out_parts)


# =================================
# Main Text Joiner
# =================================

def _join_text(
    df_with_block_ids: pd.DataFrame,
    blocks_df: pd.DataFrame,
    table_cells_df: pd.DataFrame = None,
    table_representation: str = "markdown",
) -> pd.DataFrame:
    """
    Join line texts into block text with strategy-specific formatting.
    
    Text joining strategies:
      1. Blocks with table_id (including table, toc, exhibits): Use table rendering
      2. block_type = "toc" or "exhibits" (without table_id): Join with newlines
      3. Lines with bullet tokens or hierarchy markers: Join with newlines
      4. Default: Join with spaces
    
    Args:
        df_with_block_ids: Lines df with _block_id column
        blocks_df: Aggregated blocks df (without text column)
        table_cells_df: Full table cells dataframe (for table formatting)
        table_representation: Format for table output ("markdown", "jsonl", "melted", "narrated")
    
    Returns:
        blocks_df with "text" column added
    """
    # -------------------------------------------------------------------------
    # STEP 1: COMPUTE LINE SEPARATORS
    # -------------------------------------------------------------------------
    df_with_block_ids = _compute_line_separator(df_with_block_ids)
    
    # -------------------------------------------------------------------------
    # STEP 2: BUILD TEXT PER BLOCK
    # -------------------------------------------------------------------------
    def _build_block_text(lines: pd.DataFrame) -> str:
        """Build text for one block based on block_type and table_id."""
        if lines.empty:
            return ""
        
        # Get block_type and table_id (all lines in a block share the same values)
        block_type = lines["block_type"].iloc[0] if "block_type" in lines.columns else None
        table_id = lines["table_id"].iloc[0] if "table_id" in lines.columns else None
        
        # Check if this block has a table
        has_table = table_id is not None and str(table_id).strip() != ""
        
        # Strategy 1: Blocks with tables (table, toc, exhibits)
        if block_type == "table" or (block_type in ["toc", "exhibits"] and has_table):
            return _generate_table_block(
                table_id,
                lines,
                table_cells_df,
                table_representation,
            )
        
        # Strategy 2-4: Text blocks (use computed separators)
        return _build_text_from_lines(lines)
    
    text_df = (
        df_with_block_ids.groupby("_block_id", sort=False, observed=True)
        .apply(_build_block_text, include_groups=False)
        .reset_index()
        .rename(columns={"_block_id": "block_id", 0: "text"})
    )
    
    # -------------------------------------------------------------------------
    # STEP 3: MERGE TEXT INTO BLOCKS
    # -------------------------------------------------------------------------
    blocks_df = blocks_df.merge(text_df, on="block_id", how="left")
    
    return blocks_df


# =======================================================================================================================
# Public API
# =======================================================================================================================

def merge_blocks(
    lines_df: pd.DataFrame,
    table_cells_df: pd.DataFrame = None,
    table_representation: str = "markdown",
) -> pd.DataFrame:
    """
    Merge lines into logical blocks.
    
    Process:
      1. Validate inputs and prepare for aggregation
      2. Assign block IDs (decision logic)
      3. Aggregate to block level (shared boilerplate)
      4. Join text with smart separators (tables use table_representation)
      5. Compute embed_char_count (includes whitespace for embedding estimation)
      6. Sort by document order
    
    Args:
        lines_df: Lines-level DataFrame
        table_cells_df: Table cells DataFrame (for table formatting)
        table_representation: Format for table output ("markdown", "jsonl", "melted", "narrated")
    
    Returns:
        Blocks-level DataFrame with embed_char_count column added
    """
    if lines_df is None or lines_df.empty:
        return pd.DataFrame()
    
    # -------------------------
    # STEP 1: ASSIGN BLOCK IDs
    # -------------------------
    df = _assign_block_ids(lines_df.copy())
    
    # -------------------------
    # STEP 2: AGGREGATE
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
    blocks_df = _join_text(df, blocks_df, table_cells_df, table_representation)
    
    # -------------------------
    # STEP 5: COMPUTE EMBED CHAR COUNT
    # -------------------------
    # embed_char_count includes all characters (including whitespace) for embedding estimation
    # This differs from char_count which only counts net characters
    if "text" in blocks_df.columns:
        blocks_df["embed_char_count"] = blocks_df["text"].astype("string").fillna("").str.len()
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
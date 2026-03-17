"""
Table JSON to Markdown Converter

Converts TableGrid JSON structures to beautifully aligned markdown tables.

FEATURES:
---------
- Column alignment with proper spacing
- Truncates long cell text (configurable max_cell_width)
- Shows span indicators: [→N] for colspan, [↓N] for rowspan
- Escapes pipe characters in cell content
- Uses header_rows metadata for proper separator placement

USAGE:
------
from .table_to_markdown import tablegrid_to_markdown

table_dict = {...}  # TableGrid JSON from extract_tables_from_html()
markdown = tablegrid_to_markdown(table_dict)
"""

from __future__ import annotations
from typing import Any
import json
import sys

JsonDict = dict[str, Any]


def tablegrid_to_markdown(tg_dict: JsonDict, max_cell_width: int = 40) -> str:
    """
    Convert TableGrid JSON to beautifully aligned markdown table.
    
    Features:
    - Properly aligned columns
    - Handles rowspan and colspan
    - Shows span indicators for multi-column/row cells
    - Truncates long text intelligently
    
    Note: Markdown doesn't support rowspan, so cells spanning multiple rows
    will show their content in the first row and "↓" indicators in subsequent rows.
    
    Args:
        tg_dict: TableGrid dictionary with 'shape', 'cells', 'grid' keys
        max_cell_width: Maximum cell text width before truncation
        
    Returns:
        Markdown string representation of the table
    """
    n_rows = tg_dict['shape']['rows']
    n_cols = tg_dict['shape']['cols']
    
    # Build cell lookup
    cells_by_id = {cell['id']: cell for cell in tg_dict['cells']}
    grid = tg_dict['grid']
    
    # First pass: build the markdown structure (collect cells for each row)
    # For each row, we need ALL columns represented (even if spanned from above)
    rows_data: list[list[tuple[str, int, bool]]] = []  # [(text, colspan, is_continuation), ...]
    
    for r in range(n_rows):
        row_cells: list[tuple[str, int, bool]] = []
        c = 0
        
        while c < n_cols:
            cell_id = grid[r][c]
            cell = cells_by_id[cell_id]
            
            # Check if this cell originates in this row
            if cell['r0'] == r and cell['c0'] == c:
                # New cell originating here
                text = cell['text'] if cell['text'] else ""
                colspan = cell['colspan']
                
                # Truncate if needed
                if len(text) > max_cell_width:
                    text = text[:max_cell_width-3] + "..."
                
                # Add span indicator if needed
                if cell['rowspan'] > 1:
                    text = text + f" [↓{cell['rowspan']}]" if text else f"[↓{cell['rowspan']}]"
                
                # Escape pipe characters
                text = text.replace("|", "\\|")
                
                row_cells.append((text, colspan, False))
                c += colspan
            elif cell['r0'] < r and cell['c0'] == c:
                # Cell spans down from above row - show continuation marker
                colspan = cell['colspan']
                row_cells.append(("↓", colspan, True))
                c += colspan
            else:
                # Part of a colspan from same row
                c += 1
        
        rows_data.append(row_cells)
    
    # Second pass: calculate column widths for alignment
    max_virtual_cols = max(sum(colspan for _, colspan, _ in row) for row in rows_data) if rows_data else 0
    col_widths = [0] * max_virtual_cols
    
    for row_cells in rows_data:
        virtual_col = 0
        for text, colspan, _ in row_cells:
            if colspan == 1:
                col_widths[virtual_col] = max(col_widths[virtual_col], len(text))
            virtual_col += colspan
    
    # Ensure minimum width of 3 for all columns
    col_widths = [max(3, w) for w in col_widths]
    
    # Third pass: build markdown with proper alignment
    lines: list[str] = []
    
    for r, row_cells in enumerate(rows_data):
        cells_formatted = []
        virtual_col = 0
        
        for text, colspan, is_continuation in row_cells:
            if colspan == 1:
                # Single column: left-align with padding
                width = col_widths[virtual_col]
                formatted = text.ljust(width)
            else:
                # Multi-column: show span indicator
                total_width = sum(col_widths[virtual_col:virtual_col + colspan])
                total_width += (colspan - 1) * 3  # Add space for separators
                span_indicator = f" [→{colspan}]" if text and not is_continuation else f"[→{colspan}]"
                formatted = (text + span_indicator).ljust(total_width)
            
            cells_formatted.append(formatted)
            virtual_col += colspan
        
        lines.append("| " + " | ".join(cells_formatted) + " |")
        
        # Add separator after header rows
        # Use header_rows info from source metadata if available
        header_rows = tg_dict.get('source', {}).get('header_rows', [])
        if header_rows:
            # Add separator after the last header row
            last_header_row = max(header_rows)
            if r == last_header_row:
                separator_cells = []
                virtual_col = 0
                for _, colspan, _ in row_cells:
                    if colspan == 1:
                        separator_cells.append("-" * col_widths[virtual_col])
                    else:
                        total_width = sum(col_widths[virtual_col:virtual_col + colspan])
                        total_width += (colspan - 1) * 3
                        separator_cells.append("-" * total_width)
                    virtual_col += colspan
                lines.append("| " + " | ".join(separator_cells) + " |")
        else:
            # Fallback: add separator after first row if it looks like a header
            if r == 0 and rows_data[0] and rows_data[0][0][0]:
                separator_cells = []
                virtual_col = 0
                for _, colspan, _ in row_cells:
                    if colspan == 1:
                        separator_cells.append("-" * col_widths[virtual_col])
                    else:
                        total_width = sum(col_widths[virtual_col:virtual_col + colspan])
                        total_width += (colspan - 1) * 3
                        separator_cells.append("-" * total_width)
                    virtual_col += colspan
                lines.append("| " + " | ".join(separator_cells) + " |")
    
    return "\n".join(lines)


def tablegrid_to_markdown_with_metadata(tg_dict: JsonDict, max_cell_width: int = 40) -> str:
    """
    Generate a full markdown report with metadata + table.
    
    Includes:
    - Table shape (rows × cols)
    - Total cells count
    - Table ID
    - Normalization operations (if applicable)
    - The formatted table
    
    Args:
        tg_dict: TableGrid dictionary
        max_cell_width: Maximum cell text width before truncation
        
    Returns:
        Full markdown report string
    """
    lines = []
    
    # Metadata
    shape = tg_dict['shape']
    lines.append(f"**Table {tg_dict['table_id']}** ({shape['rows']}×{shape['cols']})")
    
    # Source info (normalization operations)
    if 'source' in tg_dict:
        source = tg_dict['source']
        if source.get('type') == 'normalized' and 'operations' in source:
            ops = source['operations']
            if ops:
                lines.append(f"*{len(ops)} normalization operations applied*")
    
    lines.append("")
    lines.append(tablegrid_to_markdown(tg_dict, max_cell_width))
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Specify the path to your table JSON file here
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Default path - change this to your table JSON file path
        json_path = "backend/app/services/parsing/html/test_output/tables.json"
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict) and 'tables' in data:
        # JSON has a "tables" key containing an array
        tables = data['tables']
        for table in tables:
            print(tablegrid_to_markdown_with_metadata(table))
            print("\n" + "="*80 + "\n")
    elif isinstance(data, list):
        # JSON is directly a list of tables
        for table in data:
            print(tablegrid_to_markdown_with_metadata(table))
            print("\n" + "="*80 + "\n")
    else:
        # Single table object
        print(tablegrid_to_markdown_with_metadata(data))


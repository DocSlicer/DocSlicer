"""
SEC HTML Table Parser - Standalone Version with DataFrame Output

This module parses SEC HTML tables into clean, normalized DataFrames.

ARCHITECTURE:
-------------
- TableCell: Atomic cell with position (r0, c0), span (rowspan, colspan), and content
- TableGrid: Complete table with cells dict and grid matrix (references cell IDs)
- ColProfile: Analysis of column content (blank, currency-only, paren-only, etc.)

PIPELINE:
---------
1. parse_html_to_tablegrid() → TableGrid (standard, all columns as-is from HTML)
2. profile_columns() → analyze each column's content
3. normalize_tablegrid() → apply transformations:
   - Merge currency columns ($, €) into adjacent data columns
   - Merge paren columns into adjacent data columns
   - Remove blank spacer columns
   - Update all cell positions and spans accordingly
4. detect_cell_roles() → identify header/row_label/data cells
5. Convert to DataFrame for easy inspection and export

KEY FEATURES:
-------------
- Cell provenance: Every cell has a unique ID, tracked through all transformations
- Span handling: Rowspan/colspan are properly expanded, then adjusted during normalization
- Audit trail: All normalization operations are logged (merge, remove)
- DataFrame output: Returns pandas DataFrame for easy inspection/export as CSV
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Optional
from pathlib import Path
import re
import warnings
import tempfile

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XML warning since we intentionally parse XHTML as HTML
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

JsonDict = dict[str, Any]


# =========================
# Data Classes
# =========================

@dataclass
class TableCell:
    """Atomic cell with its position and span in the grid"""
    id: int
    r0: int              # origin row (top-left)
    c0: int              # origin col (top-left)
    rowspan: int
    colspan: int
    text: str
    ix: Optional[JsonDict] = None      # XBRL metadata
    style: Optional[str] = None
    role: Optional[str] = None         # "header", "row_label", "data", "spacer"

    def to_dict(self) -> JsonDict:
        """Serialize to JSON-friendly dict"""
        d = {
            "id": self.id,
            "r0": self.r0,
            "c0": self.c0,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "text": self.text,
        }
        if self.ix:
            d["ix"] = self.ix
        if self.role:
            d["role"] = self.role
        return d


@dataclass
class TableGrid:
    """Complete table representation"""
    table_id: str
    n_rows: int
    n_cols: int
    cells: dict[int, TableCell]        # cells by ID
    grid: list[list[int]]              # grid[r][c] = cell_id
    source: Optional[JsonDict] = None  # provenance metadata

    def to_dict(self, include_source: bool = True) -> JsonDict:
        """Serialize to JSON-friendly dict"""
        d = {
            "table_id": self.table_id,
            "shape": {"rows": self.n_rows, "cols": self.n_cols},
            "cells": [cell.to_dict() for cell in self.cells.values()],
            "grid": self.grid,
        }
        if include_source and self.source:
            d["source"] = self.source
        return d
    
    def to_cells_dataframe(self, page_number: int = 0, table_id: int = 1) -> pd.DataFrame:
        """
        Convert TableGrid to a normalized cell-based DataFrame.
        Each row in the DataFrame represents one cell in the table.
        
        Args:
            page_number: Page number this table appears on
            table_id: Table ID (1-indexed)
        
        Returns:
            DataFrame with columns: page_number, table_id, table_cell_id, 
            row_start, col_start, rowspan, colspan, text, role, ix, style
        """
        rows = []
        
        # Get unique cells (not references in grid)
        for cell_id, cell in self.cells.items():
            row_data = {
                "page_number": page_number,
                "table_id": table_id,
                "table_cell_id": cell.id,
                "row_start": cell.r0,
                "col_start": cell.c0,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "text": cell.text,
                "role": cell.role if cell.role else None,
                "ix": str(cell.ix) if cell.ix else None,
                "style": cell.style if cell.style else None,
            }
            rows.append(row_data)
        
        return pd.DataFrame(rows).sort_values(["row_start", "col_start"]).reset_index(drop=True)


@dataclass(frozen=True)
class ColProfile:
    """Analysis of a single column's content"""
    idx: int
    all_blank: bool
    currency_only: bool      # all non-blank are "$", "€", etc.
    lparen_only: bool        # all non-blank are "("
    rparen_only: bool        # all non-blank are ")"
    sample_values: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ColOperation:
    """Track normalization operations for audit trail"""
    action: str              # "merge", "remove"
    source_col: int
    target_col: Optional[int]
    reason: str


# =========================
# Utilities
# =========================

_WS_RE = re.compile(r"\s+")

def norm_ws(s: str) -> str:
    """Normalize whitespace"""
    s = (s or "").replace("\xa0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_currency_token(s: str) -> bool:
    """Check if string is just a currency symbol"""
    return norm_ws(s) in {"$", "€", "£", "¥", "₹", "₽", "₩", "₪", "₺"}


def is_lparen_token(s: str) -> bool:
    """Check if string is just left paren"""
    return norm_ws(s) in {"(", "[", "{"}


def is_rparen_token(s: str) -> bool:
    """Check if string is just right paren"""
    return norm_ws(s) in {")", "]", "}", "%"}


def looks_like_numberish(s: str) -> bool:
    """Heuristic: does this look like a number or placeholder?"""
    x = norm_ws(s)
    if x in {"—", "-", ""}:
        return True
    # Match patterns like: 123, 1,234, $1,234, (123), -123, etc.
    return bool(re.fullmatch(r"[\(\-]?\$?[\d,]+(\.\d+)?\)?", x))


def extract_ix_metadata(cell_tag) -> Optional[JsonDict]:
    """Extract XBRL inline metadata if present"""
    ix = cell_tag.find(lambda t: getattr(t, "name", "") and str(t.name).startswith("ix:"))
    if not ix:
        return None
    
    def attr(name: str) -> Optional[str]:
        v = ix.get(name)
        return str(v) if v is not None else None
    
    return {
        "name": attr("name"),
        "contextref": attr("contextref"),
        "unitref": attr("unitref"),
        "decimals": attr("decimals"),
        "scale": attr("scale"),
        "sign": attr("sign"),
        "id": attr("id"),
    }


# =========================
# HTML Parsing
# =========================

def parse_html_to_tablegrid(html: str, table_index: int = 0) -> TableGrid:
    """Parse HTML table into a fully expanded TableGrid (standard representation)"""
    
    # Parse HTML
    soup = BeautifulSoup(html, "lxml")
    
    tables = soup.find_all("table")
    if not tables:
        raise ValueError(f"No <table> found in HTML")
    if table_index < 0 or table_index >= len(tables):
        raise ValueError(f"table_index {table_index} out of range (found {len(tables)} tables)")
    
    table = tables[table_index]
    
    # Parse raw cells with rowspan/colspan
    raw_rows = []
    for tr in table.find_all("tr", recursive=True):
        cells = []
        tds = tr.find_all(["td", "th"], recursive=False)
        if not tds:
            tds = tr.find_all(["td", "th"], recursive=True)
        
        for td in tds:
            colspan = int(td.get("colspan") or 1)
            rowspan = int(td.get("rowspan") or 1)
            text = norm_ws(td.get_text(" ", strip=True))
            style = td.get("style")
            ix = extract_ix_metadata(td)
            
            cells.append({
                "text": text,
                "rowspan": rowspan,
                "colspan": colspan,
                "style": str(style) if style else None,
                "ix": ix,
            })
        
        if cells or any(cells):  # Keep even empty rows
            raw_rows.append(cells)
    
    # Expand rowspan/colspan into rectangular grid
    grid: list[list[Optional[int]]] = []
    pending: dict[int, tuple[int, int]] = {}  # col -> (cell_id, remaining_rows)
    cells_dict: dict[int, TableCell] = {}
    cell_id_counter = 0
    
    for r_idx, raw_row in enumerate(raw_rows):
        row: list[Optional[int]] = []
        c = 0
        
        for raw_cell in raw_row:
            # Skip to next available column (handle pending rowspans)
            while c in pending:
                cell_id, rem = pending[c]
                row.append(cell_id)
                if rem - 1 <= 0:
                    del pending[c]
                else:
                    pending[c] = (cell_id, rem - 1)
                c += 1
            
            # Create new cell
            cell = TableCell(
                id=cell_id_counter,
                r0=r_idx,
                c0=c,
                rowspan=raw_cell["rowspan"],
                colspan=raw_cell["colspan"],
                text=raw_cell["text"],
                ix=raw_cell["ix"],
                style=raw_cell["style"],
            )
            cells_dict[cell_id_counter] = cell
            
            # Fill grid positions for this cell
            for dc in range(cell.colspan):
                row.append(cell_id_counter)
                if cell.rowspan > 1:
                    pending[c + dc] = (cell_id_counter, cell.rowspan - 1)
            
            c += cell.colspan
            cell_id_counter += 1
        
        # Handle any remaining pending cells at end of row
        max_c = max(len(row), (max(pending.keys()) + 1) if pending else 0)
        while len(row) < max_c:
            if c in pending:
                cell_id, rem = pending[c]
                row.append(cell_id)
                if rem - 1 <= 0:
                    del pending[c]
                else:
                    pending[c] = (cell_id, rem - 1)
            else:
                row.append(None)
            c += 1
        
        grid.append(row)
    
    # Ensure rectangular grid
    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        while len(r) < max_cols:
            r.append(None)
    
    # Convert Optional[int] to int by filling Nones with empty cells
    final_grid: list[list[int]] = []
    for r_idx, row in enumerate(grid):
        final_row: list[int] = []
        for c_idx, cell_id in enumerate(row):
            if cell_id is None:
                # Create empty filler cell
                empty_cell = TableCell(
                    id=cell_id_counter,
                    r0=r_idx,
                    c0=c_idx,
                    rowspan=1,
                    colspan=1,
                    text="",
                )
                cells_dict[cell_id_counter] = empty_cell
                final_row.append(cell_id_counter)
                cell_id_counter += 1
            else:
                final_row.append(cell_id)
        final_grid.append(final_row)
    
    return TableGrid(
        table_id=f"table_{table_index}",
        n_rows=len(final_grid),
        n_cols=max_cols,
        cells=cells_dict,
        grid=final_grid,
        source={"type": "html", "index": table_index},
    )


# =========================
# Cell Role Detection
# =========================

def detect_cell_roles(tg: TableGrid) -> TableGrid:
    """
    Detect and assign roles to each cell in the table.
    
    Roles:
    - "header": Header rows (first row + additional rows based on heuristics)
    - "row_label": First cell of each data row
    - "data": All other data cells
    
    Header detection:
    1. First row is always a header
    2. Additional rows are headers if:
       - First row has rowspan cells (spanning into those rows)
       - Row contains year values (20XX without commas)
       - Row contains unit indicators (in thousands, in millions, etc.)
       - Row contains dates (September 30, 2025, etc.)
    3. Stop when a row has numberish data cells
    """
    
    # Header row detection patterns
    year_pattern = re.compile(r'\b20\d{2}\b')  # 2000-2099
    date_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}\b|\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}\b', re.IGNORECASE)
    unit_phrases = {
        'in thousands', 'in millions', 'in billions',
        'except per share', 'per share', 
        'percentage', '(%)', 
        'year ended', 'years ended',
        'months ended', 'month ended',
        'quarters ended', 'quarter ended',
        'total', 'actual', 'adjusted',
        'number', 'shares', 'amount', 'value',
    }
    
    # Determine header rows
    header_rows = set([0])  # First row is always a header
    
    # Check if first row has rowspan cells
    for c in range(tg.n_cols):
        cell = tg.cells[tg.grid[0][c]]
        if cell.r0 == 0 and cell.rowspan > 1:
            # Add all rows spanned by this cell as potential headers
            for r in range(1, cell.rowspan):
                header_rows.add(r)
    
    # Check subsequent rows (up to row 5 max for headers)
    for r in range(1, min(6, tg.n_rows)):
        if r in header_rows:
            continue  # Already marked as header
        
        # Collect text from all cells in this row
        row_texts = []
        has_numberish_data = False
        
        for c in range(tg.n_cols):
            cell = tg.cells[tg.grid[r][c]]
            if cell.r0 == r:  # Cell originates in this row
                text = cell.text.strip()
                if text:
                    row_texts.append(text.lower())
                    # Check if this looks like numeric data (not a year or date)
                    if looks_like_numberish(text) and not year_pattern.search(text):
                        # This looks like data, not a header
                        has_numberish_data = True
        
        # If row has numeric data, stop checking for more headers
        if has_numberish_data:
            break
        
        # Check for header indicators
        row_text_combined = ' '.join(row_texts)
        
        # Check for years (2024, 2025, etc.)
        if year_pattern.search(row_text_combined):
            header_rows.add(r)
            continue
        
        # Check for dates
        if date_pattern.search(row_text_combined):
            header_rows.add(r)
            continue
        
        # Check for unit phrases
        if any(phrase in row_text_combined for phrase in unit_phrases):
            header_rows.add(r)
            continue
        
        # If we didn't find any header indicators, stop
        break
    
    # Now assign roles to all cells
    for cell_id, cell in tg.cells.items():
        if cell.r0 in header_rows:
            # Header cell
            cell.role = "header"
        elif cell.c0 == 0 and cell.r0 not in header_rows:
            # First column of data row = row label
            cell.role = "row_label"
        else:
            # Data cell
            cell.role = "data"
    
    # Store header_rows info in source metadata
    if tg.source is None:
        tg.source = {}
    tg.source['header_rows'] = sorted(list(header_rows))
    
    return tg


# =========================
# Column Analysis
# =========================

def profile_columns(tg: TableGrid) -> list[ColProfile]:
    """Analyze each column to determine if it's blank, currency-only, etc."""
    profiles: list[ColProfile] = []
    
    for c in range(tg.n_cols):
        # Get unique cell texts in this column
        # For currency/paren detection, only consider standalone cells (colspan=1, rowspan=1)
        # that truly "belong" to this column, not cells spanning through it
        texts_all = []
        texts_data_only = []
        texts_standalone = []  # Cells with colspan=1 AND rowspan=1 (truly standalone)
        texts_standalone_data = []  # Standalone cells in data rows only
        seen_cell_ids = set()
        
        for r in range(tg.n_rows):
            cell_id = tg.grid[r][c]
            cell = tg.cells[cell_id]
            
            # Only count cells that originate in this column
            if cell.c0 == c and cell_id not in seen_cell_ids:
                texts_all.append(cell.text)
                
                # Track truly standalone cells (not spanning at all)
                if cell.colspan == 1 and cell.rowspan == 1:
                    texts_standalone.append(cell.text)
                    if r >= 5:  # Data rows only
                        texts_standalone_data.append(cell.text)
                
                # Track all data row cells (for general profiling)
                if r >= 5:
                    texts_data_only.append(cell.text)
                
                seen_cell_ids.add(cell_id)
        
        nonblank_all = [t for t in texts_all if t != ""]
        nonblank_data = [t for t in texts_data_only if t != ""]
        nonblank_standalone = [t for t in texts_standalone if t != ""]
        nonblank_standalone_data = [t for t in texts_standalone_data if t != ""]
        
        # Consider column blank if:
        # 1. Truly all blank, OR
        # 2. No standalone content (only spanning cells) + blank data rows
        all_blank = (len(nonblank_all) == 0) or \
                    (len(nonblank_standalone_data) == 0 and len(nonblank_standalone) == 0)
        
        # For currency/paren detection, ONLY use standalone cells in data rows
        # This ignores cells that span across this column from adjacent columns
        texts_to_check = nonblank_standalone_data if nonblank_standalone_data else nonblank_standalone
        currency_only = (not all_blank) and len(texts_to_check) > 0 and all(is_currency_token(t) for t in texts_to_check)
        lparen_only = (not all_blank) and len(texts_to_check) > 0 and all(is_lparen_token(t) for t in texts_to_check)
        rparen_only = (not all_blank) and len(texts_to_check) > 0 and all(is_rparen_token(t) for t in texts_to_check)
        
        profiles.append(ColProfile(
            idx=c,
            all_blank=all_blank,
            currency_only=currency_only,
            lparen_only=lparen_only,
            rparen_only=rparen_only,
            sample_values=nonblank_all[:3],
        ))
    
    return profiles


def find_next_data_col(profiles: list[ColProfile], start: int, direction: int = 1) -> Optional[int]:
    """Find next column that contains actual data (not blank/currency/parens)"""
    cols = range(start + direction, len(profiles), direction) if direction > 0 else range(start + direction, -1, direction)
    for c in cols:
        p = profiles[c]
        if not (p.all_blank or p.currency_only or p.lparen_only or p.rparen_only):
            return c
    return None


# =========================
# Normalization
# =========================

def merge_column_content(
    tg: TableGrid,
    source_col: int,
    target_col: int,
    prefix: bool = True,
) -> TableGrid:
    """
    Merge content from source_col into target_col, then remove source_col.
    
    - For standalone cells in source_col: merge text into target cell
    - For cells spanning across source_col: reduce colspan by 1
    - For cells after source_col: shift left by 1
    """
    new_cells: dict[int, TableCell] = {}
    updated_texts: dict[int, str] = {}  # cell_id -> updated text
    
    # First pass: identify text merges
    for r in range(tg.n_rows):
        source_cell_id = tg.grid[r][source_col]
        source_cell = tg.cells[source_cell_id]
        
        # Only merge if cell originates in source column and doesn't span beyond it
        if source_cell.c0 == source_col and source_cell.colspan == 1 and source_cell.text:
            target_cell_id = tg.grid[r][target_col]
            target_cell = tg.cells[target_cell_id]
            
            # Merge text
            if prefix:
                updated_texts[target_cell_id] = source_cell.text + target_cell.text
            else:
                updated_texts[target_cell_id] = target_cell.text + source_cell.text
    
    # Second pass: rebuild cells with updated positions and spans
    for cell_id, cell in tg.cells.items():
        c0, c_end = cell.c0, cell.c0 + cell.colspan - 1
        
        # Case 1: Cell originates in source column (standalone) -> skip it
        if c0 == source_col and cell.colspan == 1:
            continue
        
        # Case 2: Cell originates in source column but spans beyond it -> shift left, reduce colspan
        elif c0 == source_col and cell.colspan > 1:
            # When we remove source_col, this cell needs to move to the column before it
            # and reduce its colspan by 1
            new_cells[cell_id] = replace(
                cell,
                c0=cell.c0,  # Don't shift c0 yet - let Case 4 handle it
                colspan=cell.colspan - 1,
                text=updated_texts.get(cell_id, cell.text),
            )
        
        # Case 3: Cell starts before source column but spans across it -> reduce colspan
        elif c0 < source_col <= c_end:
            new_cells[cell_id] = replace(
                cell,
                colspan=cell.colspan - 1,
                text=updated_texts.get(cell_id, cell.text),
            )
        
        # Case 4: Cell starts after source column -> shift left
        elif c0 > source_col:
            new_cells[cell_id] = replace(
                cell,
                c0=cell.c0 - 1,
                text=updated_texts.get(cell_id, cell.text),
            )
        
        # Case 5: Cell before source column -> keep as is (but maybe update text)
        else:
            new_cells[cell_id] = replace(
                cell,
                text=updated_texts.get(cell_id, cell.text),
            )
    
    # Rebuild grid properly by expanding cells with their new dimensions
    new_n_cols = tg.n_cols - 1
    new_grid: list[list[int]] = []
    
    for r in range(tg.n_rows):
        new_row: list[int] = [0] * new_n_cols
        seen_in_row = set()
        
        # Place each cell in the new grid
        for cell_id, cell in new_cells.items():
            if cell.r0 <= r < cell.r0 + cell.rowspan:
                # This cell appears in this row
                for dc in range(cell.colspan):
                    col_idx = cell.c0 + dc
                    if 0 <= col_idx < new_n_cols:
                        new_row[col_idx] = cell_id
                        seen_in_row.add(cell_id)
        
        new_grid.append(new_row)
    
    return TableGrid(
        table_id=tg.table_id,
        n_rows=tg.n_rows,
        n_cols=new_n_cols,
        cells=new_cells,
        grid=new_grid,
        source=tg.source,
    )


def remove_column(tg: TableGrid, col_to_remove: int) -> TableGrid:
    """Remove a column entirely (used for blank spacers)"""
    new_cells: dict[int, TableCell] = {}
    
    for cell_id, cell in tg.cells.items():
        c0, c_end = cell.c0, cell.c0 + cell.colspan - 1
        
        # Case 1: Cell originates in removed column (standalone) -> skip
        if c0 == col_to_remove and cell.colspan == 1:
            continue
        
        # Case 2: Cell spans across removed column (including starting at it) -> reduce colspan
        elif c0 <= col_to_remove <= c_end:
            new_cells[cell_id] = replace(cell, colspan=cell.colspan - 1)
        
        # Case 3: Cell after removed column -> shift left
        elif c0 > col_to_remove:
            new_cells[cell_id] = replace(cell, c0=cell.c0 - 1)
        
        # Case 4: Cell before removed column -> keep as is
        else:
            new_cells[cell_id] = cell
    
    # Rebuild grid properly by expanding cells with their new dimensions
    new_n_cols = tg.n_cols - 1
    new_grid: list[list[int]] = []
    
    for r in range(tg.n_rows):
        new_row: list[int] = [0] * new_n_cols
        
        # Place each cell in the new grid
        for cell_id, cell in new_cells.items():
            if cell.r0 <= r < cell.r0 + cell.rowspan:
                # This cell appears in this row
                for dc in range(cell.colspan):
                    col_idx = cell.c0 + dc
                    if 0 <= col_idx < new_n_cols:
                        new_row[col_idx] = cell_id
        
        new_grid.append(new_row)
    
    return TableGrid(
        table_id=tg.table_id,
        n_rows=tg.n_rows,
        n_cols=new_n_cols,
        cells=new_cells,
        grid=new_grid,
        source=tg.source,
    )


def remove_leading_blank_rows(tg: TableGrid) -> TableGrid:
    """
    Remove completely blank rows from the start of the table.
    A row is blank if all cells in that row have empty text.
    """
    rows_to_remove = 0
    
    # Count how many leading rows are completely blank
    for r in range(tg.n_rows):
        row_cells = set(tg.grid[r])
        is_blank = True
        
        for cell_id in row_cells:
            if cell_id == 0:
                continue
            cell = tg.cells.get(cell_id)
            if cell and cell.text and cell.text.strip():
                is_blank = False
                break
        
        if is_blank:
            rows_to_remove += 1
        else:
            break  # Stop at first non-blank row
    
    if rows_to_remove == 0:
        return tg
    
    # Rebuild cells and grid without the leading blank rows
    new_cells: dict[int, TableCell] = {}
    
    for cell_id, cell in tg.cells.items():
        r0, r_end = cell.r0, cell.r0 + cell.rowspan - 1
        
        # Skip cells that are entirely in the removed rows
        if r_end < rows_to_remove:
            continue
        
        # Adjust cells that span across the boundary
        new_r0 = max(0, cell.r0 - rows_to_remove)
        new_rowspan = cell.rowspan
        
        # If cell started in removed rows but extends into kept rows
        if cell.r0 < rows_to_remove:
            new_rowspan = (r_end - rows_to_remove + 1)
        
        new_cells[cell_id] = replace(cell, r0=new_r0, rowspan=new_rowspan)
    
    # Rebuild grid
    new_n_rows = tg.n_rows - rows_to_remove
    new_grid: list[list[int]] = []
    
    for r in range(new_n_rows):
        new_row: list[int] = [0] * tg.n_cols
        
        for cell_id, cell in new_cells.items():
            if cell.r0 <= r < cell.r0 + cell.rowspan:
                for dc in range(cell.colspan):
                    col_idx = cell.c0 + dc
                    if 0 <= col_idx < tg.n_cols:
                        new_row[col_idx] = cell_id
        
        new_grid.append(new_row)
    
    return TableGrid(
        table_id=tg.table_id,
        n_rows=new_n_rows,
        n_cols=tg.n_cols,
        cells=new_cells,
        grid=new_grid,
        source=tg.source,
    )

def remove_blank_rows(tg: TableGrid) -> TableGrid:
    """
    Remove completely blank rows from anywhere in the table.
    
    A row is considered removable if:
    1. All cells that originate in that row (r0 == row) have empty text
    2. There are no cells from previous rows with rowspan extending into this row
    
    This ensures we don't break rowspan cells by removing rows they depend on.
    """
    rows_to_remove = set()
    
    # Identify which rows are blank and safe to remove
    for r in range(tg.n_rows):
        # Check if all cells originating in this row are blank
        is_blank = True
        has_spanning_cell = False
        
        # First check: are all cells originating in this row blank?
        for cell in tg.cells.values():
            if cell.r0 == r and cell.text and cell.text.strip():
                is_blank = False
                break
        
        if not is_blank:
            continue
        
        # Second check: are there any cells from previous rows spanning into this row?
        for cell in tg.cells.values():
            if cell.r0 < r and cell.r0 + cell.rowspan > r:
                has_spanning_cell = True
                break
        
        # Only remove if blank AND no spanning cells
        if is_blank and not has_spanning_cell:
            rows_to_remove.add(r)
    
    if not rows_to_remove:
        return tg
    
    # Build mapping from old row indices to new row indices
    row_mapping = {}
    new_row_idx = 0
    for old_row_idx in range(tg.n_rows):
        if old_row_idx not in rows_to_remove:
            row_mapping[old_row_idx] = new_row_idx
            new_row_idx += 1
    
    # Rebuild cells with adjusted row positions and spans
    new_cells: dict[int, TableCell] = {}
    
    for cell_id, cell in tg.cells.items():
        r0, r_end = cell.r0, cell.r0 + cell.rowspan - 1
        
        # Skip cells that originate in removed rows (they should be blank anyway)
        if cell.r0 in rows_to_remove:
            continue
        
        # Calculate new r0 and rowspan
        new_r0 = row_mapping[cell.r0]
        
        # Count how many rows this cell spans over in the new grid
        # (excluding removed rows)
        new_rowspan = 0
        for r in range(cell.r0, cell.r0 + cell.rowspan):
            if r not in rows_to_remove:
                new_rowspan += 1
        
        if new_rowspan > 0:  # Only keep cells with positive rowspan
            new_cells[cell_id] = replace(cell, r0=new_r0, rowspan=new_rowspan)
    
    # Rebuild grid
    new_n_rows = tg.n_rows - len(rows_to_remove)
    new_grid: list[list[int]] = []
    
    for r in range(new_n_rows):
        new_row: list[int] = [0] * tg.n_cols
        
        for cell_id, cell in new_cells.items():
            if cell.r0 <= r < cell.r0 + cell.rowspan:
                for dc in range(cell.colspan):
                    col_idx = cell.c0 + dc
                    if 0 <= col_idx < tg.n_cols:
                        new_row[col_idx] = cell_id
        
        new_grid.append(new_row)
    
    return TableGrid(
        table_id=tg.table_id,
        n_rows=new_n_rows,
        n_cols=tg.n_cols,
        cells=new_cells,
        grid=new_grid,
        source=tg.source,
    )


def normalize_tablegrid(tg: TableGrid) -> tuple[TableGrid, list[ColOperation]]:
    """
    Apply normalization operations:
    0. Remove leading blank rows
    1. Remove all blank rows (no spanning cells into them)
    2. Merge currency columns into adjacent data columns
    3. Merge paren columns into adjacent data columns
    4. Remove blank spacer columns
    
    Returns normalized grid and list of operations performed.
    """
    operations: list[ColOperation] = []
    
    # First, remove any leading blank rows
    current = remove_leading_blank_rows(tg)
    
    # Then remove all other blank rows throughout the table
    current = remove_blank_rows(current)
    
    # We need to re-profile after each change since column indices shift
    while True:
        profiles = profile_columns(current)
        made_change = False
        
        # Pass 1: Merge currency columns (into right neighbor)
        for p in profiles:
            if not p.currency_only:
                continue
            target = find_next_data_col(profiles, p.idx, direction=1)
            if target is None:
                continue
            
            current = merge_column_content(current, p.idx, target, prefix=True)
            operations.append(ColOperation("merge", p.idx, target, "currency"))
            made_change = True
            break  # Re-profile after change
        
        if made_change:
            continue
        
        # Pass 2: Merge left-paren columns (into right neighbor)
        for p in profiles:
            if not p.lparen_only:
                continue
            target = find_next_data_col(profiles, p.idx, direction=1)
            if target is None:
                continue
            
            current = merge_column_content(current, p.idx, target, prefix=True)
            operations.append(ColOperation("merge", p.idx, target, "lparen"))
            made_change = True
            break
        
        if made_change:
            continue
        
        # Pass 3: Merge right-paren columns (into left neighbor)
        for p in profiles:
            if not p.rparen_only:
                continue
            target = find_next_data_col(profiles, p.idx, direction=-1)
            if target is None:
                continue
            
            current = merge_column_content(current, p.idx, target, prefix=False)
            operations.append(ColOperation("merge", p.idx, target, "rparen"))
            made_change = True
            break
        
        if made_change:
            continue
        
        # Pass 4: Remove blank columns
        for p in profiles:
            if not p.all_blank:
                continue
            
            current = remove_column(current, p.idx)
            operations.append(ColOperation("remove", p.idx, None, "blank_spacer"))
            made_change = True
            break
        
        if not made_change:
            break  # No more changes to make
    
    # Update source metadata
    current.source = {
        "type": "normalized",
        "from": tg.table_id,
        "operations": [
            {"action": op.action, "source_col": op.source_col, "target_col": op.target_col, "reason": op.reason}
            for op in operations
        ],
    }
    
    return current, operations


# =========================
# Main API Functions
# =========================

def extract_tables_from_html(
    html: str,
    min_rows: int = 3,
    page_number: int = 0,
    remove_single_row_tables: bool = True,
) -> pd.DataFrame:
    """
    Extract all tables from HTML and return as a normalized cell-based DataFrame.
    
    Args:
        html: HTML string containing tables
        min_rows: Minimum number of rows required (default 3, skips small tables)
        page_number: Page number this HTML appears on (default 0)
        remove_single_row_tables: If True, skip tables with only 1 row (default True)
    
    Returns:
        DataFrame with one row per cell, containing columns:
        - page_number: Page number (0-indexed)
        - table_id: Table counter (1-indexed)
        - table_cell_id: Cell ID within the table
        - row_start: Starting row position (0-indexed)
        - col_start: Starting column position (0-indexed)
        - rowspan: Number of rows the cell spans
        - colspan: Number of columns the cell spans
        - text: Cell text content
        - role: Cell role (header, row_label, data)
        - ix: XBRL metadata (stringified dict or None)
        - style: CSS style string (or None)
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    
    if not tables:
        return pd.DataFrame()
    
    all_cells = []
    table_counter = 1  # 1-indexed
    
    for idx, table_elem in enumerate(tables):
        try:
            # Create HTML snippet for just this table
            table_html = str(table_elem)
            
            # Parse table structure
            standard_grid = parse_html_to_tablegrid(table_html, table_index=0)
            
            # Skip tables with too few rows
            if standard_grid.n_rows < min_rows:
                continue
            
            # Normalize (merge currency, remove spacers, etc.)
            normalized_grid, operations = normalize_tablegrid(standard_grid)
            
            # Skip single-row tables if requested (after normalization to get true row count)
            if remove_single_row_tables and normalized_grid.n_rows == 1:
                continue
            
            # Detect cell roles (header, row_label, data)
            normalized_grid = detect_cell_roles(normalized_grid)
            
            # Convert to normalized cell-based DataFrame
            df = normalized_grid.to_cells_dataframe(
                page_number=page_number,
                table_id=table_counter
            )
            
            all_cells.append(df)
            table_counter += 1
            
        except Exception as e:
            # Log error but continue with other tables
            print(f"Warning: Failed to parse table at position {idx}: {e}")
            continue
    
    if not all_cells:
        return pd.DataFrame()
    
    return pd.concat(all_cells, ignore_index=True)

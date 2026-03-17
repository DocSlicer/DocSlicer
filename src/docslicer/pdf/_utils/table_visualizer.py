"""
csv_to_markdown_tables.py

Turns a CSV with table cells into pretty Markdown tables.

Expected columns:
- table_cell_id
- page_number
- layout_id
- table_id
- r0
- c0
- rowspan
- colspan
- text
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import math

# ====== CONFIG ======
INPUT_CSV  = Path("backend/app/services/parsing/pdf/test_output/table_cells.csv")       # <- change to your actual CSV
OUTPUT_MD  = Path("backend/app/services/parsing/pdf/test_output/tables_output.md") # <- or None to just print
ENCODING   = "utf-8"
MAX_CELL_LEN = 50  # <- tweak as you like
# =====================


def _safe_text(val) -> str:
    """Convert cell text to a single-line, Markdown-safe string with truncation."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""

    s = str(val).replace("\n", " ").replace("|", r"\|").strip()

    if len(s) <= MAX_CELL_LEN:
        return s

    words = s.split()
    if len(words) <= 1:
        # single extremely long word → just truncate
        return s[: MAX_CELL_LEN - 3].rstrip() + "..."

    last_word = words[-1]

    # space left for " ... " + last_word
    keep_len = MAX_CELL_LEN - (len(last_word) + 5)
    keep_len = max(10, keep_len)  # never truncate ridiculously short

    prefix = s[:keep_len].rstrip()

    return f"{prefix} ... {last_word}"


def build_grid_for_table(df: pd.DataFrame) -> List[List[str]]:
    """
    Build a 2D grid (list of list of strings) for a single table_id.

    - r0 is assumed 0-based row index.
    - c0 is assumed 1-based column index.
    - colspan/rowspan: we put the text only in the top-left cell of the span.
      Markdown can't truly do spans, so we keep other cells blank.
    """
    if df.empty:
        return []

    # Ensure numeric types where needed
    for col in ["row_start", "col_start", "rowspan", "colspan"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)

    max_row = int((df["row_start"] + df["rowspan"] - 1).max())
    max_col = int((df["col_start"] + df["colspan"] - 1).max())

    # Initialize grid with empty strings
    grid: List[List[str]] = [["" for _ in range(max_col)] for _ in range(max_row + 1)]

    # Fill grid
    for _, row in df.iterrows():
        r0 = int(row["row_start"])
        c0 = int(row["col_start"]) - 1  # c0 is 1-based in input; convert to 0-based
        text = _safe_text(row.get("text", ""))

        if not text:
            continue

        # Only place text in the top-left cell of the span
        existing = grid[r0][c0]
        if existing:
            grid[r0][c0] = existing + " " + text
        else:
            grid[r0][c0] = text

        # We *do not* fill the rest of the span for Markdown (no real colspan support).
        # They stay as "", but we still keep them as physical columns so vertical
        # lines line up correctly.

    # Trim trailing completely empty rows at the bottom
    while grid and all(cell == "" for cell in grid[-1]):
        grid.pop()

    # If grid is empty after trimming, return as-is
    if not grid:
        return grid

    return grid


def compute_column_widths(grid: List[List[str]]) -> List[int]:
    """Compute the max width for each column across all rows."""
    if not grid:
        return []

    num_cols = max(len(row) for row in grid)
    widths = [0] * num_cols

    for row in grid:
        # pad row if shorter
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    return widths


def render_markdown_table(grid: List[List[str]]) -> str:
    """Render a 2D grid as a Markdown table with aligned columns."""
    if not grid:
        return ""

    widths = compute_column_widths(grid)
    num_cols = len(widths)

    # Ensure each row has correct number of columns
    normalized_rows: List[List[str]] = []
    for row in grid:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        normalized_rows.append(row)

    header = normalized_rows[0]
    body = normalized_rows[1:]

    lines: List[str] = []

    # Header row
    header_cells = [header[i].ljust(widths[i]) for i in range(num_cols)]
    lines.append("| " + " | ".join(header_cells) + " |")

    # Separator row (---) sized by column width (min 3 chars)
    sep_cells = ["-" * max(3, widths[i]) for i in range(num_cols)]
    lines.append("| " + " | ".join(sep_cells) + " |")

    # Body rows
    for row in body:
        cells = [row[i].ljust(widths[i]) for i in range(num_cols)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def tables_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert entire CSV (potentially multiple tables) to Markdown.

    Groups by table_id, sorted by (page_number, layout_id, table_id) if available.
    """
    if "table_id" not in df.columns:
        raise ValueError("CSV must have a 'table_id' column.")

    # Sort for deterministic output
    sort_cols = [col for col in ["page_number", "layout_id", "table_id"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    md_chunks: List[str] = []

    for table_id, group in df.groupby("table_id"):
        grid = build_grid_for_table(group)

        # Skip empty grids
        if not grid:
            continue

        md_chunks.append(f"### Table {table_id}\n")
        md_chunks.append(render_markdown_table(grid))
        md_chunks.append("")  # blank line after each table

    return "\n".join(md_chunks).rstrip()  # trim trailing newlines


def main() -> None:
    df = pd.read_csv(INPUT_CSV, encoding=ENCODING)
    markdown = tables_to_markdown(df)

    if OUTPUT_MD is None:
        print(markdown)
    else:
        OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_MD.write_text(markdown, encoding=ENCODING)
        print(f"Markdown written to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

"""
Canonical cross-format DataFrame schemas.

df_table_cells is produced independently by four pipelines (docx step 03,
pptx step 04, html step 05, pdf step 14) but consumed by format-agnostic
code downstream (_orchestrator._build_tables, df_export), so its schema is
defined once here. Each builder calls conform_table_cells() as its final
step instead of keeping a local column list.
"""
from __future__ import annotations

import pandas as pd

# One row per logical table cell. Columns a format cannot produce are
# NA-filled by conform_table_cells (e.g. caption outside docx, layout_id
# outside pdf).
TABLE_CELLS_COLS = [
    # Produced by individual table builders
    "page_number",
    "page_label",
    "layout_id",        # pdf — layout region the table belongs to
    "table_id",
    "table_row_id",     # docx/html/pptx — source-format row element id
    "table_cell_id",
    "caption",          # docx — caption paragraph adjacent to the table
    "row_start",
    "col_start",
    "rowspan",
    "colspan",
    "table_header_flag",
    "text",
    "char_count",
    "bold_ratio",
    "is_bold",
    # Produced by shared table_utils
    "table_cell_role",
]

# detect_cell_roles diagnostics — not part of the contract, passed through
# only when debug=True.
_DEBUG_PREFIX = "hdr_"
_DEBUG_COLS = frozenset({"table_row_style"})


def conform_table_cells(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Project *df* onto the canonical df_table_cells schema: canonical columns
    in canonical order, NA where the producing format has no value, anything
    else dropped. With debug=True, detect_cell_roles diagnostic columns
    (table_row_style, hdr_*) present in *df* are appended after the
    canonical block.
    """
    out = df.copy()
    for col in TABLE_CELLS_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    cols = list(TABLE_CELLS_COLS)
    if debug:
        cols += [
            c for c in df.columns
            if (c in _DEBUG_COLS or c.startswith(_DEBUG_PREFIX)) and c not in cols
        ]
    return out[cols].reset_index(drop=True)

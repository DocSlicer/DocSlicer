# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Canonical cross-format table-cells column reference.

df_table_cells is produced independently by four pipelines (docx step 04,
pptx step 04, html step 05, pdf step 14) but consumed by format-agnostic
code downstream (_orchestrator._build_tables, df_export). The consumer reads
columns defensively, so these lists are a shared *reference* for what each
column means and which format produces it — not an enforced schema.
"""
from __future__ import annotations

from .table_header import _ROW_DECISION_COLS, _ROW_FEATURE_COLS

# One row per logical table cell. Not every format produces every column;
# downstream code guards each access with `col in df.columns` / `.get()`.
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

# Optional header-detection diagnostics that detect_cell_roles /
# assign_header_features add on top of the canonical block (table_row_style +
# hdr_* features, then hdr_score / hdr_decision). Present only when the header
# pass runs; kept here so a table-cells dump documents every column it may show.
# Sourced from table_header so the two never drift.
TABLE_CELLS_DEBUG_COLS = [*_ROW_FEATURE_COLS, *_ROW_DECISION_COLS]

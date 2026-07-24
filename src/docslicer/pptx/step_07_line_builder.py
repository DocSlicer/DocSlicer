# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
PPTX paragraph -> line adapter.

Most PPTX paragraphs map 1:1 to a "line" row. The exception is table content:
paragraphs that belong to the same table row are collapsed into one
pipe-delimited line so downstream shared stages see a row-shaped table
surface, consistent with the DOCX and HTML pipelines.

Reading order is assigned upstream by step_06_reading_order (order_index);
this step only sorts on it and turns paragraphs into lines.
"""

from __future__ import annotations

import pandas as pd

from .._utils.df_aggregation.registry_aggregator import Agg, aggregate_to
from .._utils.df_aggregation.text_merge import (
    merge_table_rows,
    merge_text_within_line,
)


def _has_value(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _assign_line_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign an integer ``line_id``: one per paragraph, except one per table row
    (paragraphs sharing a table row collapse into a single line).

    IDs are handed out in first-appearance order, so after the ``order_index``
    sort in :func:`build_lines` they follow reading order.
    """
    out = df.copy()

    has_table_row = (
        out.get("table_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
        & out.get("table_row_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
    )

    paragraph_id = out.get("paragraph_id", pd.Series(range(1, len(out) + 1), index=out.index))
    table_id = out.get("table_id", pd.Series(pd.NA, index=out.index))
    table_row_id = out.get("table_row_id", pd.Series(pd.NA, index=out.index))

    key = "p:" + paragraph_id.astype(str)
    key = key.mask(
        has_table_row,
        "t:" + table_id.astype(str) + ":r:" + table_row_id.astype(str),
    )

    out["line_id"] = pd.factorize(key, sort=False)[0] + 1
    return out


def _build_line_text(df: pd.DataFrame) -> pd.Series:
    """
    One text string per ``line_id`` (Series indexed by line_id).

    Non-table lines are a single paragraph, so the text is just that
    paragraph's. Table-row lines join paragraphs within a cell with spaces,
    then pipe-join the cells so downstream shared stages see a row-shaped
    surface, matching HTML/PDF/DOCX.
    """
    texts = df["text"].fillna("").astype(str).str.strip()

    line_text = merge_text_within_line(texts, df["line_id"])

    if "table_id" not in df.columns:
        return line_text

    is_table = df["table_id"].map(_has_value)
    if not is_table.any():
        return line_text

    tbl = df[is_table]
    if "table_cell_id" in tbl.columns:
        cell_text = merge_text_within_line(texts[is_table], tbl["table_cell_id"])
        cells = tbl.drop_duplicates("table_cell_id")[["table_cell_id", "line_id"]].copy()
        cells["text"] = cells["table_cell_id"].map(cell_text)
        cells = cells.sort_values(["line_id", "table_cell_id"], kind="mergesort")
        table_text = merge_table_rows(cells["text"], cells["line_id"])
    else:
        table_text = merge_table_rows(texts[is_table], df.loc[is_table, "line_id"])

    line_text.loc[table_text.index] = table_text
    return line_text


# Style columns compared line-to-line for layout grouping (x_left is compared
# separately, rounded to whole points).
_LAYOUT_STYLE_COLS = [
    "font_family",
    "font_size",
    "text_align",
    "non_stroking_color",
    "is_bold",
    "is_italic",
    "is_uppercase",
]


def _neq(cur: pd.Series, prv: pd.Series) -> pd.Series:
    """NaN-safe inequality: NaN == NaN counts as equal (no change)."""
    return ~((cur == prv) | (cur.isna() & prv.isna()))


def _add_layout_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a layout_id column grouping consecutive lines into layouts.

    Rules:
    1. Start at 1 for the very first line.
    2. New layout whenever page_number changes.
    3. Grouped lines share one layout_id: a table (same table_id) or a chart
       (same chart_id) is always one layout, regardless of shape_id.
    4. Ungrouped lines: a new shape_id, or a change in block_type (e.g. a
       shape mixing normal text and math), always starts a new layout.
       Within the same shape_id and block_type, consecutive lines keep the
       same layout only if either
       - they share the same list_num_id AND list_level, or
       - they share the same style (see _LAYOUT_STYLE_COLS). x_left is not
         part of this check: all lines within one shape already share the
         same x_left.
       Otherwise a new layout starts.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    prev = df.shift(1)
    false = pd.Series(False, index=df.index)

    # Rule 2: page change always breaks (and splits any group spanning pages).
    if "page_number" in df.columns:
        new_page = _neq(df["page_number"], prev["page_number"])
    else:
        new_page = false

    # Rule 3: table/chart group key — table_id wins over chart_id.
    group_key = pd.Series(pd.NA, index=df.index, dtype=object)
    if "chart_id" in df.columns:
        has_chart = df["chart_id"].notna()
        group_key = group_key.mask(has_chart, "chart_" + df["chart_id"].astype(str))
    if "table_id" in df.columns:
        has_table = df["table_id"].notna()
        group_key = group_key.mask(has_table, "table_" + df["table_id"].astype(str))

    prev_key = group_key.shift(1)
    grouped = group_key.notna()
    group_break = grouped & _neq(group_key, prev_key)

    # Rule 4: ungrouped lines break on a new shape_id or a block_type change,
    # or (within the same shape/block_type) neither the list nor the style
    # criteria match the previous line.
    shape_change = _neq(df["shape_id"], prev["shape_id"]) if "shape_id" in df.columns else false
    block_type_change = _neq(df["block_type"], prev["block_type"]) if "block_type" in df.columns else false

    # Both lines must actually be list items — two non-list lines both having
    # a null list_num_id/list_level is not a "match" (NaN-safe _neq would
    # otherwise treat that as equal and short-circuit every style check for
    # the vast majority of non-list PPTX paragraphs).
    list_match = false.copy()
    if "list_num_id" in df.columns and "list_level" in df.columns:
        has_list = df["list_num_id"].notna() & df["list_level"].notna()
        prev_has_list = prev["list_num_id"].notna() & prev["list_level"].notna()
        same_list = ~_neq(df["list_num_id"], prev["list_num_id"]) & ~_neq(
            df["list_level"], prev["list_level"]
        )
        list_match = has_list & prev_has_list & same_list

    style_match = pd.Series(True, index=df.index)
    for col in _LAYOUT_STYLE_COLS:
        if col in df.columns:
            style_match &= ~_neq(df[col], prev[col])

    ungrouped_break = ~grouped & (
        prev_key.notna() | shape_change | block_type_change | ~(list_match | style_match)
    )

    breaks = (new_page | group_break | ungrouped_break).astype(bool)
    breaks.iloc[0] = True  # Rule 1
    df["layout_id"] = breaks.cumsum()
    return df


def build_lines(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build shared-compatible PPTX lines from paragraph rows.

    Non-table paragraphs are one line each. Paragraphs inside the same table
    row are aggregated into one line with pipe-delimited cell text.
    """
    if paragraph_df is None or paragraph_df.empty:
        return pd.DataFrame()

    # Guarantee reading order before grouping: line_id and the sort=False
    # groupby in aggregate_to both key off first-appearance order, so a stable
    # sort by order_index here is what makes line_id follow reading order.
    if "order_index" in paragraph_df.columns:
        paragraph_df = paragraph_df.sort_values("order_index", kind="stable")

    working = _assign_line_id(paragraph_df)
    line_text = _build_line_text(working)

    # Registry-driven aggregation: every column's policy comes from
    # COLUMN_REGISTRY. Only table_cell_id is line-specific — a line spans
    # several cells, so collect them as a list (registry default is "first"
    # for cells). text is DROP in the registry and rebuilt above, so it is
    # mapped back after.
    lines_df = aggregate_to(
        working,
        by="line_id",
        size_as="paragraph_count",
        overrides={"table_cell_id": Agg.UNIQUE_LIST},
        on_unknown="raise",
    )

    lines_df["text"] = lines_df["line_id"].map(line_text)

    lines_df = _add_layout_id(lines_df)

    return lines_df


__all__ = ["build_lines"]

"""
DOCX paragraph -> line adapter.

Most DOCX paragraphs map 1:1 to shared "line" rows. The main exception is
table content: paragraphs that belong to the same Word table row are collapsed
into one pipe-delimited line so downstream shared stages see a row-shaped table
surface, similar to HTML/PDF line data.
"""

from __future__ import annotations

import pandas as pd

from docslicer._utils.df_aggregation.registry_aggregator import Agg, aggregate_to
from docslicer._utils.df_aggregation.text_merge import (
    merge_table_rows,
    merge_text_within_line,
)


def _has_value(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _assign_line_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign an integer ``line_id``: one per paragraph, except one per Word table
    row (paragraphs sharing a table row collapse into a single line).

    IDs are handed out in first-appearance order, so after the ``order_index``
    sort in :func:`build_lines` they follow document order.
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

    Non-table lines are a single paragraph, so the text is just that paragraph's
    (script/strike markup already baked in by the paragraph builder). Table-row
    lines join paragraphs within a cell with spaces, then pipe-join the cells so
    downstream shared stages see a row-shaped surface, matching HTML/PDF.
    """
    texts = df["text"].fillna("").astype(str).str.strip()

    # Default: space-join paragraphs on a line. For non-table content there is
    # exactly one paragraph per line, so this returns the paragraph's own text.
    line_text = merge_text_within_line(texts, df["line_id"])

    if "table_id" not in df.columns:
        return line_text

    is_table = df["table_id"].map(_has_value)
    if not is_table.any():
        return line_text

    tbl = df[is_table]
    if "table_cell_id" in tbl.columns:
        # Cell text first: paragraphs within one cell joined by space.
        cell_text = merge_text_within_line(texts[is_table], tbl["table_cell_id"])
        cells = tbl.drop_duplicates("table_cell_id")[["table_cell_id", "line_id"]].copy()
        cells["text"] = cells["table_cell_id"].map(cell_text)
        cells = cells.sort_values(["line_id", "table_cell_id"], kind="mergesort")
        table_text = merge_table_rows(cells["text"], cells["line_id"])
    else:
        table_text = merge_table_rows(texts[is_table], df.loc[is_table, "line_id"])

    line_text.loc[table_text.index] = table_text
    return line_text


# block_type values whose consecutive lines are held together in one layout
# (but never mixed: a run of footnotes and an adjacent run of footers stay
# in separate layouts because their keys differ).
_LAYOUT_GROUP_BLOCK_TYPES = frozenset({"footnote", "footer", "header"})


def _layout_group_key(df: pd.DataFrame) -> pd.Series:
    """
    Per-line grouping key: consecutive lines sharing a non-null key are kept in
    one layout by :func:`_add_layout_id`. Ungrouped lines return NA and each get
    their own layout.

    Precedence (highest last so it wins the mask): special block_type
    (footnote/footer/header) < list (same list_num_id AND list_level) < table
    (same table_id).
    """
    keys = pd.Series(pd.NA, index=df.index, dtype=object)

    if "block_type" in df.columns:
        block_type = df["block_type"]
        is_special = block_type.isin(_LAYOUT_GROUP_BLOCK_TYPES)
        keys = keys.mask(is_special, "block:" + block_type.astype(str))

    if "list_num_id" in df.columns and "list_level" in df.columns:
        has_list = df["list_num_id"].map(_has_value) & df["list_level"].map(_has_value)
        list_key = "list:" + df["list_num_id"].astype(str) + ":" + df["list_level"].astype(str)
        keys = keys.mask(has_list, list_key)

    if "table_id" in df.columns:
        has_table = df["table_id"].map(_has_value)
        keys = keys.mask(has_table, "table:" + df["table_id"].astype(str))

    return keys


def _add_layout_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a layout_id column grouping consecutive lines into layouts.

    Rules:
    1. Start at 1 for the very first line.
    2. New layout whenever page_number changes.
    3. Grouped lines (see :func:`_layout_group_key`) share one layout_id across a
       consecutive run of the same key — a table (same table_id), a list (same
       list_num_id AND list_level), or a header/footer/footnote run (same
       block_type). Entering, leaving, or switching a group key breaks.
    4. Ungrouped lines get their own layout_id each.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    def _neq(cur: pd.Series, prv: pd.Series) -> pd.Series:
        """NaN-safe inequality: NaN == NaN counts as equal (no break)."""
        return ~((cur == prv) | (cur.isna() & prv.isna()))

    false = pd.Series(False, index=df.index)

    # Rule 2: page change always breaks (and splits any group spanning pages).
    if "page_number" in df.columns:
        new_page = _neq(df["page_number"], df["page_number"].shift(1))
    else:
        new_page = false

    # Rules 3 & 4: grouped lines break only when the key changes; every
    # ungrouped line (NA key) starts a new layout.
    group_key = _layout_group_key(df)
    grouped = group_key.notna()
    group_break = grouped & _neq(group_key, group_key.shift(1))
    ungrouped_break = ~grouped

    breaks = (new_page | group_break | ungrouped_break).astype(bool)
    breaks.iloc[0] = True  # Rule 1
    df["layout_id"] = breaks.cumsum()
    return df


def build_lines(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build shared-compatible DOCX lines from paragraph rows.

    Non-table paragraphs are one line each. Paragraphs inside the same table row
    are aggregated into one line with pipe-delimited cell text.
    """
    if paragraph_df is None or paragraph_df.empty:
        return pd.DataFrame()

    # Guarantee document order before grouping: line_id and the sort=False
    # groupby in aggregate_to both key off first-appearance order, so a stable
    # sort by order_index here is what makes line_id follow document order
    # (rather than relying on paragraph_df arriving pre-ordered upstream).
    if "order_index" in paragraph_df.columns:
        paragraph_df = paragraph_df.sort_values("order_index", kind="stable")

    working = _assign_line_id(paragraph_df)
    line_text = _build_line_text(working)

    # Registry-driven aggregation: every column's policy comes from
    # COLUMN_REGISTRY. Only table_cell_id is line-specific — a line spans several
    # cells, so collect them as a list (registry default is "first" for cells).
    # text is DROP in the registry and rebuilt above, so it is mapped back after.
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

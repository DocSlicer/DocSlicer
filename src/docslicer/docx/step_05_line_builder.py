"""
DOCX paragraph -> line adapter.

Most DOCX paragraphs map 1:1 to shared "line" rows. The main exception is
table content: paragraphs that belong to the same Word table row are collapsed
into one pipe-delimited line so downstream shared stages see a row-shaped table
surface, similar to HTML/PDF line data.
"""

from __future__ import annotations

import pandas as pd

from docslicer._utils.hierarchical_aggregator import (
    _collect_unique_list,
    aggregate_hierarchical,
    build_standard_agg_spec,
)


_DOCX_IDENTITY_COLS = [
    "page_number",
    "page_label",
    "page_width",
    "page_height",
    "section_id",
    "header_footer_type",
    "source_part",
    "source_part_id",
    "text_align",
    "num_id",
    "list_level",
    "list_label",
    "outline_level",
    "page_break_before",
    "section_break_type",
    "bookmark_ids",
    "bookmark_names",
    "comment_id",
    "footnote_id",
    "endnote_id",
    "paragraph_style_id",
    "paragraph_style_name",
    "effective_paragraph_style_id",
    "effective_paragraph_style_name",
    "style_id",
    "style_name",
    "character_style_id",
    "character_style_name",
    "effective_character_style_id",
    "effective_character_style_name",
    "block_type",
]


def _has_value(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _add_line_group_key(df: pd.DataFrame) -> pd.DataFrame:
    """Assign one group per paragraph, except one group per table row."""
    out = df.copy()

    has_table_row = (
        out.get("table_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
        & out.get("table_row_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
    )

    paragraph_id = out.get("paragraph_id", pd.Series(range(1, len(out) + 1), index=out.index))
    table_id = out.get("table_id", pd.Series(pd.NA, index=out.index))
    table_row_id = out.get("table_row_id", pd.Series(pd.NA, index=out.index))

    out["_line_group_key"] = "p:" + paragraph_id.astype(str)
    out.loc[has_table_row, "_line_group_key"] = (
        "t:"
        + table_id.loc[has_table_row].astype(str)
        + ":r:"
        + table_row_id.loc[has_table_row].astype(str)
    )
    return out


def _create_line_text(df: pd.DataFrame) -> dict[str, str]:
    """Build text for each line group, pipe-delimiting table-row cells."""
    if "_line_group_key" not in df.columns or "text" not in df.columns:
        return {}

    text_map: dict[str, str] = {}
    working = df.copy()
    working["text"] = working["text"].fillna("").astype(str).str.strip()

    for group_key, group in working.groupby("_line_group_key", sort=False):
        table_id = group["table_id"].iloc[0] if "table_id" in group.columns else None
        has_table = _has_value(table_id)

        if not has_table:
            text_map[group_key] = " ".join(t for t in group["text"].tolist() if t).strip()
            continue

        sort_cols = [col for col in ["table_cell_id", "paragraph_id"] if col in group.columns]
        ordered = group.sort_values(sort_cols, kind="mergesort") if sort_cols else group

        if "table_cell_id" in ordered.columns:
            cell_texts = (
                ordered.groupby("table_cell_id", sort=False)["text"]
                .agg(lambda parts: " ".join(p for p in parts if p).strip())
                .tolist()
            )
        else:
            cell_texts = ordered["text"].tolist()

        text_map[group_key] = " | ".join(t for t in cell_texts if t).strip()

    return text_map


def _add_layout_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add layout_id column using the same reasoning as the HTML line builder.

    Rules:
    1. Start at 1 for the very first line.
    2. Increment when page_number changes.
    3. Rows with same table_id get same layout_id.
    4. For non-table lines, increment for every new line.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    layout_ids = []
    current_layout_id = 1
    prev_page = None
    prev_table_id = None

    for _, row in df.iterrows():
        page = row.get("page_number")
        table_id = row.get("table_id")

        if prev_page is not None and page != prev_page:
            current_layout_id += 1
            prev_table_id = None

        has_table = pd.notna(table_id)

        if has_table:
            if table_id != prev_table_id:
                if prev_page is not None:
                    current_layout_id += 1
                prev_table_id = table_id
        else:
            if prev_table_id is not None:
                current_layout_id += 1
                prev_table_id = None
            else:
                if prev_page is not None:
                    current_layout_id += 1

        layout_ids.append(current_layout_id)
        prev_page = page

    df["layout_id"] = layout_ids
    return df


def build_lines(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build shared-compatible DOCX lines from paragraph rows.

    Non-table paragraphs are one line each. Paragraphs inside the same table row
    are aggregated into one line with pipe-delimited cell text.
    """
    if paragraph_df is None or paragraph_df.empty:
        return pd.DataFrame()

    working = _add_line_group_key(paragraph_df)
    working["_paragraph_count_marker"] = 1
    line_text_map = _create_line_text(working)

    agg_spec = build_standard_agg_spec(
        identity_cols=_DOCX_IDENTITY_COLS,
        include_geometry=False,
        include_hierarchy=False,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,
        extra_first=[
            "is_deleted_revision",
            "is_inserted_revision",
            "section_break_after",
        ],
        extra_agg={
            "paragraph_id": _collect_unique_list,
            "table_cell_id": _collect_unique_list,
            "hyperlink_url": _collect_unique_list,
        },
        count_col="_paragraph_count_marker",
    )

    lines_df = aggregate_hierarchical(
        df=working,
        group_col="_line_group_key",
        agg_spec=agg_spec,
        rename_count_col={"_paragraph_count_marker": "paragraph_count"},
        compute_derived=True,
    )

    lines_df["text"] = lines_df["_line_group_key"].map(line_text_map)
    lines_df.insert(0, "line_id", range(1, len(lines_df) + 1))
    lines_df = lines_df.drop(columns=["_line_group_key"])

    #if "layout_type" not in lines_df.columns:
     #   lines_df["layout_type"] = "docx"

    lines_df = _add_layout_id(lines_df)
    return lines_df

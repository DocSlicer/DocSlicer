"""
DOCX run → paragraph aggregator.

Filters text runs and merges them by paragraph_id, producing one row per logical
paragraph with concatenated text and character-count-weighted style.
"""

from __future__ import annotations

import pandas as pd

from docslicer._utils.hierarchical_aggregator import (
    ALPHA_WEIGHTED_STYLE,
    _collect_unique_list,
    aggregate_hierarchical,
    build_standard_agg_spec,
)

# Identity columns that are constant across all runs in a paragraph.
# text_orientation is intentionally excluded — include_style=True picks it up
# via ALPHA_WEIGHTED_STYLE (correct when table cells have mixed orientations).
_PARA_IDENTITY_COLS = [
    "page_number",
    "page_label",
    "page_width",
    "page_height",
    "section_id",
    "header_footer_type",
    "source_part",
    "source_part_id",
    "text_align",
    "table_id",
    "table_row_id",
    "table_cell_id",
    "nested_table_depth",
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
]


def _paragraph_text(run_df: pd.DataFrame) -> pd.Series:
    """
    Build paragraph text from the full run DataFrame (all run_types).

    text runs contribute their content; tab runs become a single space so that
    text–tab–text patterns produce a space separator rather than a direct concat.
    All other run types (field_marker, field_code, page_break, …) are discarded.
    Leading/trailing whitespace is stripped from the result.
    """
    mask = run_df["run_type"].isin({"text", "tab"})
    working = run_df.loc[mask, ["paragraph_id", "order_index", "run_type", "text"]].copy()
    working["text"] = working["text"].fillna("").astype(str)
    working.loc[working["run_type"] == "tab", "text"] = " "
    working = working.sort_values(["paragraph_id", "order_index"])
    result = working.groupby("paragraph_id", sort=False)["text"].agg("".join)
    return result.str.strip()


def build_paragraphs(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate run-level rows into paragraph-level rows.

    Keeps only ``run_type == "text"`` runs, then groups by ``paragraph_id``.
    Text is joined in document order (no separator). Style is weighted by
    character count; bold/italic/underlined ratios are recomputed from sums.

    Args:
        run_df: Output of ``extract_runs``.

    Returns:
        One row per paragraph that contained at least one text run.
    """
    if run_df.empty:
        return pd.DataFrame()

    text_runs = run_df[run_df["run_type"] == "text"].copy()
    if text_runs.empty:
        return pd.DataFrame()

    # char_count / alpha_count unlock ALPHA_WEIGHTED_STYLE fast path and correct
    # bold/italic/underlined ratio recomputation inside aggregate_hierarchical.
    text_runs["char_count"] = text_runs["text"].fillna("").str.len()
    text_runs["alpha_count"] = text_runs["text"].fillna("").str.count(r"[a-zA-Z]")

    # Pre-compute paragraph text from the full run_df so tabs between text runs
    # produce a space separator (text_runs alone has already lost the tab rows).
    para_text = _paragraph_text(run_df)

    agg_spec = build_standard_agg_spec(
        identity_cols=_PARA_IDENTITY_COLS,
        include_geometry=False,
        include_hierarchy=False,
        include_style=True,    # font_size, font_name, non_stroking_color, text_orientation
        include_counts=True,   # char_count/alpha_count summed; _bold/_italic/_underlined_char_est
        include_metadata=False,
        include_table=False,
        extra_first=["style_id", "style_name"],
        extra_agg={
            # paragraph-level flags that should OR across runs
            "section_break_after": "max",
            "is_toc_field": "max",
            "is_deleted_revision": "max",
            "is_inserted_revision": "max",
            # a paragraph can span multiple hyperlinks
            "hyperlink_url": _collect_unique_list,
            # character style: pick from the run with the most alphabetic content
            "character_style_id": ALPHA_WEIGHTED_STYLE,
            "character_style_name": ALPHA_WEIGHTED_STYLE,
            "effective_character_style_id": ALPHA_WEIGHTED_STYLE,
            "effective_character_style_name": ALPHA_WEIGHTED_STYLE,
        },
        count_col="run_id",
    )

    para_df = aggregate_hierarchical(
        df=text_runs,
        group_col="paragraph_id",
        agg_spec=agg_spec,
        rename_count_col={"run_id": "run_count"},
        compute_derived=True,
    )

    para_df["text"] = para_df["paragraph_id"].map(para_text)

    return para_df

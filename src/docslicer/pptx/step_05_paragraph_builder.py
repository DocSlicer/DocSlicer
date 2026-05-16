"""
PPTX run → paragraph aggregator.

Filters content runs and merges them by paragraph_id, producing one row per
logical paragraph with concatenated text and character-count-weighted style.

Speaker notes are included by default (include_notes=True) and can be
excluded by passing include_notes=False, which filters out runs where
header_footer_type == "notes".
"""

from __future__ import annotations

import pandas as pd

from .._utils.hierarchical_aggregator import (
    ALPHA_WEIGHTED_STYLE,
    _collect_unique_list,
    aggregate_hierarchical,
    build_standard_agg_spec,
)


# Identity columns that are constant across all runs in a paragraph.
_PARA_IDENTITY_COLS = [
    "page_number",
    "slide_index",
    "header_footer_type",
    "source_part",
    "source_part_id",
    "text_align",
    "table_id",
    "table_row_id",
    "table_cell_id",
    "list_num_id",
    "list_level",
    "list_label",
    "outline_level",
    "shape_id",
    "shape_name",
    "shape_type",
    "placeholder_type",
]

# Run types that represent self-contained paragraph content.
_CONTENT_RUN_TYPES = frozenset({"text", "image_ref", "shape_ref", "chart_ref", "graphic_ref"})


def _paragraph_text(run_df: pd.DataFrame) -> pd.Series:
    """
    Build paragraph text from the full run DataFrame.

    text runs contribute their content; line_break runs contribute a newline
    so intra-paragraph breaks produce a natural separator. field_marker runs
    and other types are discarded. Leading/trailing whitespace is stripped.
    """
    mask = run_df["run_type"].isin({"text", "line_break"})
    working = run_df.loc[mask, ["paragraph_id", "order_index", "text"]].copy()
    working["text"] = working["text"].fillna("").astype(str)
    working = working.sort_values(["paragraph_id", "order_index"])
    result = working.groupby("paragraph_id", sort=False)["text"].agg("".join)
    return result.str.strip()


def build_paragraphs(
    run_df: pd.DataFrame,
    include_notes: bool = True,
) -> pd.DataFrame:
    """
    Aggregate run-level rows into paragraph-level rows.

    Keeps run_type in {"text", "image_ref", "shape_ref", "chart_ref",
    "graphic_ref"} and groups by paragraph_id. Style is weighted by character
    count; bold/italic/underlined ratios are recomputed from sums. A block_type
    column is derived post-aggregation.

    Args:
        run_df: Output of extract_runs.
        include_notes: When False, speaker-note paragraphs are excluded
            (header_footer_type == "notes").

    Returns:
        One row per paragraph that contained at least one content run.
    """
    if run_df.empty:
        return pd.DataFrame()

    if not include_notes and "header_footer_type" in run_df.columns:
        run_df = run_df[run_df["header_footer_type"] != "notes"]

    if run_df.empty:
        return pd.DataFrame()

    content_runs = run_df[run_df["run_type"].isin(_CONTENT_RUN_TYPES)].copy()
    if content_runs.empty:
        return pd.DataFrame()

    # Drop paragraphs whose text is empty/whitespace, unless they carry an image.
    _text_para_ids = set(
        content_runs.loc[
            (content_runs["run_type"] == "text")
            & content_runs["text"].fillna("").str.strip().astype(bool),
            "paragraph_id",
        ]
    )
    _ref_para_ids = set(
        content_runs.loc[
            content_runs["run_type"].isin({"image_ref", "chart_ref"}), "paragraph_id"
        ]
    )
    content_runs = content_runs[
        content_runs["paragraph_id"].isin(_text_para_ids | _ref_para_ids)
    ]
    if content_runs.empty:
        return pd.DataFrame()

    para_has_image = set(
        content_runs.loc[content_runs["run_type"] == "image_ref", "paragraph_id"]
    )
    para_has_chart = set(
        content_runs.loc[content_runs["run_type"] == "chart_ref", "paragraph_id"]
    )
    para_has_shape = set(
        content_runs.loc[
            content_runs["run_type"].isin({"shape_ref", "graphic_ref"}), "paragraph_id"
        ]
    )

    content_runs["char_count"] = content_runs["text"].fillna("").str.len()
    content_runs["alpha_count"] = content_runs["text"].fillna("").str.count(r"[a-zA-Z]")

    para_text = _paragraph_text(run_df)

    agg_spec = build_standard_agg_spec(
        identity_cols=_PARA_IDENTITY_COLS,
        include_geometry=True,
        include_hierarchy=False,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        extra_agg={
            "hyperlink_url": _collect_unique_list,
            "chart_id": "first",
        },
        count_col="run_id",
    )

    para_df = aggregate_hierarchical(
        df=content_runs,
        group_col="paragraph_id",
        agg_spec=agg_spec,
        rename_count_col={"run_id": "run_count"},
        compute_derived=True,
    )

    para_df["text"] = para_df["paragraph_id"].map(para_text)

    # Derive block_type in ascending priority (table wins over all).
    para_df["block_type"] = None
    if "shape_type" in para_df.columns:
        para_df.loc[para_df["shape_type"] == "speaker_notes", "block_type"] = "speaker_notes"
    if para_has_shape:
        para_df.loc[para_df["paragraph_id"].isin(para_has_shape), "block_type"] = "shape"
    if para_has_image:
        para_df.loc[para_df["paragraph_id"].isin(para_has_image), "block_type"] = "image"
    if para_has_chart:
        para_df.loc[para_df["paragraph_id"].isin(para_has_chart), "block_type"] = "chart"
    if "table_id" in para_df.columns:
        para_df.loc[para_df["table_id"].notna(), "block_type"] = "table"

    return para_df


__all__ = ["build_paragraphs"]

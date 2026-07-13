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

from .._utils.df_aggregation.registry_aggregator import aggregate_to, group_join
from .._utils.df_aggregation.text_merge import apply_inline_markup


# Run types eligible to become paragraph content. Eligible is not the same as
# guaranteed to survive: build_paragraphs additionally drops any paragraph
# that has no "text"/"math" content AND isn't an image_ref/chart_ref (see
# _text_para_ids / _ref_para_ids below) — so a shape_ref or graphic_ref run
# only survives when its paragraph also carries text, an image, or a chart.
# A shape_ref/graphic_ref alone in its paragraph (the common case for
# decorative, textless shapes) is dropped before aggregation.
_CONTENT_RUN_TYPES = frozenset(
    {"text", "math", "image_ref", "shape_ref", "chart_ref", "graphic_ref"}
)

# Run types that keep their paragraph alive even with no text content.
# NOTE: shape_ref/graphic_ref are deliberately excluded here — see the
# _CONTENT_RUN_TYPES comment above.
_SELF_SUFFICIENT_REF_TYPES = frozenset({"image_ref", "chart_ref"})


def _paragraph_text(run_df: pd.DataFrame) -> pd.Series:
    """
    Build paragraph text from the full run DataFrame (all run_types).

    text and math runs contribute their content; line_break runs contribute a
    literal newline so intra-paragraph breaks produce a natural separator.
    field_marker runs and other types are discarded.

    Superscript/subscript/strikethrough runs (equations frequently carry
    baseline-shifted rPr, e.g. the subscript in g_s) are wrapped as inline
    markup (``[^x]`` / ``[_x]`` / ``~~x~~``) via :func:`apply_inline_markup`
    before the join, matching the HTML/PDF/DOCX text output. PPTX runs within
    a paragraph are contiguous character runs like DOCX, so fragments are
    concatenated with no separator (``sep=""``) — script tokens attach cleanly
    and no spurious spaces appear between styled runs. Leading/trailing
    whitespace is stripped from the result.
    """
    mask = run_df["run_type"].isin({"text", "math", "line_break"})
    cols = ["paragraph_id", "order_index", "text", "script_type", "is_strikethrough"]
    working = run_df.loc[mask, [c for c in cols if c in run_df.columns]].copy()
    working["text"] = working["text"].fillna("").astype(str)
    working["text"] = apply_inline_markup(working)
    working = working.sort_values(["paragraph_id", "order_index"])
    result = group_join(working["text"], working["paragraph_id"], sep="")
    return result.str.strip()


def build_paragraphs(
    run_df: pd.DataFrame,
    include_notes: bool = True,
) -> pd.DataFrame:
    """
    Aggregate run-level rows into paragraph-level rows.

    Considers run_type in _CONTENT_RUN_TYPES {"text", "math", "image_ref",
    "shape_ref", "chart_ref", "graphic_ref"}, but only keeps a paragraph if it
    has non-blank text/math OR a self-sufficient ref (image_ref/chart_ref —
    see _SELF_SUFFICIENT_REF_TYPES). A paragraph containing only a shape_ref
    or graphic_ref run (no text, image, or chart alongside it) is dropped
    before aggregation, even though shape_ref/graphic_ref are themselves
    "content" run types. Groups by paragraph_id; style is weighted by
    character count, bold/italic/underlined ratios are recomputed from sums,
    and a block_type column is derived post-aggregation.

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

    # Drop paragraphs whose text is empty/whitespace, unless they carry a
    # self-sufficient ref (image/chart). shape_ref/graphic_ref runs are NOT
    # self-sufficient: a paragraph containing only one of those (no text, no
    # image, no chart) is dropped here — see _SELF_SUFFICIENT_REF_TYPES.
    _text_para_ids = set(
        content_runs.loc[
            content_runs["run_type"].isin({"text", "math"})
            & content_runs["text"].fillna("").str.strip().astype(bool),
            "paragraph_id",
        ]
    )
    _ref_para_ids = set(
        content_runs.loc[
            content_runs["run_type"].isin(_SELF_SUFFICIENT_REF_TYPES), "paragraph_id"
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
    # Math block_type requires the *whole* paragraph to be math runs — a
    # paragraph mixing prose and a single math run (e.g. an inline variable
    # like "at large 𝛼′") is still prose, not a math block.
    para_run_types = content_runs.groupby("paragraph_id")["run_type"].agg(set)
    para_has_math = set(para_run_types[para_run_types == {"math"}].index)

    content_runs["char_count"] = content_runs["text"].fillna("").str.len()
    content_runs["alpha_count"] = content_runs["text"].fillna("").str.count(r"[a-zA-Z]")

    para_text = _paragraph_text(run_df)

    para_df = aggregate_to(
        content_runs,
        by="paragraph_id",
        size_as="run_count",
        on_unknown="raise",
    )

    para_df["text"] = para_df["paragraph_id"].map(para_text)

    # 1-celled tables carry no structural table information — treat as plain textboxes.
    if "table_id" in run_df.columns and "table_cell_id" in run_df.columns:
        cell_counts = (
            run_df[run_df["table_id"].notna()]
            .groupby("table_id")["table_cell_id"]
            .nunique()
        )
        single_cell_ids = set(cell_counts[cell_counts == 1].index)
        if single_cell_ids and "table_id" in para_df.columns:
            mask = para_df["table_id"].isin(single_cell_ids)
            para_df.loc[mask, ["table_id", "table_row_id", "table_cell_id"]] = None

    # Derive block_type in ascending priority (table wins over all; math is
    # lowest priority, filling only paragraphs still unassigned afterwards).
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
    # math is lowest priority: only fills paragraphs left unassigned above
    # (e.g. a formula-only paragraph with no shape/image/chart/table context).
    if para_has_math:
        mask = para_df["paragraph_id"].isin(para_has_math) & para_df["block_type"].isna()
        para_df.loc[mask, "block_type"] = "math"

    return para_df


__all__ = ["build_paragraphs"]

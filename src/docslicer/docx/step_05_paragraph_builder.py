"""
DOCX run → paragraph aggregator.

Filters text/image runs and merges them by paragraph_id, producing one row per
logical paragraph with concatenated text and character-count-weighted style.
When include_headers_footers is False (default) header/footer rows are dropped
before processing.  When True they must already be expanded per-page (via
expand_header_footer_runs) and receive block_type "header" or "footer".
"""

from __future__ import annotations

import pandas as pd

from docslicer._utils.df_aggregation.registry_aggregator import aggregate_to, group_join
from docslicer._utils.df_aggregation.text_merge import apply_inline_markup


def _paragraph_text(run_df: pd.DataFrame) -> pd.Series:
    """
    Build paragraph text from the full run DataFrame (all run_types).

    text runs contribute their content; tab runs become a single space so that
    text–tab–text patterns produce a space separator rather than a direct concat.
    footnote_reference / endnote_reference runs carry the note id in ``text`` and
    are rendered as markdown-style ``[^30]`` inline markers so the reference is
    preserved in the paragraph text. All other run types (field_marker,
    field_code, page_break, …) are discarded.

    Superscript/subscript/strikethrough runs are wrapped as inline markup
    (``[^x]`` / ``[_x]`` / ``~~x~~``) via :func:`apply_inline_markup` before the
    join, matching the HTML/PDF text output. docx runs within a paragraph are
    contiguous character runs, so fragments are concatenated with no separator
    (``sep=""``) — script tokens attach cleanly and no spurious spaces appear
    between styled runs. Leading/trailing whitespace is stripped from the result.
    """
    ref_types = {"footnote_reference", "endnote_reference"}
    mask = run_df["run_type"].isin({"text", "tab"} | ref_types)
    cols = ["paragraph_id", "order_index", "run_type", "text", "script_type", "is_strikethrough"]
    working = run_df.loc[mask, [c for c in cols if c in run_df.columns]].copy()
    working["text"] = working["text"].fillna("").astype(str)
    # Render note reference markers as [^30] before apply_inline_markup, and clear
    # their script_type so the marker isn't additionally wrapped as superscript
    # (footnote markers are typically superscript in the source).
    ref_mask = working["run_type"].isin(ref_types) & working["text"].str.strip().ne("")
    if ref_mask.any():
        working.loc[ref_mask, "text"] = "[^" + working.loc[ref_mask, "text"].str.strip() + "]"
        if "script_type" in working.columns:
            working.loc[ref_mask, "script_type"] = None
    # Wrap script/strikethrough runs before tab substitution: a tab's text is
    # blank so apply_inline_markup leaves it untouched, then we set it to a space.
    working["text"] = apply_inline_markup(working)
    working.loc[working["run_type"] == "tab", "text"] = " "
    working = working.sort_values(["paragraph_id", "order_index"])
    result = group_join(working["text"], working["paragraph_id"], sep="")
    return result.str.strip()


def build_paragraphs(
    run_df: pd.DataFrame,
    include_headers_footers: bool = False,
) -> pd.DataFrame:
    """
    Aggregate run-level rows into paragraph-level rows.

    Keeps ``run_type`` in ``{"text", "image_ref"}`` runs, groups by
    ``paragraph_id``. Style is weighted by character count; bold/italic/underlined
    ratios are recomputed from sums. A ``block_type`` column is derived
    post-aggregation.

    Args:
        run_df: Output of ``extract_runs`` (optionally post-processed by
            ``expand_header_footer_runs``).
        include_headers_footers: When False (default) header/footer rows are
            dropped before processing.  When True they are expected to have
            been expanded per-page and receive block_type "header"/"footer".

    Returns:
        One row per paragraph that contained at least one text or image run.
    """
    if run_df.empty:
        return pd.DataFrame()

    if not include_headers_footers and "header_footer_type" in run_df.columns:
        run_df = run_df[~run_df["header_footer_type"].isin({"header", "footer"})]

    if run_df.empty:
        return pd.DataFrame()

    content_runs = run_df[run_df["run_type"].isin({"text", "footnote_reference", "image_ref"})].copy()
    if content_runs.empty:
        return pd.DataFrame()

    # Track which paragraphs contain at least one image before aggregation.
    para_has_image = set(
        content_runs.loc[content_runs["run_type"] == "image_ref", "paragraph_id"]
    )
    # Charts are image_ref runs that reference a chart part; carry a chart_id.
    if "chart_id" in content_runs.columns:
        para_has_chart = set(
            content_runs.loc[content_runs["chart_id"].notna(), "paragraph_id"]
        )
    else:
        para_has_chart = set()

    # char_count / alpha_count drive the registry's DOMINANT (alpha-weighted
    # style) and WEIGHTED_RATIO (char-weighted bold/italic/underlined/strike)
    # policies. image_ref rows contribute 0 chars so they don't distort weighting.
    content_runs["char_count"] = content_runs["text"].fillna("").str.len()
    content_runs["alpha_count"] = content_runs["text"].fillna("").str.count(r"[a-zA-Z]")

    # Pre-compute paragraph text from the filtered run_df (headers/footers
    # already removed) so tabs between text runs produce a space separator.
    para_text = _paragraph_text(run_df)

    # Registry-driven aggregation: every column's policy comes from COLUMN_REGISTRY
    # (see registry_aggregator) — no call-site drop list. derived=True recomputes
    # is_bold/is_italic/is_underlined/is_strikethrough from the weighted ratios;
    # size_as gives the run count. on_unknown="raise" fails loudly if a run column
    # is missing from the registry rather than silently defaulting to "first".
    para_df = aggregate_to(
        content_runs,
        by="paragraph_id",
        size_as="run_count",
        on_unknown="raise",
    )

    para_df["text"] = para_df["paragraph_id"].map(para_text)

    # Derive block_type: image → paragraphs with at least one image_ref run;
    # table → paragraphs inside a table (table_id not null); table takes priority.
    # footnote/endnote and header/footer override both so structural context is
    # preserved.
    para_df["block_type"] = None
    if para_has_image:
        para_df.loc[para_df["paragraph_id"].isin(para_has_image), "block_type"] = "image"
    if para_has_chart:
        para_df.loc[para_df["paragraph_id"].isin(para_has_chart), "block_type"] = "chart"
    if "table_id" in para_df.columns:
        para_df.loc[para_df["table_id"].notna(), "block_type"] = "table"
    if "header_footer_type" in para_df.columns:
        para_df.loc[para_df["header_footer_type"] == "footnote", "block_type"] = "footnote"
        para_df.loc[para_df["header_footer_type"] == "endnote", "block_type"] = "endnote"
    if include_headers_footers and "header_footer_type" in para_df.columns:
        para_df.loc[para_df["header_footer_type"] == "header", "block_type"] = "header"
        para_df.loc[para_df["header_footer_type"] == "footer", "block_type"] = "footer"

    return para_df

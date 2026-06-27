# NOTE: WIP - didn't give good results

"""
step_07_reading_order.py  (rewrite)

New reading-order pipeline. Public API:

    df_words, df_struct_groups = assign_reading_order(df_words, df_shapes)

Step 1 — Build struct groups
----------------------------
The PDF struct tree / marked content gives us a `struct_group_id` on every word
(struct_tag_id > mcid+1e6 > text_object_id+2e6, see step_01). Words sharing a
struct_group_id are one logical block (a paragraph, list item, heading, …).

  * If `struct_group_id` is populated on every word, aggregate words into one row
    per struct group:
       - "text"  -> join_fragments(..., dehyphenate=True), so line-broken words
                    ("inter-" + "national" -> "inter-national") rejoin and inline
                    sub/superscript markup is preserved.
       - everything else -> the shared hierarchical aggregator (geometry, style,
                    counts, metadata) exactly like the cell / line / block builders.

  * If `struct_group_id` is missing or null anywhere (e.g. OCR text, which has no
    struct tree), there is nothing to group on — df_struct_groups falls back to
    df_words unchanged.

Later steps (TODO) will assign reading order across these struct groups.
"""

from __future__ import annotations

import pandas as pd

from .._utils.text_merge import apply_inline_markup, join_fragments
from .._utils.hierarchical_aggregator import (
    aggregate_hierarchical,
    build_standard_agg_spec,
)


# ================================================================================
# STEP 1 — STRUCT GROUP AGGREGATION
# ================================================================================

def _has_struct_groups(df: pd.DataFrame) -> bool:
    """True when struct_group_id exists and is populated on every word."""
    return "struct_group_id" in df.columns and df["struct_group_id"].notna().all()


def _build_struct_groups_df(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate words into one row per struct group.

    struct_group_id is a per-page DFS counter (step_01), so it is NOT unique across
    pages — the group key is therefore (page_number, struct_group_id). Words are
    ordered within each group by text_object_id (PDF stream order) so the joined
    text reads correctly; group order follows (page_number, struct_group_id).
    """
    df = df_words.copy()

    # Composite per-page group key (struct_group_id alone collides across pages).
    df["_sg_key"] = (
        df["page_number"].astype("int64") * 1_000_000_000
        + df["struct_group_id"].astype("int64")
    )

    # Within-group reading order: text_object_id is PDF byte-stream order and is
    # always populated by step_01. Fall back to spatial order if it is missing.
    if "text_object_id" in df.columns and df["text_object_id"].notna().any():
        sort_cols = ["_sg_key", "text_object_id"]
    else:
        sort_cols = ["_sg_key", "y_top", "x_left"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Per-fragment inline markup (sub/superscript, strikethrough) before joining.
    df["_fmt_text"] = apply_inline_markup(df)

    agg_spec = build_standard_agg_spec(
        identity_cols=[
            "page_number",
            "page_width",
            "page_height",
            "reading_column",
            "gutter_id_left",
            "gutter_id_right",
            # struct-tree provenance
            "struct_group_id",
            "struct_tag",
            "struct_tag_id",
            "reading_rank",
        ],
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=False,
        include_hierarchy=False,
        include_table=False,
        extra_agg={
            "_fmt_text": lambda s: join_fragments(s.tolist(), dehyphenate=True),
            "word_id": list,
        },
    )

    df_groups = aggregate_hierarchical(
        df,
        group_col="_sg_key",
        agg_spec=agg_spec,
        rename_count_col={"word_id": "word_ids"},
    )
    df_groups = df_groups.rename(columns={"_fmt_text": "text"})
    df_groups = df_groups.drop(columns=["_sg_key"])
    return df_groups


# ================================================================================
# PUBLIC API
# ================================================================================

def assign_reading_order(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build struct groups and (later) assign reading order.

    Returns
    -------
    df_words         : the input words (unchanged in step 1).
    df_struct_groups : one row per struct_group_id when struct groups are
                       available, otherwise a copy of df_words.
    """
    if df_words is None or df_words.empty:
        empty = df_words.copy() if df_words is not None else pd.DataFrame()
        return empty, empty

    if not _has_struct_groups(df_words):
        return df_words, df_words.copy()

    df_struct_groups = _build_struct_groups_df(df_words)
    return df_words, df_struct_groups

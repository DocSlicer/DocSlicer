"""
HTML struct-tree block_type prefiller.

Runs after the line builder, before the table extractor and the shared steps.
Uses the HTML struct-ancestor chain to pre-assign block_type for high-confidence
structural cases so the shared detectors can skip those rows.

By the time this step runs, block_type is already set for a handful of
structural elements — ``hr`` and ``image`` (box cleaner), ``page_label``
(page-label detector) and ``table`` (line builder). Those values are never
overwritten; only rows with no block_type yet are considered.

Rules (applied only to rows with no existing block_type), resolved by the
*deepest* matching struct ancestor so nesting decides the tie — a heading
inside a blockquote is a heading, a code span inside a heading is code:

    code        — a ``code`` or ``pre`` ancestor (``<pre><code>``, or a bare
                  ``<pre>`` block that isn't literally labelled ``code``)
    block_quote — a ``blockquote`` ancestor
    heading     — an ``h1``-``h6`` ancestor (deepest wins, as in the PDF path);
                  heading rows also get ``heading_source = "html"`` so the
                  shared heading detector treats them as structural facts, and
                  ``heading_level_raw`` — the number of the deepest ``h*`` tag

table_header_flag is set upstream in the box cleaner (thead / first all-TH row).

Public API:
    prefill_styles(df) -> pd.DataFrame
        Adds/updates block_type (and heading_source for headings), then
        returns *df*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Deepest matching ancestor tag → block_type. h1-h6 all map to "heading".
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_TAG_BLOCK_TYPE: dict[str, str] = {
    "code": "code",
    "pre": "code",
    "blockquote": "block_quote",
    **{tag: "heading" for tag in _HEADING_TAGS},
}


def _as_list(v) -> list:
    """Coerce a struct-ancestor cell to a real list (NaN/scalars → [])."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return []


def _unfilled(block_type: pd.Series) -> pd.Series:
    """Boolean mask: rows whose block_type is not yet assigned."""
    bt = block_type.astype(object)
    stripped = bt.astype(str).str.strip()
    return bt.isna() | (stripped == "") | (stripped == "nan")


def prefill_styles(df: pd.DataFrame) -> pd.DataFrame:
    """Assign block_type from struct_ancestors for still-unassigned rows."""
    if df is None or df.empty:
        return df

    if "block_type" not in df.columns:
        df["block_type"] = None
    if "struct_ancestors" not in df.columns:
        return df

    unfilled = _unfilled(df["block_type"]).to_numpy()
    anc_arr = df["struct_ancestors"].to_numpy(dtype=object)
    n = len(df)

    new_types = np.empty(n, dtype=object)
    new_types[:] = None
    new_levels = np.empty(n, dtype=object)
    new_levels[:] = None

    for i in range(n):
        if not unfilled[i]:
            continue
        chosen = None
        level = None
        # Ancestors run root -> leaf, so the last relevant tag is the deepest.
        for tag in _as_list(anc_arr[i]):
            bt = _TAG_BLOCK_TYPE.get(tag)
            if bt is not None:
                chosen = bt
                level = int(tag[1:]) if bt == "heading" else None
        new_types[i] = chosen
        new_levels[i] = level

    assign = pd.Series(new_types, index=df.index)
    to_set = assign.notna()
    df.loc[to_set, "block_type"] = assign[to_set]

    # Headings assigned here are structural facts from the HTML tree.
    is_heading = to_set & assign.eq("heading")
    if is_heading.any():
        if "heading_source" not in df.columns:
            df["heading_source"] = pd.NA
        df.loc[is_heading, "heading_source"] = "html"
        if "heading_level_raw" not in df.columns:
            df["heading_level_raw"] = pd.NA
        df.loc[is_heading, "heading_level_raw"] = pd.Series(new_levels, index=df.index)[is_heading]

    return df

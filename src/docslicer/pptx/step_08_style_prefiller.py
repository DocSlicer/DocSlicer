"""
PPTX shape-name block_type prefiller.

Runs after the line builder. Uses ``shape_name`` (and, for headings,
``font_size`` / position) to pre-assign block_type for high-confidence cases
so the shared detectors can skip those rows.

By the time this step runs, block_type is already set for table/chart/image/
shape/math/speaker_notes rows (paragraph builder). Those values are never
overwritten; only rows with no block_type yet are considered.

Rules (applied only to rows with no existing block_type):

    heading     — at most one line per slide, and only on slides that have a
                  shape whose name contains "title" (and not "subtitle"); a
                  slide with no title-named shape gets no prefilled heading.
                  Among a slide's title-shape lines, the winner is the one
                  with the largest font_size, then the highest position
                  (smallest y_top). heading_level_raw is always 1 — PPTX
                  titles carry no real hierarchy, unlike Word/HTML's h1-h6.
    footnote    — shape_name contains "footnote"
    page_label  — shape_name contains "slide number" or "page number"

Public API:
    prefill_styles(df) -> pd.DataFrame
        Adds/updates block_type (and heading_source/heading_level_raw for the
        picked heading), then returns *df*.
"""
from __future__ import annotations

import pandas as pd

_SUBTITLE_RE = r"subtitle"
_TITLE_RE = r"title"
_FOOTNOTE_RE = r"footnote"
_PAGE_LABEL_RE = r"slide number|page number"


def _unfilled(block_type: pd.Series) -> pd.Series:
    """Boolean mask: rows whose block_type is not yet assigned."""
    bt = block_type.astype(object)
    stripped = bt.astype(str).str.strip()
    return bt.isna() | (stripped == "") | (stripped == "nan")


def _name_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    if "shape_name" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["shape_name"].astype(str).str.contains(pattern, case=False, na=False, regex=True)


def _assign_heading(df: pd.DataFrame, group_cols: list[str]) -> None:
    """
    Pick at most one heading line per slide and set it in place.

    Only slides with at least one title-named shape get a heading — there is
    no font-size/position fallback for slides without one.
    """
    unfilled = _unfilled(df["block_type"])
    has_text = df["text"].astype(str).str.strip().ne("") if "text" in df.columns else unfilled
    is_title = _name_contains(df, _TITLE_RE) & ~_name_contains(df, _SUBTITLE_RE)
    candidates = df[unfilled & has_text & is_title]
    if candidates.empty or not group_cols:
        return

    ranked = candidates.assign(
        _font_size=pd.to_numeric(candidates.get("font_size"), errors="coerce").fillna(0.0),
        _y_top=pd.to_numeric(candidates.get("y_top"), errors="coerce").fillna(float("inf")),
    )
    picked = ranked.sort_values(
        group_cols + ["_font_size", "_y_top"],
        ascending=[True] * len(group_cols) + [False, True],
        kind="stable",
    ).drop_duplicates(subset=group_cols, keep="first")

    df.loc[picked.index, "block_type"] = "heading"
    if "heading_source" not in df.columns:
        df["heading_source"] = pd.NA
    df.loc[picked.index, "heading_source"] = "pptx"
    if "heading_level_raw" not in df.columns:
        df["heading_level_raw"] = pd.NA
    df.loc[picked.index, "heading_level_raw"] = 1


def prefill_styles(df: pd.DataFrame) -> pd.DataFrame:
    """Assign block_type from shape_name for still-unassigned rows."""
    if df is None or df.empty:
        return df

    if "block_type" not in df.columns:
        df["block_type"] = None
    if "shape_name" not in df.columns:
        return df

    df = df.copy()

    footnote_mask = _unfilled(df["block_type"]) & _name_contains(df, _FOOTNOTE_RE)
    df.loc[footnote_mask, "block_type"] = "footnote"

    page_label_mask = _unfilled(df["block_type"]) & _name_contains(df, _PAGE_LABEL_RE)
    df.loc[page_label_mask, "block_type"] = "page_label"

    group_cols = [c for c in ("page_number", "slide_index") if c in df.columns]
    _assign_heading(df, group_cols)

    return df


__all__ = ["prefill_styles"]

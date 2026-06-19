"""
step_05c_footnote_detector.py

Detects footnote blocks in PDF word extracts and marks them with
``footnote_flag = True`` so downstream steps (e.g. line-number detector)
can exclude them.

Strategy per page
-----------------
1. Compute the global font_size median across the whole document.
2. Isolate words in the bottom BOTTOM_PAGE_FRACTION of the page whose
   font_size is below FONT_SIZE_RATIO × global median.
3. Assign temporary line IDs to that subset.
4. If at least one line starts with a recognised footnote marker
   (bare digit, digit+dot, *, †, ‡, §) the entire small-font bottom
   block is labelled footnote_flag = True.
"""

from __future__ import annotations

import re

import pandas as pd

from .._utils.line_merger import assign_line_id

# =============================================================================
# CONFIG
# =============================================================================

_BOTTOM_PAGE_FRACTION: float = 0.50   # inspect bottom 50 % of page height
_FONT_SIZE_RATIO: float = 0.85        # word must be < 85 % of global median
_MARKER_RE = re.compile(r"^(\d+\.?|[*†‡§])$")


# =============================================================================
# Public API
# =============================================================================

def detect_footnotes(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df_words* with a boolean ``footnote_flag`` column.

    Parameters
    ----------
    df_words : pd.DataFrame
        Must contain: page_number, word_id, text, x_left, y_top,
        font_size, page_height.

    Returns
    -------
    pd.DataFrame
        Copy of df_words with ``footnote_flag`` (bool) added.
    """
    if df_words is None or df_words.empty:
        out = df_words.copy() if df_words is not None else pd.DataFrame()
        out["footnote_flag"] = pd.Series(dtype=bool)
        return out

    required = {
        "page_number", "word_id", "text",
        "x_left", "y_top", "font_size", "page_height",
    }
    missing = required - set(df_words.columns)
    if missing:
        raise KeyError(f"detect_footnotes: missing columns: {missing}")

    out = df_words.copy()
    out["footnote_flag"] = False

    # Global font-size median — ignore zero / NaN entries (OCR artefacts)
    valid_sizes = out["font_size"].replace(0, pd.NA).dropna()
    if valid_sizes.empty:
        return out
    global_median: float = float(valid_sizes.median())
    size_threshold: float = _FONT_SIZE_RATIO * global_median

    for page_num, page_group in out.groupby("page_number"):
        flagged = _detect_page_footnotes(page_group, size_threshold)
        if flagged:
            out.loc[out["word_id"].isin(flagged), "footnote_flag"] = True

    return out


# =============================================================================
# Internal helpers
# =============================================================================

def _detect_page_footnotes(
    page_df: pd.DataFrame,
    size_threshold: float,
) -> list[int]:
    """Return word_ids that belong to a footnote block on this page."""

    page_height: float = float(page_df["page_height"].iloc[0])
    y_cutoff: float = page_height * (1.0 - _BOTTOM_PAGE_FRACTION)

    # Keep only small-font words in the bottom half of the page
    mask = (page_df["y_top"] >= y_cutoff) & (page_df["font_size"] < size_threshold)
    candidates = page_df[mask].copy()

    if candidates.empty:
        return []

    # Restrict to LTR text before line merging
    if "text_orientation" in candidates.columns:
        orientation = candidates["text_orientation"].fillna("LTR").str.upper()
        candidates = candidates[orientation == "LTR"].copy()

    if candidates.empty:
        return []

    candidates = assign_line_id(candidates, y_alignment="center")

    # At least one line must begin with a footnote marker
    if not _has_footnote_marker(candidates):
        return []

    return candidates["word_id"].tolist()


def _has_footnote_marker(df: pd.DataFrame) -> bool:
    """Return True if any line in *df* starts with a footnote marker."""
    for _, line_group in df.groupby("line_id"):
        first_word = line_group.sort_values("x_left").iloc[0]
        if _MARKER_RE.match(str(first_word["text"]).strip()):
            return True
    return False

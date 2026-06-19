"""
step_06_line_number_detector.py

Detects margin line numbers in structured documents (legal, technical, etc.).

For each page:
1. Group words into temporary y-buckets via line_merger.
2. Per bucket, identify the leftmost word.
3. Check whether those leftmost candidates form a monotonically increasing
   sequence of integers that share a common x alignment.
4. If such a series is found, set line_number_flag = True on those words.
"""

from __future__ import annotations

import pandas as pd

from .._utils.line_merger import assign_line_id


# =============================================================================
# CONFIG
# =============================================================================

_MIN_SERIES_LENGTH: int = 3       # minimum candidates to qualify as line numbers
_X_ALIGN_TOLERANCE: float = 5.0   # pt — max x_left spread within an x-cluster
_MAX_NUMBER_WIDTH: float = 30.0   # pt — line-number token must be narrow
_MAX_MISSING_NUMBERS_PER_PAGE: int = 1  # allow at most one skipped line number


# =============================================================================
# Public API
# =============================================================================

def detect_line_numbers(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean column ``line_number_flag`` to df_words.

    Parameters
    ----------
    df_words : pd.DataFrame
        Must contain: page_number, word_id, text, x_left, x_right, y_top, y_bottom.

    Returns
    -------
    pd.DataFrame
        Copy of df_words with an extra ``line_number_flag`` (bool) column.
    """
    if df_words is None or df_words.empty:
        out = df_words.copy() if df_words is not None else pd.DataFrame()
        out["line_number_flag"] = pd.Series(dtype=bool)
        return out

    required = {"page_number", "word_id", "text", "x_left", "x_right", "y_top", "y_bottom"}
    missing = required - set(df_words.columns)
    if missing:
        raise KeyError(f"detect_line_numbers: missing columns: {missing}")

    _tmp_cols = ["page_number", "word_id", "text", "x_left", "x_right", "y_top", "y_bottom"]
    if "text_orientation" in df_words.columns:
        _tmp_cols.append("text_orientation")

    # Assign temporary line IDs on a minimal copy (avoids mutating df_words).
    # Ignore non-LTR words before line merging so rotated/vertical footer text
    # cannot become the leftmost word for a line-number row.
    # Also exclude words already identified as footnotes.
    df_source = df_words
    if "footnote_flag" in df_words.columns:
        df_source = df_words[~df_words["footnote_flag"]]

    df_tmp = df_source[_tmp_cols].copy()
    if "text_orientation" in df_tmp.columns:
        orientation = df_tmp["text_orientation"].fillna("LTR").astype(str).str.upper()
        df_tmp = df_tmp[orientation == "LTR"].copy()

    df_tmp = assign_line_id(df_tmp, y_alignment="center")

    out = df_words.copy()
    out["line_number_flag"] = False

    for page_num, page_group in df_tmp.groupby("page_number"):
        flagged_ids = _detect_page_line_numbers(page_group)
        if flagged_ids:
            mask = (out["page_number"] == page_num) & (out["word_id"].isin(flagged_ids))
            out.loc[mask, "line_number_flag"] = True

    return out


# =============================================================================
# Internal helpers
# =============================================================================

def _detect_page_line_numbers(page_df: pd.DataFrame) -> list[int]:
    """Return word_ids identified as line numbers on this page."""

    # Per line_id: pick the leftmost word — idxmin is C-level, no apply overhead
    idx = page_df.groupby("line_id")["x_left"].idxmin()
    candidates = page_df.loc[idx].reset_index(drop=True)

    # Vectorised positive-integer check (replaces row-wise _is_positive_integer)
    text_str = candidates["text"].astype(str).str.strip()
    is_pos_int = text_str.str.fullmatch(r"\d+", na=False) & (
        pd.to_numeric(text_str, errors="coerce").fillna(0) > 0
    )
    candidates = candidates[
        is_pos_int
        & ((candidates["x_right"] - candidates["x_left"]) <= _MAX_NUMBER_WIDTH)
    ].copy()

    if len(candidates) < _MIN_SERIES_LENGTH:
        return []

    candidates["_num"] = candidates["text"].astype(str).str.strip().astype(int)

    # Sort by reading order (y_top ascending)
    candidates = candidates.sort_values("y_top").reset_index(drop=True)

    # Group into x-aligned clusters, then check each for a near-contiguous series.
    flagged_ids: list[int] = []
    missing_budget = _MAX_MISSING_NUMBERS_PER_PAGE
    for cluster in _cluster_by_x(candidates):
        if len(cluster) < _MIN_SERIES_LENGTH:
            continue
        nums = cluster.sort_values("y_top")["_num"].tolist()
        ok, missing_count = _is_line_number_sequence(nums, max_missing=missing_budget)
        if ok:
            flagged_ids.extend(cluster["word_id"].tolist())
            missing_budget -= missing_count

    return flagged_ids


def _is_line_number_sequence(nums: list[int], *, max_missing: int) -> tuple[bool, int]:
    """
    Return whether nums look like margin line numbers and how many numbers are missing.

    Line numbers should be contiguous. To tolerate one missed OCR/tokenization row,
    the page gets a tiny missing-number budget, e.g. [1, 2, 4, 5] is acceptable
    with max_missing=1, while TOC/page-label jumps like [2, 12, 14] are not.
    """
    if len(nums) < _MIN_SERIES_LENGTH:
        return False, 0

    missing_count = 0
    for a, b in zip(nums, nums[1:]):
        diff = b - a
        if diff <= 0:
            return False, missing_count

        missing_count += diff - 1
        if missing_count > max_missing:
            return False, missing_count

    return True, missing_count


def _cluster_by_x(candidates: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split candidates into groups where the total x_left spread stays within
    _X_ALIGN_TOLERANCE.  Sorts by x_left before splitting.
    """
    if candidates.empty:
        return []

    sorted_c = candidates.sort_values("x_left").reset_index(drop=True)
    x_vals = sorted_c["x_left"].values

    clusters: list[pd.DataFrame] = []
    start = 0
    for i in range(1, len(x_vals)):
        if x_vals[i] - x_vals[start] > _X_ALIGN_TOLERANCE:
            clusters.append(sorted_c.iloc[start:i])
            start = i
    clusters.append(sorted_c.iloc[start:])

    return clusters

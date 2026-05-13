"""
step_05_line_number_detector.py

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
    df_tmp = df_words[_tmp_cols].copy()
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

    # Per line_id: pick the leftmost word
    candidates = (
        page_df.groupby("line_id", group_keys=False)
        .apply(lambda g: g.loc[g["x_left"].idxmin()])
        .reset_index(drop=True)
    )

    # Keep only narrow, positive-integer tokens
    candidates = candidates[
        candidates["text"].apply(_is_positive_integer)
        & ((candidates["x_right"] - candidates["x_left"]) <= _MAX_NUMBER_WIDTH)
    ].copy()

    if len(candidates) < _MIN_SERIES_LENGTH:
        return []

    candidates["_num"] = candidates["text"].astype(str).str.strip().astype(int)

    # Sort by reading order (y_top ascending)
    candidates = candidates.sort_values("y_top").reset_index(drop=True)

    # Group into x-aligned clusters, then check each for monotonic increase
    flagged_ids: list[int] = []
    for cluster in _cluster_by_x(candidates):
        if len(cluster) < _MIN_SERIES_LENGTH:
            continue
        nums = cluster.sort_values("y_top")["_num"].tolist()
        if _is_monotonically_increasing(nums):
            flagged_ids.extend(cluster["word_id"].tolist())

    return flagged_ids


def _is_positive_integer(text) -> bool:
    try:
        return int(str(text).strip()) > 0
    except (ValueError, TypeError):
        return False


def _is_monotonically_increasing(nums: list[int]) -> bool:
    """Strictly increasing; gaps are allowed."""
    return all(b > a for a, b in zip(nums, nums[1:]))


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

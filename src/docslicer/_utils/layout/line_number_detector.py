"""
step_06_line_number_detector.py

Detects margin line numbers in structured documents (legal, technical, etc.).

For each page:
1. Group words into temporary y-buckets via line_merger.
2. Per bucket, identify the leftmost word.
3. Check whether those leftmost candidates form a monotonically increasing
   sequence of integers that share a common x alignment.
4. If such a series is found, set line_number_flag = True on those words.

Thresholds live in ``LineNumberConfig``. The PDF pipeline uses the strict
defaults (``PDF_CONFIG``); the OCR pipeline passes ``OCR_CONFIG``, which is
looser because tesseract garbles digits (9 -> S, 44 -> 4A) and drops the
line-number column on some pages entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .line_merger import assign_line_id


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class LineNumberConfig:
    min_series_length: int = 3         # minimum candidates to qualify as line numbers
    x_align_tolerance: float = 7.0     # pt — max x_right spread within an x-cluster (line numbers tend to be right-aligned)
    max_number_width: float = 30.0     # pt — line-number token must be narrow
    max_missing_numbers_per_page: int = 1  # allowed skipped line numbers per page
    font_size_ratio: float = 0.85      # line numbers may not be smaller than this × doc median, otherwise likely footnotes
    min_page_coverage: float = 0.80    # line numbers must appear on at least this fraction of pages
    exclude_note_ancestors: bool = True  # drop candidates with a 'Note' struct_ancestors tag (tagged PDFs only; column never exists in OCR)
    # OCR-only KPI (inactive at the default below):
    min_line_number_line_fraction: float = 0.0  # min fraction of a page's lines that must be numbered (guards against footnote hits)


PDF_CONFIG = LineNumberConfig()

OCR_CONFIG = LineNumberConfig(
    max_missing_numbers_per_page=3,
    min_page_coverage=0.50,
    min_line_number_line_fraction=0.10,
)


# =============================================================================
# Public API
# =============================================================================

def detect_line_numbers(
    df_words: pd.DataFrame,
    config: LineNumberConfig = PDF_CONFIG,
) -> pd.DataFrame:
    """
    Adds a boolean column ``line_number_flag`` to df_words.

    Parameters
    ----------
    df_words : pd.DataFrame
        Must contain: page_number, word_id, text, x_left, x_right, y_top, y_bottom.
    config : LineNumberConfig
        Detection thresholds; defaults to the strict PDF preset.

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
    if "font_size" in df_words.columns:
        _tmp_cols.append("font_size")
    if config.exclude_note_ancestors and "struct_ancestors" in df_words.columns:
        _tmp_cols.append("struct_ancestors")

    # Assign temporary line IDs on a minimal copy (avoids mutating df_words).
    # Ignore non-LTR words before line merging so rotated/vertical footer text
    # cannot become the leftmost word for a line-number row.
    df_tmp = df_words[_tmp_cols].copy()
    if "text_orientation" in df_tmp.columns:
        orientation = df_tmp["text_orientation"].fillna("LTR").astype(str).str.upper()
        df_tmp = df_tmp[orientation == "LTR"].copy()

    df_tmp = assign_line_id(df_tmp, y_alignment="center")

    # Document-level font size median for the size KPI (None if column absent).
    font_size_threshold: float | None = None
    if "font_size" in df_words.columns:
        valid_sizes = df_words["font_size"].replace(0, pd.NA).dropna()
        if not valid_sizes.empty:
            font_size_threshold = config.font_size_ratio * float(valid_sizes.median())

    out = df_words.copy()
    out["line_number_flag"] = False

    for page_num, page_group in df_tmp.groupby("page_number"):
        flagged_ids = _detect_page_line_numbers(page_group, font_size_threshold, config)

        # QC: enough of the page's lines must carry a number, otherwise the hit
        # is likely a numbered footnote block rather than margin line numbers.
        if flagged_ids and config.min_line_number_line_fraction > 0:
            total_lines = page_group["line_id"].nunique()
            if total_lines and len(flagged_ids) / total_lines < config.min_line_number_line_fraction:
                flagged_ids = []

        if flagged_ids:
            mask = (out["page_number"] == page_num) & (out["word_id"].isin(flagged_ids))
            out.loc[mask, "line_number_flag"] = True

    # QC: line numbers must be present on at least min_page_coverage of all pages.
    # Sparse hits (e.g. a few lines starting with years like 2020/2021) are rejected.
    if out["line_number_flag"].any():
        total_pages = out["page_number"].nunique()
        pages_with_flags = out.loc[out["line_number_flag"], "page_number"].nunique()
        if pages_with_flags / total_pages < config.min_page_coverage:
            out["line_number_flag"] = False

    return out


# =============================================================================
# Internal helpers
# =============================================================================

def _detect_page_line_numbers(
    page_df: pd.DataFrame,
    font_size_threshold: float | None,
    config: LineNumberConfig,
) -> list[int]:
    """Return word_ids identified as line numbers on this page."""

    # Per line_id: pick the leftmost word — idxmin is C-level, no apply overhead
    idx = page_df.groupby("line_id")["x_left"].idxmin()
    candidates = page_df.loc[idx].reset_index(drop=True)

    if config.exclude_note_ancestors and "struct_ancestors" in candidates.columns:
        candidates = candidates[
            ~candidates["struct_ancestors"].map(_has_note_ancestor)
        ].reset_index(drop=True)

    # Vectorised positive-integer check (replaces row-wise _is_positive_integer)
    text_str = candidates["text"].astype(str).str.strip()
    is_pos_int = text_str.str.fullmatch(r"\d+", na=False) & (
        pd.to_numeric(text_str, errors="coerce").fillna(0) > 0
    )
    size_ok = (
        (candidates["font_size"] >= font_size_threshold)
        if font_size_threshold is not None and "font_size" in candidates.columns
        else True
    )
    candidates = candidates[
        is_pos_int
        & ((candidates["x_right"] - candidates["x_left"]) <= config.max_number_width)
        & size_ok
    ].copy()

    if len(candidates) < config.min_series_length:
        return []

    candidates["_num"] = candidates["text"].astype(str).str.strip().astype(int)

    # Sort by reading order (y_top ascending)
    candidates = candidates.sort_values("y_top").reset_index(drop=True)

    # Group into x-aligned clusters, then check each for a near-contiguous series.
    flagged_ids: list[int] = []
    missing_budget = config.max_missing_numbers_per_page
    for cluster in _cluster_by_x(candidates, config.x_align_tolerance):
        if len(cluster) < config.min_series_length:
            continue
        nums = cluster.sort_values("y_top")["_num"].tolist()
        ok, missing_count = _is_line_number_sequence(
            nums, max_missing=missing_budget, min_series_length=config.min_series_length
        )
        if ok:
            flagged_ids.extend(cluster["word_id"].tolist())
            missing_budget -= missing_count

    return flagged_ids


def _has_note_ancestor(ancestors: object) -> bool:
    """
    True if any struct_ancestors tag is note-like, e.g. ['Part', 'Note', 'P'].

    Matches any tag containing 'note' (case-insensitive) so unnormalized
    variants like 'Footnote' or 'Endnote' are caught too.
    """
    if not isinstance(ancestors, (list, tuple)):
        return False
    return any(isinstance(tag, str) and "note" in tag.lower() for tag in ancestors)


def _is_line_number_sequence(
    nums: list[int], *, max_missing: int, min_series_length: int
) -> tuple[bool, int]:
    """
    Return whether nums look like margin line numbers and how many numbers are missing.

    Line numbers should be contiguous. To tolerate one missed OCR/tokenization row,
    the page gets a tiny missing-number budget, e.g. [1, 2, 4, 5] is acceptable
    with max_missing=1, while TOC/page-label jumps like [2, 12, 14] are not.
    """
    if len(nums) < min_series_length:
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


def _cluster_by_x(candidates: pd.DataFrame, x_align_tolerance: float) -> list[pd.DataFrame]:
    """
    Split candidates into groups where the total x_right spread stays within
    x_align_tolerance.  Sorts by x_right before splitting.
    """
    if candidates.empty:
        return []

    sorted_c = candidates.sort_values("x_right").reset_index(drop=True)
    x_vals = sorted_c["x_right"].values

    clusters: list[pd.DataFrame] = []
    start = 0
    for i in range(1, len(x_vals)):
        if x_vals[i] - x_vals[start] > x_align_tolerance:
            clusters.append(sorted_c.iloc[start:i])
            start = i
    clusters.append(sorted_c.iloc[start:])

    return clusters

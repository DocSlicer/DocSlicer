"""
DOCX style-based block_type prefiller.

Runs before shared/ steps. Uses Word-native paragraph style metadata to
pre-assign block_type for high-confidence cases so the shared detectors
can skip those rows and focus on the remainder.

Rules (applied only to rows with no existing block_type):
  toc        — paragraph_style_id contains "toc" (case-insensitive),
               or is "TableofFigures"
  toc_heading — paragraph_style_id in {"FrontMatterHeader", "TOCHeading"},
               OR text is "table of contents / figures / tables"
               on a page that already has toc rows
  heading    — paragraph_style_id matches "Heading<N>", "Title", or "Subtitle",
               OR outline_level is 0–8, unless the heading level appears to be
               used as body/list styling throughout the document,
               EXCEPT when source_part references a footnote/endnote part
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_HEADING_STYLE_RE = re.compile(
    r"^(heading\s*\d+|title|subtitle|backmatterheading)$",
    re.IGNORECASE,
)
_HEADING_NUMBER_STYLE_RE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)

_NAMED_HEADING_STYLE_LEVELS: dict[str, int] = {
    "title": 1,
    "subtitle": 2,
    "backmatterheading": 1,
}

_HEADING_OVERUSE_RATIO = 0.35
_SUSPICIOUS_RUN_MIN_LEN = 4
_HIGH_CONSECUTIVE_RATIO = 0.70
_HIGH_CONSECUTIVE_MIN_ROWS = 6

# Exact (normalised) style IDs that map to toc_heading
_TOC_HEADER_STYLE_IDS: frozenset[str] = frozenset({"frontmatterheader", "tocheading"})

# TableofFigures style ID (no "toc" substring → explicit check)
_TABLE_OF_FIGURES_ID = "tableoffigures"

# Text values that identify a TOC header when found on a TOC page
_TOC_HEADER_TEXTS: frozenset[str] = frozenset({
    "table of contents",
    "table of content",
    "table of figures",
    "table of figure",
    "table of tables",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_style_id(s: pd.Series) -> pd.Series:
    """Lowercase and strip whitespace from a style-id series."""
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", "", regex=True)


def _unfilled(out: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows whose block_type is not yet assigned."""
    br = out["block_type"].astype(object)
    return br.isna() | (br.astype(str).str.strip() == "") | (br.astype(str).str.strip() == "nan")


def _heading_level_from_style(style_id: object) -> int | None:
    """Return 1-based Heading<N> level from a Word style id, if present."""
    if style_id is None or pd.isna(style_id):
        return None
    style_norm = str(style_id).strip().lower().replace(" ", "")
    if style_norm in _NAMED_HEADING_STYLE_LEVELS:
        return _NAMED_HEADING_STYLE_LEVELS[style_norm]

    match = _HEADING_NUMBER_STYLE_RE.match(str(style_id).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _heading_level_from_outline(outline_level: object) -> int | None:
    """Return 1-based heading level from Word's 0-based outline level."""
    try:
        if outline_level is None or pd.isna(outline_level):
            return None
        level = int(outline_level)
    except (TypeError, ValueError):
        return None
    if 0 <= level <= 8:
        return level + 1
    return None


def _heading_candidate_levels(out: pd.DataFrame) -> pd.Series:
    """Build nullable 1-based heading levels for Heading<N>/outline candidates."""
    style_levels = out.get(
        "paragraph_style_id", pd.Series(pd.NA, index=out.index)
    ).map(_heading_level_from_style)

    outline_levels = out.get(
        "outline_level", pd.Series(pd.NA, index=out.index)
    ).map(_heading_level_from_outline)

    levels = style_levels.combine_first(outline_levels)
    return pd.Series(levels, index=out.index, dtype="Int64")


def _candidate_denominator(out: pd.DataFrame) -> int:
    """Count paragraphs used as the denominator for document-level ratios."""
    if "text" not in out.columns:
        return len(out)
    text = out["text"].fillna("").astype(str).str.strip()
    return int(text.ne("").sum())


def _same_level_runs(candidate_levels: pd.Series) -> dict[int, list[list]]:
    """
    Return same-level candidate runs.

    Non-candidate rows break runs, so the detector catches repeated adjacent
    Heading3/Heading4 procedural items without punishing ordinary interleaved
    heading/content structure.
    """
    runs_by_level: dict[int, list[list]] = {}
    current_level: int | None = None
    current_indices: list = []

    def flush() -> None:
        nonlocal current_level, current_indices
        if current_level is not None and current_indices:
            runs_by_level.setdefault(current_level, []).append(current_indices)
        current_level = None
        current_indices = []

    for idx, value in candidate_levels.items():
        if pd.isna(value):
            flush()
            continue

        level = int(value)
        if current_level == level:
            current_indices.append(idx)
        else:
            flush()
            current_level = level
            current_indices = [idx]

    flush()
    return runs_by_level


def _suppressed_heading_levels(
    out: pd.DataFrame,
    heading_mask: pd.Series,
    heading_levels: pd.Series,
) -> tuple[set[int], dict[int, str]]:
    """
    Detect heading levels that are likely being used as body/list styling.

    A whole level is suppressed when any of these triggers fires:
      1. More than 35% of paragraphs are heading candidates and this level has
         at least one run of four or more adjacent candidates.
      2. This level has two or more runs of four or more adjacent candidates.
      3. At least 70% of candidates at this level appear inside same-level
         adjacent runs, with a small minimum row count guard.
    """
    candidate_levels = heading_levels.where(heading_mask)
    runs_by_level = _same_level_runs(candidate_levels)

    denominator = max(_candidate_denominator(out), 1)
    heading_ratio = float(heading_mask.sum()) / denominator

    suppressed: set[int] = set()
    reasons: dict[int, str] = {}

    for level, runs in runs_by_level.items():
        total_candidates = int((candidate_levels == level).sum())
        long_runs = [run for run in runs if len(run) >= _SUSPICIOUS_RUN_MIN_LEN]
        rows_in_adjacent_runs = sum(len(run) for run in runs if len(run) >= 2)
        consecutive_ratio = (
            rows_in_adjacent_runs / total_candidates if total_candidates else 0.0
        )

        reason_parts: list[str] = []
        if heading_ratio > _HEADING_OVERUSE_RATIO and long_runs:
            reason_parts.append("global_heading_overuse_with_long_run")
        if len(long_runs) >= 2:
            reason_parts.append("multiple_long_same_level_runs")
        if (
            total_candidates >= _HIGH_CONSECUTIVE_MIN_ROWS
            and consecutive_ratio >= _HIGH_CONSECUTIVE_RATIO
        ):
            reason_parts.append("high_same_level_consecutive_ratio")

        if reason_parts:
            suppressed.add(level)
            reasons[level] = ";".join(reason_parts)

    return suppressed, reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prefill_block_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-assign block_type using Word-native paragraph style metadata.

    Only writes to rows where block_type is not already set.
    Returns a copy with block_type updated.
    """
    out = df.copy()

    if "block_type" not in out.columns:
        out["block_type"] = pd.NA

    style_norm = _normalise_style_id(out.get("paragraph_style_id", pd.Series("", index=out.index)))

    source_part = out.get("source_part", pd.Series("", index=out.index))
    is_note_part = source_part.fillna("").astype(str).str.lower().str.contains(
        "footnote|endnote", regex=True
    )

    # ------------------------------------------------------------------
    # 1. TOC header rows — style-based (must run before generic toc check
    #    because "TOCHeading" contains "toc" and would be misclassified)
    # ------------------------------------------------------------------
    is_toc_heading_style = style_norm.isin(_TOC_HEADER_STYLE_IDS)
    out.loc[_unfilled(out) & is_toc_heading_style, "block_type"] = "toc_heading"

    # ------------------------------------------------------------------
    # 2. TOC rows
    # ------------------------------------------------------------------
    is_toc_style = (
        style_norm.str.contains("toc", regex=False)
        | (style_norm == _TABLE_OF_FIGURES_ID)
    )
    out.loc[_unfilled(out) & is_toc_style, "block_type"] = "toc"

    # ------------------------------------------------------------------
    # 3. TOC header rows — text-based (on a page that already has toc rows)
    # ------------------------------------------------------------------
    if "page_number" in out.columns and "text" in out.columns:
        toc_pages = set(
            out.loc[out["block_type"] == "toc", "page_number"].dropna().unique()
        )
        if toc_pages:
            text_norm = out["text"].fillna("").astype(str).str.strip().str.lower()
            is_toc_heading_text = text_norm.isin(_TOC_HEADER_TEXTS)
            on_toc_page = out["page_number"].isin(toc_pages)
            out.loc[_unfilled(out) & is_toc_heading_text & on_toc_page, "block_type"] = "toc_heading"

    # ------------------------------------------------------------------
    # 4. Heading rows
    # ------------------------------------------------------------------
    is_heading_style = out.get(
        "paragraph_style_id", pd.Series("", index=out.index)
    ).fillna("").astype(str).map(lambda s: bool(_HEADING_STYLE_RE.match(s)))

    outline_level = out.get("outline_level", pd.Series(pd.NA, index=out.index))
    is_heading_outline = pd.to_numeric(outline_level, errors="coerce").between(0, 8)

    is_heading = (is_heading_style | is_heading_outline) & ~is_note_part
    heading_levels = _heading_candidate_levels(out)
    fillable_heading = _unfilled(out) & is_heading

    out["docx_heading_candidate"] = is_heading
    out["docx_heading_level"] = heading_levels
    out["docx_heading_suppressed"] = False
    out["docx_heading_suppressed_reason"] = pd.NA

    suppressed_levels, suppressed_reasons = _suppressed_heading_levels(
        out,
        fillable_heading,
        heading_levels,
    )
    if suppressed_levels:
        is_suppressed_level = heading_levels.isin(suppressed_levels)
        suppress_mask = fillable_heading & is_suppressed_level
        out.loc[suppress_mask, "docx_heading_suppressed"] = True
        out.loc[suppress_mask, "docx_heading_suppressed_reason"] = (
            heading_levels.loc[suppress_mask].map(suppressed_reasons)
        )

    unsuppressed_heading = fillable_heading & ~out["docx_heading_suppressed"]
    out.loc[unsuppressed_heading, "block_type"] = "heading"

    if "heading_source" not in out.columns:
        out["heading_source"] = pd.NA
    out.loc[unsuppressed_heading, "heading_source"] = "docx"

    return out

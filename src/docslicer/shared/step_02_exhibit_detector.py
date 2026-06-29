from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .._utils.io.yaml_compilers.exhibit_patterns import ExhibitPatternConfig

# =========================
# Constants
# =========================

EXHIBIT_ROW_MAX_CHARS = 500

# =========================
# Types
# =========================

ExhibitRowPatternType = Literal[
    "multi_parens", "multi_parens_with_markers",
    "alpha_with_parens", "alpha_with_parens_with_markers",
    "number_with_parens", "number_with_parens_with_markers",
    "dotted_with_parens", "dotted_with_parens_with_markers",
    "numeric_or_dotted", "numeric_or_dotted_with_markers",
    "exhibit_prefix_row", "exhibit_prefix_row_with_markers",
    "ex_code_row", "ex_code_row_with_markers",
    "subpart_not_applicable", "subpart_not_applicable_with_markers",
    "hundred_series_exhibit", "hundred_series_exhibit_with_markers",
]
ExhibitHeaderPatternType = Literal[
    "item_any_exhibits", "exhibit_index", "index_to_exhibits", "exhibits_only",
    "paren_letter_exhibits"
]


@dataclass(frozen=True)
class ExhibitRowCandidate:
    line_id: int
    exhibit_row_pattern: Optional[ExhibitRowPatternType]
    exhibit_number: Optional[str]
    pattern_strength: Optional[str]  # "strong" or "weak"
    has_link: bool
    table_id: Optional[Any]

    # Layout fields (used for fingerprint-based grouping and cross-page merging)
    x_left: Optional[float]
    height: Optional[float]
    font_size: Optional[float]
    text_align: Optional[str]
    page_number: Optional[int]


@dataclass(frozen=True)
class LayoutFingerprint:
    """Layout fingerprint for grouping non-table rows by visual similarity."""
    x_left: float
    height: float
    font_size: float

    @classmethod
    def from_candidate(cls, candidate: ExhibitRowCandidate) -> Optional[LayoutFingerprint]:
        """Create a fingerprint from a candidate's layout values."""
        if candidate.x_left is None or candidate.font_size is None:
            return None
        return cls(
            x_left=candidate.x_left,
            height=candidate.height or 0.0,
            font_size=candidate.font_size,
        )

    def matches(
        self,
        other: LayoutFingerprint,
        left_tolerance: float = 5.0,
        height_tolerance: float = 2.0,
        font_tolerance: float = 0.5,
    ) -> bool:
        """Return True if *other* is within tolerance on all three dimensions."""
        return (
            abs(self.x_left - other.x_left) <= left_tolerance
            and abs(self.height - other.height) <= height_tolerance
            and abs(self.font_size - other.font_size) <= font_tolerance
        )


@dataclass(frozen=True)
class ExhibitSegment:
    """
    A localized cluster of exhibit candidate rows.

    Segments are scored and filtered to produce the final exhibit annotations.
    Two clustering strategies are used (mutually exclusive per segment):

    - **Table-based**: all rows share the same ``table_id``.
    - **Fingerprint-based**: rows are contiguous and share the same layout
      fingerprint (x_left, height, font_size within tolerance).
    """
    segment_id: int
    start_line_id: int
    end_line_id: int
    n_rows: int
    n_candidates: int
    candidate_ratio: float
    max_consecutive_candidates: int
    has_exhibit_heading_nearby: bool
    has_exhibit_number_header: bool
    has_other_segment_above: bool
    n_links: int
    nearby_header_line_ids: List[int]
    above_exhibit_segment_id: int
    table_id: Optional[str]
    fingerprint: Optional[LayoutFingerprint]

    @property
    def is_table_based(self) -> bool:
        return self.table_id is not None

    @property
    def is_fingerprint_based(self) -> bool:
        return self.fingerprint is not None


@dataclass
class ExhibitScore:
    """Confidence score and metadata for a single exhibit segment."""
    segment: ExhibitSegment
    confidence_score: float
    is_disqualified: bool
    disqualification_reason: Optional[str]

    # Score components
    header_nearby_score: float
    number_header_score: float
    segment_above_score: float
    links_score: float
    consecutive_score: float
    strong_pattern_score: float
    weak_pattern_score: float

    # Debug info
    root_segment_id: int
    root_has_header: bool
    n_strong_patterns: int
    n_weak_patterns: int


# =========================
# Private Helpers
# =========================

def _safe_bool01(x: Any) -> bool:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        if isinstance(x, str) and x.strip().isdigit():
            return bool(int(x.strip()))
        if isinstance(x, (int, float)):
            return bool(int(x))
        return bool(x)
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(x)
    except Exception:
        return None


def _safe_str_or_none(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return str(x).strip() or None


# ==========================================
# Exhibit Header Candidates
# ==========================================

def _identify_exhibit_heading_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    *,
    max_len: int = 150,
) -> pd.DataFrame:
    """
    Tag rows that look like exhibit section headers.

    Adds ``exhibit_heading_candidate`` (pattern name or ``pd.NA``).
    Examples of matching text: ``"Item 15. Exhibits"``, ``"EXHIBIT INDEX"``.
    """
    out = df.copy()
    out["exhibit_heading_candidate"] = pd.NA

    if "text" not in out.columns:
        return out

    texts = out["text"].astype(str).tolist()
    results: List[Any] = [pd.NA] * len(texts)

    for i, raw_text in enumerate(texts):
        txt = raw_text.strip()
        if not txt or len(txt) > max_len:
            continue
        for pattern in exhibit_config.header_patterns:
            if pattern.compiled.match(txt):
                results[i] = pattern.name
                break  # First match wins

    out["exhibit_heading_candidate"] = results
    return out


# ==========================================
# Exhibit Row Candidates
# ==========================================

def _check_exhibit_row_match(
    text: str,
    pattern: Any,
    pattern_name: str,
) -> Tuple[bool, Optional[str]]:
    """
    Test whether *text* matches an exhibit row pattern and extract its number.

    Footnote markers (``*``, ``†``, ``‡``, ``§``, ``¶``, ``#``, …) are handled
    automatically: the pattern compiler generates ``_with_markers`` variants.

    Returns:
        ``(is_valid, exhibit_number)``
    """
    txt = (text or "").strip()
    if not txt:
        return False, None

    match = pattern.compiled.match(txt)
    if not match:
        return False, None

    exhibit_number = None
    base = pattern_name.replace("_with_markers", "")

    if base in ("exhibit_prefix_row", "ex_code_row", "hundred_series_exhibit"):
        try:
            exhibit_number = match.group("code")
        except (IndexError, AttributeError):
            exhibit_number = match.group(0).strip().split()[0]

    elif base in (
        "multi_parens", "alpha_with_parens",
        "number_with_parens", "dotted_with_parens", "numeric_or_dotted",
    ):
        matched_text = match.group(0).strip()
        exhibit_number = matched_text.split()[0] if matched_text else matched_text
        # Strip footnote markers (mirrors FOOTNOTE_MARKERS in exhibit_patterns.py)
        if exhibit_number:
            for marker in "*†‡§¶#+^■●▲▼◆◇○□△▽◊~":
                exhibit_number = exhibit_number.replace(marker, "")

    elif base == "subpart_not_applicable":
        exhibit_number = txt.split()[0]

    return True, exhibit_number


def _add_exhibit_row_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    *,
    include_debug_cols: bool = True,
) -> Tuple[pd.DataFrame, List[ExhibitRowCandidate]]:
    """
    Detect exhibit row candidates and return an annotated DataFrame plus
    structured candidate objects.

    Adds ``exhibit_row_candidate`` (matched pattern name or ``pd.NA``).
    When ``include_debug_cols=True`` also adds ``exhibit_number`` and
    ``pattern_strength``.

    A row is accepted when text is non-empty, within :data:`EXHIBIT_ROW_MAX_CHARS`,
    and matches one of the exhibit row patterns from the YAML config.

    Returns:
        ``(annotated_df, candidates)`` — one :class:`ExhibitRowCandidate` per
        matched row.
    """
    out = df.copy()
    out["exhibit_row_candidate"] = pd.NA
    if include_debug_cols:
        out["exhibit_number"] = pd.NA
        out["pattern_strength"] = pd.NA

    if "text" not in out.columns:
        return out, []

    # Pre-extract columns as lists for O(1) positional access
    texts = out["text"].astype(str).tolist()
    line_ids_list   = out["line_id"].tolist()   if "line_id"   in out.columns else list(range(len(out)))
    has_links_list  = out["has_link"].tolist()  if "has_link"  in out.columns else [None] * len(out)
    table_ids_list  = out["table_id"].tolist()  if "table_id"  in out.columns else [None] * len(out)
    x_lefts_list    = out["x_left"].tolist()    if "x_left"    in out.columns else [None] * len(out)
    heights_list    = out["height"].tolist()    if "height"    in out.columns else [None] * len(out)
    font_sizes_list = out["font_size"].tolist() if "font_size" in out.columns else [None] * len(out)
    text_aligns_list= out["text_align"].tolist()if "text_align"in out.columns else [None] * len(out)
    page_nums_list  = out["page_number"].tolist()if "page_number" in out.columns else [None] * len(out)

    row_candidate_results: List[Any] = [pd.NA] * len(texts)
    exhibit_number_results: List[Any] = [pd.NA] * len(texts)
    pattern_strength_results: List[Any] = [pd.NA] * len(texts)

    candidates: List[ExhibitRowCandidate] = []

    for i, raw_text in enumerate(texts):
        text = raw_text.strip()
        if not text or len(text) > EXHIBIT_ROW_MAX_CHARS:
            continue

        matched_name = matched_strength = exhibit_number = None

        for pattern in exhibit_config.row_patterns:
            is_valid, number = _check_exhibit_row_match(text, pattern, pattern.name)
            if is_valid:
                matched_name = pattern.name
                matched_strength = pattern.strength
                exhibit_number = number
                break  # First match wins

        if matched_name is None:
            continue

        row_candidate_results[i] = matched_name
        if include_debug_cols:
            exhibit_number_results[i] = exhibit_number
            pattern_strength_results[i] = matched_strength

        candidates.append(ExhibitRowCandidate(
            line_id=int(line_ids_list[i]),
            exhibit_row_pattern=matched_name,  # type: ignore[arg-type]
            exhibit_number=exhibit_number,
            pattern_strength=matched_strength,
            has_link=_safe_bool01(has_links_list[i]),
            table_id=table_ids_list[i],
            x_left=_safe_float(x_lefts_list[i]),
            height=_safe_float(heights_list[i]),
            font_size=_safe_float(font_sizes_list[i]),
            text_align=_safe_str_or_none(text_aligns_list[i]),
            page_number=_safe_int(page_nums_list[i]),
        ))

    out["exhibit_row_candidate"] = row_candidate_results
    if include_debug_cols:
        out["exhibit_number"] = exhibit_number_results
        out["pattern_strength"] = pattern_strength_results

    return out, candidates


# ==========================================
# Build Exhibit Segments
# ==========================================


def _detect_exhibit_number_header_tables(df: pd.DataFrame) -> Set[Any]:
    """
    Return the set of ``table_id`` values whose first row contains
    ``"Exhibit Number"`` or ``"Exhibit No"`` — a strong signal that the table
    is an exhibit listing.
    """
    if "table_id" not in df.columns or "text" not in df.columns:
        return set()

    sort_col = "line_id" if "line_id" in df.columns else None
    src = df.dropna(subset=["table_id"])
    if sort_col:
        src = src.sort_values(sort_col)
    first_rows = src.drop_duplicates(subset=["table_id"], keep="first")
    mask = first_rows["text"].astype(str).str.strip().str.match(
        r"^\s*exhibit\s+(?:number|no\.?)\s*$", case=False
    )
    return set(first_rows.loc[mask, "table_id"])


def _calculate_max_consecutive(
    sorted_line_ids: List[int], candidate_set: Set[int]
) -> int:
    """Return the longest run of consecutive candidate line IDs."""
    max_run = current_run = 0
    for lid in sorted_line_ids:
        if lid in candidate_set:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _check_exhibit_heading_nearby(
    heading_arr: Optional[np.ndarray],
    line_ids: np.ndarray,
    start_pos: int,
    lookback: int,
) -> Tuple[bool, List[int]]:
    """Return ``(has_header, header_line_ids)`` scanning *lookback* positions before *start_pos*."""
    if heading_arr is None or start_pos == 0:
        return False, []
    begin = max(0, start_pos - lookback)
    header_line_ids = [
        int(line_ids[i])
        for i in range(begin, start_pos)
        if isinstance(heading_arr[i], str)
    ]
    return bool(header_line_ids), header_line_ids


def _check_other_segment_above(
    line_ids: np.ndarray,
    start_pos: int,
    processed_segments: List[ExhibitSegment],
    lookback: int,
) -> Tuple[bool, int]:
    """
    Return ``(has_segment_above, segment_id_above)`` scanning *lookback* positions before *start_pos*.
    ``segment_id_above`` is ``-1`` when none found.
    """
    if not processed_segments or start_pos == 0:
        return False, -1
    begin = max(0, start_pos - lookback)
    before_ids = {int(line_ids[i]) for i in range(begin, start_pos)}
    if not before_ids:
        return False, -1
    for seg in reversed(processed_segments):  # Most recent first
        if any(seg.start_line_id <= lid <= seg.end_line_id for lid in before_ids):
            return True, seg.segment_id
    return False, -1


def _expand_segment_by_fingerprint(
    seed_pos: int,
    n_rows: int,
    target: LayoutFingerprint,
    line_ids: np.ndarray,
    x_left_arr: np.ndarray,
    height_arr: np.ndarray,
    font_size_arr: np.ndarray,
    left_tolerance: float,
    height_tolerance: float,
    font_tolerance: float,
) -> List[int]:
    """
    Expand from *seed_pos* in both directions, collecting contiguous rows
    whose layout fingerprint matches *target* within tolerance.

    Returns:
        Sorted list of matching line IDs (including the seed).
    """
    collected = {seed_pos}
    for direction in (-1, 1):
        current = seed_pos + direction
        while 0 <= current < n_rows:
            left = x_left_arr[current]
            font = font_size_arr[current]
            if np.isnan(left) or np.isnan(font):
                break
            h = height_arr[current]
            if np.isnan(h):
                h = 0.0
            if (
                abs(target.x_left - left) > left_tolerance
                or abs(target.font_size - font) > font_tolerance
                or abs(target.height - h) > height_tolerance
            ):
                break
            collected.add(current)
            current += direction
    return [int(line_ids[i]) for i in sorted(collected)]


def _build_exhibit_segments(
    df: pd.DataFrame,
    candidates: List[ExhibitRowCandidate],
    *,
    left_tolerance: float,
    height_tolerance: float,
    font_tolerance: float,
    header_lookback: int,
    segment_lookback: int,
) -> List[ExhibitSegment]:
    """
    Cluster candidates into :class:`ExhibitSegment` objects.

    Iterates the DataFrame in ``line_id`` order. For each unprocessed candidate:

    - **Table-based**: when the row has a ``table_id``, all rows in that table
      form one segment.
    - **Fingerprint-based**: when there is no ``table_id``, expand up and down
      while adjacent rows share the same layout fingerprint.
    """
    if df.empty or "line_id" not in df.columns or "exhibit_row_candidate" not in df.columns:
        return []

    df_sorted = df.sort_values("line_id").reset_index(drop=True)
    n = len(df_sorted)

    # Pre-extract all needed columns as arrays for O(1) positional access
    line_ids: np.ndarray = df_sorted["line_id"].to_numpy()
    line_id_to_pos: Dict[int, int] = {int(lid): pos for pos, lid in enumerate(line_ids)}

    heading_arr: Optional[np.ndarray] = (
        df_sorted["exhibit_heading_candidate"].to_numpy()
        if "exhibit_heading_candidate" in df_sorted.columns else None
    )
    has_link_arr: np.ndarray = (
        df_sorted["has_link"].map(_safe_bool01).to_numpy()
        if "has_link" in df_sorted.columns
        else np.zeros(n, dtype=bool)
    )
    table_id_arr: Optional[np.ndarray] = (
        df_sorted["table_id"].to_numpy() if "table_id" in df_sorted.columns else None
    )
    x_left_arr: np.ndarray = (
        pd.to_numeric(df_sorted["x_left"], errors="coerce").to_numpy()
        if "x_left" in df_sorted.columns else np.full(n, np.nan)
    )
    height_arr: np.ndarray = (
        pd.to_numeric(df_sorted["height"], errors="coerce").to_numpy()
        if "height" in df_sorted.columns else np.zeros(n)
    )
    font_size_arr: np.ndarray = (
        pd.to_numeric(df_sorted["font_size"], errors="coerce").to_numpy()
        if "font_size" in df_sorted.columns else np.full(n, np.nan)
    )

    candidates_by_line_id: Dict[int, ExhibitRowCandidate] = {
        c.line_id: c for c in candidates
    }
    exhibit_number_header_tables = _detect_exhibit_number_header_tables(df)

    # Pre-group table row positions by table_id to avoid per-segment boolean filter
    table_line_ids_by_tid: Dict[Any, List[int]] = {}
    if table_id_arr is not None:
        for pos, tid in enumerate(table_id_arr):
            if tid is not None and not (isinstance(tid, float) and np.isnan(tid)) and str(tid).strip():
                table_line_ids_by_tid.setdefault(tid, []).append(int(line_ids[pos]))

    segments: List[ExhibitSegment] = []
    segment_id_counter = 0
    processed_line_ids: Set[int] = set()
    processed_table_ids: Set[Any] = set()

    for pos in range(n):
        line_id = int(line_ids[pos])

        if line_id in processed_line_ids or line_id not in candidates_by_line_id:
            continue

        table_id = table_id_arr[pos] if table_id_arr is not None else None
        has_table = (
            table_id is not None
            and not (isinstance(table_id, float) and np.isnan(table_id))
            and str(table_id).strip() != ""
        )

        if has_table and table_id not in processed_table_ids:
            # === TABLE-BASED SEGMENT ===
            all_line_ids = sorted(table_line_ids_by_tid.get(table_id, []))
            if not all_line_ids:
                continue

            candidate_line_ids = {lid for lid in all_line_ids if lid in candidates_by_line_id}
            start_line_id = all_line_ids[0]
            end_line_id = all_line_ids[-1]
            n_rows = len(all_line_ids)
            n_candidates = len(candidate_line_ids)

            start_pos = line_id_to_pos.get(start_line_id, pos)
            has_header, header_line_ids = _check_exhibit_heading_nearby(
                heading_arr, line_ids, start_pos, header_lookback
            )
            if has_header:
                has_above, above_id = False, -1
            else:
                has_above, above_id = _check_other_segment_above(
                    line_ids, start_pos, segments, segment_lookback
                )

            n_links = int(sum(
                has_link_arr[line_id_to_pos[lid]]
                for lid in all_line_ids
                if lid in line_id_to_pos
            ))

            segments.append(ExhibitSegment(
                segment_id=segment_id_counter,
                start_line_id=start_line_id,
                end_line_id=end_line_id,
                n_rows=n_rows,
                n_candidates=n_candidates,
                candidate_ratio=n_candidates / n_rows if n_rows else 0.0,
                max_consecutive_candidates=_calculate_max_consecutive(
                    all_line_ids, candidate_line_ids
                ),
                has_exhibit_heading_nearby=has_header,
                has_exhibit_number_header=table_id in exhibit_number_header_tables,
                has_other_segment_above=has_above,
                n_links=n_links,
                nearby_header_line_ids=header_line_ids,
                above_exhibit_segment_id=above_id,
                table_id=str(table_id),
                fingerprint=None,
            ))
            segment_id_counter += 1
            processed_line_ids.update(all_line_ids)
            processed_table_ids.add(table_id)

        elif not has_table:
            # === FINGERPRINT-BASED SEGMENT ===
            fingerprint = LayoutFingerprint.from_candidate(candidates_by_line_id[line_id])
            if fingerprint is None:
                continue

            segment_line_ids = _expand_segment_by_fingerprint(
                pos, n, fingerprint, line_ids,
                x_left_arr, height_arr, font_size_arr,
                left_tolerance, height_tolerance, font_tolerance,
            )
            if not segment_line_ids:
                continue

            candidate_line_ids = {
                lid for lid in segment_line_ids if lid in candidates_by_line_id
            }
            start_line_id = segment_line_ids[0]
            end_line_id = segment_line_ids[-1]
            n_rows = len(segment_line_ids)
            n_candidates = len(candidate_line_ids)

            n_links = int(sum(
                has_link_arr[line_id_to_pos[lid]]
                for lid in segment_line_ids
                if lid in line_id_to_pos
            ))

            start_pos = line_id_to_pos.get(start_line_id, pos)
            has_header, header_line_ids = _check_exhibit_heading_nearby(
                heading_arr, line_ids, start_pos, header_lookback
            )
            if has_header:
                has_above, above_id = False, -1
            else:
                has_above, above_id = _check_other_segment_above(
                    line_ids, start_pos, segments, segment_lookback
                )

            segments.append(ExhibitSegment(
                segment_id=segment_id_counter,
                start_line_id=start_line_id,
                end_line_id=end_line_id,
                n_rows=n_rows,
                n_candidates=n_candidates,
                candidate_ratio=n_candidates / n_rows if n_rows else 0.0,
                max_consecutive_candidates=_calculate_max_consecutive(
                    segment_line_ids, candidate_line_ids
                ),
                has_exhibit_heading_nearby=has_header,
                has_exhibit_number_header=False,
                has_other_segment_above=has_above,
                n_links=n_links,
                nearby_header_line_ids=header_line_ids,
                above_exhibit_segment_id=above_id,
                table_id=None,
                fingerprint=fingerprint,
            ))
            segment_id_counter += 1
            processed_line_ids.update(segment_line_ids)

    return segments


# ==========================================
# Score and Filter Segments
# ==========================================

def _find_root_segment(
    segment: ExhibitSegment,
    segments_by_id: Dict[int, ExhibitSegment],
) -> ExhibitSegment:
    """Follow the chain upward to the root (a segment with no segment above it)."""
    current = segment
    visited = {segment.segment_id}

    while current.has_other_segment_above:
        above_id = current.above_exhibit_segment_id
        if above_id < 0 or above_id not in segments_by_id or above_id in visited:
            break
        visited.add(above_id)
        current = segments_by_id[above_id]

    return current


def _score_exhibit_segments(
    segments: List[ExhibitSegment],
    candidates: List[ExhibitRowCandidate],
    *,
    require_header_in_chain: bool = True,
    links_weight_per_link: float = 0.5,
    links_score_cap: float = 2.0,
    consecutive_weight_per_count: float = 0.2,
    consecutive_score_cap: float = 1.5,
    strong_pattern_weight: float = 1.0,
    strong_pattern_cap: float = 2.0,
    weak_first_weight: float = 1.0,
    weak_additional_weight: float = 0.1,
    weak_pattern_cap: float = 1.5,
) -> List[ExhibitScore]:
    """
    Score exhibit segments with disqualification and confidence scoring.

    **Disqualification rules** (when ``require_header_in_chain=True``):

    1. Segment has no header nearby *and* no segment above → disqualified.
    2. Segment belongs to a chain whose root has no header → disqualified.

    **Confidence scoring components:**

    - ``has_exhibit_heading_nearby``: +1
    - ``has_exhibit_number_header``: +1
    - ``has_other_segment_above`` (continuation): +1
    - Links: ``links_weight_per_link`` per link, capped at ``links_score_cap``
    - Consecutive candidates: ``consecutive_weight_per_count`` per run,
      capped at ``consecutive_score_cap``
    - Strong patterns: ``strong_pattern_weight`` per element, capped at
      ``strong_pattern_cap``
    - Weak patterns: ``weak_first_weight`` for the first, then
      ``weak_additional_weight`` each, capped at ``weak_pattern_cap``

    Returns:
        List of :class:`ExhibitScore` objects — includes both accepted and
        disqualified segments so callers can inspect the full picture.
    """
    if not segments:
        return []

    segments_by_id = {seg.segment_id: seg for seg in segments}
    sorted_candidates = sorted(candidates, key=lambda c: c.line_id)

    scores: List[ExhibitScore] = []

    for segment in segments:
        # --- Disqualification ---
        is_disqualified = False
        disqualification_reason = None

        root = _find_root_segment(segment, segments_by_id)
        root_has_header = root.has_exhibit_heading_nearby

        if require_header_in_chain:
            if not segment.has_exhibit_heading_nearby and not segment.has_other_segment_above:
                is_disqualified = True
                disqualification_reason = "No header nearby and not part of a chain"
            elif segment.has_other_segment_above and not root_has_header:
                is_disqualified = True
                disqualification_reason = (
                    f"Chain root (segment {root.segment_id}) has no header"
                )

        # --- Confidence scoring (computed even for disqualified segments) ---
        header_nearby_score = 1.0 if segment.has_exhibit_heading_nearby else 0.0
        number_header_score = 1.0 if segment.has_exhibit_number_header else 0.0
        segment_above_score = 1.0 if segment.has_other_segment_above else 0.0
        links_score = min(segment.n_links * links_weight_per_link, links_score_cap)
        consecutive_score = min(
            segment.max_consecutive_candidates * consecutive_weight_per_count,
            consecutive_score_cap,
        )

        segment_candidates = [
            c for c in sorted_candidates
            if segment.start_line_id <= c.line_id <= segment.end_line_id
        ]
        n_strong = sum(1 for c in segment_candidates if c.pattern_strength == "strong")
        n_weak = sum(1 for c in segment_candidates if c.pattern_strength == "weak")

        strong_pattern_score = min(n_strong * strong_pattern_weight, strong_pattern_cap)
        if n_weak == 0:
            weak_pattern_score = 0.0
        elif n_weak == 1:
            weak_pattern_score = weak_first_weight
        else:
            weak_pattern_score = weak_first_weight + (n_weak - 1) * weak_additional_weight
        weak_pattern_score = min(weak_pattern_score, weak_pattern_cap)

        confidence_score = (
            header_nearby_score
            + number_header_score
            + segment_above_score
            + links_score
            + consecutive_score
            + strong_pattern_score
            + weak_pattern_score
        )

        scores.append(ExhibitScore(
            segment=segment,
            confidence_score=confidence_score,
            is_disqualified=is_disqualified,
            disqualification_reason=disqualification_reason,
            header_nearby_score=header_nearby_score,
            number_header_score=number_header_score,
            segment_above_score=segment_above_score,
            links_score=links_score,
            consecutive_score=consecutive_score,
            strong_pattern_score=strong_pattern_score,
            weak_pattern_score=weak_pattern_score,
            root_segment_id=root.segment_id,
            root_has_header=root_has_header,
            n_strong_patterns=n_strong,
            n_weak_patterns=n_weak,
        ))

    return scores


def _filter_accepted_exhibit_segments(
    scores: List[ExhibitScore],
    min_score_threshold: float,
) -> List[ExhibitScore]:
    """Return scores that are not disqualified and meet *min_score_threshold*."""
    return [
        s for s in scores
        if not s.is_disqualified and s.confidence_score >= min_score_threshold
    ]


# ==========================================
# DEBUG UTILITIES
# ==========================================
#
# Enable debug mode in the main pipeline to print segment info and add
# intermediate columns to the output:
#
#   df = detect_and_mark_exhibits(df, config, debug=True)
#

def print_exhibit_segments(
    segments: List[ExhibitSegment],
    df: Optional[pd.DataFrame] = None,
    scores: Optional[List[ExhibitScore]] = None,
    *,
    show_text: bool = True,
    max_text_len: int = 80,
) -> None:
    """
    Print a formatted debug view of all exhibit segments.

    Args:
        segments: Segments to display.
        df: Optional DataFrame — when provided, row text is shown per segment.
        scores: Optional scores from the scoring step.
        show_text: Include text content in the output.
        max_text_len: Truncation length for displayed text.
    """
    if not segments:
        print("No segments found.")
        return

    score_by_id = {s.segment.segment_id: s for s in scores} if scores else {}

    print(f"\n{'='*100}")
    print(f"EXHIBIT SEGMENTS ({len(segments)} total)")
    print(f"{'='*100}\n")

    for i, seg in enumerate(segments, 1):
        print(f"Segment #{i} (ID: {seg.segment_id})")
        print(f"  Lines: {seg.start_line_id} -> {seg.end_line_id} ({seg.n_rows} rows)")
        print(f"  Candidates: {seg.n_candidates}/{seg.n_rows} ({seg.candidate_ratio:.1%})")
        print(f"  Max consecutive: {seg.max_consecutive_candidates}")
        print(f"  Links: {seg.n_links}")

        if seg.is_table_based:
            print(f"  Type: TABLE (table_id={seg.table_id})")
        elif seg.is_fingerprint_based:
            fp = seg.fingerprint
            print(
                f"  Type: FINGERPRINT "
                f"(x_left={fp.x_left:.1f}, height={fp.height:.1f}, font={fp.font_size:.1f})"
            )
        else:
            print("  Type: UNKNOWN")

        if seg.has_exhibit_heading_nearby:
            print(f"  Has exhibit header nearby (lines: {seg.nearby_header_line_ids})")
        if seg.has_exhibit_number_header:
            print("  Has exhibit number header")
        if seg.has_other_segment_above:
            print(f"  Connected to segment {seg.above_exhibit_segment_id} above")

        if seg.segment_id in score_by_id:
            score = score_by_id[seg.segment_id]
            status = "DISQUALIFIED" if score.is_disqualified else "ACCEPTED"
            print(f"  Score: {score.confidence_score:.2f} [{status}]")
            if score.is_disqualified:
                print(f"    Reason: {score.disqualification_reason}")
            else:
                components = []
                if score.header_nearby_score:
                    components.append(f"header={score.header_nearby_score:.1f}")
                if score.number_header_score:
                    components.append(f"number_hdr={score.number_header_score:.1f}")
                if score.segment_above_score:
                    components.append(f"above={score.segment_above_score:.1f}")
                if score.links_score:
                    components.append(f"links={score.links_score:.1f}")
                if score.consecutive_score:
                    components.append(f"consec={score.consecutive_score:.1f}")
                if score.strong_pattern_score:
                    components.append(f"strong={score.strong_pattern_score:.1f}")
                if score.weak_pattern_score:
                    components.append(f"weak={score.weak_pattern_score:.1f}")
                if components:
                    print(f"    Components: {', '.join(components)}")
                print(
                    f"    Root segment: {score.root_segment_id}, "
                    f"root_has_header={score.root_has_header}"
                )
                print(f"    Patterns: {score.n_strong_patterns} strong, {score.n_weak_patterns} weak")

        if show_text and df is not None and "line_id" in df.columns and "text" in df.columns:
            rows = df[
                (df["line_id"] >= seg.start_line_id)
                & (df["line_id"] <= seg.end_line_id)
            ]
            if not rows.empty:
                print("  Text content:")
                for _, row in rows.iterrows():
                    text = str(row["text"]) if pd.notna(row["text"]) else ""
                    text = text.replace("\n", " ").replace("\r", " ")
                    if len(text) > max_text_len:
                        text = text[: max_text_len - 3] + "..."
                    print(f"    [{row['line_id']}] {text}")

        print()

    print(f"{'='*100}\n")


# ==========================================
# PUBLIC API
# ==========================================

def detect_and_mark_exhibits(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    *,
    left_tolerance: float = 5.0,
    height_tolerance: float = 2.0,
    font_tolerance: float = 0.5,
    header_lookback: int = 3,
    segment_lookback: int = 5,
    min_score_threshold: float = 2.0,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run the complete exhibit detection pipeline and annotate the DataFrame.

    Expects columns: ``text``, ``line_id``, ``has_link``, ``table_id``,
    ``block_type`` (optional — rows classified as ``'toc'`` or ``'toc_heading'``
    are skipped).

    Pipeline steps:

    1. Skip rows already classified as ``'toc'`` or ``'toc_heading'``.
    2. Identify exhibit section header candidates.
    3. Identify exhibit row candidates.
    4. Cluster candidates into :class:`ExhibitSegment` objects.
    5. Score and filter segments.
    6. Write ``block_type = 'exhibit'`` or ``'exhibit_heading'`` for accepted rows.

    Args:
        df: Input DataFrame.
        exhibit_config: Compiled exhibit patterns from YAML.
        left_tolerance: Fingerprint tolerance for left coordinate (default: 5.0).
        height_tolerance: Fingerprint tolerance for row height (default: 2.0).
        font_tolerance: Fingerprint tolerance for font size (default: 0.5).
        header_lookback: Rows to scan back for exhibit headers (default: 3).
        segment_lookback: Rows to scan back for neighbouring segments (default: 5).
        min_score_threshold: Minimum confidence score to accept a segment
            (default: 2.0).
        debug: When ``True``, adds intermediate ``exhibit_*`` columns to the
            output and prints segment details to stdout (default: ``False``).

    Returns:
        DataFrame with ``block_type`` set to ``'exhibit'`` or
        ``'exhibit_heading'`` for detected rows.  When ``debug=True``, also
        includes ``exhibit_heading_candidate``, ``exhibit_row_candidate``,
        ``exhibit_number``, and ``pattern_strength`` columns.
    """
    # STEP 0: Skip rows already classified as TOC
    if "block_type" in df.columns:
        df_work = df[~df["block_type"].isin(["toc", "toc_heading"])].copy()
    else:
        df_work = df.copy()

    if df_work.empty:
        if "block_type" not in df.columns:
            df["block_type"] = pd.NA
        return df

    # STEP 1: Header candidates
    df_work = _identify_exhibit_heading_candidates(df_work, exhibit_config)

    # STEP 2: Row candidates
    df_work, candidates = _add_exhibit_row_candidates(
        df_work, exhibit_config, include_debug_cols=True,
    )

    # STEP 3: Segments
    segments = _build_exhibit_segments(
        df_work, candidates,
        left_tolerance=left_tolerance,
        height_tolerance=height_tolerance,
        font_tolerance=font_tolerance,
        header_lookback=header_lookback,
        segment_lookback=segment_lookback,
    )

    # STEP 4: Score
    scores = _score_exhibit_segments(segments, candidates)

    # STEP 5: Filter
    accepted = _filter_accepted_exhibit_segments(scores, min_score_threshold)

    if debug:
        print_exhibit_segments(segments, df=df_work, scores=scores)

    # STEP 6: Annotate accepted segments
    if "block_type" not in df.columns:
        df["block_type"] = pd.NA

    for score_obj in accepted:
        seg = score_obj.segment
        for hid in seg.nearby_header_line_ids:
            df.loc[df["line_id"] == hid, "block_type"] = "exhibit_heading"
        mask = (df["line_id"] >= seg.start_line_id) & (df["line_id"] <= seg.end_line_id)
        df.loc[mask, "block_type"] = "exhibits"

    # STEP 7: Fill gaps between the chain root's header and the last segment
    if accepted:
        segments_by_root: Dict[int, List[ExhibitSegment]] = {}
        for score_obj in accepted:
            segments_by_root.setdefault(score_obj.root_segment_id, []).append(score_obj.segment)

        for chain_segs in segments_by_root.values():
            chain_segs = sorted(chain_segs, key=lambda s: s.start_line_id)
            root_seg = chain_segs[0]
            last_seg = chain_segs[-1]

            if not (root_seg.has_exhibit_heading_nearby and root_seg.nearby_header_line_ids):
                continue

            earliest_header = min(root_seg.nearby_header_line_ids)
            header_set = set(root_seg.nearby_header_line_ids)
            in_range = (df["line_id"] >= earliest_header) & (df["line_id"] <= last_seg.end_line_id)

            df.loc[in_range & df["line_id"].isin(header_set), "block_type"] = "exhibit_heading"
            df.loc[
                in_range
                & ~df["line_id"].isin(header_set)
                & (df["block_type"].isna() | df["block_type"].isin(["exhibits", "exhibit_heading"])),
                "block_type",
            ] = "exhibits"

    # STEP 8: Merge debug columns back into the original DataFrame
    if debug:
        debug_cols = [
            c for c in [
                "exhibit_heading_candidate", "exhibit_row_candidate",
                "exhibit_number", "pattern_strength",
            ]
            if c in df_work.columns
        ]
        if debug_cols:
            debug_df = df_work[["line_id"] + debug_cols].copy()
            df = df.merge(debug_df, on="line_id", how="left", suffixes=("", "_new"))
            for col in debug_cols:
                if f"{col}_new" in df.columns:
                    df[col] = df[f"{col}_new"]
                    df = df.drop(columns=[f"{col}_new"])

    return df

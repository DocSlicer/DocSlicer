from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from .._utils.yaml_compilers.page_label_patterns import PageLabelPatternConfig

# =========================
# Config
# =========================

# Header anchors — clean, separate patterns for each case
TOC_HEADER_PATTERNS = [
    # 1) Classic "Table of Contents"
    re.compile(r'^\s*table\s+of\s+contents?\b', re.IGNORECASE),

    # 2) Simple "Contents"
    re.compile(r'^\s*contents?\b', re.IGNORECASE),

    # 3) PRIMARY rule — starts WITH or ends WITH INDEX
    re.compile(r'^(?:\s*index\b.*|.*\bindex\s*)$', re.IGNORECASE),
]

# dot leaders = repeated visual leader glyphs, allowing whitespace between glyphs
_DOT_LEADER_CHARS = r".…⋯∙·•‧"
_DOT_LEADERS_RE = re.compile(rf"[{re.escape(_DOT_LEADER_CHARS)}](?:\s*[{re.escape(_DOT_LEADER_CHARS)}]){{2,}}")

_HIDDEN_BLOCK_TYPES = frozenset({
    "toc",
    "toc_heading",
    "heading",
    "image",
    "hr",
    "page_label",
})


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
    if x is None:
        return None
    try:
        if bool(pd.isna(x)):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    return s or None


def _is_hidden_block_type(value: Any) -> bool:
    s = _safe_str_or_none(value)
    return bool(s and s.lower() in _HIDDEN_BLOCK_TYPES)


def _hidden_block_type_mask(df: pd.DataFrame) -> pd.Series:
    if "block_type" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["block_type"].map(_is_hidden_block_type).astype(bool)


# =========================
# Dataclasses
# =========================

PageLabelType = Literal["arabic", "roman", "alpha_numeric", "alpha_roman", "roman_numeric", "unknown"]


@dataclass(frozen=True)
class TocRowCandidate:
    line_id: int
    is_row_candidate: bool
    page_label_token: Optional[str]
    page_label_type: PageLabelType
    has_link: bool
    table_id: Optional[Any]
    has_dot_leaders: bool

    # layout (helps for <p>-based TOCs and cross-page merge)
    left: Optional[float]
    height: Optional[float]
    font_size_px: Optional[float]
    text_align: Optional[str]
    non_stroking_color: Optional[str]
    paragraph_style_id: Optional[str]
    page_number: Optional[int]


@dataclass(frozen=True)
class LayoutFingerprint:
    """Fingerprint for grouping similar non-table rows using actual observed values."""
    left: float
    height: float
    font_size_px: float

    @classmethod
    def from_candidate(cls, candidate: TocRowCandidate) -> Optional['LayoutFingerprint']:
        """Create a fingerprint from a TocRowCandidate's layout fields."""
        if candidate.left is None or candidate.font_size_px is None:
            return None
        return cls(
            left=candidate.left,
            height=candidate.height or 0.0,
            font_size_px=candidate.font_size_px,
        )

    def matches(self, other: 'LayoutFingerprint',
                left_tolerance: float = 5.0,
                height_tolerance: float = 2.0,
                font_tolerance: float = 0.5) -> bool:
        """Check if another fingerprint matches within tolerances."""
        return (
            abs(self.left - other.left) <= left_tolerance and
            abs(self.height - other.height) <= height_tolerance and
            abs(self.font_size_px - other.font_size_px) <= font_tolerance
        )


@dataclass(frozen=True)
class DocxStyleFingerprint:
    """Fallback fingerprint for DOCX rows that do not have x-position data."""
    font_size_px: float
    text_align: Optional[str]
    non_stroking_color: Optional[str]
    paragraph_style_id: Optional[str]

    @classmethod
    def from_candidate(cls, candidate: TocRowCandidate) -> Optional['DocxStyleFingerprint']:
        """Create a fingerprint from DOCX paragraph/style fields."""
        if candidate.left is not None or candidate.font_size_px is None:
            return None
        if (
            candidate.text_align is None and
            candidate.non_stroking_color is None and
            candidate.paragraph_style_id is None
        ):
            return None
        return cls(
            font_size_px=candidate.font_size_px,
            text_align=candidate.text_align,
            non_stroking_color=candidate.non_stroking_color,
            paragraph_style_id=candidate.paragraph_style_id,
        )

    def matches(self, other: 'DocxStyleFingerprint',
                left_tolerance: float = 5.0,
                height_tolerance: float = 2.0,
                font_tolerance: float = 0.5) -> bool:
        """Check if another DOCX style fingerprint matches."""
        return (
            abs(self.font_size_px - other.font_size_px) <= font_tolerance and
            self.text_align == other.text_align and
            self.non_stroking_color == other.non_stroking_color and
            self.paragraph_style_id == other.paragraph_style_id
        )


@dataclass(frozen=True)
class TocSegment:
    """
    A small, localized cluster of candidate rows.
    Segments are later scored and merged into final TOCs.
    """
    segment_id: int
    start_line_id: int
    end_line_id: int
    n_rows: int
    n_candidates: int
    candidate_ratio: float
    max_consecutive_candidates: int  # Longest run of consecutive candidate rows
    has_toc_heading_nearby: bool
    has_page_header: bool
    n_links: int         # Number of rows in this segment with hyperlinks
    n_dot_leaders: int   # Number of rows in this segment with dot leaders (......)
    nearby_header_line_ids: List[int]
    # Primary clustering signal (mutually exclusive in practice)
    table_id: Optional[str]                   # Non-null if this is a table-based segment
    fingerprint: Optional[Any]                # Non-null if fingerprint-based segment

    @property
    def is_table_based(self) -> bool:
        return self.table_id is not None

    @property
    def is_fingerprint_based(self) -> bool:
        return self.fingerprint is not None


@dataclass(frozen=True)
class TocScore:
    """Score breakdown for a TOC segment."""
    segment: TocSegment
    total_score: float
    # Score components
    header_score: float
    page_header_score: float
    links_score: float
    consecutive_score: float
    ratio_score: float
    dot_leaders_score: float
    fingerprint_penalty: float
    currency_penalty: float
    # Flags
    passed_filters: bool
    accepted: bool


# ==========================================
# STEP 0: Remove TOC Pointers
# ==========================================

def _remove_toc_pointers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove TOC pointer rows (navigation links back to the TOC) and reindex line_id.

    A TOC pointer is a row whose text is exactly "table of contents" and has a hyperlink.
    These are navigation artifacts, not real content.
    """
    if "text" not in df.columns or "has_link" not in df.columns:
        return df

    out = df.copy()
    text_lower = out["text"].astype(str).str.strip().str.lower()
    is_pointer = (text_lower == "table of contents") & out["has_link"].map(_safe_bool01)
    out = out[~is_pointer].copy()
    out["line_id"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


# ==========================================
# STEP 1: Build TOC Header Candidate Column
# ==========================================

def _identify_toc_heading_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify TOC header candidates (e.g., "Table of Contents", "Index").

    Adds column `toc_heading_candidate`:
    - TRUE where text matches any TOC header pattern and length <= 80
    - NA elsewhere
    """
    out = df.copy()
    out["toc_heading_candidate"] = pd.NA

    if "text" not in out.columns:
        return out

    text = out["text"]

    mask = pd.Series([False] * len(text), index=text.index)
    for pattern in TOC_HEADER_PATTERNS:
        mask |= text.str.match(pattern, na=False)

    mask &= text.str.len().fillna(81) <= 80
    mask &= ~_hidden_block_type_mask(out)

    out.loc[mask, "toc_heading_candidate"] = True
    return out


# ==========================================
# STEP 2: Build TOC Candidate Column
# ==========================================

# ---- Helper Functions ---- #

def _classify_page_label_token(token: str, cfg: PageLabelPatternConfig) -> PageLabelType:
    t = (token or "").strip()
    if not t:
        return "unknown"
    if len(t) > cfg.max_length:
        return "unknown"

    for pat in cfg.patterns:
        if pat.compiled.match(t):
            name = (pat.name or "unknown").strip()
            if name in {"arabic", "roman", "alpha_numeric", "alpha_roman", "roman_numeric"}:
                return name  # type: ignore[return-value]
            return "unknown"
    return "unknown"


def _extract_last_token(text: str) -> str:
    # strip common trailing punctuation/closers
    s = (text or "").strip().rstrip(".,;:)]}")
    if not s:
        return ""
    parts = s.split()
    return parts[-1] if parts else ""


def _token_is_at_end(text: str, token: str) -> bool:
    s = (text or "").rstrip().rstrip(" \t\r\n.,;:)]}")
    return bool(token) and s.endswith(token)


def _has_adjacent_page_label_token_before_last(
    text: str,
    page_label_config: PageLabelPatternConfig,
) -> bool:
    """
    Return True if the token immediately before the trailing page-label token
    also looks like a page-label token (per YAML patterns).

    Example reject:
      "... 558 518"  -> last token 518 is a page label, and previous token 558 is also a page label
    """
    s = (text or "").strip().rstrip(".,;:)]}")
    if not s:
        return False

    parts = s.split()
    if len(parts) < 2:
        return False

    last = parts[-1]
    prev = parts[-2].strip(".,;:)]}")

    if _classify_page_label_token(last, page_label_config) == "unknown":
        return False

    return _classify_page_label_token(prev, page_label_config) != "unknown"


# ---- Main TOC Row Candidate Builder ---- #

def _add_toc_row_candidates(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    *,
    min_chars: int = 4,   # Excluding page label token
    max_chars: int = 250,
    include_debug_cols: bool = True,
) -> Tuple[pd.DataFrame, List[TocRowCandidate]]:
    """
    Identifies TOC row candidates and builds TocRowCandidate objects.

    Adds:
      - toc_row_candidate: TRUE where row rules pass, else NA
      - (optional debug) toc_page_label_token, toc_page_label_type, toc_has_dot_leaders

    Row rules:
      - non-empty text
      - len(text) <= max_chars
      - trailing token matches YAML patterns (page_label_config)
      - token must be at end of line
      - reject if multiple consecutive token-like items appear at right end
      - require at least `min_chars` chars before the token (basic guard)
      - reject rows with currency symbols

    Returns: (out_df, candidates)
    """
    out = df.copy()

    # initialize as NA (not False) for easier CSV inspection
    out["toc_row_candidate"] = pd.NA

    if include_debug_cols:
        out["toc_page_label_token"] = pd.NA
        out["toc_page_label_type"] = pd.NA
        out["toc_has_dot_leaders"] = pd.NA

    if "text" not in out.columns:
        return out, []

    texts = out["text"].astype(str)
    hidden_mask = _hidden_block_type_mask(out)
    candidates: List[TocRowCandidate] = []

    # Row candidates (iterate once; keeps logic clear and debuggable)
    for idx, raw_text in texts.items():
        if bool(hidden_mask.at[idx]):
            continue

        text = (raw_text or "").strip()
        if not text or len(text) > max_chars:
            continue

        # extract trailing token and classify via YAML
        token = _extract_last_token(text)
        if not token:
            continue

        token_type = _classify_page_label_token(token, page_label_config)
        if token_type == "unknown":
            continue

        if not _token_is_at_end(text, token):
            continue

        # Require minimum number of characters before the page token
        title_part = re.sub(r"\s+", "", _DOT_LEADERS_RE.sub("", text[: -len(token)].strip()))
        if len(title_part) < min_chars:
            continue

        # Reject "multiple right tokens" (table-like rows)
        if _has_adjacent_page_label_token_before_last(text, page_label_config):
            continue

        # Reject rows with currency symbols
        if any(sym in text for sym in ("$", "€", "£")):
            continue

        has_dot_leaders = bool(_DOT_LEADERS_RE.search(text))

        if include_debug_cols:
            out.loc[idx, "toc_page_label_token"] = token
            out.loc[idx, "toc_page_label_type"] = token_type
            out.loc[idx, "toc_has_dot_leaders"] = has_dot_leaders

        out.loc[idx, "toc_row_candidate"] = True

        candidates.append(
            TocRowCandidate(
                line_id=int(out.at[idx, "line_id"]) if "line_id" in out.columns else int(idx),
                is_row_candidate=True,
                page_label_token=token,
                page_label_type=token_type,
                has_link=_safe_bool01(out.at[idx, "has_link"]) if "has_link" in out.columns else False,
                table_id=out.at[idx, "table_id"] if "table_id" in out.columns else None,
                has_dot_leaders=has_dot_leaders,
                left=_safe_float(out.at[idx, "x_left"]) if "x_left" in out.columns else None,
                height=_safe_float(out.at[idx, "height"]) if "height" in out.columns else None,
                font_size_px=_safe_float(out.at[idx, "font_size"]) if "font_size" in out.columns else None,
                text_align=_safe_str_or_none(out.at[idx, "text_align"]) if "text_align" in out.columns else None,
                non_stroking_color=_safe_str_or_none(out.at[idx, "non_stroking_color"]) if "non_stroking_color" in out.columns else None,
                paragraph_style_id=_safe_str_or_none(out.at[idx, "paragraph_style_id"]) if "paragraph_style_id" in out.columns else None,
                page_number=_safe_int(out.at[idx, "page_number"]) if "page_number" in out.columns else None,
            )
        )

    return out, candidates


# ==========================================
# STEP 3: Build TOC Segments
# ==========================================

# ---- Helper Functions ---- #

_PAGE_HEADER_RE = re.compile(r"^\s*pages?\s*$", re.IGNORECASE)


def _detect_page_header_tables(df: pd.DataFrame) -> Set[Any]:
    """
    Return a set of table_id values where the FIRST line
    contains exactly 'page' or 'pages' (case-insensitive).

    Strong signal that the table is a page-number listing (TOC-like).
    """
    if "table_id" not in df.columns or "text" not in df.columns:
        return set()

    page_header_table_ids: Set[Any] = set()

    for table_id, g in df.groupby("table_id"):
        if pd.isna(table_id):
            continue

        first_row = g.sort_values("line_id").iloc[0] if "line_id" in g.columns else g.iloc[0]

        if _PAGE_HEADER_RE.match(str(first_row["text"]).strip()):
            page_header_table_ids.add(table_id)

    return page_header_table_ids


def _calculate_max_consecutive(sorted_line_ids: List[int], candidate_set: Set[int]) -> int:
    """Calculate the maximum consecutive run of candidates in sorted line IDs."""
    if not sorted_line_ids or not candidate_set:
        return 0

    max_run = 0
    current_run = 0

    for line_id in sorted_line_ids:
        if line_id in candidate_set:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def _check_toc_heading_nearby(
    df_sorted: pd.DataFrame,
    start_line_id: int,
    lookback: int,
) -> Tuple[bool, List[int]]:
    """
    Check if a TOC header appears within lookback rows before start_line_id.

    Returns:
        Tuple of (has_header: bool, header_line_ids: List[int])
    """
    if "toc_heading_candidate" not in df_sorted.columns:
        return False, []

    before_rows = df_sorted[df_sorted["line_id"] < start_line_id].tail(lookback)
    if before_rows.empty:
        return False, []

    header_rows = before_rows[before_rows["toc_heading_candidate"].notna()]
    header_line_ids = [int(x) for x in header_rows["line_id"].tolist()]
    return len(header_line_ids) > 0, header_line_ids


def _get_row_fingerprint(row: pd.Series) -> Optional[Any]:
    """Extract a positional or DOCX style fingerprint from a DataFrame row.

    Used during segment expansion to check layout continuity of non-candidate rows.
    """
    if _is_hidden_block_type(row.get("block_type")):
        return None

    left = _safe_float(row.get("x_left"))
    height = _safe_float(row.get("height"))
    font_size_px = _safe_float(row.get("font_size"))

    if font_size_px is None:
        return None

    if left is not None:
        return LayoutFingerprint(left=left, height=height or 0.0, font_size_px=font_size_px)

    text_align = _safe_str_or_none(row.get("text_align"))
    non_stroking_color = _safe_str_or_none(row.get("non_stroking_color"))
    paragraph_style_id = _safe_str_or_none(row.get("paragraph_style_id"))
    if text_align is None and non_stroking_color is None and paragraph_style_id is None:
        return None

    return DocxStyleFingerprint(
        font_size_px=font_size_px,
        text_align=text_align,
        non_stroking_color=non_stroking_color,
        paragraph_style_id=paragraph_style_id,
    )


def _expand_segment_by_fingerprint(
    df_sorted: pd.DataFrame,
    seed_line_id: int,
    target_fingerprint: Any,
    line_id_to_idx: Dict[int, int],
    left_tolerance: float,
    height_tolerance: float,
    font_tolerance: float,
) -> List[int]:
    """
    Expand from seed_line_id to find all contiguous rows with matching fingerprint.
    """
    if seed_line_id not in line_id_to_idx:
        return []

    seed_idx = line_id_to_idx[seed_line_id]
    segment_indices = {seed_idx}

    # Expand upward
    current_idx = seed_idx - 1
    while current_idx >= 0:
        fp = _get_row_fingerprint(df_sorted.iloc[current_idx])
        if fp is None or not target_fingerprint.matches(fp, left_tolerance, height_tolerance, font_tolerance):
            break
        segment_indices.add(current_idx)
        current_idx -= 1

    # Expand downward
    current_idx = seed_idx + 1
    while current_idx < len(df_sorted):
        fp = _get_row_fingerprint(df_sorted.iloc[current_idx])
        if fp is None or not target_fingerprint.matches(fp, left_tolerance, height_tolerance, font_tolerance):
            break
        segment_indices.add(current_idx)
        current_idx += 1

    return [int(df_sorted.iloc[idx]["line_id"]) for idx in sorted(segment_indices)]


# ---- Main TOC Segment Builder ---- #

def _build_toc_segments(
    df: pd.DataFrame,
    candidates: List[TocRowCandidate],
    *,
    left_tolerance: float = 5.0,
    height_tolerance: float = 2.0,
    font_tolerance: float = 0.5,
    header_lookback: int = 3,
) -> List[TocSegment]:
    """
    Build TocSegment objects from TOC row candidates.

    Strategy:
    Iterate through DataFrame in line_id order. When encountering a toc_row_candidate:
    - If it has a table_id: collect all rows in that table as one segment
    - If no table_id: expand up/down by matching fingerprint (using candidate layout data)
    Mark processed line_ids and continue to next unprocessed candidate.
    """
    if df.empty or "line_id" not in df.columns or "toc_row_candidate" not in df.columns:
        return []

    df_sorted = df.sort_values("line_id").reset_index(drop=True)
    line_id_to_idx: Dict[int, int] = {int(row["line_id"]): idx for idx, row in df_sorted.iterrows()}

    # O(1) candidate lookup by line_id — avoids re-scanning the DataFrame per row
    candidates_by_line_id: Dict[int, TocRowCandidate] = {c.line_id: c for c in candidates}

    page_header_tables = _detect_page_header_tables(df)

    segments: List[TocSegment] = []
    segment_id_counter = 0
    processed_line_ids: Set[int] = set()
    processed_table_ids: Set[Any] = set()

    for _, row in df_sorted.iterrows():
        line_id = int(row["line_id"])

        if line_id in processed_line_ids:
            continue

        if not pd.notna(row.get("toc_row_candidate")):
            continue

        table_id = row.get("table_id")
        has_table = (
            table_id is not None and
            not pd.isna(table_id) and
            str(table_id).strip() != ""
        )

        if has_table and table_id not in processed_table_ids:
            # === TABLE-BASED SEGMENT ===
            table_rows = df_sorted[df_sorted["table_id"] == table_id]

            if not table_rows.empty:
                all_line_ids = sorted(int(x) for x in table_rows["line_id"].tolist())
                candidate_line_ids = {lid for lid in all_line_ids if lid in candidates_by_line_id}

                start_line_id = all_line_ids[0]
                n_rows = len(all_line_ids)
                n_candidates = len(candidate_line_ids)

                max_consecutive = _calculate_max_consecutive(all_line_ids, candidate_line_ids)
                has_toc_heading, header_line_ids = _check_toc_heading_nearby(
                    df_sorted, start_line_id, header_lookback
                )

                n_links = int(table_rows["has_link"].map(_safe_bool01).sum()) if "has_link" in table_rows.columns else 0
                n_dot_leaders = int(table_rows["toc_has_dot_leaders"].map(_safe_bool01).sum()) if "toc_has_dot_leaders" in table_rows.columns else 0

                segments.append(TocSegment(
                    segment_id=segment_id_counter,
                    start_line_id=start_line_id,
                    end_line_id=all_line_ids[-1],
                    n_rows=n_rows,
                    n_candidates=n_candidates,
                    candidate_ratio=n_candidates / n_rows if n_rows > 0 else 0.0,
                    max_consecutive_candidates=max_consecutive,
                    has_toc_heading_nearby=has_toc_heading,
                    has_page_header=table_id in page_header_tables,
                    n_links=n_links,
                    n_dot_leaders=n_dot_leaders,
                    nearby_header_line_ids=header_line_ids,
                    table_id=str(table_id),
                    fingerprint=None,
                ))
                segment_id_counter += 1
                processed_line_ids.update(all_line_ids)
                processed_table_ids.add(table_id)

        elif not has_table:
            # === FINGERPRINT-BASED SEGMENT ===
            # Build fingerprint from the candidate object — no need to re-read the DataFrame
            candidate_obj = candidates_by_line_id.get(line_id)
            if candidate_obj is None:
                continue

            fingerprint = (
                LayoutFingerprint.from_candidate(candidate_obj) or
                DocxStyleFingerprint.from_candidate(candidate_obj)
            )
            if fingerprint is None:
                continue

            segment_line_ids = _expand_segment_by_fingerprint(
                df_sorted, line_id, fingerprint, line_id_to_idx,
                left_tolerance, height_tolerance, font_tolerance,
            )

            if not segment_line_ids:
                continue

            # Count candidates using dict lookup (O(1) per line)
            candidate_line_ids = {lid for lid in segment_line_ids if lid in candidates_by_line_id}

            start_line_id = segment_line_ids[0]
            n_rows = len(segment_line_ids)
            n_candidates = len(candidate_line_ids)

            max_consecutive = _calculate_max_consecutive(segment_line_ids, candidate_line_ids)
            has_toc_heading, header_line_ids = _check_toc_heading_nearby(
                df_sorted, start_line_id, header_lookback
            )

            # Count links and dot leaders via O(1) index lookup
            n_links = 0
            n_dot_leaders = 0
            has_link_present = "has_link" in df_sorted.columns
            has_dot_present = "toc_has_dot_leaders" in df_sorted.columns
            for lid in segment_line_ids:
                if lid in line_id_to_idx:
                    r = df_sorted.iloc[line_id_to_idx[lid]]
                    if has_link_present and _safe_bool01(r.get("has_link")):
                        n_links += 1
                    if has_dot_present and _safe_bool01(r.get("toc_has_dot_leaders")):
                        n_dot_leaders += 1

            segments.append(TocSegment(
                segment_id=segment_id_counter,
                start_line_id=start_line_id,
                end_line_id=segment_line_ids[-1],
                n_rows=n_rows,
                n_candidates=n_candidates,
                candidate_ratio=n_candidates / n_rows if n_rows > 0 else 0.0,
                max_consecutive_candidates=max_consecutive,
                has_toc_heading_nearby=has_toc_heading,
                has_page_header=False,
                n_links=n_links,
                n_dot_leaders=n_dot_leaders,
                nearby_header_line_ids=header_line_ids,
                table_id=None,
                fingerprint=fingerprint,
            ))
            segment_id_counter += 1
            processed_line_ids.update(segment_line_ids)

    return segments


# ==========================================
# STEP 4: Score and Filter Segments
# ==========================================

def _score_and_filter_toc_segments(
    segments: List[TocSegment],
    df: pd.DataFrame,
    *,
    min_rows: int,
    min_consecutive: int,
    min_candidate_ratio: float,
    min_score_threshold: float,
) -> List[TocScore]:
    """
    Score and filter TOC segments based on multiple signals.

    Returns list of TocScore objects with detailed scoring breakdown.
    Only segments with total_score > min_score_threshold are marked as accepted.
    """
    # Scoring weights — tuned empirically, adjust here if needed
    HAS_TOC_HEADER_WEIGHT       = 1.0
    HAS_PAGE_HEADER_WEIGHT      = 1.0
    LINKS_WEIGHT_PER_LINK       = 0.5
    LINKS_SCORE_CAP             = 2.0
    CONSECUTIVE_WEIGHT          = 0.2
    CONSECUTIVE_CAP             = 1.5
    LONG_CONSECUTIVE_MIN        = 10
    LONG_CONSECUTIVE_BONUS      = 0.2
    RATIO_WEIGHT                = 1.0
    DOT_LEADERS_WEIGHT          = 0.2
    DOT_LEADERS_CAP             = 1.0
    FINGERPRINT_PENALTY         = 0.0
    CURRENCY_PENALTY            = 0.5

    # Pre-build set of line_ids containing currency symbols — avoids re-scanning df per segment
    currency_line_ids: Set[int] = set()
    if "text" in df.columns and "line_id" in df.columns:
        currency_mask = df["text"].astype(str).str.contains(r'[$€£¥]', regex=True, na=False)
        currency_line_ids = {int(x) for x in df.loc[currency_mask, "line_id"].tolist()}

    scores: List[TocScore] = []

    for segment in segments:
        # === DISQUALIFICATION FILTERS ===
        if (segment.n_rows < min_rows or
                segment.max_consecutive_candidates < min_consecutive or
                segment.candidate_ratio < min_candidate_ratio):
            scores.append(TocScore(
                segment=segment,
                total_score=0.0,
                header_score=0.0,
                page_header_score=0.0,
                links_score=0.0,
                consecutive_score=0.0,
                ratio_score=0.0,
                dot_leaders_score=0.0,
                fingerprint_penalty=0.0,
                currency_penalty=0.0,
                passed_filters=False,
                accepted=False,
            ))
            continue

        # === SCORING ===
        header_score      = HAS_TOC_HEADER_WEIGHT if segment.has_toc_heading_nearby else 0.0
        page_header_score = HAS_PAGE_HEADER_WEIGHT if segment.has_page_header else 0.0
        links_score       = min(segment.n_links * LINKS_WEIGHT_PER_LINK, LINKS_SCORE_CAP)
        consecutive_score = min(segment.max_consecutive_candidates * CONSECUTIVE_WEIGHT, CONSECUTIVE_CAP)
        if segment.max_consecutive_candidates >= LONG_CONSECUTIVE_MIN:
            consecutive_score += LONG_CONSECUTIVE_BONUS
        ratio_score       = segment.candidate_ratio * RATIO_WEIGHT
        dot_leaders_score = min(segment.n_dot_leaders * DOT_LEADERS_WEIGHT, DOT_LEADERS_CAP)

        fp_penalty = FINGERPRINT_PENALTY if segment.is_fingerprint_based else 0.0

        # line_ids are consecutive after reindexing, so range check is accurate
        currency_found = any(
            lid in currency_line_ids
            for lid in range(segment.start_line_id, segment.end_line_id + 1)
        )
        curr_penalty = CURRENCY_PENALTY if currency_found else 0.0

        total_score = (
            header_score
            + page_header_score
            + links_score
            + consecutive_score
            + ratio_score
            + dot_leaders_score
            - fp_penalty
            - curr_penalty
        )

        scores.append(TocScore(
            segment=segment,
            total_score=total_score,
            header_score=header_score,
            page_header_score=page_header_score,
            links_score=links_score,
            consecutive_score=consecutive_score,
            ratio_score=ratio_score,
            dot_leaders_score=dot_leaders_score,
            fingerprint_penalty=fp_penalty,
            currency_penalty=curr_penalty,
            passed_filters=True,
            accepted=total_score > min_score_threshold,
        ))

    return scores


# ==========================================
# STEP 5: Close Gaps Between TOC Segments
# ==========================================

def _close_toc_gaps(df: pd.DataFrame, *, max_gap: int = 3) -> pd.DataFrame:
    """
    Close gaps between TOC-related rows for better continuity.

    Two gap-closing operations:
    1. Fill gaps between toc_heading and toc (mark as toc_heading)
    2. Fill small gaps between toc segments (mark as toc, max `max_gap` rows)

    Both operations overwrite existing "table" rows in the gap, in addition to NA rows.
    This handles TOCs where the table parser produced many small per-section tables that
    individually fail the min_rows filter but sit between accepted toc segments.
    """
    if df.empty or "block_type" not in df.columns or "line_id" not in df.columns:
        return df

    out = df.copy().sort_values("line_id").reset_index(drop=True)

    # === OPERATION 1: Close gaps between toc_heading and toc ===
    toc_heading_indices = out[out["block_type"] == "toc_heading"].index.tolist()
    toc_indices = out[out["block_type"] == "toc"].index.tolist()

    for header_idx in toc_heading_indices:
        header_line_id = out.at[header_idx, "line_id"]
        next_toc = [idx for idx in toc_indices if out.at[idx, "line_id"] > header_line_id]

        if next_toc:
            next_toc_line_id = out.at[min(next_toc), "line_id"]
            gap_mask = (
                (out["line_id"] > header_line_id) &
                (out["line_id"] < next_toc_line_id) &
                (out["block_type"].isna() | (out["block_type"] == "table"))
            )
            out.loc[gap_mask, "block_type"] = "toc_heading"

    # === OPERATION 2: Close small gaps between toc segments ===
    toc_indices = out[out["block_type"] == "toc"].index.tolist()
    if not toc_indices:
        return out

    # Group toc rows into truly contiguous segments (no gap)
    segments = []
    current_segment = [toc_indices[0]]

    for i in range(1, len(toc_indices)):
        prev_idx, curr_idx = toc_indices[i - 1], toc_indices[i]
        if out.at[curr_idx, "line_id"] - out.at[prev_idx, "line_id"] == 1:
            current_segment.append(curr_idx)
        else:
            segments.append(current_segment)
            current_segment = [curr_idx]

    segments.append(current_segment)

    # Merge adjacent segments if gap is small enough
    for i in range(len(segments) - 1):
        seg1_end_line_id   = out.at[segments[i][-1], "line_id"]
        seg2_start_line_id = out.at[segments[i + 1][0], "line_id"]
        gap_size = seg2_start_line_id - seg1_end_line_id - 1

        if 0 < gap_size <= max_gap:
            gap_mask = (
                (out["line_id"] > seg1_end_line_id) &
                (out["line_id"] < seg2_start_line_id) &
                (out["block_type"].isna() | (out["block_type"] == "table"))
            )
            out.loc[gap_mask, "block_type"] = "toc"

    return out


# ==========================================
# PUBLIC API
# ==========================================

def detect_and_annotate_tocs(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    *,
    min_rows: int = 4,
    min_consecutive: int = 3,
    min_candidate_ratio: float = 0.5,
    min_score_threshold: float = 2.5,
    max_gap: int = 3,
    include_debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Complete TOC detection pipeline.

    Pipeline:
    1. Remove TOC pointer links and reindex line_id
    2. Identify header candidates
    3. Identify row candidates and build TocRowCandidate objects
    4. Build TocSegment objects
    5. Score and filter segments
    6. Annotate winning rows with block_type = 'toc' or 'toc_heading'
    7. Close gaps between toc_heading/toc and between toc segments

    Args:
        df: Input DataFrame (expects columns: text, has_link, line_id)
        page_label_config: Compiled page label patterns from YAML

        Disqualification filters:
            min_rows: Minimum rows in a segment to qualify (default: 4)
            min_consecutive: Minimum consecutive candidates required (default: 3)
            min_candidate_ratio: Minimum fraction of rows that are candidates (default: 0.5)
            min_score_threshold: Minimum score to accept a segment (default: 2.5)

        Gap closing:
            max_gap: Maximum number of non-toc rows to bridge between toc segments (default: 3)

        Debug:
            include_debug_cols: Keep toc_* intermediate columns in output (default: False)

    Returns:
        DataFrame with block_type column added:
        - 'toc' for TOC entry rows
        - 'toc_heading' for header rows (e.g. "Table of Contents" title)
    """
    df = _remove_toc_pointers(df)
    df = _identify_toc_heading_candidates(df)
    df, candidates = _add_toc_row_candidates(df, page_label_config, include_debug_cols=include_debug_cols)
    segments = _build_toc_segments(df, candidates)
    scores = _score_and_filter_toc_segments(
        segments, df,
        min_rows=min_rows,
        min_consecutive=min_consecutive,
        min_candidate_ratio=min_candidate_ratio,
        min_score_threshold=min_score_threshold,
    )

    if "block_type" not in df.columns:
        df["block_type"] = pd.NA

    # Write per-row segment debug columns before annotating block_type so that
    # the mask covers the unmodified rows (no hidden-block interference).
    if include_debug_cols:
        df["toc_segment_id"]       = pd.NA
        df["toc_seg_type"]         = pd.NA  # "table" or "fingerprint"
        df["toc_seg_score"]        = pd.NA
        df["toc_seg_passed"]       = pd.NA
        df["toc_seg_accepted"]     = pd.NA
        df["toc_seg_max_consec"]   = pd.NA

        for score_obj in scores:
            seg = score_obj.segment
            mask = (df["line_id"] >= seg.start_line_id) & (df["line_id"] <= seg.end_line_id)
            df.loc[mask, "toc_segment_id"]     = seg.segment_id
            df.loc[mask, "toc_seg_type"]       = "table" if seg.is_table_based else "fingerprint"
            df.loc[mask, "toc_seg_score"]      = round(score_obj.total_score, 2)
            df.loc[mask, "toc_seg_passed"]     = score_obj.passed_filters
            df.loc[mask, "toc_seg_accepted"]   = score_obj.accepted
            df.loc[mask, "toc_seg_max_consec"] = seg.max_consecutive_candidates

    for score_obj in scores:
        if score_obj.accepted:
            seg = score_obj.segment

            for header_line_id in seg.nearby_header_line_ids:
                header_mask = (
                    (df["line_id"] == header_line_id) &
                    ~_hidden_block_type_mask(df)
                )
                df.loc[header_mask, "block_type"] = "toc_heading"

            mask = (df["line_id"] >= seg.start_line_id) & (df["line_id"] <= seg.end_line_id)
            mask &= ~_hidden_block_type_mask(df)
            df.loc[mask, "block_type"] = "toc"

    df = _close_toc_gaps(df, max_gap=max_gap)

    if not include_debug_cols:
        debug_cols = [
            "toc_heading_candidate",
            "toc_row_candidate",
            "toc_page_label_token",
            "toc_page_label_type",
            "toc_has_dot_leaders",
        ]
        df = df.drop(columns=[col for col in debug_cols if col in df.columns])
    return df

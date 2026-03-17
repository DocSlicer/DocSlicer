# d04_toc_detector.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

import pandas as pd

from .._utils.yaml_compilers.page_label_patterns import PageLabelPatternConfig

# =========================
# Config
# =========================

# Header anchors - clean, separate patterns for each case
TOC_HEADER_PATTERNS = [
    # 1) Classic "Table of Contents"
    re.compile(r'^\s*table\s+of\s+contents?\b', re.IGNORECASE),

    # 2) Simple "Contents"
    re.compile(r'^\s*contents?\b', re.IGNORECASE),

    # 3) PRIMARY rule — starts WITH or ends WITH INDEX
    re.compile(r'^(?:\s*index\b.*|.*\bindex\s*)$', re.IGNORECASE),
]



# dot leaders = "." repeated, allowing whitespace between dots
_DOT_LEADERS_RE = re.compile(r"\.(?:\s*\.){2,}")

# Token-like items at right tail (used only for "multiple right tokens" filter)
_RIGHT_TOKENLIKE_RE = re.compile(
    r"^(?:\d{1,4}|[A-Za-z]{1}-?\d{1,4}|[ivxlcdm]{1,10})$",
    re.IGNORECASE,
)

# =========================
# Dataclasses
# =========================

PageLabelType = Literal["arabic", "roman", "alpha_numeric", "alpha_roman", "roman_numeric", "unknown"]

@dataclass(frozen=True)
class TocRowCandidate:
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
    page_number: Optional[int]


@dataclass(frozen=True)
class LayoutFingerprint:
    """Fingerprint for grouping similar non-table rows using actual observed values"""
    left: float  # Actual left coordinate
    height: float  # Actual height
    font_size_px: float  # Actual font size
    
    @classmethod
    def from_candidate(cls, candidate: TocRowCandidate) -> Optional['LayoutFingerprint']:
        """Create fingerprint from actual values"""
        if candidate.left is None or candidate.font_size_px is None:
            return None
        return cls(
            left=candidate.left,
            height=candidate.height or 0.0,
            font_size_px=candidate.font_size_px
        )
    
    def matches(self, other: 'LayoutFingerprint',
                left_tolerance: float = 5.0,
                height_tolerance: float = 2.0,
                font_tolerance: float = 0.5) -> bool:
        """Check if another fingerprint matches within tolerances"""
        return (
            abs(self.left - other.left) <= left_tolerance and
            abs(self.height - other.height) <= height_tolerance and
            abs(self.font_size_px - other.font_size_px) <= font_tolerance
        )


@dataclass(frozen=True)
class TocSegment:
    """
    A small, localized cluster of candidate rows.
    Segments are later merged into final TOCs.
    """
    segment_id: int
    start_line_id: int
    end_line_id: int
    n_rows: int
    n_candidates: int
    candidate_ratio: float
    max_consecutive_candidates: int  # Longest run of consecutive candidate rows
    has_toc_header_nearby: bool
    has_page_header: bool
    n_links: int # Number of rows in this object with hyperlinks
    n_dot_leaders: int # Number of rows with dot leaders (......)
    # Header line_ids found nearby (outside segment boundaries)
    nearby_header_line_ids: List[int]
    # Primary clustering signal (mutually exclusive in practice)
    table_id: Optional[str] # Non-null if this is a table-based segment
    fingerprint: Optional[LayoutFingerprint] # Non-null if layout-based segment

    @property
    def is_table_based(self) -> bool:
        return self.table_id is not None
    
    @property
    def is_fingerprint_based(self) -> bool:
        return self.fingerprint is not None

# ==========================================
# STEP 0: Remove TOC Pointers
# ==========================================

def _is_toc_pointer(text: str, has_link: int) -> bool:
    """Check if row is a TOC navigation link (e.g., 'table of contents' with link)"""
    if not isinstance(text, str):
        return False
    return text.strip().lower() == "table of contents" and int(has_link or 0) == 1


def remove_toc_pointers(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    has_link_col: str = "has_link",
    line_id_col: str = "line_id",
) -> pd.DataFrame:
    """
    Remove TOC pointer rows (navigation links) and reindex line_id.
    
    Args:
        df: DataFrame with text and has_link columns
        text_col: Column name for text
        has_link_col: Column name for link flag
        line_id_col: Column name for line IDs
        
    Returns:
        DataFrame with TOC pointers removed and line_id reindexed
    """
    out = df.copy()
    
    if text_col not in out.columns or has_link_col not in out.columns:
        return out
    
    # Remove TOC pointer rows
    mask = out.apply(
        lambda row: not _is_toc_pointer(row.get(text_col), row.get(has_link_col)),
        axis=1
    )
    out = out[mask].copy()
    
    # Reindex line_id
    if line_id_col in out.columns:
        out[line_id_col] = range(1, len(out) + 1)
    
    return out.reset_index(drop=True)


# ==========================================
# STEP 1: Build TOC Header Candidate Column
# ==========================================

def identify_toc_header_candidates(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    header_col: str = "toc_header_candidate",
    max_len: int = 80,
) -> pd.DataFrame:
    """
    Identify TOC header candidates (e.g., "Table of Contents", "Index").
    
    Adds column `toc_header_candidate`:
    - TRUE where text matches any TOC header pattern and length <= max_len
    - NA elsewhere
    
    Args:
        df: DataFrame with text column
        text_col: Column name for text
        header_col: Column name for output header candidate flag
        max_len: Maximum text length for header candidates
        
    Returns:
        DataFrame with toc_header_candidate column added
    """
    out = df.copy()
    out[header_col] = pd.NA
    
    if text_col not in out.columns:
        return out
    
    text = out[text_col].astype(str)
    
    # Check if text matches ANY of the TOC header patterns
    mask = pd.Series([False] * len(text), index=text.index)
    for pattern in TOC_HEADER_PATTERNS:
        mask |= text.str.match(pattern, na=False)
    
    # Apply length constraint
    mask &= (text.str.len() <= max_len)
    
    out.loc[mask, header_col] = True
    return out


# ==========================================
# STEP 2: Build TOC Candidate Column
# ==========================================

# ---- Helper Functions ---- #

def _classify_page_label_token(
    token: str,
    cfg: PageLabelPatternConfig,
) -> PageLabelType:
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
    s = (text or "").rstrip()
    s = s.rstrip(" \t\r\n.,;:)]}")
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

    # last should already be a valid page label token in your pipeline,
    # but we keep this check for safety
    if _classify_page_label_token(last, page_label_config) == "unknown":
        return False

    return _classify_page_label_token(prev, page_label_config) != "unknown"


def _safe_bool01(x: Any) -> bool:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        # handle 0/1 ints stored as str, etc.
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
    s = str(x).strip()
    return s or None


# ---- Main TOC Row Candidate Builder ---- #

def add_toc_row_candidates(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    *,
    # columns
    text_col: str = "text",
    has_link_col: str = "has_link",
    table_id_col: str = "table_id",
    left_col: str = "x_left",
    height_col: str = "height",
    font_col: str = "font_size",
    align_col: str = "text_align",
    page_number_col: str = "page_number",
    # row rule
    min_chars: int = 4, # Excluding page label token
    max_chars: int = 250,
    # output
    include_debug_token_cols: bool = True,
    build_objects: bool = True,
) -> Tuple[pd.DataFrame, List[TocRowCandidate]]:
    """
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

    Returns: (out_df, objects)
      - objects list is empty if build_objects=False
    """
    out = df.copy()

    # initialize columns as NA (not False) for easier CSV inspection
    out["toc_row_candidate"] = pd.NA

    if include_debug_token_cols:
        out["toc_page_label_token"] = pd.NA
        out["toc_page_label_type"] = pd.NA
        out["toc_has_dot_leaders"] = pd.NA

    if text_col not in out.columns:
        return out, []

    texts = out[text_col].astype(str)

    candidates: List[TocRowCandidate] = []

    # Row candidates (iterate once; keeps logic clear and debuggable)
    for idx, raw_text in texts.items():
        text = (raw_text or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
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
        title_part = text[: -len(token)].strip()

        # Remove dot leaders and whitespace from title
        title_part = _DOT_LEADERS_RE.sub("", title_part)
        title_part = re.sub(r"\s+", "", title_part)

        if len(title_part) < min_chars:
            continue

        # reject "multiple right tokens" (table-like rows)
        if _has_adjacent_page_label_token_before_last(text, page_label_config):
            continue

        # Reject rows with currency symbols
        if any(sym in text for sym in ("$", "€", "£")):
            continue

        has_dot_leaders = bool(_DOT_LEADERS_RE.search(text))

        # Debug cols
        if include_debug_token_cols:
            out.loc[idx, "toc_page_label_token"] = token
            out.loc[idx, "toc_page_label_type"] = token_type
            out.loc[idx, "toc_has_dot_leaders"] = has_dot_leaders

        # Final accept
        out.loc[idx, "toc_row_candidate"] = True

        if build_objects:
            has_link = _safe_bool01(out.at[idx, has_link_col]) if has_link_col in out.columns else False
            table_id = out.at[idx, table_id_col] if table_id_col in out.columns else None
            left = _safe_float(out.at[idx, left_col]) if left_col in out.columns else None
            height = _safe_float(out.at[idx, height_col]) if height_col in out.columns else None
            font_size_px = _safe_float(out.at[idx, font_col]) if font_col in out.columns else None
            text_align = _safe_str_or_none(out.at[idx, align_col]) if align_col in out.columns else None
            page_number = _safe_int(out.at[idx, page_number_col]) if page_number_col in out.columns else None

            candidates.append(
                TocRowCandidate(
                    is_row_candidate=True,
                    page_label_token=token,
                    page_label_type=token_type,
                    has_link=has_link,
                    table_id=table_id,
                    has_dot_leaders=has_dot_leaders,
                    left=left,
                    height=height,
                    font_size_px=font_size_px,
                    text_align=text_align,
                    page_number=page_number,
                )
            )

    return out, candidates


# ==========================================
# STEP 3: Build TOC Segments
# ==========================================

# ---- Helper Functions ---- #

_PAGE_HEADER_RE = re.compile(r"^\s*pages?\s*$", re.IGNORECASE)


def detect_page_header_tables(
    df: pd.DataFrame,
    *,
    table_id_col: str = "table_id",
    text_col: str = "text",
    line_id_col: str = "line_id",
) -> Set[Any]:
    """
    Return a set of table_id values where the FIRST line
    contains exactly 'page' or 'pages' (case-insensitive).

    Strong signal that the table is a page-number listing (TOC-like).
    """
    if table_id_col not in df.columns or text_col not in df.columns:
        return set()

    page_header_table_ids: Set[Any] = set()

    for table_id, g in df.groupby(table_id_col):
        if pd.isna(table_id):
            continue

        # Determine first line of the table
        if line_id_col in g.columns:
            first_row = g.sort_values(line_id_col).iloc[0]
        else:
            first_row = g.iloc[0]

        text = str(first_row[text_col]).strip()
        if _PAGE_HEADER_RE.match(text):
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


def _check_toc_header_nearby(
    df_sorted: pd.DataFrame,
    start_line_id: int,
    line_id_col: str,
    toc_header_col: str,
    lookback: int,
) -> Tuple[bool, List[int]]:
    """
    Check if a TOC header appears within lookback rows before start_line_id.
    
    Returns:
        Tuple of (has_header: bool, header_line_ids: List[int])
    """
    if toc_header_col not in df_sorted.columns:
        return False, []
    
    # Find rows before start_line_id
    mask = df_sorted[line_id_col] < start_line_id
    before_rows = df_sorted[mask].tail(lookback)
    
    if before_rows.empty:
        return False, []
    
    # Find all header candidate rows
    header_line_ids = []
    for _, row in before_rows.iterrows():
        if pd.notna(row.get(toc_header_col)) and row.get(toc_header_col) is True:
            header_line_ids.append(int(row[line_id_col]))
    
    has_header = len(header_line_ids) > 0
    return has_header, header_line_ids


def _get_row_fingerprint(
    row: pd.Series,
    left_tolerance: float = None,  # Not used, kept for API compatibility
    height_tolerance: float = None,  # Not used, kept for API compatibility
    font_tolerance: float = None,  # Not used, kept for API compatibility
) -> Optional[LayoutFingerprint]:
    """Extract fingerprint from a DataFrame row using actual values."""
    # Try multiple column name variations for compatibility (HTML uses left/font_size_px, PDF uses x_left/font_size)
    left = _safe_float(row.get("x_left"))
    height = _safe_float(row.get("height"))
    font_size_px = _safe_float(row.get("font_size"))

    if left is None or font_size_px is None:
        return None

    return LayoutFingerprint(
        left=left,
        height=height or 0.0,
        font_size_px=font_size_px
    )


def _expand_segment_by_fingerprint(
    df_sorted: pd.DataFrame,
    seed_line_id: int,
    target_fingerprint: LayoutFingerprint,
    line_id_col: str,
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
        row = df_sorted.iloc[current_idx]
        fp = _get_row_fingerprint(row)
        
        if fp is None or not target_fingerprint.matches(fp, left_tolerance, height_tolerance, font_tolerance):
            break
        
        segment_indices.add(current_idx)
        current_idx -= 1
    
    # Expand downward
    current_idx = seed_idx + 1
    while current_idx < len(df_sorted):
        row = df_sorted.iloc[current_idx]
        fp = _get_row_fingerprint(row)
        
        if fp is None or not target_fingerprint.matches(fp, left_tolerance, height_tolerance, font_tolerance):
            break
        
        segment_indices.add(current_idx)
        current_idx += 1
    
    # Convert indices back to line_ids
    segment_line_ids = [
        df_sorted.iloc[idx][line_id_col]
        for idx in sorted(segment_indices)
    ]
    
    return segment_line_ids


# ---- Main TOC Segment Builder ---- #

def build_toc_segments(
    df: pd.DataFrame,
    candidates: List[TocRowCandidate],
    *,
    line_id_col: str = "line_id",
    table_id_col: str = "table_id",
    text_col: str = "text",
    toc_header_col: str = "toc_header_candidate",
    toc_row_col: str = "toc_row_candidate",
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
    - If no table_id: expand up/down by matching fingerprint
    Mark processed line_ids and continue to next unprocessed candidate.
    
    Args:
        df: DataFrame with toc_row_candidate column
        candidates: List of TocRowCandidate objects (not used, kept for API compatibility)
        line_id_col: Column name for line IDs (must be sortable)
        table_id_col: Column name for table IDs
        text_col: Column name for text
        toc_header_col: Column name for TOC header flags
        toc_row_col: Column name for TOC row candidates
        left_tolerance: Tolerance for grouping by left coordinate
        height_tolerance: Tolerance for grouping by height
        font_tolerance: Tolerance for grouping by font size
        header_lookback: Number of rows to look back for TOC headers
        
    Returns:
        List of TocSegment objects
    """
    if df.empty:
        return []
    
    if line_id_col not in df.columns or toc_row_col not in df.columns:
        return []
    
    # Sort DataFrame by line_id
    df_sorted = df.sort_values(line_id_col).reset_index(drop=True)
    line_id_to_idx = {row[line_id_col]: idx for idx, row in df_sorted.iterrows()}
    
    # Detect page header tables
    page_header_tables = detect_page_header_tables(
        df, table_id_col=table_id_col, text_col=text_col, line_id_col=line_id_col
    )
    
    segments: List[TocSegment] = []
    segment_id_counter = 0
    processed_line_ids: Set[int] = set()
    processed_table_ids: Set[Any] = set()
    
    # Iterate through DataFrame in line_id order
    for idx, row in df_sorted.iterrows():
        line_id = row[line_id_col]
        
        # Skip if already processed
        if line_id in processed_line_ids:
            continue
        
        # Skip if not a TOC candidate
        if not (pd.notna(row.get(toc_row_col)) and row.get(toc_row_col) is True):
            continue
        
        # We found a TOC candidate - build a segment
        table_id = row.get(table_id_col)
        
        # Check if this candidate has a valid table_id
        has_table = (table_id is not None and 
                    not pd.isna(table_id) and 
                    str(table_id).strip() != "")
        
        if has_table and table_id not in processed_table_ids:
            # === TABLE-BASED SEGMENT ===
            # Collect all rows in this table
            table_mask = df_sorted[table_id_col] == table_id
            table_rows = df_sorted[table_mask]
            
            if not table_rows.empty:
                all_line_ids = sorted(table_rows[line_id_col].tolist())
                
                # Count candidates in this table
                candidate_mask = table_rows[toc_row_col].notna() & (table_rows[toc_row_col] == True)
                candidate_line_ids = set(table_rows[candidate_mask][line_id_col].tolist())
                
                start_line_id = min(all_line_ids)
                end_line_id = max(all_line_ids)
                n_rows = len(all_line_ids)
                n_candidates = len(candidate_line_ids)
                candidate_ratio = n_candidates / n_rows if n_rows > 0 else 0.0
                
                max_consecutive = _calculate_max_consecutive(all_line_ids, candidate_line_ids)
                
                has_toc_header, header_line_ids = _check_toc_header_nearby(
                    df_sorted, start_line_id, line_id_col, toc_header_col, header_lookback
                )
                
                has_page_header = table_id in page_header_tables
                
                # Count rows with links
                n_links = 0
                if "has_link" in table_rows.columns:
                    n_links = int(table_rows["has_link"].apply(_safe_bool01).sum())
                
                # Count rows with dot leaders
                n_dot_leaders = 0
                if "toc_has_dot_leaders" in table_rows.columns:
                    n_dot_leaders = int(table_rows["toc_has_dot_leaders"].apply(_safe_bool01).sum())
                
                segments.append(TocSegment(
                    segment_id=segment_id_counter,
                    start_line_id=start_line_id,
                    end_line_id=end_line_id,
                    n_rows=n_rows,
                    n_candidates=n_candidates,
                    candidate_ratio=candidate_ratio,
                    max_consecutive_candidates=max_consecutive,
                    has_toc_header_nearby=has_toc_header,
                    has_page_header=has_page_header,
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
            # Get fingerprint for this candidate
            left = _safe_float(row.get("x_left"))
            height = _safe_float(row.get("height"))
            font_size_px = _safe_float(row.get("font_size"))
            
            if left is None or font_size_px is None:
                continue
            
            fingerprint = LayoutFingerprint(
                left=left,
                height=height or 0.0,
                font_size_px=font_size_px
            )
            
            # Expand up and down from this line_id until fingerprint changes
            segment_line_ids = _expand_segment_by_fingerprint(
                df_sorted, line_id, fingerprint,
                line_id_col, line_id_to_idx,
                left_tolerance, height_tolerance, font_tolerance
            )
            
            if not segment_line_ids:
                continue
            
            # Count candidates in this segment
            segment_set = set(segment_line_ids)
            candidate_line_ids = set()
            for seg_line_id in segment_line_ids:
                seg_row = df_sorted[df_sorted[line_id_col] == seg_line_id]
                if not seg_row.empty:
                    seg_row = seg_row.iloc[0]
                    if pd.notna(seg_row.get(toc_row_col)) and seg_row.get(toc_row_col) is True:
                        candidate_line_ids.add(seg_line_id)
            
            start_line_id = min(segment_line_ids)
            end_line_id = max(segment_line_ids)
            n_rows = len(segment_line_ids)
            n_candidates = len(candidate_line_ids)
            candidate_ratio = n_candidates / n_rows if n_rows > 0 else 0.0
            
            max_consecutive = _calculate_max_consecutive(sorted(segment_line_ids), candidate_line_ids)
            
            has_toc_header, header_line_ids = _check_toc_header_nearby(
                df_sorted, start_line_id, line_id_col, toc_header_col, header_lookback
            )
            
            # Count rows with links in this segment
            n_links = 0
            n_dot_leaders = 0
            if "has_link" in df_sorted.columns or "toc_has_dot_leaders" in df_sorted.columns:
                for seg_line_id in segment_line_ids:
                    seg_row = df_sorted[df_sorted[line_id_col] == seg_line_id]
                    if not seg_row.empty:
                        seg_row_data = seg_row.iloc[0]
                        if "has_link" in df_sorted.columns and _safe_bool01(seg_row_data.get("has_link")):
                            n_links += 1
                        if "toc_has_dot_leaders" in df_sorted.columns and _safe_bool01(seg_row_data.get("toc_has_dot_leaders")):
                            n_dot_leaders += 1
            
            segments.append(TocSegment(
                segment_id=segment_id_counter,
                start_line_id=start_line_id,
                end_line_id=end_line_id,
                n_rows=n_rows,
                n_candidates=n_candidates,
                candidate_ratio=candidate_ratio,
                max_consecutive_candidates=max_consecutive,
                has_toc_header_nearby=has_toc_header,
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
# STEP 4: Select Final TOC's - scoring
# ==========================================

@dataclass
class TocScore:
    """Score breakdown for a TOC segment"""
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


def score_and_filter_toc_segments(
    segments: List[TocSegment],
    df: pd.DataFrame,
    *,
    # Disqualification filters (hard cutoffs)
    min_rows: int = 4,
    min_consecutive: int = 3,
    min_candidate_ratio: float = 0.5,
    # Scoring weights (positive)
    has_toc_header_weight: float = 1.0,
    has_page_header_weight: float = 1.0,
    links_weight_per_link: float = 0.5,
    links_score_cap: float = 2.0,
    consecutive_weight_per_count: float = 0.2,
    consecutive_score_cap: float = 1.5,
    dot_leaders_weight_per_count: float = 0.2,
    dot_leaders_score_cap: float = 1.0,
    # Scoring weights (negative)
    fingerprint_penalty: float = 0.0,
    currency_penalty: float = 0.5,
    # Acceptance threshold
    min_score_threshold: float = 2.5,
    # DataFrame columns
    line_id_col: str = "line_id",
    text_col: str = "text",
) -> List[TocScore]:
    """
    Score and filter TOC segments based on multiple signals.
    
    Returns list of TocScore objects with detailed scoring breakdown.
    Only segments with score > min_score_threshold are marked as accepted.
    
    Args:
        segments: List of TocSegment objects to score
        df: DataFrame containing the line data
        
        Disqualification filters:
            min_rows: Minimum number of rows in segment (default: 4)
            min_consecutive: Minimum consecutive candidates (default: 3)
            min_candidate_ratio: Minimum candidate ratio (default: 0.5)
        
        Scoring weights:
            has_toc_header_weight: Score if TOC header found nearby (default: 1.0)
            has_page_header_weight: Score if page header present (default: 1.0)
            links_weight_per_link: Score per linked row (default: 0.5)
            links_score_cap: Maximum score from links (default: 2.0)
            consecutive_weight_per_count: Score per consecutive candidate (default: 0.2)
            consecutive_score_cap: Maximum score from consecutive candidates (default: 1.5)
            dot_leaders_weight_per_count: Score per row with dot leaders (default: 0.2)
            dot_leaders_score_cap: Maximum score from dot leaders (default: 1.0)
            fingerprint_penalty: Penalty for fingerprint-based segments (default: 0.5)
            currency_penalty: Penalty if currency symbols found (default: 0.5)
            min_score_threshold: Minimum score to accept segment (default: 2.5)
    
    Returns:
        List of TocScore objects (all segments scored, check .accepted flag)
    """
    scores: List[TocScore] = []
    
    for segment in segments:
        # === DISQUALIFICATION FILTERS ===
        passed_filters = True
        
        if segment.n_rows < min_rows:
            passed_filters = False
        elif segment.max_consecutive_candidates < min_consecutive:
            passed_filters = False
        elif segment.candidate_ratio < min_candidate_ratio:
            passed_filters = False
        
        # If doesn't pass filters, score is 0
        if not passed_filters:
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
        
        # Positive signals
        header_score = has_toc_header_weight if segment.has_toc_header_nearby else 0.0
        page_header_score = has_page_header_weight if segment.has_page_header else 0.0
        
        links_score = min(segment.n_links * links_weight_per_link, links_score_cap)
        consecutive_score = min(segment.max_consecutive_candidates * consecutive_weight_per_count, consecutive_score_cap)
        ratio_score = segment.candidate_ratio  # Already 0.0 to 1.0
        dot_leaders_score = min(segment.n_dot_leaders * dot_leaders_weight_per_count, dot_leaders_score_cap)
        
        # Negative signals
        fp_penalty = fingerprint_penalty if segment.is_fingerprint_based else 0.0
        
        # Check for currency symbols in segment text
        currency_found = False
        if text_col in df.columns:
            segment_rows = df[
                (df[line_id_col] >= segment.start_line_id) & 
                (df[line_id_col] <= segment.end_line_id)
            ]
            for _, row in segment_rows.iterrows():
                text = str(row.get(text_col, ""))
                if any(sym in text for sym in ("$", "€", "£", "¥")):
                    currency_found = True
                    break
        
        curr_penalty = currency_penalty if currency_found else 0.0
        
        # Calculate total score
        total_score = (
            header_score +
            page_header_score +
            links_score +
            consecutive_score +
            ratio_score +
            dot_leaders_score -
            fp_penalty -
            curr_penalty
        )
        
        accepted = total_score > min_score_threshold
        
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
            accepted=accepted,
        ))
    
    return scores





# ==========================================
# STEP 5: Close Gaps Between TOC Segments
# ==========================================

def close_toc_gaps(
    df: pd.DataFrame,
    *,
    line_id_col: str = "line_id",
    block_role_col: str = "block_role",
    max_gap_between_segments: int = 3,
) -> pd.DataFrame:
    """
    Close gaps between TOC-related rows for better continuity.
    
    Two gap-closing operations:
    1. Fill gaps between toc_header and toc (mark as toc_header)
    2. Fill small gaps between toc segments (mark as toc, max 3 rows)
    
    Args:
        df: DataFrame with block_role column
        line_id_col: Column name for line IDs
        block_role_col: Column name for block roles
        max_gap_between_segments: Maximum gap to close between toc segments (default: 3)
        
    Returns:
        DataFrame with gaps filled in block_role column
    """
    if df.empty or block_role_col not in df.columns or line_id_col not in df.columns:
        return df
    
    out = df.copy()
    
    # Sort by line_id for sequential processing
    out = out.sort_values(line_id_col).reset_index(drop=True)
    
    # === OPERATION 1: Close gaps between toc_header and toc ===
    # For each toc_header, find the next toc row and fill the gap
    
    toc_header_mask = out[block_role_col] == "toc_header"
    toc_mask = out[block_role_col] == "toc"
    
    toc_header_indices = out[toc_header_mask].index.tolist()
    toc_indices = out[toc_mask].index.tolist()
    
    for header_idx in toc_header_indices:
        header_line_id = out.at[header_idx, line_id_col]
        
        # Find the next toc row after this header
        next_toc_indices = [idx for idx in toc_indices if out.at[idx, line_id_col] > header_line_id]
        
        if next_toc_indices:
            next_toc_idx = min(next_toc_indices)
            next_toc_line_id = out.at[next_toc_idx, line_id_col]
            
            # Fill gap between header and toc with toc_header
            gap_mask = (
                (out[line_id_col] > header_line_id) & 
                (out[line_id_col] < next_toc_line_id) &
                out[block_role_col].isna()
            )
            out.loc[gap_mask, block_role_col] = "toc_header"
    
    # === OPERATION 2: Close small gaps between toc segments ===
    # Find separate toc segments and merge if gap is small enough
    
    # Refresh toc_mask after operation 1
    toc_mask = out[block_role_col] == "toc"
    toc_indices = out[toc_mask].index.tolist()
    
    if not toc_indices:
        return out
    
    # Build segments: group toc rows that are truly contiguous (consecutive line_ids)
    # or already touching (gap = 0)
    segments = []
    current_segment = [toc_indices[0]]
    
    for i in range(1, len(toc_indices)):
        prev_idx = toc_indices[i - 1]
        curr_idx = toc_indices[i]
        
        prev_line_id = out.at[prev_idx, line_id_col]
        curr_line_id = out.at[curr_idx, line_id_col]
        
        # Only group together if consecutive or touching (gap = 0)
        # This identifies truly separate segments
        if curr_line_id - prev_line_id == 1:
            current_segment.append(curr_idx)
        else:
            # Found a gap - start a new segment
            segments.append(current_segment)
            current_segment = [curr_idx]
    
    segments.append(current_segment)
    
    # Now we have truly separate segments - merge them if gap is small
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        
        seg1_end_idx = seg1[-1]
        seg2_start_idx = seg2[0]
        
        seg1_end_line_id = out.at[seg1_end_idx, line_id_col]
        seg2_start_line_id = out.at[seg2_start_idx, line_id_col]
        
        gap_size = seg2_start_line_id - seg1_end_line_id - 1
        
        if 0 < gap_size <= max_gap_between_segments:
            # Fill the gap with toc
            gap_mask = (
                (out[line_id_col] > seg1_end_line_id) & 
                (out[line_id_col] < seg2_start_line_id) &
                out[block_role_col].isna()
            )
            out.loc[gap_mask, block_role_col] = "toc"
    
    return out


# ==========================================
# PUBLIC API
# ==========================================

def detect_and_annotate_tocs(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    *,
    # Column names
    text_col: str = "text",
    has_link_col: str = "has_link",
    line_id_col: str = "line_id",
    # Scoring parameters (easily tunable)
    min_rows: int = 4,
    min_consecutive: int = 3,
    min_candidate_ratio: float = 0.5,
    min_score_threshold: float = 2.5,
    # Debug options
    include_debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Complete TOC detection pipeline.
    
    Pipeline:
    1. Remove TOC pointer links and reindex
    2. Identify header candidates
    3. Identify row candidates
    4. Build segments
    5. Score and filter segments
    6. Annotate winning rows with block_role = 'toc' or 'toc_header'
    7. Close gaps between toc_header/toc and between toc segments
    8. Clean up debug columns (optional)
    
    Args:
        df: Input DataFrame with text, has_link, line_id columns
        page_label_config: Compiled page label patterns from YAML
        
        Column names:
            text_col: Column name for text (default: "text")
            has_link_col: Column name for link flag (default: "has_link")
            line_id_col: Column name for line IDs (default: "line_id")
        
        Scoring parameters (see score_and_filter_toc_segments for full list):
            min_rows: Minimum rows in segment (default: 4)
            min_consecutive: Minimum consecutive candidates (default: 3)
            min_candidate_ratio: Minimum candidate ratio (default: 0.5)
            min_score_threshold: Minimum score to accept (default: 2.5)
        
        Debug options:
            include_debug_cols: Keep toc_* debug columns in output (default: False)
    
    Returns:
        DataFrame with added columns:
        - block_role: 'toc' for TOC entry rows, 'toc_header' for header rows
        - (if include_debug_cols=True):
          - toc_header_candidate: TRUE for header candidates
          - toc_row_candidate: TRUE for row candidates
          - toc_page_label_token, toc_page_label_type, toc_has_dot_leaders
    """
    # STEP 0: Remove TOC pointers and reindex
    df = remove_toc_pointers(
        df,
        text_col=text_col,
        has_link_col=has_link_col,
        line_id_col=line_id_col
    )
    
    # STEP 1: Identify header candidates
    df = identify_toc_header_candidates(df, text_col=text_col)
    
    # STEP 2: Identify row candidates
    df, toc_candidates = add_toc_row_candidates(
        df, 
        page_label_config, 
        text_col=text_col,
        include_debug_token_cols=include_debug_cols,
    )
    
    # STEP 3: Build segments
    segments = build_toc_segments(df, toc_candidates, line_id_col=line_id_col)
    
    # STEP 4: Score and filter
    scores = score_and_filter_toc_segments(
        segments,
        df,
        min_rows=min_rows,
        min_consecutive=min_consecutive,
        min_candidate_ratio=min_candidate_ratio,
        min_score_threshold=min_score_threshold,
        line_id_col=line_id_col,
        text_col=text_col,
    )
    
    # STEP 5: Annotate winning segments
    # Initialize block_role column if it doesn't exist
    if "block_role" not in df.columns:
        df["block_role"] = pd.NA
    
    # Mark all rows in accepted segments
    for score_obj in scores:
        if score_obj.accepted:
            seg = score_obj.segment
            
            # Mark header rows that are associated with this segment
            for header_line_id in seg.nearby_header_line_ids:
                header_mask = df[line_id_col] == header_line_id
                df.loc[header_mask, "block_role"] = "toc_header"
            
            # Mark TOC content rows
            mask = (df[line_id_col] >= seg.start_line_id) & (df[line_id_col] <= seg.end_line_id)
            df.loc[mask, "block_role"] = "toc"
    
    # STEP 6: Close gaps between TOC segments and headers
    df = close_toc_gaps(df, line_id_col=line_id_col)
    
    # STEP 7: Clean up debug columns if not requested
    if not include_debug_cols:
        debug_cols = [
            "toc_header_candidate",
            "toc_row_candidate", 
            "toc_page_label_token",
            "toc_page_label_type",
            "toc_has_dot_leaders",
        ]
        for col in debug_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
    
    return df

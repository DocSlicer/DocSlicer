# d05_exhibit_detector.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

import pandas as pd

from .._utils.yaml_compilers.exhibit_patterns import ExhibitPatternConfig

# =========================
# Config
# =========================

# Minimum characters in exhibit description (after exhibit code)
MIN_DESCRIPTION_CHARS = 3
EXHIBIT_ROW_MAX_CHARS = 500

# =========================
# Dataclasses
# =========================

# Pattern type now includes base patterns and their "_with_markers" variations
ExhibitRowPatternType = Literal[
    "multi_parens", "multi_parens_with_markers",
    "alpha_with_parens", "alpha_with_parens_with_markers",
    "number_with_parens", "number_with_parens_with_markers",
    "dotted_with_parens", "dotted_with_parens_with_markers",
    "numeric_or_dotted", "numeric_or_dotted_with_markers",
    "exhibit_prefix_row", "exhibit_prefix_row_with_markers",
    "ex_code_row", "ex_code_row_with_markers",
    "subpart_not_applicable", "subpart_not_applicable_with_markers",
    "hundred_series_exhibit", "hundred_series_exhibit_with_markers"
]
ExhibitHeaderPatternType = Literal["item_any_exhibits", "exhibit_index", "index_to_exhibits", "exhibits_only"]

@dataclass(frozen=True)
class ExhibitRowCandidate:
    is_row_candidate: bool
    exhibit_row_pattern: Optional[ExhibitRowPatternType]
    exhibit_number: Optional[str]
    pattern_strength: Optional[str]  # "strong" or "weak"
    has_link: bool
    table_id: Optional[Any]

    # layout (helps for <p>-based exhibits and cross-page merge)
    left: Optional[float]
    height: Optional[float]
    font_size: Optional[float]
    text_align: Optional[str]
    page_number: Optional[int]


@dataclass(frozen=True)
class LayoutFingerprint:
    """Fingerprint for grouping similar non-table rows using actual observed values"""
    left: float  # Actual left coordinate
    height: float  # Actual height
    font_size: float  # Actual font size
    
    @classmethod
    def from_candidate(cls, candidate: ExhibitRowCandidate) -> Optional['LayoutFingerprint']:
        """Create fingerprint from actual values"""
        if candidate.left is None or candidate.font_size is None:
            return None
        return cls(
            left=candidate.left,
            height=candidate.height or 0.0,
            font_size=candidate.font_size
        )
    
    def matches(self, other: 'LayoutFingerprint',
                left_tolerance: float = 5.0,
                height_tolerance: float = 2.0,
                font_tolerance: float = 0.5) -> bool:
        """Check if another fingerprint matches within tolerances"""
        return (
            abs(self.left - other.left) <= left_tolerance and
            abs(self.height - other.height) <= height_tolerance and
            abs(self.font_size - other.font_size) <= font_tolerance
        )


@dataclass(frozen=True)
class ExhibitSegment:
    """
    A small, localized cluster of candidate rows.
    Segments are later merged into final exhibits.
    """
    segment_id: int
    start_line_id: int
    end_line_id: int
    n_rows: int
    n_candidates: int
    candidate_ratio: float
    max_consecutive_candidates: int  # Longest run of consecutive candidate rows
    has_exhibit_header_nearby: bool
    has_exhibit_number_header: bool
    has_other_segment_above: bool
    n_links: int # Number of rows in this object with hyperlinks
    # Header line_ids found nearby (outside segment boundaries)
    nearby_header_line_ids: List[int]
    above_exhibit_segment_id: int
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
# STEP 1: Build Exhibit Header Candidate Column
# ==========================================

def identify_exhibit_header_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    *,
    text_col: str = "text",
    header_col: str = "exhibit_header_candidate",
    max_len: int = 150,
) -> pd.DataFrame:
    """
    Identify Exhibit header candidates (e.g., "Item ... Exhibits", "EXHIBIT INDEX", "INDEX TO EXHIBITS").
    
    Adds column `exhibit_header_candidate`:
    - Pattern name where text matches Exhibit header regex and length <= max_len
    - NA elsewhere
    
    Args:
        df: DataFrame with text column
        exhibit_config: Compiled exhibit patterns from YAML
        text_col: Column name for text
        header_col: Column name for output header candidate flag
        max_len: Maximum text length for header candidates
        
    Returns:
        DataFrame with exhibit_header_candidate column added
    """
    out = df.copy()
    out[header_col] = pd.NA
    
    if text_col not in out.columns:
        return out
    
    text = out[text_col].astype(str)
    
    # Check each row against all header patterns
    for idx, raw_text in text.items():
        txt = (raw_text or "").strip()
        if not txt or len(txt) > max_len:
            continue
        
        # Try each header pattern
        for pattern in exhibit_config.header_patterns:
            if pattern.compiled.match(txt):
                out.loc[idx, header_col] = pattern.name
                break  # First match wins
    
    return out


# ==========================================
# STEP 2: Build Exhibit Row Candidates
# ==========================================

# ---- Helper Functions ---- #

def _contains_anti_pattern(text: str, anti_patterns) -> bool:
    """Check if text contains patterns that indicate it's NOT an exhibit."""
    for pattern in anti_patterns:
        if pattern.compiled.search(text):
            return True
    return False


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


def _check_exhibit_row_match(
    text: str,
    pattern: Any,
    pattern_name: str,
    anti_patterns,
) -> Tuple[bool, Optional[str]]:
    """
    Check if text matches an exhibit row pattern.
    The pattern match IS the validation - if it matches, it's a valid exhibit row.
    
    Footnote markers (*, †, ‡, §, ¶, #) are handled automatically by the pattern
    compiler, which generates variations with/without markers.
    
    Returns:
        Tuple of (is_valid, exhibit_number)
        - is_valid: Whether text matches the pattern
        - exhibit_number: Extracted exhibit number (for metadata)
    """
    txt = (text or "").strip()
    if not txt:
        return False, None
    
    # Check if pattern matches - this IS the validation
    # No need to strip footnote markers - patterns handle all variations
    match = pattern.compiled.match(txt)
    if not match:
        return False, None
    
    # Extract exhibit number from the matched portion (for metadata)
    exhibit_number = None
    
    # Remove "_with_markers" suffix from pattern name if present
    base_pattern_name = pattern_name.replace("_with_markers", "")
    
    if base_pattern_name in ("exhibit_prefix_row", "ex_code_row", "hundred_series_exhibit"):
        # These patterns have named groups for the code
        try:
            exhibit_number = match.group("code")
        except (IndexError, AttributeError):
            # If no named group, extract from start of match
            exhibit_number = match.group(0).strip().split()[0]
    
    elif base_pattern_name in ("multi_parens", "alpha_with_parens", 
                          "number_with_parens", "dotted_with_parens", "numeric_or_dotted"):
        # Extract the number/code portion from start of text
        matched_text = match.group(0).strip()
        # Take up to first space as the exhibit number
        exhibit_number = matched_text.split()[0] if matched_text.split() else matched_text
        # Clean footnote markers from exhibit number for display
        # Matches the FOOTNOTE_MARKERS constant in yaml_compilers/exhibit_patterns.py
        if exhibit_number:
            for marker in '*†‡§¶#+^■●▲▼◆◇○□△▽◊~':
                exhibit_number = exhibit_number.replace(marker, '')
    
    elif base_pattern_name == "subpart_not_applicable":
        # Extract the letter in parens (e.g., "(d)")
        exhibit_number = txt.split()[0]  # First token is the code
    
    return True, exhibit_number


# ---- Main Exhibit Row Candidate Builder ---- #

def add_exhibit_row_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
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
    max_chars: int = EXHIBIT_ROW_MAX_CHARS,
    # output
    include_debug_cols: bool = True,
    build_objects: bool = True,
) -> Tuple[pd.DataFrame, List[ExhibitRowCandidate]]:
    """
    Adds:
      - exhibit_row_candidate: Pattern name where row rules pass, else NA
      - (optional debug) exhibit_number: The extracted exhibit number
      - (optional debug) pattern_strength: "strong" or "weak" indicator
    
    Row rules:
      - non-empty text
      - len(text) <= max_chars
      - text matches one of the exhibit row patterns from YAML
      - has sufficient description after exhibit number (for most patterns)
    
    Returns: (out_df, objects)
      - objects list is empty if build_objects=False
    """
    out = df.copy()
    
    # initialize columns as NA (not False) for easier CSV inspection
    out["exhibit_row_candidate"] = pd.NA
    
    if include_debug_cols:
        out["exhibit_number"] = pd.NA
        out["pattern_strength"] = pd.NA
    
    if text_col not in out.columns:
        return out, []
    
    texts = out[text_col].astype(str)
    
    candidates: List[ExhibitRowCandidate] = []
    
    # Row candidates (iterate once; keeps logic clear and debuggable)
    for idx, raw_text in texts.items():
        text = (raw_text or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            continue
        
        # Try each exhibit row pattern
        matched = False
        matched_pattern_name = None
        matched_pattern_strength = None
        exhibit_number = None
        
        for pattern in exhibit_config.row_patterns:
            is_valid, number = _check_exhibit_row_match(
                text, pattern, pattern.name, exhibit_config.anti_patterns
            )
            if is_valid:
                matched = True
                matched_pattern_name = pattern.name
                matched_pattern_strength = pattern.strength
                exhibit_number = number
                break  # First match wins
        
        if not matched:
            continue
        
        # Debug cols
        if include_debug_cols:
            out.loc[idx, "exhibit_number"] = exhibit_number
            out.loc[idx, "pattern_strength"] = matched_pattern_strength
        
        # Final accept
        out.loc[idx, "exhibit_row_candidate"] = matched_pattern_name
        
        if build_objects:
            has_link = _safe_bool01(out.at[idx, has_link_col]) if has_link_col in out.columns else False
            table_id = out.at[idx, table_id_col] if table_id_col in out.columns else None
            left = _safe_float(out.at[idx, left_col]) if left_col in out.columns else None
            height = _safe_float(out.at[idx, height_col]) if height_col in out.columns else None
            font_size = _safe_float(out.at[idx, font_col]) if font_col in out.columns else None
            text_align = _safe_str_or_none(out.at[idx, align_col]) if align_col in out.columns else None
            page_number = _safe_int(out.at[idx, page_number_col]) if page_number_col in out.columns else None
            
            candidates.append(
                ExhibitRowCandidate(
                    is_row_candidate=True,
                    exhibit_row_pattern=matched_pattern_name,  # type: ignore[arg-type]
                    exhibit_number=exhibit_number,
                    pattern_strength=matched_pattern_strength,
                    has_link=has_link,
                    table_id=table_id,
                    left=left,
                    height=height,
                    font_size=font_size,
                    text_align=text_align,
                    page_number=page_number,
                )
            )
    
    return out, candidates



# ==========================================
# STEP 3: Build Exhibit Segments
# ==========================================

# ---- Helper Functions ---- #

_EXHIBIT_NUMBER_HEADER_RE = re.compile(r"^\s*exhibit\s+(?:number|no\.?)\s*$", re.IGNORECASE)


def detect_exhibit_number_header_tables(
    df: pd.DataFrame,
    *,
    table_id_col: str = "table_id",
    text_col: str = "text",
    line_id_col: str = "line_id",
) -> Set[Any]:
    """
    Return a set of table_id values where the FIRST line
    contains 'Exhibit Number' or 'Exhibit No' (case-insensitive).
    
    Strong signal that the table is an exhibit listing.
    """
    if table_id_col not in df.columns or text_col not in df.columns:
        return set()
    
    exhibit_number_header_table_ids: Set[Any] = set()
    
    for table_id, g in df.groupby(table_id_col):
        if pd.isna(table_id):
            continue
        
        # Determine first line of the table
        if line_id_col in g.columns:
            first_row = g.sort_values(line_id_col).iloc[0]
        else:
            first_row = g.iloc[0]
        
        text = str(first_row[text_col]).strip()
        if _EXHIBIT_NUMBER_HEADER_RE.match(text):
            exhibit_number_header_table_ids.add(table_id)
    
    return exhibit_number_header_table_ids


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


def _check_exhibit_header_nearby(
    df_sorted: pd.DataFrame,
    start_line_id: int,
    line_id_col: str,
    exhibit_header_col: str,
    lookback: int,
) -> Tuple[bool, List[int]]:
    """
    Check if an exhibit header appears within lookback rows before start_line_id.
    
    Returns:
        Tuple of (has_header: bool, header_line_ids: List[int])
    """
    if exhibit_header_col not in df_sorted.columns:
        return False, []
    
    # Find rows before start_line_id
    mask = df_sorted[line_id_col] < start_line_id
    before_rows = df_sorted[mask].tail(lookback)
    
    if before_rows.empty:
        return False, []
    
    # Find all header candidate rows
    header_line_ids = []
    for _, row in before_rows.iterrows():
        # Check for non-NA values in exhibit_header_col (pattern name stored there)
        if pd.notna(row.get(exhibit_header_col)):
            header_line_ids.append(int(row[line_id_col]))
    
    has_header = len(header_line_ids) > 0
    return has_header, header_line_ids


def _check_other_segment_above(
    df_sorted: pd.DataFrame,
    start_line_id: int,
    line_id_col: str,
    processed_segments: List[ExhibitSegment],
    lookback: int,
) -> Tuple[bool, int]:
    """
    Check if another exhibit segment exists within lookback rows before start_line_id.
    
    Returns:
        Tuple of (has_segment_above: bool, segment_id_above: int)
        - segment_id_above is -1 if no segment found
    """
    if not processed_segments:
        return False, -1
    
    # Find rows before start_line_id
    mask = df_sorted[line_id_col] < start_line_id
    before_rows = df_sorted[mask].tail(lookback)
    
    if before_rows.empty:
        return False, -1
    
    # Check if any line_id in the lookback range is part of a previous segment
    before_line_ids = set(before_rows[line_id_col].tolist())
    
    for segment in reversed(processed_segments):  # Check most recent first
        # Check if segment overlaps with our lookback range
        segment_line_ids = set(range(segment.start_line_id, segment.end_line_id + 1))
        if segment_line_ids & before_line_ids:  # If there's any intersection
            return True, segment.segment_id
    
    return False, -1


def _get_row_fingerprint(
    row: pd.Series,
    left_tolerance: float = None,  # Not used, kept for API compatibility
    height_tolerance: float = None,  # Not used, kept for API compatibility
    font_tolerance: float = None,  # Not used, kept for API compatibility
) -> Optional[LayoutFingerprint]:
    """Extract fingerprint from a DataFrame row using actual values."""
    left = _safe_float(row.get("x_left"))
    height = _safe_float(row.get("height"))
    font_size = _safe_float(row.get("font_size"))
    
    if left is None or font_size is None:
        return None
    
    return LayoutFingerprint(
        left=left,
        height=height or 0.0,
        font_size=font_size
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


# ---- Main Exhibit Segment Builder ---- #

def build_exhibit_segments(
    df: pd.DataFrame,
    candidates: List[ExhibitRowCandidate],
    *,
    line_id_col: str = "line_id",
    table_id_col: str = "table_id",
    text_col: str = "text",
    exhibit_header_col: str = "exhibit_header_candidate",
    exhibit_row_col: str = "exhibit_row_candidate",
    left_tolerance: float = 5.0,
    height_tolerance: float = 2.0,
    font_tolerance: float = 0.5,
    header_lookback: int = 3,
    segment_lookback: int = 5,
) -> List[ExhibitSegment]:
    """
    Build ExhibitSegment objects from exhibit row candidates.
    
    Strategy:
    Iterate through DataFrame in line_id order. When encountering an exhibit_row_candidate:
    - If it has a table_id: collect all rows in that table as one segment
    - If no table_id: expand up/down by matching fingerprint
    Mark processed line_ids and continue to next unprocessed candidate.
    
    Key differences from TOC segments:
    - Exhibits can span multiple pages/tables, so multiple segments may share one header
    - Check for other segments above (within 5 rows) to detect continuation
    - Blank rows between segments are expected (page breaks, horizontal rules)
    
    Args:
        df: DataFrame with exhibit_row_candidate column
        candidates: List of ExhibitRowCandidate objects (not used, kept for API compatibility)
        line_id_col: Column name for line IDs (must be sortable)
        table_id_col: Column name for table IDs
        text_col: Column name for text
        exhibit_header_col: Column name for exhibit header flags
        exhibit_row_col: Column name for exhibit row candidates
        left_tolerance: Tolerance for grouping by left coordinate
        height_tolerance: Tolerance for grouping by height
        font_tolerance: Tolerance for grouping by font size
        header_lookback: Number of rows to look back for exhibit headers (default: 3)
        segment_lookback: Number of rows to look back for other segments (default: 5)
        
    Returns:
        List of ExhibitSegment objects
    """
    if df.empty:
        return []
    
    if line_id_col not in df.columns or exhibit_row_col not in df.columns:
        return []
    
    # Sort DataFrame by line_id
    df_sorted = df.sort_values(line_id_col).reset_index(drop=True)
    line_id_to_idx = {row[line_id_col]: idx for idx, row in df_sorted.iterrows()}
    
    # Detect exhibit number header tables
    exhibit_number_header_tables = detect_exhibit_number_header_tables(
        df, table_id_col=table_id_col, text_col=text_col, line_id_col=line_id_col
    )
    
    segments: List[ExhibitSegment] = []
    segment_id_counter = 0
    processed_line_ids: Set[int] = set()
    processed_table_ids: Set[Any] = set()
    
    # Iterate through DataFrame in line_id order
    for idx, row in df_sorted.iterrows():
        line_id = row[line_id_col]
        
        # Skip if already processed
        if line_id in processed_line_ids:
            continue
        
        # Skip if not an exhibit candidate (check for non-NA pattern name)
        if pd.isna(row.get(exhibit_row_col)):
            continue
        
        # We found an exhibit candidate - build a segment
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
                candidate_mask = table_rows[exhibit_row_col].notna()
                candidate_line_ids = set(table_rows[candidate_mask][line_id_col].tolist())
                
                start_line_id = min(all_line_ids)
                end_line_id = max(all_line_ids)
                n_rows = len(all_line_ids)
                n_candidates = len(candidate_line_ids)
                candidate_ratio = n_candidates / n_rows if n_rows > 0 else 0.0
                
                max_consecutive = _calculate_max_consecutive(all_line_ids, candidate_line_ids)
                
                has_exhibit_header, header_line_ids = _check_exhibit_header_nearby(
                    df_sorted, start_line_id, line_id_col, exhibit_header_col, header_lookback
                )
                
                # IMPORTANT: Only link to segment above if this segment has NO header of its own
                # A segment with its own header starts a new chain
                if has_exhibit_header:
                    # This segment has its own header - it's a chain root, don't link to segments above
                    has_other_segment_above = False
                    above_segment_id = -1
                else:
                    # No header - check if we can link to a segment above
                    has_other_segment_above, above_segment_id = _check_other_segment_above(
                        df_sorted, start_line_id, line_id_col, segments, segment_lookback
                    )
                
                has_exhibit_number_header = table_id in exhibit_number_header_tables
                
                # Count rows with links
                n_links = 0
                if "has_link" in table_rows.columns:
                    n_links = int(table_rows["has_link"].apply(_safe_bool01).sum())
                
                segments.append(ExhibitSegment(
                    segment_id=segment_id_counter,
                    start_line_id=start_line_id,
                    end_line_id=end_line_id,
                    n_rows=n_rows,
                    n_candidates=n_candidates,
                    candidate_ratio=candidate_ratio,
                    max_consecutive_candidates=max_consecutive,
                    has_exhibit_header_nearby=has_exhibit_header,
                    has_exhibit_number_header=has_exhibit_number_header,
                    has_other_segment_above=has_other_segment_above,
                    n_links=n_links,
                    nearby_header_line_ids=header_line_ids,
                    above_exhibit_segment_id=above_segment_id,
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
            font_size = _safe_float(row.get("font_size"))
            
            if left is None or font_size is None:
                continue
            
            fingerprint = LayoutFingerprint(
                left=left,
                height=height or 0.0,
                font_size=font_size
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
                    if pd.notna(seg_row.get(exhibit_row_col)):
                        candidate_line_ids.add(seg_line_id)
            
            start_line_id = min(segment_line_ids)
            end_line_id = max(segment_line_ids)
            n_rows = len(segment_line_ids)
            n_candidates = len(candidate_line_ids)
            candidate_ratio = n_candidates / n_rows if n_rows > 0 else 0.0
            
            max_consecutive = _calculate_max_consecutive(sorted(segment_line_ids), candidate_line_ids)
            
            has_exhibit_header, header_line_ids = _check_exhibit_header_nearby(
                df_sorted, start_line_id, line_id_col, exhibit_header_col, header_lookback
            )
            
            # IMPORTANT: Only link to segment above if this segment has NO header of its own
            # A segment with its own header starts a new chain
            if has_exhibit_header:
                # This segment has its own header - it's a chain root, don't link to segments above
                has_other_segment_above = False
                above_segment_id = -1
            else:
                # No header - check if we can link to a segment above
                has_other_segment_above, above_segment_id = _check_other_segment_above(
                    df_sorted, start_line_id, line_id_col, segments, segment_lookback
                )
            
            # Count rows with links in this segment
            n_links = 0
            if "has_link" in df_sorted.columns:
                for seg_line_id in segment_line_ids:
                    seg_row = df_sorted[df_sorted[line_id_col] == seg_line_id]
                    if not seg_row.empty:
                        seg_row_data = seg_row.iloc[0]
                        if _safe_bool01(seg_row_data.get("has_link")):
                            n_links += 1
            
            segments.append(ExhibitSegment(
                segment_id=segment_id_counter,
                start_line_id=start_line_id,
                end_line_id=end_line_id,
                n_rows=n_rows,
                n_candidates=n_candidates,
                candidate_ratio=candidate_ratio,
                max_consecutive_candidates=max_consecutive,
                has_exhibit_header_nearby=has_exhibit_header,
                has_exhibit_number_header=False,
                has_other_segment_above=has_other_segment_above,
                n_links=n_links,
                nearby_header_line_ids=header_line_ids,
                above_exhibit_segment_id=above_segment_id,
                table_id=None,
                fingerprint=fingerprint,
            ))
            segment_id_counter += 1
            processed_line_ids.update(segment_line_ids)
    
    return segments

# ==========================================
# STEP 4: Select Final Exhibits - scoring
# ==========================================

@dataclass
class ExhibitScore:
    """Confidence score for an exhibit segment"""
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


def _find_root_segment(
    segment: ExhibitSegment,
    segments_by_id: Dict[int, ExhibitSegment]
) -> ExhibitSegment:
    """
    Follow the segment chain upward to find the root segment.
    The root segment is one that has no segment above it.
    """
    current = segment
    visited = {segment.segment_id}  # Prevent infinite loops
    
    while current.has_other_segment_above:
        above_id = current.above_exhibit_segment_id
        
        # Safety checks
        if above_id < 0 or above_id not in segments_by_id:
            break
        if above_id in visited:  # Circular reference
            break
        
        visited.add(above_id)
        current = segments_by_id[above_id]
    
    return current


def score_exhibit_segments(
    df: pd.DataFrame,
    segments: List[ExhibitSegment],
    candidates: List[ExhibitRowCandidate],
    *,
    # Disqualification
    require_header_in_chain: bool = True,
    # Confidence scoring weights
    links_weight_per_link: float = 0.5,
    links_score_cap: float = 2.0,
    consecutive_weight_per_count: float = 0.2,
    consecutive_score_cap: float = 1.5,
    strong_pattern_weight: float = 1.0,
    strong_pattern_cap: float = 2.0,
    weak_first_weight: float = 1.0,
    weak_additional_weight: float = 0.1,
    weak_pattern_cap: float = 1.5,
    # Acceptance threshold
    min_score_threshold: float = 2.0,
    # Column names
    line_id_col: str = "line_id",
    exhibit_row_col: str = "exhibit_row_candidate",
    pattern_strength_col: str = "pattern_strength",
) -> List[ExhibitScore]:
    """
    Score exhibit segments with disqualification and confidence scoring.
    
    Disqualification rules:
    1. If segment has no header nearby AND no segment above -> disqualified
    2. If segment is in a chain, follow chain to root - root MUST have header
    
    Confidence scoring:
    - has_exhibit_header_nearby: +1
    - has_exhibit_number_header: +1
    - has_other_segment_above: +1 (continuation of exhibit list)
    - Links: links_weight_per_link per link, capped at links_score_cap
    - Consecutive candidates: consecutive_weight_per_count per count, capped at consecutive_score_cap
    - Strong patterns: strong_pattern_weight per strong element, capped at strong_pattern_cap
    - Weak patterns: weak_first_weight for first, then weak_additional_weight each, capped at weak_pattern_cap
    
    Args:
        df: DataFrame with pattern_strength column
        segments: List of ExhibitSegment objects to score
        candidates: List of ExhibitRowCandidate objects (not currently used)
        require_header_in_chain: Disqualify if chain root has no header
        
        Confidence scoring weights:
            links_weight_per_link: Score per linked row (default: 0.5)
            links_score_cap: Maximum score from links (default: 2.0)
            consecutive_weight_per_count: Score per consecutive candidate (default: 0.2)
            consecutive_score_cap: Maximum score from consecutive candidates (default: 1.5)
            strong_pattern_weight: Score per strong pattern element (default: 1.0)
            strong_pattern_cap: Maximum score from strong patterns (default: 2.0)
            weak_first_weight: Score for first weak pattern (default: 1.0)
            weak_additional_weight: Score for additional weak patterns (default: 0.1)
            weak_pattern_cap: Maximum score from weak patterns (default: 1.5)
        
        Acceptance threshold:
            min_score_threshold: Minimum score to accept segment (default: 2.0)
        
        Column names:
            line_id_col: Column name for line IDs
            exhibit_row_col: Column name for exhibit row candidates
            pattern_strength_col: Column name for pattern strength
    
    Returns:
        List of ExhibitScore objects (includes both qualified and disqualified segments)
    """
    if not segments:
        return []
    
    # Build segment lookup
    segments_by_id = {seg.segment_id: seg for seg in segments}
    
    scores: List[ExhibitScore] = []
    
    for segment in segments:
        # === DISQUALIFICATION CHECK ===
        is_disqualified = False
        disqualification_reason = None
        
        # Find root segment (follow chain upward)
        root_segment = _find_root_segment(segment, segments_by_id)
        root_has_header = root_segment.has_exhibit_header_nearby
        
        # Check disqualification rules
        if require_header_in_chain:
            # Rule 1: If no header nearby and no segment above -> disqualified
            if not segment.has_exhibit_header_nearby and not segment.has_other_segment_above:
                is_disqualified = True
                disqualification_reason = "No header nearby and not part of a chain"
            
            # Rule 2: If in a chain, root must have header
            elif segment.has_other_segment_above and not root_has_header:
                is_disqualified = True
                disqualification_reason = f"Chain root (segment {root_segment.segment_id}) has no header"
        
        # === CONFIDENCE SCORING ===
        # (Calculate even for disqualified segments for inspection)
        
        # 1. Header nearby
        header_nearby_score = 1.0 if segment.has_exhibit_header_nearby else 0.0
        
        # 2. Exhibit number header
        number_header_score = 1.0 if segment.has_exhibit_number_header else 0.0
        
        # 3. Segment above (continuation)
        segment_above_score = 1.0 if segment.has_other_segment_above else 0.0
        
        # 4. Links score (capped)
        links_score = min(segment.n_links * links_weight_per_link, links_score_cap)
        
        # 5. Consecutive candidates score (capped)
        consecutive_score = min(
            segment.max_consecutive_candidates * consecutive_weight_per_count,
            consecutive_score_cap
        )
        
        # 6. Strong/weak pattern scores
        # Count strong and weak patterns in this segment
        segment_rows = df[
            (df[line_id_col] >= segment.start_line_id) & 
            (df[line_id_col] <= segment.end_line_id) &
            (df[exhibit_row_col].notna())
        ]
        
        n_strong = 0
        n_weak = 0
        if pattern_strength_col in segment_rows.columns:
            n_strong = (segment_rows[pattern_strength_col] == "strong").sum()
            n_weak = (segment_rows[pattern_strength_col] == "weak").sum()
        
        # Strong pattern score (capped)
        strong_pattern_score = min(n_strong * strong_pattern_weight, strong_pattern_cap)
        
        # Weak pattern score (first one worth more, then diminishing)
        if n_weak == 0:
            weak_pattern_score = 0.0
        elif n_weak == 1:
            weak_pattern_score = weak_first_weight
        else:
            # First weak + additional weaks
            weak_pattern_score = weak_first_weight + (n_weak - 1) * weak_additional_weight
        weak_pattern_score = min(weak_pattern_score, weak_pattern_cap)
        
        # Total confidence score
        confidence_score = (
            header_nearby_score +
            number_header_score +
            segment_above_score +
            links_score +
            consecutive_score +
            strong_pattern_score +
            weak_pattern_score
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
            root_segment_id=root_segment.segment_id,
            root_has_header=root_has_header,
            n_strong_patterns=int(n_strong),
            n_weak_patterns=int(n_weak),
        ))
    
    return scores


def filter_accepted_exhibit_segments(
    scores: List[ExhibitScore],
    min_score_threshold: float = 2.0,
) -> List[ExhibitScore]:
    """
    Filter exhibit scores to only accepted segments.
    
    A segment is accepted if:
    - Not disqualified
    - confidence_score >= min_score_threshold
    
    Args:
        scores: List of ExhibitScore objects
        min_score_threshold: Minimum confidence score to accept
    
    Returns:
        List of accepted ExhibitScore objects
    """
    accepted = []
    for score in scores:
        if score.is_disqualified:
            continue
        if score.confidence_score < min_score_threshold:
            continue
        accepted.append(score)
    
    return accepted


# ==========================================
# DEBUG UTILITIES
# ==========================================
# 
# Usage examples:
#
# 1. Enable debug mode in the main pipeline (adds debug columns + prints segments):
#    df = detect_and_mark_exhibits(df, config, debug=True)
#
# 2. Or call the debug function directly after running the legacy API:
#    df, segments, scores = detect_and_annotate_exhibits(df, config, build_segments=True, score_segments=True)
#    print_exhibit_segments(segments, df=df, scores=scores)
#

def print_exhibit_segments(
    segments: List[ExhibitSegment],
    df: pd.DataFrame = None,
    scores: List[ExhibitScore] = None,
    *,
    text_col: str = "text",
    line_id_col: str = "line_id",
    show_text: bool = True,
    max_text_len: int = 80,
) -> None:
    """
    Print a formatted debug view of all exhibit segments.
    
    Args:
        segments: List of ExhibitSegment objects to print
        df: Optional DataFrame with text to show segment content
        scores: Optional list of ExhibitScore objects to show scores
        text_col: Column name for text in df
        line_id_col: Column name for line_id in df
        show_text: Whether to show text content from df
        max_text_len: Maximum length of text to display per line
    """
    if not segments:
        print("No segments found.")
        return
    
    # Create score lookup if scores provided
    score_by_seg_id = {}
    if scores:
        score_by_seg_id = {score.segment.segment_id: score for score in scores}
    
    print(f"\n{'='*100}")
    print(f"EXHIBIT SEGMENTS ({len(segments)} total)")
    print(f"{'='*100}\n")
    
    for i, seg in enumerate(segments, 1):
        # Basic segment info
        print(f"Segment #{i} (ID: {seg.segment_id})")
        print(f"  Lines: {seg.start_line_id} → {seg.end_line_id} ({seg.n_rows} rows)")
        print(f"  Candidates: {seg.n_candidates}/{seg.n_rows} ({seg.candidate_ratio:.1%})")
        print(f"  Max consecutive: {seg.max_consecutive_candidates}")
        print(f"  Links: {seg.n_links}")
        
        # Classification
        if seg.is_table_based:
            print(f"  Type: TABLE (table_id={seg.table_id})")
        elif seg.is_fingerprint_based:
            fp = seg.fingerprint
            print(f"  Type: FINGERPRINT (left={fp.left:.1f}, height={fp.height:.1f}, font={fp.font_size:.1f})")
        else:
            print(f"  Type: UNKNOWN")
        
        # Headers
        if seg.has_exhibit_header_nearby:
            print(f"  ✓ Has exhibit header nearby (lines: {seg.nearby_header_line_ids})")
        if seg.has_exhibit_number_header:
            print(f"  ✓ Has exhibit number header")
        
        # Chain info
        if seg.has_other_segment_above:
            print(f"  ⬆ Connected to segment {seg.above_exhibit_segment_id} above")
        
        # Score info
        if seg.segment_id in score_by_seg_id:
            score = score_by_seg_id[seg.segment_id]
            status = "✗ DISQUALIFIED" if score.is_disqualified else "✓ ACCEPTED"
            print(f"  Score: {score.confidence_score:.2f} {status}")
            if score.is_disqualified:
                print(f"    Reason: {score.disqualification_reason}")
            else:
                components = []
                if score.header_nearby_score > 0:
                    components.append(f"header={score.header_nearby_score:.1f}")
                if score.number_header_score > 0:
                    components.append(f"number_hdr={score.number_header_score:.1f}")
                if score.segment_above_score > 0:
                    components.append(f"above={score.segment_above_score:.1f}")
                if score.links_score > 0:
                    components.append(f"links={score.links_score:.1f}")
                if score.consecutive_score > 0:
                    components.append(f"consec={score.consecutive_score:.1f}")
                if score.strong_pattern_score > 0:
                    components.append(f"strong={score.strong_pattern_score:.1f}")
                if score.weak_pattern_score > 0:
                    components.append(f"weak={score.weak_pattern_score:.1f}")
                if components:
                    print(f"    Components: {', '.join(components)}")
                print(f"    Root segment: {score.root_segment_id}, root_has_header={score.root_has_header}")
                print(f"    Patterns: {score.n_strong_patterns} strong, {score.n_weak_patterns} weak")
        
        # Show text content if available
        if show_text and df is not None and line_id_col in df.columns and text_col in df.columns:
            segment_rows = df[
                (df[line_id_col] >= seg.start_line_id) & 
                (df[line_id_col] <= seg.end_line_id)
            ]
            
            if not segment_rows.empty:
                print(f"  Text content:")
                for _, row in segment_rows.iterrows():
                    line_id = row[line_id_col]
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    # Truncate and clean text
                    text = text.replace("\n", " ").replace("\r", " ")
                    if len(text) > max_text_len:
                        text = text[:max_text_len-3] + "..."
                    print(f"    [{line_id}] {text}")
        
        print()  # Blank line between segments
    
    print(f"{'='*100}\n")


# ==========================================
# PUBLIC API
# ==========================================

def detect_and_mark_exhibits(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    *,
    # Column names
    text_col: str = "text",
    has_link_col: str = "has_link",
    line_id_col: str = "line_id",
    table_id_col: str = "table_id",
    block_role_col: str = "block_role",
    # Segmentation options
    left_tolerance: float = 5.0,
    height_tolerance: float = 2.0,
    font_tolerance: float = 0.5,
    header_lookback: int = 3,
    segment_lookback: int = 5,
    # Scoring options
    min_score_threshold: float = 2.0,
    # Debug options
    debug: bool = False,
) -> pd.DataFrame:
    """
    Complete exhibit detection pipeline.
    
    Pipeline:
    1. Mask rows where block_role = 'toc' or 'toc_header'
    2. Identify header candidates
    3. Identify row candidates
    4. Build segments
    5. Score and filter segments
    6. Annotate winning rows with block_role = 'exhibit' or 'exhibit_header'
    
    Args:
        df: Input DataFrame with text, has_link, line_id columns
        exhibit_config: Compiled exhibit patterns from YAML
        
        Column names:
            text_col: Column name for text (default: "text")
            has_link_col: Column name for link flag (default: "has_link")
            line_id_col: Column name for line IDs (default: "line_id")
            table_id_col: Column name for table IDs (default: "table_id")
            block_role_col: Column name for block_role (default: "block_role")
        
        Segmentation options:
            left_tolerance: Tolerance for fingerprint matching (default: 5.0)
            height_tolerance: Tolerance for fingerprint matching (default: 2.0)
            font_tolerance: Tolerance for fingerprint matching (default: 0.5)
            header_lookback: Rows to look back for headers (default: 3)
            segment_lookback: Rows to look back for other segments (default: 5)
        
        Scoring options:
            min_score_threshold: Minimum score to accept segment (default: 2.0)
        
        Debug options:
            debug: Enable debug mode - adds exhibit_* columns and prints segment info (default: False)
    
    Returns:
        DataFrame with added columns:
        - block_role: 'exhibit' for exhibit entry rows, 'exhibit_header' for header rows
        - (if debug=True):
          - exhibit_header_candidate: Pattern name for header candidates
          - exhibit_row_candidate: Pattern name for row candidates
          - exhibit_number: Extracted exhibit number
          - pattern_strength: "strong" or "weak" indicator
          
        When debug=True, also prints formatted segment information to console.
    """
    # STEP 0: Mask TOC rows (don't process them as exhibits)
    # Create a mask for rows that are NOT toc or toc_header
    if block_role_col in df.columns:
        mask = ~df[block_role_col].isin(["toc", "toc_header"])
        df_to_process = df[mask].copy()
    else:
        df_to_process = df.copy()
    
    if df_to_process.empty:
        # No rows to process - just return original with block_role column
        if block_role_col not in df.columns:
            df[block_role_col] = pd.NA
        return df
    
    # STEP 1: Identify header candidates
    df_to_process = identify_exhibit_header_candidates(
        df_to_process, 
        exhibit_config,
        text_col=text_col
    )
    
    # STEP 2: Identify row candidates
    df_to_process, exhibit_candidates = add_exhibit_row_candidates(
        df_to_process, 
        exhibit_config,
        text_col=text_col,
        include_debug_cols=True,  # Always build these for scoring
    )
    
    # STEP 3: Build segments
    segments = build_exhibit_segments(
        df_to_process,
        exhibit_candidates,
        line_id_col=line_id_col,
        table_id_col=table_id_col,
        text_col=text_col,
        exhibit_header_col="exhibit_header_candidate",
        exhibit_row_col="exhibit_row_candidate",
        left_tolerance=left_tolerance,
        height_tolerance=height_tolerance,
        font_tolerance=font_tolerance,
        header_lookback=header_lookback,
        segment_lookback=segment_lookback,
    )
    
    # STEP 4: Score segments
    scores = score_exhibit_segments(
        df_to_process,
        segments,
        exhibit_candidates,
        min_score_threshold=min_score_threshold,
        line_id_col=line_id_col,
    )
    
    # STEP 5: Filter to accepted segments
    accepted_scores = filter_accepted_exhibit_segments(scores, min_score_threshold)
    
    # Debug: Print segments if requested
    if debug:
        print_exhibit_segments(
            segments,
            df=df_to_process,
            scores=scores,
            text_col=text_col,
            line_id_col=line_id_col,
            show_text=True,
        )
    
    # STEP 6: Annotate winning segments
    # Initialize block_role column if it doesn't exist
    if block_role_col not in df.columns:
        df[block_role_col] = pd.NA
    
    # Mark all rows in accepted segments
    for score_obj in accepted_scores:
        seg = score_obj.segment
        
        # Mark header rows that are associated with this segment
        for header_line_id in seg.nearby_header_line_ids:
            header_mask = df[line_id_col] == header_line_id
            df.loc[header_mask, block_role_col] = "exhibit_header"
        
        # Mark exhibit content rows (from start to end of segment)
        mask = (df[line_id_col] >= seg.start_line_id) & (df[line_id_col] <= seg.end_line_id)
        df.loc[mask, block_role_col] = "exhibit"
    
    # STEP 7: For chain continuations, mark from header to last segment
    # Build segment chains and mark from root header to final segment end
    if accepted_scores:
        # Group segments by their root
        segments_by_root: Dict[int, List[ExhibitSegment]] = {}
        for score_obj in accepted_scores:
            root_id = score_obj.root_segment_id
            if root_id not in segments_by_root:
                segments_by_root[root_id] = []
            segments_by_root[root_id].append(score_obj.segment)
        
        # For each chain, mark from header to last segment
        for root_id, chain_segments in segments_by_root.items():
            if not chain_segments:
                continue
            
            # Sort segments by start_line_id
            chain_segments = sorted(chain_segments, key=lambda s: s.start_line_id)
            
            # Find the root segment (should be the first one)
            root_segment = chain_segments[0]
            last_segment = chain_segments[-1]
            
            # If root has header, mark from header to last segment end
            if root_segment.has_exhibit_header_nearby and root_segment.nearby_header_line_ids:
                # Find earliest header line_id
                earliest_header = min(root_segment.nearby_header_line_ids)
                
                # Mark all rows from earliest header to last segment end as exhibit/exhibit_header
                for line_id in range(earliest_header, last_segment.end_line_id + 1):
                    line_mask = df[line_id_col] == line_id
                    if line_mask.any():
                        # Check if it's a header line
                        if line_id in root_segment.nearby_header_line_ids:
                            df.loc[line_mask, block_role_col] = "exhibit_header"
                        else:
                            # Only mark as exhibit if not already marked as something else important
                            current_role = df.loc[line_mask, block_role_col].iloc[0] if line_mask.any() else None
                            if pd.isna(current_role) or current_role in ["exhibit", "exhibit_header"]:
                                df.loc[line_mask, block_role_col] = "exhibit"
    
    # STEP 8: Merge debug columns from df_to_process back to df if requested
    if debug:
        debug_cols = [
            "exhibit_header_candidate",
            "exhibit_row_candidate",
            "exhibit_number",
            "pattern_strength",
        ]
        # Create a mapping from line_id to debug column values
        if any(col in df_to_process.columns for col in debug_cols):
            # Select only line_id and debug columns that exist
            merge_cols = [line_id_col] + [col for col in debug_cols if col in df_to_process.columns]
            debug_df = df_to_process[merge_cols].copy()
            
            # Merge debug columns into df based on line_id
            df = df.merge(debug_df, on=line_id_col, how="left", suffixes=("", "_new"))
            
            # If there are any duplicate columns (shouldn't be), use the new values
            for col in debug_cols:
                if f"{col}_new" in df.columns:
                    df[col] = df[f"{col}_new"]
                    df = df.drop(columns=[f"{col}_new"])
    
    return df




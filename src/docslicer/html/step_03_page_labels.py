# step_03_page_labels.py
"""
Page Label Detection and Sequence Identification

This module detects and assigns page labels in SEC EDGAR HTML documents.
Page labels are typically small text elements (numbers, roman numerals, etc.)
that appear at consistent positions across pages.

Public API:
    assign_page_labels(): Main entry point for page label detection

Pipeline Overview:
    1. Token Extraction: Identify potential page label tokens from raw text
    2. Token Filtering: Exclude TOC entries, XBRL content, tables, crowded bands
    3. Candidate Building: Create PageLabelCandidate objects with formatting info
    4. Sequence Building: Group candidates into sequences with consistent formatting
    5. Winner Selection: Choose non-overlapping sequences using greedy scoring
    6. Group Conversion: Convert to final PageLabel and PageLabelGroup objects

Key Concepts:
    - FormattingSignature: Identifies sequences by height, font, tag, wrappers
    - AlternationMode: "fixed" (same position) vs "alternating" (left/right pattern)
    - Sequence Scoring: length^1.5 scoring favors longer sequences
    - Alignment Tracking: Enforces consistent alignment within sequences

Author: MarketFramer
"""
from __future__ import annotations

import re
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, Set, Union

import pandas as pd

# =========================
# Constants
# =========================

TOP_TOL_PX = 4 # px tolerance for grouping rows into the same "top" band

# =========================
# Core Dataclasses
# =========================

# -------- Stage 1: Detection ------- #

@dataclass
class FormattingSignature:
    """Properties that define a page label series"""
    height: float
    font_size: float
    font_weight: str  # Add this - important for differentiation
    structure_tag: str  # Add this - 'p' vs 'td' matters
    has_table_id: bool
    has_dash_wrapper: bool
    has_paren_wrapper: bool
    
    def __hash__(self):
        return hash((self.height, self.font_size, self.font_weight, self.structure_tag, self.has_table_id, 
                     self.has_dash_wrapper, self.has_paren_wrapper))

@dataclass
class CandidateSequence:
    """A contiguous run of candidates that form a potential sequence"""
    sequence_id: int
    page_label_type: PageLabelType
    formatting_signature: FormattingSignature
    text_align: str
    alternation_mode: AlternationMode
    start_box_id: int  # First box_id in sequence
    end_box_id: int    # Last box_id in sequence
    length: int # Number of actual labels in the sequence
    box_ids: List[int] = None  # All box_ids that are part of this sequence
    score: Optional[float] = None # Score of the sequence -- only applicable for overlapping sequences

@dataclass
class PageLabelCandidate:
    """All info needed for detection algorithm"""
    box_id: int
    raw_token: str
    normalized_token: str
    page_label_type: PageLabelType
    # Formatting properties
    height: float
    font_size: float
    font_weight: str
    structure_tag: str
    text_align: str
    # Table context
    has_table_id: bool
    # Wrapper detection
    has_dash_wrapper: bool
    has_paren_wrapper: bool
    
    def get_signature(self) -> FormattingSignature:
        return FormattingSignature(self.height, self.font_size, 
                                   self.font_weight, self.structure_tag, self.has_table_id,
                                   self.has_dash_wrapper, self.has_paren_wrapper)

# -------- Stage 2: Validated Output ------- #

@dataclass
class PageLabel:
    """Confirmed page label after validation"""
    box_id: int
    raw_token: Optional[str]
    normalized_token: Optional[str]
    corrected_token: Optional[str]  # if fixed during validation
    page_label_type: PageLabelType
    detection_method: DetectionMethod
    # Reference back to group (set after group creation)
    group_id: Optional[int] = None


@dataclass
class PageLabelGroup:
    """A sequence of validated page labels with consistent formatting"""
    group_id: int
    page_label_type: PageLabelType
    position: PageLabelGroupPosition
    alternation_mode: AlternationMode
    start_token: str  # "i" or "1" or "F-1"
    end_token: str    # "ix" or "130" or "F-7"
    page_labels: List[PageLabel]  # The actual labels in this group
    formatting_signature: FormattingSignature

# -------- Enums ------- #

PageLabelType = Literal["arabic", "roman", "alpha_numeric", "alpha_roman", "roman_numeric", "unknown"]

PageLabelGroupPosition = Literal["left", "center", "right", "left_right", "unknown"]

AlternationMode = Literal[
    "fixed",        # same position throughout
    "alternating",  # e.g., left/right alternating by page
    "mixed",        # unstable
    "unknown",
]

DetectionMethod = Literal[
    "standard",              # inferred from global candidates + monotonic series
    "manual_fix",            # patched during validation (e.g., 55,56,C,58)
    "gap_fill_from_above",   # filled missing value by incrementing previous
    "gap_fill_from_below",   # filled missing value by copying/deriving from next
]




# =========================
# PUBLIC API
# =========================

def assign_page_labels(
    df: pd.DataFrame, 
    page_label_config, 
    tol_px: float = TOP_TOL_PX
) -> Tuple[pd.DataFrame, List[PageLabel], List[PageLabelGroup]]:
    """
    Detect and assign page labels to document boxes.
    
    Main entry point for page label detection. Runs full pipeline to identify
    and validate page label sequences.

    Pipeline:
    1. Extract candidate tokens from text
    2. Filter tokens (exclude TOC, XBRL, tables, crowded bands, etc.)
    3. Build candidate sequences (group by formatting)
    4. Select winning non-overlapping sequences
    5. Convert to validated PageLabel and PageLabelGroup objects
    6. Add page_label columns to DataFrame
    7. Post-processing:
       - Add block_role column ("page_label" for labeled rows)
       - Propagate page_label upwards (first label until hr, others until prior label)
       - Infer page_no if all rows have page_no == 1 (increment on label changes)

    Required columns in df:
      - box_id, y_top, page_number, text
      - structure_tag, height, font_size, font_weight, text_align
      
    Optional columns:
      - has_link (improves TOC detection if present)
      - ixbrl_id (excludes XBRL-tagged content if present)
      - table_id (excludes large tables if present)
    
    Args:
        df: DataFrame with box metadata
        page_label_config: Compiled page label pattern configuration
        tol_px: Tolerance in pixels for top-band grouping (default: 4)

    Returns:
        Tuple of (df, page_labels, page_label_groups) where:
        - df: DataFrame with added columns:
              - page_label_token: all candidate tokens (for debugging)
              - page_label: validated page labels (propagated upwards)
              - page_label_group_id: which group each label belongs to
              - alternation_mode: fixed/alternating pattern
              - block_role: "page_label" for labeled rows, None for others
              - page_no: inferred page numbers (if originally all 1s)
        - page_labels: List of detected PageLabel objects
        - page_label_groups: List of PageLabelGroup objects
    """
    required = {"box_id", "y_top", "page_number", "text", 
                "structure_tag", "height", "font_size", "font_weight", "text_align"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    out = df.copy()

    # Step 1: Extract all candidate tokens
    inv = _build_page_label_token_inventory(out, page_label_config)

    # Step 2: Apply filters (page-aware)
    cand = _filter_page_label_tokens(out, inv, tol_px=tol_px)

    out["page_label_token"] = cand
    
    # Add wrapper information temporarily (needed for candidate building)
    out["_has_dash_wrapper"] = inv["has_dash_wrapper"]
    out["_has_paren_wrapper"] = inv["has_paren_wrapper"]
    
    # Step 3-5: Run full sequence detection pipeline
    page_labels, groups = _detect_page_label_sequence(out, page_label_config)
    
    # Step 6: Add page_label columns
    out["page_label"] = None
    out["page_label_group_id"] = None
    out["alternation_mode"] = None
    out["page_label_type"] = None
    
    # Build lookup: group_id -> alternation_mode
    group_alternation = {g.group_id: g.alternation_mode for g in groups}
    
    # Map page labels to rows
    for label in page_labels:
        mask = out["box_id"] == label.box_id
        if mask.any():
            out.loc[mask, "page_label"] = label.normalized_token
            out.loc[mask, "page_label_group_id"] = label.group_id
            out.loc[mask, "alternation_mode"] = group_alternation.get(label.group_id, "unknown")
            out.loc[mask, "page_label_type"] = label.page_label_type
    
    # Drop temporary wrapper columns
    out = out.drop(columns=["_has_dash_wrapper", "_has_paren_wrapper", "page_label_token", "page_label_group_id", "alternation_mode"])
    
    # ===== POST-PROCESSING STEPS ===== #
    
    # Step 1: Add block_role column (only if it doesn't exist, then update page label rows)
    if "block_role" not in out.columns:
        out["block_role"] = None
    # Only update rows with page labels, preserve existing values
    out.loc[out["page_label"].notna(), "block_role"] = "page_label"
    
    # Step 2: Fill page_label upwards with propagation rules
    out = _propagate_page_labels_upward(out)
    
    # Step 3: Populate page_no intelligently
    out = _infer_page_numbers(out)
    
    return out, page_labels, groups


# ---------------------------------------------------------------------------- PART 1: Extraction ---------------------------------------------------------------------------- #

# =========================
# Helpers to remove rows with links
# =========================

def _top_band_mask(df, target_y_top: float, target_page_number: int, tol_px: float = TOP_TOL_PX):
    """
    Boolean mask for rows whose `y_top` lies within ± tol_px of target_y_top
    AND are on the same page_number.
    
    Args:
        df: DataFrame with columns 'y_top' and 'page_number'
        target_y_top: Target y_top coordinate
        target_page_number: Target page number
        tol_px: Tolerance in pixels for grouping rows into same "top" band
    
    Returns:
        Boolean Series mask
    """
    top_match = (df["y_top"] >= target_y_top - tol_px) & (df["y_top"] <= target_y_top + tol_px)
    page_match = df["page_number"] == target_page_number
    return top_match & page_match


# Exclude all box_id's in top bands that has_link = 1
def _get_box_ids_in_linked_top_bands(df, tol_px: float = TOP_TOL_PX) -> Set[int]:
    """
    Returns all box_id's that belong to any top-band which contains
    at least one row with has_link = 1.
    
    Considers page_number to avoid cross-page contamination in paginated files.

    Example:
      page_number=1, y_top=100, has_link=1
      page_number=1, y_top=101, has_link=0
    → both box_ids from page 1 are returned (same page, same band).
    
      page_number=2, top=100, has_link=0
    → this box_id is NOT returned (different page).
    
    Args:
        df: DataFrame with columns 'box_id', 'y_top', 'page_number'
        Optional: 'has_link' (if not present, returns empty set)
        tol_px: Tolerance in pixels for grouping rows into same "y_top" band
    
    Returns:
        Set of box_ids to exclude
    """
    required = {"box_id", "y_top", "page_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    
    out: Set[int] = set()
    
    # If has_link column not present, return empty set (no filtering)
    if "has_link" not in df.columns:
        return out

    # rows that explicitly have links (handle both boolean True and integer 1)
    link_rows = df[df["has_link"].isin([1, True])]

    if link_rows.empty:
        return out

    for _, row in link_rows.iterrows():
        band_mask = _top_band_mask(df, row["y_top"], row["page_number"], tol_px)
        out.update(df.loc[band_mask, "box_id"].tolist())

    return out


# =========================
# STEP 1: Build Token Inventory
# =========================

def _match_page_label_type(token: str, cfg) -> str:
    for p in cfg.patterns:
        if p.compiled.match(token):
            return p.name
    return "unknown"


_PAREN_WRAPPER_RE = re.compile(r"^\((.+)\)$")
_DASH_WRAPPER_RE  = re.compile(r"^[-–—]\s*(.+?)\s*[-–—]$")


def _extract_page_label_tokens(text: Any) -> Optional[str]:
    """
    Extract a candidate page label token from text.

    Intentionally conservative:
    - Skip obvious pipeline markers ([[HR]], [[IMAGE:...]])
    - Return the cleaned string (still not normalized)
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    upper = s.upper()
    if upper.startswith("[[HR"):
        return None
    if upper.startswith("[[IMAGE"):
        return None

    return s


def _normalize_page_label_token(token: str) -> str:
    """
    Normalization used for regex matching + series comparisons.
    Keep this deterministic and minimal; expand iteratively.
    """
    s = token.strip()

    # normalize unicode dashes to ASCII hyphen
    s = s.replace("–", "-").replace("—", "-")

    # unwrap (ii) -> ii
    m = _PAREN_WRAPPER_RE.match(s)
    if m:
        s = m.group(1).strip()

    # unwrap "- 2 -" -> "2" (also handles "—2—" after dash normalization)
    m = _DASH_WRAPPER_RE.match(s)
    if m:
        s = m.group(1).strip()

    # collapse whitespace around hyphen (e.g. "II - 3" -> "II-3")
    s = re.sub(r"\s*-\s*", "-", s)

    # remove trailing punctuation like "." ":" ";" "," (common in headings)
    s = re.sub(r"[.:;,]+$", "", s).strip()

    # collapse any internal whitespace
    s = re.sub(r"\s+", " ", s)

    return s


def _has_dash_wrapper(token: str) -> bool:
    """
    Check if token has dash wrapper like "- 2 -" or "—ii—".
    
    Args:
        token: Raw token string
        
    Returns:
        True if token is wrapped in dashes, False otherwise
    """
    if not token:
        return False
    s = token.strip()
    # Normalize unicode dashes to ASCII hyphen first
    s = s.replace("–", "-").replace("—", "-")
    return _DASH_WRAPPER_RE.match(s) is not None


def _has_paren_wrapper(token: str) -> bool:
    """
    Check if token has parenthesis wrapper like "(ii)" or "(2)".
    
    Args:
        token: Raw token string
        
    Returns:
        True if token is wrapped in parentheses, False otherwise
    """
    if not token:
        return False
    return _PAREN_WRAPPER_RE.match(token.strip()) is not None


def _build_page_label_token_inventory(df: pd.DataFrame, page_label_config) -> pd.DataFrame:
    """
    Build a token inventory used by both HR-anchored and no-HR candidate selection.

    Returns a small DataFrame indexed the same as df, with:
      - raw_token
      - normalized_token
      - page_label_type
      - is_token_like
      - has_dash_wrapper
      - has_paren_wrapper

    Requires df columns:
      - text
    Optional but commonly present:
      - box_id, y_top, x_left, structure_tag, has_link (not required here)
    
    Args:
        df: DataFrame with text column
        page_label_config: Compiled PageLabelPatternConfig from load_and_compile_patterns()
    """
    if "text" not in df.columns:
        raise ValueError("build_page_label_token_inventory: df must contain column 'text'")

    max_len = int(getattr(page_label_config, "max_length", 8))

    raw_tokens = []
    norm_tokens = []
    types = []
    is_like = []
    has_dash_wrappers = []
    has_paren_wrappers = []

    # iterate once; keep logic explicit for debugging
    for text in df["text"].tolist():
        raw_tok = _extract_page_label_tokens(text)
        if raw_tok is None:
            raw_tokens.append(None)
            norm_tokens.append(None)
            types.append("unknown")
            is_like.append(False)
            has_dash_wrappers.append(None)
            has_paren_wrappers.append(None)
            continue

        norm = _normalize_page_label_token(raw_tok)

        # Detect wrappers from raw token
        has_dash = _has_dash_wrapper(raw_tok)
        has_paren = _has_paren_wrapper(raw_tok)

        # length gate AFTER normalization
        if not norm or len(norm) > max_len:
            raw_tokens.append(raw_tok)
            norm_tokens.append(norm if norm else None)
            types.append("unknown")
            is_like.append(False)
            has_dash_wrappers.append(has_dash)
            has_paren_wrappers.append(has_paren)
            continue

        t = _match_page_label_type(norm, page_label_config)
        ok = (t != "unknown")

        raw_tokens.append(raw_tok)
        norm_tokens.append(norm)
        types.append(t)
        is_like.append(ok)
        has_dash_wrappers.append(has_dash)
        has_paren_wrappers.append(has_paren)

    out = pd.DataFrame(
        {
            "raw_token": raw_tokens,
            "normalized_token": norm_tokens,
            "page_label_type": types,
            "is_token_like": is_like,
            "has_dash_wrapper": has_dash_wrappers,
            "has_paren_wrapper": has_paren_wrappers,
        },
        index=df.index,
    )

    return out


# =========================
# STEP 2: Filter Token Inventory
# =========================

def _filter_page_label_tokens(
    df: pd.DataFrame,
    token_inventory: pd.DataFrame,
    tol_px: float = TOP_TOL_PX
) -> pd.Series:
    """
    Filter token inventory to valid page label tokens.

    Returns a Series (indexed like df) with normalized_token values,
    blanked to None when:
      - token does not match any page label pattern (is_token_like=False)
      - token is a single letter c, d, l, or m (common list markers, not page labels)
      - row's box_id belongs to any top-band that has has_link=1 (TOC-ish)
      - row has an ixbrl_id (XBRL-tagged financial data, not page labels)
      - token is a pure negative number like -1, -2, -3 (financial table data)
      - row belongs to a table with more than 3 rows (complex data tables)
      - row belongs to a top band with more than 2 items (likely headers or content)
      - row belongs to a top band with total character length > 50 (likely text content)

    Requirements in df:
      - box_id, y_top, page_number, structure_tag
    Optional:
      - has_link (if present, used to filter TOC entries)
      - ixbrl_id (if present, rows with non-blank ixbrl_id are excluded)
      - table_id (if present, used to filter large tables)
      - text (if present, used to filter bands with long text)

    Args:
        df: DataFrame with box metadata
        token_inventory: DataFrame from extract_token_inventory() with columns:
                         raw_token, normalized_token, page_label_type, is_token_like
        tol_px: Tolerance in pixels for grouping rows into same "top" band

    Returns:
        Series of normalized tokens (or None) indexed like df
    """
    required = {"box_id", "y_top", "page_number", "structure_tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    # Start with token-like candidates
    cand = token_inventory["normalized_token"].where(token_inventory["is_token_like"], None)

    # Drop parenthesized integers: (1), (2), (10), etc.
    raw = token_inventory["raw_token"].astype("string")

    is_paren_integer = (
        raw.fillna("")
        .str.strip()
        .str.match(r"^\(\s*\d+\s*\)$")
    )

    cand = cand.where(~is_paren_integer, None)


    # Exclude single-letter list markers: (c), (d), (l), (m)
    # After normalization these become: c, d, l, m
    # These are valid roman numeral chars but almost never used as single-char page numbers
    excluded_single_letters = {"c", "d", "l", "m", "C", "D", "L", "M"}
    is_excluded_letter = cand.isin(excluded_single_letters)
    cand = cand.where(~is_excluded_letter, None)

    # Exclude rows in link bands (TOC entries) - now page-aware
    # Only apply if has_link column is present
    if "has_link" in df.columns:
        linked_band_box_ids = _get_box_ids_in_linked_top_bands(df, tol_px=tol_px)
        in_link_band = df["box_id"].isin(linked_band_box_ids)
        cand = cand.where(~in_link_band, None)

    # Exclude rows with ixbrl_id (XBRL-tagged content)
    if "ixbrl_id" in df.columns:
        has_ixbrl = df["ixbrl_id"].notna() & (df["ixbrl_id"] != "")
        cand = cand.where(~has_ixbrl, None)
    
    # Exclude pure negative numbers like -1, -2, -3 (but not wrapped ones like -2-)
    # These are common in financial tables as negative values
    def is_pure_negative(token):
        """Check if token is a pure negative number like '-1', '-2' (not '-1-')"""
        if pd.isna(token):
            return False
        s = str(token)
        if s.startswith("-") and not s.endswith("-") and len(s) > 1:
            # Check if everything after the '-' is digits
            return s[1:].isdigit()
        return False
    
    is_negative = cand.apply(is_pure_negative)
    cand = cand.where(~is_negative, None)
    
    # Exclude rows from tables that are "large enough" to be real data tables:
    # condition: table_id is non-blank AND table has >= 3 box_ids
    if "table_id" in df.columns:
        table_id = df["table_id"].astype("string")
        table_id_nonblank = table_id.notna() & (table_id.str.strip() != "")

        # count box_ids per table (only for non-blank table_ids)
        table_box_counts = (
            df.loc[table_id_nonblank]
              .groupby(table_id.loc[table_id_nonblank])["box_id"]
              .size()
        )

        tables_ge3 = set(table_box_counts[table_box_counts >= 3].index)

        in_table_ge3 = table_id_nonblank & table_id.isin(tables_ge3)
        cand = cand.where(~in_table_ge3, None)
    
    # Exclude rows where the same top band has more than 2 items
    # Create top band groups (page_number, rounded_top)
    df_with_rounded_top = df.copy()
    df_with_rounded_top['rounded_top'] = (df['y_top'] / tol_px).round() * tol_px
    
    # Count items per top band (page_number, rounded_top)
    band_counts = df_with_rounded_top.groupby(['page_number', 'rounded_top']).size()
    
    # Identify bands with more than 2 items
    crowded_bands = band_counts[band_counts > 2].index
    
    # Mark rows that belong to crowded bands
    in_crowded_band = df_with_rounded_top.apply(
        lambda row: (row['page_number'], row['rounded_top']) in crowded_bands, 
        axis=1
    )
    cand = cand.where(~in_crowded_band, None)
    
    # Exclude rows where the same top band has total character length > 50
    # Need text column for this
    if "text" in df.columns:
        # Calculate total character length per band
        df_with_text_len = df_with_rounded_top.copy()
        df_with_text_len['text_len'] = df['text'].fillna("").astype(str).str.len()
        
        band_char_totals = df_with_text_len.groupby(['page_number', 'rounded_top'])['text_len'].sum()
        
        # Identify bands with total char length > 50
        long_text_bands = band_char_totals[band_char_totals > 50].index
        
        # Mark rows that belong to long text bands
        in_long_text_band = df_with_rounded_top.apply(
            lambda row: (row['page_number'], row['rounded_top']) in long_text_bands,
            axis=1
        )
        cand = cand.where(~in_long_text_band, None)

    return cand


# ---------------------------------------------------------------------------- PART 2: Main Logic ---------------------------------------------------------------------------- #

# =========================
# STEP 3: Candidate Building
# =========================

def _build_candidates_from_df(df: pd.DataFrame, page_label_config) -> List[PageLabelCandidate]:
    """
    Build a list of PageLabelCandidate objects from a DataFrame with page_label_token column.
    
    Requirements in df:
      - box_id, page_label_token, structure_tag, height, font_size, font_weight, text_align
      - _has_dash_wrapper, _has_paren_wrapper (temporary wrapper columns)
    Optional:
      - table_id (for has_table_id detection)
      
    Args:
        df: DataFrame from assign_page_labels() with page_label_token column
        page_label_config: Compiled PageLabelPatternConfig (not used here but kept for consistency)
        
    Returns:
        List of PageLabelCandidate objects (only for rows with valid page_label_token)
    """
    required = {"box_id", "page_label_token", "structure_tag", "height", "font_size", "font_weight", "text_align",
                "_has_dash_wrapper", "_has_paren_wrapper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    
    # Filter to only rows with valid page_label_token
    valid_df = df[df["page_label_token"].notna()].copy()
    
    candidates = []
    for _, row in valid_df.iterrows():
        token = row["page_label_token"]
        
        # Determine page_label_type
        label_type = row.get("page_label_type", "unknown")
        if label_type == "unknown" or pd.isna(label_type):
            # Try to infer from token
            label_type = _match_page_label_type(token, page_label_config)
        
        # Check if table_id is actually present (not None, not NaN, not empty string)
        table_id_val = row.get("table_id")
        has_table_id = pd.notna(table_id_val) and str(table_id_val).strip() != ""
        
        # Handle wrapper flags - if None (token was never extracted), default to False
        dash_wrapper = row["_has_dash_wrapper"]
        has_dash_wrapper = bool(dash_wrapper) if pd.notna(dash_wrapper) else False
        
        paren_wrapper = row["_has_paren_wrapper"]
        has_paren_wrapper = bool(paren_wrapper) if pd.notna(paren_wrapper) else False
        
        candidate = PageLabelCandidate(
            box_id=int(row["box_id"]),
            raw_token=token,  # In this case raw_token = normalized_token
            normalized_token=token,
            page_label_type=label_type,
            height=float(row["height"]),
            font_size=float(str(row["font_size"]).replace("px", "")),
            font_weight=str(row["font_weight"]),
            structure_tag=str(row["structure_tag"]),
            text_align=str(row["text_align"]),
            has_table_id=has_table_id,
            has_dash_wrapper=has_dash_wrapper,
            has_paren_wrapper=has_paren_wrapper
        )
        candidates.append(candidate)
    
    return candidates


# ================================
# STEP 4: Sequence Building
# ================================

# ----- Page Label Value Helpers ----- #

_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> Optional[int]:
    """
    Convert roman numeral string to integer using standard algorithm.
    Supports unlimited range (I to MMMM...).
    Returns None if not a valid roman numeral.
    
    Examples:
        i, ii, iii → 1, 2, 3
        iv, v, vi → 4, 5, 6
        ix, x, xi → 9, 10, 11
        xlix, l, li → 49, 50, 51
        xcix, c, ci → 99, 100, 101
    """
    if not s:
        return None
    t = re.sub(r"[^IVXLCDM]", "", str(s).upper())
    if not t:
        return None

    total = 0
    prev = 0
    for ch in reversed(t):
        v = _ROMAN_MAP.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    if total <= 0:
        return None
    return total


def _extract_custom_label_parts(token: str) -> Optional[Tuple[str, int]]:
    """
    Extract prefix and numeric suffix from custom labels.
    
    Handles:
    - alpha_numeric: F-1, A-23, EX-99 → (prefix, number)
    - alpha_roman: S-iv, F-ii, A-vii → (prefix, roman_as_int)
    - roman_numeric: II-3, IV-12, III-10 → (roman_as_int, number)
    
    Returns (prefix_or_int, number) or None if not a valid custom label.
    
    Note: Order matters! Check roman patterns before alpha patterns since
    roman numerals also match [A-Z]+.
    """
    upper_token = token.upper()
    
    # Try roman-numeric FIRST: II-3, IV-12 → convert roman prefix to int
    # Must come before alpha-numeric since roman chars match [A-Z]+
    match = re.match(r'^([IVXLCDM]+)-(\d+)$', upper_token)
    if match:
        roman_val = _roman_to_int(match.group(1))
        if roman_val:
            return (roman_val, int(match.group(2)))
    
    # Try alpha-roman: S-iv, F-ii → convert roman to int
    match = re.match(r'^([A-Z]+)-([IVXLCDM]+)$', upper_token)
    if match:
        roman_val = _roman_to_int(match.group(2))
        if roman_val:
            return (match.group(1), roman_val)
    
    # Try alpha-numeric: F-1, A-23
    # Comes last since it's most general
    match = re.match(r'^([A-Z]+)-(\d+)$', upper_token)
    if match:
        return (match.group(1), int(match.group(2)))
    
    return None


def _extract_page_label_value(token: str, label_type: PageLabelType) -> Union[int, str, Tuple, None]:
    """
    Extract the comparable value from a page label token.
    
    Returns:
        - For arabic: int
        - For roman: str (original token, will be converted during comparison)
        - For custom types: tuple of (prefix, number)
        - None if extraction fails
    """
    if label_type == "arabic":
        try:
            return int(token)
        except (ValueError, TypeError):
            return None
            
    elif label_type == "roman":
        # Return as-is, will be converted to int during comparison
        return token
        
    elif label_type in ["alpha_numeric", "alpha_roman", "roman_numeric"]:
        parts = _extract_custom_label_parts(token)
        return parts
        
    return None


def _compare_page_label_values(val1: Union[int, str], val2: Union[int, str], label_type: PageLabelType) -> int:
    """
    Compare two page label values for ordering.
    
    Returns:
        -1 if val1 < val2 (val1 comes before val2)
         0 if val1 == val2 (same value)
         1 if val1 > val2 (val1 comes after val2)
        None if comparison not possible
        
    Examples:
        _compare_page_label_values(1, 2, "arabic") -> -1
        _compare_page_label_values("ii", "v", "roman") -> -1
        _compare_page_label_values(("S", 5), ("S", 3), "alpha_numeric") -> 1
    """
    if label_type == "arabic":
        try:
            return -1 if int(val1) < int(val2) else (0 if int(val1) == int(val2) else 1)
        except (ValueError, TypeError):
            return None
            
    elif label_type == "roman":
        r1 = _roman_to_int(str(val1))
        r2 = _roman_to_int(str(val2))
        if r1 is None or r2 is None:
            return None
        return -1 if r1 < r2 else (0 if r1 == r2 else 1)
        
    elif label_type in ["alpha_numeric", "alpha_roman", "roman_numeric"]:
        # These are tuples (prefix, number)
        if not isinstance(val1, tuple) or not isinstance(val2, tuple):
            return None
        prefix1, num1 = val1
        prefix2, num2 = val2
        
        # Prefixes must match for comparison
        if prefix1 != prefix2:
            return None
            
        # Compare numeric parts
        return -1 if num1 < num2 else (0 if num1 == num2 else 1)
        
    return None


# ----- Page Label Alignment Helpers ----- #

def _detect_alignment_mode_from_list(text_aligns: List[str]) -> AlternationMode:
    """
    Detect alignment mode from a list of text_align values.
    
    Args:
        text_aligns: List of text_align values in sequence order
        
    Returns:
        AlternationMode: "fixed", "alternating", "mixed", or "unknown"
    """
    if len(text_aligns) < 2:
        return "fixed"
    
    unique_aligns = set(text_aligns)
    
    if len(unique_aligns) == 1:
        return "fixed"
    
    # Check for alternating pattern
    if len(unique_aligns) == 2:
        alternates = True
        for i in range(len(text_aligns) - 1):
            if text_aligns[i] == text_aligns[i + 1]:
                alternates = False
                break
        
        if alternates:
            return "alternating"
    
    return "mixed"


def _is_alignment_compatible(candidate_align: str, seq_info: Dict) -> bool:
    """
    Check if a candidate's alignment is compatible with an existing sequence.
    
    Args:
        candidate_align: The text_align value of the candidate
        seq_info: Sequence info dictionary with text_aligns, alignment_mode, expected_next_align
        
    Returns:
        True if compatible, False otherwise
    """
    text_aligns = seq_info.get('text_aligns', [])
    
    # If sequence has less than 2 items, we haven't detected the mode yet
    # Allow any alignment (mode will be detected after adding)
    if len(text_aligns) < 1:
        return True
    
    alignment_mode = seq_info.get('alignment_mode')
    
    # If mode not detected yet (length == 1), allow any alignment
    if alignment_mode is None:
        return True
    
    # For fixed mode: must match the first alignment
    if alignment_mode == "fixed":
        return candidate_align == text_aligns[0]
    
    # For alternating mode: must match the expected next alignment
    if alignment_mode == "alternating":
        expected_next = seq_info.get('expected_next_align')
        if expected_next is None:
            return True
        return candidate_align == expected_next
    
    # For mixed or unknown: be permissive (shouldn't happen in practice)
    return True


def _get_alignment_key(text_aligns: List[str], alignment_mode: Optional[str]) -> str:
    """
    Get the alignment key to use in sequence key.
    
    Args:
        text_aligns: List of text_align values
        alignment_mode: The detected alignment mode (or None if not yet detected)
        
    Returns:
        String key representing the alignment pattern
    """
    if not text_aligns:
        return "unknown"
    
    if len(text_aligns) == 1:
        # Single item - use its alignment
        return text_aligns[0]
    
    if alignment_mode == "alternating":
        # Alternating - use sorted combination
        unique_aligns = sorted(set(text_aligns))
        return "_".join(unique_aligns)  # e.g., "left_right"
    
    # Fixed or unknown - use the first alignment
    return text_aligns[0]


# ----- Sequence Tracking ----- #

def _create_sequence_info(sequence_id: int, candidate: PageLabelCandidate, token_value, prefix=None) -> Dict:
    """
    Create a new sequence info dictionary.
    
    Args:
        sequence_id: Unique sequence identifier
        candidate: PageLabelCandidate to start the sequence
        token_value: Extracted value from the token
        prefix: Optional prefix for custom labels
        
    Returns:
        Dictionary with sequence tracking information
    """
    return {
        'sequence_id': sequence_id,
        'page_label_type': candidate.page_label_type,
        'formatting_signature': candidate.get_signature(),
        'start_box_id': candidate.box_id,
        'end_box_id': candidate.box_id,
        'length': 1,
        'last_value': token_value,
        'box_ids': [candidate.box_id],
        'text_aligns': [candidate.text_align],
        'alignment_mode': None,
        'expected_next_align': None,
        'alignment_key': candidate.text_align  # Initial key based on first item
    }


def _extend_sequence_info(seq_info: Dict, candidate: PageLabelCandidate, token_value):
    """
    Extend an existing sequence with a new candidate.
    Updates alignment tracking and detects alternation mode when length reaches 2.
    
    Args:
        seq_info: Sequence info dictionary to extend
        candidate: PageLabelCandidate to add
        token_value: Extracted value from the token
    """
    seq_info['end_box_id'] = candidate.box_id
    seq_info['length'] += 1
    seq_info['last_value'] = token_value
    seq_info['box_ids'].append(candidate.box_id)
    seq_info['text_aligns'].append(candidate.text_align)
    
    # Detect alignment mode once we have 2+ items
    old_mode = seq_info.get('alignment_mode')
    if seq_info['length'] >= 2 and old_mode is None:
        seq_info['alignment_mode'] = _detect_alignment_mode_from_list(seq_info['text_aligns'])
        
        # Update alignment_key if mode changed to alternating
        if seq_info['alignment_mode'] == 'alternating':
            seq_info['alignment_key'] = _get_alignment_key(seq_info['text_aligns'], 'alternating')
    
    # Update expected next alignment for alternating sequences
    if seq_info['alignment_mode'] == 'alternating':
        last_align = candidate.text_align
        unique_aligns = list(set(seq_info['text_aligns']))
        if len(unique_aligns) == 2:
            # Expect the opposite of the last one
            seq_info['expected_next_align'] = unique_aligns[0] if last_align == unique_aligns[1] else unique_aligns[1]


def _get_sequence_text_align(seq_info: Dict) -> str:
    """
    Get the text_align value for a CandidateSequence from sequence info.
    
    Args:
        seq_info: Sequence info dictionary
        
    Returns:
        Text align string for the sequence
    """
    text_aligns = seq_info.get('text_aligns', [])
    alignment_mode = seq_info.get('alignment_mode', 'unknown')
    
    if not text_aligns:
        return "unknown"
    
    if alignment_mode == "fixed":
        # Return the actual alignment
        return text_aligns[0]
    elif alignment_mode == "alternating":
        # Return combined representation
        unique_aligns = sorted(set(text_aligns))
        return "_".join(unique_aligns)  # e.g., "left_right"
    else:
        # Mixed or unknown - return the most common one
        from collections import Counter
        most_common = Counter(text_aligns).most_common(1)
        return most_common[0][0] if most_common else "unknown"


def _seq_info_to_candidate_sequence(seq_info: Dict) -> CandidateSequence:
    """
    Convert sequence info dictionary to CandidateSequence object.
    
    Args:
        seq_info: Sequence info dictionary
        
    Returns:
        CandidateSequence object
    """
    alignment_mode = seq_info.get('alignment_mode', 'unknown')
    if alignment_mode is None:
        alignment_mode = 'fixed'  # Single item sequences are fixed
    
    return CandidateSequence(
        sequence_id=seq_info['sequence_id'],
        page_label_type=seq_info['page_label_type'],
        formatting_signature=seq_info['formatting_signature'],
        text_align=_get_sequence_text_align(seq_info),
        alternation_mode=alignment_mode,
        start_box_id=seq_info['start_box_id'],
        end_box_id=seq_info['end_box_id'],
        length=seq_info['length'],
        box_ids=seq_info['box_ids']
    )


# ----- Main Sequence Building ----- #

def _build_candidate_sequences(candidates: List[PageLabelCandidate]) -> List[CandidateSequence]:
    """
    Build CandidateSequence objects from page label candidates.
    
    Algorithm:
    1. Iterate through candidates by box_id order
    2. For each candidate:
       - Get its sequence key (FormattingSignature + PageLabelType + prefix if applicable)
       - Check if there's an active sequence with this key
       - If yes and value increments AND alignment matches: extend that sequence
       - If yes but value equal/decreases OR alignment mismatch: close that sequence, start new one
       - If no active sequence: check paused sequences
         - If paused sequence exists with this key: resume it
         - If no paused sequence: start a new sequence
       - If key doesn't match any active sequence: pause the active sequence
    3. Once a sequence reaches length >= 2, detect alignment mode (fixed/alternating)
    4. Enforce alignment compatibility for all subsequent additions
    5. Close all sequences at the end
    
    Args:
        candidates: List of PageLabelCandidate objects, should be sorted by box_id
        
    Returns:
        List of CandidateSequence objects
    """
    if not candidates:
        return []
    
    # Sort by box_id to ensure proper ordering
    candidates = sorted(candidates, key=lambda c: c.box_id)
    
    # Track sequences
    completed_sequences: List[CandidateSequence] = []
    active_sequences: Dict[Tuple, Dict] = {}  # (sig, type, prefix) -> {seq_info}
    paused_sequences: Dict[Tuple, List[Dict]] = {}  # (sig, type, prefix) -> [seq_info_list]
    sequence_counter = 0
    
    for candidate in candidates:
        sig = candidate.get_signature()
        label_type = candidate.page_label_type
        token_value = _extract_page_label_value(candidate.normalized_token, label_type)
        
        if token_value is None:
            # Can't process this token
            continue
        
        # Build sequence key: signature + type + prefix (for custom labels) + text_align
        # This ensures:
        # - S-1, S-2 is different from plain 1, 2
        # - S-1, S-2 is different from F-1, F-2
        # - Center-aligned sequences are different from left-aligned sequences
        # - Alternating sequences use "left_right" key
        prefix = None
        if label_type in ["alpha_numeric", "alpha_roman", "roman_numeric"]:
            if isinstance(token_value, tuple):
                prefix = token_value[0]  # Extract prefix
        
        # Try to find an active sequence that could accept this candidate
        # First try exact alignment match
        seq_key = (sig, label_type, prefix, candidate.text_align)
        
        # Also check for alternating sequence key if candidate could be part of alternation
        alternating_key = None
        if candidate.text_align in ["left", "right"]:
            alternating_key = (sig, label_type, prefix, "left_right")
        elif candidate.text_align in ["center", "left"]:
            alternating_key = (sig, label_type, prefix, "center_left")
        elif candidate.text_align in ["center", "right"]:
            alternating_key = (sig, label_type, prefix, "center_right")
        
        # Check if there's an active sequence with this key
        if seq_key in active_sequences:
            active_seq = active_sequences[seq_key]
            
            # Compare values to decide what to do
            last_value = active_seq['last_value']
            comparison = _compare_page_label_values(last_value, token_value, label_type)
            
            if comparison is None:
                # Can't compare - different prefixes or invalid comparison
                # Close current sequence and start new one
                completed_sequences.append(_seq_info_to_candidate_sequence(active_seq))
                
                # Start new sequence
                sequence_counter += 1
                active_sequences[seq_key] = _create_sequence_info(sequence_counter, candidate, token_value, prefix)
                
            elif comparison >= 0:
                # last_value >= token_value (equal or decreased)
                # Close current sequence and start new one
                completed_sequences.append(_seq_info_to_candidate_sequence(active_seq))
                
                # Start new sequence
                sequence_counter += 1
                active_sequences[seq_key] = _create_sequence_info(sequence_counter, candidate, token_value, prefix)
                
            else:
                # last_value < token_value (increasing) - check alignment compatibility
                if not _is_alignment_compatible(candidate.text_align, active_seq):
                    # Alignment doesn't match - close current sequence and start new one
                    completed_sequences.append(_seq_info_to_candidate_sequence(active_seq))
                    
                    # Start new sequence
                    sequence_counter += 1
                    active_sequences[seq_key] = _create_sequence_info(sequence_counter, candidate, token_value, prefix)
                else:
                    # Alignment matches - extend sequence
                    old_key = seq_key
                    _extend_sequence_info(active_seq, candidate, token_value)
                    
                    # Check if alignment_key changed (e.g., from "left" to "left_right" after detecting alternation)
                    new_alignment_key = active_seq.get('alignment_key')
                    new_key = (sig, label_type, prefix, new_alignment_key)
                    
                    if new_key != old_key:
                        # Key changed - update the dictionary
                        del active_sequences[old_key]
                        active_sequences[new_key] = active_seq
                
        elif alternating_key and alternating_key in active_sequences:
            # Check if there's an alternating sequence that could accept this candidate
            active_seq = active_sequences[alternating_key]
            
            last_value = active_seq['last_value']
            comparison = _compare_page_label_values(last_value, token_value, label_type)
            
            if comparison is not None and comparison < 0 and _is_alignment_compatible(candidate.text_align, active_seq):
                # Valid continuation - extend the alternating sequence
                _extend_sequence_info(active_seq, candidate, token_value)
            else:
                # Can't extend - close and start new
                completed_sequences.append(_seq_info_to_candidate_sequence(active_seq))
                del active_sequences[alternating_key]
                
                sequence_counter += 1
                active_sequences[seq_key] = _create_sequence_info(sequence_counter, candidate, token_value, prefix)
        
        else:
            # Check if there's a single-item sequence with opposite alignment that could start alternation
            # For example: active sequence has "left" (length=1), candidate is "right" with value=2
            opposite_align_found = False
            
            if candidate.text_align in ["left", "right"]:
                opposite_align = "right" if candidate.text_align == "left" else "left"
                opposite_key = (sig, label_type, prefix, opposite_align)
                
                if opposite_key in active_sequences:
                    opposite_seq = active_sequences[opposite_key]
                    
                    # Check if it's a single-item sequence and value increments
                    if opposite_seq['length'] == 1:
                        last_value = opposite_seq['last_value']
                        comparison = _compare_page_label_values(last_value, token_value, label_type)
                        
                        if comparison is not None and comparison < 0:
                            # This could be an alternating sequence! Extend it
                            _extend_sequence_info(opposite_seq, candidate, token_value)
                            
                            # The alignment_key should now be "left_right"
                            new_alignment_key = opposite_seq.get('alignment_key')
                            new_key = (sig, label_type, prefix, new_alignment_key)
                            
                            # Move from opposite_key to new_key
                            del active_sequences[opposite_key]
                            active_sequences[new_key] = opposite_seq
                            opposite_align_found = True
            
            if not opposite_align_found:
                # No active sequence with this key
                # Check if there's a paused sequence we can resume (try both keys)
                resumed = False
                for try_key in [seq_key, alternating_key]:
                    if try_key and try_key in paused_sequences and paused_sequences[try_key]:
                        # Resume the most recently paused sequence with this key
                        resumed_seq = paused_sequences[try_key].pop()
                        
                        # Check if we can resume (value should increment from where we left off)
                        last_value = resumed_seq['last_value']
                        comparison = _compare_page_label_values(last_value, token_value, label_type)
                        
                        if comparison is not None and comparison < 0 and _is_alignment_compatible(candidate.text_align, resumed_seq):
                            # Valid continuation AND alignment matches - resume the sequence
                            old_key = try_key
                            _extend_sequence_info(resumed_seq, candidate, token_value)
                            
                            # Use the updated alignment_key from the sequence
                            new_alignment_key = resumed_seq.get('alignment_key')
                            new_key = (sig, label_type, prefix, new_alignment_key)
                            active_sequences[new_key] = resumed_seq
                            resumed = True
                            break
                        else:
                            # Can't resume with this value or alignment doesn't match - put it back
                            paused_sequences[try_key].append(resumed_seq)
                
                if not resumed:
                    # No paused sequence either - start a new sequence
                    sequence_counter += 1
                    active_sequences[seq_key] = _create_sequence_info(sequence_counter, candidate, token_value, prefix)
            
            # Pause all other active sequences (different keys)
            # Find which sequence is now active
            # Could be at: seq_key, alternating_key, or a new key from opposite_align detection
            current_active_seq_id = None
            for k in list(active_sequences.keys()):
                if k in [seq_key, alternating_key] or (
                    k[0] == sig and k[1] == label_type and k[2] == prefix
                ):
                    # Check if this key has a sequence (might be the one we just activated)
                    if k in active_sequences:
                        # Check if this could be our current sequence by looking at recent additions
                        seq = active_sequences[k]
                        if seq.get('box_ids') and seq['box_ids'][-1] == candidate.box_id:
                            current_active_seq_id = seq.get('sequence_id')
                            break
            
            for other_key, other_seq in list(active_sequences.items()):
                if other_seq.get('sequence_id') != current_active_seq_id:
                    # Move to paused - use the key it's currently stored under
                    if other_key not in paused_sequences:
                        paused_sequences[other_key] = []
                    paused_sequences[other_key].append(other_seq)
                    del active_sequences[other_key]
    
    # Close out all remaining sequences (active + paused)
    for seq in active_sequences.values():
        completed_sequences.append(_seq_info_to_candidate_sequence(seq))
    
    for key_seqs in paused_sequences.values():
        for seq in key_seqs:
            completed_sequences.append(_seq_info_to_candidate_sequence(seq))
    
    # Sort by sequence_id for consistent output
    completed_sequences.sort(key=lambda s: s.sequence_id)
    
    return completed_sequences

# ================================
# STEP 5: Winner Selection
# ================================

def _select_winning_sequences(sequences: List[CandidateSequence]) -> List[CandidateSequence]:
    """
    Select non-overlapping sequences using greedy algorithm with weighted scoring.
    
    Algorithm:
    1. Score each sequence: score = (length^2) * 1.0 + (span/100) * 0.1
    2. Sort by score descending, with tie-breakers:
       - Prefer has_table_id=False (page labels usually not in tables)
       - Prefer earlier start_box_id (document order)
    3. Greedily select sequences that don't overlap claimed box_ids
    4. Return selected sequences sorted by start_box_id
    
    Args:
        sequences: List of CandidateSequence objects
        
    Returns:
        List of selected CandidateSequence objects in document order (sorted by start_box_id)
    """
    if not sequences:
        return []
    
    # Step 1: Score and annotate sequences
    scored_sequences = []
    for seq in sequences:
        span = seq.end_box_id - seq.start_box_id
        score = (seq.length ** 2) * 1.0 + (span / 100.0) * 0.1
        
        # Store score on the sequence object
        seq.score = score
        scored_sequences.append(seq)
    
    # Step 2: Sort by score (descending), with tie-breakers
    # Tie-breaker 1: Prefer has_table_id=False (lower is better, so False=0 < True=1)
    # Tie-breaker 2: Prefer earlier start_box_id (lower is better)
    sorted_sequences = sorted(
        scored_sequences,
        key=lambda s: (
            -s.score,  # Higher score first (negate for descending)
            s.formatting_signature.has_table_id,  # False < True (prefer non-table)
            s.start_box_id  # Earlier start first
        )
    )
    
    # Step 3: Greedy selection
    selected = []
    claimed_ranges = []  # List of (start, end) tuples
    
    for seq in sorted_sequences:
        # Check if this sequence's range overlaps with any claimed range
        seq_start = seq.start_box_id
        seq_end = seq.end_box_id
        
        has_overlap = False
        for claimed_start, claimed_end in claimed_ranges:
            # Check if ranges overlap
            # Two ranges [a,b] and [c,d] overlap if NOT (b < c OR d < a)
            # Which simplifies to: (b >= c AND d >= a)
            if seq_end >= claimed_start and claimed_end >= seq_start:
                has_overlap = True
                break
        
        if has_overlap:
            # Overlap detected - skip this sequence
            continue
        
        # No overlap - select this sequence and claim its entire range
        selected.append(seq)
        claimed_ranges.append((seq_start, seq_end))
    
    # Step 4: Sort by document order (start_box_id)
    selected.sort(key=lambda s: s.start_box_id)
    
    return selected

# ================================
# STEP 6: Group Conversion
# ================================

def _detect_alternation_mode(df: pd.DataFrame, box_ids: List[int]) -> AlternationMode:
    """
    Detect alternation mode for a sequence of page labels.
    
    Checks if page labels alternate between left/right positions in sequence order.
    
    Args:
        df: DataFrame with box_id, text_align columns
        box_ids: List of box_ids in the sequence (order matters)
        
    Returns:
        AlternationMode: "fixed", "alternating", "mixed", or "unknown"
    """
    if not box_ids:
        return "unknown"
    
    # Single label can't alternate - it's fixed
    if len(box_ids) == 1:
        return "fixed"
    
    # Get rows for these box_ids
    seq_df = df[df["box_id"].isin(box_ids)].copy()
    
    if seq_df.empty or "text_align" not in seq_df.columns:
        return "unknown"
    
    # Preserve the order of box_ids in the sequence
    # Create a mapping of box_id to its position in the sequence
    box_id_order = {box_id: idx for idx, box_id in enumerate(box_ids)}
    seq_df["seq_order"] = seq_df["box_id"].map(box_id_order)
    
    # Sort by sequence order and get text_align values
    seq_df = seq_df.sort_values("seq_order")
    text_aligns = seq_df["text_align"].tolist()
    
    # Delegate to list-based detection logic
    return _detect_alignment_mode_from_list(text_aligns)


def _detect_position(df: pd.DataFrame, box_ids: List[int]) -> PageLabelGroupPosition:
    """
    Detect the position(s) where page labels appear.
    
    Args:
        df: DataFrame with box_id, text_align columns
        box_ids: List of box_ids in the sequence
        
    Returns:
        PageLabelGroupPosition: "left", "center", "right", "left_right", or "unknown"
    """
    if not box_ids:
        return "unknown"
    
    seq_df = df[df["box_id"].isin(box_ids)]
    
    if seq_df.empty or "text_align" not in seq_df.columns:
        return "unknown"
    
    aligns = set(seq_df["text_align"].dropna().tolist())
    
    if not aligns:
        return "unknown"
    
    # Map to position
    if aligns == {"left"}:
        return "left"
    elif aligns == {"center"}:
        return "center"
    elif aligns == {"right"}:
        return "right"
    elif aligns == {"left", "right"}:
        return "left_right"
    elif "center" in aligns:
        return "center"  # If mixed with center, call it center
    else:
        return "unknown"


def _convert_sequences_to_groups(
    sequences: List[CandidateSequence],
    df: pd.DataFrame,
    candidates_map: Dict[int, PageLabelCandidate]
) -> Tuple[List[PageLabel], List[PageLabelGroup]]:
    """
    Convert winning sequences to PageLabel and PageLabelGroup objects.
    
    Args:
        sequences: List of winning CandidateSequence objects (sorted by start_box_id)
        df: DataFrame with box_id, text_align columns
        candidates_map: Dict mapping box_id -> PageLabelCandidate (for token lookup)
        
    Returns:
        Tuple of (page_labels, page_label_groups)
    """
    all_page_labels = []
    page_label_groups = []
    
    for group_id, seq in enumerate(sequences, start=1):
        # Get tokens for this sequence
        tokens = []
        page_labels_in_group = []
        
        for box_id in seq.box_ids:
            candidate = candidates_map.get(box_id)
            if candidate:
                # Create PageLabel
                page_label = PageLabel(
                    box_id=box_id,
                    raw_token=candidate.raw_token,
                    normalized_token=candidate.normalized_token,
                    corrected_token=None,
                    page_label_type=seq.page_label_type,
                    detection_method="standard",
                    group_id=group_id
                )
                page_labels_in_group.append(page_label)
                all_page_labels.append(page_label)
                tokens.append(candidate.normalized_token)
        
        if not tokens:
            continue
        
        # Detect position and alternation
        position = _detect_position(df, seq.box_ids)
        alternation_mode = _detect_alternation_mode(df, seq.box_ids)
        
        # Create PageLabelGroup
        group = PageLabelGroup(
            group_id=group_id,
            page_label_type=seq.page_label_type,
            position=position,
            alternation_mode=alternation_mode,
            start_token=tokens[0] if tokens else "",
            end_token=tokens[-1] if tokens else "",
            page_labels=page_labels_in_group,
            formatting_signature=seq.formatting_signature
        )
        page_label_groups.append(group)
    
    return all_page_labels, page_label_groups


# =========================
#  Sequencing Pipeline Orchestrator
# =========================

def _detect_page_label_sequence(
    df: pd.DataFrame,
    page_label_config
) -> Tuple[List[PageLabel], List[PageLabelGroup]]:
    """
    Internal function to detect page label sequences.
    
    Pipeline:
    1. Build candidates from filtered tokens
    2. Build candidate sequences
    3. Select winning sequences (non-overlapping)
    4. Convert to PageLabel and PageLabelGroup objects
    
    Args:
        df: DataFrame with page_label_token, formatting columns, wrapper flags
        page_label_config: Compiled pattern config
        
    Returns:
        Tuple of (page_labels, page_label_groups)
    """
    # Step 1: Build candidates from the DataFrame
    candidates = _build_candidates_from_df(df, page_label_config)
    
    if not candidates:
        return [], []
    
    # Step 2: Build candidate sequences
    sequences = _build_candidate_sequences(candidates)
    
    if not sequences:
        return [], []
    
    # Step 3: Select winning (non-overlapping) sequences
    winning_sequences = _select_winning_sequences(sequences)
    
    if not winning_sequences:
        return [], []
    
    # Step 4: Convert to PageLabel and PageLabelGroup objects
    # Need to pass df for alignment detection
    candidates_map = {c.box_id: c for c in candidates}
    page_labels, page_label_groups = _convert_sequences_to_groups(
        winning_sequences, df, candidates_map
    )
    
    return page_labels, page_label_groups

# ---------------------------------------------------------------------------- PART 3: Post-Processing ---------------------------------------------------------------------------- #

# =========================
# POST-PROCESSING HELPERS
# =========================

def _propagate_page_labels_upward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill page_label upwards with propagation rules.
    
    Rules:
    - First label (lowest box_id): Copy upwards until hitting structure_tag='hr'
    - All others: Copy upwards until hitting the prior page_label
    
    Args:
        df: DataFrame with box_id, page_label, structure_tag columns
        
    Returns:
        DataFrame with page_label filled upwards
    """
    if "page_label" not in df.columns or "box_id" not in df.columns or "structure_tag" not in df.columns:
        return df
    
    out = df.copy()
    
    # Find all rows with page_label
    labeled_rows = out[out["page_label"].notna()].sort_values("box_id")
    
    if labeled_rows.empty:
        return out

    has_text = "text" in out.columns
    
    # Process each page label
    for idx, (row_idx, row) in enumerate(labeled_rows.iterrows()):
        current_box_id = row["box_id"]
        current_label = row["page_label"]
        current_label_type = row.get("page_label_type")
        
        # Determine the stop condition based on whether this is the first label
        is_first = (idx == 0)
        
        # Get all rows with box_id < current_box_id
        rows_above = out[out["box_id"] < current_box_id].copy()
        
        if rows_above.empty:
            continue
        
        # Sort by box_id descending (start from just below current_box_id)
        rows_above = rows_above.sort_values("box_id", ascending=False)
        
        # Propagate upwards
        for above_idx, above_row in rows_above.iterrows():
            # Stop conditions
            if is_first:
                # First label: stop at hr tag
                if above_row["structure_tag"] == "hr":
                    break
                # Stop at table of contents
                if has_text:
                    t = above_row["text"]
                    if isinstance(t, str) and t.strip().lower() == "table of contents":
                        break
            else:
                # Other labels: stop at prior page_label
                if pd.notna(above_row["page_label"]):
                    break
            
            # Fill this row with current label and type
            out.loc[above_idx, "page_label"] = current_label
            if current_label_type is not None and "page_label_type" in out.columns:
                out.loc[above_idx, "page_label_type"] = current_label_type
    
    return out


def _infer_page_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate page_number intelligently based on page_label changes.
    
    Logic:
    - Check if all rows have page_number == 1 (indicating unpaginated/needs inference)
    - If yes: Start at 1, increment each time page_label changes (including blank -> something)
    - If no: Skip (already has real page numbers)
    
    Args:
        df: DataFrame with page_number, page_label, box_id columns
        
    Returns:
        DataFrame with page_number populated
    """
    if "page_number" not in df.columns or "page_label" not in df.columns:
        return df
    
    # Check if all rows have page_number == 1 (needs inference)
    needs_inference = (df["page_number"] == 1).all()
    
    if not needs_inference:
        # Already has real page numbers, skip
        return df
    
    out = df.copy()
    
    # Sort by box_id to process in document order
    out = out.sort_values("box_id")
    
    # Track page number and previous label
    current_page = 1
    prev_label = None
    is_first_row = True
    
    # Iterate through rows and assign page numbers
    new_page_nos = []
    for idx, row in out.iterrows():
        current_label = row["page_label"]
        
        # Normalize None/NaN to None for comparison
        if pd.isna(current_label):
            current_label = None
        
        # First row always gets page 1
        # For subsequent rows, increment when page_label changes
        if not is_first_row and current_label != prev_label:
            current_page += 1
        
        # Update prev_label for next iteration
        prev_label = current_label
        is_first_row = False
        
        new_page_nos.append(current_page)
    
    # Assign new page numbers
    out["page_number"] = new_page_nos
    
    # Restore original order (by index)
    out = out.sort_index()
    
    return out
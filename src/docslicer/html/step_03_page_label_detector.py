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
    - Sequence Scoring: length^2 scoring favors longer sequences
    - Alignment Tracking: Enforces consistent alignment within sequences

Author: MarketFramer
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, Set, Union

import pandas as pd

# =========================
# Constants
# =========================

TOP_TOL_PX = 4           # px tolerance for grouping rows into the same "top" band
HR_PAGE_BREAK_MIN_PCT = 80  # HR must span at least this % of page width to signal a page break

# =========================
# Core Dataclasses
# =========================

# -------- Stage 1: Detection ------- #

@dataclass
class FormattingSignature:
    """Properties that define a page label series"""
    height: float
    font_size: float
    font_weight: str
    struct_tag: str
    has_table_id: bool
    has_dash_wrapper: bool
    has_paren_wrapper: bool
    
    def __hash__(self):
        return hash((self.height, self.font_size, self.font_weight, self.struct_tag, self.has_table_id, 
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
    struct_tag: str
    text_align: str
    # Table context
    has_table_id: bool
    # Wrapper detection
    has_dash_wrapper: bool
    has_paren_wrapper: bool
    
    def get_signature(self) -> FormattingSignature:
        return FormattingSignature(self.height, self.font_size, 
                                   self.font_weight, self.struct_tag, self.has_table_id,
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
    tol_px: float = TOP_TOL_PX,
    debug: bool = False,
    use_coordinate_filters: bool = True,
) -> Tuple[pd.DataFrame, List[PageLabel], List[PageLabelGroup]]:
    """
    Detect and assign page labels to document boxes.

    Main entry point for page label detection. Runs the full pipeline to identify
    and validate page label sequences.

    Pipeline:
    1. Extract candidate tokens from text
    2. Filter tokens (exclude TOC, XBRL, tables, crowded bands, etc.)
    3. Build candidate sequences (group by formatting + value monotonicity)
    4. Select winning non-overlapping sequences
    5. Convert to validated PageLabel and PageLabelGroup objects
    6. Add page_label columns to DataFrame
    7. Post-processing:
       - Set block_type = "page_label" for labeled rows
       - Propagate page_label upward (first label until <hr>, others until prior label)
       - Infer page_number if all rows had page_number == 1

    Required columns in df:
      - box_id, y_top, page_number, text
      - struct_tag, height, font_size, font_weight, text_align

    Optional columns:
      - has_link    — improves TOC detection
      - ixbrl_id   — excludes XBRL-tagged content
      - table_id   — excludes large data tables

    Args:
        df: DataFrame with box metadata.
        page_label_config: Compiled page label pattern configuration.
        tol_px: Tolerance in pixels for top-band grouping (default: 4).
        use_coordinate_filters: When False, skip band-based filters that rely on y_top
            (crowded-band, long-text-band, linked-top-bands). Set to False for statically
            extracted boxes where y_top is always 0 — all boxes collapse into one band,
            which would wipe every candidate.
        debug: If True, retain intermediate columns in the returned DataFrame:
               ``page_label_token`` (filtered candidates),
               ``page_label_group_id``, and ``alternation_mode``.
               These columns are dropped by default.

    Returns:
        Tuple of (df, page_labels, page_label_groups) where:

        - df: Original DataFrame with the following columns added/updated:

          Always present:
            - page_label        : validated label propagated upward (or None)
            - page_label_type   : "arabic", "roman", "alpha_numeric", etc. (or None)
            - block_type        : "page_label" for label rows; existing values preserved
            - page_number       : updated in-place if originally all 1s

          Only when debug=True:
            - page_label_token  : filtered, normalized candidate token
            - page_label_group_id : integer group the label belongs to
            - alternation_mode  : "fixed" / "alternating" / "mixed" / "unknown"

        - page_labels: List of PageLabel objects for every detected label.
        - page_label_groups: List of PageLabelGroup objects (one per sequence).
    """
    required = {"box_id", "y_top", "page_number", "text", 
                "struct_tag", "height", "font_size", "font_weight", "text_align"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    out = df.copy()

    # Step 1: Extract all candidate tokens
    inv = _build_page_label_token_inventory(out, page_label_config)

    # Step 2: Apply filters (page-aware)
    cand = _filter_page_label_tokens(out, inv, tol_px=tol_px, use_coordinate_filters=use_coordinate_filters)

    # Attach columns needed by the sequence-detection pipeline.
    # _raw_token preserves the original string before normalization so that
    # PageLabel.raw_token carries the true raw value (e.g. "(iv)" not "iv").
    out["page_label_token"] = cand
    out["_raw_token"] = inv["raw_token"]
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

    # Drop internal temp columns always; drop debug columns only in non-debug mode.
    always_drop = ["_raw_token", "_has_dash_wrapper", "_has_paren_wrapper"]
    debug_cols = ["page_label_token", "page_label_group_id", "alternation_mode"]
    drop_cols = always_drop + ([] if debug else debug_cols)
    out = out.drop(columns=drop_cols)
    
    # ===== POST-PROCESSING STEPS ===== #
    
    # Step 1: Add block_type column (only if it doesn't exist, then update page label rows)
    if "block_type" not in out.columns:
        out["block_type"] = None
    # Only update rows with page labels, preserve existing values
    out.loc[out["page_label"].notna(), "block_type"] = "page_label"
    
    # Step 2: Fill page_label upwards with propagation rules
    out = _propagate_page_labels_upward(out)
    
    # Step 3: Infer page_number from label changes when all rows have page_number == 1
    out = _infer_page_numbers(out)
    
    return out, page_labels, groups


# ---------------------------------------------------------------------------- PART 1: Extraction ---------------------------------------------------------------------------- #

# =========================
# Helpers to remove rows with links
# =========================

def _get_box_ids_in_linked_top_bands(df, tol_px: float = TOP_TOL_PX) -> Set[int]:
    """
    Return all box_ids that belong to any top-band containing at least one linked row.

    Rows are grouped into fixed-grid bands of width ``tol_px`` per page, so that
    closely positioned elements (within ±tol_px) are treated as the same band.
    Page-aware: bands are scoped per page_number to avoid cross-page contamination.

    Example:
      page_number=1, y_top=100, has_link=1
      page_number=1, y_top=101, has_link=0
    → both box_ids are returned (same page, same band).

      page_number=2, y_top=100, has_link=0
    → this box_id is NOT returned (different page).

    Args:
        df: DataFrame with columns ``box_id``, ``y_top``, ``page_number``.
            If ``has_link`` is absent, an empty set is returned immediately.
        tol_px: Band width in pixels (default: TOP_TOL_PX).

    Returns:
        Set of box_ids to exclude.
    """
    required = {"box_id", "y_top", "page_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    if "has_link" not in df.columns:
        return set()

    # Assign every row to a fixed-grid band key: (page_number, band_index).
    # Using integer band indices avoids floating-point key collisions.
    band_idx = (df["y_top"] / tol_px).round().astype(int)
    band_key = pd.Series(
        list(zip(df["page_number"], band_idx)), index=df.index, dtype=object
    )

    # Identify band keys that contain at least one linked row.
    linked_band_keys = set(band_key[df["has_link"].isin([1, True])])
    if not linked_band_keys:
        return set()

    # Return every box_id whose band key is in the linked set.
    return set(df.loc[band_key.isin(linked_band_keys), "box_id"])


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

# Embedded label patterns — used to extract page numbers from compound text.
#
# N-of-M: matches "2 of 20", "Page 2 of 20", "... 2 of 20".
#   - Anchored at end of string to avoid matching "3 of 10 items".
#   - Limits digits to 1-4 to avoid matching large financial numbers.
#   - Validated post-match: 0 < N ≤ M.
_N_OF_M_RE = re.compile(r"\b(\d{1,4})\s+of\s+(\d{1,4})\s*$", re.IGNORECASE)

# HR page-break marker: matches [[HR: 100%]], [[HR: 85.5%]], etc.
_HR_PCT_RE = re.compile(r"^\[\[HR:\s*(\d+(?:\.\d+)?)\s*%\s*\]\]$", re.IGNORECASE)


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


def _is_pure_negative_number(token) -> bool:
    """
    Return True if *token* is a bare negative integer like ``-1`` or ``-23``.

    Excludes dash-wrapped labels such as ``-2-`` (which are legitimate page
    label wrappers) and non-numeric strings.
    """
    if pd.isna(token):
        return False
    s = str(token)
    return s.startswith("-") and not s.endswith("-") and len(s) > 1 and s[1:].isdigit()


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


def _try_extract_embedded_label(text: str) -> Optional[str]:
    """
    Try to extract a page number token from compound text that would otherwise
    be rejected by the length gate or fail type matching.

    Supported patterns (checked in order of specificity):

    1. **N-of-M suffix** — ``"... 2 of 20"``, ``"Page 2 of 20"``, ``"2 of 20"``

       Extracts N.  Constraints that limit false positives:

       - End-anchored (``$``): ``"3 of 10 items"`` does **not** match.
       - Requires a bare integer before ``"of"``: prose such as
         ``"on behalf of 20"`` does not match.
       - Validates ``0 < N ≤ M``.

    2. **Pipe-terminated label** — ``"Apple Inc. | Q1 2026 | 2"``

       Takes the last segment after the final ``"|"`` and strips whitespace.
       Type-matching by the caller rejects segments that are not valid page
       label tokens (e.g. ``"Header | Some Text"`` is rejected because
       ``"Some Text"`` does not match any page label pattern).

    Args:
        text: Raw text string from a document box.

    Returns:
        Extracted candidate string (still un-normalized), or ``None``.
    """
    if not text:
        return None

    stripped = text.strip()

    # Pattern 1: N of M — most specific, check first.
    m = _N_OF_M_RE.search(stripped)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        if 0 < n <= total:
            return m.group(1)

    # Pattern 2: pipe-terminated — take the last segment.
    if "|" in stripped:
        last = stripped.rsplit("|", 1)[-1].strip()
        if last:
            return last

    # Pattern 3: word-prefix + structured label — "Annex A-1-86", "Exhibit F-5".
    # Requires a hyphen in the last space-segment to exclude bare arabic ("Revenue 5")
    # and bare roman ("Section iv") which are too ambiguous without document context.
    if " " in stripped:
        last = stripped.rsplit(" ", 1)[-1].strip()
        if last and "-" in last:
            return last

    return None


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
      - box_id, y_top, x_left, struct_tag, has_link (not required here)
    
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

        # Direct match: token must be within the length gate and match a known type.
        if norm and len(norm) <= max_len:
            t = _match_page_label_type(norm, page_label_config)
            if t != "unknown":
                raw_tokens.append(raw_tok)
                norm_tokens.append(norm)
                types.append(t)
                is_like.append(True)
                has_dash_wrappers.append(has_dash)
                has_paren_wrappers.append(has_paren)
                continue

        # Direct match failed (too long or unrecognized type).
        # Try extracting an embedded page number from compound text formats
        # such as "Header | Subheader | 2" or "Page 2 of 20".
        # raw_token stays as the original full text; only normalized_token changes.
        embedded = _try_extract_embedded_label(raw_tok)
        if embedded:
            embedded_norm = _normalize_page_label_token(embedded)
            if embedded_norm and len(embedded_norm) <= max_len:
                t = _match_page_label_type(embedded_norm, page_label_config)
                if t != "unknown":
                    raw_tokens.append(raw_tok)
                    norm_tokens.append(embedded_norm)
                    types.append(t)
                    is_like.append(True)
                    has_dash_wrappers.append(False)  # wrappers don't apply to embedded labels
                    has_paren_wrappers.append(False)
                    continue

        # Nothing matched — record what we have for debugging visibility.
        raw_tokens.append(raw_tok)
        norm_tokens.append(norm if norm else None)
        types.append("unknown")
        is_like.append(False)
        has_dash_wrappers.append(has_dash)
        has_paren_wrappers.append(has_paren)

    out = pd.DataFrame(
        {
            "raw_token": pd.array(raw_tokens, dtype=object),
            "normalized_token": pd.array(norm_tokens, dtype=object),
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
    tol_px: float = TOP_TOL_PX,
    use_coordinate_filters: bool = True,
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
      - box_id, y_top, page_number, struct_tag
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
    required = {"box_id", "y_top", "page_number", "struct_tag"}
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
    # Only apply if has_link column is present and coordinates are reliable
    if use_coordinate_filters and "has_link" in df.columns:
        linked_band_box_ids = _get_box_ids_in_linked_top_bands(df, tol_px=tol_px)
        in_link_band = df["box_id"].isin(linked_band_box_ids)
        cand = cand.where(~in_link_band, None)

    # Exclude rows with ixbrl_id (XBRL-tagged content)
    if "ixbrl_id" in df.columns:
        has_ixbrl = df["ixbrl_id"].notna() & (df["ixbrl_id"] != "")
        cand = cand.where(~has_ixbrl, None)
    
    # Exclude bare negative integers like -1, -2, -3 (common in financial tables).
    # Dash-wrapped labels like -2- are intentionally preserved.
    filled = cand.fillna("")
    is_negative = (
        filled.str.startswith("-")
        & ~filled.str.endswith("-")
        & (filled.str.len() > 1)
        & filled.str[1:].str.isdigit()
    )
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
    
    # Build a lightweight working frame for band-level aggregations.
    # We only add derived columns here; the original df is never mutated.
    band_df = df[["page_number", "y_top"]].copy()
    band_df["rounded_top"] = (df["y_top"] / tol_px).round() * tol_px
    band_key = ["page_number", "rounded_top"]

    # Build a MultiIndex once; reused for all band membership tests below.
    # pd.MultiIndex.isin() is implemented in C and avoids any Python-level row loop.
    row_band_midx = pd.MultiIndex.from_arrays(
        [band_df["page_number"], band_df["rounded_top"]]
    )

    # Exclude rows where the same top band has more than 2 items (likely headers/content).
    # Skipped when use_coordinate_filters=False: all y_top=0 collapses everything into one
    # band, which would wipe every candidate.
    if use_coordinate_filters:
        band_counts = band_df.groupby(band_key).size()
        crowded_midx = band_counts[band_counts > 2].index
        in_crowded_band = pd.Series(row_band_midx.isin(crowded_midx), index=df.index)
        cand = cand.where(~in_crowded_band, None)

    # Exclude rows where the same top band has total character length > 50 (likely text content).
    if use_coordinate_filters and "text" in df.columns:
        band_df["text_len"] = df["text"].fillna("").astype(str).str.len()
        band_char_totals = band_df.groupby(band_key)["text_len"].sum()
        long_text_midx = band_char_totals[band_char_totals > 50].index
        in_long_text_band = pd.Series(row_band_midx.isin(long_text_midx), index=df.index)
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
      - box_id, page_label_token, _raw_token
      - struct_tag, height, font_size, font_weight, text_align
      - _has_dash_wrapper, _has_paren_wrapper (temporary wrapper columns)
    Optional:
      - table_id (for has_table_id detection)

    Args:
        df: DataFrame from assign_page_labels() with page_label_token and _raw_token columns.
        page_label_config: Compiled PageLabelPatternConfig used to infer page_label_type.

    Returns:
        List of PageLabelCandidate objects (only for rows with a valid page_label_token).
    """
    required = {
        "box_id", "page_label_token", "_raw_token",
        "struct_tag", "height", "font_size", "font_weight", "text_align",
        "_has_dash_wrapper", "_has_paren_wrapper",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    # Only process rows that survived filtering
    valid_df = df[df["page_label_token"].notna()].copy()

    candidates = []
    for _, row in valid_df.iterrows():
        normalized = row["page_label_token"]
        raw = row["_raw_token"] if pd.notna(row["_raw_token"]) else normalized

        # Prefer the type already computed during inventory; re-infer only if missing.
        label_type = row["page_label_type"] if "page_label_type" in row.index else "unknown"
        if not label_type or label_type == "unknown" or pd.isna(label_type):
            label_type = _match_page_label_type(normalized, page_label_config)

        # table_id is truthy only when non-blank
        table_id_val = row.get("table_id")
        has_table_id = pd.notna(table_id_val) and str(table_id_val).strip() != ""

        # Wrapper flags default to False when the token was never extracted
        has_dash_wrapper = bool(row["_has_dash_wrapper"]) if pd.notna(row["_has_dash_wrapper"]) else False
        has_paren_wrapper = bool(row["_has_paren_wrapper"]) if pd.notna(row["_has_paren_wrapper"]) else False

        candidates.append(PageLabelCandidate(
            box_id=int(row["box_id"]),
            raw_token=raw,
            normalized_token=normalized,
            page_label_type=label_type,
            height=float(row["height"]),
            font_size=float(str(row["font_size"]).replace("px", "")),
            font_weight=str(row["font_weight"]),
            struct_tag=str(row["struct_tag"]),
            text_align=str(row["text_align"]),
            has_table_id=has_table_id,
            has_dash_wrapper=has_dash_wrapper,
            has_paren_wrapper=has_paren_wrapper,
        ))

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
    
    # Try alpha-numeric-sub: A-1-86, A-2-3 → prefix is "letter-major", number is minor
    match = re.match(r'^([A-Z]-\d{1,3})-(\d+)$', upper_token)
    if match:
        return (match.group(1), int(match.group(2)))

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
        
    elif label_type in ["alpha_numeric", "alpha_roman", "roman_numeric", "alpha_numeric_sub"]:
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
        
    elif label_type in ["alpha_numeric", "alpha_roman", "roman_numeric", "alpha_numeric_sub"]:
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
        if label_type in ["alpha_numeric", "alpha_roman", "roman_numeric", "alpha_numeric_sub"]:
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

def _passes_global_quality_gate(sequences: List[CandidateSequence], doc_char_len: int) -> bool:
    """
    Return False if the winning sequences do not meet document-scale quality thresholds.

    Both criteria must pass, or all detection is discarded:

    1. **Five-consecutive requirement**: at least one winning sequence must have
       length >= 5.  A document where no run of five sequential page labels was
       found is not reliably paginated enough to trust any label, including lone
       singletons.

    2. **Minimum label count**: for long documents the total number of detected
       labels must be proportional to the implied page count:

           minimum_labels = floor(doc_char_len * 0.90 / 5000)

       For a 350 000-char document this yields 63; if fewer than 63 labels are
       found across all winning sequences the detection is considered too sparse
       and is rejected in full.  The minimum is 0 for documents shorter than
       ~5 556 chars, so very short files are never penalised.

    Args:
        sequences: Winning (non-overlapping) CandidateSequence objects.
        doc_char_len: Total character length of the document text.

    Returns:
        True if detection should be kept, False if all sequences should be discarded.
    """
    if not sequences:
        return False

    # Criterion 1: need at least one run of 5 consecutive labels.
    if not any(seq.length >= 5 for seq in sequences):
        return False

    # Criterion 2: total labels must meet the doc-length-scaled minimum.
    minimum_labels = int(doc_char_len * 0.90) // 5000
    if minimum_labels > 0:
        total_labels = sum(seq.length for seq in sequences)
        if total_labels < minimum_labels:
            return False

    return True


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
    3a. Apply global quality gate — reject all if thresholds not met
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

    # Step 3a: Global quality gate — discard all detection if the document does
    # not have enough evidence to trust the result in totality.
    doc_char_len = int(df["text"].fillna("").astype(str).str.len().sum())
    if not _passes_global_quality_gate(winning_sequences, doc_char_len):
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
    Fill page_label upwards according to two rules:

    - **First label** (lowest box_id): propagate upward until hitting
      ``struct_tag='hr'`` or a "table of contents" text row (inclusive stop —
      the stop row and everything above it remain unlabeled).
    - **All other labels**: propagate upward until hitting the prior page_label
      (i.e., each unlabeled gap between two consecutive labels is filled with the
      label of the one immediately below it).

    Uses ``bfill()`` on the box_id-sorted frame for O(n) vectorized propagation,
    followed by a single boundary-clipping step for the first-label rule.

    Args:
        df: DataFrame with ``box_id``, ``page_label``, and ``struct_tag`` columns.

    Returns:
        DataFrame with ``page_label`` (and ``page_label_type`` if present) filled upward.
    """
    if "page_label" not in df.columns or "box_id" not in df.columns or "struct_tag" not in df.columns:
        return df

    out = df.copy().sort_values("box_id")

    labeled_rows = out[out["page_label"].notna()]
    if labeled_rows.empty:
        return df

    # bfill() on the ascending-box_id frame propagates each label to all unlabeled
    # rows above it, stopping naturally at the previous labeled row.
    # This handles all "other labels" cases in one vectorized pass.
    out["page_label"] = out["page_label"].bfill()
    if "page_label_type" in out.columns:
        out["page_label_type"] = out["page_label_type"].bfill()

    # Handle the first-label boundary: clip any fill that crossed an <hr> or
    # a "table of contents" row above the first label.
    first_box_id = labeled_rows["box_id"].min()
    before_first = out["box_id"] < first_box_id

    cutoff_candidates = out.loc[before_first & (out["struct_tag"] == "hr"), "box_id"]

    if "text" in out.columns:
        toc_mask = before_first & (
            out["text"].astype(str).str.strip().str.lower() == "table of contents"
        )
        cutoff_candidates = pd.concat([cutoff_candidates, out.loc[toc_mask, "box_id"]])

    if not cutoff_candidates.empty:
        # Null out every row at or above the last stop marker before the first label.
        cutoff = cutoff_candidates.max()
        null_mask = out["box_id"] <= cutoff
        out.loc[null_mask, "page_label"] = None
        if "page_label_type" in out.columns:
            out.loc[null_mask, "page_label_type"] = None

    return out.sort_index()


def _passes_hr_quality_gate(hr_count: int, doc_char_len: int) -> bool:
    """
    Return False if the number of HR page breaks is too sparse for the document size.

    Uses the same scaling formula as ``_passes_global_quality_gate``:

        minimum_breaks = floor(doc_char_len * 0.90 / 5000)

    For a 500 000-char document this yields 90; a single accidental HR would be
    rejected.  The minimum is 0 for documents shorter than ~5 556 chars, so very
    short files are never penalised.

    Args:
        hr_count: Number of qualifying HR page-break rows detected.
        doc_char_len: Total character length of the document text.

    Returns:
        True if HR-based pagination should be applied, False if it should be skipped.
    """
    minimum_breaks = int(doc_char_len * 0.90) // 5000
    if minimum_breaks > 0 and hr_count < minimum_breaks:
        return False
    return True


def _infer_page_numbers_from_hr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer page numbers from full-width HR markers when no page labels exist.

    Each row whose ``text`` matches ``[[HR: X%]]`` with X >= ``HR_PAGE_BREAK_MIN_PCT``
    (default 80) is treated as a page-break separator.  The HR row itself stays on
    page N; the following row starts page N+1.

    A quality gate rejects inference when the HR count is too sparse relative to the
    document size (same scaling formula as the label-based quality gate).  This
    prevents a single accidental HR in a large document from splitting it into two
    spurious pages.

    This is the fallback strategy for filings such as 8-Ks that use SEC-style
    horizontal rules to divide pages but carry no numeric page labels.

    Args:
        df: DataFrame sorted by ``box_id``, must contain a ``text`` column.

    Returns:
        DataFrame with ``page_number`` set to inferred values (starting at 1),
        or the original DataFrame unchanged if the quality gate rejects inference.
    """
    if "text" not in df.columns:
        return df

    out = df.copy()

    # Vectorized HR percentage extraction.
    hr_pct = (
        out["text"]
        .astype(str)
        .str.strip()
        .str.extract(_HR_PCT_RE, expand=False)
    )
    hr_pct = pd.to_numeric(hr_pct, errors="coerce")

    # A row triggers a page break if its HR width meets the threshold.
    is_page_break = hr_pct >= HR_PAGE_BREAK_MIN_PCT

    # Quality gate: reject if HR breaks are too sparse for the document size.
    hr_count = int(is_page_break.sum())
    doc_char_len = int(out["text"].fillna("").astype(str).str.len().sum())
    if not _passes_hr_quality_gate(hr_count, doc_char_len):
        return df

    # Shift forward by one so the HR itself stays on page N and the next
    # row begins page N+1.  fill_value=False keeps the first row on page 1.
    out["page_number"] = is_page_break.shift(1, fill_value=False).cumsum() + 1

    return out


def _infer_page_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate ``page_number`` when the source file carries no real pagination.

    Only runs when every row has ``page_number == 1`` (the sentinel that signals
    an unpaginated file).  Chooses between two strategies:

    - **Label-based** (preferred): increment on each ``page_label`` change.
      Used whenever at least one page label was detected.
    - **HR-based** (fallback): increment after each full-width ``[[HR: X%]]``
      row (X >= ``HR_PAGE_BREAK_MIN_PCT``).  Used for filings such as 8-Ks
      that have no page labels but use horizontal rules as page separators.

    Args:
        df: DataFrame with ``page_number``, ``page_label``, and ``box_id`` columns.

    Returns:
        DataFrame with ``page_number`` updated in-place.
    """
    if "page_number" not in df.columns or "page_label" not in df.columns:
        return df

    out = df.copy().sort_values("box_id")

    if out["page_label"].notna().any():
        # Label-based inference — always runs when labels were detected,
        # regardless of existing page_number values.
        #
        # Why: some HTML parsers populate page_number only on the exact label
        # rows (the rows that carry the printed page number), leaving all other
        # rows at page_number=1.  The old all()==1 guard incorrectly skipped
        # inference for those files, so propagated-label rows kept page_number=1.
        # Since our page labels are the authoritative pagination signal, we
        # always recompute from them.
        labels = out["page_label"].fillna("__NONE__")
        changed = labels != labels.shift(1)
        changed.iloc[0] = False
        out["page_number"] = changed.cumsum() + 1

    elif (out["page_number"] == 1).all():
        # No labels detected and every row still has the default page_number=1
        # (truly unpaginated file).  Fall back to HR-based inference.
        out = _infer_page_numbers_from_hr(out)

    # else: no labels but page_numbers already vary → trust the existing values.

    return out.sort_index()
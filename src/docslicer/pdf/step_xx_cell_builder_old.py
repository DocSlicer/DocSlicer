"""
step_05_cell_builder.py (vectorized, DataFrame-only version)

Public API:
    df_cells, df_words = build_cells(df_words, df_shapes, df_links)

Incoming df_words columns (expected):
    [
        "page_number", "word_id", "text",
        "x_left", "y_top", "x_right", "y_bottom", "width", "height",
        "font_name", "font_size",
        "non_stroking_color", "stroking_color",
        "bold_ratio", "italic_ratio",
        "char_count", "alpha_count", "digit_count", "uppercase_count",
        "word_count", "alpha_word_count", "capitalized_word_count",
        "page_width", "page_height",
        ...
    ]

This script:
  Step 1-3: Cell Building
    - adds: top_bucket, temp_line_id, cell_id to df_words
    - builds df_cells by grouping on cell_id and recomputing layout + stats
  
  Step 4: Link Relationships (vectorized)
    - adds: has_link, link_url, link_dest, link_type to df_cells
    - uses bbox overlap to match cells with links from df_links
    - much faster than old approach via vectorized overlap calculations
  
  Step 5: Rectangle Relationships (vectorized)
    - adds: inside_rect_shape, background_*_color, shape_id_container to df_cells
    - checks if cells are contained within rectangles from df_shapes
    - vectorized containment checks per page
  
  Step 6: Vertical Line Relationships (vectorized)
    - adds: has_vertical_line, shape_id_vertical_line to df_cells
    - checks if cell centers fall within vertical line ranges
    - must run BEFORE horizontal band assignment to enable vertical-line-based merging
  
  Step 7: Horizontal Band Assignment
    - adds: horizontal_band_id, page_median_gap, page_gap_thresh to df_cells
    - groups cells into horizontal content bands based on vertical gaps
    - uses temp_line_id groups and adaptive gap thresholds
    - merges bands that share vertical lines (tunable threshold)
  
  Step 8: Underline Assignment
    - adds: shape_id_underline to df_cells
    - associates cells with underline shapes
  
  Note: Column reordering is now handled centrally by the orchestrator
    before CSV export via utils.reorder_columns.reorder_columns()
"""

from __future__ import annotations

from typing import Tuple, Optional, Any, Dict, List

import numpy as np
import pandas as pd

from .step_07_cell_builder import assign_cell_underlines
from .._utils.line_merger import assign_line_id
    
# ================================================================================
# Step 2: assign cell_id within each temp_line_id
# ================================================================================

# ---------------------
# Cell Merging Config
# ---------------------

# Base merge thresholds
def _interpolate_font_gap(
    font_size: Optional[float],

    # ▼▼▼ MAIN TOGGLES BELOW ▼▼▼
    gap_at_low: float = 6.0,  # --> Base Merge Threshold (also fallback when font_size is None/invalid)
    gap_at_high: float = 10.0, # --> Max Merge Threshold
     # ▲▲▲ MAIN TOGGLES ABOVE ▲▲▲

    low_size: float = 10.0,   # --> Low Font Size
    high_size: float = 24.0,  # --> High Font Size
) -> float:
    """
    Returns an adaptive cell-merge gap threshold based on font size.

    - if font_size is None, 0, or NaN → gap_at_low (fallback)
    - if font_size <= low_size  → gap_at_low  (e.g. 5pt at size ≤ 10)
    - if font_size >= high_size → gap_at_high (e.g. 7pt at size ≥ 24)
    - else linear interpolation between them.
    """
    # Handle invalid/missing font_size - return the base threshold
    if font_size is None or font_size <= 0 or np.isnan(font_size):
        return gap_at_low
    
    if font_size <= low_size:
        return gap_at_low
    if font_size >= high_size:
        return gap_at_high

    t = (font_size - low_size) / (high_size - low_size)
    return gap_at_low + t * (gap_at_high - gap_at_low)


# Special merge cases
_BULLET_MERGE_ENABLED = True
_BULLET_MERGE_MAX_GAP = 30.0  # max allowed gap for bullet→text
_DOLLAR_MERGE_MAX_GAP = 60.0  # max allowed gap for "$"→number merges
_SENTENCE_MERGE_MAX_GAP = 10.0  # Increased tolerance compared to [interpolate_font_gap], to prevent justified text from exploding into multiple cells


# Common bullet-like tokens; extend as needed
_BULLET_TOKENS = {
    "-", "–", "—",          # en/em dash / hyphen – you can remove "-" if too aggressive
    "•", "·",               # classic bullets
    "■", "▪", "",          # squares / special bullet glyphs
    "…",                    # ellipsis, if used as leader
    "+", "☒", "☐",
    "○", "◦", "►", "▸", "‣", "⁃",
    "✓", "✔", "✗", "✘", "✖", "✕",
}
# TODO: Potentially add support for "(1)", "1.", etc.

# ---------------------
# Small helpers
# ---------------------

def _is_numeric_like(text: str) -> bool:
    """
    Heuristic: treat as numeric-ish if it mainly consists of digits,
    commas, dots, parentheses, minus and percent signs.
    """
    if not text:
        return False
    text = str(text).strip()
    if not text:
        return False
    # Allow things like "16,654", "0.62", "-6%", "(3)"
    return all(ch.isdigit() or ch in ",.()-+% " for ch in text)


def _is_bullet_token(text: str) -> bool:
    """
    Return True if `text` looks like a standalone bullet/leader token.
    """
    if not text:
        return False
    t = str(text).strip()
    return t in _BULLET_TOKENS


def _mode_or_first(series: pd.Series) -> Any:
    """
    Fast-ish "most prevalent" helper:
      - if there is a mode, return the first mode
      - else fall back to first non-null
    """
    if series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if not vc.empty:
        return vc.index[0]
    # All NA
    return series.dropna().iloc[0] if series.dropna().size else None

# ---------------------
# Sentence helper to catch justified text
# ---------------------

# Minimal, high-signal stopwords for “sentence vs table” (grammar glue only)
_STOPWORDS = {
    "the","and","of","to","in","for","with","as","on","by","from","at",
    "into","among","including","that","which","who","whose","its",
    "is","are","was","were","be","been","being"
}
_PUNCT_CHARS = ".,;:"
_STRIP_CHARS = ".,;:()[]{}\"'"


def is_sentence_like_line(words_df_line) -> tuple[bool, int]:
    """
    Fast, line-local heuristic to identify sentence-like (paragraph) lines.

    Designed to be cheap:
      - Uses existing aggregate columns: word_count, alpha_word_count, digit_count,
        uppercase_count, capitalized_word_count, char_count, alpha_count.
      - One lightweight pass over tokens for stopwords + punctuation.

    words_df_line: slice of words_df for ONE temp_line_id (one row per token)
    
    Returns:
        tuple[bool, int]: (is_sentence_like, score)
    """

    if words_df_line is None or words_df_line.empty:
        return False, 0

    # Count number of tokens (words) in this line
    # word_count could be per-word (1 per row) or a repeated line total
    # Use len() as the reliable count of words in the line
    n = len(words_df_line)

    # Too short → not a sentence
    if n < 5:
        return False, 0

    # === Aggregate metrics (cheap) ===
    # If your schema guarantees these columns exist, you can drop the .get fallbacks.
    alpha_tokens = int(words_df_line.get("alpha_word_count", 0).sum())
    digit_chars = int(words_df_line.get("digit_count", 0).sum())
    uppercase_chars = int(words_df_line.get("uppercase_count", 0).sum())  # not used directly, but kept
    capitalized_tokens = int(words_df_line.get("capitalized_word_count", 0).sum())
    total_chars = int(words_df_line.get("char_count", 0).sum())

    alpha_ratio = alpha_tokens / max(n, 1)
    numeric_ratio = digit_chars / max(total_chars, 1)

    # === Cheap text-based checks (single pass) ===
    stop_hits = 0
    has_punct = False

    # Require alpha_count column for the “8.4% shouldn’t count as punctuation” rule
    has_alpha_count = "alpha_count" in words_df_line.columns

    for row in words_df_line.itertuples(index=False):
        t = row.text

        # stopwords: normalize by lowercasing + stripping surrounding punctuation
        t_norm = t.lower().strip(_STRIP_CHARS)
        if t_norm in _STOPWORDS:
            stop_hits += 1

        # sentence punctuation: only if token contains alphabetic chars
        if not has_punct:
            alpha_ok = (row.alpha_count > 0) if has_alpha_count else any(ch.isalpha() for ch in t)
            if alpha_ok and any(p in t for p in _PUNCT_CHARS):
                has_punct = True

    # === Scoring ===
    score = 0

    # Length
    if n >= 8:
        score += 2
    elif n >= 6:
        score += 1

    # Stopwords
    if stop_hits >= 2:
        score += 2
    elif stop_hits == 1:
        score += 1

    # Punctuation
    if has_punct:
        score += 1

    # Alphabetic dominance
    if alpha_ratio >= 0.75:
        score += 2
    elif alpha_ratio >= 0.60:
        score += 1

    # Numeric penalty (using numeric chars share)
    if numeric_ratio >= 0.35:
        score -= 3
    elif numeric_ratio >= 0.20:
        score -= 2
    elif numeric_ratio >= 0.12:
        score -= 1

    # Lowercase body-text proxy:
    # lots of alpha tokens but relatively few capitalized tokens → paragraph
    if alpha_tokens >= 4 and capitalized_tokens <= alpha_tokens * 0.5:
        score += 1

    # Table-header penalty: many capitalized tokens and no stopwords → header-like
    if capitalized_tokens >= 4 and stop_hits == 0:
        score -= 2

    return score >= 4, score


# ---------------------
# Main decision engine
# ---------------------
    
"""
Assign 2 words on the same line to 1 cell if they are close enough, otherwise create a new cell.
"""

def _assign_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with temp_line_id, assign a global cell_id based on horizontal gaps.

    Rules PER temp_line_id (in left→right order):
      - Check if line is sentence-like via is_sentence_like_line()
      - If sentence-like, use _SENTENCE_MERGE_MAX_GAP as threshold
      - Otherwise, compute threshold via _interpolate_font_gap(median_font_size)
      - Compute gap[i] = x_left[i+1] - x_right[i].
      - Merge words into same cell if:
          gap <= line_merge_threshold
          OR (left is bullet AND gap <= _SPECIAL_MERGE_MAX_GAP)
          OR (left == "$" AND right is numeric-like AND gap <= _SPECIAL_MERGE_MAX_GAP)
      - cell_id is unique across entire DataFrame.
      - Adds is_sentence_like debug column to track detection results.
    """
    if df.empty:
        df["cell_id"] = pd.Series(dtype="int64")
        df["is_sentence_like"] = pd.Series(dtype="bool")
        df["sentence_score"] = pd.Series(dtype="int32")
        return df

    # Ensure sorted by temp_line_id, then left→right
    df = df.sort_values(
        ["page_number", "temp_line_id", "x_left", "y_top"],
        kind="mergesort",
    ).reset_index(drop=True)


    line_arr = df["temp_line_id"].to_numpy(dtype=np.int64)
    x_left_arr = df["x_left"].to_numpy(dtype=float)
    x_right_arr = df["x_right"].to_numpy(dtype=float)
    font_size_arr = df["font_size"].to_numpy(dtype=float)
    text_arr = df["text"].astype(str).to_numpy()

    n = len(df)
    cell_id_arr = np.empty(n, dtype=np.int64)
    is_sentence_like_arr = np.empty(n, dtype=bool)
    sentence_score_arr = np.empty(n, dtype=np.int32)

    next_cell_id = 1
    i = 0

    while i < n:
        current_line = line_arr[i]
        # Find end of this temp_line_id block [i:j)
        j = i + 1
        while j < n and line_arr[j] == current_line:
            j += 1

        # Slice indices for this line
        idx_start, idx_end = i, j
        line_indices = np.arange(idx_start, idx_end)

        # Check if this line is sentence-like
        line_df = df.iloc[idx_start:idx_end]
        is_sentence, sentence_score = is_sentence_like_line(line_df)
        
        # Store sentence detection result and score for all words in this line
        is_sentence_like_arr[idx_start:idx_end] = is_sentence
        sentence_score_arr[idx_start:idx_end] = sentence_score

        # Determine merge threshold based on sentence detection
        if is_sentence:
            line_threshold = _SENTENCE_MERGE_MAX_GAP
        else:
            # Compute median font_size for line where > 0
            line_fonts = font_size_arr[idx_start:idx_end]
            valid_fonts = line_fonts[line_fonts > 0]
            median_font = float(np.median(valid_fonts)) if valid_fonts.size > 0 else None
            line_threshold = _interpolate_font_gap(median_font)

        # Assign cell_ids within this line
        # Always start new cell for first word in line
        current_cell = next_cell_id
        cell_id_arr[idx_start] = current_cell

        # Walk pairs (k, k+1)
        for k in range(idx_start, idx_end - 1):
            left_x_right = x_right_arr[k]
            right_x_left = x_left_arr[k + 1]
            gap = right_x_left - left_x_right

            left_text = text_arr[k].strip()
            right_text = text_arr[k + 1].strip()

            # --- NEW: guard against negative gaps (mis-ordered or overlapping) ---
            if gap < 0:
                # Force a new cell; don't allow "wrap-around" merges
                next_cell_id += 1
                current_cell = next_cell_id
                cell_id_arr[k + 1] = current_cell
                continue

            bullet_pair = False
            if _BULLET_MERGE_ENABLED and _is_bullet_token(left_text) and right_text:
                if gap <= _BULLET_MERGE_MAX_GAP:
                    bullet_pair = True

            dollar_pair = False
            if left_text == "$" and _is_numeric_like(right_text):
                if gap <= _DOLLAR_MERGE_MAX_GAP:
                    dollar_pair = True

            if gap <= line_threshold or bullet_pair or dollar_pair:
                # Same cell
                cell_id_arr[k + 1] = current_cell
            else:
                # New cell
                next_cell_id += 1
                current_cell = next_cell_id
                cell_id_arr[k + 1] = current_cell


        # Prepare for next line
        next_cell_id += 1
        i = j

    df["cell_id"] = cell_id_arr
    df["is_sentence_like"] = is_sentence_like_arr
    df["sentence_score"] = sentence_score_arr
    return df


# ---------------------
# Step 3: build df_cells from df_words (with cell_id)
# ---------------------

def _build_cells_df(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Group df_words by cell_id and recompute geometry + stats for each cell.

    Rules:
      - Geometry:
          x_left  = min(x_left)
          x_right = max(x_right)
          y_top   = min(y_top)
          y_bottom= max(y_bottom)
          width   = x_right - x_left
          height  = y_bottom - y_top
      - Text: join word texts with a space.
      - Counts: sum the per-word counts (char_count, alpha_count, etc.).
      - bold_ratio / italic_ratio:
          weighted by char_count, via per-word bold_ratio * char_count.
      - font_name / colors: most prevalent (mode; fall back to first).
    """
    if df_words.empty:
        # Note: link and rect columns are added later by separate functions
        return pd.DataFrame(
            columns=[
                "cell_id",
                "page_number",
                "top_bucket",
                "temp_line_id",
                "is_sentence_like",
                "sentence_score",
                "text",
                "x_left",
                "x_right",
                "y_top",
                "y_bottom",
                "width",
                "height",
                "font_name",
                "font_family",
                "font_size",
                "text_orientation",
                "non_stroking_color",
                "stroking_color",
                "bold_ratio",
                "italic_ratio",
                "char_count",
                "alpha_count",
                "digit_count",
                "uppercase_count",
                "word_count",
                "alpha_word_count",
                "capitalized_word_count",
                "page_width",
                "page_height",
                "word_ids",
            ]
        )

    df = df_words.copy()

    # NEW: ensure words inside each cell are ordered left→right
    # (fallback on y_top just to break ties deterministically)
    df = df.sort_values(["cell_id", "x_left", "y_top"], kind="mergesort").reset_index(drop=True)

    # Helper: approximate bold/italic char counts for weighted ratios
    df["bold_char_est"] = df["bold_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)
    df["italic_char_est"] = df["italic_ratio"].fillna(0.0) * df["char_count"].fillna(0.0)

    # Aggregate word_ids per cell for debugging/tracing
    # (keeps the original word_id column as list)
    def _listify(series: pd.Series) -> List[Any]:
        return list(series)

    agg_spec: Dict[str, Any] = {
        "page_number": "first",
        #"top_bucket": "first",
        "temp_line_id": "first",
        "page_width": "first",
        "page_height": "first",
        "is_sentence_like": "first",  # All words in a cell share same line → same value
        "sentence_score": "first",  # All words in a cell share same line → same value

        "x_left": "min",
        "x_right": "max",
        "y_top": "min",
        "y_bottom": "max",

        "char_count": "sum",
        "alpha_count": "sum",
        "digit_count": "sum",
        "uppercase_count": "sum",
        "word_count": "sum",
        "alpha_word_count": "sum",
        "capitalized_word_count": "sum",

        "bold_char_est": "sum",
        "italic_char_est": "sum",

        "font_name": _mode_or_first,
        "font_family": _mode_or_first,
        "font_size": _mode_or_first,
        "non_stroking_color": _mode_or_first,
        "stroking_color": _mode_or_first,
        "text_orientation": _mode_or_first,

        "word_id": _listify,
        "text": lambda s: " ".join(t for t in s.astype(str) if t),
    }

    grouped = df.groupby("cell_id", sort=True, observed=True).agg(agg_spec).reset_index()

    # Recompute geometry
    grouped["width"] = grouped["x_right"] - grouped["x_left"]
    grouped["height"] = grouped["y_bottom"] - grouped["y_top"]

    # Recompute ratios from estimated bold/italic chars
    total_chars = grouped["char_count"].replace(0, np.nan)
    grouped["bold_ratio"] = (grouped["bold_char_est"] / total_chars).fillna(0.0)
    grouped["italic_ratio"] = (grouped["italic_char_est"] / total_chars).fillna(0.0)

    # Rename word_id list → word_ids
    grouped = grouped.rename(columns={"word_id": "word_ids"})

    # Drop helper bold/italic estimates
    grouped = grouped.drop(columns=["bold_char_est", "italic_char_est"])

    return grouped


# ================================================================================
# ADD RELATIONSHIPS TO CELLS
# ================================================================================

# ---------------------
# Step 4: Add link relationships to df_cells
# ---------------------

def _calculate_bbox_overlap_ratio_vectorized(
    x_left_a: np.ndarray, x_right_a: np.ndarray, y_top_a: np.ndarray, y_bottom_a: np.ndarray,
    x_left_b: float, x_right_b: float, y_top_b: float, y_bottom_b: float,
) -> np.ndarray:
    """
    Calculate the overlap ratio between multiple boxes A and a single box B.
    Vectorized for speed.
    
    Returns the ratio of intersection area to the area of each box A.
    If there's no overlap, returns 0.0 for that box.
    
    Args:
        x_left_a, x_right_a, y_top_a, y_bottom_a: Arrays for boxes A (cells)
        x_left_b, x_right_b, y_top_b, y_bottom_b: Scalars for box B (link/rect)
    
    Returns:
        Array of overlap ratios (same length as input arrays)
    """
    # Calculate intersection (vectorized)
    x_left_inter = np.maximum(x_left_a, x_left_b)
    x_right_inter = np.minimum(x_right_a, x_right_b)
    y_top_inter = np.maximum(y_top_a, y_top_b)
    y_bottom_inter = np.minimum(y_bottom_a, y_bottom_b)
    
    # Check if there's actual overlap
    has_overlap = (x_left_inter < x_right_inter) & (y_top_inter < y_bottom_inter)
    
    # Calculate areas
    inter_width = np.where(has_overlap, x_right_inter - x_left_inter, 0.0)
    inter_height = np.where(has_overlap, y_bottom_inter - y_top_inter, 0.0)
    intersection_area = inter_width * inter_height
    
    # Area of boxes A
    area_a = (x_right_a - x_left_a) * (y_bottom_a - y_top_a)
    
    # Avoid division by zero
    ratios = np.where(area_a > 0, intersection_area / area_a, 0.0)
    
    return ratios


def _add_link_relationships(
    df_cells: pd.DataFrame,
    df_links: pd.DataFrame,
    min_overlap_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Add link features to cells by finding bbox overlaps (vectorized).
    
    For each cell:
    - Find all links on the same page that overlap with it
    - Calculate overlap ratio for each (vectorized per page)
    - Assign the link with the largest overlap (if >= min_overlap_ratio)
    
    This is much faster than the old word_enhancer approach because:
    - Uses vectorized overlap calculations per page
    - Minimizes row-by-row iterations
    
    Args:
        df_cells: DataFrame with cell data (must have: page_number, x_left, x_right, y_top, y_bottom)
        df_links: DataFrame with link data (must have: page_number, x_left, x_right, y_top, y_bottom, link_type, link_url, link_dest)
        min_overlap_ratio: Minimum overlap ratio to consider a match (0.0 to 1.0)
    
    Returns:
        DataFrame with added columns: has_link, link_url, link_dest, link_type
    """
    out = df_cells.copy()
    
    # Initialize link columns with defaults
    out["has_link"] = False
    out["link_url"] = None
    out["link_dest"] = None
    out["link_type"] = None
    
    # If no links, return early
    if df_links.empty:
        return out
    
    # Process page by page for efficiency
    for page_num in out["page_number"].unique():
        # Get cells and links for this page
        cell_mask = out["page_number"] == page_num
        link_mask = df_links["page_number"] == page_num
        
        page_links = df_links[link_mask]
        
        if page_links.empty:
            continue
        
        # Get cells on this page - convert to numpy arrays for speed
        page_cell_indices = out.index[cell_mask].tolist()
        cell_x_left = out.loc[page_cell_indices, "x_left"].values
        cell_x_right = out.loc[page_cell_indices, "x_right"].values
        cell_y_top = out.loc[page_cell_indices, "y_top"].values
        cell_y_bottom = out.loc[page_cell_indices, "y_bottom"].values
        
        # For each link, vectorized check overlap with all cells on page
        best_overlap_ratios = np.zeros(len(page_cell_indices))
        best_link_indices = np.full(len(page_cell_indices), -1, dtype=np.int64)
        
        for link_idx, link in page_links.iterrows():
            # Vectorized overlap calculation for all cells on this page
            overlap_ratios = _calculate_bbox_overlap_ratio_vectorized(
                cell_x_left, cell_x_right, cell_y_top, cell_y_bottom,
                link["x_left"], link["x_right"], link["y_top"], link["y_bottom"],
            )
            
            # Update best matches
            better_mask = overlap_ratios > best_overlap_ratios
            best_overlap_ratios = np.where(better_mask, overlap_ratios, best_overlap_ratios)
            best_link_indices = np.where(better_mask, link_idx, best_link_indices)
        
        # Assign links to cells that have sufficient overlap
        for i, cell_idx in enumerate(page_cell_indices):
            if best_overlap_ratios[i] >= min_overlap_ratio and best_link_indices[i] >= 0:
                best_link = df_links.loc[best_link_indices[i]]
                
                out.at[cell_idx, "has_link"] = True
                out.at[cell_idx, "link_type"] = best_link["link_type"]
                out.at[cell_idx, "link_url"] = best_link.get("link_url")
                out.at[cell_idx, "link_dest"] = best_link.get("link_dest")
    
    return out


# ---------------------
# Step 5: Add rectangle relationships to df_cells
# ---------------------

def _add_rect_relationships(
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add rectangle shape relationships to cells (vectorized).
    
    For each cell, check if it falls completely inside a rect shape.
    If so, assign the rect's properties as background properties.
    
    This is faster than the old word_enhancer approach because:
    - Uses vectorized containment checks per page
    - Processes all cells on a page at once for each rectangle
    
    Args:
        df_cells: DataFrame with cell data (must have: page_number, x_left, x_right, y_top, y_bottom)
        df_shapes: DataFrame with shape data (must have: shape_type='rect')
    
    Returns:
        DataFrame with added columns: inside_rect_shape, background_*_color, shape_id_container
    """
    out = df_cells.copy()
    
    # Initialize columns
    out["inside_rect_shape"] = False
    out["background_non_stroking_color"] = None
    out["background_stroking_color"] = None
    out["shape_id_container"] = None
    
    if df_shapes.empty:
        return out
    
    # Filter to only rect shapes
    rect_shapes = df_shapes[df_shapes["shape_type"] == "rect"].copy()
    
    if rect_shapes.empty:
        return out
    
    # Process page by page
    for page_num in out["page_number"].unique():
        cell_mask = out["page_number"] == page_num
        rect_mask = rect_shapes["page_number"] == page_num
        
        page_rects = rect_shapes[rect_mask]
        
        if page_rects.empty:
            continue
        
        # Get cells on this page - convert to numpy arrays for speed
        page_cell_indices = out.index[cell_mask].tolist()
        cell_x_left = out.loc[page_cell_indices, "x_left"].values
        cell_x_right = out.loc[page_cell_indices, "x_right"].values
        cell_y_top = out.loc[page_cell_indices, "y_top"].values
        cell_y_bottom = out.loc[page_cell_indices, "y_bottom"].values
        
        # For each rectangle, vectorized check if cells are inside
        for rect_idx, rect in page_rects.iterrows():
            # Vectorized containment check for all cells on this page
            cells_inside = (
                (cell_x_left >= rect["x_left"]) &
                (cell_x_right <= rect["x_right"]) &
                (cell_y_top >= rect["y_top"]) &
                (cell_y_bottom <= rect["y_bottom"])
            )
            
            # Assign rect properties to cells that are inside
            # (Only assign if not already inside another rect - first rect wins)
            for i, cell_idx in enumerate(page_cell_indices):
                if cells_inside[i] and not out.at[cell_idx, "inside_rect_shape"]:
                    out.at[cell_idx, "inside_rect_shape"] = True
                    out.at[cell_idx, "background_non_stroking_color"] = rect.get("non_stroking_color")
                    out.at[cell_idx, "background_stroking_color"] = rect.get("stroking_color")
                    out.at[cell_idx, "shape_id_container"] = rect.get("shape_id")
    
    return out


# ---------------------
# Add vertical line relationships to cells
# ---------------------

def add_vertical_line_relationships_to_cells(
    cells_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add vertical line relationships to cells.

    For each cell, we compute its vertical center:
        center_y = (y_top + y_bottom) / 2

    Then, on the same page, we check if that center falls within the vertical
    range [y_top, y_bottom] of any *vertical* line shapes.

    X-position does NOT matter for this relationship (pure vertical alignment).

    Columns added to cells_df:
        - has_vertical_line: bool
        -   shape_id_vertical_line: list[int] or None if no lines match
    """
    if shapes_df.empty:
        out = cells_df.copy()
        out["has_vertical_line"] = False
        out["shape_id_vertical_line"] = None
        return out

    out = cells_df.copy()

    # Normalized page column name (cells use 'page_number' in your pipeline)
    page_col = "page_number"
    if page_col not in out.columns:
        raise ValueError(f"Expected '{page_col}' column in cells_df")

    if "page_number" not in shapes_df.columns and "page_num" in shapes_df.columns:
        shapes = shapes_df.rename(columns={"page_num": "page_number"}).copy()
    else:
        shapes = shapes_df.copy()

    # Filter to vertical line shapes
    v_lines = shapes[
        (shapes.get("shape_type") == "line")
        & (shapes.get("orientation") == "vertical")
    ].copy()

    # Initialize output columns
    out["has_vertical_line"] = False
    out["shape_id_vertical_line"] = None

    if v_lines.empty:
        return out

    # Ensure numeric y_top / y_bottom for both cells and shapes
    for col in ("y_top", "y_bottom"):
        out[col] = out[col].astype(float, copy=False)
        v_lines[col] = v_lines[col].astype(float, copy=False)

    # Precompute center_y per cell
    out_center_y = (out["y_top"].to_numpy() + out["y_bottom"].to_numpy()) / 2.0

    # We'll operate by page to keep things small & cache-friendly
    all_pages = out[page_col].unique()

    for page in all_pages:
        cell_mask = out[page_col] == page
        line_mask = v_lines["page_number"] == page

        page_lines = v_lines[line_mask]
        if page_lines.empty:
            continue

        # Positions of these cells in the full DataFrame
        cell_indices = np.where(cell_mask.to_numpy())[0]
        if cell_indices.size == 0:
            continue

        # Center y for these cells
        center_y_page = out_center_y[cell_indices]

        # Line ranges as numpy arrays
        line_y_top = page_lines["y_top"].to_numpy()
        line_y_bottom = page_lines["y_bottom"].to_numpy()
        line_ids = page_lines["shape_id"].to_numpy(dtype=int)

        # Prepare storage: a list for each cell on this page
        matches_per_cell = [[] for _ in range(cell_indices.size)]

        # For each vertical line, find all cells whose center_y lies within [top, bottom]
        for j in range(line_ids.size):
            lt = line_y_top[j]
            lb = line_y_bottom[j]
            lid = line_ids[j]

            in_range = (center_y_page >= lt) & (center_y_page <= lb)
            if not np.any(in_range):
                continue

            matched_positions = np.where(in_range)[0]
            for pos in matched_positions:
                matches_per_cell[pos].append(lid)

        # Write back to DataFrame
        for local_i, global_idx in enumerate(cell_indices):
            ids = matches_per_cell[local_i]
            if ids:
                # force Python ints
                ids = [int(x) for x in ids]

                out.at[global_idx, "has_vertical_line"] = True
                out.at[global_idx, "shape_id_vertical_line"] = ids


    return out


# ================================================================================
# ASSIGN CELLS TO HORIZONTAL BANDS
# ================================================================================

# ---------------------
# Horizontal Band Assignment Config
# ---------------------

def _interpolate_gap_multiplier(
    median_gap: float,
    low_gap: float = 3.0,
    high_gap: float = 10.0,
    mult_at_low: float = 1.60,
    mult_at_high: float = 1.10,
) -> float:
    """
    Returns an adaptive gap multiplier based on the page's median gap.

    - if median_gap <= low_gap  → mult_at_low
    - if median_gap >= high_gap → mult_at_high
    - else linear interpolation between them
    """
    if median_gap <= low_gap:
        return mult_at_low
    if median_gap >= high_gap:
        return mult_at_high

    # linear interpolation
    t = (median_gap - low_gap) / (high_gap - low_gap)
    return mult_at_low + t * (mult_at_high - mult_at_low)


# ---------------------
# Helpers
# ---------------------

def _renumber_band_ids_monotonic(df: pd.DataFrame) -> None:
    """
    Renumber band IDs to be monotonically increasing (in-place modification).
    
    After merging bands by vertical lines, band IDs may have jumps or be out of order.
    This function renumbers them sequentially while maintaining vertical order (top to bottom).
    
    Args:
        df: DataFrame with horizontal_band_id column
    
    Modifies df in-place by updating horizontal_band_id column.
    """
    if df.empty or "horizontal_band_id" not in df.columns:
        return
    
    # Track the next available global band ID
    next_global_band_id = 1
    
    # Process per page
    for (page_num), page_cells in df.groupby(
        ["page_number"], sort=False
    ):
        page_mask = (df["page_number"] == page_num)
        page_indices = df.index[page_mask]
        
        if page_indices.empty:
            continue
        
        # Get unique band IDs on this page with their minimum y_top (for vertical ordering)
        band_info = []
        unique_bands = df.loc[page_indices, "horizontal_band_id"].dropna().unique()
        
        for band_id in unique_bands:
            if band_id < 0:
                continue
            
            band_mask = page_mask & (df["horizontal_band_id"] == band_id)
            band_indices = df.index[band_mask]
            
            if band_indices.empty:
                continue
            
            # Get the minimum y_top for this band (top-most cell)
            min_y_top = df.loc[band_indices, "y_top"].min()
            band_info.append((band_id, min_y_top))
        
        # Sort bands by vertical position (top to bottom)
        band_info.sort(key=lambda x: x[1])
        
        # Create mapping from old band ID to new sequential ID
        band_id_mapping = {}
        for old_band_id, _ in band_info:
            band_id_mapping[old_band_id] = next_global_band_id
            next_global_band_id += 1
        
        # Apply the new band IDs
        for idx in page_indices:
            old_band_id = df.at[idx, "horizontal_band_id"]
            if pd.notna(old_band_id) and old_band_id >= 0:
                old_band_id = int(old_band_id)
                if old_band_id in band_id_mapping:
                    df.at[idx, "horizontal_band_id"] = band_id_mapping[old_band_id]


def _merge_bands_by_shared_vertical_lines(
    df: pd.DataFrame,
    min_shared_lines: int,
) -> None:
    """
    Merge bands that share vertical lines (in-place modification).
    
    For each page, finds bands that share at least `min_shared_lines` 
    vertical line IDs and merges them to use the same band_id (the smallest one).
    
    Args:
        df: DataFrame with horizontal_band_id and shape_id_vertical_line columns
        min_shared_lines: Minimum number of shared vertical line IDs to merge bands
    
    Modifies df in-place by updating horizontal_band_id column.
    """
    if df.empty or "horizontal_band_id" not in df.columns:
        return
    
    # Process per page
    for (page_num), page_cells in df.groupby(
        ["page_number"], sort=False
    ):
        # Get unique bands on this page
        page_mask = (df["page_number"] == page_num)
        page_indices = df.index[page_mask]
        
        # Build a mapping: band_id -> set of all vertical line IDs in that band
        band_to_vlines: Dict[int, set] = {}
        
        for idx in page_indices:
            band_id = df.at[idx, "horizontal_band_id"]
            if pd.isna(band_id) or band_id < 0:
                continue
            
            band_id = int(band_id)
            vlines = df.at[idx, "shape_id_vertical_line"]
            
            # Skip cells with no vertical lines
            if vlines is None or (isinstance(vlines, float) and pd.isna(vlines)):
                continue
            
            # Ensure vlines is a list
            if not isinstance(vlines, list):
                vlines = [vlines]
            
            # Add to band's vertical line set
            if band_id not in band_to_vlines:
                band_to_vlines[band_id] = set()
            band_to_vlines[band_id].update(vlines)
        
        # Find bands that should merge (union-find structure)
        band_ids = sorted(band_to_vlines.keys())
        if len(band_ids) < 2:
            continue  # Nothing to merge
        
        # parent[band_id] = the root band_id of its merge group
        parent = {bid: bid for bid in band_ids}
        
        def find_root(bid: int) -> int:
            """Find root of merge group with path compression."""
            if parent[bid] != bid:
                parent[bid] = find_root(parent[bid])
            return parent[bid]
        
        def union(bid1: int, bid2: int) -> None:
            """Merge two bands (use smaller ID as root)."""
            root1 = find_root(bid1)
            root2 = find_root(bid2)
            if root1 != root2:
                # Use smaller ID as parent
                if root1 < root2:
                    parent[root2] = root1
                else:
                    parent[root1] = root2
        
        # Compare all pairs of bands
        for i in range(len(band_ids)):
            for j in range(i + 1, len(band_ids)):
                bid1 = band_ids[i]
                bid2 = band_ids[j]
                
                # Count shared vertical lines
                shared = band_to_vlines[bid1] & band_to_vlines[bid2]
                
                if len(shared) >= min_shared_lines:
                    union(bid1, bid2)
        
        # Apply the merges: reassign band_ids to their roots
        for idx in page_indices:
            band_id = df.at[idx, "horizontal_band_id"]
            if pd.isna(band_id) or band_id < 0:
                continue
            
            band_id = int(band_id)
            if band_id in parent:
                new_band_id = find_root(band_id)
                df.at[idx, "horizontal_band_id"] = new_band_id


# ---------------------
# Step 6: Assign cells to horizontal bands
# ---------------------

def _assign_horizontal_bands(
    df_cells: pd.DataFrame,
    min_shared_vertical_lines: int = 1,
) -> pd.DataFrame:
    """
    Group cells into horizontal content bands per (page_number).
    
    Uses temp_line_id groups as the base unit - calculates gaps between
    consecutive temp_line groups and clusters them into bands.
    
    Additionally merges bands that share vertical lines:
    - If cells in different bands share at least `min_shared_vertical_lines`
      vertical line IDs, those bands are merged together.
    - After merging, band IDs are renumbered to be monotonically increasing
      in vertical order (top to bottom) while remaining globally unique.
    
    Args:
        df_cells: DataFrame with cell data
        min_shared_vertical_lines: Minimum number of shared vertical line IDs
            required to merge bands (default: 1)
    
    Adds columns to df_cells:
        - horizontal_band_id: int (unique and monotonically increasing per page, top to bottom)
        - page_median_gap: float (median gap between temp_line groups on page)
        - page_gap_thresh: float (threshold used to split bands on page)
        - line_gap: float (actual vertical gap between this line and the previous line)
    
    Returns:
        DataFrame with band columns added
    """
    if df_cells.empty:
        df_cells["horizontal_band_id"] = pd.Series(dtype="int64")
        df_cells["page_median_gap"] = pd.Series(dtype="float64")
        df_cells["page_gap_thresh"] = pd.Series(dtype="float64")
        df_cells["line_gap"] = pd.Series(dtype="float64")
        return df_cells
    
    df = df_cells.copy()
    
    # Ensure proper sort order
    df = df.sort_values(
        ["page_number", "y_top", "x_left"]
    ).reset_index(drop=True)
    
    # Initialize output columns
    df["horizontal_band_id"] = -1
    df["page_median_gap"] = np.nan
    df["page_gap_thresh"] = np.nan
    df["line_gap"] = np.nan
    
    global_band_id = 1
    
    # Process per page
    for (page_num), page_cells in df.groupby(
        ["page_number"], sort=False
    ):
        page_cells = page_cells.sort_values(["y_top", "x_left"])
        
        if page_cells.empty:
            continue
        
        # Group by temp_line_id to get representative rows per line
        # Each temp_line represents a horizontal row of cells
        line_groups = page_cells.groupby("temp_line_id", sort=False).agg({
            "y_top": "min",
            "y_bottom": "max",
        }).sort_values("y_top")
        
        if len(line_groups) == 0:
            continue
        
        # --- First pass: compute gaps between temp_lines + page median gap ---
        gaps: List[float] = []
        prev_y_bottom = None
        
        for _, line_row in line_groups.iterrows():
            if prev_y_bottom is None:
                prev_y_bottom = line_row["y_bottom"]
                continue
            
            gap = float(line_row["y_top"] - prev_y_bottom)
            
            if gap > 0:
                gaps.append(gap)
            
            prev_y_bottom = line_row["y_bottom"]
        
        # Calculate page statistics
        if gaps:
            median_gap = float(np.median(gaps))
            adaptive_multiplier = _interpolate_gap_multiplier(median_gap)
            gap_thresh = adaptive_multiplier * median_gap
        else:
            median_gap = 0.0
            gap_thresh = float("inf")
        
        # Attach per-page stats to all cells on this page
        df.loc[page_cells.index, "page_median_gap"] = median_gap
        df.loc[page_cells.index, "page_gap_thresh"] = gap_thresh
        
        # --- Second pass: walk temp_lines and assign bands ---
        current_band_temp_line_ids: List[int] = []
        prev_y_bottom = None
        
        def _flush_current_band():
            nonlocal global_band_id, current_band_temp_line_ids
            
            if not current_band_temp_line_ids:
                return
            
            # Assign band ID to all cells in these temp_lines
            band_mask = page_cells["temp_line_id"].isin(current_band_temp_line_ids)
            band_cell_indices = page_cells[band_mask].index
            
            df.loc[band_cell_indices, "horizontal_band_id"] = global_band_id
            
            global_band_id += 1
            current_band_temp_line_ids = []
        
        # Iterate through temp_line groups in vertical order
        for temp_line_id, line_row in line_groups.iterrows():
            y_top = float(line_row["y_top"])
            y_bottom = float(line_row["y_bottom"])
            
            if prev_y_bottom is None:
                # Start first band on this page
                # First line has no previous line, so line_gap is NaN (already initialized)
                current_band_temp_line_ids = [temp_line_id]
                prev_y_bottom = y_bottom
                continue
            
            gap = float(y_top - prev_y_bottom)
            
            # Assign this gap to all cells in this temp_line
            line_mask = page_cells["temp_line_id"] == temp_line_id
            line_cell_indices = page_cells[line_mask].index
            df.loc[line_cell_indices, "line_gap"] = gap
            
            if gap > gap_thresh:
                # Big vertical valley → close current band, start a new one
                _flush_current_band()
                current_band_temp_line_ids = [temp_line_id]
            else:
                # Same band
                current_band_temp_line_ids.append(temp_line_id)
            
            prev_y_bottom = y_bottom
        
        # Flush last band on this page
        _flush_current_band()
    
    # --- Post-processing: merge bands that share vertical lines ---
    if min_shared_vertical_lines > 0 and "shape_id_vertical_line" in df.columns:
        _merge_bands_by_shared_vertical_lines(df, min_shared_vertical_lines)
    
    # --- Renumber band IDs to be monotonically increasing ---
    _renumber_band_ids_monotonic(df)
    
    # Re-sort to logical order
    df = df.sort_values(
        ["page_number", "cell_id"]
    ).reset_index(drop=True)
    
    return df


# ================================================================================
# Public API
# ================================================================================

def build_cells(
    df_words: pd.DataFrame,
    df_shapes: Optional[pd.DataFrame] = None,
    df_links: Optional[pd.DataFrame] = None,
    min_shared_vertical_lines: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Public API used by the PDF pipeline.

    Parameters
    ----------
    df_words : pd.DataFrame
        Word-level dataframe from the previous PDF extraction step.
        Must contain at least:
            page_number, word_id, text,
            x_left, x_right, y_top, y_bottom,
            font_size, bold_ratio, italic_ratio,
            char_count, alpha_count, digit_count, uppercase_count,
            word_count, alpha_word_count, capitalized_word_count,
            page_width, page_height

    df_shapes : pd.DataFrame, optional
        Shape dataframe (from step_04_shape_merger).
        If provided, will add rectangle and vertical line relationships to cells.

    df_links : pd.DataFrame, optional
        Link dataframe (from step_03_link_extractor).
        If provided, will add link relationships to cells.

    min_shared_vertical_lines : int, optional
        Minimum number of shared vertical line IDs required to merge bands.
        Default is 1. Set to 0 to disable vertical line merging.

    Returns
    -------
    df_cells : pd.DataFrame
        One row per cell, with aggregated geometry, stats, and relationships.
        Includes columns: 
            - Relationships: has_link, link_url, link_dest, link_type, inside_rect_shape, background_*_color
            - Bands: horizontal_band_id, page_median_gap, page_gap_thresh
    df_words_out : pd.DataFrame
        The input df_words, sorted and augmented with:
            top_bucket, temp_line_id, cell_id
    """
    # Work on a copy to avoid mutating caller's DataFrame in-place
    df = df_words.copy()

    ## TODO: add support for vertical text (TTB / BTT)

    # --- NEW: mask out vertical text (TTB / BTT) from cell-building ---
    if "text_orientation" in df.columns:
        vertical_mask = df["text_orientation"].isin(["TTB", "BTT"])
    else:
        vertical_mask = pd.Series(False, index=df.index)

    df_vert = df[vertical_mask].copy()        # kept unassigned
    df_horiz = df[~vertical_mask].copy()      # only these go through pipeline

    if df.empty:
        # Return empty frames with expected columns
        df_cells = _build_cells_df(df.assign(#top_bucket=pd.Series(dtype="float64"),
                                             temp_line_id=pd.Series(dtype="int64"),
                                             cell_id=pd.Series(dtype="int64")))
        return df_cells, df

    # Step 1: assign top_bucket + temp_line_id
    #df_horiz = _assign_top_buckets_and_lines(df_horiz)
    df_horiz = assign_line_id(df_horiz)
    df_horiz = df_horiz.rename(columns={"line_id": "temp_line_id"})

    # Step 2: assign cell_id within each temp_line_id
    df_horiz = _assign_cell_ids(df_horiz)

    # Step 3: aggregate to cell-level df
    df_cells = _build_cells_df(df_horiz)

    # Step 4: add link relationships (if df_links provided)
    if df_links is not None and not df_links.empty:
        df_cells = _add_link_relationships(df_cells, df_links)

    # Step 5: add rectangle relationships (if df_shapes provided)
    if df_shapes is not None and not df_shapes.empty:
        df_cells = _add_rect_relationships(df_cells, df_shapes)

    # Step 6: assign vertical line relationships BEFORE horizontal bands (if df_shapes provided)
    # This allows bands to be merged based on shared vertical lines
    if df_shapes is not None and not df_shapes.empty:
        df_cells = add_vertical_line_relationships_to_cells(df_cells, df_shapes)

    # Step 7: assign cells to horizontal bands (uses vertical line info for merging)
    df_cells = _assign_horizontal_bands(df_cells, min_shared_vertical_lines=min_shared_vertical_lines)

    # Step 8: assign cell underlines (if df_shapes provided)
    if df_shapes is not None and not df_shapes.empty:
        df_cells, df_shapes = assign_cell_underlines(df_cells, df_shapes)

    # Recombine words: horizontals (with IDs) + verticals (unassigned)
    df_words_out = pd.concat(
        [df_horiz, df_vert],
        axis=0,
        ignore_index=True
    ).sort_values(
        ["page_number", "y_top", "x_left"],
        kind="mergesort"
    ).reset_index(drop=True)

    return df_cells, df_words_out

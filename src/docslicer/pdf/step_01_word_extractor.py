"""
Step 01 – Raw word extraction (PyMuPDF version, semi-vectorized)

Responsibility:
    - Open a PDF with PyMuPDF (fitz)
    - Extract raw word tokens via `page.get_text("words")`
    - Attach geometry + a representative span's raw style info
      (font_name, font_size, non_stroking_color, stroking_color)
    - Convert all color values to hex format (#rrggbb)
    - NO high-level features (no bold/italic guesses, no ratios, etc.)

Output columns (per row), matching a *subset* of WordSchema:
    page_number
    word_id
    text

    x_left, y_top, x_right, y_bottom
    width, height
    page_width, page_height

    font_name
    font_size
    non_stroking_color (hex string: #rrggbb or None)
    stroking_color (hex string: #rrggbb or None)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings
import sys
import os
from contextlib import contextmanager

import fitz  # PyMuPDF
import pandas as pd
import numpy as np

from .._utils.color_utils import pdf_color_to_hex
from .._utils.text_utils import add_calculated_text_features


@contextmanager
def _suppress_pdf_warnings():
    """
    Suppress warnings from PDF libraries about invalid color values.
    
    Many PDFs have malformed color definitions (e.g., name objects like '/P189'
    instead of numeric values). These cause warnings in the underlying PDF library
    that we can't fix and don't affect our extraction.
    """
    # Save stderr
    old_stderr = sys.stderr
    
    try:
        # Redirect stderr to devnull to suppress PDF library warnings
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            
            # Also suppress Python warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Cannot set.*color.*invalid.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                yield
    finally:
        # Restore stderr
        sys.stderr = old_stderr


# ==================================================
# Build the Span DF, equivalent to Char in pdfplumber
# ==================================================

def _build_span_df(page) -> pd.DataFrame:
    """
    Build a DataFrame with one row per text span on the page.

    Uses page.get_text("rawdict") → blocks → lines → spans.
    We treat spans as our style carriers (font, size, color, direction).
    """
    raw = page.get_text("rawdict")
    if not raw or "blocks" not in raw:
        return pd.DataFrame()

    span_rows: List[Dict[str, Any]] = []

    for block in raw["blocks"]:
        # Text blocks only
        if block.get("type", 0) != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = bbox

                # --- derive direction from chars, if possible ---
                dx = dy = 0.0
                chars = span.get("chars")  # list of char dicts, if present

                if chars and len(chars) >= 2:
                    # first + last char centers
                    c0 = chars[0]
                    c1 = chars[-1]

                    bx0, by0, bx1, by1 = c0["bbox"]
                    cx0 = (bx0 + bx1) / 2.0
                    cy0 = (by0 + by1) / 2.0

                    bx0, by0, bx1, by1 = c1["bbox"]
                    cx1 = (bx0 + bx1) / 2.0
                    cy1 = (by0 + by1) / 2.0

                    dx = float(cx1 - cx0)
                    dy = float(cy1 - cy0)

                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm > 0:
                        dx /= norm
                        dy /= norm
                    else:
                        dx = 1.0
                        dy = 0.0
                else:
                    # fallback to span["dir"] if chars missing or degenerate
                    dir_vec = span.get("dir")
                    if dir_vec and len(dir_vec) == 2:
                        ddx, ddy = map(float, dir_vec)
                        norm = (ddx * ddx + ddy * ddy) ** 0.5
                        if norm > 0:
                            dx = ddx / norm
                            dy = ddy / norm
                        else:
                            dx = 1.0
                            dy = 0.0
                    else:
                        dx, dy = 1.0, 0.0  # default LTR

                # Convert color to hex format (handles all formats and invalid values)
                non_stroking_color_hex = pdf_color_to_hex(span.get("color"))
                stroking_color_hex = pdf_color_to_hex(span.get("stroke_color"))

                span_rows.append(
                    {
                        "x0": float(x0),
                        "top": float(y0),
                        "x1": float(x1),
                        "bottom": float(y1),
                        "fontname": span.get("font"),
                        "size": span.get("size"),
                        "non_stroking_color": non_stroking_color_hex,
                        "stroking_color": stroking_color_hex,
                        "dir_x": dx,
                        "dir_y": dy,
                        "wmode": span.get("wmode", 0),
                        "text": span.get("text", ""),
                    }
                )

    if not span_rows:
        return pd.DataFrame()

    df = pd.DataFrame(span_rows)

    # Compute centers (vectorized)
    df["cx"] = (df["x0"] + df["x1"]) / 2.0
    df["cy"] = (df["top"] + df["bottom"]) / 2.0

    # Give spans a stable ID for debugging
    #df["span_id"] = range(1, len(df) + 1)

    return df


# ==================================================
# Build the Words DF, equivalent to Word in pdfplumber
# ==================================================

def _build_words_df(words: List[Tuple[Any, ...]]) -> pd.DataFrame:
    """
    Build a DataFrame from PyMuPDF's 'words' list.

    PyMuPDF page.get_text("words") tuples:
        (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    """
    if not words:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for w in words:
        # Defensive: allow len 5 or 8 (older vs newer versions)
        x0, y0, x1, y1, text = w[:5]
        rows.append(
            {
                "x_left": float(x0),
                "x_right": float(x1),
                "y_top": float(y0),
                "y_bottom": float(y1),
                "text": text or "",
            }
        )

    df = pd.DataFrame(rows)

    # Sort by (y_top, x_left) to mimic pdfplumber behavior
    df = df.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    # Geometry (vectorized)
    df["width"] = df["x_right"] - df["x_left"]
    df["height"] = df["y_bottom"] - df["y_top"]

    # Normalize text
    df["text"] = df["text"].fillna("").astype(str)

    return df


# ==================================================
# Vectorized word-span matching, for Style parameters
# ==================================================

def _orientation_from_dir(dx: float, dy: float) -> str:
    """
    Map a direction vector (dx, dy) to a coarse orientation label.

    dx, dy are from the first→last character centers.
    PDF coords: y increases downward.
    """
    THRESH = 0.5

    # Decide whether it's more horizontal or vertical
    if abs(dx) >= abs(dy):
        # Mostly horizontal
        if dx >= THRESH:
            return "LTR"
        if dx <= -THRESH:
            return "RTL"
    else:
        # Mostly vertical
        if dy >= THRESH:
            # y increasing down → visually top-to-bottom
            return "TTB"
        if dy <= -THRESH:
            # y decreasing up → visually bottom-to-top
            return "BTT"

    return "UNKNOWN"



def _attach_representative_spans(
    words_df: pd.DataFrame,
    span_df: pd.DataFrame,
    eps: float = 0.5,
) -> pd.DataFrame:
    """
    For each word, pick a representative span and attach its style columns
    + text_orientation.

    Strategy (per word):
        1. Find spans whose bbox intersects the word bbox (with eps tolerance).
        2. For those spans, compute:
           - intersection area / word area
           - whether span height is comparable to word height
           - whether span is vertical (|dy| > |dx|)
        3. Build a score from these features and choose the max.
        4. If *no* spans intersect, fall back to globally nearest span center.

    This avoids tiny decorative spans stealing matches from the big vertical
    TOC spans that actually represent the text.
    """
    if words_df.empty or span_df.empty:
        out = words_df.copy()
        for col in ("font_name", "font_size", "non_stroking_color", "stroking_color", "text_orientation"):
            out[col] = None
        return out

    # Word geometry (W,)
    W_x0 = words_df["x_left"].to_numpy(dtype=np.float64)
    W_x1 = words_df["x_right"].to_numpy(dtype=np.float64)
    W_y0 = words_df["y_top"].to_numpy(dtype=np.float64)
    W_y1 = words_df["y_bottom"].to_numpy(dtype=np.float64)

    # Span geometry and style (S,)
    S_x0    = span_df["x0"].to_numpy(dtype=np.float64)
    S_x1    = span_df["x1"].to_numpy(dtype=np.float64)
    S_y0    = span_df["top"].to_numpy(dtype=np.float64)
    S_y1    = span_df["bottom"].to_numpy(dtype=np.float64)
    S_cx    = span_df["cx"].to_numpy(dtype=np.float64)
    S_cy    = span_df["cy"].to_numpy(dtype=np.float64)
    S_dir_x = span_df["dir_x"].to_numpy(dtype=np.float64)
    S_dir_y = span_df["dir_y"].to_numpy(dtype=np.float64)
    S_font  = span_df["fontname"].to_numpy()
    S_size  = span_df["size"].to_numpy()
    S_ns    = span_df["non_stroking_color"].to_numpy()
    S_sc    = span_df["stroking_color"].to_numpy()

    # Word metrics (W,)
    W_cx     = (W_x0 + W_x1) * 0.5
    W_cy     = (W_y0 + W_y1) * 0.5
    w_height = np.maximum(W_y1 - W_y0, 1e-6)
    w_area   = np.maximum(W_x1 - W_x0, 1e-6) * w_height

    # Intersection mask (W, S)
    intersects = (
        (S_x0[np.newaxis, :] <= W_x1[:, np.newaxis] + eps)
        & (S_x1[np.newaxis, :] >= W_x0[:, np.newaxis] - eps)
        & (S_y0[np.newaxis, :] <= W_y1[:, np.newaxis] + eps)
        & (S_y1[np.newaxis, :] >= W_y0[:, np.newaxis] - eps)
    )

    # Intersection area (W, S)
    inter_area = (
        np.maximum(0.0, np.minimum(S_x1[np.newaxis, :], W_x1[:, np.newaxis])
                        - np.maximum(S_x0[np.newaxis, :], W_x0[:, np.newaxis]))
        * np.maximum(0.0, np.minimum(S_y1[np.newaxis, :], W_y1[:, np.newaxis])
                          - np.maximum(S_y0[np.newaxis, :], W_y0[:, np.newaxis]))
    )

    # Span-level features, broadcast over words
    span_h        = S_y1 - S_y0                                              # (S,)
    height_ok     = span_h[np.newaxis, :] >= 0.7 * w_height[:, np.newaxis]  # (W, S)
    vertical_flag = np.abs(S_dir_y) > np.abs(S_dir_x)                       # (S,)

    # Intersection score (W, S); non-intersecting entries → -inf
    score = np.where(
        intersects,
        inter_area / w_area[:, np.newaxis]
            + 0.5 * height_ok
            + 0.2 * vertical_flag[np.newaxis, :],
        -np.inf,
    )

    # Words that have ≥1 intersecting span with a positive score use intersection
    # scoring; all others (no intersection or only zero-area touches) fall back to
    # nearest-centre matching — preserving the original per-word logic exactly.
    use_intersection = intersects.any(axis=1) & (score > 0).any(axis=1)  # (W,)

    # Fallback: negative squared distance so argmax gives the nearest span centre
    dist2 = (
        (S_cx[np.newaxis, :] - W_cx[:, np.newaxis]) ** 2
        + (S_cy[np.newaxis, :] - W_cy[:, np.newaxis]) ** 2
    )

    final_score = np.where(use_intersection[:, np.newaxis], score, -dist2)
    best = final_score.argmax(axis=1)  # (W,) — positional indices into span arrays

    # Vectorised orientation label from direction vectors
    dx, dy = S_dir_x[best], S_dir_y[best]
    mostly_horiz = np.abs(dx) >= np.abs(dy)
    THRESH = 0.5
    text_orientation = np.select(
        [
            mostly_horiz & (dx >= THRESH),
            mostly_horiz & (dx <= -THRESH),
            ~mostly_horiz & (dy >= THRESH),
            ~mostly_horiz & (dy <= -THRESH),
        ],
        ["LTR", "RTL", "TTB", "BTT"],
        default="UNKNOWN",
    )

    out = words_df.copy()
    out["font_name"]          = S_font[best]
    out["font_size"]          = S_size[best]
    out["non_stroking_color"] = S_ns[best]
    out["stroking_color"]     = S_sc[best]
    out["text_orientation"]   = text_orientation
    return out


# ==================================================
# Per page orchestrator
# ==================================================

def _extract_raw_words_for_page(
    page,
    page_number: int,
    *,
    start_word_id: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Page extraction using PyMuPDF:

        - spans_df from rawdict (style carriers)
        - words_df from get_text("words")
        - attach representative span per word
        - return DataFrame with controlled column order
    """
    spans_df = _build_span_df(page)
    words = page.get_text("words")
    if not words:
        return pd.DataFrame(), start_word_id

    words_df = _build_words_df(words)
    words_df = _attach_representative_spans(words_df, spans_df)

    n_words = len(words_df)
    word_ids = range(start_word_id + 1, start_word_id + 1 + n_words)

    # Add IDs + context columns (vectorized)
    words_df = words_df.copy()
    words_df["word_id"] = list(word_ids)
    words_df["page_number"] = page_number
    words_df["page_width"] = float(page.rect.width)
    words_df["page_height"] = float(page.rect.height)

    next_word_id = start_word_id + n_words
    return words_df, next_word_id


# =============================
# Public API
# =============================

def extract_words(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract all words from a PDF and return a DataFrame with one row per word.

    Columns match a subset of WordSchema: page_number, word_id, text,
    x_left, x_right, y_top, y_bottom, width, height, font_name, font_size,
    non_stroking_color, stroking_color (both as hex #rrggbb or None),
    text_orientation, bold_ratio, italic_ratio, char_count, alpha_count,
    digit_count, uppercase_count, token_count, alpha_token_count,
    capitalized_token_count, page_width, page_height.

    DocumentIdentity fields (document_name, document_id) are intentionally
    excluded — the orchestrator adds those after extraction.
    """

    pdf_path = Path(pdf_path).expanduser().resolve()

    all_words_dfs: List[pd.DataFrame] = []
    next_word_id = 0

    # Suppress PDF parsing warnings about invalid color values
    with _suppress_pdf_warnings():
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count

            if pages_to_process is None:
                page_numbers = range(1, total_pages + 1)
            else:
                page_numbers = pages_to_process

            for page_number in page_numbers:
                if page_number < 1 or page_number > total_pages:
                    continue

                page = doc.load_page(page_number - 1)

                page_words_df, next_word_id = _extract_raw_words_for_page(
                    page,
                    page_number=page_number,
                    start_word_id=next_word_id,
                )
                if not page_words_df.empty:
                    all_words_dfs.append(page_words_df)

        if not all_words_dfs:
            return pd.DataFrame()

        df = pd.concat(all_words_dfs, ignore_index=True)
        
        # --------------------
        # Add calculated features (bold, italic, char, word, etc.)
        # --------------------
        df = add_calculated_text_features(df)
        
        return df

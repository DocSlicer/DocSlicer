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
        words_df = words_df.copy()
        words_df["font_name"] = None
        words_df["font_size"] = None
        words_df["non_stroking_color"] = None
        words_df["stroking_color"] = None
        words_df["text_orientation"] = None
        #words_df["debug_span_id"] = None
        return words_df

    fontnames: List[Optional[str]] = []
    font_sizes: List[Optional[float]] = []
    ns_colors: List[Optional[Any]] = []
    s_colors: List[Optional[Any]] = []
    orientations: List[Optional[str]] = []
    #chosen_span_ids: List[Optional[int]] = []

    # Local arrays for span geometry + style
    sx0 = span_df["x0"].values
    sx1 = span_df["x1"].values
    stop = span_df["top"].values
    sbot = span_df["bottom"].values
    scx = span_df["cx"].values
    scy = span_df["cy"].values
    s_font = span_df["fontname"].values
    s_size = span_df["size"].values
    s_ns = span_df["non_stroking_color"].values
    s_s = span_df["stroking_color"].values
    s_dir_x = span_df["dir_x"].values
    s_dir_y = span_df["dir_y"].values
    #s_span_id = span_df["span_id"].values

    span_index = span_df.index.to_numpy()

    for _, w in words_df.iterrows():
        x_left = float(w["x_left"])
        x_right = float(w["x_right"])
        y_top = float(w["y_top"])
        y_bottom = float(w["y_bottom"])

        # word center + size
        w_cx = (x_left + x_right) / 2.0
        w_cy = (y_top + y_bottom) / 2.0
        w_width = max(x_right - x_left, 1e-6)
        w_height = max(y_bottom - y_top, 1e-6)
        w_area = w_width * w_height

        # --- primary: spans whose bbox intersects the word bbox ---
        mask = (
            (sx0 <= x_right + eps)
            & (sx1 >= x_left - eps)
            & (stop <= y_bottom + eps)
            & (sbot >= y_top - eps)
        )

        if mask.any():
            # candidate spans
            cand_idx = np.where(mask)[0]

            # intersection bbox per candidate
            inter_x0 = np.maximum(sx0[cand_idx], x_left)
            inter_y0 = np.maximum(stop[cand_idx], y_top)
            inter_x1 = np.minimum(sx1[cand_idx], x_right)
            inter_y1 = np.minimum(sbot[cand_idx], y_bottom)

            inter_w = np.maximum(0.0, inter_x1 - inter_x0)
            inter_h = np.maximum(0.0, inter_y1 - inter_y0)
            inter_area = inter_w * inter_h

            # how much of the word is covered?
            area_ratio = inter_area / w_area  # 0..1+

            # is span height comparable to word height?
            span_h = sbot[cand_idx] - stop[cand_idx]
            height_ok = span_h >= 0.7 * w_height  # big spans get a bonus

            # is span vertical-ish?
            vertical_flag = np.abs(s_dir_y[cand_idx]) > np.abs(s_dir_x[cand_idx])

            # build a simple score:
            #   - area_ratio is the main signal
            #   - +0.5 bonus if span is tall enough
            #   - +0.2 bonus if vertical (for rotated TOC labels)
            score = area_ratio.copy()
            score += 0.5 * height_ok.astype(float)
            score += 0.2 * vertical_flag.astype(float)

            # if everything is zero (e.g. weird geometry), fall back to nearest
            if (score > 0).any():
                best_local = score.argmax()
                idx_abs = span_index[cand_idx[best_local]]
            else:
                # fallback to nearest center among all spans
                dx_all = scx - w_cx
                dy_all = scy - w_cy
                dist2_all = dx_all * dx_all + dy_all * dy_all
                idx_rel = dist2_all.argmin()
                idx_abs = span_index[idx_rel]
        else:
            # --- no intersecting spans: fallback to globally nearest center ---
            dx_all = scx - w_cx
            dy_all = scy - w_cy
            dist2_all = dx_all * dx_all + dy_all * dy_all
            idx_rel = dist2_all.argmin()
            idx_abs = span_index[idx_rel]

        # attach style + orientation from the chosen span
        fontnames.append(s_font[idx_abs])
        font_sizes.append(s_size[idx_abs])
        ns_colors.append(s_ns[idx_abs])
        s_colors.append(s_s[idx_abs])

        dx_span = s_dir_x[idx_abs]
        dy_span = s_dir_y[idx_abs]
        orientations.append(_orientation_from_dir(dx_span, dy_span))

        #chosen_span_ids.append(s_span_id[idx_abs])

    out = words_df.copy()
    out["font_name"] = fontnames
    out["font_size"] = font_sizes
    out["non_stroking_color"] = ns_colors
    out["stroking_color"] = s_colors
    out["text_orientation"] = orientations
    #out["debug_span_id"] = chosen_span_ids
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

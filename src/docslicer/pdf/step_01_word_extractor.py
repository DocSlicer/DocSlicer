"""
Step 01 – Raw word extraction (pypdfium2 version)

Responsibility:
    - Open a PDF with pypdfium2
    - Iterate characters via textpage, group into whitespace-delimited words
    - Attach font name, font size, fill/stroke color from the first char of each word
    - Derive text_orientation from first→last char centre direction within the word
    - Convert all color values to hex format (#rrggbb)
    - NO high-level features (no bold/italic guesses, no ratios, etc.)

Output columns match the _pymu version exactly:
    page_number, word_id, text,
    x_left, y_top, x_right, y_bottom, width, height,
    page_width, page_height,
    font_name, font_size,
    non_stroking_color (#rrggbb or None),
    stroking_color     (#rrggbb or None),
    text_orientation   (LTR | RTL | TTB | BTT | UNKNOWN)

Coordinate system: pypdfium2 charboxes are (left, bottom, right, top) in PDF
space (y increases upward). We convert to screen space (y increases downward)
to match the _pymu output: y_top = page_height - pdf_top.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pandas as pd

from .._utils.text_utils import add_calculated_text_features


_WHITESPACE = frozenset(' \t\r\n\x0c\xa0​‌‍﻿')

# PDFium emits U+FFFE as a line-break marker in place of a soft hyphen.
# When its tight bbox has non-zero width the hyphen is visually rendered in the PDF.
_FFFE = '￾'

_BYREF = ctypes.byref


# ── Low-level per-character helpers ──────────────────────────────────────────

def _font_info(tp, i: int) -> Tuple[Optional[str], float]:
    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int(0)
    pdfium_c.FPDFText_GetFontInfo(tp, i, buf, 256, _BYREF(flags))
    name = buf.value.decode("utf-8", errors="replace") or None
    # FPDFText_GetMatrix returns the text matrix (Tm) WITHOUT the Tf font size factor.
    # FPDFText_GetFontSize returns only the Tf value.
    # Effective size = Tf × |Tm scale| covers both encoding styles:
    #   Type A: Tf=12, Tm≈identity (scale=1)  → 12 × 1 = 12
    #   Type B: Tf=1,  Tm=[fs,0,0,fs,...] → 1 × fs = fs
    tf = float(pdfium_c.FPDFText_GetFontSize(tp, i))
    m = pdfium_c.FS_MATRIX()
    if pdfium_c.FPDFText_GetMatrix(tp, i, _BYREF(m)):
        scale = max(math.hypot(m.a, m.b), math.hypot(m.c, m.d))
    else:
        scale = 1.0
    size = tf * scale if scale > 1e-6 else tf
    return name, size


def _fill_color(tp, i: int) -> Optional[str]:
    r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    if pdfium_c.FPDFText_GetFillColor(tp, i, _BYREF(r), _BYREF(g), _BYREF(b), _BYREF(a)):
        return f"#{r.value:02x}{g.value:02x}{b.value:02x}"
    return None


def _stroke_color(tp, i: int) -> Optional[str]:
    r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    if pdfium_c.FPDFText_GetStrokeColor(tp, i, _BYREF(r), _BYREF(g), _BYREF(b), _BYREF(a)):
        return f"#{r.value:02x}{g.value:02x}{b.value:02x}"
    return None


# ── Orientation ───────────────────────────────────────────────────────────────

_THRESH = 0.5


def _orientation_from_angle(angle_rad: float) -> str:
    """Map a single-char rotation angle (radians, PDF space y-up) to an orientation label."""
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)  # positive = upward in PDF = BTT on screen
    if abs(dx) >= abs(dy):
        return "LTR" if dx >= _THRESH else ("RTL" if dx <= -_THRESH else "UNKNOWN")
    # vertical — flip sign because screen y is downward
    return "BTT" if dy >= _THRESH else ("TTB" if dy <= -_THRESH else "UNKNOWN")


def _orientation_from_centers(
    boxes: List[Tuple[float, float, float, float]],
) -> str:
    """
    Derive orientation from the vector between the first and last char centres.
    boxes are in screen coords (y increases downward).
    """
    if len(boxes) < 2:
        return "UNKNOWN"
    x0 = (boxes[0][0] + boxes[0][2]) / 2.0
    y0 = (boxes[0][1] + boxes[0][3]) / 2.0
    x1 = (boxes[-1][0] + boxes[-1][2]) / 2.0
    y1 = (boxes[-1][1] + boxes[-1][3]) / 2.0
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return "UNKNOWN"
    dx /= norm
    dy /= norm
    if abs(dx) >= abs(dy):
        return "LTR" if dx >= _THRESH else ("RTL" if dx <= -_THRESH else "UNKNOWN")
    return "TTB" if dy >= _THRESH else ("BTT" if dy <= -_THRESH else "UNKNOWN")


# ── Per-page extraction ───────────────────────────────────────────────────────

def _extract_words_for_page(
    page,
    page_number: int,
    *,
    start_word_id: int,
) -> Tuple[pd.DataFrame, int]:
    page_width  = float(page.get_width())
    page_height = float(page.get_height())

    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        if n == 0:
            return pd.DataFrame(), start_word_id

        rows: List[Dict[str, Any]] = []

        # Pre-fetch all chars at once — avoids per-char ctypes buffer alloc + UTF-16 decode.
        all_chars = tp.get_text_range(0, n)

        # Per-word accumulators
        char_texts:  List[str] = []
        word_font:   Optional[str] = None
        word_size:   float         = 0.0
        word_fill:   Optional[str] = None
        word_stroke: Optional[str] = None
        first_angle: float         = 0.0
        prev_x_right: float        = -1.0
        # Running bbox — updated per char to avoid list + min/max in _flush.
        word_x_left:   float = float("inf")
        word_y_top:    float = float("inf")
        word_x_right:  float = float("-inf")
        word_y_bottom: float = float("-inf")
        # Only first/last screen box needed for orientation.
        word_first_box: Optional[Tuple[float, float, float, float]] = None
        word_last_box:  Optional[Tuple[float, float, float, float]] = None
        word_n_chars:   int = 0

        # A gap wider than this fraction of font size → implicit word boundary.
        # With loose charboxes, intra-word gaps are always ≤ 0 (boxes overlap).
        # Inter-word gaps without an explicit space char are ~0.12–0.15 × font_size.
        # 0.10 sits safely between those two ranges.
        _GAP_FACTOR = 0.10

        _INF = float("inf")

        def _flush() -> None:
            nonlocal word_first_box, word_last_box, word_n_chars
            nonlocal word_x_left, word_y_top, word_x_right, word_y_bottom
            if not char_texts:
                return
            orientation = (
                _orientation_from_centers([word_first_box, word_last_box])
                if word_n_chars >= 2
                else _orientation_from_angle(first_angle)
            )
            rows.append({
                "text":               "".join(char_texts),
                "x_left":             word_x_left,
                "y_top":              word_y_top,
                "x_right":            word_x_right,
                "y_bottom":           word_y_bottom,
                "width":              word_x_right  - word_x_left,
                "height":             word_y_bottom - word_y_top,
                "font_name":          word_font,
                "font_size":          word_size,
                "non_stroking_color": word_fill,
                "stroking_color":     word_stroke,
                "text_orientation":   orientation,
            })
            char_texts.clear()
            word_first_box = word_last_box = None
            word_n_chars   = 0
            word_x_left    = word_y_top   = _INF
            word_x_right   = word_y_bottom = -_INF

        def _start_word(i: int) -> None:
            nonlocal word_font, word_size, word_fill, word_stroke, first_angle
            word_font, word_size = _font_info(tp, i)
            word_fill            = _fill_color(tp, i)
            word_stroke          = _stroke_color(tp, i)
            first_angle          = float(pdfium_c.FPDFText_GetCharAngle(tp, i))

        def _add_char(ch: str, sb: Tuple[float, float, float, float]) -> None:
            nonlocal word_first_box, word_last_box, word_n_chars
            nonlocal word_x_left, word_y_top, word_x_right, word_y_bottom
            char_texts.append(ch)
            if word_first_box is None:
                word_first_box = sb
            word_last_box   = sb
            word_n_chars   += 1
            if sb[0] < word_x_left:    word_x_left   = sb[0]
            if sb[1] < word_y_top:     word_y_top    = sb[1]
            if sb[2] > word_x_right:   word_x_right  = sb[2]
            if sb[3] > word_y_bottom:  word_y_bottom = sb[3]

        for i in range(n):
            ch = all_chars[i]

            # charbox loose=True → full character cell height (matches PyMuPDF word height)
            l, b, r, t = tp.get_charbox(i, loose=True)

            if ch == _FFFE:
                # PDFium line-break marker. Visible (non-zero tight width) → soft hyphen.
                tl, _, tr, _ = tp.get_charbox(i, loose=False)
                if tr - tl > 0.5 and char_texts:
                    _add_char('-', (l, page_height - t, r, page_height - b))
                _flush()
                prev_x_right = -1.0
                continue

            if ch in _WHITESPACE:
                _flush()
                prev_x_right = -1.0
                continue

            # convert to screen space (y-down)
            screen_box = (l, page_height - t, r, page_height - b)

            if not char_texts:
                _start_word(i)
            elif prev_x_right >= 0:
                gap = l - prev_x_right
                if gap > _GAP_FACTOR * (word_size or 8.0):
                    # Implicit space — start a new word
                    _flush()
                    _start_word(i)

            _add_char(ch, screen_box)
            prev_x_right = r

        _flush()  # last word on page

    finally:
        tp.close()

    if not rows:
        return pd.DataFrame(), start_word_id

    df = pd.DataFrame(rows)

    # Sort to match _pymu output order
    df = df.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    n_words = len(df)
    df["word_id"]     = range(start_word_id + 1, start_word_id + 1 + n_words)
    df["page_number"] = page_number
    df["page_width"]  = page_width
    df["page_height"] = page_height

    return df, start_word_id + n_words


# ── Public API ────────────────────────────────────────────────────────────────

def extract_words(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract all words from a PDF and return a DataFrame with one row per word.

    Columns match the _pymu version: page_number, word_id, text,
    x_left, x_right, y_top, y_bottom, width, height, font_name, font_size,
    non_stroking_color, stroking_color (both as hex #rrggbb or None),
    text_orientation, plus calculated text features.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    all_dfs: List[pd.DataFrame] = []
    next_word_id = 0

    with pdfium.PdfDocument(pdf_path) as doc:
        total_pages = len(doc)

        page_numbers = (
            range(1, total_pages + 1)
            if pages_to_process is None
            else pages_to_process
        )

        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue

            page = doc[page_number - 1]
            try:
                page_df, next_word_id = _extract_words_for_page(
                    page,
                    page_number=page_number,
                    start_word_id=next_word_id,
                )
            finally:
                page.close()

            if not page_df.empty:
                all_dfs.append(page_df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = add_calculated_text_features(df)
    return df

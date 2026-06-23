"""
Step 01 – Raw word extraction

Responsibilities:
    - Open a PDF with pypdfium2
    - Iterate characters via textpage, group into whitespace-delimited words
    - Attach font name, font size, fill/stroke color from the first char of each word
    - Derive text_orientation from first→last char centre direction within the word
    - Convert all color values to hex format (#rrggbb)
    - Detect superscript and subscript characters and emit them as separate words
    - Annotate each word with marked-content and struct-tree signals (separate page-object pass)
    - NO high-level features (no bold/italic guesses, no ratios, etc.)

Output columns:
    page_number, word_id, text,
    x_left, y_top, x_right, y_bottom, width, height,
    page_width, page_height,
    font_name, font_size,
    non_stroking_color (#rrggbb or None),
    stroking_color     (#rrggbb or None),
    text_orientation   (LTR | RTL | TTB | BTT | UNKNOWN),
    script_type        (None | "superscript" | "subscript"),
    mcid               (int | None  — marked-content ID from enclosing BDC),
    marked_tag         (str | None  — mark name: "Span", "P", "Artifact", …),
    struct_tag         (str | None  — struct-tree element type: "P", "Span", …),
    struct_tag_id     (int | None  — DFS counter of the struct-tree element owning this word;
                        two words with different ids are in different struct elements even if
                        both have struct_tag == "P"),
    struct_group_id    (int | None  — struct_tag_id > mcid+1e6 > text_object_id+2e6;
                        namespaced to prevent cross-tranche collisions; same value = same logical block),
    reading_rank       (int | None  — DFS position in struct tree; global reading-order key),
    text_object_id     (int | None  — 0-based sequential id per distinct PDFium text object,
                        assigned via FPDFText_GetTextObject during the character loop;
                        works inside form XObjects; increments on first encounter in
                        textpage character order)

Coordinate system: pypdfium2 charboxes are (left, bottom, right, top) in PDF
space (y increases upward). We convert to screen space (y increases downward):
y_top = page_height - pdf_top.

Super/subscript detection:
    Characters are split into their own word when two conditions both hold,
    checked in order (cheap first):
      1. y-centre shift > 20% of the current word's font size
      2. font size ratio outside 0.40–0.82× (entering script) or above 0.88× of
         the pre-entry size (exiting script)
    Requiring both conditions prevents false positives from same-font-size glyphs
    that sit at different y positions (e.g. digits next to descender letters in an
    email address) and from decorative drop caps whose body text would otherwise
    read as subscript. The font-size API call is only made when condition 1 fires.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)  # positive = upward in PDF = BTT on screen
    if abs(dx) >= abs(dy):
        return "LTR" if dx >= _THRESH else ("RTL" if dx <= -_THRESH else "UNKNOWN")
    # vertical — flip sign because screen y is downward
    return "BTT" if dy >= _THRESH else ("TTB" if dy <= -_THRESH else "UNKNOWN")


def _orientation_from_centers(
    boxes: List[Tuple[float, float, float, float]],
) -> str:
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

# ── Script-detection thresholds ───────────────────────────────────────────────
_SCRIPT_Y_FACTOR  = 0.20   # y-centre must shift > 20% of word font-size to be candidate
_SCRIPT_SIZE_MIN  = 0.40   # incoming char must be ≥ 40% of word size (excludes drop caps)
_SCRIPT_SIZE_DOWN = 0.82   # incoming char < 82% of word size  → entering script
_SCRIPT_SIZE_INV  = 1.22   # incoming char > 122% of word size → current word was script
_SCRIPT_SIZE_UP   = 0.88   # in script word: incoming ≥ 88% of ref_size → back to normal

_GAP_FACTOR = 0.10


# ── Marked-content & struct-tree helpers ────────────────────────────────────

_TEXT_OBJ_TYPE = pdfium_c.FPDF_PAGEOBJ_TEXT
_MARK_BUF_BYTES = 256  # bytes for UTF-16LE name buffers (128 chars)


def _decode_utf16le(raw: bytes) -> str | None:
    try:
        return raw.decode("utf-16-le").rstrip("\x00") or None
    except UnicodeDecodeError:
        return None


def _get_mark_name(mark) -> str | None:
    buf = ctypes.create_string_buffer(_MARK_BUF_BYTES)
    out_len = ctypes.c_ulong(0)
    ok = pdfium_c.FPDFPageObjMark_GetName(
        mark,
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort)),
        ctypes.c_ulong(_MARK_BUF_BYTES),
        _BYREF(out_len),
    )
    if not ok or out_len.value < 2:
        return None
    return _decode_utf16le(buf.raw[: out_len.value])


def _extract_text_obj_marks(
    page,
) -> list[tuple[float, float, float, float, int | None, str | None]]:
    """
    Iterate TEXT page objects in content-stream order.

    Returns a list of (l, b, r, t, mcid, marked_tag) in PDF space (y-up).
    Used only for mcid/marked_tag assignment; text_object_id is now derived
    directly from FPDFText_GetTextObject during the character loop.
    """
    results: list[tuple[float, float, float, float, int | None, str | None]] = []
    _l = ctypes.c_float(); _b = ctypes.c_float()
    _r = ctypes.c_float(); _t = ctypes.c_float()
    _mcid_tmp = ctypes.c_int(0)

    n_obj = pdfium_c.FPDFPage_CountObjects(page)
    for i in range(n_obj):
        obj = pdfium_c.FPDFPage_GetObject(page, i)
        if not obj or pdfium_c.FPDFPageObj_GetType(obj) != _TEXT_OBJ_TYPE:
            continue

        pdfium_c.FPDFPageObj_GetBounds(obj, _BYREF(_l), _BYREF(_b), _BYREF(_r), _BYREF(_t))

        raw_mcid = pdfium_c.FPDFPageObj_GetMarkedContentID(obj)
        mcid: int | None = raw_mcid if raw_mcid >= 0 else None

        # Walk marks: prefer the MCID-bearing mark's name; fall back to first mark.
        marked_tag: str | None = None
        fallback_tag: str | None = None
        n_marks = pdfium_c.FPDFPageObj_CountMarks(obj)
        for m in range(n_marks):
            mark = pdfium_c.FPDFPageObj_GetMark(obj, m)
            if not mark:
                continue
            tag = _get_mark_name(mark)
            has_mcid = pdfium_c.FPDFPageObjMark_GetParamIntValue(mark, b"MCID", _BYREF(_mcid_tmp))
            if has_mcid:
                marked_tag = tag
                break
            if fallback_tag is None:
                fallback_tag = tag

        if marked_tag is None:
            marked_tag = fallback_tag

        results.append((_l.value, _b.value, _r.value, _t.value, mcid, marked_tag))

    return results


def _extract_struct_info(
    page,
) -> tuple[dict[int, str], dict[int, int], dict[int, int]]:
    """
    Walk the struct tree for *page* in DFS order.

    Returns:
        mcid_to_struct_tag   – mcid → element type ("P", "Span", …)
        mcid_to_struct_group – mcid → sequential element id (same id = same logical block)
        mcid_to_rank         – mcid → DFS reading-order position
    """
    tree = pdfium_c.FPDF_StructTree_GetForPage(page)
    if not tree:
        return {}, {}, {}

    mcid_to_tag:   dict[int, str] = {}
    mcid_to_group: dict[int, int] = {}
    mcid_to_rank:  dict[int, int] = {}
    counters = [0, 0]  # [elem_id, rank]
    _type_buf = ctypes.create_string_buffer(_MARK_BUF_BYTES)

    def _elem_type(elem) -> str | None:
        n = pdfium_c.FPDF_StructElement_GetType(elem, _type_buf, ctypes.c_ulong(_MARK_BUF_BYTES))
        return _decode_utf16le(_type_buf.raw[:n]) if n > 0 else None

    def _walk(elem) -> None:
        etype   = _elem_type(elem)
        elem_id = counters[0]
        counters[0] += 1

        n_ch = pdfium_c.FPDF_StructElement_CountChildren(elem)
        for ci in range(n_ch):
            child = pdfium_c.FPDF_StructElement_GetChildAtIndex(elem, ci)
            if child:
                # Struct-element child — recurse.
                _walk(child)
            else:
                # Content-item child — get its MCID.
                mc = pdfium_c.FPDF_StructElement_GetChildMarkedContentID(elem, ci)
                if mc >= 0:
                    if etype:
                        mcid_to_tag[mc] = etype
                    mcid_to_group[mc] = elem_id
                    mcid_to_rank[mc]  = counters[1]
                    counters[1] += 1

    n_root = pdfium_c.FPDF_StructTree_CountChildren(tree)
    for ri in range(n_root):
        root = pdfium_c.FPDF_StructTree_GetChildAtIndex(tree, ri)
        if root:
            _walk(root)

    pdfium_c.FPDF_StructTree_Close(tree)
    return mcid_to_tag, mcid_to_group, mcid_to_rank


def _annotate_words(
    df: pd.DataFrame,
    page,
    page_height: float,
) -> None:
    """
    Enrich *df* in-place with marked-content and struct-tree columns.

    Adds: mcid, marked_tag, struct_tag, struct_tag_id,
          struct_group_id, reading_rank.
    text_object_id is already set by the character loop before this is called.
    """
    n = len(df)
    obj_marks = _extract_text_obj_marks(page)

    # ── Assign MCID/tag to each word via bbox containment (vectorised) ────────
    mcid_arr    = np.empty(n, dtype=object); mcid_arr[:] = None
    tag_arr     = np.empty(n, dtype=object); tag_arr[:]  = None
    matched_arr = np.zeros(n, dtype=bool)   # guard: each word claimed by at most one obj

    if obj_marks and n > 0:
        cx     = (df["x_left"].to_numpy(float) + df["x_right"].to_numpy(float)) * 0.5
        cy_pdf = page_height - (df["y_top"].to_numpy(float) + df["y_bottom"].to_numpy(float)) * 0.5
        TOL = 1.5

        for l, b, r, t, mc, tag in obj_marks:
            mask = (
                (cx     >= l - TOL) & (cx     <= r + TOL) &
                (cy_pdf >= b - TOL) & (cy_pdf <= t + TOL) &
                ~matched_arr
            )
            if mask.any():
                mcid_arr[mask]    = mc
                tag_arr[mask]     = tag
                matched_arr[mask] = True

    # ── Struct tree ────────────────────────────────────────────────────────────
    mcid_to_stag, mcid_to_group, mcid_to_rank = _extract_struct_info(page)

    stag_arr   = np.empty(n, dtype=object); stag_arr[:]   = None
    sgroup_arr = np.empty(n, dtype=object); sgroup_arr[:] = None
    srank_arr  = np.empty(n, dtype=object); srank_arr[:]  = None

    for i, mc in enumerate(mcid_arr):
        if mc is not None:
            stag  = mcid_to_stag.get(mc)
            sgrp  = mcid_to_group.get(mc)
            srank = mcid_to_rank.get(mc)
            if stag  is not None: stag_arr[i]   = stag
            if sgrp  is not None: sgroup_arr[i] = sgrp
            if srank is not None: srank_arr[i]  = srank

    # ── struct_group_id: struct_tag_id > mcid+1e6 > text_object_id+2e6 ──────
    # Each tranche gets its own integer range to prevent cross-tranche collisions.
    _MCID_OFFSET  = 1_000_000
    _TXOBJ_OFFSET = 2_000_000

    txobj_arr = df["text_object_id"].to_numpy(dtype=object)

    sg_arr = np.empty(n, dtype=object); sg_arr[:] = None
    for i in range(n):
        if sgroup_arr[i] is not None:
            sg_arr[i] = int(sgroup_arr[i])
        elif mcid_arr[i] is not None:
            sg_arr[i] = _MCID_OFFSET + int(mcid_arr[i])
        elif txobj_arr[i] is not None:
            sg_arr[i] = _TXOBJ_OFFSET + int(txobj_arr[i])

    # ── Write columns ──────────────────────────────────────────────────────────
    df["mcid"]           = mcid_arr
    df["marked_tag"]     = tag_arr
    df["struct_tag"]     = stag_arr
    df["struct_tag_id"]  = sgroup_arr  # raw DFS counter; two P's with different ids are different blocks
    df["struct_group_id"] = sg_arr
    df["reading_rank"]   = srank_arr


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
        all_chars = tp.get_text_range(0, n)

        # ── Word accumulators ─────────────────────────────────────────────────
        char_texts:  List[str]              = []
        word_font:   Optional[str]          = None
        word_size:   float                  = 0.0
        word_fill:   Optional[str]          = None
        word_stroke: Optional[str]          = None
        first_angle: float                  = 0.0
        prev_x_right: float                 = -1.0
        word_x_left:   float = float("inf")
        word_y_top:    float = float("inf")
        word_x_right:  float = float("-inf")
        word_y_bottom: float = float("-inf")
        word_first_box: Optional[Tuple[float, float, float, float]] = None
        word_last_box:  Optional[Tuple[float, float, float, float]] = None
        word_n_chars:   int  = 0
        # Script state
        word_script_type:      Optional[str] = None   # "superscript" | "subscript" | None
        word_ref_size:         float         = 0.0    # normal font-size before entering script
        word_ref_cy:           float         = 0.0    # normal baseline before entering script
        word_first_baseline:   float         = 0.0    # baseline of first char in current word
        # Text-object tracking via FPDFText_GetTextObject (works inside form XObjects)
        _ptr_to_obj_id: Dict[int, int] = {}
        _next_obj_id   = [0]              # list so closures can mutate without nonlocal
        word_text_obj_id: Optional[int] = None

        _INF = float("inf")

        # ── Closures ──────────────────────────────────────────────────────────

        def _flush() -> None:
            nonlocal word_first_box, word_last_box, word_n_chars
            nonlocal word_x_left, word_y_top, word_x_right, word_y_bottom
            nonlocal word_script_type, word_first_baseline
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
                "script_type":        word_script_type,
                "text_object_id":     word_text_obj_id,
            })
            char_texts.clear()
            word_first_box = word_last_box = None
            word_n_chars   = 0
            word_x_left    = word_y_top    = _INF
            word_x_right   = word_y_bottom = -_INF
            word_script_type = None
            word_first_baseline = 0.0

        def _start_word(
            i: int,
            script_type: Optional[str] = None,
            ref_size: float = 0.0,
            ref_cy: float = 0.0,
        ) -> None:
            nonlocal word_font, word_size, word_fill, word_stroke, first_angle
            nonlocal word_script_type, word_ref_size, word_ref_cy, word_text_obj_id
            word_font, word_size = _font_info(tp, i)
            word_fill            = _fill_color(tp, i)
            word_stroke          = _stroke_color(tp, i)
            first_angle          = float(pdfium_c.FPDFText_GetCharAngle(tp, i))
            word_script_type     = script_type
            word_ref_size        = ref_size
            word_ref_cy          = ref_cy
            raw_ptr  = pdfium_c.FPDFText_GetTextObject(tp, i)
            ptr_int  = ctypes.cast(raw_ptr, ctypes.c_void_p).value
            if ptr_int:
                if ptr_int not in _ptr_to_obj_id:
                    _ptr_to_obj_id[ptr_int] = _next_obj_id[0]
                    _next_obj_id[0] += 1
                word_text_obj_id = _ptr_to_obj_id[ptr_int]
            else:
                word_text_obj_id = None

        def _add_char(ch: str, sb: Tuple[float, float, float, float]) -> None:
            nonlocal word_first_box, word_last_box, word_n_chars
            nonlocal word_x_left, word_y_top, word_x_right, word_y_bottom
            char_texts.append(ch)
            if word_first_box is None:
                word_first_box = sb
            word_last_box  = sb
            word_n_chars  += 1
            if sb[0] < word_x_left:   word_x_left   = sb[0]
            if sb[1] < word_y_top:    word_y_top    = sb[1]
            if sb[2] > word_x_right:  word_x_right  = sb[2]
            if sb[3] > word_y_bottom: word_y_bottom = sb[3]

        # ── Main character loop ───────────────────────────────────────────────

        for i in range(n):
            ch = all_chars[i]
            l, b, r, t = tp.get_charbox(i, loose=True)

            if ch == _FFFE:
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

            screen_box      = (l, page_height - t, r, page_height - b)
            char_baseline   = screen_box[3]   # screen-coord baseline (bottom of glyph)

            if not char_texts:
                _start_word(i)
                word_first_baseline = char_baseline

            elif prev_x_right >= 0:
                gap = l - prev_x_right
                if gap > _GAP_FACTOR * (word_size or 8.0):
                    _flush()
                    _start_word(i)
                    word_first_baseline = char_baseline
                else:
                    # ── Script detection (fast-path: y-shift first) ───────────
                    y_shift = abs(char_baseline - word_first_baseline)
                    if y_shift > _SCRIPT_Y_FACTOR * (word_size or 8.0):
                        char_size = _font_info(tp, i)[1]   # ctypes call – only here

                        if word_script_type is None:
                            if _SCRIPT_SIZE_MIN * word_size < char_size < _SCRIPT_SIZE_DOWN * word_size:
                                # Forward entry: normal → script
                                direction = "superscript" if char_baseline < word_first_baseline else "subscript"
                                _ref_sz = word_size
                                _ref_cy = word_first_baseline
                                _flush()
                                _start_word(i, direction, _ref_sz, _ref_cy)
                                word_first_baseline = char_baseline

                            elif char_size > _SCRIPT_SIZE_INV * word_size:
                                # Retroactive entry: word started with script char,
                                # now a normal char arrives. Tag before flush.
                                direction = "superscript" if word_first_baseline < char_baseline else "subscript"
                                word_script_type = direction   # set before _flush reads it
                                _flush()
                                _start_word(i)
                                word_first_baseline = char_baseline

                        else:
                            # Exit: in script word, normal-sized char returns
                            if char_size >= word_ref_size * _SCRIPT_SIZE_UP:
                                _flush()
                                _start_word(i)
                                word_first_baseline = char_baseline

            _add_char(ch, screen_box)
            prev_x_right = r

        _flush()

    finally:
        tp.close()

    if not rows:
        return pd.DataFrame(), start_word_id

    df = pd.DataFrame(rows)
    df = df.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    n_words = len(df)
    df["word_id"]     = range(start_word_id + 1, start_word_id + 1 + n_words)
    df["page_number"] = page_number
    df["page_width"]  = page_width
    df["page_height"] = page_height

    _annotate_words(df, page, page_height)

    return df, start_word_id + n_words


# ── Public API ────────────────────────────────────────────────────────────────

def extract_words(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract all words from a PDF, emitting superscript and subscript characters as
    separate words flagged with ``script_type`` ("superscript" | "subscript" | None).

    Columns: page_number, word_id, text,
    x_left, x_right, y_top, y_bottom, width, height, font_name, font_size,
    non_stroking_color, stroking_color (both as hex #rrggbb or None),
    text_orientation, script_type,
    mcid, marked_tag, struct_tag, struct_group_id,
    reading_rank, text_matrix_id, text_object_id,
    plus calculated text features.
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

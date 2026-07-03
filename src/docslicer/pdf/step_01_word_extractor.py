"""
Step 01 – Raw word extraction

Responsibilities:
    - Open a PDF with pypdfium2
    - Iterate characters via textpage, group into whitespace-delimited words
    - Attach font name, font size, fill/stroke color from the first char of each word
    - Derive text_orientation from first→last char centre direction within the word
    - Convert all color values to hex format (#rrggbb)
    - Detect superscript and subscript characters and emit them as separate words
    - Annotate each word with marked-content (pdfium page-object pass) and
      struct-tree signals (pikepdf, via _utils.struct_tree.build_struct_index_with_links)
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
    bdc_tag            (str | None  — mark name: "Span", "P", "Artifact", …),
    struct_tag         (str | None  — RoleMap-resolved struct-tree type: "P", "H1", "TD", …),
    struct_raw_tag     (str | None  — original /S before RoleMap resolution; equals struct_tag
                        for standard tags, preserves custom tags e.g. "CorporateHeader"),
    struct_tag_id     (int | None  — global DFS counter of the struct-tree element owning this word;
                        two words with different ids are in different struct elements even if
                        both have struct_tag == "P"),
    dfs_position       (int | None  — global DFS position in struct tree; reading-order key),
    struct_ancestors   (list[str] | None  — resolved ancestor tag names root→direct-parent,
                        e.g. ["Document", "Table", "TR", "TD", "P"]),
    struct_raw_ancestors (list[str] | None — original /S values root→direct-parent before
                        RoleMap resolution, e.g. ["Document", "Header", "P"]; parallel to
                        struct_ancestors; use when custom container tags matter),
    struct_ancestor_ids (list[int] | None — parallel DFS elem_ids for each ancestor tag;
                        same index as struct_ancestors; use to distinguish e.g. TD#3 from TD#7),
    struct_col_span    (int | None  — owning TD/TH ColSpan; None outside a table cell),
    struct_row_span    (int | None  — owning TD/TH RowSpan; None outside a table cell),
    struct_scope       (str | None  — TH Scope: "Row" | "Column" | "Both"),
    struct_headers     (list[str] | None — header-cell ID references for this cell),
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
import io
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pandas as pd

from .._utils.cpu import resolve_worker_count
from .._utils.text_utils import add_calculated_text_features
from ._utils.struct_tree import StructInfo
from ._utils.form_fields import FormField
from ._utils.struct_context import StructContext, build_struct_context
from ._utils.page_rotation import make_rotation_transform


_WHITESPACE = frozenset(' \t\r\n\x0c\xa0​‌‍﻿')

# PDFium emits a sentinel char as a line-break marker in place of a soft hyphen.
# When its tight bbox has non-zero width the hyphen is visually rendered in the PDF.
# The sentinel's value depends on the extraction API: FPDFText_GetText (used by
# pypdfium2's get_text_range) maps it to U+FFFE, but the per-index
# FPDFText_GetUnicode we build all_chars from reports U+0002. We read per-index,
# so the loop matches U+0002.
_HYPHEN_BREAK = '\x02'

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

    Returns a list of (l, b, r, t, mcid, bdc_tag) in PDF space (y-up).
    Used only for mcid/bdc_tag assignment; text_object_id is now derived
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
        bdc_tag: str | None = None
        fallback_tag: str | None = None
        n_marks = pdfium_c.FPDFPageObj_CountMarks(obj)
        for m in range(n_marks):
            mark = pdfium_c.FPDFPageObj_GetMark(obj, m)
            if not mark:
                continue
            tag = _get_mark_name(mark)
            has_mcid = pdfium_c.FPDFPageObjMark_GetParamIntValue(mark, b"MCID", _BYREF(_mcid_tmp))
            if has_mcid:
                bdc_tag = tag
                break
            if fallback_tag is None:
                fallback_tag = tag

        if bdc_tag is None:
            bdc_tag = fallback_tag

        results.append((_l.value, _b.value, _r.value, _t.value, mcid, bdc_tag))

    return results


def _annotate_words(
    df: pd.DataFrame,
    page,
    to_screen,
    struct_index: Dict[Tuple[Optional[int], int], StructInfo],
    page_index: int,
) -> None:
    """
    Enrich *df* in-place with marked-content and struct-tree columns.

    Marked-content (mcid, bdc_tag) is read from pdfium page objects. The
    struct-tree columns are looked up in *struct_index* — a doc-level
    ``{(page_index, mcid): StructInfo}`` map built once by the pikepdf-backed
    parser in ``_utils.struct_tree`` (pikepdf resolves /RoleMap, /ClassMap and
    attribute objects such as ColSpan/RowSpan that pdfium's struct API cannot).

    Adds: mcid, bdc_tag, struct_tag, struct_raw_tag, struct_tag_id,
          dfs_position, struct_ancestors, struct_ancestor_ids,
          struct_col_span, struct_row_span, struct_scope, struct_headers.
    text_object_id is already set by the character loop before this is called.
    """
    n = len(df)
    obj_marks = _extract_text_obj_marks(page)

    # ── Assign MCID/tag to each word via bbox containment (vectorised) ────────
    mcid_arr    = np.empty(n, dtype=object); mcid_arr[:] = None
    tag_arr     = np.empty(n, dtype=object); tag_arr[:]  = None
    matched_arr = np.zeros(n, dtype=bool)   # guard: each word claimed by at most one obj

    if obj_marks and n > 0:
        # Word coords are already screen-space (post-rotation). Object marks
        # come back in raw pdfium bounds space, so run them through the same
        # to_screen transform used for glyph boxes before comparing.
        cx = (df["x_left"].to_numpy(float) + df["x_right"].to_numpy(float)) * 0.5
        cy = (df["y_top"].to_numpy(float) + df["y_bottom"].to_numpy(float)) * 0.5
        TOL = 1.5

        for l, b, r, t, mc, tag in obj_marks:
            sl, st, sr, sb = to_screen(l, b, r, t)
            mask = (
                (cx >= sl - TOL) & (cx <= sr + TOL) &
                (cy >= st - TOL) & (cy <= sb + TOL) &
                ~matched_arr
            )
            if mask.any():
                mcid_arr[mask]    = mc
                tag_arr[mask]     = tag
                matched_arr[mask] = True

    # ── Struct tree (pikepdf index, keyed by (page_index, mcid)) ────────────────
    stag_arr    = np.empty(n, dtype=object); stag_arr[:]    = None
    sraw_arr    = np.empty(n, dtype=object); sraw_arr[:]    = None
    sgroup_arr  = np.empty(n, dtype=object); sgroup_arr[:]  = None
    srank_arr   = np.empty(n, dtype=object); srank_arr[:]   = None
    sanc_arr    = np.empty(n, dtype=object); sanc_arr[:]    = None
    srancanc_arr = np.empty(n, dtype=object); srancanc_arr[:] = None
    sancid_arr  = np.empty(n, dtype=object); sancid_arr[:]  = None
    scol_arr    = np.empty(n, dtype=object); scol_arr[:]    = None
    srow_arr    = np.empty(n, dtype=object); srow_arr[:]    = None
    sscope_arr  = np.empty(n, dtype=object); sscope_arr[:]  = None
    shdr_arr    = np.empty(n, dtype=object); shdr_arr[:]    = None

    for i, mc in enumerate(mcid_arr):
        if mc is None:
            continue
        # Elements with no resolvable /Pg are stored under page key None.
        info = struct_index.get((page_index, mc)) or struct_index.get((None, mc))
        if info is None:
            continue
        if info.tag      is not None: stag_arr[i]   = info.tag
        if info.raw_tag  is not None: sraw_arr[i]   = info.raw_tag
        sgroup_arr[i] = info.elem_id
        srank_arr[i]  = info.rank
        if info.ancestors:     sanc_arr[i]    = info.ancestors
        if info.raw_ancestors: srancanc_arr[i] = info.raw_ancestors
        if info.ancestor_ids:  sancid_arr[i]  = info.ancestor_ids
        # ColSpan/RowSpan only carry meaning inside a table cell; leave None
        # elsewhere so non-table words stay clean.
        if "TD" in info.ancestors or "TH" in info.ancestors:
            scol_arr[i] = info.col_span
            srow_arr[i] = info.row_span
        if info.scope:   sscope_arr[i] = info.scope
        if info.headers: shdr_arr[i]   = info.headers

    # ── Write columns ──────────────────────────────────────────────────────────
    df["mcid"]                = mcid_arr
    df["bdc_tag"]          = tag_arr
    df["struct_tag"]          = stag_arr
    df["struct_raw_tag"]      = sraw_arr    # original /S before RoleMap (custom tags), e.g. "CorporateHeader"
    df["struct_tag_id"]       = sgroup_arr  # raw DFS counter; two P's with different ids are different blocks
    df["dfs_position"]        = srank_arr
    df["struct_ancestors"]     = sanc_arr     # list[str] resolved root→direct-parent
    df["struct_raw_ancestors"] = srancanc_arr # list[str] raw /S values root→direct-parent (pre-RoleMap)
    df["struct_ancestor_ids"]  = sancid_arr   # list[int] parallel elem_ids for each ancestor tag
    df["struct_col_span"]     = scol_arr    # int | None — TD/TH ColSpan (None outside a table cell)
    df["struct_row_span"]     = srow_arr    # int | None — TD/TH RowSpan (None outside a table cell)
    df["struct_scope"]        = sscope_arr  # "Row"|"Column"|"Both" | None — TH scope
    df["struct_headers"]      = shdr_arr    # list[str] | None — header-cell ID references


_SKIP_WIDGET_TYPES = {"pushbutton"}

# Struct/font columns cloned from the last label word into each injected row.
_TEMPLATE_COLS = (
    "struct_tag", "struct_raw_tag", "struct_tag_id",
    "dfs_position", "struct_ancestors", "struct_ancestor_ids",
    "struct_col_span", "struct_row_span", "struct_scope", "struct_headers",
    "font_name", "font_size", "non_stroking_color", "stroking_color",
    "text_object_id", "mcid", "bdc_tag",
)


def _normalize_form_text(widget_type: str, value: Optional[str], is_empty: bool) -> str:
    if widget_type in ("checkbox", "radio"):
        return "[Unchecked]" if is_empty else "[Checked]"
    if is_empty or value is None:
        return "[blank]"
    return value


def _inject_form_value_rows(
    df: pd.DataFrame,
    form_fields: List[FormField],
    to_screen,
    page_number: int,
    page_width: float,
    page_height: float,
) -> List[Dict[str, Any]]:
    """
    Build one synthetic row per form field, carrying the field's value (or an
    empty/checked marker) as ``text``, positioned at the widget's bbox.

    Each injected row inherits struct and font metadata from the *last* label
    word for that field (the bottommost/rightmost word in the spatial sort),
    so downstream block/chunk assembly rolls the value up into the same group
    as its label. Fields with no label word in *df* get struct columns as None.

    Pushbuttons are skipped. Fields whose value is already present as a
    content-stream word at the widget position (overlap > 30% of widget area)
    are also skipped to avoid duplication on flattened or XFA-rendered forms.
    """
    if not form_fields or df.empty:
        return []

    # last label word per field_name (spatial sort order → bottommost/rightmost)
    last_label: Dict[str, Any] = {}
    if "form_field_name" in df.columns:
        for fn, grp in df[df["form_field_name"].notna()].groupby(
            "form_field_name", sort=False
        ):
            last_label[str(fn)] = grp.iloc[-1].to_dict()

    word_xl = df["x_left"].to_numpy(float)
    word_xr = df["x_right"].to_numpy(float)
    word_yt = df["y_top"].to_numpy(float)
    word_yb = df["y_bottom"].to_numpy(float)

    rows: List[Dict[str, Any]] = []
    for fld in form_fields:
        if fld.widget_type in _SKIP_WIDGET_TYPES:
            continue

        llx, lly, urx, ury = fld.pdf_rect
        fx_left, fy_top, fx_right, fy_bottom = to_screen(llx, lly, urx, ury)
        fw        = fx_right - fx_left
        fh        = fy_bottom - fy_top

        # Skip if any content-stream word already covers the widget bbox.
        widget_area = max(fw * fh, 1.0)
        ox = np.minimum(word_xr, fx_right) - np.maximum(word_xl, fx_left)
        oy = np.minimum(word_yb, fy_bottom) - np.maximum(word_yt, fy_top)
        if (np.maximum(ox, 0) * np.maximum(oy, 0) / widget_area).max() > 0.30:
            continue

        tmpl = last_label.get(fld.field_name)
        row: Dict[str, Any] = {
            "text":            _normalize_form_text(fld.widget_type, fld.value, fld.is_empty),
            "x_left":          fx_left,
            "y_top":           fy_top,
            "x_right":         fx_right,
            "y_bottom":        fy_bottom,
            "width":           fw,
            "height":          fh,
            "page_number":     page_number,
            "page_width":      page_width,
            "page_height":     page_height,
            "text_orientation":"LTR",
            "script_type":     None,
            "word_source":     "form_value",
            "form_widget":     fld.widget_type,
            "form_value":      None if fld.is_empty else fld.value,
            "form_is_empty":   fld.is_empty,
            "form_field_name": fld.field_name,
            "form_tooltip":    fld.label,
        }
        for col in _TEMPLATE_COLS:
            row[col] = tmpl.get(col) if tmpl is not None else None

        rows.append(row)

    return rows


def _annotate_form_fields(
    df: pd.DataFrame,
    form_fields: List[FormField],
    to_screen,
    form_label_index: Dict[Tuple[Optional[int], int], FormField],
    page_index: int,
) -> None:
    """
    Enrich *df* in-place with AcroForm field metadata on the label words that
    describe each field.

    Widget values live in /AcroForm annotations, not in the content stream, so
    pdfium never returns them as text. Rather than injecting synthetic rows, we
    find the PDF text words that serve as visible labels for each field and
    annotate them directly. This keeps the label–value relationship explicit for
    downstream RAG / LLM use without disturbing the word count or spatial sort.

    Two passes, structural first:

      1. STRUCT — for tagged PDFs, *form_label_index* maps (page, mcid) to the
         owning field via the structure tree (widget /OBJR ↔ label MCID). Words
         carrying such an mcid are labelled unambiguously; no geometry involved.
         This resolves cases spatial matching cannot, e.g. two side-by-side
         Yes/No checkboxes, or a label that happens to sit nearer another field.

      2. SPATIAL fallback — only for fields with no structural label (untagged
         PDFs, or widgets absent from the struct tree). Label candidates:
           • LEFT  — words whose right edge is within _MAX_LEFT_GAP left of the
                      field's left edge, vertical centre inside its height band.
           • ABOVE — words whose bottom edge is within _MAX_ABOVE_GAP above the
                      field's top edge, overlapping horizontally.
         When several fields map to one word, the horizontally closer one wins.
         Words already claimed structurally are never overridden.

    Adds columns: form_widget, form_value, form_is_empty, form_field_name,
    form_tooltip (the field's /TU authored label/question — the most reliable,
    verbatim label text; present on both struct- and spatially-matched words).
    All are None on words that don't serve as a label for any field.
    """
    n = len(df)
    fw_arr  = np.empty(n, dtype=object); fw_arr[:]  = None  # widget type
    fv_arr  = np.empty(n, dtype=object); fv_arr[:]  = None  # filled value
    fe_arr  = np.empty(n, dtype=object); fe_arr[:]  = None  # is_empty bool
    fn_arr  = np.empty(n, dtype=object); fn_arr[:]  = None  # field_name
    ft_arr  = np.empty(n, dtype=object); ft_arr[:]  = None  # /TU tooltip / authored label
    # Track closest-field distance per word for the multi-field conflict rule.
    dist_arr = np.full(n, np.inf)
    # Words labelled by the structure tree are locked against the spatial pass.
    struct_claimed = np.zeros(n, dtype=bool)

    def _write() -> None:
        df["form_widget"]     = fw_arr
        df["form_value"]      = fv_arr
        df["form_is_empty"]   = fe_arr
        df["form_field_name"] = fn_arr
        df["form_tooltip"]    = ft_arr

    if n == 0:
        _write()
        return

    # ── Pass 1: structural label assignment via (page, mcid) ──────────────────
    if form_label_index and "mcid" in df.columns:
        mcids = df["mcid"].to_numpy(dtype=object)
        for i in range(n):
            mc = mcids[i]
            if mc is None:
                continue
            # Elements with no resolvable /Pg are keyed under page None.
            fld = (form_label_index.get((page_index, int(mc)))
                   or form_label_index.get((None, int(mc))))
            if fld is None:
                continue
            fw_arr[i] = fld.widget_type
            fv_arr[i] = None if fld.is_empty else fld.value
            fe_arr[i] = fld.is_empty
            fn_arr[i] = fld.field_name
            ft_arr[i] = fld.label
            struct_claimed[i] = True

    # ── Pass 2: spatial fallback for fields without a structural label ────────
    # Fields whose label was already placed structurally are skipped entirely so
    # the brittle geometry heuristic only runs where nothing better exists.
    struct_field_names = {
        fld.field_name for fld in form_label_index.values()
    } if form_label_index else set()
    spatial_fields = [
        f for f in form_fields if f.field_name not in struct_field_names
    ]

    if not spatial_fields:
        _write()
        return

    wx_left   = df["x_left"].to_numpy(float)
    wx_right  = df["x_right"].to_numpy(float)
    wy_top    = df["y_top"].to_numpy(float)
    wy_bottom = df["y_bottom"].to_numpy(float)
    wy_center = (wy_top + wy_bottom) * 0.5

    _MAX_LEFT_GAP  = 100.0   # max horizontal distance from word right-edge to field left
    _MAX_ABOVE_GAP =  18.0   # max vertical distance from word bottom to field top
    # Widget types whose label is to the RIGHT (option text) — skip left/above matching.
    _NO_LABEL_TYPES = {"checkbox", "radio", "pushbutton"}

    for fld in spatial_fields:
        if fld.widget_type in _NO_LABEL_TYPES:
            continue

        llx, lly, urx, ury = fld.pdf_rect
        fx_left, fy_top, fx_right, fy_bottom = to_screen(llx, lly, urx, ury)
        fheight   = fy_bottom - fy_top

        # Words to the LEFT: end before field starts, centre within the field's
        # vertical band (±20% of height to stay on the same visual row).
        left_mask = (
            (wx_right <= fx_left + 2.0) &
            (wx_right >= fx_left - _MAX_LEFT_GAP) &
            (wy_center >= fy_top    - fheight * 0.2) &
            (wy_center <= fy_bottom + fheight * 0.2)
        )
        # Words ABOVE: bottom within gap above field top, horizontally overlapping.
        above_mask = (
            (wy_bottom >= fy_top - _MAX_ABOVE_GAP) &
            (wy_bottom <= fy_top + 2.0) &
            (wx_right  >= fx_left  - 5.0) &
            (wx_left   <= fx_right + 5.0)
        )

        candidate = left_mask | above_mask
        if not candidate.any():
            continue

        # Proximity = horizontal gap for left-labels; 0 for above-labels (treat
        # above as same priority as touching-left).
        horiz_gap = np.where(left_mask, fx_left - wx_right, 0.0)

        idxs = np.where(candidate)[0]
        for i in idxs:
            if struct_claimed[i]:
                continue  # never override a structurally-assigned label
            if horiz_gap[i] < dist_arr[i]:
                dist_arr[i]  = horiz_gap[i]
                fw_arr[i]    = fld.widget_type
                fv_arr[i]    = None if fld.is_empty else fld.value
                fe_arr[i]    = fld.is_empty
                fn_arr[i]    = fld.field_name
                ft_arr[i]    = fld.label

    _write()


# ── Per-page extraction ───────────────────────────────────────────────────────

def _extract_words_for_page(
    page,
    page_number: int,
    *,
    start_word_id: int,
    struct_index: Dict[Tuple[Optional[int], int], StructInfo],
    form_fields: List[FormField],
    form_label_index: Dict[Tuple[Optional[int], int], FormField],
) -> Tuple[pd.DataFrame, int]:
    page_width  = float(page.get_width())
    page_height = float(page.get_height())

    # pdfium's get_charbox/GetBounds return coordinates in raw, unrotated page
    # space (CropBox-space before /Rotate), while get_width/get_height already
    # reflect /Rotate. to_screen folds crop-offset + rotation + the y-flip into
    # one step so every raw box lands in the displayed page's screen space.
    to_screen, rotation = make_rotation_transform(page, page_width, page_height)
    _rotation_rad = math.radians(rotation)

    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        if n == 0:
            return pd.DataFrame(), start_word_id

        rows: List[Dict[str, Any]] = []
        # Build per-index to stay aligned with get_charbox(i) etc. A bulk
        # get_text_range desyncs when non-BMP glyphs (e.g. math symbols) are
        # present, since the decoded string length no longer matches n.
        all_chars = "".join(chr(pdfium_c.FPDFText_GetUnicode(tp, i)) for i in range(n))

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
            # GetCharAngle is in raw (unrotated) space like GetCharBox; rotate
            # it into displayed space to match screen_box orientation below.
            first_angle          = float(pdfium_c.FPDFText_GetCharAngle(tp, i)) - _rotation_rad
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

            if ch == _HYPHEN_BREAK:
                tl, _, tr, _ = tp.get_charbox(i, loose=False)
                if tr - tl > 0.5 and char_texts:
                    _add_char('-', to_screen(l, b, r, t))
                _flush()
                prev_x_right = -1.0
                continue

            if ch in _WHITESPACE:
                _flush()
                prev_x_right = -1.0
                continue

            screen_box      = to_screen(l, b, r, t)
            char_baseline   = screen_box[3]   # screen-coord baseline (bottom of glyph)

            if not char_texts:
                _start_word(i)
                word_first_baseline = char_baseline

            elif prev_x_right >= 0:
                # Gap/baseline both read off screen_box (post-rotation) so word-
                # break detection tracks the reading direction as displayed, not
                # the raw content-stream axis (which is swapped under rotation).
                gap = screen_box[0] - prev_x_right
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
            prev_x_right = screen_box[2]

        _flush()

    finally:
        tp.close()

    if not rows:
        return pd.DataFrame(), start_word_id

    df = pd.DataFrame(rows)
    df = df.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    df["page_number"]  = page_number
    df["page_width"]   = page_width
    df["page_height"]  = page_height
    df["word_source"]  = "content_stream"

    _annotate_words(df, page, to_screen, struct_index, page_number - 1)
    _annotate_form_fields(
        df, form_fields, to_screen, form_label_index, page_number - 1
    )
    value_rows = _inject_form_value_rows(
        df, form_fields, to_screen, page_number, page_width, page_height
    )
    if value_rows:
        df = pd.concat([df, pd.DataFrame(value_rows)], ignore_index=True)
        df = df.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    # Assign word_ids after inject+sort so synthetic rows get sequential ids.
    df["word_id"] = range(start_word_id + 1, start_word_id + 1 + len(df))

    return df, start_word_id + len(df)


# ── Parallel helpers ──────────────────────────────────────────────────────────

_PARALLEL_PAGE_THRESHOLD = 50


def _chunk_pages(page_numbers: List[int], n_chunks: int) -> List[List[int]]:
    k, rem = divmod(len(page_numbers), n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            chunks.append(page_numbers[start:end])
        start = end
    return chunks


def _extract_words_chunk(
    pdf_bytes: bytes,
    page_numbers: List[int],
    struct_index: Dict[Tuple[Optional[int], int], StructInfo],
    form_index: Dict[int, List[FormField]],
    form_label_index: Dict[Tuple[Optional[int], int], FormField],
) -> List[pd.DataFrame]:
    """Worker: opens its own in-memory PdfDocument and processes a chunk of pages.

    Runs in a separate *process* (PDFium is not thread-safe — concurrent document
    loads/parses race on shared C state). Accepts bytes (read once by the caller,
    pickled to each worker) so it uses FPDF_LoadMemDocument rather than
    FPDF_LoadDocument. word_id values are page-local (start_word_id=0); the caller
    reassigns them globally after merging all chunks.
    """
    dfs: List[pd.DataFrame] = []
    with pdfium.PdfDocument(io.BytesIO(pdf_bytes)) as doc:
        total_pages = len(doc)
        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue
            try:
                page = doc[page_number - 1]
            except Exception:
                continue
            try:
                page_df, _ = _extract_words_for_page(
                    page,
                    page_number=page_number,
                    start_word_id=0,
                    struct_index=struct_index,
                    form_fields=form_index.get(page_number - 1, []),
                    form_label_index=form_label_index,
                )
            finally:
                page.close()
            if not page_df.empty:
                dfs.append(page_df)
    return dfs


# ── Public API ────────────────────────────────────────────────────────────────

def extract_words(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
    struct_ctx: Optional[StructContext] = None,
) -> pd.DataFrame:
    """
    Extract all words from a PDF, emitting superscript and subscript characters as
    separate words flagged with ``script_type`` ("superscript" | "subscript" | None).

    Columns: page_number, word_id, text,
    x_left, x_right, y_top, y_bottom, width, height, font_name, font_size,
    non_stroking_color, stroking_color (both as hex #rrggbb or None),
    text_orientation, script_type,
    mcid, bdc_tag, struct_tag, struct_raw_tag, struct_tag_id,
    dfs_position, struct_ancestors, struct_ancestor_ids,
    struct_col_span, struct_row_span, struct_scope, struct_headers,
    text_object_id,
    form_widget, form_value, form_is_empty, form_field_name, form_tooltip,
    plus calculated text features.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    # The pikepdf-derived indices (struct tree + AcroForm) are normally built once
    # by the orchestrator and passed in as *struct_ctx*, so a single pikepdf open
    # serves words, images and shapes. When called standalone (tests, __main__) we
    # build our own — degrading to an empty context on any failure so untagged /
    # encrypted-without-context PDFs still extract text.
    #   struct_index     : {(page, mcid): StructInfo}   — struct-tree leaves
    #   form_index       : {page_index: [FormField]}    — AcroForm fields
    #   form_label_index : {(page, mcid): FormField}    — struct-tree widget→label
    #                      join; words at these MCIDs are the field's visible label.
    if struct_ctx is None:
        try:
            struct_ctx = build_struct_context(pdf_path)
        except Exception:
            struct_ctx = StructContext()
    struct_index = struct_ctx.struct_index
    form_index = struct_ctx.form_index
    form_label_index = struct_ctx.form_label_index

    with pdfium.PdfDocument(pdf_path) as doc:
        total_pages = len(doc)
        page_numbers_list: List[int] = list(
            range(1, total_pages + 1)
            if pages_to_process is None
            else pages_to_process
        )

    n_workers = 1
    if len(page_numbers_list) >= _PARALLEL_PAGE_THRESHOLD:
        n_workers = resolve_worker_count(None, n_items=len(page_numbers_list))

    all_dfs: List[pd.DataFrame] = []

    if n_workers > 1:
        pdf_bytes = pdf_path.read_bytes()
        chunks = _chunk_pages(page_numbers_list, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _extract_words_chunk,
                    pdf_bytes, chunk, struct_index, form_index, form_label_index,
                )
                for chunk in chunks
            ]
            for f in futures:
                all_dfs.extend(f.result())
    else:
        next_word_id = 0
        with pdfium.PdfDocument(pdf_path) as doc:
            total_pages = len(doc)
            for page_number in page_numbers_list:
                if page_number < 1 or page_number > total_pages:
                    continue
                try:
                    page = doc[page_number - 1]
                except Exception:
                    continue
                try:
                    page_df, next_word_id = _extract_words_for_page(
                        page,
                        page_number=page_number,
                        start_word_id=next_word_id,
                        struct_index=struct_index,
                        form_fields=form_index.get(page_number - 1, []),
                        form_label_index=form_label_index,
                    )
                finally:
                    page.close()
                if not page_df.empty:
                    all_dfs.append(page_df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    if n_workers > 1:
        # Chunks arrive out of order; restore reading order then assign global ids.
        df = df.sort_values(["page_number", "word_id"], kind="mergesort").reset_index(drop=True)
        df["word_id"] = range(1, len(df) + 1)

    df = add_calculated_text_features(df)
    return df

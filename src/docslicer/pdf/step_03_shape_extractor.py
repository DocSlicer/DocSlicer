"""
Step 03 – Raw shape extraction

Output columns:
    page_number, raw_shape_id, raw_shape_type,
    x_left, y_top, x_right, y_bottom, width, height, area,
    non_stroking_color, stroking_color, linewidth,
    fill, stroke, paint_op

Coordinate system: FPDFPageObj_GetBounds returns (left, bottom, right, top) in
raw, unrotated PDF space (y increases upward). We convert to screen space
(y increases downward, in the page's *displayed* orientation) via
_utils.page_rotation.make_rotation_transform, which also accounts for the
page's /Rotate entry — see that module's docstring for why this is needed.
"""

from __future__ import annotations

from ctypes import c_float, c_int, c_uint
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pandas as pd

from ._utils.page_rotation import make_rotation_transform
from ._utils.struct_tree import StructInfo, struct_info_to_columns


# ── Module-level function references (avoid attribute lookup in tight loops) ──

_count_objects  = pdfium_c.FPDFPage_CountObjects
_get_object     = pdfium_c.FPDFPage_GetObject
_get_obj_type   = pdfium_c.FPDFPageObj_GetType
_get_bounds     = pdfium_c.FPDFPageObj_GetBounds
_get_draw_mode  = pdfium_c.FPDFPath_GetDrawMode
_get_fill_clr   = pdfium_c.FPDFPageObj_GetFillColor
_get_stroke_clr = pdfium_c.FPDFPageObj_GetStrokeColor
_get_width      = pdfium_c.FPDFPageObj_GetStrokeWidth
_count_segs     = pdfium_c.FPDFPath_CountSegments
_get_seg        = pdfium_c.FPDFPath_GetPathSegment
_get_seg_type   = pdfium_c.FPDFPathSegment_GetType
_get_mcid       = pdfium_c.FPDFPageObj_GetMarkedContentID

_PATH_TYPE  = pdfium_c.FPDF_PAGEOBJ_PATH
_SEG_LINE   = pdfium_c.FPDF_SEGMENT_LINETO    # 0
_SEG_BEZIER = pdfium_c.FPDF_SEGMENT_BEZIERTO  # 1
_SEG_MOVE   = pdfium_c.FPDF_SEGMENT_MOVETO    # 2


# ── Classify ──────────────────────────────────────────────────────────────────

def _classify_path(raw_obj: Any) -> str:
    """
    Classify a path into 'rect', 'line', or 'curve'.

    Observed segment patterns (LINETO=0, BEZIERTO=1, MOVETO=2, CLOSE=-1):
        rect  → (2, 0, 0, 0, 0)       MOVETO + 4 LINETOs
        line  → (2,0) or (2,0,2,0,…)  one LINETO per MOVETO
        curve → any BEZIERTO
    """
    n = _count_segs(raw_obj)
    if n <= 0:
        return "other"

    move_count = 0
    line_count = 0
    for i in range(n):
        seg = _get_seg(raw_obj, i)
        if not seg:
            continue
        t = _get_seg_type(seg)
        if t == _SEG_BEZIER:
            return "curve"       # early exit
        elif t == _SEG_MOVE:
            move_count += 1
        elif t == _SEG_LINE:
            line_count += 1
        # _SEG_CLOSE (-1) intentionally ignored

    if move_count == 0:
        return "other"
    if move_count == 1 and line_count == 4:
        return "rect"
    if move_count == line_count:
        return "line"
    return "curve"


# ── Per-page extraction ───────────────────────────────────────────────────────

def _extract_raw_shapes_for_page(
    page: pdfium.PdfPage,
    page_number: int,
    *,
    include_types: Optional[Sequence[str]] = None,
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]] = None,
) -> List[Dict[str, Any]]:
    raw_shapes: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    page_width  = float(page.get_width())
    page_height = float(page.get_height())
    to_screen, _rotation = make_rotation_transform(page, page_width, page_height)

    # Pre-allocate all ctypes slots once — reused for every shape on this page.
    _l = c_float(); _b = c_float(); _r = c_float(); _t = c_float()
    _fm = c_int();  _sf = c_int()
    _cr = c_uint(); _cg = c_uint(); _cb = c_uint(); _ca = c_uint()
    _w  = c_float()

    n_obj = _count_objects(page)
    for i in range(n_obj):
        raw_obj = _get_object(page, i)
        if not raw_obj or _get_obj_type(raw_obj) != _PATH_TYPE:
            continue

        # ── Bounds ──────────────────────────────────────────────────────────
        _get_bounds(raw_obj, _l, _b, _r, _t)
        x_left, y_top, x_right, y_bottom = to_screen(_l.value, _b.value, _r.value, _t.value)

        width  = x_right - x_left
        height = y_bottom - y_top
        if width < 0.1 and height < 0.1:
            continue

        area = width * height

        # ── Draw mode ───────────────────────────────────────────────────────
        _get_draw_mode(raw_obj, _fm, _sf)
        is_fill   = _fm.value != 0
        is_stroke = _sf.value != 0

        paint_op = (
            "fs" if is_fill and is_stroke
            else "f" if is_fill
            else "s" if is_stroke
            else ""
        )

        # ── Colors (inline hex — we know values are always 0-255 ints) ─────
        non_stroking_color: Optional[str] = None
        if is_fill:
            _get_fill_clr(raw_obj, _cr, _cg, _cb, _ca)
            non_stroking_color = f"#{_cr.value:02x}{_cg.value:02x}{_cb.value:02x}"

        stroking_color: Optional[str] = None
        if is_stroke:
            _get_stroke_clr(raw_obj, _cr, _cg, _cb, _ca)
            stroking_color = f"#{_cr.value:02x}{_cg.value:02x}{_cb.value:02x}"

        # ── Linewidth ────────────────────────────────────────────────────────
        _get_width(raw_obj, _w)
        linewidth = float(_w.value)

        # ── Shape type ───────────────────────────────────────────────────────
        raw_shape_type = _classify_path(raw_obj)

        if include_types and raw_shape_type not in include_types:
            continue

        # ── Deduplication (built-in hash, much faster than md5) ─────────────
        h = hash((
            round(x_left,   2), round(y_top,    2),
            round(x_right,  2), round(y_bottom, 2),
            raw_shape_type, non_stroking_color, stroking_color,
            round(linewidth or 0, 2),
        ))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        row: Dict[str, Any] = {
            "page_number":       page_number,
            "page_width":        page_width,
            "page_height":       page_height,
            "raw_shape_id":      0,  # assigned after sorting
            "raw_shape_type":    raw_shape_type,
            "x_left":            x_left,
            "x_right":           x_right,
            "y_top":             y_top,
            "y_bottom":          y_bottom,
            "width":             width,
            "height":            height,
            "area":              area,
            "non_stroking_color": non_stroking_color,
            "stroking_color":    stroking_color,
            "linewidth":         linewidth,
            "fill":              is_fill,
            "stroke":            is_stroke,
            "paint_op":          paint_op or None,
        }

        # Struct-tree enrichment by marked-content id (Figure/Artifact tag,
        # ancestors, table spans). Only when a struct index is supplied, so
        # standalone extraction keeps its legacy shape-only schema.
        if struct_index is not None:
            mcid = _get_mcid(raw_obj)
            info = None
            if mcid >= 0:
                info = (struct_index.get((page_number - 1, mcid))
                        or struct_index.get((None, mcid)))
            row["mcid"] = mcid if mcid >= 0 else None
            row.update(struct_info_to_columns(info))

        raw_shapes.append(row)

    return raw_shapes


# ── Public API ────────────────────────────────────────────────────────────────

def extract_shapes(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
    include_types: Optional[Sequence[str]] = None,
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]] = None,
) -> pd.DataFrame:
    """
    Extract all shapes from a PDF and return a DataFrame with one row per shape.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: Page numbers (1-indexed), or None for all pages
        include_types: Shape types to include — None for all, or a subset of
            ``['rect', 'line', 'curve']``
        struct_index: Optional ``{(page, mcid): StructInfo}`` from the shared
            :class:`StructContext`. When supplied, each shape is joined to its
            struct-tree leaf by marked-content id, adding ``mcid`` and the same
            ``struct_*`` columns words carry. Omit it to keep the legacy schema.

    Shapes are sorted by page_number → y_top → x_left.
    ``raw_shape_id`` is assigned sequentially (1-based) after sorting.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    all_shapes: List[Dict[str, Any]] = []

    with pdfium.PdfDocument(pdf_path) as doc:
        total_pages = len(doc)

        page_numbers = (
            range(1, total_pages + 1) if pages_to_process is None else pages_to_process
        )

        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue

            try:
                page = doc[page_number - 1]
            except Exception:
                continue
            all_shapes.extend(
                _extract_raw_shapes_for_page(
                    page,
                    page_number,
                    include_types=include_types,
                    struct_index=struct_index,
                )
            )

    if not all_shapes:
        return pd.DataFrame()

    df = pd.DataFrame(all_shapes)
    df = df.sort_values(by=["page_number", "y_top", "x_left"], ignore_index=True)
    df["raw_shape_id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step_03_shape_extractor.py <pdf_path>")
        sys.exit(1)

    df = extract_shapes(sys.argv[1])

    if df.empty:
        print("No shapes found")
    else:
        print(f"Found {len(df)} shapes")
        print(df[["page_number", "raw_shape_type", "width", "height",
                  "non_stroking_color", "stroking_color", "linewidth"]].head(10))

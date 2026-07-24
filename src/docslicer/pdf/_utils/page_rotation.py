# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Shared page-rotation handling for pdfium-based extractors.

pdfium's low-level bounds APIs (FPDFText_GetCharBox, FPDFPageObj_GetBounds,
FPDFLink_GetAnnotRect, AcroForm /Rect via pikepdf) all report coordinates in
raw, unrotated page space — the content stream's own coordinate system.
page.get_width()/get_height(), by contrast, already reflect the page's
/Rotate entry. A page with /Rotate 90 and a portrait 612x792 MediaBox
displays as a landscape 792x612 page, but every raw box from those bounds
APIs is still in the 612x792 frame. Left unhandled, this makes rotated-page
content (e.g. PPT-to-PDF exports with baked-in /Rotate) come out with
swapped/rotated positions and orientations for every object on the page.

make_rotation_transform() derives the exact rotation from the page's raw
CropBox and /Rotate value, and returns a to_screen(l, b, r, t) converter that
folds crop-offset + rotation + the PDF-to-screen y-flip into one step.
"""
from __future__ import annotations

import ctypes
from typing import Callable, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

ToScreen = Callable[[float, float, float, float], Tuple[float, float, float, float]]


def _raw_crop_box(page: "pdfium.PdfPage", page_width: float, page_height: float) -> Tuple[float, float, float, float]:
    """Return (left, bottom, right, top) CropBox in raw MediaBox space.

    Falls back to a box built from page_width/page_height when the CropBox
    query fails — only correct for rotation 0/180, but that failure mode is
    itself rare (CropBox falls back to MediaBox inside pdfium already).
    """
    left = ctypes.c_float()
    bottom = ctypes.c_float()
    right = ctypes.c_float()
    top = ctypes.c_float()
    ok = pdfium_c.FPDFPage_GetCropBox(
        page.raw,
        ctypes.byref(left),
        ctypes.byref(bottom),
        ctypes.byref(right),
        ctypes.byref(top),
    )
    if ok:
        return float(left.value), float(bottom.value), float(right.value), float(top.value)
    return 0.0, 0.0, page_width, page_height


def make_rotation_transform(
    page: "pdfium.PdfPage", page_width: float, page_height: float
) -> Tuple[ToScreen, int]:
    """Build a (l, b, r, t) raw-PDF-space -> (x_left, y_top, x_right, y_bottom)
    screen-space converter that accounts for the page's /Rotate.

    Returns (to_screen, rotation_degrees).
    """
    crop_l, crop_b, crop_r, crop_t = _raw_crop_box(page, page_width, page_height)
    raw_w = crop_r - crop_l
    raw_h = crop_t - crop_b
    rotation = page.get_rotation() % 360

    def _rotate(x: float, y: float) -> Tuple[float, float]:
        if rotation == 90:
            return y, raw_w - x
        if rotation == 180:
            return raw_w - x, raw_h - y
        if rotation == 270:
            return raw_h - y, x
        return x, y

    def to_screen(l: float, b: float, r: float, t: float) -> Tuple[float, float, float, float]:
        x0, y0 = _rotate(l - crop_l, b - crop_b)
        x1, y1 = _rotate(r - crop_l, t - crop_b)
        disp_l, disp_r = (x0, x1) if x0 <= x1 else (x1, x0)
        disp_b, disp_t = (y0, y1) if y0 <= y1 else (y1, y0)
        return disp_l, page_height - disp_t, disp_r, page_height - disp_b

    return to_screen, rotation

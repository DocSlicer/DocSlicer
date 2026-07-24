# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Color conversion and perceptual comparison (hex/RGB/Lab, CIEDE2000 delta-E)."""

from __future__ import annotations

import math
import re
from numbers import Integral, Real
from typing import Any, Optional, Sequence

import numpy as np


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _component_to_hex(value: float) -> str:
    return f"{int(round(_clamp01(value) * 255)):02x}"


def _rgb01_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = rgb
    return f"#{_component_to_hex(r)}{_component_to_hex(g)}{_component_to_hex(b)}"


def _normalize_hex(value: str) -> Optional[str]:
    match = _HEX_RE.match(value.strip())
    if not match:
        return None

    hex_value = match.group(1).lower()
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    return f"#{hex_value}"


def _sequence_to_rgb01(color_value: Sequence[Any]) -> Optional[tuple[float, float, float]]:
    try:
        values = [float(v) for v in color_value]
    except (TypeError, ValueError):
        return None

    if not values or any(math.isnan(v) for v in values):
        return None

    if len(values) == 1:
        v = values[0]
        if v > 1.0:
            v /= 255.0
        v = _clamp01(v)
        return (v, v, v)

    if len(values) == 3:
        if max(values) > 1.0:
            values = [v / 255.0 for v in values]
        r, g, b = (_clamp01(v) for v in values)
        return (r, g, b)

    if len(values) == 4:
        if max(values) > 1.0:
            values = [v / 100.0 for v in values]
        c, m, y, k = (_clamp01(v) for v in values)
        return (
            (1.0 - c) * (1.0 - k),
            (1.0 - m) * (1.0 - k),
            (1.0 - y) * (1.0 - k),
        )

    return None


def pdf_color_to_hex(color_value: Any) -> Optional[str]:
    """
    Convert PDF color values from pypdfium2 to hex.

    Handles:
      - packed RGB integers, e.g. 16777215 -> '#ffffff'
      - RGB tuples/lists in 0-1 or 0-255 range
      - CMYK tuples/lists in 0-1 or 0-100 range
      - grayscale floats/ints
      - '#RGB' and '#RRGGBB' strings

    Returns '#rrggbb' or None for invalid/missing colors.
    """
    if color_value is None:
        return None

    try:
        if isinstance(color_value, str):
            return _normalize_hex(color_value)

        if isinstance(color_value, Integral):
            if color_value < 0 or color_value > 0xFFFFFF:
                return None
            return f"#{int(color_value):06x}"

        if isinstance(color_value, Real):
            value = float(color_value)
            if math.isnan(value):
                return None
            return _rgb01_to_hex(_sequence_to_rgb01([value]) or ())

        if isinstance(color_value, Sequence):
            rgb = _sequence_to_rgb01(color_value)
            return _rgb01_to_hex(rgb) if rgb else None

    except Exception:
        return None

    return None


# ==================================================================================================
# PERCEPTUAL COLOR COMPARISON (for OCR)
# ==================================================================================================
#
# The OCR pipeline extracts `non_stroking_color` from rasterized scans, so the value jitters even
# after two rounds of quantization/snapping. Downstream code (layout.py) treats any change in
# non_stroking_color as a signal to spin off a new layout, so we need a comparison that answers the
# *human* question — "do these two colors look the same?" — rather than a raw RGB one.
#
# RGB Euclidean distance is not perceptually uniform: the same numeric step reads very differently
# depending on where it sits in the cube (dark tones especially get exaggerated). #202020 vs #000000
# is a large RGB step (~55) yet both look near-black. The standard fix is to convert sRGB -> CIELAB
# (a perceptually-oriented space) and compute CIEDE2000 ΔE, the CIE's model of perceived difference.
#
# ΔE rule of thumb:  <1 imperceptible · ~2.3 just-noticeable · 2.3-10 "same family" · >~12 distinct.
# We default to 10: #202020≈#000000 (ΔE 7.4) and scan drift like #404040≈#405050 (ΔE 9.0) collapse,
# while #000000 vs #0000FF (ΔE 39.7) or vs dark-red #800000 (ΔE 30.8) stay distinct.

OCR_COLOR_DELTA_E_THRESHOLD = 10.0


def _hex_to_rgb01(value: str) -> Optional[tuple[float, float, float]]:
    normalized = _normalize_hex(value)
    if normalized is None:
        return None
    h = normalized[1:]
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an sRGB triple (0-1) to CIELAB (D65 reference white)."""

    def _linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (_linearize(c) for c in rgb)

    # linear sRGB -> XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # normalize by D65 white point
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = _f(x), _f(y), _f(z)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def delta_e_ciede2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    """Perceptual color difference (CIEDE2000) between two CIELAB colors."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - math.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    d_lp = l2 - l1
    d_cp = c2p - c1p

    if c1p * c2p == 0.0:
        d_hp = 0.0
    elif h2p - h1p > 180.0:
        d_hp = h2p - h1p - 360.0
    elif h2p - h1p < -180.0:
        d_hp = h2p - h1p + 360.0
    else:
        d_hp = h2p - h1p
    d_big_hp = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(d_hp) / 2.0)

    l_bar_p = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0

    if c1p * c2p == 0.0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        h_bar_p = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        h_bar_p = (h1p + h2p + 360.0) / 2.0
    else:
        h_bar_p = (h1p + h2p - 360.0) / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )
    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p**7
    rc = 2.0 * math.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7))
    sl = 1.0 + (0.015 * (l_bar_p - 50.0) ** 2) / math.sqrt(20.0 + (l_bar_p - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -math.sin(math.radians(2.0 * d_theta)) * rc

    return math.sqrt(
        (d_lp / sl) ** 2
        + (d_cp / sc) ** 2
        + (d_big_hp / sh) ** 2
        + rt * (d_cp / sc) * (d_big_hp / sh)
    )


def ocr_color_delta_e(color_a: Any, color_b: Any) -> Optional[float]:
    """
    Perceptual CIEDE2000 ΔE between two colors (hex strings or pdf color values).

    Returns None if either color is missing/unparseable.
    """
    hex_a = pdf_color_to_hex(color_a)
    hex_b = pdf_color_to_hex(color_b)
    if hex_a is None or hex_b is None:
        return None
    rgb_a = _hex_to_rgb01(hex_a)
    rgb_b = _hex_to_rgb01(hex_b)
    if rgb_a is None or rgb_b is None:
        return None
    return delta_e_ciede2000(_srgb_to_lab(rgb_a), _srgb_to_lab(rgb_b))


def ocr_colors_match(
    color_a: Any,
    color_b: Any,
    threshold: float = OCR_COLOR_DELTA_E_THRESHOLD,
) -> bool:
    """
    Whether two OCR colors look the same to a human (perceptual CIEDE2000 ΔE).

    Use this instead of raw hex/RGB equality when deciding whether a change in
    `non_stroking_color` is real ink-color change vs. scan jitter — e.g. so that
    #202020 vs #000000 does NOT trigger a new layout spinoff, while #000000 vs
    #0000FF does.

    Missing colors are treated as matching only when *both* are missing; a single
    missing color counts as a difference.
    """
    if color_a is None and color_b is None:
        return True

    dist = ocr_color_delta_e(color_a, color_b)
    if dist is None:
        return False
    return dist <= threshold


# ==================================================================================================
# VECTORIZED PERCEPTUAL COMPARISON
# ==================================================================================================
#
# The scalar helpers above answer "do these two colors look the same?" one pair at a time. For
# whole-column work (e.g. the layout builder comparing every line's non_stroking_color to the line
# above it), parse each color to CIELAB ONCE with `colors_to_lab`, then feed consecutive rows to
# `ciede2000_vec`. That is O(n) conversions + one vectorized ΔE pass, not O(pairs) Python calls, and
# is numerically identical to `delta_e_ciede2000` / `ocr_color_delta_e` (verified elementwise).


def colors_to_lab(values: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a sequence of pdf/hex color values to CIELAB, vectorized.

    Parsing each color to sRGB is inherently per-value (ints, 0-1 / 0-255 RGB
    tuples, CMYK, grayscale, '#rgb' strings — see ``pdf_color_to_hex``), so it
    runs once per element; the sRGB->CIELAB step is then vectorized.

    Returns ``(lab, present)``:
        lab      float array of shape (n, 3); rows for missing/unparseable
                 colors are NaN.
        present  bool array; False where the color was missing/unparseable.
    """
    n = len(values)
    rgb = np.full((n, 3), np.nan, dtype=float)
    for i, v in enumerate(values):
        hex_v = pdf_color_to_hex(v)
        if hex_v is None:
            continue
        rgb01 = _hex_to_rgb01(hex_v)
        if rgb01 is not None:
            rgb[i] = rgb01

    present = ~np.isnan(rgb).any(axis=1)
    return _srgb_to_lab_vec(rgb), present


def _srgb_to_lab_vec(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB(0-1)->CIELAB (D65); NaN rows pass through as NaN."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    def _linearize(c: np.ndarray) -> np.ndarray:
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r, g, b = _linearize(r), _linearize(g), _linearize(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x, z = x / 0.95047, z / 1.08883   # normalize by D65 white (y/1.0 == y)

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=1)


def ciede2000_vec(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Vectorized CIEDE2000 ΔE between two (n, 3) CIELAB arrays.

    Elementwise-identical to ``delta_e_ciede2000``.  Rows where either input is
    NaN return NaN.
    """
    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    d_lp = l2 - l1
    d_cp = c2p - c1p

    cp_zero = (c1p * c2p) == 0.0
    dh = h2p - h1p
    d_hp = np.where(
        cp_zero, 0.0,
        np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh)),
    )
    d_big_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(d_hp) / 2.0)

    l_bar_p = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0

    hsum = h1p + h2p
    h_bar_p = np.where(
        cp_zero, hsum,
        np.where(
            np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
            np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p ** 7
    rc = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0 ** 7))
    sl = 1.0 + (0.015 * (l_bar_p - 50.0) ** 2) / np.sqrt(20.0 + (l_bar_p - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -np.sin(np.radians(2.0 * d_theta)) * rc

    return np.sqrt(
        (d_lp / sl) ** 2
        + (d_cp / sc) ** 2
        + (d_big_hp / sh) ** 2
        + rt * (d_cp / sc) * (d_big_hp / sh)
    )


__all__ = [
    "pdf_color_to_hex",
    "ocr_colors_match",
    "ocr_color_delta_e",
    "delta_e_ciede2000",
    "colors_to_lab",
    "ciede2000_vec",
    "OCR_COLOR_DELTA_E_THRESHOLD",
]

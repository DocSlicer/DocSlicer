from __future__ import annotations

import math
import re
from numbers import Integral, Real
from typing import Any, Optional, Sequence


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


__all__ = ["pdf_color_to_hex"]

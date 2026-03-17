"""Page geometry helpers — format detection and mixed-size checking."""
from __future__ import annotations

import pandas as pd


def _detect_page_format(width: float, height: float) -> str:
    """
    Classify page format based on dimensions (in points).

    Args:
        width: Page width in points
        height: Page height in points

    Returns:
        Page format string (e.g., "A4_PORTRAIT", "US_LETTER_LANDSCAPE")
    """
    if width <= 0 or height <= 0:
        return "UNKNOWN"

    # Determine orientation
    if height > width:
        orientation = "PORTRAIT"
        w, h = width, height
    else:
        orientation = "LANDSCAPE"
        w, h = height, width  # Swap to normalize

    # Common page sizes (in points, portrait orientation)
    # Allow 5pt tolerance for matching
    TOLERANCE = 5.0

    formats = {
        "A4": (595.276, 841.890),
        "US_LETTER": (612.0, 792.0),
        "US_LEGAL": (612.0, 1008.0),
        "A3": (841.890, 1190.551),
        "TABLOID": (792.0, 1224.0),
        "SLIDE_16_9": (720.0, 540.0),  # 10x5.625 inches at 72 DPI
        "SLIDE_4_3": (720.0, 540.0),   # 10x7.5 inches at 72 DPI
    }

    for format_name, (fmt_w, fmt_h) in formats.items():
        if abs(w - fmt_w) <= TOLERANCE and abs(h - fmt_h) <= TOLERANCE:
            return f"{format_name}_{orientation}"

    # Check for presentation slides by aspect ratio
    ratio = h / w if w > 0 else 0
    if 0.55 <= ratio <= 0.78:  # Between 16:9 (0.56) and 4:3 (0.75)
        if ratio < 0.65:
            return f"SLIDE_16_9_{orientation}"
        else:
            return f"SLIDE_4_3_{orientation}"

    return f"CUSTOM_{orientation}"


def _check_mixed_page_sizes(df_cells: pd.DataFrame) -> bool:
    """
    Check if document has mixed page sizes.

    Args:
        df_cells: DataFrame with page_width and page_height columns

    Returns:
        True if multiple distinct page sizes exist
    """
    if "page_width" not in df_cells.columns or "page_height" not in df_cells.columns:
        return False

    # Get unique page sizes (rounded to avoid floating point issues)
    unique_sizes = df_cells[["page_width", "page_height"]].round(1).drop_duplicates()

    return len(unique_sizes) > 1

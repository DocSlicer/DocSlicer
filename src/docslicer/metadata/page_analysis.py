"""Page geometry + char totals — format-agnostic page info for all pipelines."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def add_page_info(doc_meta: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add page geometry and character totals to the metadata dict (in-place).

    Format-agnostic: ``df`` is whatever text-bearing layout table the pipeline
    produces — PDF cells/words, HTML boxes, DOCX paragraphs/lines. Only the
    columns used below are needed; missing columns degrade gracefully.

    Sets:
        page_count, page_width, page_height, page_format,
        has_mixed_page_sizes, chars

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        df: Layout DataFrame with page_number and optionally
            page_width, page_height, char_count
    """
    if df.empty:
        doc_meta["page_count"] = 0
        doc_meta["chars"] = 0
        return

    # --- page count ---
    if "page_number" in df.columns:
        doc_meta["page_count"] = int(df["page_number"].max())
    else:
        doc_meta["page_count"] = 0

    # --- page dimensions + format ---
    if "page_width" in df.columns and "page_height" in df.columns:
        page_width = df["page_width"].iloc[0]
        page_height = df["page_height"].iloc[0]

        # NaN dimensions are common for HTML/web pages without a fixed size.
        if pd.isna(page_width) or pd.isna(page_height):
            content_type = doc_meta.get("content_type", "").upper()
            doc_meta["page_width"] = None
            doc_meta["page_height"] = None
            doc_meta["page_format"] = "WEB" if content_type == "HTML" else "UNKNOWN"
        else:
            doc_meta["page_width"] = float(page_width)
            doc_meta["page_height"] = float(page_height)
            doc_meta["page_format"] = _detect_page_format(
                doc_meta["page_width"], doc_meta["page_height"]
            )
        doc_meta["has_mixed_page_sizes"] = bool(_check_mixed_page_sizes(df))

    # --- character total ---
    if "char_count" in df.columns:
        doc_meta["chars"] = int(df["char_count"].sum())
    else:
        doc_meta["chars"] = 0


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
        "SLIDE_16_9": (540.0, 960.0),  # 13.33x7.5 inches at 72 DPI (widescreen)
        "SLIDE_4_3": (540.0, 720.0),   # 10x7.5 inches at 72 DPI
    }

    for format_name, (fmt_w, fmt_h) in formats.items():
        if abs(w - fmt_w) <= TOLERANCE and abs(h - fmt_h) <= TOLERANCE:
            return f"{format_name}_{orientation}"

    # Check for presentation slides by aspect ratio (w <= h after normalization)
    ratio = w / h if h > 0 else 0
    if 0.55 <= ratio <= 0.78:  # Between 16:9 (0.5625) and 4:3 (0.75)
        if ratio < 0.65:
            return f"SLIDE_16_9_{orientation}"
        else:
            return f"SLIDE_4_3_{orientation}"

    return f"CUSTOM_{orientation}"


def _check_mixed_page_sizes(df: pd.DataFrame) -> bool:
    """
    Check if document has mixed page sizes.

    Args:
        df: Layout DataFrame with page_width and page_height columns

    Returns:
        True if multiple distinct page sizes exist
    """
    if "page_width" not in df.columns or "page_height" not in df.columns:
        return False

    # Get unique page sizes (rounded to avoid floating point issues)
    unique_sizes = df[["page_width", "page_height"]].round(1).drop_duplicates()

    return len(unique_sizes) > 1

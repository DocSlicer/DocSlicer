# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Document metadata: the format-agnostic half of the two-channel pipeline.

Each format pipeline runs its own ``native_metadata.extract_native_metadata``
(the *native* channel) and then leans on this package for the shared second half:

    text_fallback.add_text_fallbacks   text channel — heuristics over df_lines
    consolidate.consolidate            fold native + text into title/author/language
    page_analysis.add_page_info        page geometry + char totals (all formats)
    ocr_detector.add_page_and_ocr_info page info + OCR/scanned detection (PDF only)
"""
from .schema import DocumentMetadata
from .page_analysis import add_page_info
from .ocr_detector import add_page_and_ocr_info
from .text_fallback import add_text_fallbacks
from .consolidate import consolidate

__all__ = [
    "DocumentMetadata",
    "add_page_info",
    "add_page_and_ocr_info",
    "add_text_fallbacks",
    "consolidate",
]

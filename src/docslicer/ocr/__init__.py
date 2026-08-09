# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""OCR pipeline for scanned PDFs (tesserocr-based word extraction)."""

# Safe to import without the `ocr` extra installed — this module pulls in no
# OCR dependency, which is the point of it.
from ._availability import OcrUnavailableError, ocr_unavailable_reason

__all__ = ["OcrUnavailableError", "ocr_unavailable_reason"]

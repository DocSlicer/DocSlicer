# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""OCR / scanned-document detection — image coverage and add_page_and_ocr_info."""
from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from .page_analysis import add_page_info


def _calculate_page_image_coverage(
    df_images: pd.DataFrame,
    df_cells: pd.DataFrame
) -> pd.Series:
    """
    Calculate image area coverage ratio per page (vectorized).

    Args:
        df_images: DataFrame with page_number, x_left, y_top, x_right, y_bottom
        df_cells: DataFrame with page_number, page_width, page_height

    Returns:
        Series indexed by page_number with image coverage ratios [0, 1]
    """
    if df_images.empty:
        return pd.Series(dtype=float)

    required_cols = ["page_number", "x_left", "y_top", "x_right", "y_bottom"]
    if not all(col in df_images.columns for col in required_cols):
        return pd.Series(dtype=float)

    # Calculate image areas (vectorized)
    df_img = df_images.copy()
    df_img["image_area"] = (
        (df_img["x_right"] - df_img["x_left"]) *
        (df_img["y_bottom"] - df_img["y_top"])
    ).clip(lower=0)

    # Sum image area per page (vectorized)
    page_image_area = df_img.groupby("page_number", sort=False)["image_area"].sum()

    # Get page areas (vectorized)
    if "page_width" in df_cells.columns and "page_height" in df_cells.columns:
        page_areas = (
            df_cells
            .groupby("page_number", sort=False)
            [["page_width", "page_height"]]
            .first()
        )
        page_areas["page_area"] = page_areas["page_width"] * page_areas["page_height"]

        # Calculate coverage ratio (vectorized)
        coverage = page_image_area / page_areas["page_area"].replace(0, float('nan'))
        return coverage.fillna(0).clip(upper=1.0)

    return pd.Series(dtype=float)


def _has_full_page_images(
    df_images: pd.DataFrame,
    df_cells: pd.DataFrame,
    coverage_threshold: float = 0.95
) -> bool:
    """
    Check if document has full-page images (scanned document indicator).

    Args:
        df_images: DataFrame with image geometries
        df_cells: DataFrame with page dimensions
        coverage_threshold: Min coverage to consider "full page" (default: 0.95)

    Returns:
        True if majority of pages have near-full-page images
    """
    page_coverage = _calculate_page_image_coverage(df_images, df_cells)

    if page_coverage.empty:
        return False

    # Count pages with near-full coverage
    full_page_count = (page_coverage >= coverage_threshold).sum()
    total_pages = len(page_coverage)

    return (full_page_count / max(total_pages, 1)) > 0.5


def add_page_and_ocr_info(
    doc_meta: Dict[str, Any],
    df_cells: pd.DataFrame,
    df_images: Optional[pd.DataFrame] = None,
    char_threshold: int = 80,
    image_coverage_threshold: float = 0.6,
    page_ratio_threshold: float = 0.6,
) -> None:
    """
    Add page info, OCR needs, and scanned detection after page label detection.

    Page geometry + char total are delegated to ``add_page_info``; this function
    adds the OCR-specific detection on top (vectorized for speed):
    - needs_ocr: True if >60% of pages have <80 chars AND >60% image coverage
    - is_scanned: True if entire doc has 0 chars and only full-page images

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        df_cells: DataFrame with page_number, char_count, page_width, page_height
        df_images: Optional DataFrame with page_number, x_left, y_top, x_right, y_bottom
        char_threshold: Min chars per page to avoid OCR flag (default: 80)
        image_coverage_threshold: Min image area ratio to flag page for OCR (default: 0.6)
        page_ratio_threshold: Min ratio of OCR pages to flag entire doc (default: 0.6)

    Modifies:
        doc_meta dictionary in-place
    """
    # Page geometry + char total: page_count, dims, format, mixed sizes, chars.
    add_page_info(doc_meta, df_cells)

    if (
        df_cells.empty
        or "char_count" not in df_cells.columns
        or "page_number" not in df_cells.columns
    ):
        doc_meta["needs_ocr"] = False
        doc_meta["is_scanned"] = False
        return

    # Per-page char counts drive OCR detection; the total is set by add_page_info.
    page_char_counts = df_cells.groupby("page_number", sort=False)["char_count"].sum()
    total_chars = doc_meta["chars"]

    # =============================
    # is_scanned: Zero chars + only full-page images
    # =============================

    if total_chars == 0:
        # Check if we have full-page images
        if df_images is not None and not df_images.empty:
            has_full_page_images = _has_full_page_images(df_images, df_cells)
            doc_meta["is_scanned"] = bool(has_full_page_images)
        else:
            doc_meta["is_scanned"] = True  # No text, no images = likely scanned PDF
        doc_meta["needs_ocr"] = bool(doc_meta["is_scanned"])
        return
    else:
        doc_meta["is_scanned"] = False

    # =============================
    # needs_ocr: Low text + high image coverage (vectorized)
    # =============================

    if df_images is None or df_images.empty:
        # No images, just check char count
        low_text_pages = (page_char_counts < char_threshold).sum()
        total_pages = len(page_char_counts)
        doc_meta["needs_ocr"] = bool((low_text_pages / max(total_pages, 1)) > page_ratio_threshold)
        return

    # Calculate image coverage per page (vectorized)
    page_image_coverage = _calculate_page_image_coverage(df_images, df_cells)

    # For each page: check if low text AND high image coverage (vectorized)
    # Align page_char_counts and page_image_coverage by page_number
    page_stats = pd.DataFrame({
        "char_count": page_char_counts,
        "image_coverage": page_image_coverage,
    }).fillna(0)

    # Boolean mask: low text AND high image coverage
    needs_ocr_mask = (
        (page_stats["char_count"] < char_threshold) &
        (page_stats["image_coverage"] > image_coverage_threshold)
    )

    ocr_page_count = needs_ocr_mask.sum()
    total_pages = len(page_stats)

    doc_meta["needs_ocr"] = bool((ocr_page_count / max(total_pages, 1)) > page_ratio_threshold)

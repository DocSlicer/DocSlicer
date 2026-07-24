# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the OCR pipeline: render PDF pages to images, then extract, colorize, and clean words."""

# ocr/ocr_orchestrator.py
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import pypdfium2 as pdfium
import numpy as np
import pandas as pd

# Suppress MPS pin_memory warning
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not supported on MPS.*", category=UserWarning)

from .step_01_word_extractor import extract_words_from_images, TesserocrConfig
from .step_02_word_colorizer import colorize_words_df, WordColorizerConfig
from .step_03_shape_extractor import extract_shapes_df, ShapeExtractorConfig
from .step_04_text_cleaner import clean_words_df
from .step_05_font_size_estimator import estimate_ocr_font_sizes

from .._utils.cpu import resolve_worker_count
from .._utils.text_utils import add_calculated_text_features
from .._utils.layout.line_number_detector import (
    detect_line_numbers,
    OCR_CONFIG as OCR_LINE_NUMBER_CONFIG,
)
from .._utils.layout.reading_order import assign_reading_order
from .._utils.layout.layouts import LayoutConfig, assign_layouts
from .._utils.layout.shape_processor import process_shapes
from .._utils.timing import timed_step

logger = logging.getLogger(__name__)


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class OCRPipelineConfig:
    # Rendering & coordinate conversion
    # 4.0 ≈ ~288 DPI effective for typical 72 DPI PDF coords
    # Also used to convert final output from pixels (PX) to points (PT)
    dpi_scale: float = 4.0 # ~50% faster on 2.0, but misses large headings, and small text, and produces less accurate bboxes

    # OCR word extraction (tesserocr)
    # ocr_workers: page-level parallel width. None -> performance-core count
    #   (resolved at runtime via _utils.cpu); set explicitly to cap/override.
    # ocr_tessdata_path: directory holding *.traineddata. None -> TESSDATA_PREFIX
    #   env, else auto-detected from the installed `tesseract` CLI. Only set to
    #   override a non-standard model location.
    ocr_workers: Optional[int] = None
    ocr_tessdata_path: Optional[str] = None

    # Word colorization (ink color, background color, ink coverage)
    colorizer: WordColorizerConfig = WordColorizerConfig()

    # Shape extraction (rule lines / borders)
    shapes: ShapeExtractorConfig = ShapeExtractorConfig()


# ==================================================================================================
# PDF -> Images (render ONCE)
# ==================================================================================================

def _render_pdf_bytes_to_images_bgr(
    pdf_bytes: bytes,
    dpi_scale: float,
) -> List[np.ndarray]:
    """
    Render each page to a BGR uint8 image. Returns list aligned to page_number = idx + 1.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        images: List[np.ndarray] = []
        for i in range(len(doc)):
            bitmap = doc[i].render(scale=dpi_scale)  # BGR by default
            images.append(bitmap.to_numpy())
        return images
    finally:
        doc.close()


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def run_ocr_pipeline(
    file_bytes: bytes,
    *,
    config: OCRPipelineConfig = OCRPipelineConfig(),
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Production entrypoint.

    Input:
      - file_bytes: PDF bytes (scanned doc detector decides when to call this)
      - on_stage: Optional callback for progress updates.

    Output:
      - df_words: words-level dataframe, includes line_id and is_line_start.
                  line_id is assigned via gutter-aware reading order (multi-column aware).
                  is_line_start=1 flags the leftmost word per line, useful for
                  correcting OCR noise on line-initial tokens (e.g. "m=" → bullet).
      - df_shapes: shapes dataframe (rule lines / borders)
      - df_grid_cells: grid cell dataframe (from shape enrichment)
      - df_gutters: gutter zone dataframe from gutter-aware reading order detection
    """
    # --------------------
    # Pipeline in PX
    # --------------------

    if on_stage:
        on_stage("render_images")

    # Render PDF to images
    with timed_step("pdf_rendering", logger=logger):
        images_bgr = _render_pdf_bytes_to_images_bgr(file_bytes, dpi_scale=config.dpi_scale)

    if on_stage:
        on_stage("extract_elements")

    # 1) Words (Tesseract, page-level parallel)
    with timed_step("ocr_word_extraction", logger=logger):
        ocr_workers = resolve_worker_count(config.ocr_workers, n_items=len(images_bgr))
        df_words = extract_words_from_images(
            images_bgr,
            ocr_config=TesserocrConfig(tessdata_path=config.ocr_tessdata_path),
            max_workers=ocr_workers,
        )

    # 2) Colorize words and add ink coverage
    with timed_step("word_colorization", logger=logger):
        df_words = colorize_words_df(df_words, images_bgr, config=config.colorizer, max_workers=ocr_workers)
        ink_cov = df_words["ink_coverage"].to_numpy(dtype=float)
        median_ink = float(np.median(ink_cov[ink_cov > 0])) if np.any(ink_cov > 0) else 0.0
        df_words["bold_ratio"] = (ink_cov >= config.colorizer.bold_ink_multiplier * median_ink).astype(int)

        # Drop words where the colorizer found no ink — these are Tesseract hallucinations
        # in whitespace/gutter areas where no actual text pixels exist.
        # NOTE: This operation removes rows from the df
        df_words = df_words[df_words["non_stroking_color"].notna()].reset_index(drop=True)

    # 3) Extract rule shapes (horizontal/vertical lines)
    with timed_step("shape_extraction", logger=logger):
        df_shapes = extract_shapes_df(images_bgr, config=config.shapes)

    # --------------------
    # Pipeline in PT (convert from pixels to points)
    # --------------------

    if on_stage:
        on_stage("process_layouts")

    with timed_step("conversion_px_to_pt", logger=logger):
        # Convert words bbox from PX to PT
        bbox_cols_words = ['x_left', 'x_right', 'y_top', 'y_bottom', 'width', 'height']
        for col in bbox_cols_words:
            if col in df_words.columns:
                df_words[col] = df_words[col] / config.dpi_scale

        # Derive font_size from bbox height (best available proxy from Tesseract output)
        if "height" in df_words.columns:
            df_words["font_size"] = df_words["height"]

        # Derive page_width / page_height from rendered image dimensions (converted to PT)
        if "page_number" in df_words.columns and images_bgr:
            page_dims = {
                idx + 1: (img.shape[1] / config.dpi_scale, img.shape[0] / config.dpi_scale)
                for idx, img in enumerate(images_bgr)
                if img is not None
            }
            df_words["page_width"]  = df_words["page_number"].map(lambda p: page_dims.get(p, (0, 0))[0])
            df_words["page_height"] = df_words["page_number"].map(lambda p: page_dims.get(p, (0, 0))[1])

        # Convert shapes bbox from PX to PT and remove x1,y1,x2,y2
        bbox_cols_shapes = ['x_left', 'x_right', 'y_top', 'y_bottom', 'width', 'height', 'area', 'linewidth', 'length']
        for col in bbox_cols_shapes:
            if col in df_shapes.columns:
                df_shapes[col] = df_shapes[col] / config.dpi_scale
        # area is width*height so it scales by dpi_scale^2 — correct after both dims are divided once each
        if "area" in df_shapes.columns:
            df_shapes["area"] = df_shapes["area"] / config.dpi_scale

        # Remove x1, y1, x2, y2 from shapes (not used downstream)
        cols_to_drop = ['x1', 'y1', 'x2', 'y2']
        df_shapes = df_shapes.drop(columns=[c for c in cols_to_drop if c in df_shapes.columns])

    # Detect and remove line numbers
    with timed_step("line_number_detection", logger=logger):
        df_words = detect_line_numbers(df_words, config=OCR_LINE_NUMBER_CONFIG)

        # NOTE: This operation removes rows from the df
        # Step 05c - Drop line-number words
        # Line numbers are margin artefacts that must be removed entirely — unlike
        # other annotations they cannot be represented as a meaningful block_type.
        if "line_number_flag" in df_words.columns:
            df_words = df_words[~df_words["line_number_flag"]].copy()


    # Enrich df_shapes for the gutter detector
    with timed_step("shape_processing", logger=logger):
        df_shapes, df_grid_cells = process_shapes(df_shapes)

    # 4) Assign reading order (gutter-aware)
    with timed_step("reading_order", logger=logger):
        df_words, df_gutters = assign_reading_order(
            df_words, df_shapes, df_grid_cells, max_workers=config.ocr_workers
        )
        df_words = df_words.drop(columns=["center_bucket"], errors="ignore")

        # Flag the leftmost word in each line (useful for OCR noise correction,
        # e.g. "m=" or "e" at line start may be a misread bullet)
        min_x = df_words.groupby(["page_number", "line_id"])["x_left"].transform("min")
        df_words["is_line_start"] = (df_words["x_left"] == min_x).astype(int)

    # 5) Text cleaning (runs after is_line_start is available for bullet detection)
    with timed_step("text_cleaning", logger=logger):
        df_words = clean_words_df(df_words, df_shapes)
        # Remove all blanked out words
        df_words = df_words[df_words["text"].fillna("").str.strip() != ""].reset_index(drop=True)
        df_words = add_calculated_text_features(df_words)

    # 6) Estimate font size (per layout, too noisy on a line level)
    with timed_step("font_size_estimation", logger=logger):
        df_words = assign_layouts(
            df_words, line_level=False, config=LayoutConfig.for_ocr_prelim()
        )
        df_words = estimate_ocr_font_sizes(df_words, method="word")
        
        # Drop the layout_id column (will later be assessed more accurately, this one was just to set the font sizes)
        #df_words = df_words.drop(columns=["layout_id"], errors="ignore")

    return df_words, df_shapes, df_grid_cells, df_gutters


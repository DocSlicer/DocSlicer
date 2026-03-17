# ocr/ocr_orchestrator.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
from time import perf_counter

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import cv2

# Suppress MPS pin_memory warning
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not supported on MPS.*", category=UserWarning)

from .step_01_word_extractor import extract_words_from_images
from .step_02_word_colorizer import colorize_words_df
from .step_03_shape_extractor import extract_shapes_df, ShapeExtractorConfig
from .step_04_text_cleaner import clean_words_df
from .._utils.line_merger import assign_line_id, LineMergerConfig


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class OCRPipelineConfig:
    # Rendering & coordinate conversion
    # 2.0 ≈ ~144 DPI effective for typical 72 DPI PDF coords
    # Also used to convert final output from pixels (PX) to points (PT)
    dpi_scale: float = 2.0

    # EasyOCR
    #easyocr: EasyOCRConfig = EasyOCRConfig(langs=("en",), use_gpu=True)


    # Shape extraction (rule lines / borders)
    shapes: ShapeExtractorConfig = ShapeExtractorConfig()

    # Temp line assignment (tolerances in PT)
    temp_lines: LineMergerConfig = LineMergerConfig()


# ==================================================================================================
# PDF -> Images (render ONCE)
# ==================================================================================================

def _pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    """
    Convert PyMuPDF pixmap samples to an OpenCV BGR image (uint8).
    """
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        # RGBA -> BGR
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # RGB -> BGR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _render_pdf_bytes_to_images_bgr(
    pdf_bytes: bytes,
    dpi_scale: float,
) -> List[np.ndarray]:
    """
    Render each page to a BGR uint8 image. Returns list aligned to page_number = idx + 1.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        images: List[np.ndarray] = []
        mat = fitz.Matrix(dpi_scale, dpi_scale)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(_pixmap_to_bgr(pix))

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
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, float]]]:
    """
    Production entrypoint.

    Input:
      - file_bytes: PDF bytes (scanned doc detector decides when to call this)

    Output:
      - df_words: words-level dataframe, includes temp_line_id and is_line_start.
                  temp_line_id is a provisional line grouping (assumes single-column layout;
                  multi-column correction happens later in the PDF pipeline).
                  is_line_start=1 flags the leftmost word per temp line, useful for
                  correcting OCR noise on line-initial tokens (e.g. "m=" → bullet).
      - df_shapes: shapes dataframe (rule lines / borders)
      - timings: list of (step_name, duration_seconds) tuples
    """
    timings = []

    # --------------------
    # Pipeline in PX
    # --------------------
    
    # Render PDF to images
    t0 = perf_counter()
    images_bgr = _render_pdf_bytes_to_images_bgr(file_bytes, dpi_scale=config.dpi_scale)
    timings.append(("PDF rendering", perf_counter() - t0))

    # 1) Words (Tesseract)
    t0 = perf_counter()
    df_words = extract_words_from_images(images_bgr) # Pure Tesseract version
    timings.append(("STEP 1: OCR word extraction", perf_counter() - t0))

    # 2) Colorize words and add ink coverage
    t0 = perf_counter()
    df_words = colorize_words_df(df_words, images_bgr)
    timings.append(("STEP 2: Word colorization", perf_counter() - t0))

    # 3) Extract rule shapes (horizontal/vertical lines)
    t0 = perf_counter()
    df_shapes = extract_shapes_df(images_bgr, words_df=df_words, config=config.shapes)
    timings.append(("STEP 3: Shape extraction", perf_counter() - t0))

    # --------------------
    # Pipeline in PT (convert from pixels to points)
    # --------------------
    
    t0 = perf_counter()
    
    # Convert words bbox from PX to PT
    bbox_cols_words = ['x_left', 'x_right', 'y_top', 'y_bottom', 'width', 'height']
    for col in bbox_cols_words:
        if col in df_words.columns:
            df_words[col] = df_words[col] / config.dpi_scale
    
    # Convert shapes bbox from PX to PT and remove x1,y1,x2,y2
    bbox_cols_shapes = ['x_left', 'x_right', 'y_top', 'y_bottom', 'length']
    for col in bbox_cols_shapes:
        if col in df_shapes.columns:
            df_shapes[col] = df_shapes[col] / config.dpi_scale

    # Remove x1, y1, x2, y2 from shapes (not used downstream)
    cols_to_drop = ['x1', 'y1', 'x2', 'y2']
    df_shapes = df_shapes.drop(columns=[c for c in cols_to_drop if c in df_shapes.columns])
    
    timings.append(("Conversion PX -> PT", perf_counter() - t0))

    # 4) Assign provisional temp lines onto words
    # Uses the shared line_merger utility. Called after PX→PT so tolerances are in PT.
    # This is a single-column assumption; multi-column correction happens later in the PDF pipeline.
    t0 = perf_counter()
    df_words = assign_line_id(df_words, y_alignment="center", config=config.temp_lines)
    df_words = df_words.rename(columns={"line_id": "temp_line_id"})
    df_words = df_words.drop(columns=["center_bucket"], errors="ignore")

    # Flag the leftmost word in each temp line (useful for OCR noise correction,
    # e.g. "m=" or "e" at line start may be a misread bullet)
    min_x = df_words.groupby(["page_number", "temp_line_id"])["x_left"].transform("min")
    df_words["is_line_start"] = (df_words["x_left"] == min_x).astype(int)
    timings.append(("STEP 4: Temp line assignment", perf_counter() - t0))

    # 5) Text cleaning (runs after is_line_start is available for bullet detection)
    t0 = perf_counter()
    df_words = clean_words_df(df_words)
    timings.append(("STEP 5: Text cleaning", perf_counter() - t0))

    return df_words, df_shapes, timings


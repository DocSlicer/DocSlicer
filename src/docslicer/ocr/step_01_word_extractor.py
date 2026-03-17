# ocr/step_01_word_extractor.py  (Tesseract version, PURE extractor)
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pandas as pd
import pytesseract


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class TesseractConfig:
    lang: str = "eng"
    psm: int = 6
    oem: int = 3


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def extract_words_from_images(
    images_bgr: List[np.ndarray],
    *,
    ocr_config: TesseractConfig = TesseractConfig(),
    include_text_orientation: bool = True,
) -> pd.DataFrame:
    """
    Tesseract-only word extractor (pure).

    Input:
      - images_bgr: list of uint8 BGR images, one per page (page_number = idx + 1)

    Output schema:
      page_number, word_id, text,
      x_left, x_right, y_top, y_bottom,
      width, height,
      text_orientation (optional)

    Notes:
      - No cleaning.
      - No trimming.
      - No garbage filtering.
      - No bullet logic.
      - Just raw OCR geometry.
      - Sorting is stable: (page_number, y_top, x_left).
      - word_id is stable AFTER sort (1..N).
    """

    base_cols = [
        "page_number",
        "word_id",
        "text",
        "x_left",
        "x_right",
        "y_top",
        "y_bottom",
        "width",
        "height",
    ]
    if include_text_orientation:
        base_cols.append("text_orientation")

    if not images_bgr:
        return pd.DataFrame(columns=base_cols)

    tcfg = f"--oem {int(ocr_config.oem)} --psm {int(ocr_config.psm)}"

    rows = []

    for page_idx, img_bgr in enumerate(images_bgr):
        if img_bgr is None:
            continue

        # Tesseract prefers grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            lang=ocr_config.lang,
            config=tcfg,
            output_type=pytesseract.Output.DICT,
        )

        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()

            # Tesseract uses conf = -1 for non-text layout rows
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0

            if conf < 0:
                continue

            left   = float(data["left"][i])
            top    = float(data["top"][i])
            width  = float(data["width"][i])
            height = float(data["height"][i])

            row = {
                "page_number": int(page_idx + 1),
                "text": text,
                "x_left": left,
                "x_right": left + width,
                "y_top": top,
                "y_bottom": top + height,
                "width": width,
                "height": height,
            }

            if include_text_orientation:
                # Tesseract provides axis-aligned boxes only.
                # Without running OSD, this is conservatively LTR.
                row["text_orientation"] = "LTR"

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    # Stable deterministic ordering
    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)

    # Stable word_id after sort
    df.insert(1, "word_id", np.arange(1, len(df) + 1, dtype=np.int64))

    return df

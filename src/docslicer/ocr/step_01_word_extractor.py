# ocr/step_01_word_extractor_tesserocr.py  (tesserocr version, PURE extractor)
#
# Drop-in alternative to step_01_word_extractor.py that talks to libtesseract
# in-process (no subprocess / temp-file / TSV round-trip).
#
# Public entry point:
#   - extract_words_from_images : page-level parallel (process pool); max_workers=1
#                                 runs the in-process serial path.
#
# Output is value-for-value identical to the pytesseract extractor (and across any
# worker count) because the global sort + word_id assignment is always done once,
# at the end, in the parent.
from __future__ import annotations

import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from .._utils.parallel import warn_pool_fell_back
from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import tesserocr
from PIL import Image


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class TesserocrConfig:
    lang: str = "eng"
    psm: int = 3 # 6 is slightly faster but 1) less accurate on text/numbers, 2) misses non LTR text
    oem: int = 3
    # Path to the tessdata directory. If None, resolution order is:
    #   TESSDATA_PREFIX env -> auto-detect via the system `tesseract` CLI.
    # Only set this to override (e.g. a non-standard model location).
    tessdata_path: Optional[str] = None


BASE_COLS_NO_ORIENT = [
    "page_number",
    "word_id",
    "text",
    "x_left",
    "x_right",
    "y_top",
    "y_bottom",
    "width",
    "height",
    "ocr_confidence",
    "font_pointsize",
]

_WRITING_DIRECTION_MAP = {
    tesserocr.WritingDirection.LEFT_TO_RIGHT: "LTR",
    tesserocr.WritingDirection.RIGHT_TO_LEFT: "RTL",
    tesserocr.WritingDirection.TOP_TO_BOTTOM: "TTB",
}

def _word_orientation(r) -> str:
    """
    Use block-level Orientation (physical page rotation) rather than WordDirection
    (script direction). WordDirection detects LTR/RTL scripts but doesn't capture
    whether a text block is physically rotated on the page.
    """
    try:
        orientation, writing_dir, _, _ = r.Orientation()
    except Exception:
        return "LTR"
    if orientation == tesserocr.Orientation.PAGE_UP:
        return _WRITING_DIRECTION_MAP.get(writing_dir, "LTR")
    if orientation == tesserocr.Orientation.PAGE_RIGHT:
        return "TTB"
    if orientation == tesserocr.Orientation.PAGE_LEFT:
        return "BTT"
    return "LTR"  # PAGE_DOWN (upside-down) — treat as LTR fallback


# ==================================================================================================
# SHARED INTERNALS
# ==================================================================================================

@lru_cache(maxsize=1)
def _autodetect_tessdata() -> Optional[str]:
    """
    Locate the tessdata directory via the installed `tesseract` CLI.

    The prebuilt tesserocr wheel ships its own libtesseract that does NOT know
    where the traineddata models live, so a bare `pip install` + `brew install
    tesseract` (or apt) would otherwise fail with empty languages. The system
    `tesseract` binary, however, knows its own model path — so we ask it.
    """
    # 1) Parse `tesseract --list-langs`, which prints: List of available languages in "<dir>" (N):
    try:
        out = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=2.0,
        )
        m = re.search(r'in "(.+?)"', (out.stdout or "") + (out.stderr or ""))
        if m and os.path.isdir(m.group(1)):
            return m.group(1)
    except Exception:
        pass

    # 2) Fall back to the standard <bin>/../share/tessdata layout next to the binary.
    exe = shutil.which("tesseract")
    if exe:
        cand = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "tessdata")
        if os.path.isdir(cand):
            return cand

    return None


def _resolve_tessdata_path(ocr_config: TesserocrConfig) -> str:
    # Explicit config / env always win.
    explicit = ocr_config.tessdata_path or os.environ.get("TESSDATA_PREFIX")
    if explicit:
        return explicit

    auto = _autodetect_tessdata()
    if auto:
        return auto

    raise RuntimeError(
        "Could not locate Tesseract traineddata. Install the Tesseract engine "
        "(e.g. `brew install tesseract` or `apt install tesseract-ocr`), or set "
        "TESSDATA_PREFIX / TesserocrConfig.tessdata_path to the directory "
        "containing eng.traineddata."
    )


def _new_api(ocr_config: TesserocrConfig) -> "tesserocr.PyTessBaseAPI":
    return tesserocr.PyTessBaseAPI(
        path=_resolve_tessdata_path(ocr_config),
        lang=ocr_config.lang,
        psm=int(ocr_config.psm),
        oem=int(ocr_config.oem),
    )


def _extract_page_rows(api, img_bgr, page_idx, include_text_orientation) -> list:
    """OCR a single page with an already-initialized API; return raw row dicts."""
    if img_bgr is None:
        return []

    # Match the pytesseract path byte-for-byte: feed grayscale.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    api.SetImage(Image.fromarray(gray))
    api.Recognize()

    ri = api.GetIterator()
    if ri is None:
        return []

    level = tesserocr.RIL.WORD
    rows = []
    for r in tesserocr.iterate_level(ri, level):
        try:
            raw = r.GetUTF8Text(level)
        except RuntimeError:
            continue
        text = (raw or "").strip()
        # Drop blank / invisible-only words (PSM 3 hits lines and shapes as text)
        if not text or not text.strip('\x00\xa0​‌‍﻿'):
            continue

        # Word-level confidence is 0..100 for recognized words; the pytesseract
        # conf == -1 rows simply never appear at this iteration level.
        try:
            conf = float(r.Confidence(level))
        except Exception:
            conf = -1.0
        if conf < 0:
            continue

        bbox = r.BoundingBox(level)  # (left, top, right, bottom) or None
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        left = float(left)
        top = float(top)
        width = float(right - left)
        height = float(bottom - top)

        try:
            fa = r.WordFontAttributes() or {}
        except Exception:
            fa = {}

        row = {
            "page_number": int(page_idx + 1),
            "text": text,
            "x_left": left,
            "x_right": left + width,
            "y_top": top,
            "y_bottom": top + height,
            "width": width,
            "height": height,
            "ocr_confidence": conf,
            "font_pointsize": fa.get("pointsize"),
        }
        if include_text_orientation:
            row["text_orientation"] = _word_orientation(r)
        rows.append(row)

    return rows


def _finalize(rows: list, include_text_orientation: bool) -> pd.DataFrame:
    """Identical finalization for serial + parallel: sort once, assign word_id once."""
    base_cols = list(BASE_COLS_NO_ORIENT)
    if include_text_orientation:
        base_cols.append("text_orientation")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)
    df.insert(1, "word_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


# ==================================================================================================
# SERIAL (internal — used directly when max_workers resolves to 1)
# ==================================================================================================

def _extract_words_serial(
    images_bgr: List[np.ndarray],
    *,
    ocr_config: TesserocrConfig,
    include_text_orientation: bool,
) -> pd.DataFrame:
    """tesserocr word extraction in-process, one reused API. Mirrors the pytesseract extractor exactly."""
    if not images_bgr:
        return _finalize([], include_text_orientation)

    rows = []
    with _new_api(ocr_config) as api:  # one API reused across all pages
        for page_idx, img_bgr in enumerate(images_bgr):
            rows.extend(_extract_page_rows(api, img_bgr, page_idx, include_text_orientation))

    return _finalize(rows, include_text_orientation)


# ==================================================================================================
# PUBLIC API (page-level parallel; falls back to serial for a single worker)
# ==================================================================================================

def _worker_init():
    # Each worker stays single-threaded internally so W processes map cleanly to W
    # cores instead of oversubscribing via Tesseract's OpenMP threads.
    # (Also set in the parent before pool creation; spawned children inherit it, but
    # we set it here too as a belt-and-suspenders guard.)
    os.environ["OMP_THREAD_LIMIT"] = "1"


def _worker_run_chunk(args):
    """Top-level (picklable) worker: OCR a chunk of (page_idx, img) pairs with one API."""
    page_chunk, ocr_config, include_text_orientation = args
    rows = []
    with _new_api(ocr_config) as api:
        for page_idx, img_bgr in page_chunk:
            rows.extend(_extract_page_rows(api, img_bgr, page_idx, include_text_orientation))
    return rows


def _chunk_pages(indexed_pages: list, n_chunks: int) -> list:
    """Round-robin pages into n_chunks so slow pages spread across workers."""
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(indexed_pages):
        chunks[i % n_chunks].append(item)
    return [c for c in chunks if c]


def extract_words_from_images(
    images_bgr: List[np.ndarray],
    *,
    ocr_config: TesserocrConfig = TesserocrConfig(),
    include_text_orientation: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    tesserocr word extractor (pure), page-level parallel via a process pool.
    Output is identical to the serial path (global sort + word_id assignment
    happen once here in the parent).

    max_workers:
      - None  -> safe fallback of min(#pages, os.cpu_count()). Callers that know
                 their machine (e.g. the orchestrator) should pass an explicit,
                 policy-resolved value via _utils.cpu.resolve_worker_count.
      - 1     -> runs the serial path (no process overhead)
    """
    if not images_bgr:
        return _finalize([], include_text_orientation)

    indexed_pages = [(i, img) for i, img in enumerate(images_bgr) if img is not None]
    if not indexed_pages:
        return _finalize([], include_text_orientation)

    if max_workers is None:
        max_workers = min(len(indexed_pages), os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, len(indexed_pages)))

    if max_workers == 1:
        return _extract_words_serial(
            images_bgr, ocr_config=ocr_config, include_text_orientation=include_text_orientation
        )

    # Children inherit this; cap Tesseract's internal OpenMP threading.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    chunks = _chunk_pages(indexed_pages, max_workers)
    payload = [(chunk, ocr_config, include_text_orientation) for chunk in chunks]

    all_rows = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
            for rows in ex.map(_worker_run_chunk, payload):
                all_rows.extend(rows)
    except BrokenProcessPool:
        warn_pool_fell_back("OCR word extraction")
        return _extract_words_serial(
            images_bgr, ocr_config=ocr_config, include_text_orientation=include_text_orientation
        )

    return _finalize(all_rows, include_text_orientation)

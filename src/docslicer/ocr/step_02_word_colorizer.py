# ocr/step_02_word_colorizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class WordColorizerConfig:
    # Expand bbox slightly to stabilize sampling
    text_pad_px: int = 2

    # Background sampling: ring thickness around bbox
    bg_ring_px: int = 6

    # Ink sampling: pixels darker than this (grayscale) are "ink"
    ink_dark_thr: int = 180

    # --- background robustness + normalization ---
    # Ignore dark pixels in the ring when estimating background (prevents ink bleed / border bleed)
    bg_exclude_dark_thr: int = 220

    # Quantize colors to reduce drift (channel rounding)
    bg_quant_step: int = 8
    ink_quant_step: int = 16

    # Snap near-white backgrounds to #FFFFFF
    bg_snap_white_min_rgb: int = 248

    # Also snap near-neutral bright backgrounds to a canonical light gray
    bg_neutral_max_chroma: int = 8       # max(R,G,B)-min(R,G,B)
    bg_neutral_min_brightness: int = 232
    bg_neutral_snap_hex: str = "#F8F8F8"

    # Bold detection: a word is bold if its ink_coverage >= this multiplier * median ink_coverage
    bold_ink_multiplier: float = 1.2


# ==================================================================================================
# COLOR HELPERS
# ==================================================================================================

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _clip_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), W - 1)
    y2 = min(int(y2), H - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _safe_median_bgr(pixels_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Median is much more stable than mean for scanned/compressed pages.
    pixels_bgr: (N, 3) uint8/float
    """
    if pixels_bgr is None or len(pixels_bgr) == 0:
        return None
    med = np.median(pixels_bgr.astype(np.float32), axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _bgr_tuple_to_rgb_int(bgr: Optional[Tuple[float, float, float]]) -> Optional[Tuple[int, int, int]]:
    if bgr is None:
        return None
    b, g, r = [int(round(v)) for v in bgr]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return (r, g, b)


def _rgb_to_hex(rgb: Optional[Tuple[int, int, int]]) -> Optional[str]:
    if rgb is None:
        return None
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _quantize_rgb(rgb: Optional[Tuple[int, int, int]], step: int) -> Optional[Tuple[int, int, int]]:
    if rgb is None:
        return None
    if step <= 1:
        return rgb

    def q(v: int) -> int:
        return int(round(v / step) * step)

    r, g, b = rgb
    r = max(0, min(255, q(r)))
    g = max(0, min(255, q(g)))
    b = max(0, min(255, q(b)))
    return (r, g, b)


def _normalize_bg_rgb(rgb: Optional[Tuple[int, int, int]], cfg: WordColorizerConfig) -> Optional[Tuple[int, int, int]]:
    """
    Normalize background to reduce scan drift while preserving real shading differences.
    Steps:
      1) quantize in small steps
      2) snap near-white to pure white
      3) snap near-neutral bright to canonical light gray (optional)
    """
    if rgb is None:
        return None

    rgb_q = _quantize_rgb(rgb, cfg.bg_quant_step)
    if rgb_q is None:
        return None

    r, g, b = rgb_q
    mn = min(r, g, b)
    mx = max(r, g, b)
    chroma = mx - mn

    # snap to white
    if mn >= int(cfg.bg_snap_white_min_rgb):
        return (255, 255, 255)

    # snap near-neutral bright whites to canonical gray/white bucket
    if chroma <= int(cfg.bg_neutral_max_chroma) and mn >= int(cfg.bg_neutral_min_brightness):
        # parse cfg.bg_neutral_snap_hex
        hx = cfg.bg_neutral_snap_hex.lstrip("#")
        if len(hx) == 6:
            rr = int(hx[0:2], 16)
            gg = int(hx[2:4], 16)
            bb = int(hx[4:6], 16)
            return (rr, gg, bb)

    return rgb_q


def _normalize_ink_rgb(rgb: Optional[Tuple[int, int, int]], cfg: WordColorizerConfig) -> Optional[Tuple[int, int, int]]:
    """
    Ink normalization: mostly quantization only (you typically *don't* want to snap ink).
    """
    return _quantize_rgb(rgb, cfg.ink_quant_step)


# ==================================================================================================
# SAMPLERS
# ==================================================================================================

def _sample_background_color_bgr_median(
    img_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    ring_px: int,
    exclude_dark_thr: int,
) -> Optional[Tuple[float, float, float]]:
    """
    Background = median color of a "ring" around the bbox, excluding dark pixels.

    outer_rect(x1-ring..x2+ring) minus inner_rect(x1..x2)
    """
    H, W = img_bgr.shape[:2]
    clipped = _clip_box(x1, y1, x2, y2, W, H)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped

    rx1 = max(x1 - ring_px, 0)
    ry1 = max(y1 - ring_px, 0)
    rx2 = min(x2 + ring_px, W - 1)
    ry2 = min(y2 + ring_px, H - 1)

    # ring mask
    mask = np.zeros((ry2 - ry1 + 1, rx2 - rx1 + 1), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (rx2 - rx1, ry2 - ry1), 255, thickness=-1)
    cv2.rectangle(mask, (x1 - rx1, y1 - ry1), (x2 - rx1, y2 - ry1), 0, thickness=-1)

    patch = img_bgr[ry1 : ry2 + 1, rx1 : rx2 + 1]
    ring_pixels = patch[mask == 255]
    if ring_pixels is None or len(ring_pixels) == 0:
        return None

    # exclude dark pixels to avoid border/ink bleed
    gray = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    keep = gray >= int(exclude_dark_thr)
    ring_pixels = ring_pixels[keep]
    if ring_pixels is None or len(ring_pixels) == 0:
        # fallback: median on all ring pixels
        ring_pixels = patch[mask == 255]

    return _safe_median_bgr(ring_pixels)

# ==================================================================================================
# BOLDNESS ESTIMATOR
# ==================================================================================================

def _sample_ink_color_and_coverage_bgr_median(
    img_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    ink_dark_thr: int,
) -> Tuple[Optional[Tuple[float, float, float]], float]:
    """
    Ink color = median of "dark" pixels inside bbox.
    ink_coverage = fraction of pixels inside bbox considered ink.
    """
    H, W = img_bgr.shape[:2]
    clipped = _clip_box(x1, y1, x2, y2, W, H)
    if clipped is None:
        return None, 0.0
    x1, y1, x2, y2 = clipped

    patch = img_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return None, 0.0

    gray = _to_gray(patch)
    ink_mask = gray < int(ink_dark_thr)
    ink_cov = float(ink_mask.mean())  # [0..1]

    ink_pixels = patch[ink_mask]
    if ink_pixels is None or len(ink_pixels) == 0:
        return None, ink_cov

    return _safe_median_bgr(ink_pixels), ink_cov


# ==================================================================================================
# ITALIC DETECTION (TODO)
# ==================================================================================================
#
# Current experiments are too noisy because they measure whole-word geometry instead of
# true stem/slant behavior. A better approach should focus only on informative glyph parts.
#
# General idea:
# 1) Work on the word image patch only after OCR bbox extraction.
# 2) Threshold ink pixels and remove tiny noise components.
# 3) Ignore round / non-informative character parts as much as possible:
#    - exclude blobs that are too round or too wide relative to height
#    - exclude components with low eccentricity
#    - exclude near-circular letters/parts like o, e, c, 0 where possible
# 4) Focus only on upright or slightly right-tilted stroke-like components:
#    - keep tall, narrow connected components
#    - keep component orientations near vertical
#    - allow small right tilt for italic candidates
#    - exclude strong "/" diagonals like the right leg of "A"
#    - exclude strong "\" diagonals
# 5) Estimate raw slant from the remaining stroke-like components only.
# 6) Aggregate at line/page level afterwards:
#    - compute page slant / scan skew
#    - subtract page slant from raw word slant
#    - use line-level median/mean for stability
# 7) Treat italic as a proxy signal, not ground truth metadata.
#
# Likely useful signals to test:
# - connected-component PCA orientation
# - component eccentricity / aspect ratio
# - vertical stroke filtering
# - edge orientation histogram restricted to near-vertical edges
# - line-level aggregation instead of trusting single short words
#
# Important:
# - short words like "and", "of", "to" will remain noisy
# - round words should probably be marked as "insufficient evidence"
# - goal is not perfect italic recovery, but a stable visual slant proxy

# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def colorize_words_df(
    words_df: pd.DataFrame,
    images_bgr: List[np.ndarray],
    *,
    config: WordColorizerConfig = WordColorizerConfig(),
) -> pd.DataFrame:
    """
    Adds *raw* and *normalized* color columns.

    Output columns added:
      - non_stroking_color_hex_raw
      - background_non_stroking_color_hex_raw
      - non_stroking_color_hex
      - background_non_stroking_color_hex
      - ink_coverage
    """
    if words_df is None or words_df.empty:
        return pd.DataFrame() if words_df is None else words_df.copy()

    required = ["page_number", "x_left", "x_right", "y_top", "y_bottom"]
    missing = [c for c in required if c not in words_df.columns]
    if missing:
        raise ValueError(f"words_df missing required columns: {missing}")

    out = words_df.copy()

    n = len(out)

    ink_hex_raw = np.empty(n, dtype=object)
    bg_hex_raw = np.empty(n, dtype=object)
    ink_hex_norm = np.empty(n, dtype=object)
    bg_hex_norm = np.empty(n, dtype=object)
    ink_cov = np.zeros(n, dtype=float)

    pad = int(config.text_pad_px)
    ring = int(config.bg_ring_px)

    page = out["page_number"].to_numpy()
    x_left = out["x_left"].to_numpy(dtype=float)
    x_right = out["x_right"].to_numpy(dtype=float)
    y_top = out["y_top"].to_numpy(dtype=float)
    y_bottom = out["y_bottom"].to_numpy(dtype=float)

    for i in range(n):
        p = int(page[i])
        if p <= 0 or p > len(images_bgr):
            ink_hex_raw[i] = None
            bg_hex_raw[i] = None
            ink_hex_norm[i] = None
            bg_hex_norm[i] = None
            ink_cov[i] = 0.0
            continue

        img_bgr = images_bgr[p - 1]
        if img_bgr is None:
            ink_hex_raw[i] = None
            bg_hex_raw[i] = None
            ink_hex_norm[i] = None
            bg_hex_norm[i] = None
            ink_cov[i] = 0.0
            continue

        # Expand bbox for more stable sampling
        x1 = int(round(x_left[i])) - pad
        y1 = int(round(y_top[i])) - pad
        x2 = int(round(x_right[i])) + pad
        y2 = int(round(y_bottom[i])) + pad

        ink_bgr, cov = _sample_ink_color_and_coverage_bgr_median(
            img_bgr,
            x1, y1, x2, y2,
            ink_dark_thr=int(config.ink_dark_thr),
        )
        bg_bgr = _sample_background_color_bgr_median(
            img_bgr,
            x1, y1, x2, y2,
            ring_px=ring,
            exclude_dark_thr=int(config.bg_exclude_dark_thr),
        )

        # --- raw hex ---
        ink_rgb_raw = _bgr_tuple_to_rgb_int(ink_bgr)
        bg_rgb_raw = _bgr_tuple_to_rgb_int(bg_bgr)
        ink_hex_raw[i] = _rgb_to_hex(ink_rgb_raw)
        bg_hex_raw[i] = _rgb_to_hex(bg_rgb_raw)
        ink_cov[i] = float(cov)

        # --- normalized hex ---
        ink_rgb_norm = _normalize_ink_rgb(ink_rgb_raw, config)
        bg_rgb_norm = _normalize_bg_rgb(bg_rgb_raw, config)
        ink_hex_norm[i] = _rgb_to_hex(ink_rgb_norm)
        bg_hex_norm[i] = _rgb_to_hex(bg_rgb_norm)

    # keep both raw and normalized for debugging + stability
    out["non_stroking_color_hex_raw"] = ink_hex_raw
    out["background_non_stroking_color_hex_raw"] = bg_hex_raw

    out["non_stroking_color_hex"] = ink_hex_norm
    out["background_non_stroking_color_hex"] = bg_hex_norm

    out["ink_coverage"] = ink_cov

    # Bold: compute the median ink_coverage across all words with any ink,
    # then flag words whose coverage exceeds the median by the configured multiplier.
    # Using the median (not mean) makes it robust to outliers like large black logos.
    median_ink = float(np.median(ink_cov[ink_cov > 0])) if np.any(ink_cov > 0) else 0.0
    out["bold_ratio"] = (ink_cov >= config.bold_ink_multiplier * median_ink).astype(int)

    return out

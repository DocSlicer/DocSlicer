# ocr/step_02_word_colorizer.py
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
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

    # Post-hoc color snapping: snap infrequent ink colors to the nearest high-frequency
    # canonical within this RGB Euclidean distance. 0 = disabled.
    # 30 collapses scan-noise variants (~1-2 channel steps apart) without merging
    # visually distinct colors (e.g. dark-gray vs. near-black stay separate).
    ink_snap_threshold: int = 30


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

    For uint8 input (the common case) this uses a per-channel histogram
    (np.bincount) which is O(N) and avoids both the full sort that np.median
    performs and the float32 copy of the whole pixel array. The histogram
    returns the lower median, which after rounding + quantization is
    indistinguishable from np.median for our purposes.
    """
    if pixels_bgr is None or len(pixels_bgr) == 0:
        return None

    if pixels_bgr.dtype != np.uint8:
        med = np.median(pixels_bgr, axis=0)
        return (float(med[0]), float(med[1]), float(med[2]))

    n = pixels_bgr.shape[0]
    target = (n - 1) // 2  # 0-based index of the lower median in sorted order
    med = []
    for c in range(3):
        counts = np.bincount(pixels_bgr[:, c], minlength=256)
        med.append(float(np.searchsorted(np.cumsum(counts), target + 1)))
    return (med[0], med[1], med[2])


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

    # Gather the ring as the 4 border bands around the inner box directly via
    # slicing, instead of allocating a mask + drawing two rectangles + boolean
    # fancy-indexing per word. The bands are disjoint and cover exactly the
    # outer-rect-minus-inner-box region (corners go to the top/bottom bands).
    parts = []
    if y1 > ry1:  # top band (full width)
        parts.append(img_bgr[ry1:y1, rx1 : rx2 + 1].reshape(-1, 3))
    if y2 < ry2:  # bottom band (full width)
        parts.append(img_bgr[y2 + 1 : ry2 + 1, rx1 : rx2 + 1].reshape(-1, 3))
    if x1 > rx1:  # left band (inner rows only)
        parts.append(img_bgr[y1 : y2 + 1, rx1:x1].reshape(-1, 3))
    if x2 < rx2:  # right band (inner rows only)
        parts.append(img_bgr[y1 : y2 + 1, x2 + 1 : rx2 + 1].reshape(-1, 3))

    if not parts:
        return None
    ring_pixels = np.concatenate(parts, axis=0)
    if len(ring_pixels) == 0:
        return None

    # exclude dark pixels to avoid border/ink bleed; vectorized luminance
    # (cv2 BGR2GRAY weights) instead of a per-word cv2.cvtColor with reshape
    gray = (
        ring_pixels[:, 0].astype(np.float32) * 0.114
        + ring_pixels[:, 1].astype(np.float32) * 0.587
        + ring_pixels[:, 2].astype(np.float32) * 0.299
    )
    kept = ring_pixels[gray >= float(exclude_dark_thr)]
    if len(kept) == 0:
        # fallback: median on all ring pixels
        kept = ring_pixels

    return _safe_median_bgr(kept)

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
    ink_cov = float(np.count_nonzero(ink_mask)) / ink_mask.size  # [0..1]

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
# COLOR SNAPPING (post-hoc, document-level)
# ==================================================================================================

def _snap_similar_colors(series: pd.Series, threshold: int) -> pd.Series:
    """
    Snap scan-noise color variants to the nearest high-frequency canonical.

    Algorithm: visit colors in descending frequency order. Each color either
    joins the nearest already-established canonical (if within `threshold` RGB
    Euclidean distance) or becomes a new canonical itself.

    This runs after per-word quantization, so it collapses residual drift
    (e.g. #404040 vs #405050) without touching truly distinct ink colors.
    """
    if threshold <= 0:
        return series

    freq = series.dropna().value_counts()
    if freq.empty:
        return series

    def _parse(h: str):
        h = h.lstrip("#")
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32)

    canon_rgbs: list = []   # np.array per canonical
    canon_hexes: list = []  # corresponding hex string
    mapping: dict = {}

    for hex_color in freq.index:
        try:
            rgb = _parse(hex_color)
        except (ValueError, IndexError):
            mapping[hex_color] = hex_color
            continue

        best_dist = float("inf")
        best_idx = -1
        for i, can_rgb in enumerate(canon_rgbs):
            d = float(np.linalg.norm(rgb - can_rgb))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_dist <= threshold:
            mapping[hex_color] = canon_hexes[best_idx]
        else:
            canon_rgbs.append(rgb)
            canon_hexes.append(hex_color)
            mapping[hex_color] = hex_color

    return series.map(mapping)


# ==================================================================================================
# WORKER (picklable — must be top-level)
# ==================================================================================================

def _colorize_page_worker(args):
    """Colorize all words on a single page. Runs in a worker process or inline."""
    indices, x_left, x_right, y_top, y_bottom, img_bgr, config = args

    n = len(indices)
    ink_hex_raw = np.empty(n, dtype=object)
    bg_hex_raw = np.empty(n, dtype=object)
    ink_hex_norm = np.empty(n, dtype=object)
    bg_hex_norm = np.empty(n, dtype=object)
    ink_cov = np.zeros(n, dtype=float)

    pad = int(config.text_pad_px)
    ring = int(config.bg_ring_px)

    for i in range(n):
        x1 = int(round(x_left[i])) - pad
        y1 = int(round(y_top[i])) - pad
        x2 = int(round(x_right[i])) + pad
        y2 = int(round(y_bottom[i])) + pad

        ink_bgr, cov = _sample_ink_color_and_coverage_bgr_median(
            img_bgr, x1, y1, x2, y2, ink_dark_thr=int(config.ink_dark_thr),
        )
        bg_bgr = _sample_background_color_bgr_median(
            img_bgr, x1, y1, x2, y2,
            ring_px=ring, exclude_dark_thr=int(config.bg_exclude_dark_thr),
        )

        ink_rgb_raw = _bgr_tuple_to_rgb_int(ink_bgr)
        bg_rgb_raw = _bgr_tuple_to_rgb_int(bg_bgr)
        ink_hex_raw[i] = _rgb_to_hex(ink_rgb_raw)
        bg_hex_raw[i] = _rgb_to_hex(bg_rgb_raw)
        ink_cov[i] = float(cov)

        ink_rgb_norm = _normalize_ink_rgb(ink_rgb_raw, config)
        bg_rgb_norm = _normalize_bg_rgb(bg_rgb_raw, config)
        ink_hex_norm[i] = _rgb_to_hex(ink_rgb_norm)
        bg_hex_norm[i] = _rgb_to_hex(bg_rgb_norm)

    return indices, ink_hex_raw, bg_hex_raw, ink_hex_norm, bg_hex_norm, ink_cov


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def colorize_words_df(
    words_df: pd.DataFrame,
    images_bgr: List[np.ndarray],
    *,
    config: WordColorizerConfig = WordColorizerConfig(),
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adds *raw* and *normalized* color columns. Page-level parallel via a process
    pool when max_workers > 1; falls back to serial for single-page or max_workers=1.

    Output columns added:
      - non_stroking_color_hex_raw
      - background_non_stroking_color_hex_raw
      - non_stroking_color
      - background_non_stroking_color
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

    page_arr = out["page_number"].to_numpy()
    x_left = out["x_left"].to_numpy(dtype=float)
    x_right = out["x_right"].to_numpy(dtype=float)
    y_top = out["y_top"].to_numpy(dtype=float)
    y_bottom = out["y_bottom"].to_numpy(dtype=float)

    # Build one task per page; words with an out-of-range page stay as None/0.
    page_to_indices: dict = {}
    for i in range(n):
        p = int(page_arr[i])
        if p <= 0 or p > len(images_bgr) or images_bgr[p - 1] is None:
            continue
        page_to_indices.setdefault(p, []).append(i)

    tasks = []
    for p, idxs in page_to_indices.items():
        idx_arr = np.array(idxs, dtype=np.intp)
        tasks.append((
            idx_arr,
            x_left[idx_arr],
            x_right[idx_arr],
            y_top[idx_arr],
            y_bottom[idx_arr],
            images_bgr[p - 1],
            config,
        ))

    n_pages = len(tasks)
    if max_workers is None:
        max_workers = min(n_pages, os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, n_pages))

    if max_workers == 1 or n_pages <= 1:
        results = [_colorize_page_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_colorize_page_worker, tasks))

    for idxs, ihr, bhr, ihn, bhn, ic in results:
        ink_hex_raw[idxs] = ihr
        bg_hex_raw[idxs] = bhr
        ink_hex_norm[idxs] = ihn
        bg_hex_norm[idxs] = bhn
        ink_cov[idxs] = ic

    out["non_stroking_color_hex_raw"] = ink_hex_raw
    out["background_non_stroking_color_hex_raw"] = bg_hex_raw
    out["non_stroking_color"] = ink_hex_norm
    out["background_non_stroking_color"] = bg_hex_norm
    out["ink_coverage"] = ink_cov

    if config.ink_snap_threshold > 0:
        out["non_stroking_color"] = _snap_similar_colors(
            out["non_stroking_color"], config.ink_snap_threshold
        )

    return out

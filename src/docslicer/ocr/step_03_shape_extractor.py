# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Extract ruled lines and shapes from a rendered page image."""

# ocr/step_03_shape_extractor.py
#
# Rule-line extractor for rendered (OCR) pages.
#
# Approach
# --------
# Detect axis-aligned rule lines directly with morphology + connected components:
#
#   gray -> binarize (adaptive threshold, ink=white)
#     horizontal: open with a long (k,1) kernel -> bridge dashes with a short (k,1) close
#     vertical:   open with a long (1,k) kernel -> bridge dashes with a short (1,k) close
#   connectedComponentsWithStats on each mask  ->  each component is one rule
#
# Properties:
#   * A component spans the whole rule, so detection is fragment-free.
#   * Text strokes are shorter than the opening kernel and cannot survive it,
#     so word edges do not produce false lines (no text-overlap filtering needed).
#   * Cost is a threshold + a few morph ops + 2 CC calls per page, all in OpenCV C.
#
# A Python collinear-merge step rejoins rules split by threshold dropouts.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class ShapeExtractorConfig:
    # --- binarization ---
    # Adaptive thresholding is local, so it recovers faint grey rules that a single
    # global Otsu threshold drops. Otsu remains available for clean dark-on-white.
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 15      # odd; only used when use_adaptive_threshold
    adaptive_C: int = 6                # only used when use_adaptive_threshold

    # --- line isolation (in pixels, at the rendered scale) ---
    # Minimum run length for a structure to be considered a rule. A structuring
    # element of this length erases anything shorter (incl. glyph strokes).
    min_absolute_len_px: int = 60
    # Optionally also require a fraction of the page dimension. 0 disables.
    min_rule_frac: float = 0.0

    # Bridge tiny gaps (anti-aliasing) in a real rule BEFORE opening. Keep this
    # small: a value large enough to bridge inter-letter gaps will fuse text rows
    # into false horizontal lines, so 0 (off) is the safe default.
    gap_bridge_px: int = 0
    # Bridge gaps between already-detected collinear segments AFTER opening
    # (dashed rules). Safe because text is already removed by this point.
    post_open_bridge_px: int = 12

    # --- collinear merge (rejoin rules split by threshold dropouts) ---
    # Two segments merge if they are nearly collinear (centers within
    # merge_perp_tol_px on the perpendicular axis) and their along-axis gap is
    # <= merge_gap_px. Keep merge_gap_px smaller than typical inter-column spacing
    # so separate underlines stay separate.
    #
    # UNITS: all *_px values here are RENDER PIXELS, i.e. points * dpi_scale.
    # With the default dpi_scale=2, a value of 40 bridges a 20pt gap. The exported
    # CSV is in points, so a gap that reads as Npt in the CSV is 2*N px here.
    merge_collinear: bool = True
    merge_perp_tol_px: int = 3
    merge_gap_px: int = 60

    # --- block rejection ---
    # A rule is thin in its short dimension. Reject components thicker than this
    # (filled cells, shaded bands, image edges) so they don't become "lines".
    max_line_thickness_px: int = 12
    # Also reject components that aren't elongated enough (short side / long side).
    max_thickness_ratio: float = 0.30

    # --- dedupe near-identical shapes ---
    dedupe: bool = True
    dedupe_endpoint_tol_px: int = 3


# ==================================================================================================
# HELPERS
# ==================================================================================================

def _binarize_ink(gray: np.ndarray, config: ShapeExtractorConfig) -> np.ndarray:
    """Return a uint8 mask where ink (dark pixels) == 255, background == 0."""
    if config.use_adaptive_threshold:
        bs = config.adaptive_block_size
        if bs % 2 == 0:
            bs += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            blockSize=bs, C=config.adaptive_C,
        )
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _detect_axis_lines(
    bw: np.ndarray,
    *,
    horizontal: bool,
    min_len: int,
    gap_bridge: int,
    config: ShapeExtractorConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Isolate axis-aligned rules with morphology and return component bboxes
    as (left, top, width, height) in pixels.
    """
    if horizontal:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, min_len), 1))
        post_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, config.post_open_bridge_px), 1))
        pre_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, gap_bridge), 1))
    else:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, min_len)))
        post_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, config.post_open_bridge_px)))
        pre_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, gap_bridge)))

    # 1) Optional tiny pre-bridge for anti-aliased solid rules (off by default;
    #    larger values fuse text into false lines).
    src = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, pre_k) if gap_bridge > 0 else bw
    # 2) Open: erase anything without a continuous run of >= min_len. This is what
    #    removes text (gaps between letters) while keeping real rules.
    isolated = cv2.morphologyEx(src, cv2.MORPH_OPEN, open_k)
    # 3) Optional post-bridge: reconnect collinear pieces of dashed rules. Safe now
    #    that text is gone.
    if config.post_open_bridge_px > 0:
        isolated = cv2.morphologyEx(isolated, cv2.MORPH_CLOSE, post_k)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(isolated, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, n_labels):  # skip background (0)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        long_side = w if horizontal else h
        short_side = h if horizontal else w

        if long_side < min_len:
            continue
        # Block rejection: a rule is thin in its short dimension.
        if short_side > config.max_line_thickness_px:
            continue
        if long_side > 0 and (short_side / long_side) > config.max_thickness_ratio:
            continue

        boxes.append((x, y, w, h))

    return boxes


def _merge_collinear_boxes(
    boxes: List[Tuple[bool, float, float, float, float]],
    perp_tol: float,
    gap_tol: float,
) -> List[Tuple[bool, float, float, float, float]]:
    """
    Merge nearly-collinear, axis-aligned segments that were split by mask dropouts.

    Each box is (is_horizontal, x_left, y_top, x_right, y_bottom). Segments join when
    their perpendicular-axis centers are within `perp_tol` and the along-axis gap is
    <= `gap_tol`. Horizontal and vertical sets are merged independently.
    """
    out: List[Tuple[bool, float, float, float, float]] = []

    for is_h in (True, False):
        group = [b for b in boxes if b[0] == is_h]
        if not group:
            continue

        # Sort by perpendicular center, then along-axis start, so collinear
        # segments arrive left-to-right (or top-to-bottom) and adjacent.
        def perp_center(b):
            _, xl, yt, xr, yb = b
            return (yt + yb) / 2.0 if is_h else (xl + xr) / 2.0

        def along_start(b):
            _, xl, yt, xr, yb = b
            return xl if is_h else yt

        group.sort(key=lambda b: (perp_center(b), along_start(b)))

        merged: List[List[float]] = []  # [is_h, x_left, y_top, x_right, y_bottom]
        for _, xl, yt, xr, yb in group:
            c = (yt + yb) / 2.0 if is_h else (xl + xr) / 2.0
            placed = False
            for m in merged:
                mc = (m[2] + m[4]) / 2.0 if is_h else (m[1] + m[3]) / 2.0
                if abs(mc - c) > perp_tol:
                    continue
                # True 1-D separation regardless of along-axis order: positive when
                # disjoint, negative when overlapping. The sort is by perpendicular
                # center first, so segments may arrive out of along-axis order; a
                # one-sided (xl - m[3]) reads "entirely before" as a huge negative
                # "overlap" and fuses rules across arbitrary whitespace.
                gap = max(xl - m[3], m[1] - xr) if is_h else max(yt - m[4], m[2] - yb)
                if gap <= gap_tol:
                    m[1] = min(m[1], xl)
                    m[2] = min(m[2], yt)
                    m[3] = max(m[3], xr)
                    m[4] = max(m[4], yb)
                    placed = True
                    break
            if not placed:
                merged.append([is_h, xl, yt, xr, yb])

        out.extend((bool(m[0]), m[1], m[2], m[3], m[4]) for m in merged)

    return out


def _sample_shape_color(
    img_bgr: np.ndarray,
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
) -> Optional[str]:
    """Median pixel color inside the shape bbox as #RRGGBB (line ink color)."""
    H, W = img_bgr.shape[:2]
    x1 = max(int(round(x_left)), 0)
    y1 = max(int(round(y_top)), 0)
    x2 = min(int(round(x_right)), W - 1)
    y2 = min(int(round(y_bottom)), H - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return None
    pixels = patch.reshape(-1, 3).astype(np.float32)
    med = np.median(pixels, axis=0)  # BGR
    b = max(0, min(255, int(round(float(med[0])))))
    g = max(0, min(255, int(round(float(med[1])))))
    r = max(0, min(255, int(round(float(med[2])))))
    return f"#{r:02X}{g:02X}{b:02X}"


_OUTPUT_COLUMNS = [
    "page_number", "raw_shape_id", "raw_shape_type",
    "x1", "y1", "x2", "y2",
    "x_left", "x_right", "y_top", "y_bottom",
    "width", "height", "area", "linewidth", "non_stroking_color",
    "length", "angle_deg",
]


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def extract_shapes_df(
    images_bgr: List[np.ndarray],
    *,
    config: ShapeExtractorConfig = ShapeExtractorConfig(),
) -> pd.DataFrame:
    """
    Extract axis-aligned rule lines via morphology + connected components.

    Inputs:
      - images_bgr: list of BGR uint8 images, page_number = idx + 1

    Output schema:
      page_number, raw_shape_id, raw_shape_type,
      x1, y1, x2, y2,
      x_left, x_right, y_top, y_bottom,
      width, height, area, linewidth, non_stroking_color,
      length, angle_deg
    """
    if not images_bgr:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    all_rows: List[dict] = []

    for page_idx, img_bgr in enumerate(images_bgr):
        if img_bgr is None:
            continue

        page_number = int(page_idx + 1)
        H, W = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bw = _binarize_ink(gray, config)

        min_h_len = max(int(W * float(config.min_rule_frac)), int(config.min_absolute_len_px))
        min_v_len = max(int(H * float(config.min_rule_frac)), int(config.min_absolute_len_px))

        h_boxes = _detect_axis_lines(
            bw, horizontal=True, min_len=min_h_len,
            gap_bridge=int(config.gap_bridge_px), config=config,
        )
        v_boxes = _detect_axis_lines(
            bw, horizontal=False, min_len=min_v_len,
            gap_bridge=int(config.gap_bridge_px), config=config,
        )

        # Convert (left, top, w, h) -> (is_horizontal, x_left, y_top, x_right, y_bottom)
        boxes: List[Tuple[bool, float, float, float, float]] = (
            [(True, float(x), float(y), float(x + w - 1), float(y + h - 1)) for (x, y, w, h) in h_boxes]
            + [(False, float(x), float(y), float(x + w - 1), float(y + h - 1)) for (x, y, w, h) in v_boxes]
        )

        # Rejoin segments split by mask dropouts (dense tables).
        if config.merge_collinear:
            boxes = _merge_collinear_boxes(
                boxes,
                perp_tol=float(config.merge_perp_tol_px),
                gap_tol=float(config.merge_gap_px),
            )

        seen = set()
        page_rows: List[dict] = []

        for is_horiz, x_left, y_top, x_right, y_bottom in boxes:
            w = x_right - x_left + 1.0
            h = y_bottom - y_top + 1.0

            if is_horiz:
                yc = (y_top + y_bottom) / 2.0
                x1, y1, x2, y2 = int(x_left), int(round(yc)), int(x_right), int(round(yc))
                length = x_right - x_left
                linewidth = float(h)
                angle_deg = 0.0
            else:
                xc = (x_left + x_right) / 2.0
                x1, y1, x2, y2 = int(round(xc)), int(y_top), int(round(xc)), int(y_bottom)
                length = y_bottom - y_top
                linewidth = float(w)
                angle_deg = 90.0

            if config.dedupe:
                tol = max(1, int(config.dedupe_endpoint_tol_px))
                key = (
                    page_number,
                    int(round(x_left / tol)),
                    int(round(y_top / tol)),
                    int(round(x_right / tol)),
                    int(round(y_bottom / tol)),
                )
                if key in seen:
                    continue
                seen.add(key)

            page_rows.append({
                "page_number": page_number,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x_left": x_left, "x_right": x_right,
                "y_top": y_top, "y_bottom": y_bottom,
                "width": float(w), "height": float(h), "area": float(w * h),
                "linewidth": linewidth,
                "non_stroking_color": _sample_shape_color(img_bgr, x_left, y_top, x_right, y_bottom),
                "length": float(length),
                "angle_deg": angle_deg,
            })

        all_rows.extend(page_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)
    df.insert(1, "raw_shape_id", np.arange(1, len(df) + 1, dtype=np.int64))
    df.insert(2, "raw_shape_type", "line")

    return df[_OUTPUT_COLUMNS]

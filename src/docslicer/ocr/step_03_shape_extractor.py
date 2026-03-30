# ocr/shapes/step_03_line_extractor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import cv2


# ==================================================================================================
# CONFIG
# ==================================================================================================

@dataclass(frozen=True)
class ShapeExtractorConfig:
    # --- preprocessing ---
    mask_words: bool = True
    word_mask_pad_px: int = 2

    # --- edge detection ---
    canny_low: int = 50
    canny_high: int = 150

    # --- Hough transform (more sensitive defaults) ---
    hough_threshold: int = 80  # Lower to detect fainter shapes
    hough_min_shape_len_px: int = 30  # Catch shorter table shapes
    hough_max_shape_gap_px: int = 10  # Bridge gaps in dashed/faint shapes

    # --- keep only long, axis-aligned rules ---
    keep_horiz_deg_tol: float = 3.0
    keep_vert_deg_tol: float = 3.0
    min_rule_frac: float = 0.0  # Disabled: catch all shapes regardless of length
    min_absolute_len_px: int = 30  # Minimum absolute length in pixels

    # --- optional: dedupe near-identical shapes ---
    dedupe: bool = True
    dedupe_endpoint_tol_px: int = 3

    # --- multi-pass detection ---
    use_multipass: bool = True  # Try multiple detection strategies

    # --- shape merging ---
    merge_collinear: bool = True  # Merge collinear shape segments
    merge_distance_tol_px: int = 10  # Max gap between segments to merge
    merge_angle_tol_deg: float = 2.0  # Max angle difference for collinear

    # --- text filter (reduce false positives from aligned text) ---
    filter_text_shapes: bool = True  # Remove shapes that pass through too much text
    text_intersection_max_ratio: float = 0.15  # Max ratio of shape covered by text (very strict)
    text_intersection_min_words: int = 2  # Min words intersecting to reject (aligned text pattern)


# ==================================================================================================
# HELPERS
# ==================================================================================================

def _clip_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), W - 1)
    y2 = min(int(y2), H - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _mask_words(img_bgr: np.ndarray, words_df: pd.DataFrame, page_number: int, pad: int) -> np.ndarray:
    """
    Remove word regions to prevent text strokes from becoming "shapes".
    """
    out = img_bgr.copy()
    if words_df is None or words_df.empty:
        return out

    w = words_df[words_df["page_number"].astype(int) == int(page_number)]
    if w.empty:
        return out

    H, W = out.shape[:2]

    # Iterate rows: words_df is typically large; keep this simple and safe.
    for _, r in w.iterrows():
        x1 = int(round(float(r["x_left"]))) - pad
        y1 = int(round(float(r["y_top"]))) - pad
        x2 = int(round(float(r["x_right"]))) + pad
        y2 = int(round(float(r["y_bottom"]))) + pad
        clipped = _clip_box(x1, y1, x2, y2, W, H)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    return out


def _normalize_angle_deg(angle: float) -> float:
    """
    Normalize to [-90, 90] for easier horiz/vert checks.
    """
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return float(angle)


def _shape_key_axis_aligned(x1: int, y1: int, x2: int, y2: int, tol: int) -> Tuple[int, int, int, int]:
    """
    Quantize endpoints to allow cheap dedupe.
    Sort endpoints so direction doesn't matter.
    """
    def q(v: int) -> int:
        return int(round(v / max(1, tol))) * max(1, tol)

    ax1, ay1, ax2, ay2 = q(x1), q(y1), q(x2), q(y2)
    if (ax2, ay2) < (ax1, ay1):
        ax1, ay1, ax2, ay2 = ax2, ay2, ax1, ay1
    return ax1, ay1, ax2, ay2


def _shapes_are_collinear(
    x1_a: float, y1_a: float, x2_a: float, y2_a: float,
    x1_b: float, y1_b: float, x2_b: float, y2_b: float,
    angle_tol_deg: float,
    distance_tol_px: float,
) -> bool:
    """
    Check if two shape segments are collinear and close enough to merge.
    """
    import math

    # Calculate angles
    angle_a = math.atan2(y2_a - y1_a, x2_a - x1_a)
    angle_b = math.atan2(y2_b - y1_b, x2_b - x1_b)

    # Normalize angles to [0, pi]
    angle_a = abs(angle_a)
    angle_b = abs(angle_b)

    # Check angle similarity
    angle_diff_deg = abs(math.degrees(angle_a - angle_b))
    if angle_diff_deg > 180:
        angle_diff_deg = 360 - angle_diff_deg

    if angle_diff_deg > angle_tol_deg:
        return False

    # Check if shapes are close to each other
    # For horizontal shapes, check y-coordinate proximity
    # For vertical shapes, check x-coordinate proximity

    is_horizontal = abs(math.degrees(angle_a)) < 45 or abs(math.degrees(angle_a)) > 135

    if is_horizontal:
        # Check if y-coordinates are similar
        y_a = (y1_a + y2_a) / 2
        y_b = (y1_b + y2_b) / 2
        if abs(y_a - y_b) > distance_tol_px:
            return False

        # Check if x-ranges overlap or are close
        x_min_a, x_max_a = min(x1_a, x2_a), max(x1_a, x2_a)
        x_min_b, x_max_b = min(x1_b, x2_b), max(x1_b, x2_b)

        # Check overlap or gap
        gap = max(x_min_a, x_min_b) - min(x_max_a, x_max_b)
        return gap <= distance_tol_px
    else:
        # Vertical shape - check x-coordinates
        x_a = (x1_a + x2_a) / 2
        x_b = (x1_b + x2_b) / 2
        if abs(x_a - x_b) > distance_tol_px:
            return False

        # Check if y-ranges overlap or are close
        y_min_a, y_max_a = min(y1_a, y2_a), max(y1_a, y2_a)
        y_min_b, y_max_b = min(y1_b, y2_b), max(y1_b, y2_b)

        # Check overlap or gap
        gap = max(y_min_a, y_min_b) - min(y_max_a, y_max_b)
        return gap <= distance_tol_px


def _merge_shape_segments(shapes: List[dict], config: ShapeExtractorConfig) -> List[dict]:
    """
    Merge collinear shape segments into longer shapes.
    """
    if not shapes or not config.merge_collinear:
        return shapes

    import math

    merged = []
    used = set()

    for i, shape_a in enumerate(shapes):
        if i in used:
            continue

        # Start with this shape
        x1, y1 = shape_a["x1"], shape_a["y1"]
        x2, y2 = shape_a["x2"], shape_a["y2"]

        # Try to merge with other shapes
        changed = True
        while changed:
            changed = False
            for j, shape_b in enumerate(shapes):
                if j == i or j in used:
                    continue

                # Check if collinear
                if _shapes_are_collinear(
                    x1, y1, x2, y2,
                    shape_b["x1"], shape_b["y1"], shape_b["x2"], shape_b["y2"],
                    angle_tol_deg=config.merge_angle_tol_deg,
                    distance_tol_px=config.merge_distance_tol_px,
                ):
                    # Merge: extend to cover both shapes
                    all_x = [x1, x2, shape_b["x1"], shape_b["x2"]]
                    all_y = [y1, y2, shape_b["y1"], shape_b["y2"]]

                    # For horizontal shapes, use min/max x
                    # For vertical shapes, use min/max y
                    angle = math.atan2(y2 - y1, x2 - x1)
                    is_horizontal = abs(math.degrees(angle)) < 45 or abs(math.degrees(angle)) > 135

                    if is_horizontal:
                        idx_min = all_x.index(min(all_x))
                        idx_max = all_x.index(max(all_x))
                        x1, y1 = all_x[idx_min], all_y[idx_min]
                        x2, y2 = all_x[idx_max], all_y[idx_max]
                    else:
                        idx_min = all_y.index(min(all_y))
                        idx_max = all_y.index(max(all_y))
                        x1, y1 = all_x[idx_min], all_y[idx_min]
                        x2, y2 = all_x[idx_max], all_y[idx_max]

                    used.add(j)
                    changed = True

        # Add merged shape
        dx = x2 - x1
        dy = y2 - y1
        length = float(math.hypot(dx, dy))
        angle = float(math.degrees(math.atan2(dy, dx)))
        angle_n = _normalize_angle_deg(angle)

        merged.append({
            "page_number": shape_a["page_number"],
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "x_left": float(min(x1, x2)),
            "x_right": float(max(x1, x2)),
            "y_top": float(min(y1, y2)),
            "y_bottom": float(max(y1, y2)),
            "length": length,
            "angle_deg": angle_n,
        })

        used.add(i)

    return merged


def _sample_shape_color(
    img_bgr: np.ndarray,
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
) -> Optional[str]:
    """
    Sample the median pixel color inside the shape bbox, returned as a hex string.
    For rule lines this captures the ink/line color rather than the background.
    """
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
    med = np.median(pixels, axis=0)  # BGR order
    b = max(0, min(255, int(round(float(med[0])))))
    g = max(0, min(255, int(round(float(med[1])))))
    r = max(0, min(255, int(round(float(med[2])))))
    return f"#{r:02X}{g:02X}{b:02X}"


def _filter_text_overlapping_shapes(
    shapes: List[dict],
    words_df: Optional[pd.DataFrame],
    page_number: int,
    config: ShapeExtractorConfig,
) -> List[dict]:
    """
    Filter out shapes that overlap heavily with text (likely false positives from aligned text).

    This is particularly important for vertical shapes that might be detected from
    aligned text columns in tables.
    """
    if not config.filter_text_shapes or words_df is None or words_df.empty:
        return shapes

    # Get words for this page
    page_words = words_df[words_df["page_number"].astype(int) == int(page_number)]
    if page_words.empty:
        return shapes

    filtered = []

    for shape in shapes:
        x1, y1, x2, y2 = shape["x1"], shape["y1"], shape["x2"], shape["y2"]
        shape_len = shape["length"]

        if shape_len == 0:
            continue

        # Calculate how much of the shape intersects with text bboxes
        intersection_length = 0.0
        intersecting_words = 0

        # Tighter tolerance for intersection check
        pad = 2

        for _, word in page_words.iterrows():
            wx1 = float(word["x_left"])
            wy1 = float(word["y_top"])
            wx2 = float(word["x_right"])
            wy2 = float(word["y_bottom"])

            word_intersects = False

            # Check if shape intersects with word bbox
            # For horizontal shapes
            if abs(shape["angle_deg"]) < 45:
                # Check y overlap
                if not (y1 - pad <= wy2 and y2 + pad >= wy1):
                    continue

                # Check x overlap
                x_overlap_start = max(min(x1, x2), wx1)
                x_overlap_end = min(max(x1, x2), wx2)

                if x_overlap_end > x_overlap_start:
                    intersection_length += (x_overlap_end - x_overlap_start)
                    word_intersects = True

            # For vertical shapes (more strict - common false positive from aligned text)
            else:
                # Check x overlap (must be close to the shape)
                shape_x = (x1 + x2) / 2
                if not (shape_x - pad <= wx2 and shape_x + pad >= wx1):
                    continue

                # Check y overlap
                y_overlap_start = max(min(y1, y2), wy1)
                y_overlap_end = min(max(y1, y2), wy2)

                if y_overlap_end > y_overlap_start:
                    intersection_length += (y_overlap_end - y_overlap_start)
                    word_intersects = True

            if word_intersects:
                intersecting_words += 1

        # Calculate ratio of shape covered by text
        text_ratio = intersection_length / shape_len if shape_len > 0 else 0

        # Rejection criteria:
        # 1. High text overlap ratio (>15%)
        # 2. Multiple words intersecting (likely aligned text column)
        reject = False

        if text_ratio > config.text_intersection_max_ratio:
            reject = True

        # Special case for vertical shapes: if multiple words align, it's likely false positive
        if abs(shape["angle_deg"]) > 45 and intersecting_words >= config.text_intersection_min_words:
            # For vertical shapes, be even more strict
            # If it passes through many words, it's almost certainly aligned text
            if text_ratio > 0.08 or intersecting_words >= 3:  # Even stricter for vertical
                reject = True

        if not reject:
            filtered.append(shape)

    return filtered


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def extract_shapes_df(
    images_bgr: List[np.ndarray],
    *,
    words_df: Optional[pd.DataFrame] = None,
    config: ShapeExtractorConfig = ShapeExtractorConfig(),
) -> pd.DataFrame:
    """
    Extract long axis-aligned rule shapes (horizontal/vertical lines) using HoughLinesP.

    Inputs:
      - images_bgr: list of BGR uint8 images, page_number = idx + 1
      - words_df: optional; used for masking text regions if config.mask_words=True

    Output schema:
      page_number, raw_shape_id, raw_shape_type,
      x1, y1, x2, y2,
      x_left, x_right, y_top, y_bottom,
      width, height, area, linewidth, non_stroking_color,
      length, angle_deg
    """
    if not images_bgr:
        return pd.DataFrame(
            columns=[
                "page_number",
                "raw_shape_id",
                "raw_shape_type",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_left",
                "x_right",
                "y_top",
                "y_bottom",
                "width",
                "height",
                "area",
                "linewidth",
                "non_stroking_color",
                "length",
                "angle_deg",
            ]
        )

    all_rows: List[dict] = []

    for page_idx, img_bgr in enumerate(images_bgr):
        if img_bgr is None:
            continue

        page_number = int(page_idx + 1)
        H, W = img_bgr.shape[:2]

        work = img_bgr
        if config.mask_words and words_df is not None and not words_df.empty:
            work = _mask_words(work, words_df, page_number=page_number, pad=int(config.word_mask_pad_px))

        page_rows: List[dict] = []
        seen = set()  # dedupe keys per page

        # Convert to grayscale
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        # Multi-pass detection for better coverage
        all_hough_shapes = []

        if config.use_multipass:
            # Pass 1: Standard OTSU threshold
            _, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges1 = cv2.Canny(bw1, int(config.canny_low), int(config.canny_high))
            shapes1 = cv2.HoughLinesP(
                edges1, rho=1, theta=np.pi / 180.0,
                threshold=int(config.hough_threshold),
                minLineLength=int(config.hough_min_shape_len_px),
                maxLineGap=int(config.hough_max_shape_gap_px),
            )
            if shapes1 is not None:
                all_hough_shapes.extend(shapes1.reshape(-1, 4))

            # Pass 2: Inverted for light-on-dark shapes
            gray_inv = 255 - gray
            _, bw2 = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges2 = cv2.Canny(bw2, int(config.canny_low), int(config.canny_high))
            shapes2 = cv2.HoughLinesP(
                edges2, rho=1, theta=np.pi / 180.0,
                threshold=int(config.hough_threshold),
                minLineLength=int(config.hough_min_shape_len_px),
                maxLineGap=int(config.hough_max_shape_gap_px),
            )
            if shapes2 is not None:
                all_hough_shapes.extend(shapes2.reshape(-1, 4))

            # Pass 3: Morphological closing to connect broken shapes
            _, bw3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw3_inv = 255 - bw3
            # Horizontal kernel for horizontal shapes
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            h_closed = cv2.morphologyEx(bw3_inv, cv2.MORPH_CLOSE, h_kernel)
            # Vertical kernel for vertical shapes
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            v_closed = cv2.morphologyEx(bw3_inv, cv2.MORPH_CLOSE, v_kernel)
            # Combine
            morph_combined = cv2.bitwise_or(h_closed, v_closed)
            edges3 = cv2.Canny(morph_combined, int(config.canny_low), int(config.canny_high))
            shapes3 = cv2.HoughLinesP(
                edges3, rho=1, theta=np.pi / 180.0,
                threshold=int(config.hough_threshold),
                minLineLength=int(config.hough_min_shape_len_px),
                maxLineGap=int(config.hough_max_shape_gap_px),
            )
            if shapes3 is not None:
                all_hough_shapes.extend(shapes3.reshape(-1, 4))

            # Pass 4: Direct edge detection on grayscale for very faint shapes
            edges4 = cv2.Canny(gray, int(config.canny_low) // 2, int(config.canny_high) // 2)
            shapes4 = cv2.HoughLinesP(
                edges4, rho=1, theta=np.pi / 180.0,
                threshold=max(50, int(config.hough_threshold) // 2),
                minLineLength=int(config.hough_min_shape_len_px),
                maxLineGap=int(config.hough_max_shape_gap_px) * 2,
            )
            if shapes4 is not None:
                all_hough_shapes.extend(shapes4.reshape(-1, 4))
        else:
            # Single pass (original behavior)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(bw, int(config.canny_low), int(config.canny_high))
            shapes = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180.0,
                threshold=int(config.hough_threshold),
                minLineLength=int(config.hough_min_shape_len_px),
                maxLineGap=int(config.hough_max_shape_gap_px),
            )
            if shapes is not None:
                all_hough_shapes.extend(shapes.reshape(-1, 4))

        if not all_hough_shapes:
            continue

        # Calculate minimum lengths
        min_horiz_len = max(int(W * float(config.min_rule_frac)), int(config.min_absolute_len_px))
        min_vert_len = max(int(H * float(config.min_rule_frac)), int(config.min_absolute_len_px))

        for (x1, y1, x2, y2) in all_hough_shapes:
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)

            dx = x2 - x1
            dy = y2 - y1
            length = float(math.hypot(dx, dy))
            angle = float(math.degrees(math.atan2(dy, dx)))
            angle_n = _normalize_angle_deg(angle)

            is_horiz = abs(angle_n) <= float(config.keep_horiz_deg_tol) and abs(x2 - x1) >= min_horiz_len
            is_vert = abs(abs(angle_n) - 90.0) <= float(config.keep_vert_deg_tol) and abs(y2 - y1) >= min_vert_len

            if not (is_horiz or is_vert):
                continue

            if config.dedupe:
                key = (page_number,) + _shape_key_axis_aligned(
                    x1, y1, x2, y2, tol=int(config.dedupe_endpoint_tol_px)
                )
                if key in seen:
                    continue
                seen.add(key)

            page_rows.append(
                {
                    "page_number": page_number,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_left": float(min(x1, x2)),
                    "x_right": float(max(x1, x2)),
                    "y_top": float(min(y1, y2)),
                    "y_bottom": float(max(y1, y2)),
                    "length": float(length),
                    "angle_deg": float(angle_n),
                }
            )

        # Post-processing for this page
        # 1. Merge collinear shape segments first (combines broken shapes)
        page_rows = _merge_shape_segments(page_rows, config)

        # 2. Filter out shapes that overlap heavily with text
        # (do this after merging so we have complete shapes to evaluate)
        page_rows = _filter_text_overlapping_shapes(page_rows, words_df, page_number, config)

        # 3. Augment with derived fields required by the shape merger
        for shape in page_rows:
            w = shape["x_right"] - shape["x_left"]
            h = shape["y_bottom"] - shape["y_top"]
            shape["width"] = w
            shape["height"] = h
            shape["area"] = w * h
            # linewidth = physical thickness of the rule
            shape["linewidth"] = h if abs(shape.get("angle_deg", 0.0)) < 45 else w
            shape["non_stroking_color"] = _sample_shape_color(
                img_bgr, shape["x_left"], shape["y_top"], shape["x_right"], shape["y_bottom"]
            )

        # Add to all rows
        all_rows.extend(page_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "page_number",
                "raw_shape_id",
                "raw_shape_type",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_left",
                "x_right",
                "y_top",
                "y_bottom",
                "width",
                "height",
                "area",
                "linewidth",
                "non_stroking_color",
                "length",
                "angle_deg",
            ]
        )

    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)
    df.insert(1, "raw_shape_id", np.arange(1, len(df) + 1, dtype=np.int64))
    df.insert(2, "raw_shape_type", "line")

    return df

# ocr/font_size_estimator.py
from __future__ import annotations

import pandas as pd


# ============================================================
# Typographic character sets
# ============================================================

# Any uppercase letter gives a capital-height reference
_CAPITAL_CHARS: frozenset[str] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Lowercase letters with ascenders (stem above x-height: b d f h i j k l t)
_ASCENDER_CHARS: frozenset[str] = frozenset("bdfhijklt")

# Lowercase letters with descenders (tail below baseline: g j p q y)
_DESCENDER_CHARS: frozenset[str] = frozenset("gjpqy")

# Standard font-size buckets: (upper_bound_exclusive_pt, canonical_pt)
_FONT_SIZE_BUCKETS: list[tuple[float, float]] = [
    (5.5,  5.0),
    (6.5,  6.0),
    (7.5,  7.0),
    (8.5,  8.0),
    (9.5,  9.0),
    (10.5, 10.0),
    (11.5, 11.0),
    (13.0, 12.0),
    (15.0, 14.0),
    (17.0, 16.0),
    (19.0, 18.0),
    (21.0, 20.0),
    (26.0, 24.0),
    (32.0, 28.0),
    (40.0, 36.0),
]


def _snap_font_size(height_pt: float) -> float:
    """Snap a height in points to the nearest canonical font size."""
    for upper, label in _FONT_SIZE_BUCKETS:
        if height_pt < upper:
            return label
    return round(height_pt)


def _mode_or_median(s: pd.Series) -> float:
    counts = s.value_counts()
    return float(counts.index[0]) if not counts.empty else float(s.median())


# ============================================================
# Public API
# ============================================================

def estimate_ocr_font_sizes(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate stable font sizes for OCR lines using typographic analysis.

    Must be called after build_tables() so that layout_id is available.

    Algorithm
    ---------
    1. Recompute line height from geometry (y_bottom - y_top).
    2. Detect typographic coverage per line from its text:
         has_capital   — any uppercase letter (A-Z)
         has_ascender  — any b d f h i j k l t
         has_descender — any g j p q y
    3. Compute adjusted_height to compensate for missing typographic references:
         missing top reference (no capital AND no ascender) → +20 %
         missing descender                                  → +20 %
         (both missing                                      → +40 %)
       Lines with all three present get adjusted_height = height as-is.
    4. Per layout_id, take the mode of adjusted_height (rounded to 1 pt)
       as the canonical line height for that layout block.
    5. Snap canonical height to the nearest standard font size.

    Added / updated columns
    -----------------------
    has_capital, has_ascender, has_descender  bool
    font_size                                 float  (replaces noisy per-line value)
    """
    required = {"text", "y_top", "y_bottom", "layout_id"}
    missing_cols = required - set(df_lines.columns)
    if missing_cols:
        raise ValueError(f"estimate_ocr_font_sizes: missing columns: {sorted(missing_cols)}")

    df = df_lines.copy()
    texts = df["text"].astype(str)

    # --- typography flags ---
    df["has_capital"]   = texts.apply(lambda t: any(c in _CAPITAL_CHARS   for c in t))
    df["has_ascender"]  = texts.apply(lambda t: any(c in _ASCENDER_CHARS  for c in t))
    df["has_descender"] = texts.apply(lambda t: any(c in _DESCENDER_CHARS for c in t))

    # --- adjusted height ---
    line_height    = (df["y_bottom"] - df["y_top"]).clip(lower=0.0)
    missing_top    = ~(df["has_capital"] | df["has_ascender"])
    missing_bottom = ~df["has_descender"]
    adjustment     = missing_top.astype(float) * 0.20 + missing_bottom.astype(float) * 0.20
    adj_height     = line_height * (1.0 + adjustment)

    # --- mode per layout_id (rounded to 1 pt for stability) ---
    df["_adj_h"] = adj_height.round(1)
    layout_canonical = (
        df.groupby("layout_id")["_adj_h"]
        .agg(_mode_or_median)
        .rename("_canonical_h")
    )
    df = df.join(layout_canonical, on="layout_id")

    # --- snap to standard font size ---
    df["font_size"] = df["_canonical_h"].apply(_snap_font_size)

    return df.drop(columns=["_adj_h", "_canonical_h"])

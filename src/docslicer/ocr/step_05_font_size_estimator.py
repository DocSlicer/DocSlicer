# ocr/step_05_font_size_estimator.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ============================================================
# Typographic character sets
# ============================================================

_RE_CAPITAL   = "[A-Z]"
_RE_ASCENDER  = "[bdfhijklt]"
_RE_DESCENDER = "[gjpqyQ]"

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
_BUCKET_BOUNDS = np.array([u for u, _ in _FONT_SIZE_BUCKETS], dtype=float)
_BUCKET_LABELS = np.array([l for _, l in _FONT_SIZE_BUCKETS], dtype=float)


def _snap_vectorized(heights: np.ndarray) -> np.ndarray:
    """Snap an array of heights (pts) to the nearest canonical font size."""
    idx      = np.searchsorted(_BUCKET_BOUNDS, heights, side="right")
    in_range = idx < len(_BUCKET_LABELS)
    return np.where(
        in_range,
        _BUCKET_LABELS[np.minimum(idx, len(_BUCKET_LABELS) - 1)],
        np.round(heights),
    )


def _band_mode(group_col: pd.Series, adj_h_col: pd.Series) -> pd.Series:
    """
    Vectorized mode of adj_h_col per group_col value.
    Returns a Series indexed by group_col values.
    """
    counts = (
        pd.DataFrame({"_g": group_col, "_h": adj_h_col.round(1)})
        .groupby(["_g", "_h"], sort=False)
        .size()
        .reset_index(name="_cnt")
        .sort_values("_cnt", ascending=False)
    )
    return counts.groupby("_g", sort=False)["_h"].first().rename("_canonical_h")


# ============================================================
# Public API
# ============================================================

def estimate_ocr_font_sizes(
    df_words: pd.DataFrame,
    method: Literal["line", "word"] = "line",
) -> pd.DataFrame:
    """
    Estimate stable font sizes grouped by layout_id, from word-level input.

    Parameters
    ----------
    df_words : pd.DataFrame
        One row per word.  Required columns: line_id, layout_id,
        y_top, y_bottom, text.
    method : "line" | "word"
        "line"  — aggregate typographic flags per line_id (any word in the line
                  has the character), take median word height per line, then mode
                  of adjusted line heights per band.  More accurate because lines
                  almost always have full typographic coverage → minimal adjustment.
        "word"  — adjust each word independently, then mode of all word heights
                  per band.  Simpler, more data points; over-adjusted pure-x-height
                  words (e.g. "of", "to") are typically outvoted by the mode.

    Returns
    -------
    df_words with added columns:
        has_capital, has_ascender, has_descender  bool   (word-level)
        font_size                                 float  (band-level canonical)
    """
    required = {"line_id", "layout_id", "y_top", "y_bottom", "text"}
    missing  = required - set(df_words.columns)
    if missing:
        raise ValueError(f"estimate_ocr_font_sizes: missing columns: {sorted(missing)}")

    df = df_words.copy()
    texts = df["text"].astype(str)

    # ── Step 1: word-level typographic flags + height (always vectorized) ─────
    df["has_capital"]   = texts.str.contains(_RE_CAPITAL,   regex=True, na=False)
    df["has_ascender"]  = texts.str.contains(_RE_ASCENDER,  regex=True, na=False)
    df["has_descender"] = texts.str.contains(_RE_DESCENDER, regex=True, na=False)
    df["_word_height"]  = (df["y_bottom"] - df["y_top"]).clip(lower=0.0)

    if method == "line":
        # ── Step 2a: aggregate flags + height to line level ──────────────────
        line_agg = df.groupby("line_id", sort=False).agg(
            _has_cap  =("has_capital",        "any"),
            _has_asc  =("has_ascender",       "any"),
            _has_desc =("has_descender",      "any"),
            _height   =("_word_height",       "median"),
            _band_id  =("layout_id",          "first"),
        )

        # ── Step 3a: adjusted height per line ────────────────────────────────
        missing_top    = ~(line_agg["_has_cap"] | line_agg["_has_asc"])
        missing_bottom = ~line_agg["_has_desc"]
        adjustment     = missing_top.astype(float) * 0.20 + missing_bottom.astype(float) * 0.20
        line_agg["_adj_h"] = line_agg["_height"] * (1.0 + adjustment)

        band_canonical = _band_mode(line_agg["_band_id"], line_agg["_adj_h"])

    else:  # method == "word"
        # ── Step 2b: adjusted height per word ────────────────────────────────
        missing_top    = ~(df["has_capital"] | df["has_ascender"])
        missing_bottom = ~df["has_descender"]
        adjustment     = missing_top.astype(float) * 0.20 + missing_bottom.astype(float) * 0.20
        df["_adj_h"]   = df["_word_height"] * (1.0 + adjustment)

        band_canonical = _band_mode(df["layout_id"], df["_adj_h"])

    # ── Step 4: scale glyph bbox → nominal em-square, then snap ─────────────
    # OCR measures pixel bboxes; even fully-covered lines span only ~85% of the
    # typographic em-square.  The 1.18 factor corrects this systematic undercount.
    _GLYPH_TO_EM: float = 1.18
    band_font_size = pd.Series(
        _snap_vectorized(band_canonical.to_numpy() * _GLYPH_TO_EM),
        index=band_canonical.index,
        name="font_size",
    )

    # ── Join back onto input df ───────────────────────────────────────────────
    df["font_size"] = df["layout_id"].map(band_font_size)

    return df.drop(columns=["_word_height", "_adj_h"], errors="ignore")

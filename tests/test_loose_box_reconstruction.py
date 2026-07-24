"""Lock the pdfium-version-independent loose-box vertical reconstruction.

pdfium changed how it synthesizes the *loose* char box's vertical extent
between build 147 (pypdfium2 5.6, ~1.3x em from FontBBox) and build 152
(5.12, ~0.9x em from Ascent/Descent). docslicer's geometry (line merging,
heading detection) is tuned to the FontBBox-based box, so step_01 rebuilds it
from baseline + FontBBox in `_loose_charbox_y`. These tests fail on the raw
5.12 box (~0.9x em) and pass only when the reconstruction is in place, so they
guard against a silent regression on any future pdfium bump.
"""
from pathlib import Path

import pytest

from docslicer.pdf import step_01_word_extractor as W

SAMPLE = (
    Path(__file__).resolve().parent.parent
    / "examples" / "sample_docs" / "academic_paper.pdf"
)


def test_parse_font_vbbox_type1_cleartext():
    # Type1 /FontBBox is in 1000-em glyph units; we take (ymin, ymax).
    data = b"%!PS-AdobeFont\n/FontBBox {-168 -341 1000 960} readonly def\n"
    assert W._parse_font_vbbox(data) == (-341.0, 960.0)


def test_parse_font_vbbox_rejects_garbage():
    assert W._parse_font_vbbox(b"") is None
    assert W._parse_font_vbbox(b"not a font program") is None


def test_std14_fallback_matches_family_and_style():
    assert W._std14_vbbox("Helvetica-Bold") == (-228.0, 962.0)
    assert W._std14_vbbox("Times-Roman")[1] == 898.0
    assert W._std14_vbbox("Courier") == (-250.0, 805.0)
    assert W._std14_vbbox("SomeUnknownFont") is None


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_loose_box_is_fontbbox_scale_not_ascent_descent():
    """First heading word of the sample is 14.35pt Nimbus Roman (FontBBox
    ymax-ymin = 1301/1000 em). Its reconstructed height must be ~1.30x the font
    size (the pre-5.7 convention), NOT the ~0.90x that raw pdfium 5.12 returns.
    """
    df = W.extract_words(SAMPLE)
    assert not df.empty
    row = df.sort_values(["page_number", "y_top", "x_left"]).iloc[0]
    ratio = row["height"] / row["font_size"]
    # FontBBox-based box is ~1.30x em; Ascent/Descent box is ~0.90x. A midpoint
    # threshold cleanly separates the two regimes.
    assert ratio > 1.15, (
        f"loose box height/em = {ratio:.3f}; expected ~1.30 (FontBBox), got the "
        "squeezed ~0.90 Ascent/Descent box — reconstruction is not applied."
    )

"""
constants.py

Allowable values for every categorical field in the parsing pipeline.
One value per line so each option can carry an inline explanation.

Import pattern:
    from docslicer.constants import BlockType, SectionType
    from docslicer.constants import values  # runtime membership test
"""

from __future__ import annotations
from typing import Literal, get_args


# ==========================================
# HELPERS
# ==========================================

def values(literal_type: type) -> frozenset:
    """Return the allowable values of a Literal type as a frozenset."""
    return frozenset(get_args(literal_type))


# ==========================================
# DOCUMENT STRUCTURE
# ==========================================

SectionType = Literal[
    "coverpage",     # leading page(s) before any numbered content (title page, cover sheet)
    "front_matter",  # roman-numbered pages before the body (preface, introduction, TOC if roman-paged)
    "toc",           # table-of-contents pages (detected by block_type or page content)
    "exhibits",      # exhibits / appendix listing section
    "body",          # main numbered content (arabic page labels)
    "annex",         # lettered annex sections (e.g. A-1, B-2)
    "financials",    # financial statements annex (letter prefix "F")
    "schedules",     # schedules annex (letter prefix "S")
    "back_matter",   # roman-numbered pages after the body (signatures, certifications)
    "last_page",     # standalone closing page (contact info, low density — not part of back_matter)
]


# ==========================================
# BLOCK / LAYOUT CLASSIFICATION
# ==========================================

BlockType = Literal[
    "heading",                    # section or sub-section heading
    "paragraph",                  # regular body text
    "table",                      # tabular content
    "image",                      # standalone image
    "hr",                         # horizontal rule / visual separator
    "page_label",                 # running page number / header label
    "toc",                        # table-of-contents entry line
    "toc_heading",                # heading that introduces a TOC section
    "exhibits",                   # exhibit / appendix entry line
    "exhibit_heading",            # heading that introduces an exhibits section
    "hybrid_heading_paragraph",   # visually looks like heading but flows as paragraph
    "suppressed_repeated_heading", # heading de-duplicated from a repeated header band
    "speaker_notes",              # pptx speaker notes
    "shape",                      # pptx non-text shape with content
    "chart",                      # pptx chart object
]

LayoutType = Literal[
    "text_singlecol",   # single-column prose band
    "text_multicol",    # multi-column prose band (gutter detected)
    "table",            # tabular band
]


# ==========================================
# SHAPE
# ==========================================

ShapeType = Literal[
    "line",     # straight line segment
    "rect",     # rectangle / filled box
    "curve",    # bezier curve or arc
    "unknown",  # could not be classified
]

ShapeOrientation = Literal[
    "horizontal",   # width > height
    "vertical",     # height > width
    "unknown",      # square or indeterminate
]
Orientation = ShapeOrientation  # backwards-compatible alias

ShapeRole = Literal[
    "page_background",  # filled rect covering (almost) the entire page (ppt slides)
    "table_grid",       # grid line inside a table
    "underline",        # underline decoration under text
    "separator",        # horizontal rule separating content regions
    "background_band",  # filled rectangle used as a background highlight
    "other",            # none of the above
]


# ==========================================
# TABLE
# ==========================================

CellRole = Literal[
    "header",       # column header row
    "data",         # regular data cell
    "row_label",    # leftmost column acting as a row label
]


# ==========================================
# STYLE
# ==========================================

TextAlign = Literal[
    "left",
    "center",
    "right",
    "justify",
]

TextOrientation = Literal[
    "LTR",      # left-to-right (standard)
    "RTL",      # right-to-left
    "BTT",
    "TTB",
    "vertical", # rotated / vertical text
]


# ==========================================
# LINKS
# ==========================================

LinkType = Literal[
    "external",  # http(s) URL pointing outside the document
    "internal",  # named destination within the same document
    "anchor",    # bookmark anchor within the same document
]


# ==========================================
# DOCX / PPTX RUN TYPES
# ==========================================

RunType = Literal[
    # --- content ---
    "text",               # regular text run
    "tab",                # tab character (docx)
    "line_break",         # explicit line break within a paragraph (pptx <a:br>)
    # --- media references ---
    "image_ref",          # inline image placeholder
    "shape_ref",          # reference to a pptx shape with text
    "chart_ref",          # reference to a pptx chart
    "graphic_ref",        # reference to a pptx SmartArt / generic graphic
    # --- fields (docx) ---
    "field_marker",       # fldChar begin/end marker
    "field_code",         # INSTRTEXT field instruction (e.g. PAGE, HYPERLINK)
    # --- breaks (docx) ---
    "page_break",         # explicit <w:lastRenderedPageBreak> or break type="page"
    "rendered_page_break", # pagination-inferred page break inserted by the renderer
    "section_break",      # synthetic row inserted at a section boundary
    # --- cross-references (docx) ---
    "footnote_reference", # <w:footnoteReference> — inline marker in body text
    "endnote_reference",  # <w:endnoteReference> — inline marker in body text
]

HeaderFooterType = Literal[
    "body",    # main document body (not a header or footer)
    "header",  # page header region
    "footer",  # page footer region
    "notes",   # pptx notes panel
]

SectionBreakType = Literal[
    "nextPage",    # break starts new page (default)
    "continuous",  # break continues on the same page
    "evenPage",    # break starts on the next even-numbered page
    "oddPage",     # break starts on the next odd-numbered page
]


# ==========================================
# PPTX PLACEHOLDER TYPES
# ==========================================

PlaceholderType = Literal[
    "title",     # slide title
    "ctrTitle",  # centered title (title slide)
    "body",      # main content area
    "subTitle",  # subtitle (title slide)
    "pic",       # picture placeholder
    "chart",     # chart placeholder
    "tbl",       # table placeholder
    "dgm",       # SmartArt / diagram placeholder
    "media",     # audio / video placeholder
    # sldImg / sldNum / dt / ftr / hdr are skipped during extraction
]

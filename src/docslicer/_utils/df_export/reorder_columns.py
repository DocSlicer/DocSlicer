# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
reorder_columns.py

Centralised column ordering for all DataFrames in the parsing pipeline
(PDF, DOCX, PPTX, HTML, OCR).

MASTER_COLUMN_ORDER is the single source of truth for every column name
produced anywhere in the pipeline.  Columns not listed here end up appended
at the end of any reordered DataFrame — a useful signal that the list is
out of date.

The list is split into two parts:

    PART A — STABLE OUTPUT COLUMNS
        Columns that survive to (or near) the final output, ordered so that a
        human inspecting an exported frame reads identity → hierarchy →
        content → geometry → style → features → provenance left-to-right.

    PART B — INTERMEDIATE / DIAGNOSTIC COLUMNS
        Scratch, debug and per-atom columns consumed at some stage and
        dropped before the final output. Grouped by the pipeline stage that
        produces them and pushed to the back so they never clutter the front
        of an inspected frame. Ordering here mirrors the sections in
        ``df_aggregation/registry_aggregator.py`` (the aggregation source of
        truth); the two files should be kept in sync.

Usage:
    from docslicer._utils.reorder_columns import reorder_columns

    df_ordered = reorder_columns(df, dataframe_type="cells")
"""

from __future__ import annotations
import pandas as pd
from typing import List, Literal

DataFrameType = Literal[
    "words",
    "runs",
    "boxes",
    "shapes",
    "links",
    "cells",
    "temp_lines",
    "table_cells",
    "paragraphs",
    "lines",
    "blocks",
    "chunks",
    "images",
    "gutters",
]


# =====================
# Master Column Order
# =====================

MASTER_COLUMN_ORDER: List[str] = [

    # =========================================================================
    # PART A — STABLE OUTPUT COLUMNS
    # =========================================================================

    # ===== 1. Document / page identity =====
    "page_number",
    "page_format",
    "page_label",
    "page_label_type",
    "page_label_value",
    "section",
    "section_id",           # docx
    "slide_index",          # pptx

    # ===== 2. Run / atom provenance (docx, pptx) =====
    # (every row comes from a single XML run; these say *where* it lives)
    "run_type",             # text / tab / image_ref / chart_ref / …
    "run_index",            # position within parent paragraph
    "order_index",          # global sequential position in document
    "header_footer_type",   # body / header / footer
    "nested_table_depth",   # 0 for top-level, >0 inside table cells
    "page_break_before",    # paragraph-level page break flag
    "section_break_after",  # section break appended after paragraph
    "section_break_type",   # nextPage / continuous / evenPage / oddPage

    # ===== 3. IDs and hierarchy =====
    # Atomic-level IDs (raw extraction)
    "word_id",              # pdf, ocr, html
    "text_object_id",       # native pdf
    "box_id",               # html
    "run_id",               # docx, pptx
    "image_id",             # pdf, docx, pptx
    "raw_shape_id",         # pdf
    "shape_id",             # pdf, pptx
    "link_id",              # pdf, html
    "chart_id",             # pptx

    # Mid-pipeline IDs (pdf + html)
    "cell_id",
    "line_id",
    "layout_id",
    "cell_count",
    "paragraph_id",
    "gutter_id",
    "reading_order",
    "struct_group_id",
    "stream_group_id",

    # Shared-pipeline IDs
    "block_id",
    "chunk_index",
    "chunk_id",

    # Aggregated hierarchical ID lists
    "word_ids",
    "line_ids",
    "cell_ids",
    "raw_shape_ids",
    "group_ids",            # PPTX element grouping
    "group_names",          # PPTX element grouping
    "group_depth",          # PPTX element grouping
    "container_shape_ids",  # PPTX element grouping

    # Aggregated hierarchical counts
    "run_count",            # runs collapsed into this paragraph
    "paragraph_count",      # paragraphs collapsed into this line
    "block_count",

    # Layout classification
    "layout_type",
    "layout_score",
    "block_type",
    "shape_role",           # page_background / table_grid / underline / …

    # ===== 4. Table IDs and structure =====
    "table_id",
    "row_start",
    "col_start",
    "col_end",
    "rowspan",
    "colspan",
    "band_total_cols",
    "table_cell_role",      # header / data / row_label
    "role",                 # html generic cell role
    "row_complete",
    "is_last_tr_below",
    "is_subheading",
    "is_numeric_cell",
    "is_numeric_line",
    "grid_cell_trustworthy",
    "grid_row_trustworthy",

    "table_cell_id",
    "table_cell_index",
    "table_header_flag",
    "table_row_id",
    "table_row_cell_count",

    "struct_table_id",
    "struct_table_row_id",
    "struct_table_cell_id",
    "struct_col_span",
    "struct_row_span",

    "table_grid_id",        # detected grids in df_shapes (≠ table_id)
    "grid_cell_id",         # detected grid cells in df_shapes (≠ table_cell_id)
    "grid_row_start",
    "grid_rowspan",
    "grid_colspan",

    # ===== 5. Shape core =====
    "raw_shape_type",
    "shape_type",
    "shape_orientation",    # horizontal / vertical / unknown

    # ===== 6. Gutter / reading column (pdf — on words df) =====
    "gutter_id_left",
    "gutter_id_right",
    "reading_column",

    # ===== 7. Content =====
    "chunk_heading",
    "chunk_path",
    "text",
    "text_raw",             # pre-processed OCR text
    "cells",

    # Heading hierarchy
    "heading_level",
    "heading_level_raw",
    "heading_score",
    "hierarchy_marker",
    "hierarchy_type",
    "heading_type",
    "hybrid_heading_text",
    "embed_char_count",
    "heading_fp_id",
    "heading_fingerprint",
    "active_heading_id",
    "heading_id",
    "parent_heading_id",
    "heading_hash",

    # PDF form fields
    "form_widget",
    "form_field_name",
    "form_value",
    "form_tooltip",
    "form_is_empty",

    # Image content
    "img_alt",
    "img_src",

    # ===== 8. Geometry =====
    "x_left",
    "x_right",
    "y_top",
    "y_bottom",
    "y_center",
    "top_bucket",
    "center_bucket",
    "bottom_bucket",
    "width",
    "height",
    "area",

    # ===== 9. Style =====
    "font_name",
    "font_family",
    "font_size",
    "font_size_ratio",
    "font_weight",
    "text_align",
    "text_orientation",
    "non_stroking_color",
    "stroking_color",
    "linewidth",
    "has_vertical_line",

    # Gutter geometry (pdf — on gutters df)
    "gutter_x_left",
    "gutter_x_right",
    "gutter_y_top",
    "gutter_y_bottom",
    "gutter_width",

    # Shape paint state (pdf)
    "fill",
    "stroke",
    "paint_op",

    # ===== 10. Calculated features =====
    # Counts
    "token_count",          # from tiktoken
    "char_count",
    "alpha_count",
    "digit_count",
    "uppercase_count",
    "word_count",
    "alpha_word_count",
    "capitalized_word_count",

    # Ratios
    "bold_ratio",
    "italic_ratio",
    "capitalized_token_ratio",
    "underlined_ratio",
    "strikethrough_ratio",

    # Boolean style flags (derived from ratios / direct XML attributes)
    "is_bold",
    "is_italic",
    "is_uppercase",
    "is_underlined",
    "is_strikethrough",
    "script_type",          # "superscript", "subscript", or None

    # Row / line classification
    "line_number_flag",     # pdf — row looks like a line-number margin label
    "line_class",
    "line_score",

    # ===== 11. Page context =====
    "page_width",
    "page_height",

    # ===== 12. List / outline =====
    "list_num_id",          # docx, pptx
    "list_level",           # docx, pptx
    "list_label",           # docx, pptx  — rendered label ("1.", "a)", "•")
    "list_type",            # pptx        — bullet / autoNumber / none
    "list_auto_type",       # pptx        — arabicPeriod / romanUC / …
    "list_start_at",        # pptx        — override start value
    "outline_level",        # docx, pptx  — 0-based heading depth from style

    # ===== 13. Relationships =====
    # Links and bookmarks
    "has_link",
    "link_url",
    "link_dest",
    "link_type",
    "ixbrl_id",
    "ix",
    "html_data_attrs",
    "hyperlink_url",        # docx
    "bookmark_id",          # docx  — single bookmark on this run
    "bookmark_ids",         # docx  — all bookmark IDs on paragraph
    "bookmark_names",       # docx  — all bookmark names on paragraph

    # Header-detection features from detect_cell_roles (row-level → cells)
    "table_row_style",
    "hdr_n_populated",
    "hdr_frac_numeric",
    "hdr_frac_bold",
    "hdr_frac_th",
    "hdr_has_year",
    "hdr_has_date",
    "hdr_is_currency_unit",
    "hdr_has_unit_phrase",
    "hdr_col0_blank",
    "hdr_in_row0_span",

    # Shape overlaps / decorations (pdf)
    "shape_id_container",         # shape acting as a bounding rectangle
    "shape_id_underline",         # shape that provides the underline flag
    "shape_id_strikethrough",
    "shape_id_tr_above",
    "shape_id_tr_below",
    "inside_rect_shape",
    "background_non_stroking_color",
    "background_stroking_color",

    # ===== 14. PDF struct-tree provenance =====
    "word_source",
    "struct_ancestors",
    "struct_raw_ancestors",
    "struct_ancestor_ids",
    "textbox_id",
    "struct_tag_id",
    "struct_tag",
    "struct_raw_tag",
    "struct_scope",
    "struct_headers",
    "mcid",
    "bdc_tag",
    "dfs_position",         # mostly inaccurate, even in well-tagged pdfs

    # ===== 15. HTML DOM provenance =====
    "dom_id",
    "dom_class",
    "wrapping_tag",
    "split_reason",
    "ancestor_ids",
    "ancestor_classes",
    "ancestor_aria_roles",

    # ===== 16. Style inheritance (docx, pptx) =====
    "source_part",          # XML part: body / footnotes / header / footer / …
    "source_part_id",       # ID of the item within that part
    "placeholder_type",     # pptx — title / body / subtitle / …
    "shape_name",           # pptx
    "style_id",
    "style_name",
    "paragraph_style_id",
    "paragraph_style_name",
    "effective_paragraph_style_id",
    "effective_paragraph_style_name",
    "character_style_id",
    "character_style_name",
    "effective_character_style_id",
    "effective_character_style_name",

    # ===== 17. Image metadata (pdf) =====
    "image_width",
    "image_height",
    "bpc",                  # bits per component
    "colorspace",
    "colorspace_name",
    "ext",                  # file extension (png / jpg / …)
    "filter",               # pdf compression filter
    "smask",                # soft-mask xref (transparency)
    "has_transparency",
    "dpi_x",
    "dpi_y",

    # ===== 18. Shape internals (pdf) =====
    "candidate_group_id",   # grouping scratch ID during merge
    "has_intersection",
    "intersection_count",
    "intersecting_line_ids",
    "color_hex",
    "color_label",
    "hl_class",
    "is_line_start",

    # Cross-document references (docx) — rarely needed during inspection
    "comment_id",
    "footnote_id",
    "endnote_id",

    # =========================================================================
    # PART B — INTERMEDIATE / DIAGNOSTIC COLUMNS
    # Consumed at some stage and dropped before final output. Grouped by the
    # pipeline stage that produces them; ordering mirrors registry_aggregator.
    # =========================================================================

    # ----- OCR word-level internals (color / font-size estimation) -----
    "non_stroking_color_hex_raw",
    "background_non_stroking_color_hex_raw",
    "ink_coverage",
    "font_pointsize",
    "ocr_confidence",
    "has_capital",
    "has_ascender",
    "has_descender",

    # ----- Native PDF stream-group pair features (step_07) -----
    "large_gap",
    "new_textbox",
    "objects_between",
    "same_line",
    "same_struct",
    "same_table",
    "shifted_left",
    "y_decreases",
    "encapsulation_split",

    # ----- Cell-builder line-splitting diagnostics (word/line level) -----
    "line_em_threshold",
    "line_is_bimodal",
    "line_has_punct",
    "line_max_em",
    "line_median_em",
    "line_n_gaps",
    "line_n_words",
    "line_split_em",
    "line_jump_ratio",
    "line_alpha_ratio",
    "line_numeric_ratio",
    "line_stopword_hits",
    "line_cap_ratio",

    # ----- Cell grouper (vstack / grouped-row) diagnostics -----
    "cell_id_orig",
    "line_id_orig",
    "grouped_row_id",
    "grouped_row_n_vstacks",
    "grouped_row_x_left",
    "grouped_row_x_right",
    "grouped_row_y_top",
    "grouped_row_y_bottom",
    "vstack_id",
    "vstack_gap_em",
    "vstack_n_cells",
    "vstack_n_lines",
    "vstack_score",
    "vstack_width",
    "vstack_alone_in_band",
    "x_movement",
    "y_movement",
    "gap_em_right",

    # ----- PDF layout grouping (line → block) -----
    "line_gap",
    "page_gap_thresh",
    "median_gap",
    "width_ratio",

    # ----- Heading detector debug -----
    "pdf_heading_candidate",
    "pdf_heading_suppressed",
    "pdf_heading_suppressed_reason",
    "docx_heading_candidate",
    "docx_heading_suppressed",
    "docx_heading_suppressed_reason",
    "heading_source",
    "heading_score_debug",
    "heading_weight_static",
    "heading_weight_dynamic",
    "line_gap_below",
    "numbered_heading_group",
    "parent_heading_text",
    "style_change",
    "temp_section_id",

    # ----- TOC detector KPI columns (step_02) -----
    "toc_heading_candidate",
    "toc_has_dot_leaders",
    "toc_row_candidate",
    "toc_row_page_token",
    "toc_row_page_type",
    "toc_segment_id",
    "toc_segment_score",
    "toc_segment_score_detail",

    # ----- Exhibit detector KPI columns -----
    "exhibit_heading_candidate",
    "exhibit_row_candidate",
    "exhibit_row_number",
    "exhibit_row_strength",
    "exhibit_segment_id",
    "exhibit_segment_score",
    "exhibit_segment_score_detail",

    # ----- Table-builder analysis leftovers (pdf) -----
    "total_cols",
    "total_lines",

    # ----- PPTX reading-order intermediates (step_06) -----
    "container_group_ids",
    "reading_group_key",
    "reading_group_order",
    "reading_group_path",
    "reading_group_bboxes",
    "reading_group_rank",

    # ----- DOCX run-level detail -----
    "hyperlink_id",
    "field_id",
    "field_type",
    "field_phase",
    "event_tag",
    "chart_rel_id",
    "is_deleted_revision",
    "is_inserted_revision",

    # ----- Page-label detector scratch / debug -----
    "page_label_series_id",     # kept for the section classifier
    "page_label_group_id",      # kept for the section classifier
    "page_label_raw",
    "page_label_candidate",
    "page_label_cell_sharing",
    "page_label_wrapper",
    "page_label_score",
    "page_label_token",
    "alternation_mode",
]

_seen: set[str] = set()
_dupes = {col for col in MASTER_COLUMN_ORDER if col in _seen or _seen.add(col)}
if _dupes:
    raise ValueError(
        f"MASTER_COLUMN_ORDER contains duplicate entries: {sorted(_dupes)}. "
        "Selecting a duplicated name twice (df[existing_cols]) produces "
        "repeated columns, which pandas then disambiguates as 'name.1', etc. "
        "Remove the duplicate(s) from MASTER_COLUMN_ORDER."
    )
del _seen, _dupes

# =====================
# Reorder Function
# =====================

def reorder_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reorder columns in a DataFrame to match MASTER_COLUMN_ORDER.

    Columns present in df but absent from MASTER_COLUMN_ORDER are appended
    at the end — a useful signal that the master list needs updating.

    Args:
        df: DataFrame to reorder.
        dataframe_type: Optional label for documentation / future customisation.

    Returns:
        DataFrame with reordered columns (same data, new column order).
    """
    if df.empty:
        return df

    existing_cols = [col for col in MASTER_COLUMN_ORDER if col in df.columns]
    other_cols = [col for col in df.columns if col not in MASTER_COLUMN_ORDER]
    return df[existing_cols + other_cols]

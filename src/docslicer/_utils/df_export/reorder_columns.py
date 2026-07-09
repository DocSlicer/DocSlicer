"""
reorder_columns.py

Centralised column ordering for all DataFrames in the parsing pipeline
(PDF, DOCX, PPTX, HTML, OCR).

MASTER_COLUMN_ORDER is the single source of truth for every column name
produced anywhere in the pipeline.  Columns not listed here end up appended
at the end of any reordered DataFrame — a useful signal that the list is
out of date.

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
# Every column produced anywhere in the pipeline belongs here.
# Groups:
#   1.  Document / page identifiers          (all formats)
#   2.  Run / atom provenance                (docx, pptx)
#   3.  IDs and hierarchy                    (all formats)
#   4.  Content                              (all formats)
#   5.  Geometry                             (all formats)
#   6.  Style                                (all formats)
#   7.  Calculated features                  (all formats)
#   8.  Page context                         (all formats)
#   9.  List / outline                       (docx, pptx, html)
#   10. Relationships                        (all formats)
#   11. HTML DOM provenance                  (html)
#   12. Style inheritance                    (docx, pptx)
#   13. Table-specific fields                (all formats)
#   14. Image metadata                       (pdf)
#   15. Shape internals                      (pdf)
#   16. Line / band statistics               (pdf, html)
#   17. Table-builder analysis               (pdf)
#   18. Scores                               (all formats)
#   19. Cross-document references            (docx)

MASTER_COLUMN_ORDER: List[str] = [

    # ===== 1. Document / page identifiers =====
    "page_number",
    "hl_class", # TODO: Temp table hr debug
    "page_label",
    "section",
    "section_id",           # docx
    "slide_index",          # pptx

    # ===== 2. Run / atom provenance =====
    # (docx + pptx: every row comes from a single XML run;
    #  these fields say *where* in the document structure it lives)
    "run_type",             # docx, pptx  — text / tab / image_ref / chart_ref / …
    "run_index",            # docx, pptx  — position within parent paragraph
    "order_index",          # docx, pptx  — global sequential position in document
    "source_part",          # docx, pptx  — XML part: body / footnotes / endnotes / header / footer / …
    "source_part_id",       # docx, pptx  — ID of the item within that part
    "header_footer_type",   # docx        — body / header / footer
    "nested_table_depth",   # docx        — 0 for top-level, >0 for tables inside table cells
    "page_break_before",    # docx        — paragraph-level page break flag
    "section_break_after",  # docx        — section break appended after paragraph
    "section_break_type",   # docx        — nextPage / continuous / evenPage / oddPage

    # ===== 3. IDs and hierarchy =====

    # Atomic-level IDs (raw extraction)
    "word_id",              # pdf, ocr, html
    "text_object_id",       # native pdf's
    "box_id",               # html
    "run_id",               # docx, pptx
    "image_id",             # pdf, docx, pptx
    "raw_shape_id",         # pdf
    "shape_id",             # pdf, pptx
    "link_id",              # pdf, html
    "chart_id",             # pptx
    "placeholder_type",     # pptx        — title / body / subtitle / …

    # Mid-pipeline IDs (pdf + html)
    "cell_id",
    "line_id",
    "layout_id",
    "cell_count",
    "temp_line_id",
    "paragraph_id",
    "gutter_id",
    "gutter_candidate_id",
    "reading_order",
    "struct_group_id",
    "stream_group_id",
    "stream_group_trigger",


    # Shared-pipeline IDs
    "block_id",
    "chunk_index",
    "chunk_id",

    # Aggregated hierarchical ID lists
    "word_ids",
    "line_ids",
    "run_ids",              # docx, pptx  — list of run_id values aggregated here
    "paragraph_ids",        # docx, pptx  — list of paragraph_id values aggregated here
    "raw_shape_ids",

    

    # Secondary / structural IDs HTML
    "structure_tag",
    "structure_tag_id",

    # Aggregated hierarchical counts
    "run_count",            # docx, pptx  — runs collapsed into this paragraph
    "paragraph_count",      # docx, pptx  — paragraphs collapsed into this line
    "block_count",

    # Layout IDs
    "layout_type",
    "layout_score",
    "block_type",

    # Table IDs # TODO WIP
    "table_id",

    "row_start",
    "col_start",
    "col_end",
    "rowspan", 
    "colspan",
    "band_total_cols",
    "role",                 # header / data / row_label

    "row_complete",
    "is_last_tr_below",
    "is_subheading",
    "is_numeric_like",
    "grid_trustworthy",

    "table_cell_id",
    "table_header_flag",

    "struct_table_id",
    "table_row_id", # Not a column in table_cell_df
    "struct_table_row_id",
    "struct_table_cell_id",
    "struct_col_span",
    "struct_row_span",

    "table_grid_id", # Detected grids in df_shapes, not the same as table_id (there are more tables than grids ~ tables with only horizontal lines)
    "grid_cell_id",  # Detected grid cells in df_shapes, not the same as table_cell_id 
                     # (1 grid cell can have multiple table cells, for example in tables without inside borders)

    "grid_row_start",	
    "grid_col_start",	
    "grid_rowspan",	
    "grid_colspan",


    





    

    # Shape core
    "raw_shape_type",
    "shape_type",
    "shape_orientation",    # pdf  — horizontal / vertical / unknown

    # Gutter / reading column (pdf — on words df)
    "gutter_id_left",       # ID of the gutter immediately to the left of this word
    "gutter_id_right",      # ID of the gutter immediately to the right
    "reading_column",       # 1-based column index assigned by gutter split

    # ===== 4. Content =====
    "chunk_heading",
    "chunk_path",
    "shape_id_tr_above", # TODO: Reorder after stable debug
    "shape_id_tr_below", # TODO: Reorder after stable debug
    "text",
    "text_raw", # Pre-processed OCR text
    "cells",

    # Heading hierarchy
    "heading_level",
    "heading_score",
    "heading_score_debug",
    "hierarchy_marker",
    "hierarchy_type",
    "heading_type",
    "embed_char_count", #Temporarily
    "heading_fp_id",
    "heading_fingerprint",
    "active_heading_id",
    "heading_id",
    "parent_heading_id",
    "heading_hash",
    "heading_weight_static",
    "heading_weight_dynamic",
    "heading_sequence",
    

    # ===== 5. Geometry =====
    "x_left",
    "x_right",
    "y_top",
    "y_bottom",
    "x_center",
    "y_center",
    "top_bucket",
    "center_bucket",
    "bottom_bucket",
    "width",
    "height",
    "area",

    # ===== 6. Style =====
    "font_name",
    "font_family",
    "font_size",
    "font_size_ratio",
    "font_weight",
    "text_align",
    "layout_align",
    "text_orientation",
    "non_stroking_color",
    "stroking_color",
    "linewidth",

    # Gutter geometry (pdf — on gutters df)
    "gutter_x_left",
    "gutter_x_right",
    "gutter_y_top",
    "gutter_y_bottom",
    "gutter_width",
    "intersecting_horizontal_line_ids",

    # Gutter candidate scratch columns (pdf — intermediate)
    "sliding_window_id",
    "sliding_window",
    "x_page_min",
    "x_page_max",
    "gutter_candidate_x_left",
    "gutter_candidate_x_right",

    # Shape paint state (pdf)
    "fill",
    "stroke",
    "paint_op",

    # ===== 7. Calculated features =====
    # Counts
    "token_count", # From Tiktoken
    #"embed_char_count",
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
    "digit_ratio",
    "uppercase_ratio",
    "capitalized_token_ratio",
    "underlined_ratio",
    "strikethrough_ratio",
    "inside_rect_fraction",

    # Boolean style flags (derived from ratios / direct XML attributes)
    "is_bold",
    "is_italic",
    "is_uppercase",
    "is_underlined",
    "is_strikethrough",
    "script_type", # "superscript", "subscript", or None (blank)

    # Average metrics
    "avg_word_len",
    "avg_cell_count",

    # PDF-specific detection flags
    "line_number_flag",     # pdf  — row looks like a line-number margin label

    # ===== 8. Page context =====
    "page_width",
    "page_height",

    # Temporary page-label scratch IDs
    "page_label_series_id",
    "page_label_raw",
    "page_label_candidate",
    "page_label_type",
    "page_label_value",
    "page_label_cell_sharing",
    "page_label_wrapper",
    "page_label_score",

    # ===== 9. List / outline =====
    "list_num_id",          # docx, pptx
    "list_level",           # docx, pptx
    "list_label",           # docx, pptx  — rendered label ("1.", "a)", "•")
    "list_type",            # pptx        — bullet / autoNumber / none
    "list_auto_type",       # pptx        — arabicPeriod / romanUC / …
    "list_start_at",        # pptx        — override start value
    "outline_level",        # docx, pptx  — 0-based heading depth from style

    # ===== 10. Relationships =====
    # Links and bookmarks
    "has_link",
    "link_url",
    "link_dest",
    "link_type",
    "ixbrl_id",
    "html_data_attrs",
    "bookmark_id",          # docx  — single bookmark on this run
    "bookmark_ids",         # docx  — all bookmark IDs on paragraph
    "bookmark_names",       # docx  — all bookmark names on paragraph

    # Shape overlaps / decorations (pdf)
    "has_horizontal_grid_line",
    "has_vertical_grid_line",
    "shape_id_underline",       # pdf — shape that provides the underline flag
    "shape_id_strikethrough",
    "shape_id_horizontal_grid_line",
    "shape_id_vertical_grid_line",   # pdf — shape that provides the vertical line flag
    "inside_rect_shape",
    "background_non_stroking_color",
    "background_stroking_color",
    "shape_id_container",       # pdf — shape acting as a bounding rectangle

    # ===== 11. PDF Struct Tree provenance =====
    # Main cols
    "struct_ancestors",	
    "struct_raw_ancestors",	
    "struct_ancestor_ids",
    "textbox_id",
    # Other
    "struct_tag_id",
    "struct_tag",
    "struct_raw_tag",
    
    "struct_scope",
    "struct_headers",
    "mcid",
    "bdc_tag",
    "dfs_position",	 # mostly inaccurate, even in well-tagged pdf's

    

    # ===== 11. HTML DOM provenance =====
    "dom_id",
    "dom_class",
    "wrapping_tag",
    "split_reason",
    "ancestor_ids",
    "ancestor_classes",
    "ancestor_tags",
    "ancestor_aria_roles",

    # ===== 12. Style inheritance =====
    # (docx: resolved through the style-inheritance chain)
    "paragraph_style_id",
    "paragraph_style_name",
    "effective_paragraph_style_id",
    "effective_paragraph_style_name",
    "character_style_id",
    "character_style_name",
    "effective_character_style_id",
    "effective_character_style_name",

    # ===== 13. Table-specific fields =====
    """"
    "row_start",
    "col_start",
    "col_end",
    "rowspan",
    "colspan",
    "role",                 # header / data / row_label
    "band_total_cols",
    """

    # ===== 14. Image metadata (pdf) =====
    "image_width",
    "image_height",
    "xref",                 # pdf internal object reference
    "bpc",                  # bits per component
    "colorspace",
    "colorspace_name",
    "ext",                  # file extension (png / jpg / …)
    "filter",               # pdf compression filter
    "smask",                # soft-mask xref (transparency)
    "has_transparency",
    "dpi_x",
    "dpi_y",

    # ===== 15. Shape internals (pdf) =====
    # Merger intermediate
    "candidate_group_id",       # grouping scratch ID during merge
    "shape_role",               # page_background / table_grid / underline / separator / background_band / other
    #"table_grid_id",            # shared ID across the lines of one detected table grid
    "has_intersection",
    "intersection_count",
    "intersecting_line_ids",
    "color_hex",
    "color_label",

    # Cell-builder shape diagnostics
    "is_sentence_like",
    "sentence_score",
    "is_line_start",

    # ===== 16. Line / band statistics (pdf, html) =====
    "line_gap",
    "line_gap_to_prev",
    "page_median_gap",
    "page_gap_thresh",

    # Column / gutter gap metrics (pdf)
    "median_x0x1_gap",
    "max_x0x1_gap",
    "gap_ratio",
    "width_ratio",

    # ===== 17. Table-builder analysis (pdf intermediate) =====
    # These columns live on the lines/bands df during table classification
    # and are typically dropped before the final output.
    "band_table_score",
    "row_pattern",
    "row_pattern_max",
    "row_pattern_reuse",
    "column_reuse",
    "reused_cols",
    "atomic_cells",
    "atomic_cell_ratio",
    "total_cols",
    "total_lines",
    "max_cell_count",
    "max_colspan",
    "is_multi_cell",
    "is_strong_multi_cell",
    "is_prose_candidate",
    "is_justified_prose_like",
    "is_full_span_line",
    "multi_cell_line_ratio",
    "multi_cell_lines",
    "strong_multi_cell_line_ratio",
    "strong_multi_cell_lines",
    "prose_candidate_lines",
    "justified_prose_lines",
    "justified_prose_ratio",
    "full_span_lines",
    "full_span_line_ratio",
    "large_gap_lines",
    "large_gap_ratio",
    "gap_count",
    "gap_std",
    "gap_cv",
    "median_cell_words",

    # ===== 18. Scores =====
    "table_row_score",
    "table_header_score",
    "table_title_score",
    "avg_table_row_score",
    "average_table_score",

    # ===== 18. OCR Color Estimation =====
    "non_stroking_color_hex_raw",
    "background_non_stroking_color_hex_raw",
    "ink_coverage",

    # ===== 19. Cross-document references (docx) =====
    # IDs pointing to objects elsewhere in the document (comments, footnotes, endnotes).
    # Sit at the end because they're rarely needed during inspection.
    "comment_id",
    "footnote_id",
    "endnote_id",
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
    dataframe_type: DataFrameType | None = None,
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

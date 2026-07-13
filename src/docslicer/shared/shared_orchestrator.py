# shared_orchestrator.py - Shared document processing steps
"""
Shared pipeline steps that run for both HTML and PDF documents.

These steps operate on a lines_df DataFrame that has already been
processed by the type-specific orchestrator (HTML or PDF).

Pipeline Steps:
    01. TOC Detection - Detect table of contents
    02. Exhibit Detection - Detect exhibits
    03. Navigation Detection - Detect navigation blocks
    04. Doc Region Assignment - Assign document regions
    05. Hierarchy Assignment - Detect headings and structural markers
    06. Hierarchy Builder - Fingerprints, weights, parent/level assignment
    07. Block Merger (lines -> blocks)
    08. Chunk Builder (blocks -> chunks)
"""
import pandas as pd
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Shared Preprocessing Steps
from .step_01_toc_detector import detect_and_annotate_tocs
from .step_02_exhibit_detector import detect_and_mark_exhibits
from .step_03_navigation_detector import detect_navigation_blocks
from .step_05_heading_detector import detect_headings
from .step_04_section_classifier import classify_sections
from .step_06_hierarchy_builder import assign_doc_hierarchy

# Shared Pipeline Steps
from .step_07_block_merger import merge_blocks
from .step_08_chunk_builder import build_chunks

# Config loaders
from .._utils.io.yaml_loader import load_yamls
from .._utils.timing import timed_step

# Settings schema
from .._config import ParseConfig, DEFAULT_CONFIG


def run_pipeline(
    lines_df: pd.DataFrame,
    df_table_cells: Optional[pd.DataFrame] = None,
    chart_points_df: Optional[pd.DataFrame] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    config: ParseConfig = None,
    metadata: Dict[str, Any] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run shared document processing steps.
    
    Takes df_lines from HTML or PDF orchestrator and:
    1. Runs shared preprocessing (TOC, exhibits, doc regions, hierarchy)
    2. Merges lines into blocks
    3. Builds chunks from blocks
    
    Args:
        lines_df: DataFrame with lines from HTML or PDF orchestrator
        df_table_cells: Optional DataFrame with table cells
        chart_points_df: Optional DataFrame with chart datapoints (docx/pptx),
            used by the block merger to render "chart" blocks
        on_stage: Optional callback for progress updates
        config: ParseConfig with optimal_chars, max_chars, etc.
        metadata: Document metadata to attach to output
    
    Returns:
        DataFrame with chunks
    """
    config = config or DEFAULT_CONFIG
    
    if lines_df.empty:
        return lines_df
    
    df = lines_df.copy()
    
    # Load configs for shared preprocessing
    _, page_label_config, hierarchy_type_pattern_config, exhibit_pattern_config = load_yamls()
    
    # ============================================================
    # STAGE: SHARED PREPROCESSING
    # ============================================================
    if on_stage:
        on_stage("detect_hierarchy")
    
    # Step 01 - TOC Detection
    if page_label_config:
        with timed_step("toc_detection", logger=logger):
            df = detect_and_annotate_tocs(df, page_label_config)

    # Step 02 - Exhibit Detection
    if exhibit_pattern_config:
        with timed_step("exhibit_detection", logger=logger):
            df = detect_and_mark_exhibits(df, exhibit_pattern_config)

    # Step 03 - Navigation Detection
    with timed_step("navigation_detection", logger=logger):
        df = detect_navigation_blocks(df)

    # Step 04 - Doc Region Assignment
    with timed_step("doc_region_assignment", logger=logger):
        df = classify_sections(df)

    # Step 05 - Heading Detection
    if hierarchy_type_pattern_config:
        with timed_step("heading_detection", logger=logger):
            df = detect_headings(df, hierarchy_type_pattern_config)

    # Step 06 - Hierarchy Assignment
    with timed_step("hierarchy_assignment", logger=logger):
        df = assign_doc_hierarchy(df)
    
    # ============================================================
    # STAGE: CHUNKING
    # ============================================================
    if on_stage:
        on_stage("build_chunks")
    
    # Step 06 - Block Merging (lines -> blocks)
    # Use df_table_cells passed from HTML/PDF orchestrator (has proper cell structure)
    # Fall back to extracting from df only if not provided
    if df_table_cells is None and "table_id" in df.columns:
        table_lines = df[df["table_id"].notna()]
        if not table_lines.empty:
            df_table_cells = table_lines
    
    with timed_step("block_merging", logger=logger):
        df_blocks = merge_blocks(
            df,
            df_table_cells,
            chart_points_df,
            table_representation=config.table_representation,
        )

    # Step 07 - Chunk Building (blocks -> chunks)
    if not config.chunking:
        return df_blocks, pd.DataFrame()

    with timed_step("chunk_building", logger=logger):
        df_chunks = build_chunks(
            df_blocks,
            max_chunk_chars=config.max_chunk_size,
            optimal_chunk_chars=config.optimal_chunk_size,
            softmin_chunk_chars=config.min_chunk_size,
            min_chunk_chars=400,
            merge_small_chunks=config.merge_small_chunks,
            exact_tokens=config.exact_tokens,
        )

    # ============================================================
    # Attach Metadata to Chunks
    # ============================================================
    if metadata:
        for key, value in metadata.items():
            if key not in df_chunks.columns:
                df_chunks[key] = value

    return df_blocks, df_chunks

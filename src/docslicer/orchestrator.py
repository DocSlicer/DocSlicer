# orchestrator.py - Main document processing entry point
"""
Document processing orchestrator.

This is the main entry point for document processing. It:
1. Detects document type (HTML or PDF)
2. Runs the appropriate type-specific pipeline (HTML or PDF)
3. Runs the shared pipeline (Block Merger → Chunk Builder)

Usage:
    from docslicer.orchestrator import run_pipeline

    # For HTML
    df_chunks, df_table_cells = await run_pipeline(
        content=html_string,
        content_type="html",
        source_url="https://example.com/doc.html",
        config=config,
    )

    # For PDF
    df_chunks, df_table_cells = await run_pipeline(
        content=pdf_bytes,
        content_type="pdf",
        source_url="https://example.com/doc.pdf",
        config=config,
    )
"""
import pandas as pd
from typing import Dict, Any, Callable, Optional, Union, Literal, Tuple
import json
import logging

_log = logging.getLogger(__name__)

from .html.html_orchestrator import run_pipeline as run_html_pipeline
from .pdf.pdf_orchestrator import run_pipeline as run_pdf_pipeline
from .shared.shared_orchestrator import run_pipeline as run_shared_pipeline

from docslicer._config import ParseConfig, DEFAULT_CONFIG
from ._utils.df_exports import export_production


ContentType = Literal["html", "pdf"]


async def run_pipeline(
    content: Union[str, bytes],
    content_type: ContentType,
    source_url: str = None,
    metadata: Dict[str, Any] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    config: ParseConfig = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run the full document processing pipeline.
    
    This is the main entry point for document processing. It orchestrates:
    1. Type-specific processing (HTML steps or PDF steps)
    2. Shared processing (Preprocessing → Block Merger → Chunk Builder)
    3. Metadata resolution and embedding
    4. Production filtering
    
    Args:
        content: Document content (HTML string or PDF bytes)
        content_type: Type of document ("html" or "pdf")
        source_url: Original URL (for link normalization and metadata)
        metadata: Document metadata to attach to output
        on_stage: Optional callback for progress updates ("parsing", "hierarchy", "chunking")
        config: ParseConfig with optimal_chars, max_chars, etc.
    
    Returns:
        Tuple of (chunks_df DataFrame, df_table_cells DataFrame)
        - chunks_df: DataFrame with chunk-level data + resolved metadata
        - df_table_cells: Optional DataFrame with table cells (if any)
    
    Raises:
        ValueError: If content_type is not supported
    """
    config = config or DEFAULT_CONFIG
    
    # ============================================================
    # Step 1: Run type-specific pipeline
    # ============================================================
    if content_type == "html":
        if not isinstance(content, str):
            raise ValueError("HTML content must be a string")
        
        discovered_metadata, df_lines, df_table_cells = await run_html_pipeline(
            html=content,
            source_url=source_url,
            on_stage=on_stage,
        )
    
    elif content_type == "pdf":
        if not isinstance(content, bytes):
            raise ValueError("PDF content must be bytes")
        
        discovered_metadata, df_lines, df_table_cells = await run_pdf_pipeline(
            pdf_bytes=content,
            source_url=source_url,
            on_stage=on_stage,
        )
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}. Use 'html' or 'pdf'.")
    
    # Return early if type-specific pipeline returned empty
    if df_lines.empty:
        _log.warning(f"Type-specific pipeline returned empty df_lines")
        return pd.DataFrame(), df_table_cells
    
    # ============================================================
    # Step 2: Run shared pipeline (lines → chunks)
    # ============================================================
    df_chunks = run_shared_pipeline(
        lines_df=df_lines,
        df_table_cells=df_table_cells,
        on_stage=on_stage,
        config=config,
        metadata=metadata,
    )
    
    # ============================================================
    # Step 3: Extract screenshot before adding metadata to chunks
    # ============================================================
    
    # Extract screenshot for HTML only (don't add to every chunk - wasteful)
    # PDF uses native coordinates and doesn't need screenshots
    screenshot_base64 = discovered_metadata.pop("screenshot_base64", None) if discovered_metadata else None
    page_dimensions = discovered_metadata.pop("page_dimensions", None) if discovered_metadata else None
    
    # ============================================================
    # Step 4: Resolve and embed discovered metadata into chunks
    # ============================================================
    
    if discovered_metadata and not df_chunks.empty:
        # Resolve author: pick the longer of author_meta vs author_text
        author_meta = discovered_metadata.get("author_meta", [])
        author_text = discovered_metadata.get("author_text", [])
        if author_meta or author_text:
            author_meta_str = json.dumps(author_meta) if author_meta else ""
            author_text_str = json.dumps(author_text) if author_text else ""
            if len(author_text_str) > len(author_meta_str):
                df_chunks["author"] = json.dumps(author_text) if isinstance(author_text, list) else author_text
            else:
                df_chunks["author"] = json.dumps(author_meta) if isinstance(author_meta, list) else author_meta
        else:
            df_chunks["author"] = None
        
        # Resolve title: pick the longer of title_meta vs title_text
        title_meta = discovered_metadata.get("title_meta")
        title_text = discovered_metadata.get("title_text")
        if title_meta or title_text:
            title_meta_len = len(title_meta) if title_meta else 0
            title_text_len = len(title_text) if title_text else 0
            if title_text_len > title_meta_len:
                df_chunks["title"] = title_text
            else:
                df_chunks["title"] = title_meta
        else:
            df_chunks["title"] = None
        
        # Add ALL other metadata fields - export_production will filter unwanted ones
        # Skip screenshot-related fields (already extracted)
        for k, v in discovered_metadata.items():
            if k in ("screenshot_base64", "page_dimensions"):
                continue
            if k not in df_chunks.columns:  # Don't overwrite existing columns
                if isinstance(v, (list, dict)):
                    df_chunks[k] = json.dumps(v) if v else None
                else:
                    df_chunks[k] = v
    
    # ============================================================
    # Step 5: Filter for production (remove debug/internal columns)
    # ============================================================
    df_chunks = export_production(df_chunks, drop_none=True)
    
    # Return chunks, table cells, and screenshot info (for separate storage)
    # For HTML: screenshot_base64 is a single base64 string
    # For PDF: no screenshots needed (uses native PDF coordinates for overlay)
    screenshot_data = None
    if screenshot_base64:
        screenshot_data = {"html": screenshot_base64, "page_dimensions": page_dimensions}
    
    return df_chunks, df_table_cells, screenshot_data

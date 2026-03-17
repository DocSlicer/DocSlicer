"""Document metadata initialization — creates the baseline metadata dict."""
from __future__ import annotations

from typing import Optional, Dict, Any


def init_document_metadata(
    content_type: str,
    run_id: Optional[str] = None,
    document_id: Optional[str] = None,
    source_filename: Optional[str] = None,
    source_url: Optional[str] = None,
    is_password_protected: bool = False,
) -> Dict[str, Any]:
    """
    Initialize document metadata at pipeline start.

    This function can be used in two modes:
    1. Full mode (with run_id, document_id, source_filename): Used in production run orchestrator
    2. Minimal mode (only content_type): Used in parsing-only scripts/debug tools

    The parsing layer only discovers metadata fields (language, title, etc.) and doesn't
    require high-level orchestration fields (run_id, document_id, etc.).

    Args:
        content_type: "PDF" or "HTML" (required)
        run_id: Run/job identifier (optional, for orchestrator)
        document_id: Internal stable document ID (optional, for orchestrator)
        source_filename: Original filename (optional, for orchestrator)
        source_url: Optional source URL
        is_password_protected: Whether PDF is password protected

    Returns:
        Initial metadata dictionary with only non-None values
    """
    metadata: Dict[str, Any] = {"content_type": content_type.upper()}

    # Only add fields if they are provided (not None)
    if run_id is not None:
        metadata["run_id"] = run_id
    if document_id is not None:
        metadata["document_id"] = document_id
    if source_filename is not None:
        metadata["source_filename"] = source_filename
    if source_url is not None:
        metadata["source_url"] = source_url
    if is_password_protected:
        metadata["is_password_protected"] = is_password_protected

    return metadata

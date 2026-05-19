"""DocumentMetadata dataclass — single source of truth for document-level metadata."""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def _gen_uuid() -> str:
    return str(_uuid.uuid4())


@dataclass
class DocumentMetadata:
    """
    Document-level metadata produced by the parsing layer.

    Public fields are exposed via to_dict(). Internal pipeline fields
    (author_meta, author_text, title_meta, title_text, language_meta,
    language_text) are used during enrichment and excluded from output.
    """

    # ------------------------------------------------------------------
    # Core identifiers
    # ------------------------------------------------------------------

    document_id: str = field(default_factory=_gen_uuid)
    run_id: str = field(default_factory=_gen_uuid)
    processed_at: Optional[str] = None
    content_type: str = "unknown"          # pdf | html | docx | pptx
    source_filename: Optional[str] = None
    source_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    is_password_protected: bool = False

    # ------------------------------------------------------------------
    # Page geometry + OCR (after extraction)
    # ------------------------------------------------------------------

    page_count: Optional[int] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    page_format: Optional[str] = None      # A4_PORTRAIT | US_LETTER_LANDSCAPE | …
    has_mixed_page_sizes: bool = False
    has_ocr: bool = False
    needs_ocr: bool = False
    is_scanned: bool = False

    # ------------------------------------------------------------------
    # Content statistics
    # ------------------------------------------------------------------

    chars: Optional[int] = None
    estimated_tokens: Optional[int] = None

    # ------------------------------------------------------------------
    # Resolved document metadata (public output)
    # ------------------------------------------------------------------

    title: Optional[str] = None
    author: Optional[list[str]] = None
    language: Optional[str] = None
    language_confidence: Optional[str] = None   # high | medium | low
    language_source: Optional[str] = None       # meta | text

    profile: Optional[str] = None              # finance | legal | government | …
    document_type: Optional[str] = None        # sec_10k | earnings_deck | …

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    parsing_quality_score: Optional[float] = None
    processing_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal pipeline intermediates (not serialised)
    # ------------------------------------------------------------------

    author_meta: Optional[list[str]] = None
    author_text: Optional[list[str]] = None
    title_meta: Optional[str] = None
    title_text: Optional[str] = None
    language_meta: Optional[str] = None
    language_text: Optional[str] = None

    # ------------------------------------------------------------------
    # Extensibility
    # ------------------------------------------------------------------

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize public fields only — pipeline intermediates are excluded."""
        out: dict = {
            "document_id": self.document_id,
            "run_id": self.run_id,
            "processed_at": self.processed_at,
            "content_type": self.content_type,
        }
        if self.source_filename is not None:
            out["source_filename"] = self.source_filename
        if self.source_url is not None:
            out["source_url"] = self.source_url
        if self.file_size_bytes is not None:
            out["file_size_bytes"] = self.file_size_bytes
        out["is_password_protected"] = self.is_password_protected
        out["page_count"] = self.page_count
        if self.page_width is not None:
            out["page_width"] = self.page_width
        if self.page_height is not None:
            out["page_height"] = self.page_height
        if self.page_format is not None:
            out["page_format"] = self.page_format
        out["has_mixed_page_sizes"] = self.has_mixed_page_sizes
        out["has_ocr"] = self.has_ocr
        out["needs_ocr"] = self.needs_ocr
        out["is_scanned"] = self.is_scanned
        if self.chars is not None:
            out["chars"] = self.chars
        if self.estimated_tokens is not None:
            out["estimated_tokens"] = self.estimated_tokens
        if self.title is not None:
            out["title"] = self.title
        if self.author is not None:
            out["author"] = self.author
        if self.language is not None:
            out["language"] = self.language
        if self.language_confidence is not None:
            out["language_confidence"] = self.language_confidence
        if self.language_source is not None:
            out["language_source"] = self.language_source
        if self.profile is not None:
            out["profile"] = self.profile
        if self.document_type is not None:
            out["document_type"] = self.document_type
        if self.parsing_quality_score is not None:
            out["parsing_quality_score"] = self.parsing_quality_score
        if self.processing_time is not None:
            out["processing_time"] = self.processing_time
        if self.extra:
            out["extra"] = self.extra
        return out

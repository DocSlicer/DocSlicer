"""DocumentMetadata dataclass — single source of truth for document-level metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


@dataclass
class DocumentMetadata:
  """
  Document-level metadata produced by the parsing layer.

  Design
  ------
  - One instance per source document (HTML/PDF).
  - Created early in the parsing pipeline.
  - Enriched throughout the pipeline as more information becomes available.
  - Ultimately stored in `dim_document` and used by tagging/embedding layers.

  The dataclass is primarily used as a schema definition for document metadata.
  """

  # ------------------------------------------------------------------
  # Core identifiers (available before parsing)
  # ------------------------------------------------------------------

  run_id: str                         # Run / job identifier
  document_id: str                    # Stable internal ID (uuid/sha)
  content_type: Literal["html", "pdf"]  # Document type
  source_filename: str                # Sanitized filename
  source_url: Optional[str] = None    # Original URL if available
  is_password_protected: bool = False # For PDFs if detectable

  # ------------------------------------------------------------------
  # Page + extraction statistics (after box/word extraction)
  # ------------------------------------------------------------------

  chars: Optional[int] = None         # Total characters extracted
  estimated_tokens: Optional[int] = None  # Rough estimate (chars / 4)

  page_count: Optional[int] = None
  page_format: Optional[str] = None   # e.g. A4_PORTRAIT, US_LETTER_LANDSCAPE
  page_width: Optional[float] = None
  page_height: Optional[float] = None

  has_mixed_page_sizes: bool = False
  needs_ocr: bool = False             # Heuristic OCR requirement
  is_scanned: bool = False            # Likely scanned document

  # ------------------------------------------------------------------
  # Document metadata (after text extraction / line merging)
  # ------------------------------------------------------------------

  author_meta: Optional[list[str]] = None
  author_text: Optional[list[str]] = None

  title_meta: Optional[str] = None
  title_text: Optional[str] = None

  language_meta: Optional[str] = None
  language_text: Optional[str] = None
  language: Optional[str] = None

  language_confidence: Optional[str] = None   # high / medium / low
  language_source: Optional[str] = None       # meta / text

  profile: Optional[str] = None               # finance, legal, government, etc.
  document_type: Optional[str] = None         # sec_10k, earnings_deck, etc.

  # ------------------------------------------------------------------
  # Quality signals
  # ------------------------------------------------------------------

  parsing_quality_score: Optional[float] = None  # Heuristic 0–1 score

  # ------------------------------------------------------------------
  # Extensibility
  # ------------------------------------------------------------------

  extra: Dict[str, Any] = field(default_factory=dict)
  """
  Free-form metadata extension slot.

  Used for experimental or pipeline-specific fields
  (e.g. sec_cik, issuer_name, source_system).

  This avoids frequent schema changes to the dataclass.
  """
"""Document metadata extraction and enrichment package."""
from .schema import DocumentMetadata
from .init import init_document_metadata
from .ocr_detector import add_page_and_ocr_info
from .author import (
    extract_author_from_pdf_metadata,
    extract_author_from_html_metadata,
    extract_author_from_text,
    add_author_info,
)
from .title import (
    extract_title_from_pdf_metadata,
    extract_title_from_html_metadata,
    extract_title_from_text,
    add_title_info,
)
from .language import (
    extract_language_from_pdf_metadata,
    extract_language_from_html_metadata,
    extract_language_from_text,
    add_language_info,
)
from .profile import detect_document_profile, add_profile_info
from .enricher import add_document_information

__all__ = [
    "DocumentMetadata",
    "init_document_metadata",
    "add_page_and_ocr_info",
    "add_document_information",
    "add_author_info",
    "add_title_info",
    "add_language_info",
    "add_profile_info",
    "extract_author_from_pdf_metadata",
    "extract_author_from_html_metadata",
    "extract_author_from_text",
    "extract_title_from_pdf_metadata",
    "extract_title_from_html_metadata",
    "extract_title_from_text",
    "extract_language_from_pdf_metadata",
    "extract_language_from_html_metadata",
    "extract_language_from_text",
    "detect_document_profile",
]

"""Language extraction and resolution helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


def _normalize_language_code(lang: str) -> str:
    """
    Normalize language codes to base ISO 639-1 format.

    Examples:
        "en-US", "en_US", "en" → "en"
        "fr-FR", "fr_FR" → "fr"
        "en_US" → "en"

    Args:
        lang: Language code in any format

    Returns:
        Normalized 2-letter language code (lowercase)
    """
    if not lang or not isinstance(lang, str):
        return ""

    lang = lang.strip().lower()

    # Split on common delimiters
    for delimiter in ['-', '_']:
        if delimiter in lang:
            lang = lang.split(delimiter)[0]

    # Return first 2 letters if valid
    if len(lang) >= 2:
        return lang[:2]

    return lang


def extract_language_from_pdf_metadata(pdf_path: Path) -> Optional[str]:
    """
    Extract language from PDF metadata fields.

    Priority order:
    1. XMP dc:language
    2. /Lang field (doc.language property)
    3. xmp:Language

    Args:
        pdf_path: Path to PDF file

    Returns:
        Normalized language code (e.g., "en") or None
    """
    import fitz  # PyMuPDF
    import xml.etree.ElementTree as ET

    try:
        with fitz.open(pdf_path) as doc:
            # First, try the direct language property (reads /Lang)
            if hasattr(doc, 'language') and doc.language:
                lang = doc.language.strip()
                if lang:
                    return _normalize_language_code(lang)

            # Try XMP metadata
            xmp_metadata = doc.get_xml_metadata()
            if xmp_metadata:
                try:
                    root = ET.fromstring(xmp_metadata)

                    namespaces = {
                        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                        'dc': 'http://purl.org/dc/elements/1.1/',
                        'xmp': 'http://ns.adobe.com/xap/1.0/'
                    }

                    # Look for dc:language
                    for lang_elem in root.findall('.//dc:language', namespaces):
                        # Can contain rdf:Alt with rdf:li items
                        for li in lang_elem.findall('.//rdf:li', namespaces):
                            if li.text and li.text.strip():
                                return _normalize_language_code(li.text.strip())

                        # Or direct text
                        if lang_elem.text and lang_elem.text.strip():
                            return _normalize_language_code(lang_elem.text.strip())

                    # Look for xmp:Language
                    for lang_elem in root.findall('.//xmp:Language', namespaces):
                        if lang_elem.text and lang_elem.text.strip():
                            return _normalize_language_code(lang_elem.text.strip())

                except Exception as e:
                    pass  # Fall through to standard metadata

            # Fallback to standard metadata dict
            metadata = doc.metadata or {}
            for key in ['language', 'Language', 'lang', 'Lang']:
                lang = metadata.get(key)
                if lang and lang.strip():
                    return _normalize_language_code(lang.strip())

            return None

    except Exception as e:
        return None


def extract_language_from_html_metadata(html_content: str) -> Optional[str]:
    """
    Extract language from HTML metadata fields.

    Priority order:
    1. <html lang="en">
    2. <meta http-equiv="content-language" content="en">
    3. <meta name="language" content="en">
    4. JSON-LD "inLanguage": "en"
    5. <meta property="og:locale" content="en_US"> → normalize to "en"

    Args:
        html_content: Raw HTML string

    Returns:
        Normalized language code (e.g., "en") or None
    """
    try:
        from bs4 import BeautifulSoup
        import json

        soup = BeautifulSoup(html_content, 'html.parser')

        # =============================
        # 1. <html lang="en">
        # =============================
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            lang = html_tag['lang'].strip()
            if lang:
                return _normalize_language_code(lang)

        # =============================
        # 2. <meta http-equiv="content-language">
        # =============================
        content_lang = soup.find('meta', attrs={'http-equiv': lambda x: x and x.lower() == 'content-language'})
        if content_lang and content_lang.get('content'):
            lang = content_lang['content'].strip()
            if lang:
                return _normalize_language_code(lang)

        # =============================
        # 3. <meta name="language">
        # =============================
        meta_lang = soup.find('meta', attrs={'name': 'language'})
        if meta_lang and meta_lang.get('content'):
            lang = meta_lang['content'].strip()
            if lang:
                return _normalize_language_code(lang)

        # =============================
        # 4. JSON-LD "inLanguage"
        # =============================
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if isinstance(item, dict):
                        in_language = item.get('inLanguage')
                        if in_language:
                            if isinstance(in_language, str):
                                return _normalize_language_code(in_language)
                            elif isinstance(in_language, dict):
                                # Sometimes it's {"@value": "en"}
                                value = in_language.get('@value')
                                if value:
                                    return _normalize_language_code(value)

            except (json.JSONDecodeError, AttributeError):
                continue

        # =============================
        # 5. OpenGraph locale
        # =============================
        og_locale = soup.find('meta', attrs={'property': 'og:locale'})
        if og_locale and og_locale.get('content'):
            locale = og_locale['content'].strip()
            if locale:
                return _normalize_language_code(locale)

        return None

    except Exception:
        return None


def extract_language_from_text(df_lines: pd.DataFrame) -> Optional[str]:
    """
    Detect language from document text using langdetect.

    Uses first 10,000 characters, filtering out table content.

    Args:
        df_lines: DataFrame with 'text' and optionally 'block_type' columns

    Returns:
        Detected language code (e.g., "en") or None
    """
    try:
        if df_lines.empty or 'text' not in df_lines.columns:
            return None

        # Filter out table content if block_type column exists
        if 'block_type' in df_lines.columns:
            text_lines = df_lines[df_lines['block_type'] != 'table'].copy()
        else:
            text_lines = df_lines.copy()

        if text_lines.empty:
            return None

        # Concatenate text up to 10k characters
        all_text = ' '.join(text_lines['text'].dropna().astype(str))
        text_sample = all_text[:10000]

        if not text_sample.strip() or len(text_sample.strip()) < 50:
            return None

        # Try to import and use langdetect
        try:
            from langdetect import detect, LangDetectException

            # Detect language
            detected = detect(text_sample)

            # Normalize to 2-letter code
            return _normalize_language_code(detected)

        except ImportError:
            # langdetect not installed
            return None
        except Exception as e:
            # Detection failed (not enough text, etc.)
            return None

    except Exception as e:
        return None


def add_language_info(
    doc_meta: Dict[str, Any],
    pdf_path: Optional[Path] = None,
    html_content: Optional[str] = None,
    df_lines: Optional[pd.DataFrame] = None,
) -> None:
    """
    Extract and resolve language information.

    Populates:
    - language_meta: Language from metadata (or None)
    - language_text: Language from text detection (or None)
    - language: Final resolved language
    - language_confidence: "high" / "medium" / "low"
    - language_source: "meta" / "text" / None

    Resolution logic:
    - If meta and text exist and match (base code): language=meta, confidence=high, source=meta
    - If meta and text exist and differ: language=text, confidence=medium, source=text
    - If only text exists: language=text, confidence=medium, source=text
    - If only meta exists: language=meta, confidence=medium, source=meta
    - If neither exists: language=unknown, confidence=low, source=None

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        pdf_path: Path to PDF file (for PDF content_type)
        html_content: HTML content string (for HTML content_type)
        df_lines: DataFrame with text lines (for text-based detection)

    Modifies:
        doc_meta dictionary in-place
    """
    content_type = doc_meta.get("content_type", "").upper()

    # =============================
    # Extract from metadata
    # =============================
    language_meta = None
    if content_type == "PDF" and pdf_path:
        language_meta = extract_language_from_pdf_metadata(pdf_path)
    elif content_type == "HTML" and html_content:
        language_meta = extract_language_from_html_metadata(html_content)

    doc_meta["language_meta"] = language_meta

    # =============================
    # Extract from text
    # =============================
    language_text = None
    if df_lines is not None:
        language_text = extract_language_from_text(df_lines)

    doc_meta["language_text"] = language_text

    # =============================
    # Resolution logic
    # =============================

    # Normalize both for comparison
    meta_base = _normalize_language_code(language_meta) if language_meta else None
    text_base = _normalize_language_code(language_text) if language_text else None

    if meta_base and text_base:
        if meta_base == text_base:
            # Both exist and match
            doc_meta["language"] = meta_base
            doc_meta["language_confidence"] = "high"
            doc_meta["language_source"] = "meta"
        else:
            # Both exist but differ - trust text detection
            doc_meta["language"] = text_base
            doc_meta["language_confidence"] = "medium"
            doc_meta["language_source"] = "text"

    elif text_base and not meta_base:
        # Only text exists
        doc_meta["language"] = text_base
        doc_meta["language_confidence"] = "medium"
        doc_meta["language_source"] = "text"

    elif meta_base and not text_base:
        # Only meta exists
        doc_meta["language"] = meta_base
        doc_meta["language_confidence"] = "medium"
        doc_meta["language_source"] = "meta"

    else:
        # Neither exists
        doc_meta["language"] = "unknown"
        doc_meta["language_confidence"] = "low"
        doc_meta["language_source"] = None

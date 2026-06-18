"""Title extraction helpers — PDF metadata, HTML metadata, and text-based detection."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


def extract_title_from_pdf_metadata(pdf_path: Path) -> Optional[str]:
    """
    Extract title from PDF metadata fields.

    Priority order:
    1. XMP dc:title (Dublin Core title from XML metadata)
    2. PDF Info /Title (standard PDF title field)

    Args:
        pdf_path: Path to PDF file

    Returns:
        Title string or None if not found
    """
    import xml.etree.ElementTree as ET
    import pypdfium2 as pdfium
    from ._pdf_xmp import read_xmp

    try:
        # 1. XMP dc:title
        xmp = read_xmp(pdf_path)
        if xmp:
            root = ET.fromstring(xmp)
            ns = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'dc': 'http://purl.org/dc/elements/1.1/',
            }
            for elem in root.findall('.//dc:title', ns):
                for li in elem.findall('.//rdf:li', ns):
                    if li.text and li.text.strip():
                        return li.text.strip()
                if elem.text and elem.text.strip():
                    return elem.text.strip()

        # 2. PDF info dict /Title
        with pdfium.PdfDocument(pdf_path) as doc:
            title = doc.get_metadata_value("Title")
            if title and title.strip():
                return title.strip()

        return None

    except Exception:
        return None


def extract_title_from_html_metadata(html_content: str) -> Optional[str]:
    """
    Extract title from HTML metadata fields.

    Priority order:
    1. <meta property="og:title">
    2. JSON-LD headline
    3. <title> tag
    4. First visible <h1>
    5. Layout fallback (largest visible heading near top)

    Args:
        html_content: Raw HTML string

    Returns:
        Title string or None if not found
    """
    try:
        from bs4 import BeautifulSoup
        import json

        soup = BeautifulSoup(html_content, 'html.parser')

        # =============================
        # 1. Open Graph title
        # =============================
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title and og_title.get('content'):
            content = og_title['content'].strip()
            if content:
                return content

        # =============================
        # 2. JSON-LD headline
        # =============================
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if isinstance(item, dict):
                        headline = item.get('headline')
                        if headline and isinstance(headline, str) and headline.strip():
                            return headline.strip()

                        # Also check name field as fallback
                        name = item.get('name')
                        if name and isinstance(name, str) and name.strip():
                            return name.strip()

            except (json.JSONDecodeError, AttributeError):
                continue

        # =============================
        # 3. <title> tag
        # =============================
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title_text = title_tag.string.strip()
            if title_text:
                # Clean up common suffixes (e.g., "Title | Site Name")
                for sep in [' | ', ' - ', ' – ', ' — ']:
                    if sep in title_text:
                        # Take the first part (before separator)
                        parts = title_text.split(sep)
                        if parts[0].strip():
                            return parts[0].strip()
                return title_text

        # =============================
        # 4. First visible <h1>
        # =============================
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            text = h1.get_text(strip=True)
            if text and len(text) > 5:  # Reasonable title length
                return text

        # =============================
        # 5. Layout fallback - largest heading near top
        # =============================
        headings = soup.find_all(['h1', 'h2', 'h3'])[:10]  # First 10 headings
        for heading in headings:
            text = heading.get_text(strip=True)
            if text and len(text) > 5:
                return text

        return None

    except Exception:
        return None


def extract_title_from_text(df_lines: pd.DataFrame) -> Optional[str]:
    """
    Extract title from document text by finding largest font size line.

    Algorithm:
    1. Look at page_number == 1
    2. Find line with largest font_size
    3. If multiple lines with same max font size, take the first
    4. If no words on page 1, expand to pages 2 and 3

    Args:
        df_lines: DataFrame with 'page_number', 'font_size', 'text' columns

    Returns:
        Title string or None if not found
    """
    if df_lines.empty:
        return None

    required_cols = ['page_number', 'font_size', 'text']
    if not all(col in df_lines.columns for col in required_cols):
        return None

    # Try pages 1, 2, 3 in order
    for page_num in [1, 2, 3]:
        page_lines = df_lines[df_lines['page_number'] == page_num].copy()

        if page_lines.empty:
            continue

        # Filter out empty/short text
        page_lines = page_lines[
            page_lines['text'].notna() &
            (page_lines['text'].str.len() > 5)
        ]

        if page_lines.empty:
            continue

        # Find maximum font size
        max_font_size = page_lines['font_size'].max()

        if pd.isna(max_font_size):
            continue

        # Get all lines with max font size
        max_font_lines = page_lines[page_lines['font_size'] == max_font_size]

        if not max_font_lines.empty:
            # Take the first one
            title = max_font_lines.iloc[0]['text']
            if isinstance(title, str) and title.strip():
                # Clean up the title
                title = title.strip()
                # Remove common artifacts
                if len(title) > 5 and not title.lower().startswith(('page ', 'chapter ')):
                    return title

    return None


def add_title_info(
    doc_meta: Dict[str, Any],
    pdf_path: Optional[Path] = None,
    html_content: Optional[str] = None,
    df_lines: Optional[pd.DataFrame] = None,
) -> None:
    """
    Extract and add title information to document metadata.

    Populates:
    - title_meta: Title from PDF/HTML metadata (or None)
    - title_text: Title detected from document text (largest font) (or None)

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        pdf_path: Path to PDF file (for PDF content_type)
        html_content: HTML content string (for HTML content_type)
        df_lines: DataFrame with text lines (for text-based detection)

    Modifies:
        doc_meta dictionary in-place
    """
    content_type = doc_meta.get("content_type", "").upper()

    # Extract from metadata
    title_meta = None
    if content_type == "PDF" and pdf_path:
        title_meta = extract_title_from_pdf_metadata(pdf_path)
    elif content_type == "HTML" and html_content:
        title_meta = extract_title_from_html_metadata(html_content)

    doc_meta["title_meta"] = title_meta

    # Extract from text
    title_text = None
    if df_lines is not None:
        title_text = extract_title_from_text(df_lines)

    doc_meta["title_text"] = title_text

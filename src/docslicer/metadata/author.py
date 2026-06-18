"""Author extraction helpers — PDF metadata, HTML metadata, and text-based detection."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


_FAKE_AUTHORS = {
    # Placeholders (all lowercase for case-insensitive matching)
    "", "unknown", "n/a", "na", "none", "null", "anonymous",
    "author", "user", "admin", "content", "version",
    "system", "default", "root", "test", "demo", "sample",
    "posted by", "written by", "contributor", "guest",
    "rss", "contact", "info", "webmaster", "reporter",
    # PDF / software artifacts
    "microsoft", "office", "word", "powerpoint", "excel",
    "adobe", "acrobat", "indesign", "quartz", "pdfcontext",
    "preview", "scan", "hp", "canon", "epson", "pdf",
    "abbyy", "finereader", "scanner",
    # Filing / financial system generators
    "workiva", "platform", "xbrl", "edgar",
    # Web / CMS junk
    "wordpress", "drupal", "joomla", "latex", "package",
    "sitecore", "contentful", "squarespace",
}


def _is_fake_author(author: str) -> bool:
    """
    Check if author name is likely a false positive.

    Uses substring matching against known fake author patterns.

    Args:
        author: Author name to check

    Returns:
        True if likely fake
    """
    if not author:
        return True

    author_lower = author.lower().strip()

    # Check for exact matches
    if author_lower in _FAKE_AUTHORS:
        return True

    # Check for substring matches (fake authors may appear in longer strings)
    for fake in _FAKE_AUTHORS:
        if fake and fake in author_lower:
            return True

    # Additional heuristics
    if len(author_lower) < 3:  # Too short
        return True

    if author_lower.startswith(("v", "ver", "version", "rev")):  # Version strings
        return True

    if any(c.isdigit() for c in author_lower):  # Contains numbers
        # Exception: names like "John Smith 2" might be valid
        # But "v1.2.3" or "user123" are not
        if re.match(r'^[a-z\s]+\d+$', author_lower):  # Ends with number
            pass  # Could be valid
        elif re.search(r'\d+\.\d+', author_lower):  # Version pattern
            return True

    return False


def _is_title_case(text: str) -> bool:
    """
    Check if text is in title case (each word starts with capital).

    Args:
        text: Text to check

    Returns:
        True if text is title case (not all caps, not all lowercase)
    """
    # Extract words (skip punctuation, numbers)
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    if not words:
        return False

    # Check if all words are uppercase (ALL CAPS) - reject
    all_uppercase = all(w.isupper() for w in words)
    if all_uppercase:
        return False

    # Check if all words are lowercase - reject
    all_lowercase = all(w.islower() for w in words)
    if all_lowercase:
        return False

    # Check if all significant words start with uppercase
    # Allow small words (a, the, of, etc.) to be lowercase
    small_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but"}

    capitalized_count = 0
    significant_word_count = 0

    for i, word in enumerate(words):
        if word.lower() in small_words and i > 0:  # Small words can be lowercase (except first)
            continue

        significant_word_count += 1
        if word[0].isupper():
            capitalized_count += 1

    # At least 70% of significant words should be capitalized
    if significant_word_count == 0:
        return False

    return (capitalized_count / significant_word_count) >= 0.7


def _load_common_author_names() -> set[str]:
    """
    Load common author names from CSV file.

    Returns:
        Set of lowercase author names
    """
    from importlib.resources import files

    try:
        with (files("docslicer") / "config" / "common_author_names.csv").open("rb") as f:
            df = pd.read_csv(f)
        if "name" in df.columns:
            # Names are already lowercase in the CSV
            return set(df["name"].dropna().astype(str).str.strip())
        return set()
    except Exception:
        return set()


def extract_author_from_pdf_metadata(pdf_path: Path) -> list[str]:
    """
    Extract author from PDF metadata fields (dc:creator, /Author, pdf:Author).

    Priority order:
    1. dc:creator (Dublin Core) - checks both XMP and standard metadata
    2. /Author (standard PDF field)
    3. pdf:Author (alternative format)

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of author names (empty if none found or all are fake)
    """
    import xml.etree.ElementTree as ET
    import pypdfium2 as pdfium
    from ._pdf_xmp import read_xmp

    def _split_authors(value: str) -> list[str]:
        for delimiter in [";", " and ", " & "]:
            if delimiter in value:
                return [p.strip() for p in value.split(delimiter)]
        return [value.strip()]

    try:
        # 1. XMP dc:creator
        xmp = read_xmp(pdf_path)
        if xmp:
            root = ET.fromstring(xmp)
            ns = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'dc': 'http://purl.org/dc/elements/1.1/',
            }
            creators = []
            for elem in root.findall('.//dc:creator', ns):
                for li in elem.findall('.//rdf:li', ns):
                    if li.text:
                        creators.append(li.text.strip())
                if elem.text and elem.text.strip():
                    creators.append(elem.text.strip())
            valid = [a for a in creators if a and not _is_fake_author(a)]
            if valid:
                return valid[:5]

        # 2. PDF info dict /Author then /Creator
        with pdfium.PdfDocument(pdf_path) as doc:
            for key in ("Author", "Creator"):
                field_value = doc.get_metadata_value(key)
                if not field_value:
                    continue
                authors = [a for a in _split_authors(field_value) if a and not _is_fake_author(a)]
                if authors:
                    return authors[:5]

        return []

    except Exception:
        return []


def extract_author_from_html_metadata(html_content: str) -> list[str]:
    """
    Extract author from HTML metadata fields.

    Checks (in order):
    1. JSON-LD schema: { "@type": "Person", "name": "..." }
    2. <meta name="author" content="...">
    3. <meta property="article:author" content="...">
    4. Elements with itemprop="author" or rel="author"
    5. DOM elements whose class/id contains author-related keywords

    Args:
        html_content: Raw HTML string

    Returns:
        List of author names (empty if none found or all are fake)
    """
    try:
        from bs4 import BeautifulSoup
        import json

        soup = BeautifulSoup(html_content, 'html.parser')
        authors = []

        # =============================
        # 1. JSON-LD Schema
        # =============================
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)

                # Handle both single objects and arrays
                items = data if isinstance(data, list) else [data]

                for item in items:
                    # Look for Person type
                    if isinstance(item, dict):
                        item_type = item.get('@type', '')

                        # Direct Person object
                        if item_type == 'Person':
                            name = item.get('name')
                            if name and not _is_fake_author(name):
                                authors.append(name.strip())

                        # Author field with Person object
                        author_field = item.get('author')
                        if author_field:
                            if isinstance(author_field, dict):
                                if author_field.get('@type') == 'Person':
                                    name = author_field.get('name')
                                    if name and not _is_fake_author(name):
                                        authors.append(name.strip())
                            elif isinstance(author_field, list):
                                for auth in author_field:
                                    if isinstance(auth, dict) and auth.get('@type') == 'Person':
                                        name = auth.get('name')
                                        if name and not _is_fake_author(name):
                                            authors.append(name.strip())
                            elif isinstance(author_field, str):
                                if not _is_fake_author(author_field):
                                    authors.append(author_field.strip())

                if authors:
                    return authors[:5]

            except (json.JSONDecodeError, AttributeError):
                continue

        # =============================
        # 2. <meta name="author">
        # =============================
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            content = meta_author['content'].strip()
            if content and not _is_fake_author(content):
                # Split on common delimiters
                for delimiter in [',', ';', ' and ', ' & ']:
                    if delimiter in content:
                        parts = [p.strip() for p in content.split(delimiter)]
                        valid = [p for p in parts if p and not _is_fake_author(p)]
                        if valid:
                            return valid[:5]
                return [content][:5]

        # =============================
        # 3. <meta property="article:author">
        # =============================
        article_author = soup.find('meta', attrs={'property': 'article:author'})
        if article_author and article_author.get('content'):
            content = article_author['content'].strip()
            if content and not _is_fake_author(content):
                return [content][:5]

        # =============================
        # 4. Elements with itemprop="author" or rel="author"
        # =============================
        # itemprop="author"
        itemprop_authors = soup.find_all(attrs={'itemprop': 'author'})
        for elem in itemprop_authors:
            # Check for name in itemprop
            name_elem = elem.find(attrs={'itemprop': 'name'})
            if name_elem:
                text = name_elem.get_text(strip=True)
                if text and not _is_fake_author(text):
                    authors.append(text)
            else:
                text = elem.get_text(strip=True)
                if text and not _is_fake_author(text):
                    authors.append(text)

        if authors:
            return authors[:5]

        # rel="author"
        rel_authors = soup.find_all(attrs={'rel': 'author'})
        for elem in rel_authors:
            text = elem.get_text(strip=True)
            if text and not _is_fake_author(text):
                authors.append(text)

        if authors:
            return authors[:5]

        # =============================
        # 5. DOM elements with author-related class/id
        # =============================
        author_keywords = [
            'author', 'byline', 'contributor', 'posted-by', 'posted_by',
            'writer', 'article-author', 'article_author', 'by-line', 'by_line',
            'post-author', 'post_author', 'entry-author', 'entry_author'
        ]

        for keyword in author_keywords:
            # Check class
            elements = soup.find_all(class_=re.compile(keyword, re.IGNORECASE))
            for elem in elements:
                text = elem.get_text(strip=True)
                # Only accept if reasonable length and title case
                if text and 5 < len(text) < 100 and _is_title_case(text):
                    if not _is_fake_author(text):
                        authors.append(text)
                        if len(authors) >= 5:
                            return authors

            # Check id
            elements = soup.find_all(id=re.compile(keyword, re.IGNORECASE))
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and 5 < len(text) < 100 and _is_title_case(text):
                    if not _is_fake_author(text):
                        authors.append(text)
                        if len(authors) >= 5:
                            return authors

        return authors[:5] if authors else []

    except Exception:
        return []


def extract_author_from_text(
    df_lines: pd.DataFrame,
    common_names: Optional[set[str]] = None
) -> list[str]:
    """
    Extract author names from document text by finding common names in title case.

    Algorithm:
    1. Load common author names (lowercase) from config/common_author_names.csv
    2. For each line, check if it contains exact word matches to common names
    3. If match found, verify the line is in title case
    4. If title case, extract as author name

    Args:
        df_lines: DataFrame with 'text' column
        common_names: Optional pre-loaded set of common names (lowercase)

    Returns:
        List of author names found in text (max 5)
    """
    if df_lines.empty or "text" not in df_lines.columns:
        return []

    # Load common names if not provided
    if common_names is None:
        common_names = _load_common_author_names()

    if not common_names:
        return []

    authors = []

    # Process first 20 lines only (authors typically at start)
    lines_to_check = df_lines.head(20)

    for idx, row in lines_to_check.iterrows():
        text = str(row.get("text", "")).strip()

        if not text or len(text) > 100:  # Skip empty or very long lines
            continue

        # Check if line is in title case
        if not _is_title_case(text):
            continue

        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        words_lower = [w.lower() for w in words]

        # Check if any word is a common name
        matched_names = [w for w in words if w.lower() in common_names]

        if matched_names:
            # If we found a match and line is title case, it's likely an author
            authors.append(text)

            if len(authors) >= 5:  # Max 5 authors
                break

    return authors


def add_author_info(
    doc_meta: Dict[str, Any],
    pdf_path: Optional[Path] = None,
    html_content: Optional[str] = None,
    df_lines: Optional[pd.DataFrame] = None,
) -> None:
    """
    Extract and add author information to document metadata.

    Populates:
    - author_meta: Authors from PDF/HTML metadata
    - author_text: Authors detected from document text

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
    author_meta = []
    if content_type == "PDF" and pdf_path:
        author_meta = extract_author_from_pdf_metadata(pdf_path)
    elif content_type == "HTML" and html_content:
        author_meta = extract_author_from_html_metadata(html_content)

    doc_meta["author_meta"] = author_meta if author_meta else None

    # Extract from text
    author_text = []
    if df_lines is not None:
        author_text = extract_author_from_text(df_lines)

    doc_meta["author_text"] = author_text if author_text else None

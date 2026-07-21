"""Extract document metadata from an HTML document's <head>.

Mirrors the pdf/docx/pptx native extractors: same output keys, parsed once off
the rendered HTML so the metadata/ heuristics never re-parse. Field sourcing
follows what real pages actually carry (see test_jelle/html_metadata_dump.csv);
each field is a short priority chain from most- to least-authoritative signal:

    title_meta        og:title -> twitter:title -> JSON-LD headline/name -> <title> -> <h1>/heading
    author_meta       JSON-LD author -> <meta name=author> -> article:author -> itemprop/rel=author -> byline class/id
    language_meta     <html lang> -> og:locale -> content-language -> JSON-LD inLanguage
    created           article:published_time -> JSON-LD datePublished -> <meta name=date>
    modified          article:modified_time -> JSON-LD dateModified -> og:updated_time
    last_modified_by  None            (HTML has no "last editor" concept — OOXML only)
    application       <meta name=generator> -> <meta name=application-name>  (raw tool)
    generator         normalized vendor label from ``application`` (or None)

SEC EDGAR filings ship an empty ``<head>`` — every field comes back None there,
and the text-based fallbacks in metadata/ are the only source.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from ..metadata.generator import classify_generator

# ----------------------------
# Config
# ----------------------------

# Cap what we DOM-parse (in characters, not bytes) so a 400-page SEC filing is
# never parsed in full. Bounds two slices: the head window when a page has no
# </head>, and the body window for the <h1>/byline fallbacks. Metadata and the
# first heading/byline both sit near the top, well within this.
_MAX_PARSE_CHARS = 50_000

# JSON-LD can live in <head> or late in <body>, so pull it from the whole doc.
_JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
# Trailing " - Site" / " – Site" / " — Site" / " | Site" suffix on a <title>.
_TITLE_SUFFIX_RE = re.compile(r"\s+[-–—|]\s+[^-–—|]+$")

# class/id substrings that mark an author byline in the body DOM.
_AUTHOR_KEYWORDS_RE = re.compile(
    r"author|byline|contributor|posted[-_]?by|writer|by[-_]?line|"
    r"post[-_]?author|entry[-_]?author",
    re.IGNORECASE,
)


# ----------------------------
# Helpers
# ----------------------------

def _clean(value: Any) -> str | None:
    """Coerce to a stripped, whitespace-collapsed str, or None if empty."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _first(*values: str | None) -> str | None:
    """First non-empty cleaned value in the priority chain."""
    for v in values:
        cleaned = _clean(v)
        if cleaned:
            return cleaned
    return None


def _iter_json_ld(html: str):
    """Yield each parsed JSON-LD object (flattening @graph and top-level lists)."""
    for match in _JSON_LD_RE.findall(html):
        raw = match.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                for node in graph:
                    if isinstance(node, dict):
                        yield node
            else:
                yield item


def _json_ld_name(value: Any) -> list[str]:
    """Pull display name(s) out of a JSON-LD author/creator value."""
    names: list[str] = []
    if isinstance(value, str):
        names.append(value)
    elif isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            names.append(name)
    elif isinstance(value, list):
        for v in value:
            names.extend(_json_ld_name(v))
    return [n for n in (_clean(n) for n in names) if n]


def _meta(soup: BeautifulSoup, **attrs: str) -> str | None:
    """Content of the first <meta> matching the given identifying attribute."""
    tag = soup.find("meta", attrs=attrs)
    if tag and tag.get("content"):
        return _clean(tag["content"])
    return None


def _http_equiv(soup: BeautifulSoup, name: str) -> str | None:
    """Content of the first <meta http-equiv=name> (case-insensitive)."""
    tag = soup.find("meta", attrs={"http-equiv": re.compile(f"^{name}$", re.I)})
    if tag and tag.get("content"):
        return _clean(tag["content"])
    return None


def _split_authors(raw: str | None) -> list[str]:
    """Split a comma-separated author string, dropping URL-like values."""
    if not raw or raw.lower().startswith(("http://", "https://")):
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p and not p.lower().startswith(("http://", "https://"))]


def _head_slice(html: str) -> str:
    """Up to </head> (meta/title/lang live there); else a bounded top window.

    Without the cap, a page missing </head> would parse in full — the exact
    400-page-SEC blow-up we are avoiding. The cap still covers <html lang>,
    <title> and the first heading.
    """
    end = html.lower().find("</head>")
    return html[: end + 7] if end != -1 else html[:_MAX_PARSE_CHARS]


def _body_soup(html: str) -> BeautifulSoup:
    """Parse a bounded top-of-document window for the DOM-based fallbacks."""
    return BeautifulSoup(html[:_MAX_PARSE_CHARS], "html.parser")


def _title_from_dom(soup: BeautifulSoup) -> str | None:
    """Title fallback: first <h1>, else the first h1/h2/h3 near the top.

    Pure cascade — the first non-empty heading text wins, verbatim. Any vetting
    of that text is a downstream step's job, not this extractor's.
    """
    for h1 in soup.find_all("h1"):
        text = _clean(h1.get_text(strip=True))
        if text:
            return text
    for heading in soup.find_all(["h1", "h2", "h3"])[:10]:
        text = _clean(heading.get_text(strip=True))
        if text:
            return text
    return None


def _authors_from_dom(soup: BeautifulSoup) -> list[str]:
    """Author fallback: itemprop=author, then rel=author, then a class/id byline.

    Pure cascade — takes the first thing it sees at each tier and returns it
    verbatim, with no author validation (a downstream step vets the result).
    The class/id substring match is only how the byline element is located.
    """
    # itemprop="author" — prefer a nested itemprop="name", else the element text.
    names: list[str] = []
    for elem in soup.find_all(attrs={"itemprop": "author"}):
        name_elem = elem.find(attrs={"itemprop": "name"})
        text = _clean((name_elem or elem).get_text(strip=True))
        if text:
            names.append(text)
    if names:
        return names

    # rel="author"
    for elem in soup.find_all(attrs={"rel": "author"}):
        text = _clean(elem.get_text(strip=True))
        if text:
            names.append(text)
    if names:
        return names

    # class/id byline (byline, contributor, post-author, …) — first hit wins.
    for elem in [*soup.find_all(class_=_AUTHOR_KEYWORDS_RE), *soup.find_all(id=_AUTHOR_KEYWORDS_RE)]:
        text = _clean(elem.get_text(strip=True))
        if text:
            return [text]

    return []


# ----------------------------
# Public API
# ----------------------------

def extract_native_metadata(html: str | None) -> dict[str, Any]:
    """Read an HTML page's <head> metadata into the shared native-metadata shape.

    Args:
        html: The rendered/served HTML string (may be None or empty).

    Returns:
        dict with keys: title_meta, author_meta, language_meta, created, modified,
        last_modified_by, application, generator.
    """
    empty: dict[str, Any] = {
        "title_meta": None,
        "author_meta": None,
        "language_meta": None,
        "created": None,
        "modified": None,
        "last_modified_by": None,
        "application": None,
        "generator": None,
    }
    if not html:
        return empty

    soup = BeautifulSoup(_head_slice(html), "html.parser")

    # The <h1>/byline fallbacks need the body; parse it at most once, and only
    # if a head-metadata signal actually comes up empty.
    _body_cache: list[BeautifulSoup] = []
    def body() -> BeautifulSoup:
        if not _body_cache:
            _body_cache.append(_body_soup(html))
        return _body_cache[0]

    # Gather the first useful JSON-LD article/webpage node once.
    ld_headline = ld_name = ld_published = ld_modified = ld_lang = None
    ld_authors: list[str] = []
    for node in _iter_json_ld(html):
        ld_headline = ld_headline or _clean(node.get("headline"))
        ld_name = ld_name or _clean(node.get("name"))
        ld_published = ld_published or _clean(node.get("datePublished"))
        ld_modified = ld_modified or _clean(node.get("dateModified"))
        ld_lang = ld_lang or _clean(node.get("inLanguage"))
        if not ld_authors:
            ld_authors = _json_ld_name(node.get("author") or node.get("creator"))

    # --- title: prefer the pre-cleaned social/LD titles; strip site suffix off <title> ---
    title_tag = soup.title.string if soup.title else None
    title_bare = _clean(title_tag)
    if title_bare:
        stripped = _TITLE_SUFFIX_RE.sub("", title_bare).strip()
        title_bare = stripped or title_bare
    title_meta = _first(
        _meta(soup, property="og:title"),
        _meta(soup, name="twitter:title"),
        ld_headline,
        ld_name,
        title_bare,
    )
    if not title_meta:                               # body fallback: first <h1>/heading
        title_meta = _title_from_dom(body())

    # --- author: JSON-LD is cleanest; article:author is often a URL, so filter ---
    author_meta = ld_authors
    if not author_meta:
        author_meta = _split_authors(_meta(soup, name="author"))
    if not author_meta:
        author_meta = _split_authors(_meta(soup, property="article:author"))
    if not author_meta:                              # body fallback: itemprop/rel/byline
        author_meta = _authors_from_dom(body())
    if author_meta:                                  # order-preserving dedupe
        author_meta = list(dict.fromkeys(author_meta))
    author_meta = author_meta or None

    # --- language: raw (the metadata/ language step normalizes) ---
    html_tag = soup.find("html")
    html_lang = _clean(html_tag.get("lang")) if html_tag else None
    language = _first(
        html_lang,
        _meta(soup, property="og:locale"),
        _http_equiv(soup, "content-language"),
        _meta(soup, name="language"),
        ld_lang,
    )

    # --- dates: mostly ISO-8601 already; passed through as-is ---
    created = _first(
        _meta(soup, property="article:published_time"),
        ld_published,
        _meta(soup, name="date"),
        _meta(soup, name="pubdate"),
    )
    modified = _first(
        _meta(soup, property="article:modified_time"),
        ld_modified,
        _meta(soup, property="og:updated_time"),
    )

    # --- generator: the authoring tool/CMS, raw + normalized vendor label.
    # og:site_name is the *publisher* brand (CNBC, Federal Register), a different
    # concept, so it is deliberately not folded in here. ---
    application = _first(
        _meta(soup, name="generator"),
        _meta(soup, name="application-name"),
    )
    generator = classify_generator(application=application)

    return {
        "title_meta": title_meta,
        "author_meta": author_meta,
        "language_meta": language,
        "created": created,
        "modified": modified,
        "last_modified_by": None,   # no HTML equivalent (OOXML-only field)
        "application": application,
        "generator": generator,
    }


if __name__ == "__main__":   # quick manual check: python -m docslicer.html.native_metadata file.html
    import sys
    for path in sys.argv[1:]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            print(path, "->", extract_native_metadata(fh.read()))

# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# step_01_static_box_extractor.py
"""
Fallback HTML box extractor using BeautifulSoup — no Playwright required.

Produces the same box schema as extract_boxes_with_playwright, with these
known limitations:

- Coordinates (x_left, y_top, x_right, y_bottom, width, height) are all 0.0.
  Layout-dependent features in downstream steps (y_top-based ordering, region
  detection, coordinate-based merging) will degrade or produce no-ops.

- Style resolution is inline-only. CSS class rules and external stylesheets
  are NOT applied. Bold/italic/font-size are detected from:
    * Inline `style` attributes on the element and its ancestors
    * Semantic tags: <b>, <strong>, <i>, <em>, <u>, <h1>-<h6>

- Block elements are split into runs at inline-context boundaries (mirroring the
  Playwright extractor's INLINE_SPLIT_TAGS walk), so bold_ratio / italic_ratio are
  per-run rather than per-block. Styles are resolved from the text node's own
  parent upward, so a `<span style="font-weight:700">` inside a plain `<div>` is
  detected as bold.

Works well for documents that use inline styles — SEC filings, Word-exported
HTML, legal documents. Less useful for modern CSS-class-heavy pages.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import warnings

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, XMLParsedAsHTMLWarning

# ---------------------------------------------------------------------------
# Tag sets (mirrors JS STRUCTURE_TAGS)
# ---------------------------------------------------------------------------

STRUCTURE_TAGS: frozenset[str] = frozenset({
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "td", "th", "blockquote", "pre", "article",
    "section", "header", "footer", "aside", "main",
    "figcaption", "caption", "dt", "dd", "address",
})

# Default browser font sizes for heading tags (px)
_HEADING_FONT_PX: Dict[str, float] = {
    "h1": 32.0, "h2": 24.0, "h3": 18.72,
    "h4": 16.0, "h5": 13.28, "h6": 10.72,
}

_NAMED_SIZES: Dict[str, float] = {
    "xx-small": 9.0, "x-small": 10.0, "small": 13.0, "medium": 16.0,
    "large": 18.0, "x-large": 24.0, "xx-large": 32.0,
    "smaller": 13.0, "larger": 18.0,
}

_DEFAULT_FONT_PX = 16.0
_DEFAULT_VIEWPORT_WIDTH = 1280

# Inline tags that open a new inline context inside a block element (mirrors the
# JS extractor's INLINE_SPLIT_TAGS). A text run is anchored on its nearest such
# ancestor; when that anchor changes, a new box starts.
#
# <span>/<font> are included deliberately: styling in SEC filings, Word exports and
# legal HTML lives on the span, so anchoring there is what makes bold/italic/font
# detection work at all. Boxes from the same block element are re-merged downstream
# by struct_tag_id (step_02), so finer splits do not fragment the output.
_INLINE_SPLIT_TAGS: frozenset[str] = frozenset({
    "strong", "b", "em", "i", "u", "mark",
    "a", "code", "kbd", "samp", "var",
    "del", "ins", "s", "strike", "small", "big",
    "span", "font", "tt", "label",
    "sup", "sub",  # split so script_type (superscript/subscript) is tagged per box
})


# ---------------------------------------------------------------------------
# CSS helpers
# ---------------------------------------------------------------------------

def _parse_style(style_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (style_str or "").split(";"):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        out[k.strip().lower()] = v.strip()
    return out


def _css_size_to_px(value: str) -> Optional[float]:
    value = value.strip().lower()
    if not value or value in ("inherit", "initial", "unset", "auto", "normal"):
        return None
    if value in _NAMED_SIZES:
        return _NAMED_SIZES[value]
    m = re.match(r"^([\d.]+)\s*(px|pt|em|rem|%)?$", value)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2) or "px"
    if unit == "px":
        return n
    # Not rounded: getComputedStyle reports 8pt as 10.6667px, and the orchestrator
    # multiplies font_size by 0.75 to get points back. Rounding here to 2 decimals
    # made 8pt round-trip to 8.0025pt and broke exact comparison with the
    # Playwright path.
    if unit == "pt":
        return n * 96.0 / 72.0
    if unit in ("em", "rem"):
        return n * _DEFAULT_FONT_PX
    if unit == "%":
        return n / 100.0 * _DEFAULT_FONT_PX
    return n


def _is_bold_weight(fw: str) -> bool:
    fw = fw.strip().lower()
    if fw in ("bold", "bolder"):
        return True
    m = re.match(r"^(\d+)$", fw)
    return bool(m) and int(m.group(1)) >= 600


# ---------------------------------------------------------------------------
# Ancestor-walk style resolution
# ---------------------------------------------------------------------------

def _resolve_style(el: Tag) -> Dict[str, Any]:
    """
    Walk el and its ancestors resolving typography from inline styles and semantic
    tag names. Call this on the element that directly wraps the text (the text
    node's parent) so the walk passes through every inline wrapper — <span>, <font>,
    <b>, … — before reaching the block, mirroring how a browser resolves
    getComputedStyle on a text node.

    Inherited properties (font-size, font-family, font-weight, font-style, color,
    text-align) take the innermost declaration: once resolved, ancestors cannot
    override it, and neither can a semantic tag. `<b><span style="font-weight:400">`
    is therefore not bold, matching CSS cascade order.

    Non-inherited decorations (underline, line-through) are OR-ed up the tree
    instead: a browser paints an ancestor's decoration over descendant text, so
    `text-decoration:none` on the child does not remove it.
    """
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    font_weight: Optional[str] = None
    bold = False
    italic = False
    italic_resolved = False   # an explicit font-style was found at some level
    underline = False
    strikethrough = False
    script_type = ""   # "superscript" | "subscript" | ""
    color: Optional[str] = None

    node: Any = el
    while node is not None and isinstance(node, Tag) and node.name:
        tag = node.name.lower()

        # Inline style wins over this element's own semantic default, so read it first.
        props = _parse_style(node.get("style", "") or "")

        if font_weight is None and "font-weight" in props:
            fw = props["font-weight"].strip().lower()
            if fw not in ("inherit", "initial", "unset", ""):
                font_weight = fw
                bold = _is_bold_weight(fw)

        if not italic_resolved:
            fs = props.get("font-style", "").strip().lower()
            if fs in ("italic", "oblique", "normal"):
                italic = fs != "normal"
                italic_resolved = True

        # Semantic signals — only where the cascade has not already decided.
        if font_weight is None and tag in ("b", "strong"):
            bold = True
        if tag in ("s", "strike", "del"):
            strikethrough = True
        # Innermost sup/sub wins (walk runs innermost → outermost)
        if not script_type and tag == "sup":
            script_type = "superscript"
        if not script_type and tag == "sub":
            script_type = "subscript"
        if tag in _HEADING_FONT_PX:
            # h1-h6 are bold and have a default font size in every browser stylesheet
            if font_weight is None:
                bold = True
            if font_size is None:
                font_size = _HEADING_FONT_PX[tag]
        if not italic_resolved and tag in ("i", "em"):
            italic = True
            italic_resolved = True
        if tag == "u":
            underline = True

        if font_size is None and "font-size" in props:
            font_size = _css_size_to_px(props["font-size"])

        if font_family is None and "font-family" in props:
            fam = props["font-family"].split(",")[0].strip().strip("'\"")
            if fam:
                font_family = fam

        if color is None and "color" in props:
            color = props["color"].lower()

        td = (props.get("text-decoration", "") or props.get("text-decoration-line", "")).lower()
        if not underline and "underline" in td:
            underline = True
        if not strikethrough and "line-through" in td:
            strikethrough = True

        if not script_type:
            va = props.get("vertical-align", "").lower()
            if va == "super":
                script_type = "superscript"
            elif va == "sub":
                script_type = "subscript"

        # Legacy presentational attributes
        if font_family is None and tag == "font" and node.get("face"):
            fam = str(node["face"]).split(",")[0].strip().strip("'\"")
            if fam:
                font_family = fam
        if color is None and tag == "font" and node.get("color"):
            color = str(node["color"]).strip().lower()

        node = node.parent

    return {
        "font_size": font_size,
        "font_family": font_family or "",
        "font_weight_raw": font_weight,
        "bold": bold,
        "italic": italic,
        "underline": underline,
        "strikethrough": strikethrough,
        "script_type": script_type,
        "color": color or "",
    }


def _resolve_block_text_align(el: Tag) -> str:
    """
    Resolve text-align for a block element, walking up for the inherited value.

    text-align is a block-level property, so it is resolved once per structure
    element rather than per inline run (a `text-align` on a <span> has no effect in
    a browser). Logical and vendor-prefixed values are normalized exactly as the JS
    extractor does, so both paths emit the same vocabulary — notably
    "justify" → "justified".
    """
    node: Any = el
    align = ""
    direction = ""
    while node is not None and isinstance(node, Tag) and node.name:
        props = _parse_style(node.get("style", "") or "")
        if not align:
            align = (props.get("text-align", "") or str(node.get("align", "") or "")).strip().lower()
        if not direction:
            d = props.get("direction", "").strip().lower() or str(node.get("dir", "") or "").strip().lower()
            if d in ("ltr", "rtl"):
                direction = d
        if align and direction:
            break
        node = node.parent
    direction = direction or "ltr"

    align = re.sub(r"^-(webkit|moz|ms|o)-", "", align)
    if not align:
        # Browser UA default: `text-align: start` everywhere, `center` for <th>.
        # getComputedStyle always reports a value, so emitting "" here would leave a
        # systematic gap against the Playwright path on documents that never declare
        # text-align inline.
        align = "center" if (el.name or "").lower() == "th" else "start"
    if align == "start":
        return "right" if direction == "rtl" else "left"
    if align == "end":
        return "left" if direction == "rtl" else "right"
    if align == "justify":
        return "justified"
    return align


def _resolve_x_left(el: Tag) -> float:
    """
    Read x_left from the block element's own inline style only.
    `left:` for absolutely-positioned HTML; `margin-left:` as an indentation proxy.
    Not walked up ancestors — parent margins compound unpredictably without a layout engine.
    """
    props = _parse_style(el.get("style", "") or "")
    return (
        _css_size_to_px(props.get("left", ""))
        or _css_size_to_px(props.get("margin-left", ""))
        or 0.0
    )


# ---------------------------------------------------------------------------
# Inline context — nearest inline-split ancestor of a text node
# ---------------------------------------------------------------------------

def _inline_split_parent(start: Any, block: Tag) -> Tag:
    """
    Return the nearest ancestor of `start` (inclusive) that is an inline-split tag,
    or `block` itself when there is none. Mirrors the JS findInlineSplitParent.
    """
    node: Any = start
    while node is not None and isinstance(node, Tag) and node is not block:
        if (node.name or "").lower() in _INLINE_SPLIT_TAGS:
            return node
        node = node.parent
    return block


# ---------------------------------------------------------------------------
# Ancestor metadata
# ---------------------------------------------------------------------------

def _ancestor_meta(el: Tag, dom_order: Dict[int, int]) -> Dict[str, List[Any]]:
    ids, classes, tags, tag_ids, roles = [], [], [], [], []
    # Start at el itself so struct_ancestors INCLUDES the box's own tag as the last entry.
    node: Any = el
    while node is not None and isinstance(node, Tag) and node.name:
        # Skip the BeautifulSoup '[document]' pseudo-root — it is a parser artifact,
        # not a real ancestor element, and has no dom_order id.
        if node.name == "[document]":
            break
        if node.get("id"):
            ids.append(str(node["id"]))
        cls = node.get("class")
        if cls:
            if isinstance(cls, list):
                classes.extend(cls)
            else:
                classes.extend(str(cls).split())
        # struct_ancestors / struct_ancestor_ids stay index-parallel (one entry per
        # ancestor element). tag_id is a document-order unique id, so two boxes under
        # the same ancestor share its id. Whitelist gate intentionally absent here.
        tags.append(node.name.lower())
        tag_ids.append(dom_order.get(id(node), -1))
        role = node.get("role") or node.get("aria-role")
        if role:
            roles.append(str(role).strip().lower())
        node = node.parent
    # Reverse so arrays run highest-ancestor → direct-parent (matches JS extractor)
    return {
        "ancestor_ids": ids[::-1],
        "ancestor_classes": classes[::-1],
        "struct_ancestors": tags[::-1],
        "struct_ancestor_ids": tag_ids[::-1],
        "ancestor_aria_roles": roles[::-1],
    }


# ---------------------------------------------------------------------------
# Table context
# ---------------------------------------------------------------------------

def _table_context(
    el: Tag,
    table_id_map: Dict[int, int],
    row_id_map: Dict[int, int],
) -> Dict[str, Any]:
    tbl = el.find_parent("table")
    table_id = table_id_map.get(id(tbl)) if tbl else None

    tr = el.find_parent("tr")
    table_row_id = row_id_map.get(id(tr)) if tr else None

    cell_index = None
    row_cell_count = None
    if tr:
        cells = tr.find_all(["td", "th"], recursive=False)
        row_cell_count = len(cells)
        # Find which cell contains el
        cell_ancestor = el if el.name and el.name.lower() in ("td", "th") else el.find_parent(["td", "th"])
        if cell_ancestor is not None:
            for i, c in enumerate(cells):
                if c is cell_ancestor:
                    cell_index = i
                    break

    return {
        "table_id": table_id,
        "table_row_id": table_row_id,
        "table_cell_index": cell_index,
        "table_row_cell_count": row_cell_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_url(el: Tag, allow_descendants: bool = True) -> str:
    """
    Resolve the href covering `el`: el itself, then (optionally) a descendant
    <a>, then an ancestor <a>.

    `allow_descendants=False` is used for plain text runs anchored on the block
    element — a sibling <a> deeper in the block does not link that text.
    """
    if (el.name or "").lower() == "a" and el.has_attr("href"):
        return str(el["href"])
    a = (el.find("a", href=True) if allow_descendants else None) or el.find_parent("a", href=True)
    return str(a["href"]) if a else ""


def _ixbrl_id(el: Tag) -> str:
    """Return the iXBRL id if el itself is an ix:* element (mirrors JS findIxbrlId)."""
    if el and isinstance(el, Tag) and el.name:
        name = el.name.upper()
        if name.startswith("IX:") or name.startswith("IX-") or ":IX" in name:
            return str(el.get("id", "")).strip()
    return ""


def _data_attrs(el: Tag) -> Dict[str, str]:
    return {k: str(v) for k, v in el.attrs.items() if k.startswith("data-")}


# ---------------------------------------------------------------------------
# Atomic structure detection (mirrors JS findAtomicStructures)
# ---------------------------------------------------------------------------

def _find_atomic_structures(soup: BeautifulSoup) -> List[Tag]:
    """
    Block elements that have text but contain no block-level descendants with text.
    These are the smallest meaningful content units — one box per element.
    """
    candidates: List[Tag] = soup.find_all(list(STRUCTURE_TAGS))

    has_text = [el for el in candidates if (el.get_text() or "").strip()]
    has_text_ids = {id(el) for el in has_text}

    # Mark non-atomic: a structure element that has another structure element
    # with text somewhere in its subtree
    non_atomic: set[int] = set()
    for el in has_text:
        node: Any = el.parent
        while node is not None and isinstance(node, Tag):
            if id(node) in has_text_ids:
                non_atomic.add(id(node))
            node = node.parent

    return [el for el in has_text if id(el) not in non_atomic]


# ---------------------------------------------------------------------------
# Shared box skeleton
# ---------------------------------------------------------------------------

def _base_box(
    el: Tag,
    structure_tag_id: int,
    table_id_map: Dict[int, int],
    row_id_map: Dict[int, int],
    dom_order: Dict[int, int],
) -> Dict[str, Any]:
    """Fields common to all box types."""
    dom_class_raw = el.get("class", [])
    dom_class = " ".join(dom_class_raw) if isinstance(dom_class_raw, list) else str(dom_class_raw or "")
    return {
        "box_id": structure_tag_id,
        "struct_tag_id": structure_tag_id,
        "x_left": _resolve_x_left(el),
        "x_right": 0.0,
        "y_top": 0.0,
        "y_bottom": 0.0,
        "width": 0.0,
        "height": 0.0,
        "text_orientation": "LTR",
        "stroking_color": "",
        "dom_id": str(el.get("id", "") or ""),
        "dom_class": dom_class,
        "html_data_attrs": _data_attrs(el),
        "ixbrl_id": _ixbrl_id(el),
        "page_number": 1,
        "page_width": _DEFAULT_VIEWPORT_WIDTH,
        "page_height": 0,
        "page_format": "html_static",
        **_table_context(el, table_id_map, row_id_map),
        **_ancestor_meta(el, dom_order),
    }


# ---------------------------------------------------------------------------
# Inline segment walker
# ---------------------------------------------------------------------------

def _pre_line_segments(el: Tag) -> List[tuple[str, Tag, Tag, str]]:
    """
    Segment a <pre> block into one segment per code line.

    Inside <pre>, whitespace IS the layout: split on newline characters (and
    <br>) instead of on inline tags, so syntax-highlighter token spans (shiki,
    Pygments, highlight.js) are transparent. Leading indentation is preserved;
    only trailing whitespace is stripped. split_reason "code_line" tells
    step_02 not to re-merge these boxes by struct_tag_id (same contract as
    "br_tag"), and step_04 skips leading-whitespace stripping for pre boxes.
    """
    parts: List[str] = []
    for d in el.descendants:
        if isinstance(d, Comment):
            continue
        if isinstance(d, NavigableString):
            parts.append(str(d))
        elif isinstance(d, Tag) and (d.name or "").lower() == "br":
            parts.append("\n")

    segments: List[tuple[str, Tag, Tag, str]] = []
    for raw_line in "".join(parts).split("\n"):
        line = raw_line.rstrip()
        if line.strip():
            segments.append((line, el, el, "code_line"))
    return segments


def _iter_inline_segments(el: Tag) -> List[tuple[str, Tag, Tag, str]]:
    """
    Walk el's content in document order and return
    (text, inline_anchor, style_element, split_reason) tuples.

    <pre> blocks are delegated to :func:`_pre_line_segments` (one segment per
    code line, indentation preserved).

    Port of the JS extractor's text-node walk: text accumulates into the current
    box until the *inline context* changes, where the context is the triple
    (nearest _INLINE_SPLIT_TAGS ancestor, covering href, iXBRL id of the direct
    parent). <br>, <hr> and <img> force a break as well.

    `inline_anchor` is that context element (becomes wrapping_tag); `style_element`
    is the direct parent of the segment's first text node, which is where
    :func:`_resolve_style` starts so the innermost inline styles are picked up even
    when they sit on a tag that is not itself a split tag (e.g. <ix:nonnumeric>).
    """
    if (el.name or "").lower() == "pre":
        return _pre_line_segments(el)

    segments: List[tuple[str, Tag, Tag, str]] = []
    parts: List[str] = []

    anchor: Tag = el          # current inline context element
    style_el: Tag = el        # direct parent of this segment's first text node
    link_url = ""
    ixbrl_id = ""
    split_reason = "new_structure"
    pending_space = False     # a whitespace-only text node separated two runs

    def flush() -> None:
        nonlocal parts
        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        parts = []
        if text:
            segments.append((text, anchor, style_el, split_reason))

    for node in el.descendants:
        if isinstance(node, Comment):
            continue

        if isinstance(node, Tag):
            tag = (node.name or "").lower()
            if tag in ("br", "hr", "img"):
                flush()
                anchor = el
                style_el = el
                link_url = ""
                ixbrl_id = ""
                pending_space = False
                split_reason = {"br": "br_tag", "hr": "after_hr", "img": "after_image"}[tag]
            continue

        raw = str(node)
        if not raw.strip():
            # Whitespace between two runs still separates words; remember it rather
            # than dropping the node, and let the next run carry it.
            if raw:
                pending_space = True
            continue

        parent = node.parent
        if not isinstance(parent, Tag):
            continue

        new_anchor = _inline_split_parent(parent, el)
        new_link = _link_url(parent, allow_descendants=False)
        new_ixbrl = _ixbrl_id(parent)

        if new_anchor is not anchor or new_link != link_url or new_ixbrl != ixbrl_id:
            had_text = bool("".join(parts).strip())
            flush()
            if had_text:
                if new_ixbrl and new_ixbrl != ixbrl_id:
                    split_reason = "ixbrl_change"
                elif new_link and new_link != link_url:
                    split_reason = "link_change"
                elif new_anchor is not anchor:
                    split_reason = "inline_exit" if new_anchor is el else "inline_tag"
            anchor, link_url, ixbrl_id = new_anchor, new_link, new_ixbrl
            style_el = parent
            pending_space = False

        if pending_space and parts:
            parts.append(" ")
        parts.append(raw)
        pending_space = False

    flush()
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_boxes_static(html: str) -> List[Dict[str, Any]]:
    """
    Extract boxes from HTML using BeautifulSoup.

    Returns a list of box dicts matching the schema of extract_boxes_with_playwright.
    x_left is taken from inline `left`/`margin-left` CSS on the block element;
    x_right / y_top / y_bottom / width / height are always 0.0 (no layout engine).
    Block elements are split into multiple boxes at <br> and semantic inline tag
    boundaries (strong, b, em, i, u, a …), each with its own style resolution.
    <hr> and <img> elements are emitted as separate boxes in DOM order.
    """
    with warnings.catch_warnings():
        # Input is intentionally HTML; silence bs4's XML-vs-HTML guess.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    for unwanted in soup(["script", "style", "noscript"]):
        unwanted.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for el in soup.find_all(True):
        if not el.attrs:
            continue
        props = _parse_style(el.get("style", "") or "")
        if props.get("display", "").lower() == "none" or props.get("visibility", "").lower() == "hidden":
            el.decompose()

    table_id_map = {id(t): i for i, t in enumerate(soup.find_all("table"), 1)}
    row_id_map = {id(r): i for i, r in enumerate(soup.find_all("tr"), 1)}

    dom_order: Dict[int, int] = {id(el): i for i, el in enumerate(soup.descendants) if isinstance(el, Tag)}

    text_elements = _find_atomic_structures(soup)
    hr_elements: List[Tag] = soup.find_all("hr")
    img_elements: List[Tag] = soup.find_all("img")

    tagged: List[tuple[int, str, Tag]] = (
        [(dom_order.get(id(el), 0), "text", el) for el in text_elements]
        + [(dom_order.get(id(el), 0), "hr", el) for el in hr_elements]
        + [(dom_order.get(id(el), 0), "img", el) for el in img_elements]
    )
    tagged.sort(key=lambda t: t[0])

    boxes: List[Dict[str, Any]] = []
    box_counter = 0       # increments per emitted box
    struct_counter = 0    # increments per atomic element (shared by all segments from the same element)

    for (_, kind, el) in tagged:
        struct_counter += 1
        if kind in ("hr", "img"):
            box_counter += 1
            base = _base_box(el, struct_counter, table_id_map, row_id_map, dom_order)
            base["box_id"] = box_counter

            if kind == "hr":
                hr_width = (
                    el.get("width")
                    or _parse_style(el.get("style", "") or "").get("width")
                )
                boxes.append({
                    **base,
                    "struct_tag": "hr",
                    "wrapping_tag": "hr",
                    "split_reason": "horizontal_rule",
                    "text": f"[[HR: {hr_width}]]" if hr_width else "[[HR]]",
                    "font_size": "", "font_family": "", "font_weight": 400,
                    "bold_ratio": 0.0, "italic_ratio": 0.0, "underlined_ratio": 0.0,
                    "strikethrough_ratio": 0.0, "is_strikethrough": False, "script_type": "",
                    "non_stroking_color": "", "text_align": "",
                    "link_url": "", "img_alt": "", "img_src": "",
                })
            else:
                alt = str(el.get("alt", "") or "")
                boxes.append({
                    **base,
                    "struct_tag": "img",
                    "wrapping_tag": "img",
                    "split_reason": "image",
                    "text": f"[[IMAGE: {alt}]]" if alt else "[[IMAGE]]",
                    "font_size": "", "font_family": "", "font_weight": 400,
                    "bold_ratio": 0.0, "italic_ratio": 0.0, "underlined_ratio": 0.0,
                    "strikethrough_ratio": 0.0, "is_strikethrough": False, "script_type": "",
                    "non_stroking_color": "", "text_align": "",
                    "link_url": _link_url(el),
                    "img_alt": alt, "img_src": str(el.get("src", "") or ""),
                })

        else:  # text
            segments = _iter_inline_segments(el)
            if not segments:
                continue

            tag = el.name.lower()
            # text-align is block-level: resolve once per structure element, not per run.
            block_text_align = _resolve_block_text_align(el)

            # For <pre> blocks, anchor ancestry on a direct <code> wrapper when
            # present (pre > code is the standard highlighter structure), so
            # struct_ancestors ends [..., "pre", "code"].
            ancestor_el = el
            if tag == "pre":
                code_child = el.find("code", recursive=False)
                if code_child is not None:
                    ancestor_el = code_child

            dom_class_raw = el.get("class", [])
            dom_class = " ".join(dom_class_raw) if isinstance(dom_class_raw, list) else str(dom_class_raw or "")
            shared = {
                "struct_tag": tag,
                "x_left": _resolve_x_left(el),
                "x_right": 0.0, "y_top": 0.0, "y_bottom": 0.0,
                "width": 0.0, "height": 0.0,
                "text_orientation": "LTR",
                "stroking_color": "",
                "dom_id": str(el.get("id", "") or ""),
                "dom_class": dom_class,
                "html_data_attrs": _data_attrs(el),
                "page_number": 1,
                "page_width": _DEFAULT_VIEWPORT_WIDTH,
                "page_height": 0,
                "page_format": "html_static",
                **_table_context(el, table_id_map, row_id_map),
                **_ancestor_meta(ancestor_el, dom_order),
            }

            for seg_text, seg_anchor, seg_style_el, seg_split_reason in segments:
                box_counter += 1
                style = _resolve_style(seg_style_el)
                font_size_px = style["font_size"] or _DEFAULT_FONT_PX
                fw_raw = style["font_weight_raw"]
                if fw_raw is not None:
                    m = re.match(r"^(\d+)$", fw_raw)
                    font_weight = int(m.group(1)) if m else (700 if style["bold"] else 400)
                else:
                    font_weight = 700 if style["bold"] else 400

                wrapping = (seg_anchor.name or tag).lower() if seg_anchor is not el else tag

                boxes.append({
                    **shared,
                    "box_id": box_counter,
                    "struct_tag_id": struct_counter,
                    "ixbrl_id": _ixbrl_id(seg_style_el),
                    "wrapping_tag": wrapping,
                    "split_reason": seg_split_reason,
                    "text": seg_text,
                    # 4 decimals mirrors getComputedStyle ("10.6667px" for 8pt)
                    "font_size": f"{font_size_px:.4f}px",
                    "font_family": style["font_family"],
                    "font_weight": font_weight,
                    "bold_ratio": 1.0 if style["bold"] else 0.0,
                    "italic_ratio": 1.0 if style["italic"] else 0.0,
                    "underlined_ratio": 1.0 if style["underline"] else 0.0,
                    "strikethrough_ratio": 1.0 if style["strikethrough"] else 0.0,
                    "is_strikethrough": bool(style["strikethrough"]),
                    "script_type": style["script_type"],
                    "non_stroking_color": style["color"],
                    "text_align": block_text_align,
                    "link_url": _link_url(seg_style_el, allow_descendants=False),
                    "img_alt": "", "img_src": "",
                })

    return boxes

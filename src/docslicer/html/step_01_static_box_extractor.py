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

- One box per block element (no intra-element inline splits). bold_ratio and
  italic_ratio reflect whether the element or any descendant carries that style.

Works well for documents that use inline styles — SEC filings, Word-exported
HTML, legal documents. Less useful for modern CSS-class-heavy pages.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

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

# Inline tags that create a segment boundary within a block element.
# Conservative subset: only tags carrying meaningful semantic style changes.
# <span>/<font> are excluded — too frequent on CSS-styled pages, causes fragmentation.
_SEGMENT_SPLIT_TAGS: frozenset[str] = frozenset({
    "strong", "b", "em", "i", "u", "a", "mark", "s", "del", "ins",
    "strike", "sup", "sub",  # strike → strikethrough; sup/sub → script_type
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
    if unit == "pt":
        return round(n * 96.0 / 72.0, 2)
    if unit in ("em", "rem"):
        return round(n * _DEFAULT_FONT_PX, 2)
    if unit == "%":
        return round(n / 100.0 * _DEFAULT_FONT_PX, 2)
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
    tag names. Call this on _style_anchor(block_el) so the walk starts at the
    innermost text-wrapping element (e.g. the <span>) and naturally inherits
    upward through all wrappers to the block.
    """
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    font_weight: Optional[str] = None
    bold = False
    italic = False
    underline = False
    strikethrough = False
    script_type = ""   # "superscript" | "subscript" | ""
    text_align: Optional[str] = None
    color: Optional[str] = None

    node: Any = el
    while node is not None and isinstance(node, Tag) and node.name:
        tag = node.name.lower()

        # Semantic signals
        if tag in ("b", "strong"):
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
            bold = True
            if font_size is None:
                font_size = _HEADING_FONT_PX[tag]
        if tag in ("i", "em"):
            italic = True
        if tag == "u":
            underline = True

        # Inline style
        props = _parse_style(node.get("style", "") or "")

        if font_size is None and "font-size" in props:
            font_size = _css_size_to_px(props["font-size"])

        if font_family is None and "font-family" in props:
            fam = props["font-family"].split(",")[0].strip().strip("'\"")
            if fam:
                font_family = fam

        if font_weight is None and "font-weight" in props:
            font_weight = props["font-weight"].lower()
            if not bold and _is_bold_weight(font_weight):
                bold = True

        if text_align is None and "text-align" in props:
            text_align = props["text-align"].lower()

        if color is None and "color" in props:
            color = props["color"].lower()

        if not italic:
            fs = props.get("font-style", "").lower()
            if fs in ("italic", "oblique"):
                italic = True

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

        # Legacy align attribute
        if text_align is None and node.get("align"):
            text_align = str(node["align"]).strip().lower()

        if all(v is not None for v in [font_size, font_family, font_weight, text_align, color]):
            break

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
        "text_align": text_align or "",
        "color": color or "",
    }


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
# Style anchor — find the innermost element that wraps the first text node
# ---------------------------------------------------------------------------

def _style_anchor(el: Tag) -> Tag:
    """
    Return the deepest element that directly wraps the first non-empty text node
    inside `el`. Starting _resolve_style from here means the upward walk passes
    through all wrapping inline tags (span, font, b, …) before reaching the block,
    mirroring how a browser resolves getComputedStyle on a text node.
    """
    for string in el.strings:
        if string.strip():
            parent = string.parent
            if isinstance(parent, Tag):
                return parent
    return el


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

    table_header_flag = bool(el.name and el.name.lower() == "th")

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
        "table_header_flag": table_header_flag,
        "table_cell_index": cell_index,
        "table_row_cell_count": row_cell_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_url(el: Tag) -> str:
    a = el.find("a", href=True) or el.find_parent("a", href=True)
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
    style = _resolve_style(_style_anchor(el))
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
        # style fields — caller may override
        "_style": style,
    }


# ---------------------------------------------------------------------------
# Inline segment walker
# ---------------------------------------------------------------------------

def _iter_inline_segments(el: Tag) -> List[tuple[str, Tag, str]]:
    """
    Walk el's content and return (text, style_anchor, split_reason) tuples.

    A new segment is started at:
    - <br>  → split_reason "br_tag"  (step_02 respects this and won't re-merge)
    - _SEGMENT_SPLIT_TAGS (strong, b, em, i, u, a, …) → each gets its own segment
      with the inline tag as the style anchor so _resolve_style picks up its styles.

    Plain text runs between split tags use `el` (the block) as the style anchor.
    """
    segments: List[tuple[str, Tag, str]] = []
    current_parts: List[str] = []

    def flush(anchor: Tag, split_reason: str = "static_extractor") -> None:
        text = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
        current_parts.clear()
        if text:
            segments.append((text, anchor, split_reason))

    def walk(node: Tag, anchor: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t:
                    current_parts.append(t)
            elif isinstance(child, Tag):
                child_tag = (child.name or "").lower()
                if child_tag == "br":
                    flush(anchor, "br_tag")
                elif child_tag in _SEGMENT_SPLIT_TAGS:
                    flush(anchor)
                    inner = re.sub(r"\s+", " ", child.get_text() or "").strip()
                    if inner:
                        segments.append((inner, child, "static_extractor"))
                elif child_tag.startswith("ix:") or child_tag.startswith("ix-"):
                    # iXBRL element — flush current buffer and emit as its own segment
                    # so seg_anchor == the ix: element and _ixbrl_id picks up its id
                    flush(anchor)
                    inner = re.sub(r"\s+", " ", child.get_text() or "").strip()
                    if inner:
                        segments.append((inner, child, "static_extractor"))
                else:
                    walk(child, anchor)

    walk(el, el)
    flush(el)
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
            base.pop("_style")

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
                **_ancestor_meta(el, dom_order),
            }

            for seg_text, seg_anchor, seg_split_reason in segments:
                box_counter += 1
                style = _resolve_style(seg_anchor)
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
                    "ixbrl_id": _ixbrl_id(seg_anchor),
                    "wrapping_tag": wrapping,
                    "split_reason": seg_split_reason,
                    "text": seg_text,
                    "font_size": f"{font_size_px:.2f}px",
                    "font_family": style["font_family"],
                    "font_weight": font_weight,
                    "bold_ratio": 1.0 if style["bold"] else 0.0,
                    "italic_ratio": 1.0 if style["italic"] else 0.0,
                    "underlined_ratio": 1.0 if style["underline"] else 0.0,
                    "strikethrough_ratio": 1.0 if style["strikethrough"] else 0.0,
                    "is_strikethrough": bool(style["strikethrough"]),
                    "script_type": style["script_type"],
                    "non_stroking_color": style["color"],
                    "text_align": style["text_align"],
                    "link_url": _link_url(seg_anchor if seg_anchor is not el else el),
                    "img_alt": "", "img_src": "",
                })

    return boxes

"""
PPTX run extractor.

Builds a run-level DataFrame from DrawingML slide content. Every output row
carries its slide/shape/paragraph context so paragraph and table layers can
be reconstructed without re-walking the XML package.

Column schema is kept as close to the DOCX run_df as possible so shared
downstream steps (paragraph builder, line builder, hierarchy) work unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from lxml import etree

from .step_01_package_reader import PptxPackage, PptxSlide
from .._utils.text_utils import add_calculated_text_features


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

A = f"{{{NS['a']}}}"
C = f"{{{NS['c']}}}"
P = f"{{{NS['p']}}}"
R = f"{{{NS['r']}}}"
MC = f"{{{NS['mc']}}}"
A14 = f"{{{NS['a14']}}}"
M = f"{{{NS['m']}}}"

_TABLE_GRAPHIC_URI = "http://schemas.openxmlformats.org/drawingml/2006/table"
_REL_NOTES_SLIDE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
)
_PRESENTATION_PART = "ppt/presentation.xml"

# Placeholder types that carry title-level content
_TITLE_PH_TYPES = frozenset({"title", "ctrTitle"})

# Placeholder types that are decorative / non-content
_SKIP_PH_TYPES = frozenset({"sldImg", "sldNum", "dt", "ftr", "hdr"})

_LIST_STYLE_LEVEL_TAGS = {
    "lvl1pPr": 0,
    "lvl2pPr": 1,
    "lvl3pPr": 2,
    "lvl4pPr": 3,
    "lvl5pPr": 4,
    "lvl6pPr": 5,
    "lvl7pPr": 6,
    "lvl8pPr": 7,
    "lvl9pPr": 8,
}

_SYMBOL_BULLET_MAP = {
    ("wingdings", "§"): "▪",
    ("wingdings", "l"): "●",
    ("wingdings", "n"): "■",
    ("wingdings", "u"): "◆",
    ("wingdings", "v"): "◇",
    ("wingdings", "Ø"): "➢",
    ("wingdings", "ü"): "✓",
    ("wingdings 2", "–"): "➢",
    ("symbol", "·"): "•",
    ("symbol", "\uf0b7"): "•",
}


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Counters:
    run_id: int = 0
    paragraph_id: int = 0
    shape_id: int = 0
    chart_id: int = 0
    table_id: int = 0
    table_row_id: int = 0
    table_cell_id: int = 0
    order_index: int = 0


@dataclass(frozen=True)
class _Box:
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_right - self.x_left)

    @property
    def height(self) -> float:
        return max(0.0, self.y_bottom - self.y_top)


@dataclass(frozen=True)
class _Transform:
    """Composed affine map from a group's child (chOff/chExt) space to slide EMUs.

    A p:grpSp's children store a:off/a:ext in the group's local chOff/chExt
    coordinate system, not slide-absolute EMUs. Nested groups compose their
    transforms, so a shape several groups deep needs the full chain applied
    to land on its true slide position.
    """
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + self.scale_x * x, self.offset_y + self.scale_y * y

    def apply_extent(self, cx: float, cy: float) -> tuple[float, float]:
        return self.scale_x * cx, self.scale_y * cy


_IDENTITY_TRANSFORM = _Transform()


def _child_transform(parent: _Transform, grp_sp_pr: etree._Element) -> _Transform:
    """Derive the transform for a p:grpSp's children from its grpSpPr/a:xfrm."""
    xfrm = grp_sp_pr.find(f"{A}xfrm")
    if xfrm is None:
        return parent
    off = xfrm.find(f"{A}off")
    ext = xfrm.find(f"{A}ext")
    ch_off = xfrm.find(f"{A}chOff")
    ch_ext = xfrm.find(f"{A}chExt")
    if off is None or ext is None or ch_off is None or ch_ext is None:
        return parent
    try:
        off_x, off_y = float(off.get("x")), float(off.get("y"))
        ext_cx, ext_cy = float(ext.get("cx")), float(ext.get("cy"))
        ch_off_x, ch_off_y = float(ch_off.get("x")), float(ch_off.get("y"))
        ch_ext_cx, ch_ext_cy = float(ch_ext.get("cx")), float(ch_ext.get("cy"))
    except (TypeError, ValueError):
        return parent
    if ch_ext_cx == 0 or ch_ext_cy == 0:
        return parent

    scale_x = parent.scale_x * (ext_cx / ch_ext_cx)
    scale_y = parent.scale_y * (ext_cy / ch_ext_cy)
    offset_x = parent.offset_x + parent.scale_x * off_x - scale_x * ch_off_x
    offset_y = parent.offset_y + parent.scale_y * off_y - scale_y * ch_off_y
    return _Transform(scale_x, scale_y, offset_x, offset_y)


@dataclass(frozen=True)
class _Context:
    source_part: str
    slide_index: int
    slide_number: int
    shape_id: int
    shape_name: str
    shape_type: str
    placeholder_type: str | None
    is_notes: bool = False
    chart_id: int | None = None
    table_id: int | None = None
    table_row_id: int | None = None
    table_cell_id: int | None = None
    box: _Box | None = None
    # Enclosing p:grpSp ancestry, outermost → innermost, as (id, name) pairs.
    group_path: tuple[tuple[str, str], ...] = ()


class _PptxListResolver:
    """Resolve DrawingML bullet/auto-number labels while walking one text body."""

    def __init__(self) -> None:
        self.counters: dict[str, list[int]] = {}

    def next_label(self, props: dict[str, Any]) -> str | None:
        if props.get("list_type") == "bullet":
            return props.get("list_label")

        if props.get("list_type") != "auto":
            return None

        list_num_id = props.get("list_num_id")
        list_level = props.get("list_level")
        if list_num_id is None or list_level is None:
            return None

        try:
            level = int(list_level)
        except (TypeError, ValueError):
            return None

        start = int(props.get("list_start_at") or 1)
        counters = self.counters.setdefault(str(list_num_id), [0] * 9)
        counters[level] = counters[level] + 1 if counters[level] else start
        for reset_level in range(level + 1, len(counters)):
            counters[reset_level] = 0

        return _format_auto_number_label(
            counters[level],
            str(props.get("list_auto_type") or "arabicPeriod"),
        )


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

_MULTISPACE_RE = re.compile(r"\s{2,}")


def _collapse_spaces(text: str) -> str:
    """Fold runs of 2+ whitespace into a single space.

    Authors sometimes pad a run with many spaces to fake alignment or a line
    break (e.g. "Less Administrative      Burden"). That padding is visual, not
    semantic, so collapse it — but keep a single space so runs that are just a
    separator between two styled runs survive.
    """
    return _MULTISPACE_RE.sub(" ", text)


def _bool_attr(val: str | None) -> bool | None:
    """Parse a DrawingML boolean attribute (absent = inherit → None)."""
    if val is None:
        return None
    return val not in {"0", "false", "False"}


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _emu_to_pt(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 12700.0
    except (TypeError, ValueError):
        return None


def _extract_xfrm_box(
    elem: etree._Element,
    transform: _Transform = _IDENTITY_TRANSFORM,
) -> _Box | None:
    """
    Extract a DrawingML transform box as points.

    PPTX coordinates are EMUs. Converting to points keeps these geometry
    columns close to the PDF pipeline's coordinate scale. `transform` maps
    the shape's own a:off/a:ext (expressed in its parent p:grpSp's local
    chOff/chExt space, if any) into slide-absolute EMUs before conversion.
    """
    xfrm = elem.find(f".//{A}xfrm")
    if xfrm is None:
        xfrm = elem.find(f"{P}xfrm")
    if xfrm is None:
        return None

    off = xfrm.find(f"{A}off")
    ext = xfrm.find(f"{A}ext")
    if off is None or ext is None:
        return None

    try:
        raw_x, raw_y = float(off.get("x")), float(off.get("y"))
        raw_cx, raw_cy = float(ext.get("cx")), float(ext.get("cy"))
    except (TypeError, ValueError):
        return None
    abs_x, abs_y = transform.apply_point(raw_x, raw_y)
    abs_cx, abs_cy = transform.apply_extent(raw_cx, raw_cy)

    x_left = _emu_to_pt(abs_x)
    y_top = _emu_to_pt(abs_y)
    width = _emu_to_pt(abs_cx)
    height = _emu_to_pt(abs_cy)
    if None in (x_left, y_top, width, height):
        return None

    return _Box(
        x_left=x_left,
        y_top=y_top,
        x_right=x_left + width,
        y_bottom=y_top + height,
    )


def _placeholder_type(ph: etree._Element | None) -> str | None:
    if ph is None:
        return None
    return ph.get("type") or "body"


def _placeholder_idx(ph: etree._Element | None) -> str | None:
    return ph.get("idx") if ph is not None else None


def _placeholder_types_match(requested: str | None, candidate: str | None) -> bool:
    if requested == candidate:
        return True
    title_types = {"title", "ctrTitle"}
    return requested in title_types and candidate in title_types


def _find_placeholder_shape_in_part(
    root: etree._Element | None,
    placeholder_type: str | None,
    placeholder_idx: str | None,
) -> etree._Element | None:
    if root is None or placeholder_type is None:
        return None

    fallback: etree._Element | None = None
    for sp in root.findall(f".//{P}sp"):
        ph = sp.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
        if ph is None:
            continue
        cand_type = _placeholder_type(ph)
        cand_idx = _placeholder_idx(ph)
        if not _placeholder_types_match(placeholder_type, cand_type):
            continue
        if placeholder_idx is not None and cand_idx == placeholder_idx:
            return sp
        if placeholder_idx is None and cand_idx is None:
            return sp
        if fallback is None:
            fallback = sp
    return fallback


def _find_placeholder_shapes(
    slide: PptxSlide,
    package: PptxPackage,
    ph: etree._Element | None,
) -> list[etree._Element]:
    """Return matching placeholder shapes from layout then master (both levels, not just first)."""
    if ph is None:
        return []
    placeholder_type = _placeholder_type(ph)
    placeholder_idx = _placeholder_idx(ph)
    results = []
    for part_name in (slide.layout_part_name, slide.master_part_name):
        inherited = _find_placeholder_shape_in_part(
            package.get_xml(part_name) if part_name else None,
            placeholder_type,
            placeholder_idx,
        )
        if inherited is not None:
            results.append(inherited)
    return results


def _box_columns(box: _Box | None) -> dict[str, float | None]:
    return {
        "x_left": box.x_left if box else None,
        "y_top": box.y_top if box else None,
        "x_right": box.x_right if box else None,
        "y_bottom": box.y_bottom if box else None,
        "width": box.width if box else None,
        "height": box.height if box else None,
    }


def _alpha_number(value: int, uppercase: bool) -> str:
    if value <= 0:
        value = 1
    chars: list[str] = []
    while value:
        value -= 1
        chars.append(chr(ord("A") + (value % 26)))
        value //= 26
    text = "".join(reversed(chars))
    return text if uppercase else text.lower()


def _roman_number(value: int, uppercase: bool) -> str:
    if value <= 0:
        value = 1
    pairs = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    )
    out: list[str] = []
    for arabic, roman in pairs:
        while value >= arabic:
            out.append(roman)
            value -= arabic
    text = "".join(out)
    return text if uppercase else text.lower()


def _format_auto_number_value(value: int, auto_type: str) -> str:
    if auto_type.startswith("alphaLc"):
        return _alpha_number(value, uppercase=False)
    if auto_type.startswith("alphaUc"):
        return _alpha_number(value, uppercase=True)
    if auto_type.startswith("romanLc"):
        return _roman_number(value, uppercase=False)
    if auto_type.startswith("romanUc"):
        return _roman_number(value, uppercase=True)
    return str(value)


def _format_auto_number_label(value: int, auto_type: str) -> str:
    number = _format_auto_number_value(value, auto_type)
    if auto_type.endswith("ParenR"):
        return f"{number})"
    if auto_type.endswith("ParenBoth"):
        return f"({number})"
    if auto_type.endswith("Period"):
        return f"{number}."
    return number


def _normalize_text_align(val: str | None) -> str | None:
    if not val:
        return None
    mapping = {"l": "left", "r": "right", "ctr": "center", "just": "justified",
               "dist": "justified", "thaiDist": "justified"}
    return mapping.get(val, val)


def _normalize_bullet_char(p_pr: etree._Element, char: str | None) -> str:
    if not char:
        return "•"

    bu_font = p_pr.find(f"{A}buFont")
    typeface = (bu_font.get("typeface") if bu_font is not None else None) or ""
    normalized_typeface = typeface.strip().lower()

    # Symbol/Wingdings fonts often encode glyphs in the U+F000–U+F0FF private-use
    # area (e.g. U+F0FC for the Wingdings check). Fold to the low byte so a single
    # map entry (keyed on the plain char) covers both encodings.
    if len(char) == 1 and 0xF000 <= ord(char) <= 0xF0FF:
        folded = chr(ord(char) - 0xF000)
        if (normalized_typeface, folded) in _SYMBOL_BULLET_MAP:
            return _SYMBOL_BULLET_MAP[(normalized_typeface, folded)]

    return _SYMBOL_BULLET_MAP.get((normalized_typeface, char), char)


def _script_type(baseline: str | None) -> str | None:
    """Map a DrawingML rPr baseline (percentage) to super/subscript.

    baseline is a percentage of the run's font size; positive raises the text
    (superscript), negative lowers it (subscript), absent/zero is neither.
    """
    if baseline is None:
        return None
    try:
        value = int(baseline)
    except ValueError:
        return None
    if value > 0:
        return "superscript"
    if value < 0:
        return "subscript"
    return None


def _extract_color(elem: etree._Element) -> str | None:
    """Return #rrggbb from a:solidFill/a:srgbClr, or None for scheme colors."""
    solid = elem.find(f"{A}solidFill")
    if solid is None:
        return None
    srgb = solid.find(f"{A}srgbClr")
    if srgb is not None:
        val = srgb.get("val", "")
        if len(val) == 6:
            return "#" + val.lower()
    return None


# ---------------------------------------------------------------------------
# Property extractors
# ---------------------------------------------------------------------------

def _extract_r_props(r_pr: etree._Element | None) -> dict[str, Any]:
    if r_pr is None:
        return {}

    sz_raw = r_pr.get("sz")
    font_size = int(sz_raw) / 100.0 if sz_raw else None

    latin = r_pr.find(f"{A}latin")
    font_name = latin.get("typeface") if latin is not None else None
    # Keep theme font references (+mj-lt, +mn-lt, etc.) as-is; callers resolve them.

    u_val = r_pr.get("u")
    is_underlined = None if u_val is None else (u_val != "none")

    strike_val = r_pr.get("strike")
    is_strikethrough = None if strike_val is None else (strike_val != "noStrike")

    return _drop_none({
        "is_bold": _bool_attr(r_pr.get("b")),
        "is_italic": _bool_attr(r_pr.get("i")),
        "is_underlined": is_underlined,
        "is_strikethrough": is_strikethrough,
        "script_type": _script_type(r_pr.get("baseline")),
        "font_size": font_size,
        "font_name": font_name,
        "non_stroking_color": _extract_color(r_pr),
    })


def _extract_p_props(p_pr: etree._Element | None) -> dict[str, Any]:
    if p_pr is None:
        return {}
    lvl_raw = p_pr.get("lvl")
    try:
        level = int(lvl_raw) if lvl_raw is not None else None
    except ValueError:
        level = None

    bu_none = p_pr.find(f"{A}buNone")
    bu_char = p_pr.find(f"{A}buChar")
    bu_auto_num = p_pr.find(f"{A}buAutoNum")

    list_type = None
    list_label = None
    list_auto_type = None
    list_start_at = None

    if bu_none is not None:
        list_type = "none"
    elif bu_auto_num is not None:
        list_type = "auto"
        list_auto_type = bu_auto_num.get("type") or "arabicPeriod"
        start_raw = bu_auto_num.get("startAt")
        try:
            list_start_at = int(start_raw) if start_raw is not None else 1
        except ValueError:
            list_start_at = 1
    elif bu_char is not None:
        list_type = "bullet"
        list_label = _normalize_bullet_char(p_pr, bu_char.get("char"))

    return _drop_none({
        "list_num_id": None,
        "list_level": str(level) if level is not None and list_type not in (None, "none") else None,
        "list_label": list_label,
        "list_type": list_type,
        "list_auto_type": list_auto_type,
        "list_start_at": list_start_at,
        "outline_level": level,
        "text_align": _normalize_text_align(p_pr.get("algn")),
    })


def _extract_list_style_props(txbody: etree._Element) -> dict[int, dict[str, Any]]:
    lst_style = txbody.find(f"{A}lstStyle")
    if lst_style is None:
        return {}

    props_by_level: dict[int, dict[str, Any]] = {}
    for child in lst_style:
        local_name = etree.QName(child).localname
        level = _LIST_STYLE_LEVEL_TAGS.get(local_name)
        if level is None:
            continue
        props = _extract_p_props(child)
        props.pop("outline_level", None)
        props_by_level[level] = props
    return props_by_level


def _extract_list_style_r_props(txbody: etree._Element | None) -> dict[int, dict[str, Any]]:
    if txbody is None:
        return {}
    lst_style = txbody.find(f"{A}lstStyle")
    if lst_style is None:
        return {}

    props_by_level: dict[int, dict[str, Any]] = {}
    for child in lst_style:
        local_name = etree.QName(child).localname
        level = _LIST_STYLE_LEVEL_TAGS.get(local_name)
        if level is None:
            continue
        r_props = _extract_r_props(child.find(f"{A}defRPr"))
        if r_props:
            props_by_level[level] = r_props
    return props_by_level


def _extract_theme_fonts(theme_root: etree._Element | None) -> dict[str, str]:
    """Return {'major': typeface, 'minor': typeface} from the theme's fontScheme."""
    if theme_root is None:
        return {}
    font_scheme = theme_root.find(f".//{A}fontScheme")
    if font_scheme is None:
        return {}
    fonts: dict[str, str] = {}
    major = font_scheme.find(f"{A}majorFont/{A}latin")
    minor = font_scheme.find(f"{A}minorFont/{A}latin")
    if major is not None and major.get("typeface"):
        fonts["major"] = major.get("typeface")  # type: ignore[assignment]
    if minor is not None and minor.get("typeface"):
        fonts["minor"] = minor.get("typeface")  # type: ignore[assignment]
    return fonts


def _resolve_font_name(font_name: str | None, theme_fonts: dict[str, str]) -> str | None:
    if not font_name:
        return None
    if font_name.startswith("+mj-"):
        return theme_fonts.get("major") or None
    if font_name.startswith("+mn-"):
        return theme_fonts.get("minor") or None
    return font_name


def _resolve_font_name_with_fallback(
    raw_font_name: str | None,
    theme_fonts: dict[str, str],
    default_style_r_props: dict[int, dict[str, Any]],
    level: int,
) -> str | None:
    """Resolve a (possibly theme-referenced) font name, falling back to
    presentation.xml's p:defaultTextStyle when the theme's fontScheme has a
    blank major/minor typeface for that slot (seen when authors clear it)."""
    resolved = _resolve_font_name(raw_font_name, theme_fonts)
    if resolved:
        return resolved
    fallback_font = default_style_r_props.get(level, {}).get("font_name")
    if fallback_font:
        return _resolve_font_name(fallback_font, theme_fonts)
    return None


def _tx_styles_element(
    master_root: etree._Element | None,
    placeholder_type: str | None,
) -> etree._Element | None:
    """Select the p:txStyles style block that applies to a placeholder type."""
    if master_root is None:
        return None
    tx_styles = master_root.find(f"{P}txStyles")
    if tx_styles is None:
        return None

    if placeholder_type in _TITLE_PH_TYPES:
        return tx_styles.find(f"{P}titleStyle")
    if placeholder_type == "body":
        return tx_styles.find(f"{P}bodyStyle")
    return tx_styles.find(f"{P}otherStyle")


def _extract_tx_styles_r_props(
    master_root: etree._Element | None,
    placeholder_type: str | None,
) -> dict[int, dict[str, Any]]:
    """
    Read per-level defRPr from p:txStyles in the master slide.

    p:txStyles is the presentation-wide base for text formatting — it sits below
    the placeholder lstStyle in the inheritance chain and is the source of font
    size / name when placeholder lstStyles are empty.
    """
    style_elem = _tx_styles_element(master_root, placeholder_type)
    if style_elem is None:
        return {}

    props_by_level: dict[int, dict[str, Any]] = {}
    for child in style_elem:
        local_name = etree.QName(child).localname
        level = _LIST_STYLE_LEVEL_TAGS.get(local_name)
        if level is None:
            continue
        r_props = _extract_r_props(child.find(f"{A}defRPr"))
        if r_props:
            props_by_level[level] = r_props
    return props_by_level


def _extract_default_text_style_r_props(
    presentation_root: etree._Element | None,
) -> dict[int, dict[str, Any]]:
    """
    Read per-level defRPr from p:defaultTextStyle in presentation.xml.

    This is the outermost fallback in the run-property inheritance chain —
    below master p:txStyles — and is where authoring tools often stash a
    literal font name (e.g. "Arial") even when the theme's fontScheme itself
    has blank major/minor typefaces. Applies uniformly regardless of
    placeholder type, since it's the presentation-wide default for any text.
    """
    if presentation_root is None:
        return {}
    style_elem = presentation_root.find(f"{P}defaultTextStyle")
    if style_elem is None:
        return {}

    props_by_level: dict[int, dict[str, Any]] = {}
    for child in style_elem:
        local_name = etree.QName(child).localname
        level = _LIST_STYLE_LEVEL_TAGS.get(local_name)
        if level is None:
            continue
        r_props = _extract_r_props(child.find(f"{A}defRPr"))
        if r_props:
            props_by_level[level] = r_props
    return props_by_level


def _extract_tx_styles_p_props(
    master_root: etree._Element | None,
    placeholder_type: str | None,
) -> dict[int, dict[str, Any]]:
    """
    Read per-level list/paragraph props from p:txStyles in the master slide.

    Body placeholders usually inherit their bullet/auto-number definition from
    p:txStyles (bodyStyle), not from a placeholder lstStyle. This is the base of
    the list-property inheritance chain — the counterpart to the r-props reader.
    """
    style_elem = _tx_styles_element(master_root, placeholder_type)
    if style_elem is None:
        return {}

    props_by_level: dict[int, dict[str, Any]] = {}
    for child in style_elem:
        local_name = etree.QName(child).localname
        level = _LIST_STYLE_LEVEL_TAGS.get(local_name)
        if level is None:
            continue
        props = _extract_p_props(child)
        props.pop("outline_level", None)
        if props:
            props_by_level[level] = props
    return props_by_level


def _merge_props(*props: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for prop in props:
        merged.update(prop)
    return merged


def _paragraph_level(p_pr: etree._Element | None) -> int:
    if p_pr is None:
        return 0
    lvl_raw = p_pr.get("lvl")
    try:
        return int(lvl_raw) if lvl_raw is not None else 0
    except ValueError:
        return 0


def _resolve_p_props(
    p_elem: etree._Element,
    ctx: _Context,
    list_style_props: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    p_pr = p_elem.find(f"{A}pPr")
    level = _paragraph_level(p_pr)
    p_props = dict(list_style_props.get(level, {}))
    p_props.update(_extract_p_props(p_pr))

    list_type = p_props.get("list_type")
    if list_type in (None, "none"):
        p_props.pop("list_num_id", None)
        p_props.pop("list_level", None)
        p_props.pop("list_label", None)
        return p_props

    p_props["list_level"] = str(p_props.get("list_level", level))
    p_props["outline_level"] = level
    if p_props.get("list_num_id") is None:
        key_parts = [
            "pptx",
            "auto" if list_type == "auto" else "bullet",
            ctx.source_part,
            str(ctx.shape_id),
            str(ctx.table_cell_id or ""),
            str(p_props.get("list_auto_type") or p_props.get("list_label") or ""),
        ]
        p_props["list_num_id"] = ":".join(key_parts)
    return p_props


def _has_visible_paragraph_text(p_elem: etree._Element) -> bool:
    for child in p_elem:
        if child.tag in {f"{A}r", f"{A}fld"}:
            t_elem = child.find(f"{A}t")
            if t_elem is not None and (t_elem.text or "").strip():
                return True
    return False


def _paragraph_default_r_props(
    p_elem: etree._Element,
    level: int,
    inherited_level_r_props: dict[int, dict[str, Any]],
    level_r_props: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    p_pr = p_elem.find(f"{A}pPr")
    end_para_r_pr = p_elem.find(f"{A}endParaRPr")
    return _merge_props(
        inherited_level_r_props.get(level, {}),
        level_r_props.get(level, {}),
        _extract_r_props(p_pr.find(f"{A}defRPr") if p_pr is not None else None),
        _extract_r_props(end_para_r_pr),
    )


def _extract_hyperlink(
    r_pr: etree._Element | None,
    ctx: _Context,
    package: PptxPackage,
) -> tuple[str | None, str | None]:
    """Return (rel_id, url) from a:rPr/a:hlinkClick, if present."""
    if r_pr is None:
        return None, None
    hlink = r_pr.find(f"{A}hlinkClick")
    if hlink is None:
        return None, None
    rel_id = hlink.get(f"{R}id")
    rel = package.get_relationship(ctx.source_part, rel_id)
    url = rel.target if rel is not None else None
    return rel_id, url


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _append_row(
    rows: list[dict[str, Any]],
    counters: _Counters,
    ctx: _Context,
    paragraph_id: int,
    run_type: str,
    text: str,
    p_props: dict[str, Any],
    r_props: dict[str, Any],
    hyperlink_id: str | None,
    hyperlink_url: str | None,
) -> None:
    list_label = p_props.get("list_label")
    if (
        run_type in ("text", "math")
        and list_label
        and not p_props.get("_list_label_applied")
        and str(text).strip()
    ):
        label_prefix = f"{list_label} "
        if not str(text).lstrip().startswith(str(list_label)):
            text = label_prefix + text
        p_props["_list_label_applied"] = True

    counters.run_id += 1
    counters.order_index += 1
    is_bold = r_props.get("is_bold")
    is_italic = r_props.get("is_italic")
    is_underlined = r_props.get("is_underlined")
    is_strikethrough = r_props.get("is_strikethrough")
    rows.append({
        "run_id": counters.run_id,
        "paragraph_id": paragraph_id,
        "order_index": counters.order_index,
        "page_number": ctx.slide_number,
        "slide_index": ctx.slide_index,
        "source_part": ctx.source_part,
        "source_part_id": ctx.source_part,
        "header_footer_type": "notes" if ctx.is_notes else "body",
        "text": text,
        "run_type": run_type,
        "shape_id": ctx.shape_id,
        "chart_id": ctx.chart_id,
        "shape_name": ctx.shape_name,
        "shape_type": "speaker_notes" if ctx.is_notes else ctx.shape_type,
        "placeholder_type": ctx.placeholder_type,
        "table_id": ctx.table_id,
        "table_row_id": ctx.table_row_id,
        "table_cell_id": ctx.table_cell_id,
        "group_ids": [gid for gid, _ in ctx.group_path],
        "group_names": [name for _, name in ctx.group_path],
        "group_depth": len(ctx.group_path),
        **_box_columns(ctx.box),
        "list_num_id": p_props.get("list_num_id"),
        "list_level": p_props.get("list_level"),
        "list_label": list_label,
        "outline_level": p_props.get("outline_level"),
        "text_align": p_props.get("text_align"),
        "is_bold": is_bold,
        "is_italic": is_italic,
        "is_underlined": is_underlined,
        "is_strikethrough": is_strikethrough,
        "script_type": r_props.get("script_type"),
        "bold_ratio": 1.0 if is_bold is True else 0.0,
        "italic_ratio": 1.0 if is_italic is True else 0.0,
        "underlined_ratio": 1.0 if is_underlined is True else 0.0,
        "strikethrough_ratio": 1.0 if is_strikethrough is True else 0.0,
        "font_size": r_props.get("font_size"),
        "font_name": r_props.get("font_name"),
        "non_stroking_color": r_props.get("non_stroking_color"),
        "hyperlink_id": hyperlink_id,
        "hyperlink_url": hyperlink_url,
        "has_link": bool(hyperlink_id),
        "link_type": "external" if hyperlink_url else ("internal" if hyperlink_id else None),
        "link_url": hyperlink_url,
    })


# ---------------------------------------------------------------------------
# Document walkers
# ---------------------------------------------------------------------------

def _walk_txbody(
    txbody: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
    inherited_txbodies: list[etree._Element] | None = None,
    tx_styles_r_props: dict[int, dict[str, Any]] | None = None,
    tx_styles_p_props: dict[int, dict[str, Any]] | None = None,
    theme_fonts: dict[str, str] | None = None,
    default_text_style_r_props: dict[int, dict[str, Any]] | None = None,
) -> None:
    # Merge lstStyle props in inheritance order (least → most specific):
    #   p:txStyles base → master lstStyle → layout lstStyle → slide lstStyle
    # inherited_txbodies is [layout_txbody, master_txbody], so reversed = master first.
    # Per-level merges (not wholesale replace) so a higher lstStyle that only sets
    # e.g. indentation doesn't wipe the bullet inherited from p:txStyles.
    inherited_level_r_props: dict[int, dict[str, Any]] = {
        lvl: dict(props) for lvl, props in (tx_styles_r_props or {}).items()
    }
    list_style_props: dict[int, dict[str, Any]] = {
        lvl: dict(props) for lvl, props in (tx_styles_p_props or {}).items()
    }
    for itxbody in reversed(inherited_txbodies or []):
        for lvl, props in _extract_list_style_props(itxbody).items():
            list_style_props.setdefault(lvl, {}).update(props)
        for lvl, props in _extract_list_style_r_props(itxbody).items():
            inherited_level_r_props.setdefault(lvl, {}).update(props)
    for lvl, props in _extract_list_style_props(txbody).items():
        list_style_props.setdefault(lvl, {}).update(props)
    level_r_props = _extract_list_style_r_props(txbody)
    list_resolver = _PptxListResolver()
    tf = theme_fonts or {}
    default_style_r_props = default_text_style_r_props or {}

    for p_elem in txbody.findall(f"{A}p"):
        counters.paragraph_id += 1
        paragraph_id = counters.paragraph_id
        p_pr = p_elem.find(f"{A}pPr")
        level = _paragraph_level(p_pr)
        p_props = _resolve_p_props(p_elem, ctx, list_style_props)
        default_r_props = _paragraph_default_r_props(
            p_elem,
            level,
            inherited_level_r_props,
            level_r_props,
        )
        default_r_props["font_name"] = _resolve_font_name_with_fallback(
            default_r_props.get("font_name"), tf, default_style_r_props, level
        )
        if p_props.get("list_type") == "auto" and _has_visible_paragraph_text(p_elem):
            p_props["list_label"] = list_resolver.next_label(p_props)

        for child in p_elem:
            if child.tag == f"{A}r":
                r_pr = child.find(f"{A}rPr")
                r_props = _merge_props(default_r_props, _extract_r_props(r_pr))
                # Run's own rPr may also carry a theme font reference; resolve it too.
                r_props["font_name"] = _resolve_font_name_with_fallback(
                    r_props.get("font_name"), tf, default_style_r_props, level
                )
                hyperlink_id, hyperlink_url = _extract_hyperlink(r_pr, ctx, package)
                t_elem = child.find(f"{A}t")
                text = _collapse_spaces((t_elem.text or "") if t_elem is not None else "")
                _append_row(rows, counters, ctx, paragraph_id,
                            "text", text, p_props, r_props, hyperlink_id, hyperlink_url)
            elif child.tag == f"{A}br":
                _append_row(rows, counters, ctx, paragraph_id,
                            "line_break", "\n", p_props, default_r_props, None, None)
            elif child.tag == f"{A}fld":
                t_elem = child.find(f"{A}t")
                text = _collapse_spaces((t_elem.text or "") if t_elem is not None else "")
                _append_row(rows, counters, ctx, paragraph_id,
                            "field_marker", text, p_props, default_r_props, None, None)
            elif child.tag == f"{A14}m":
                # OMML equation (a14:m/m:oMath). The m:t leaves already hold the
                # rendered unicode glyphs (e.g. math-italic alpha), so flatten
                # them into a plain text run inline with the surrounding a:r runs.
                math_text = _collapse_spaces("".join(
                    t.text or "" for t in child.findall(f".//{M}t")
                ))
                if math_text:
                    m_r_pr = child.find(f".//{A}rPr")
                    r_props = _merge_props(default_r_props, _extract_r_props(m_r_pr))
                    r_props["font_name"] = _resolve_font_name_with_fallback(
                        r_props.get("font_name"), tf, default_style_r_props, level
                    )
                    _append_row(rows, counters, ctx, paragraph_id,
                                "math", math_text, p_props, r_props, None, None)


def _table_cell_boxes(tbl: etree._Element, table_box: _Box | None) -> dict[int, _Box]:
    if table_box is None:
        return {}

    col_widths = [
        _emu_to_pt(col.get("w")) or 0.0
        for col in tbl.findall(f"{A}tblGrid/{A}gridCol")
    ]
    row_heights = [
        _emu_to_pt(tr.get("h")) or 0.0
        for tr in tbl.findall(f"{A}tr")
    ]
    if not col_widths or not row_heights:
        return {}

    # Scale declared grid dimensions to the graphic frame if PowerPoint stores
    # slightly stale row/column totals.
    col_total = sum(col_widths)
    row_total = sum(row_heights)
    if col_total > 0 and abs(col_total - table_box.width) > 0.01:
        col_widths = [w * table_box.width / col_total for w in col_widths]
    if row_total > 0 and abs(row_total - table_box.height) > 0.01:
        row_heights = [h * table_box.height / row_total for h in row_heights]

    x_edges = [table_box.x_left]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)

    y_edges = [table_box.y_top]
    for height in row_heights:
        y_edges.append(y_edges[-1] + height)

    boxes: dict[int, _Box] = {}
    cell_counter = 0
    for row_idx, tr in enumerate(tbl.findall(f"{A}tr")):
        col_idx = 0
        for tc in tr.findall(f"{A}tc"):
            cell_counter += 1
            grid_span_raw = tc.get("gridSpan")
            try:
                grid_span = max(1, int(grid_span_raw)) if grid_span_raw else 1
            except ValueError:
                grid_span = 1

            end_col_idx = min(len(x_edges) - 1, col_idx + grid_span)
            if row_idx + 1 < len(y_edges) and col_idx < end_col_idx:
                boxes[cell_counter] = _Box(
                    x_left=x_edges[col_idx],
                    y_top=y_edges[row_idx],
                    x_right=x_edges[end_col_idx],
                    y_bottom=y_edges[row_idx + 1],
                )
            col_idx += grid_span

    return boxes


def _walk_table(
    graphic_frame: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
) -> None:
    tbl = graphic_frame.find(f".//{A}tbl")
    if tbl is None:
        return

    counters.table_id += 1
    table_id = counters.table_id
    cell_boxes = _table_cell_boxes(tbl, ctx.box)
    table_cell_index = 0

    for tr in tbl.findall(f"{A}tr"):
        counters.table_row_id += 1
        row_id = counters.table_row_id
        for tc in tr.findall(f"{A}tc"):
            counters.table_cell_id += 1
            table_cell_index += 1
            cell_id = counters.table_cell_id
            cell_ctx = _Context(
                source_part=ctx.source_part,
                slide_index=ctx.slide_index,
                slide_number=ctx.slide_number,
                shape_id=ctx.shape_id,
                shape_name=ctx.shape_name,
                shape_type="table",
                placeholder_type=None,
                is_notes=ctx.is_notes,
                table_id=table_id,
                table_row_id=row_id,
                table_cell_id=cell_id,
                box=cell_boxes.get(table_cell_index, ctx.box),
                group_path=ctx.group_path,
            )
            txbody = tc.find(f"{A}txBody")
            if txbody is not None:
                _walk_txbody(txbody, cell_ctx, counters, rows, package)


def _group_identity(grp_sp: etree._Element) -> tuple[str, str]:
    """Return (id, name) for a p:grpSp from its p:nvGrpSpPr/p:cNvPr."""
    cnv_pr = grp_sp.find(f"{P}nvGrpSpPr/{P}cNvPr")
    if cnv_pr is None:
        return "", ""
    return cnv_pr.get("id", ""), cnv_pr.get("name", "")


def _walk_sp_tree(
    sp_tree: etree._Element,
    slide: PptxSlide,
    is_notes: bool,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
    group_path: tuple[tuple[str, str], ...] = (),
    transform: _Transform = _IDENTITY_TRANSFORM,
) -> None:
    for child in sp_tree:
        if child.tag == f"{P}sp":
            _walk_sp(child, slide, is_notes, counters, rows, package, group_path, transform)
        elif child.tag == f"{P}graphicFrame":
            _walk_graphic_frame(child, slide, is_notes, counters, rows, package, group_path, transform)
        elif child.tag == f"{P}grpSp":
            # Recurse into shape groups, extending the ancestry path. A p:grpSp
            # holds its children directly (no nested p:spTree), so iterate it.
            # Children's a:off/a:ext are expressed in this group's chOff/chExt
            # space, not slide-absolute EMUs, so compose the transform too.
            grp_sp_pr = child.find(f"{P}grpSpPr")
            child_transform = (
                _child_transform(transform, grp_sp_pr) if grp_sp_pr is not None else transform
            )
            _walk_sp_tree(child, slide, is_notes, counters, rows, package,
                          group_path + (_group_identity(child),), child_transform)
        elif child.tag == f"{P}pic":
            _walk_pic(child, slide, is_notes, counters, rows, package, group_path, transform)
        elif child.tag == f"{MC}AlternateContent":
            # Equation-bearing shapes (a14:m) are wrapped in mc:AlternateContent
            # with a richer mc:Choice (Requires="a14") and a rasterized/blank
            # mc:Fallback for older readers. Prefer Choice; both wrap ordinary
            # p:sp/p:graphicFrame/p:grpSp/p:pic children, so recurse the same way.
            target = child.find(f"{MC}Choice")
            if target is None:
                target = child.find(f"{MC}Fallback")
            if target is not None:
                _walk_sp_tree(target, slide, is_notes, counters, rows, package, group_path, transform)


def _walk_sp(
    sp: etree._Element,
    slide: PptxSlide,
    is_notes: bool,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
    group_path: tuple[tuple[str, str], ...] = (),
    transform: _Transform = _IDENTITY_TRANSFORM,
) -> None:
    nv = sp.find(f"{P}nvSpPr")
    if nv is None:
        return

    cnv_pr = nv.find(f"{P}cNvPr")
    shape_name = cnv_pr.get("name", "") if cnv_pr is not None else ""

    nv_pr = nv.find(f"{P}nvPr")
    ph = nv_pr.find(f"{P}ph") if nv_pr is not None else None
    placeholder_type = ph.get("type") if ph is not None else None

    # Skip decorative / non-content placeholders
    if placeholder_type in _SKIP_PH_TYPES:
        return

    cnv_sp_pr = nv.find(f"{P}cNvSpPr")
    is_txbox = cnv_sp_pr is not None and cnv_sp_pr.get("txBox") == "1"

    if ph is not None:
        shape_type = "placeholder"
    elif is_txbox:
        shape_type = "text_box"
    else:
        shape_type = "shape"

    inherited_sps = _find_placeholder_shapes(slide, package, ph)
    sp_pr = sp.find(f"{P}spPr")
    box = _extract_xfrm_box(sp_pr, transform) if sp_pr is not None else None
    if box is None:
        for isp in inherited_sps:
            isp_pr = isp.find(f"{P}spPr")
            box = _extract_xfrm_box(isp_pr) if isp_pr is not None else None
            if box is not None:
                break
    counters.shape_id += 1
    ctx = _Context(
        source_part=slide.part_name if not is_notes else _notes_part_for(slide, package),
        slide_index=slide.slide_index,
        slide_number=slide.slide_number,
        shape_id=counters.shape_id,
        shape_name=shape_name,
        shape_type=shape_type,
        placeholder_type=placeholder_type,
        is_notes=is_notes,
        box=box,
        group_path=group_path,
    )

    txbody = sp.find(f"{P}txBody")
    inherited_txbodies = [
        isp.find(f"{P}txBody") for isp in inherited_sps
        if isp.find(f"{P}txBody") is not None
    ]
    master_root = package.get_xml(slide.master_part_name) if slide.master_part_name else None
    theme_fonts = _extract_theme_fonts(package.get_xml(slide.theme_part_name) if slide.theme_part_name else None)
    tx_styles_r_props = _extract_tx_styles_r_props(master_root, placeholder_type)
    tx_styles_p_props = _extract_tx_styles_p_props(master_root, placeholder_type)
    default_text_style_r_props = _extract_default_text_style_r_props(
        package.get_xml(_PRESENTATION_PART)
    )
    if txbody is None:
        if shape_type != "placeholder" and box is not None:
            counters.paragraph_id += 1
            _append_row(rows, counters, ctx, counters.paragraph_id,
                        "shape_ref", shape_name, {}, {}, None, None)
        return

    row_count_before = len(rows)
    _walk_txbody(txbody, ctx, counters, rows, package, inherited_txbodies,
                 tx_styles_r_props, tx_styles_p_props, theme_fonts,
                 default_text_style_r_props)
    if len(rows) == row_count_before and shape_type != "placeholder" and box is not None:
        counters.paragraph_id += 1
        _append_row(rows, counters, ctx, counters.paragraph_id,
                    "shape_ref", shape_name, {}, {}, None, None)


def _walk_graphic_frame(
    frame: etree._Element,
    slide: PptxSlide,
    is_notes: bool,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
    group_path: tuple[tuple[str, str], ...] = (),
    transform: _Transform = _IDENTITY_TRANSFORM,
) -> None:
    nv = frame.find(f"{P}nvGraphicFramePr")
    cnv_pr = nv.find(f"{P}cNvPr") if nv is not None else None
    shape_name = cnv_pr.get("name", "") if cnv_pr is not None else ""

    graphic = frame.find(f"{A}graphic")
    graphic_data = graphic.find(f"{A}graphicData") if graphic is not None else None
    uri = graphic_data.get("uri", "") if graphic_data is not None else ""
    chart_elem = graphic_data.find(f"{C}chart") if graphic_data is not None else None

    counters.shape_id += 1
    source_part = slide.part_name if not is_notes else _notes_part_for(slide, package)
    box = _extract_xfrm_box(frame, transform)

    if uri == _TABLE_GRAPHIC_URI:
        ctx = _Context(
            source_part=source_part,
            slide_index=slide.slide_index,
            slide_number=slide.slide_number,
            shape_id=counters.shape_id,
            shape_name=shape_name,
            shape_type="table",
            placeholder_type=None,
            is_notes=is_notes,
            box=box,
            group_path=group_path,
        )
        _walk_table(frame, ctx, counters, rows, package)
    elif chart_elem is not None:
        chart_id = None
        if not is_notes:
            counters.chart_id += 1
            chart_id = counters.chart_id
        counters.paragraph_id += 1
        ctx = _Context(
            source_part=source_part,
            slide_index=slide.slide_index,
            slide_number=slide.slide_number,
            shape_id=counters.shape_id,
            chart_id=chart_id,
            shape_name=shape_name,
            shape_type="chart",
            placeholder_type=None,
            is_notes=is_notes,
            box=box,
            group_path=group_path,
        )
        _append_row(rows, counters, ctx, counters.paragraph_id,
                    "chart_ref", shape_name, {}, {}, None, None)
    else:
        # Other non-text graphic frame: emit a marker row without chart_id.
        counters.paragraph_id += 1
        ctx = _Context(
            source_part=source_part,
            slide_index=slide.slide_index,
            slide_number=slide.slide_number,
            shape_id=counters.shape_id,
            shape_name=shape_name,
            shape_type="graphic",
            placeholder_type=None,
            is_notes=is_notes,
            box=box,
            group_path=group_path,
        )
        _append_row(rows, counters, ctx, counters.paragraph_id,
                    "graphic_ref", shape_name, {}, {}, None, None)


def _walk_pic(
    pic: etree._Element,
    slide: PptxSlide,
    is_notes: bool,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: PptxPackage,
    group_path: tuple[tuple[str, str], ...] = (),
    transform: _Transform = _IDENTITY_TRANSFORM,
) -> None:
    nv = pic.find(f"{P}nvPicPr")
    cnv_pr = nv.find(f"{P}cNvPr") if nv is not None else None
    if cnv_pr is None:
        return
    shape_name = cnv_pr.get("name", "")
    descr = cnv_pr.get("descr") or cnv_pr.get("title") or shape_name

    sp_pr = pic.find(f"{P}spPr")
    counters.shape_id += 1
    counters.paragraph_id += 1
    source_part = slide.part_name if not is_notes else _notes_part_for(slide, package)
    ctx = _Context(
        source_part=source_part,
        slide_index=slide.slide_index,
        slide_number=slide.slide_number,
        shape_id=counters.shape_id,
        shape_name=shape_name,
        shape_type="image",
        placeholder_type=None,
        is_notes=is_notes,
        box=_extract_xfrm_box(sp_pr, transform) if sp_pr is not None else None,
        group_path=group_path,
    )
    _append_row(rows, counters, ctx, counters.paragraph_id,
                "image_ref", descr, {}, {}, None, None)


def _notes_part_for(slide: PptxSlide, package: PptxPackage) -> str:
    """Return the notes slide part name for a given slide, or empty string."""
    rels = package.package.relationships.get(slide.part_name, {})
    for rel in rels.values():
        if rel.rel_type == _REL_NOTES_SLIDE:
            import posixpath
            source_dir = posixpath.dirname(slide.part_name)
            return posixpath.normpath(posixpath.join(source_dir, rel.target))
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_runs(
    package: PptxPackage,
    include_speaker_notes: bool = True,
) -> pd.DataFrame:
    """
    Extract run-level content from a PPTX package.

    Args:
        package: Parsed PPTX package (from step 01).
        include_speaker_notes: Include speaker notes slides.

    Returns:
        DataFrame with one row per text/control/image/chart/shape run event.
        page_number maps to slide_index + 1 so shared downstream steps work
        without modification.
    """
    rows: list[dict[str, Any]] = []
    counters = _Counters()

    for slide in package.slides:
        root = package.get_xml(slide.part_name)
        if root is None:
            continue
        sp_tree = root.find(f".//{P}spTree")
        if sp_tree is not None:
            _walk_sp_tree(sp_tree, slide, False, counters, rows, package)

        if include_speaker_notes:
            notes_part = _notes_part_for(slide, package)
            if notes_part:
                notes_root = package.get_xml(notes_part)
                if notes_root is not None:
                    notes_sp_tree = notes_root.find(f".//{P}spTree")
                    if notes_sp_tree is not None:
                        _walk_sp_tree(notes_sp_tree, slide, True, counters, rows, package)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "non_stroking_color" in df.columns:
        df["non_stroking_color"] = df["non_stroking_color"].fillna("#000000")
    if "text_align" in df.columns:
        df["text_align"] = df["text_align"].fillna("left")
    df = add_calculated_text_features(df)
    return df

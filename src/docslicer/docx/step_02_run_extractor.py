"""
DOCX run extractor.

Builds a run-level dataframe from WordprocessingML. Every output row carries its
paragraph/table/part context so paragraph and table layers can be reconstructed
without re-walking the XML package.
"""

from __future__ import annotations

import re
import posixpath
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from lxml import etree

from .step_01_package_reader import DocxPackage
from .._utils.yaml_compilers.page_label_patterns import PageLabelPatternConfig
from .._utils.text_utils import add_calculated_text_features


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

W = f"{{{NS['w']}}}"
R = f"{{{NS['r']}}}"


_CONTENT_PARTS = (
    ("word/document.xml", "body"),
    ("word/footnotes.xml", "footnote"),
    ("word/endnotes.xml", "endnote"),
    ("word/comments.xml", "comment"),
)


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    run_id: int = 0
    paragraph_id: int = 0
    table_id: int = 0
    table_row_id: int = 0
    table_cell_id: int = 0
    field_id: int = 0
    order_index: int = 0


@dataclass
class _SectionTracker:
    current_section_id: int = 1


@dataclass(frozen=True)
class _Context:
    source_part: str
    source_part_id: str
    header_footer_type: str
    section_id: int | None = None
    table_id: int | None = None
    table_row_id: int | None = None
    table_cell_id: int | None = None
    nested_table_depth: int = 0
    text_orientation: str | None = None
    footnote_id: str | None = None
    endnote_id: str | None = None
    comment_id: str | None = None


@dataclass(frozen=True)
class _SectionInfo:
    section_id: int
    break_type: str | None
    page_number_start: int | None
    page_number_format: str | None
    has_title_page: bool
    has_even_and_odd_headers: bool
    footer_parts: dict[str, str]
    header_parts: dict[str, str]
    page_width: float | None = None
    page_height: float | None = None


@dataclass(frozen=True)
class _StyleDef:
    style_id: str
    style_type: str
    name: str | None
    based_on: str | None
    p_pr: etree._Element | None
    r_pr: etree._Element | None
    is_default: bool


# ---------------------------------------------------------------------------
# Style and numbering resolvers
# ---------------------------------------------------------------------------


class _StyleResolver:
    """Resolve DOCX style inheritance into effective paragraph/run properties."""

    def __init__(self, styles_root: etree._Element | None):
        self.styles_root = styles_root
        self.styles: dict[tuple[str, str], _StyleDef] = {}
        self.default_style_ids: dict[str, str] = {}
        self.doc_default_r_props: dict[str, Any] = {}
        self.doc_default_p_props: dict[str, Any] = {}
        self._p_cache: dict[str | None, dict[str, Any]] = {}
        self._r_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.styles_root is None:
            return

        doc_defaults = self.styles_root.find("w:docDefaults", namespaces=NS)
        r_pr_default = doc_defaults.find("w:rPrDefault/w:rPr", namespaces=NS) if doc_defaults is not None else None
        p_pr_default = doc_defaults.find("w:pPrDefault/w:pPr", namespaces=NS) if doc_defaults is not None else None
        self.doc_default_r_props = _extract_r_pr_props(r_pr_default)
        self.doc_default_p_props = _extract_p_pr_props(p_pr_default)

        for style in self.styles_root.findall("w:style", namespaces=NS):
            style_id = style.get(f"{W}styleId")
            style_type = style.get(f"{W}type")
            if not style_id or not style_type:
                continue
            name = _attr(_child(style, "name"), "val")
            based_on = _attr(_child(style, "basedOn"), "val")
            is_default = _bool_val_attr(style.get(f"{W}default")) is True
            style_def = _StyleDef(
                style_id=style_id,
                style_type=style_type,
                name=name,
                based_on=based_on,
                p_pr=_child(style, "pPr"),
                r_pr=_child(style, "rPr"),
                is_default=is_default,
            )
            self.styles[(style_type, style_id)] = style_def
            if is_default:
                self.default_style_ids[style_type] = style_id

    def style_name(self, style_type: str, style_id: str | None) -> str | None:
        if not style_id:
            return None
        style = self.styles.get((style_type, style_id))
        return style.name if style is not None else style_id

    def resolve_paragraph_props(
        self,
        paragraph_style_id: str | None,
        direct_p_pr: etree._Element | None,
    ) -> dict[str, Any]:
        props = dict(self.doc_default_p_props)
        default_style_id = self.default_style_ids.get("paragraph")
        if default_style_id and default_style_id != paragraph_style_id:
            props.update(self._style_p_props(default_style_id))
        if paragraph_style_id:
            props.update(self._style_p_props(paragraph_style_id))
        props.update(_extract_p_pr_props(direct_p_pr))

        effective_style_id = paragraph_style_id or default_style_id
        props["paragraph_style_id"] = paragraph_style_id
        props["paragraph_style_name"] = self.style_name("paragraph", paragraph_style_id)
        props["effective_paragraph_style_id"] = effective_style_id
        props["effective_paragraph_style_name"] = self.style_name("paragraph", effective_style_id)
        return props

    def resolve_run_props(
        self,
        paragraph_style_id: str | None,
        direct_r_pr: etree._Element | None,
    ) -> dict[str, Any]:
        direct_props = _extract_r_pr_props(direct_r_pr)
        character_style_id = direct_props.pop("character_style_id", None)

        props = dict(self.doc_default_r_props)
        default_char_style_id = self.default_style_ids.get("character")
        if paragraph_style_id:
            props.update(self._style_r_props("paragraph", paragraph_style_id))
        if default_char_style_id and default_char_style_id != character_style_id:
            props.update(self._style_r_props("character", default_char_style_id))
        if character_style_id:
            props.update(self._style_r_props("character", character_style_id))
        props.update(direct_props)

        props["character_style_id"] = character_style_id
        props["character_style_name"] = self.style_name("character", character_style_id)
        props["effective_character_style_id"] = character_style_id or default_char_style_id
        props["effective_character_style_name"] = self.style_name(
            "character",
            character_style_id or default_char_style_id,
        )
        return props

    def _style_p_props(self, style_id: str | None) -> dict[str, Any]:
        if style_id in self._p_cache:
            return self._p_cache[style_id]
        style = self.styles.get(("paragraph", style_id or ""))
        if style is None:
            self._p_cache[style_id] = {}
            return {}
        props: dict[str, Any] = {}
        if style.based_on:
            props.update(self._style_p_props(style.based_on))
        props.update(_extract_p_pr_props(style.p_pr))
        self._p_cache[style_id] = props
        return props

    def _style_r_props(self, style_type: str, style_id: str | None) -> dict[str, Any]:
        cache_key = (style_type, style_id)
        if cache_key in self._r_cache:
            return self._r_cache[cache_key]
        style = self.styles.get((style_type, style_id or ""))
        if style is None:
            self._r_cache[cache_key] = {}
            return {}
        props: dict[str, Any] = {}
        if style.based_on:
            props.update(self._style_r_props(style_type, style.based_on))
        props.update(_extract_r_pr_props(style.r_pr))
        self._r_cache[cache_key] = props
        return props


class _NumberingResolver:
    """Resolve numbering.xml labels and maintain counters while walking paragraphs."""

    def __init__(self, numbering_root: etree._Element | None):
        self.num_to_abstract: dict[str, str] = {}
        self.levels: dict[tuple[str, int], dict[str, Any]] = {}
        self.counters: dict[str, list[int]] = {}
        self._load(numbering_root)

    def _load(self, numbering_root: etree._Element | None) -> None:
        if numbering_root is None:
            return

        for abstract in numbering_root.findall("w:abstractNum", namespaces=NS):
            abstract_id = abstract.get(f"{W}abstractNumId")
            if abstract_id is None:
                continue
            for lvl in abstract.findall("w:lvl", namespaces=NS):
                ilvl_raw = lvl.get(f"{W}ilvl")
                if ilvl_raw is None:
                    continue
                try:
                    ilvl = int(ilvl_raw)
                except ValueError:
                    continue
                start_raw = _attr(_child(lvl, "start"), "val")
                try:
                    start = int(start_raw) if start_raw is not None else 1
                except ValueError:
                    start = 1
                self.levels[(abstract_id, ilvl)] = {
                    "start": start,
                    "num_fmt": _attr(_child(lvl, "numFmt"), "val") or "decimal",
                    "lvl_text": _attr(_child(lvl, "lvlText"), "val") or "",
                }

        for num in numbering_root.findall("w:num", namespaces=NS):
            list_num_id = num.get(f"{W}numId")
            abstract_id = _attr(_child(num, "abstractNumId"), "val")
            if list_num_id is not None and abstract_id is not None:
                self.num_to_abstract[list_num_id] = abstract_id

    def next_label(self, list_num_id: str | None, list_level: str | None) -> str | None:
        if list_num_id is None or list_level is None:
            return None
        abstract_id = self.num_to_abstract.get(str(list_num_id))
        if abstract_id is None:
            return None
        try:
            ilvl = int(list_level)
        except ValueError:
            return None

        counters = self.counters.setdefault(str(list_num_id), [0] * 9)
        lvl_def = self.levels.get((abstract_id, ilvl), {})
        start = int(lvl_def.get("start", 1))
        counters[ilvl] = counters[ilvl] + 1 if counters[ilvl] else start
        for reset_level in range(ilvl + 1, len(counters)):
            counters[reset_level] = 0

        lvl_text = str(lvl_def.get("lvl_text", ""))
        num_fmt = str(lvl_def.get("num_fmt", "decimal"))
        if not lvl_text or num_fmt == "none":
            return None

        def repl(match: re.Match[str]) -> str:
            level = int(match.group(1)) - 1
            if level < 0 or level >= len(counters):
                return match.group(0)
            value = counters[level] or start
            fmt = str(self.levels.get((abstract_id, level), {}).get("num_fmt", "decimal"))
            return _format_number_value(value, fmt)

        label = re.sub(r"%([1-9])", repl, lvl_text).strip()
        return label or None


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _local_name(elem: etree._Element) -> str:
    return etree.QName(elem).localname


def _child(parent: etree._Element | None, name: str) -> etree._Element | None:
    if parent is None:
        return None
    return parent.find(f"w:{name}", namespaces=NS)


def _children(parent: etree._Element | None, name: str) -> list[etree._Element]:
    if parent is None:
        return []
    return list(parent.findall(f"w:{name}", namespaces=NS))


def _attr(elem: etree._Element | None, name: str) -> str | None:
    if elem is None:
        return None
    return elem.get(f"{W}{name}")


def _bool_val(elem: etree._Element | None) -> bool | None:
    if elem is None:
        return None
    val = elem.get(f"{W}val")
    if val is None:
        return True
    return val not in {"0", "false", "False", "off"}


def _bool_val_attr(val: str | None) -> bool | None:
    if val is None:
        return None
    return val not in {"0", "false", "False", "off"}


def _underline_val(elem: etree._Element | None) -> bool | None:
    if elem is None:
        return None
    val = elem.get(f"{W}val")
    if val is None:
        return True
    return val not in {"none", "0", "false", "False", "off"}


def _font_name(font: etree._Element | None) -> str | None:
    if font is None:
        return None
    return (
        font.get(f"{W}ascii")
        or font.get(f"{W}hAnsi")
        or font.get(f"{W}cs")
        or font.get(f"{W}eastAsia")
    )


def _font_size(size: etree._Element | None) -> float | None:
    size_val = _attr(size, "val")
    if not size_val:
        return None
    try:
        return int(size_val) / 2
    except ValueError:
        return None


def _drop_none(props: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in props.items() if value is not None}


def _normalize_text_align(value: str | None) -> str | None:
    if not value:
        return None
    mapping = {
        "both": "justified",
        "distribute": "justified",
        "start": "left",
        "end": "right",
    }
    return mapping.get(value, value)


def _ratio_from_bool(value: Any) -> float:
    return 1.0 if value is True else 0.0


def _extract_p_pr_props(p_pr: etree._Element | None) -> dict[str, Any]:
    p_style = _child(p_pr, "pStyle")
    num_pr = _child(p_pr, "numPr")
    list_num_id = _child(num_pr, "numId")
    ilvl = _child(num_pr, "ilvl")
    outline = _child(p_pr, "outlineLvl")
    page_break_before = _child(p_pr, "pageBreakBefore")
    text_direction = _child(p_pr, "textDirection")
    justification = _child(p_pr, "jc")

    return _drop_none({
        "paragraph_style_id": _attr(p_style, "val"),
        "list_num_id": _attr(list_num_id, "val"),
        "list_level": _attr(ilvl, "val"),
        "outline_level": _attr(outline, "val"),
        "page_break_before": _bool_val(page_break_before),
        "text_orientation": _attr(text_direction, "val"),
        "text_align": _normalize_text_align(_attr(justification, "val")),
    })


def _normalize_color(val: str | None) -> str | None:
    if not val or val.lower() == "auto":
        return None
    hex_str = val.lstrip("#")
    if len(hex_str) == 6:
        return "#" + hex_str.lower()
    return None


def _script_type(elem: etree._Element | None) -> str | None:
    if elem is None:
        return None
    val = elem.get(f"{W}val")
    if val in {"superscript", "subscript"}:
        return val
    return None


def _extract_r_pr_props(r_pr: etree._Element | None) -> dict[str, Any]:
    r_style = _child(r_pr, "rStyle")
    color = _child(r_pr, "color")
    size = _child(r_pr, "sz")
    font = _child(r_pr, "rFonts")

    return _drop_none({
        "character_style_id": _attr(r_style, "val"),
        "is_bold": _bool_val(_child(r_pr, "b")),
        "is_italic": _bool_val(_child(r_pr, "i")),
        "is_underline": _underline_val(_child(r_pr, "u")),
        "is_strikethrough": _bool_val(_child(r_pr, "strike")),
        "script_type": _script_type(_child(r_pr, "vertAlign")),
        "font_size": _font_size(size),
        "font_name": _font_name(font),
        "color": _normalize_color(_attr(color, "val")),
    })


def _paragraph_props(p: etree._Element, style_resolver: _StyleResolver) -> dict[str, Any]:
    p_pr = _child(p, "pPr")
    p_style = _child(p_pr, "pStyle")
    return style_resolver.resolve_paragraph_props(_attr(p_style, "val"), p_pr)


# ---------------------------------------------------------------------------
# Field and section helpers
# ---------------------------------------------------------------------------


def _field_type(instr_text: str | None) -> str | None:
    if not instr_text:
        return None
    stripped = instr_text.strip()
    if not stripped:
        return None
    match = re.match(r"([A-Za-z]+)", stripped)
    return match.group(1).upper() if match else None


def _paragraph_bookmarks(p: etree._Element) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    names: list[str] = []
    for bookmark in p.findall("w:bookmarkStart", namespaces=NS):
        bookmark_id = bookmark.get(f"{W}id")
        bookmark_name = bookmark.get(f"{W}name")
        if bookmark_id is not None:
            ids.append(bookmark_id)
        if bookmark_name is not None:
            names.append(bookmark_name)
    return ids, names


def _paragraph_sect_pr(p: etree._Element) -> etree._Element | None:
    return p.find("w:pPr/w:sectPr", namespaces=NS)


def _section_break_type(sect_pr: etree._Element | None) -> str | None:
    if sect_pr is None:
        return None
    type_elem = _child(sect_pr, "type")
    return _attr(type_elem, "val") or "nextPage"


def _resolve_related_part(source_part: str, target: str | None) -> str | None:
    if not target:
        return None
    if target.startswith("/"):
        return target.lstrip("/")
    source_dir = posixpath.dirname(source_part)
    return posixpath.normpath(posixpath.join(source_dir, target))


def _section_ref_parts(
    package: DocxPackage,
    sect_pr: etree._Element | None,
    ref_name: str,
) -> dict[str, str]:
    if sect_pr is None:
        return {}
    refs: dict[str, str] = {}
    for ref in sect_pr.findall(f"w:{ref_name}", namespaces=NS):
        ref_type = ref.get(f"{W}type") or "default"
        rel_id = ref.get(f"{R}id")
        rel = package.get_relationship("word/document.xml", rel_id)
        if rel is None or rel.is_external:
            continue
        target = _resolve_related_part("word/document.xml", rel.target)
        if target:
            refs[ref_type] = target
    return refs


def _section_info_from_sect_pr(
    package: DocxPackage,
    section_id: int,
    sect_pr: etree._Element | None,
    previous: _SectionInfo | None,
) -> _SectionInfo:
    pg_num_type = _child(sect_pr, "pgNumType")
    start_raw = _attr(pg_num_type, "start")
    try:
        page_number_start = int(start_raw) if start_raw is not None else None
    except ValueError:
        page_number_start = None

    explicit_footer_parts = _section_ref_parts(package, sect_pr, "footerReference")
    explicit_header_parts = _section_ref_parts(package, sect_pr, "headerReference")
    footer_parts = explicit_footer_parts or (dict(previous.footer_parts) if previous is not None else {})
    header_parts = explicit_header_parts or (dict(previous.header_parts) if previous is not None else {})
    settings = package.get_xml("word/settings.xml")

    pg_sz = _child(sect_pr, "pgSz")
    page_width: float | None = None
    page_height: float | None = None
    if pg_sz is not None:
        w_raw = _attr(pg_sz, "w")
        h_raw = _attr(pg_sz, "h")
        try:
            page_width = int(w_raw) / 20 if w_raw is not None else None
        except ValueError:
            pass
        try:
            page_height = int(h_raw) / 20 if h_raw is not None else None
        except ValueError:
            pass
    if page_width is None and previous is not None:
        page_width = previous.page_width
    if page_height is None and previous is not None:
        page_height = previous.page_height

    return _SectionInfo(
        section_id=section_id,
        break_type=_section_break_type(sect_pr),
        page_number_start=page_number_start,
        page_number_format=_attr(pg_num_type, "fmt"),
        has_title_page=_child(sect_pr, "titlePg") is not None,
        has_even_and_odd_headers=(
            settings is not None and settings.find("w:evenAndOddHeaders", namespaces=NS) is not None
        ),
        footer_parts=footer_parts,
        header_parts=header_parts,
        page_width=page_width,
        page_height=page_height,
    )


def _collect_section_infos(package: DocxPackage) -> dict[int, _SectionInfo]:
    root = package.get_xml("word/document.xml")
    if root is None:
        return {}
    body = root.find("w:body", namespaces=NS)
    if body is None:
        return {}

    infos: dict[int, _SectionInfo] = {}
    current_section_id = 1
    previous: _SectionInfo | None = None

    for child in body:
        sect_pr = _paragraph_sect_pr(child) if child.tag == f"{W}p" else None
        if sect_pr is not None:
            info = _section_info_from_sect_pr(package, current_section_id, sect_pr, previous)
            infos[current_section_id] = info
            previous = info
            current_section_id += 1

    body_sect_pr = body.find("w:sectPr", namespaces=NS)
    if body_sect_pr is not None:
        infos[current_section_id] = _section_info_from_sect_pr(
            package,
            current_section_id,
            body_sect_pr,
            previous,
        )
    elif current_section_id not in infos:
        infos[current_section_id] = _section_info_from_sect_pr(
            package,
            current_section_id,
            None,
            previous,
        )

    return infos


def _part_has_page_field(package: DocxPackage, part_name: str | None) -> bool:
    if not part_name:
        return False
    root = package.get_xml(part_name)
    if root is None:
        return False
    for instr in root.findall(".//w:instrText", namespaces=NS):
        if _field_type(instr.text or "") == "PAGE":
            return True
    for fld_simple in root.findall(".//w:fldSimple", namespaces=NS):
        if _field_type(fld_simple.get(f"{W}instr") or "") == "PAGE":
            return True
    return False


def _select_footer_part(
    info: _SectionInfo | None,
    page_number: int,
    section_first_page: int,
) -> tuple[str | None, str | None]:
    if info is None:
        return None, None
    page_in_section = page_number - section_first_page + 1
    if info.has_title_page and page_in_section == 1 and "first" in info.footer_parts:
        return info.footer_parts["first"], "first"
    if info.has_even_and_odd_headers and page_number % 2 == 0 and "even" in info.footer_parts:
        return info.footer_parts["even"], "even"
    if "default" in info.footer_parts:
        return info.footer_parts["default"], "default"
    if info.footer_parts:
        footer_type, footer_part = next(iter(info.footer_parts.items()))
        return footer_part, footer_type
    return None, None


def _select_header_part(
    info: _SectionInfo | None,
    page_number: int,
    section_first_page: int,
) -> tuple[str | None, str | None]:
    if info is None:
        return None, None
    page_in_section = page_number - section_first_page + 1
    if info.has_title_page and page_in_section == 1 and "first" in info.header_parts:
        return info.header_parts["first"], "first"
    if info.has_even_and_odd_headers and page_number % 2 == 0 and "even" in info.header_parts:
        return info.header_parts["even"], "even"
    if "default" in info.header_parts:
        return info.header_parts["default"], "default"
    if info.header_parts:
        header_type, header_part = next(iter(info.header_parts.items()))
        return header_part, header_type
    return None, None


def expand_header_footer_runs(
    run_df: pd.DataFrame,
    package: DocxPackage,
) -> pd.DataFrame:
    """
    Replace raw header/footer rows (one set per XML part) with per-page copies.

    For each body page the appropriate header/footer XML part is selected using
    section rules (first-page, even/odd, default).  Those runs are cloned, given
    the correct page_number/section_id/page_label metadata, and reordered so
    headers precede body content and footers follow it, per page.

    Footnote/endnote/comment rows are appended unchanged at the end.
    """
    if "header_footer_type" not in run_df.columns:
        return run_df

    hf_mask = run_df["header_footer_type"].isin({"header", "footer"})
    if not hf_mask.any():
        return run_df

    hf_df = run_df[hf_mask].copy()
    other_df = run_df[~hf_mask]

    body_mask = other_df["header_footer_type"].eq("body")
    body_df = other_df[body_mask]
    non_body_df = other_df[~body_mask]  # footnotes, endnotes, comments

    # Build per-page metadata from first body run of each page (body rows have
    # correct page_number, section_id, and page_label from prior pipeline steps).
    page_meta: dict[int, dict] = {}
    for page_num, group in body_df.groupby("page_number", sort=True):
        first = group.iloc[0]
        page_meta[int(page_num)] = {
            "page_number": int(page_num),
            "section_id": first.get("section_id"),
            "page_label": first.get("page_label"),
            "page_label_type": first.get("page_label_type"),
            "page_width": first.get("page_width"),
            "page_height": first.get("page_height"),
        }

    if not page_meta:
        return other_df

    section_infos = _collect_section_infos(package)

    # section_first_pages[section_id_int] = min page_number in that section
    section_first_pages: dict[int, int] = {}
    for page_num, meta in page_meta.items():
        sid = meta["section_id"]
        if sid is None or (isinstance(sid, float) and pd.isna(sid)):
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        if sid_int not in section_first_pages or page_num < section_first_pages[sid_int]:
            section_first_pages[sid_int] = page_num

    # Group h/f runs by source_part for fast lookup
    hf_by_part: dict[str, pd.DataFrame] = {
        part: grp for part, grp in hf_df.groupby("source_part", sort=False)
    }

    next_para_id = int(run_df["paragraph_id"].max()) + 1

    def _clone_for_page(part_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
        nonlocal next_para_id
        chunk = part_df.copy()
        orig_ids = sorted(chunk["paragraph_id"].unique())
        id_map = {old: next_para_id + i for i, old in enumerate(orig_ids)}
        chunk["paragraph_id"] = chunk["paragraph_id"].map(id_map)
        next_para_id += len(orig_ids)
        for col, val in meta.items():
            chunk[col] = val
        # Replace static PAGE field rendered values with the actual page label.
        # The rendered value (e.g. "4") is stored as run_type="text" inside a
        # PAGE field (field_type="PAGE"); the correct label comes from body rows.
        if "field_type" in chunk.columns and "run_type" in chunk.columns:
            page_label_val = meta.get("page_label")
            if page_label_val is not None:
                page_field_mask = (chunk["run_type"] == "text") & (chunk["field_type"] == "PAGE")
                if page_field_mask.any():
                    chunk.loc[page_field_mask, "text"] = str(page_label_val)
        return chunk

    page_header_chunks: dict[int, pd.DataFrame] = {}
    page_footer_chunks: dict[int, pd.DataFrame] = {}

    for page_num in sorted(page_meta):
        meta = page_meta[page_num]
        sid = meta["section_id"]
        try:
            sid_int = int(sid) if (sid is not None and not (isinstance(sid, float) and pd.isna(sid))) else None
        except (TypeError, ValueError):
            sid_int = None

        info = section_infos.get(sid_int) if sid_int is not None else None
        first_page = section_first_pages.get(sid_int, page_num) if sid_int is not None else page_num

        header_part, _ = _select_header_part(info, page_num, first_page)
        if header_part and header_part in hf_by_part:
            page_header_chunks[page_num] = _clone_for_page(hf_by_part[header_part], meta)

        footer_part, _ = _select_footer_part(info, page_num, first_page)
        if footer_part and footer_part in hf_by_part:
            page_footer_chunks[page_num] = _clone_for_page(hf_by_part[footer_part], meta)

    # Rebuild: for each page [header_rows][body_rows][footer_rows], then non-body
    ordered: list[pd.DataFrame] = []
    for page_num in sorted(page_meta):
        if page_num in page_header_chunks:
            ordered.append(page_header_chunks[page_num])
        ordered.append(body_df[body_df["page_number"] == page_num])
        if page_num in page_footer_chunks:
            ordered.append(page_footer_chunks[page_num])

    ordered.append(non_body_df)
    return pd.concat(ordered, ignore_index=True)


# ---------------------------------------------------------------------------
# Number and page label formatting
# ---------------------------------------------------------------------------


def _to_roman(value: int) -> str:
    if value <= 0:
        return str(value)
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    remaining = value
    for n, symbol in numerals:
        while remaining >= n:
            out.append(symbol)
            remaining -= n
    return "".join(out)


def _to_alpha(value: int) -> str:
    if value <= 0:
        return str(value)
    chars = []
    n = value
    while n:
        n -= 1
        chars.append(chr(ord("A") + (n % 26)))
        n //= 26
    return "".join(reversed(chars))


def _format_value(value: int, fmt: str | None) -> str:
    fmt_norm = (fmt or "decimal").strip()
    if fmt_norm == "lowerRoman":
        return _to_roman(value).lower()
    if fmt_norm == "upperRoman":
        return _to_roman(value)
    if fmt_norm == "lowerLetter":
        return _to_alpha(value).lower()
    if fmt_norm == "upperLetter":
        return _to_alpha(value)
    return str(value)


def _format_page_label(value: int, fmt: str | None) -> str:
    if (fmt or "").strip() == "decimalZero":
        return f"{value:02d}"
    return _format_value(value, fmt)


def _format_number_value(value: int, fmt: str | None) -> str:
    if (fmt or "").strip() == "bullet":
        return ""
    return _format_value(value, fmt)


def _classify_page_label(label: str, config: PageLabelPatternConfig) -> str:
    for p in config.patterns:
        if p.compiled.fullmatch(label):
            return p.name
    return "unknown"


def _add_page_labels_from_sections(
    run_df: pd.DataFrame,
    package: DocxPackage,
    page_label_config: PageLabelPatternConfig | None = None,
) -> pd.DataFrame:
    if run_df.empty or "section_id" not in run_df.columns or "page_number" not in run_df.columns:
        return run_df

    section_infos = _collect_section_infos(package)
    out = run_df.copy()
    out["page_label"] = None
    out["page_label_type"] = None
    out["page_label_format"] = None
    out["page_label_footer_part"] = None
    out["page_label_footer_type"] = None
    out["page_label_source"] = None
    out["page_width"] = out["section_id"].map(
        {sid: info.page_width for sid, info in section_infos.items()}
    )
    out["page_height"] = out["section_id"].map(
        {sid: info.page_height for sid, info in section_infos.items()}
    )

    body_mask = out["header_footer_type"].eq("body") & out["section_id"].notna()
    if not body_mask.any():
        return out

    section_first_pages = (
        out.loc[body_mask]
        .groupby("section_id")["page_number"]
        .min()
        .to_dict()
    )
    section_last_pages = (
        out.loc[body_mask]
        .groupby("section_id")["page_number"]
        .max()
        .to_dict()
    )
    section_label_starts: dict[int, int] = {}
    next_continued_label: int | None = None
    for raw_section_id in sorted(section_first_pages, key=lambda x: int(x)):
        section_id = int(raw_section_id)
        first_page = int(section_first_pages[raw_section_id])
        last_page = int(section_last_pages[raw_section_id])
        info = section_infos.get(section_id)
        if info is not None and info.page_number_start is not None:
            label_start = info.page_number_start
        elif next_continued_label is not None:
            label_start = next_continued_label
        else:
            label_start = first_page
        section_label_starts[section_id] = label_start
        next_continued_label = label_start + max(1, last_page - first_page + 1)

    footer_page_field_cache: dict[str, bool] = {}

    for idx, row in out.loc[body_mask].iterrows():
        section_id = int(row["section_id"])
        page_number = int(row["page_number"])
        info = section_infos.get(section_id)
        section_first_page = int(section_first_pages.get(row["section_id"], page_number))
        footer_part, footer_type = _select_footer_part(info, page_number, section_first_page)

        out.at[idx, "page_label_footer_part"] = footer_part
        out.at[idx, "page_label_footer_type"] = footer_type
        if info is not None:
            out.at[idx, "page_label_format"] = info.page_number_format or "decimal"

        if not footer_part:
            continue
        if footer_part not in footer_page_field_cache:
            footer_page_field_cache[footer_part] = _part_has_page_field(package, footer_part)
        if not footer_page_field_cache[footer_part]:
            continue

        label_number = section_label_starts.get(section_id, page_number) + (
            page_number - section_first_page
        )

        label = _format_page_label(
            label_number,
            info.page_number_format if info is not None else None,
        )
        out.at[idx, "page_label"] = label
        if page_label_config is not None:
            out.at[idx, "page_label_type"] = _classify_page_label(label, page_label_config)
        out.at[idx, "page_label_source"] = "section_footer_page_field"

    return out


def _assign_page_numbers(run_df: pd.DataFrame) -> pd.Series:
    """
    Page numbers from explicit and rendered page breaks.

    `w:lastRenderedPageBreak` is useful when Word cached layout but can duplicate
    an immediately preceding explicit `w:br w:type="page"`. Count rendered
    breaks unless there has been no visible content since the last counted
    explicit break.
    """
    page_numbers: list[int] = []
    page_number = 1
    content_since_break = True

    for row in run_df.itertuples(index=False):
        run_type = getattr(row, "run_type")
        text = getattr(row, "text", "")
        has_visible_content = run_type not in {
            "page_break",
            "rendered_page_break",
            "section_break",
            "field_marker",
        } and bool(str(text).strip())

        if run_type == "page_break":
            page_number += 1
            content_since_break = False
        elif run_type == "rendered_page_break":
            if content_since_break:
                page_number += 1
            content_since_break = False

        page_numbers.append(page_number)

        if has_visible_content:
            content_since_break = True

    return pd.Series(page_numbers, index=run_df.index, dtype="int64")


# ---------------------------------------------------------------------------
# Content part discovery and run events
# ---------------------------------------------------------------------------


def _part_kind(part_name: str) -> str:
    if "/header" in part_name:
        return "header"
    if "/footer" in part_name:
        return "footer"
    if part_name.endswith("footnotes.xml"):
        return "footnote"
    if part_name.endswith("endnotes.xml"):
        return "endnote"
    if part_name.endswith("comments.xml"):
        return "comment"
    return "body"


def _content_part_specs(package: DocxPackage) -> list[tuple[str, str, str | None]]:
    specs: list[tuple[str, str, str | None]] = []
    for part_name, part_type in _CONTENT_PARTS:
        if package.get_xml(part_name) is not None:
            specs.append((part_name, part_type, None))

    for part_name in package.part_names:
        norm = part_name.lstrip("/")
        if norm.startswith("word/header") and norm.endswith(".xml"):
            specs.append((norm, "header", None))
        elif norm.startswith("word/footer") and norm.endswith(".xml"):
            specs.append((norm, "footer", None))

    return specs


def _iter_part_roots(root: etree._Element, part_type: str) -> list[tuple[etree._Element, str | None]]:
    if part_type == "footnote":
        return [(node, node.get(f"{W}id")) for node in root.findall("w:footnote", namespaces=NS)]
    if part_type == "endnote":
        return [(node, node.get(f"{W}id")) for node in root.findall("w:endnote", namespaces=NS)]
    if part_type == "comment":
        return [(node, node.get(f"{W}id")) for node in root.findall("w:comment", namespaces=NS)]
    body = root.find("w:body", namespaces=NS)
    return [(body if body is not None else root, None)]


def _extract_hyperlink_context(
    package: DocxPackage,
    source_part: str,
    ancestors: list[etree._Element],
) -> tuple[str | None, str | None, str | None]:
    for ancestor in reversed(ancestors):
        if ancestor.tag != f"{W}hyperlink":
            continue
        rel_id = ancestor.get(f"{R}id")
        anchor = ancestor.get(f"{W}anchor")
        rel = package.get_relationship(source_part, rel_id)
        url = rel.target if rel is not None else None
        return rel_id, url, anchor
    return None, None, None


def _image_alt_text(elem: etree._Element) -> str:
    doc_pr = elem.find(".//wp:docPr", namespaces=NS)
    if doc_pr is not None:
        descr = doc_pr.get("descr")
        title = doc_pr.get("title")
        if descr:
            return descr
        if title:
            return title

    for node in elem.iter():
        for attr_name in ("alt", "title"):
            val = node.get(attr_name)
            if val:
                return val
    return ""


def _run_events(r: etree._Element) -> list[tuple[str, str, etree._Element | None]]:
    events: list[tuple[str, str, etree._Element | None]] = []
    for child in r:
        if child.tag == f"{W}t":
            events.append(("text", child.text or "", child))
        elif child.tag == f"{W}tab":
            events.append(("tab", "\t", child))
        elif child.tag == f"{W}br":
            br_type = child.get(f"{W}type")
            events.append(("page_break" if br_type == "page" else "line_break", "\n", child))
        elif child.tag == f"{W}cr":
            events.append(("line_break", "\n", child))
        elif child.tag == f"{W}lastRenderedPageBreak":
            events.append(("rendered_page_break", "\n", child))
        elif child.tag == f"{W}instrText":
            events.append(("field_code", child.text or "", child))
        elif child.tag == f"{W}fldChar":
            fld_type = child.get(f"{W}fldCharType")
            events.append(("field_marker", fld_type or "", child))
        elif child.tag in {f"{W}drawing", f"{W}pict"}:
            events.append(("image_ref", _image_alt_text(child), child))
        elif child.tag == f"{W}footnoteReference":
            fn_id = child.get(f"{W}id")
            if fn_id is not None:
                events.append(("footnote_reference", fn_id, child))
        elif child.tag == f"{W}endnoteReference":
            en_id = child.get(f"{W}id")
            if en_id is not None:
                events.append(("endnote_reference", en_id, child))
    return events


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _append_run_row(
    rows: list[dict[str, Any]],
    counters: _Counters,
    ctx: _Context,
    paragraph_id: int,
    run_index: int,
    r: etree._Element,
    event: tuple[str, str, etree._Element | None],
    p_props: dict[str, Any],
    field_state: dict[str, Any],
    package: DocxPackage,
    style_resolver: _StyleResolver,
    ancestors: list[etree._Element],
) -> None:
    run_type, text, event_elem = event
    list_label = p_props.get("list_label")
    if (
        run_type == "text"
        and list_label
        and not p_props.get("_list_label_applied")
        and str(text).strip()
    ):
        label_prefix = f"{list_label} "
        if not str(text).lstrip().startswith(str(list_label)):
            text = label_prefix + text
        p_props["_list_label_applied"] = True

    if run_type == "field_marker":
        if text == "begin":
            counters.field_id += 1
            field_state["field_id"] = counters.field_id
            field_state["phase"] = "begin"
            field_state["field_type"] = None
        elif text == "separate":
            field_state["phase"] = "result"
        elif text == "end":
            field_state["phase"] = "end"

    if run_type == "field_code":
        parsed_type = _field_type(text)
        if parsed_type:
            field_state["field_type"] = parsed_type
        if field_state.get("phase") in {None, "begin"}:
            field_state["phase"] = "instr"

    hyperlink_id, hyperlink_url, bookmark_id = _extract_hyperlink_context(
        package,
        ctx.source_part,
        ancestors,
    )
    r_props = style_resolver.resolve_run_props(
        p_props.get("effective_paragraph_style_id") or p_props.get("paragraph_style_id"),
        _child(r, "rPr"),
    )
    character_style_id = r_props.get("character_style_id")
    paragraph_style_id = p_props.get("paragraph_style_id")
    style_id = character_style_id or paragraph_style_id
    style_name = (
        r_props.get("character_style_name")
        or p_props.get("paragraph_style_name")
        or style_id
    )

    counters.run_id += 1
    counters.order_index += 1
    rows.append(
        {
            "run_id": counters.run_id,
            "paragraph_id": paragraph_id,
            "source_part": ctx.source_part,
            "source_part_id": ctx.source_part_id,
            "order_index": counters.order_index,
            "run_index": run_index,
            "text": text,
            "run_type": run_type,
            "section_id": ctx.section_id,
            "text_orientation": p_props.get("text_orientation") or ctx.text_orientation,
            "text_align": p_props.get("text_align"),
            "section_break_after": bool(p_props.get("section_break_after")),
            "section_break_type": p_props.get("section_break_type"),
            "header_footer_type": ctx.header_footer_type,
            "table_id": ctx.table_id,
            "table_row_id": ctx.table_row_id,
            "table_cell_id": ctx.table_cell_id,
            "nested_table_depth": ctx.nested_table_depth,
            "style_id": style_id,
            "style_name": style_name,
            "paragraph_style_id": paragraph_style_id,
            "paragraph_style_name": p_props.get("paragraph_style_name"),
            "effective_paragraph_style_id": p_props.get("effective_paragraph_style_id"),
            "effective_paragraph_style_name": p_props.get("effective_paragraph_style_name"),
            "character_style_id": character_style_id,
            "character_style_name": r_props.get("character_style_name"),
            "effective_character_style_id": r_props.get("effective_character_style_id"),
            "effective_character_style_name": r_props.get("effective_character_style_name"),
            "is_bold": r_props.get("is_bold"),
            "is_italic": r_props.get("is_italic"),
            "is_underline": r_props.get("is_underline"),
            "is_strikethrough": r_props.get("is_strikethrough"),
            "script_type": r_props.get("script_type"),
            "bold_ratio": _ratio_from_bool(r_props.get("is_bold")),
            "italic_ratio": _ratio_from_bool(r_props.get("is_italic")),
            "underlined_ratio": _ratio_from_bool(r_props.get("is_underline")),
            "font_size": r_props.get("font_size"),
            "font_name": r_props.get("font_name"),
            "non_stroking_color": r_props.get("color"),
            "hyperlink_id": hyperlink_id,
            "hyperlink_url": hyperlink_url,
            "bookmark_id": bookmark_id,
            "bookmark_ids": p_props.get("bookmark_ids"),
            "bookmark_names": p_props.get("bookmark_names"),
            "comment_id": ctx.comment_id,
            "footnote_id": ctx.footnote_id,
            "endnote_id": ctx.endnote_id,
            "field_id": field_state.get("field_id"),
            "field_type": field_state.get("field_type"),
            "field_phase": field_state.get("phase"),
            "has_link": bool(hyperlink_id or p_props.get("bookmark_ids")),
            "link_type": "external" if hyperlink_url else ("internal" if hyperlink_id or p_props.get("bookmark_ids") else None),
            "is_deleted_revision": any(a.tag == f"{W}del" for a in ancestors),
            "is_inserted_revision": any(a.tag == f"{W}ins" for a in ancestors),
            "list_num_id": p_props.get("list_num_id"),
            "list_level": p_props.get("list_level"),
            "list_label": list_label,
            "outline_level": p_props.get("outline_level"),
            "page_break_before": bool(p_props.get("page_break_before")),
            "event_tag": _local_name(event_elem) if event_elem is not None else None,
        }
    )

    if run_type == "field_marker" and text == "end":
        field_state.clear()


def _append_section_break_row(
    rows: list[dict[str, Any]],
    counters: _Counters,
    ctx: _Context,
    paragraph_id: int,
    p_props: dict[str, Any],
) -> None:
    counters.run_id += 1
    counters.order_index += 1
    paragraph_style_id = p_props.get("paragraph_style_id")
    rows.append(
        {
            "run_id": counters.run_id,
            "paragraph_id": paragraph_id,
            "source_part": ctx.source_part,
            "source_part_id": ctx.source_part_id,
            "order_index": counters.order_index,
            "run_index": 0,
            "text": "",
            "run_type": "section_break",
            "section_id": ctx.section_id,
            "text_orientation": p_props.get("text_orientation") or ctx.text_orientation,
            "text_align": p_props.get("text_align"),
            "section_break_after": True,
            "section_break_type": p_props.get("section_break_type"),
            "header_footer_type": ctx.header_footer_type,
            "table_id": ctx.table_id,
            "table_row_id": ctx.table_row_id,
            "table_cell_id": ctx.table_cell_id,
            "nested_table_depth": ctx.nested_table_depth,
            "style_id": paragraph_style_id,
            "style_name": p_props.get("paragraph_style_name") or paragraph_style_id,
            "paragraph_style_id": paragraph_style_id,
            "paragraph_style_name": p_props.get("paragraph_style_name"),
            "effective_paragraph_style_id": p_props.get("effective_paragraph_style_id"),
            "effective_paragraph_style_name": p_props.get("effective_paragraph_style_name"),
            "character_style_id": None,
            "character_style_name": None,
            "effective_character_style_id": None,
            "effective_character_style_name": None,
            "is_bold": None,
            "is_italic": None,
            "is_underline": None,
            "is_strikethrough": None,
            "script_type": None,
            "bold_ratio": 0.0,
            "italic_ratio": 0.0,
            "underlined_ratio": 0.0,
            "font_size": None,
            "font_name": None,
            "non_stroking_color": None,
            "hyperlink_id": None,
            "hyperlink_url": None,
            "bookmark_id": None,
            "bookmark_ids": p_props.get("bookmark_ids"),
            "bookmark_names": p_props.get("bookmark_names"),
            "comment_id": ctx.comment_id,
            "footnote_id": ctx.footnote_id,
            "endnote_id": ctx.endnote_id,
            "field_id": None,
            "field_type": None,
            "field_phase": None,
            "has_link": bool(p_props.get("bookmark_ids")),
            "link_type": "internal" if p_props.get("bookmark_ids") else None,
            "is_deleted_revision": False,
            "is_inserted_revision": False,
            "list_num_id": p_props.get("list_num_id"),
            "list_level": p_props.get("list_level"),
            "list_label": p_props.get("list_label"),
            "outline_level": p_props.get("outline_level"),
            "page_break_before": bool(p_props.get("page_break_before")),
            "event_tag": "sectPr",
        }
    )


# ---------------------------------------------------------------------------
# Document walkers
# ---------------------------------------------------------------------------


def _walk_paragraph(
    p: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: DocxPackage,
    style_resolver: _StyleResolver,
    numbering_resolver: _NumberingResolver,
    ancestors: list[etree._Element],
) -> None:
    counters.paragraph_id += 1
    paragraph_id = counters.paragraph_id
    p_props = _paragraph_props(p, style_resolver)
    bookmark_ids, bookmark_names = _paragraph_bookmarks(p)
    sect_pr = _paragraph_sect_pr(p)
    p_props["bookmark_ids"] = bookmark_ids
    p_props["bookmark_names"] = bookmark_names
    p_props["section_break_after"] = sect_pr is not None
    p_props["section_break_type"] = _section_break_type(sect_pr)
    if p_props.get("list_num_id") is not None and p_props.get("list_level") is None:
        p_props["list_level"] = "0"
    p_props["list_label"] = numbering_resolver.next_label(
        p_props.get("list_num_id"),
        p_props.get("list_level"),
    )
    field_state: dict[str, Any] = {}
    run_index = 0
    row_count_before = len(rows)

    for node in p.iter():
        if node is p:
            continue
        if node.tag != f"{W}r":
            continue
        run_index += 1
        node_ancestors = list(ancestors)
        parent = node.getparent()
        while parent is not None and parent is not p:
            node_ancestors.append(parent)
            parent = parent.getparent()
        for event in _run_events(node):
            _append_run_row(
                rows,
                counters,
                ctx,
                paragraph_id,
                run_index,
                node,
                event,
                p_props,
                field_state,
                package,
                style_resolver,
                node_ancestors,
            )

    if sect_pr is not None and len(rows) == row_count_before:
        _append_section_break_row(rows, counters, ctx, paragraph_id, p_props)


def _walk_table(
    tbl: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: DocxPackage,
    style_resolver: _StyleResolver,
    numbering_resolver: _NumberingResolver,
    section_tracker: _SectionTracker,
    ancestors: list[etree._Element],
) -> None:
    counters.table_id += 1
    table_id = counters.table_id

    for tr in tbl.findall("w:tr", namespaces=NS):
        counters.table_row_id += 1
        row_id = counters.table_row_id
        for tc in tr.findall("w:tc", namespaces=NS):
            counters.table_cell_id += 1
            cell_id = counters.table_cell_id
            tc_pr = _child(tc, "tcPr")
            text_direction = _child(tc_pr, "textDirection")
            cell_ctx = _Context(
                source_part=ctx.source_part,
                source_part_id=ctx.source_part_id,
                header_footer_type=ctx.header_footer_type,
                section_id=ctx.section_id,
                table_id=table_id,
                table_row_id=row_id,
                table_cell_id=cell_id,
                nested_table_depth=ctx.nested_table_depth + 1,
                text_orientation=_attr(text_direction, "val") or ctx.text_orientation,
                footnote_id=ctx.footnote_id,
                endnote_id=ctx.endnote_id,
                comment_id=ctx.comment_id,
            )
            _walk_container(
                tc,
                cell_ctx,
                counters,
                rows,
                package,
                style_resolver,
                numbering_resolver,
                section_tracker,
                ancestors + [tbl, tr, tc],
            )


def _walk_sdt(
    sdt: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: DocxPackage,
    style_resolver: _StyleResolver,
    numbering_resolver: _NumberingResolver,
    section_tracker: _SectionTracker,
    ancestors: list[etree._Element],
) -> None:
    content = sdt.find("w:sdtContent", namespaces=NS)
    if content is not None:
        _walk_container(
            content,
            ctx,
            counters,
            rows,
            package,
            style_resolver,
            numbering_resolver,
            section_tracker,
            ancestors + [sdt],
        )


def _walk_container(
    container: etree._Element,
    ctx: _Context,
    counters: _Counters,
    rows: list[dict[str, Any]],
    package: DocxPackage,
    style_resolver: _StyleResolver,
    numbering_resolver: _NumberingResolver,
    section_tracker: _SectionTracker,
    ancestors: list[etree._Element] | None = None,
) -> None:
    ancestors = ancestors or []
    for child in container:
        child_ctx = ctx
        if ctx.header_footer_type == "body":
            child_ctx = replace(ctx, section_id=section_tracker.current_section_id)

        if child.tag == f"{W}p":
            _walk_paragraph(
                child,
                child_ctx,
                counters,
                rows,
                package,
                style_resolver,
                numbering_resolver,
                ancestors,
            )
            if child_ctx.nested_table_depth == 0 and _paragraph_sect_pr(child) is not None:
                section_tracker.current_section_id += 1
        elif child.tag == f"{W}tbl":
            _walk_table(
                child,
                child_ctx,
                counters,
                rows,
                package,
                style_resolver,
                numbering_resolver,
                section_tracker,
                ancestors,
            )
        elif child.tag == f"{W}sdt":
            _walk_sdt(
                child,
                child_ctx,
                counters,
                rows,
                package,
                style_resolver,
                numbering_resolver,
                section_tracker,
                ancestors,
            )


# ---------------------------------------------------------------------------
# Footnote and endnote inlining
# ---------------------------------------------------------------------------


def _inline_footnotes(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder footnote and endnote content rows to appear immediately after their
    in-body reference markers, inheriting the reference's page context columns.

    footnote_reference / endnote_reference rows carry the referenced id in ``text``.
    Content rows from footnotes.xml / endnotes.xml carry the matching id in
    ``footnote_id`` / ``endnote_id``.

    The first text run of each footnote/endnote is prefixed with a bracketed
    label ("[Footnote] " / "[Endnote] ") so RAG pipelines and LLMs can
    unambiguously identify inline footnote content.

    Special DOCX separator footnotes (id <= 0) have no body reference and are dropped.
    """
    fn_ref_mask = run_df["run_type"] == "footnote_reference"
    en_ref_mask = run_df["run_type"] == "endnote_reference"
    if not fn_ref_mask.any() and not en_ref_mask.any():
        return run_df

    # Build maps: id_str → (ref_order_index, page_number, page_label, section_id, page_width, page_height)
    def _build_ref_map(mask: pd.Series) -> dict[str, tuple]:
        m: dict[str, tuple] = {}
        for _, row in run_df[mask].iterrows():
            m[str(row["text"])] = (
                float(row["order_index"]),
                row["page_number"],
                row.get("page_label"),
                row.get("section_id"),
                row.get("page_width"),
                row.get("page_height"),
            )
        return m

    fn_ref_map = _build_ref_map(fn_ref_mask)
    en_ref_map = _build_ref_map(en_ref_mask)

    # Drop DOCX special separator footnotes/endnotes (id <= 0, e.g. "-1", "0")
    fn_part_mask = run_df["source_part"].str.endswith("footnotes.xml", na=False)
    en_part_mask = run_df["source_part"].str.endswith("endnotes.xml", na=False)

    def _special_id_mask(part_mask: pd.Series, id_col: str) -> pd.Series:
        numeric = pd.to_numeric(run_df[id_col], errors="coerce")
        return part_mask & (numeric.fillna(1) <= 0)

    drop_mask = (
        _special_id_mask(fn_part_mask, "footnote_id")
        | _special_id_mask(en_part_mask, "endnote_id")
    )
    df = run_df[~drop_mask].reset_index(drop=True).copy()

    # Recompute part masks on trimmed df
    fn_part_mask = df["source_part"].str.endswith("footnotes.xml", na=False)
    en_part_mask = df["source_part"].str.endswith("endnotes.xml", na=False)

    has_page_label = "page_label" in df.columns
    has_section_id = "section_id" in df.columns
    has_page_dims = "page_width" in df.columns and "page_height" in df.columns

    df["_sort_key"] = df["order_index"].astype(float)

    def _assign_order(part_mask: pd.Series, id_col: str, ref_map: dict, label: str) -> None:
        if not part_mask.any():
            return
        id_series = df[id_col].astype(str)
        for ref_id, (ref_idx, ref_page, ref_pg_label, ref_section, ref_pw, ref_ph) in ref_map.items():
            positions = df.index[part_mask & (id_series == ref_id)]
            if positions.empty:
                continue
            n = len(positions)
            offsets = pd.Series(range(1, n + 1), index=positions, dtype=float)

            df.loc[positions, "_sort_key"]  = ref_idx + offsets / (n + 1)
            df.loc[positions, "page_number"] = ref_page
            if has_page_label:
                df.loc[positions, "page_label"] = ref_pg_label
            if has_section_id:
                df.loc[positions, "section_id"] = ref_section
            if has_page_dims:
                df.loc[positions, "page_width"]  = ref_pw
                df.loc[positions, "page_height"] = ref_ph

            # Prefix the first text run so LLMs/RAG identify it as a footnote.
            text_positions = df.index[part_mask & (id_series == ref_id) & (df["run_type"] == "text")]
            if not text_positions.empty:
                first = text_positions[0]
                df.at[first, "text"] = label + df.at[first, "text"]

    _assign_order(fn_part_mask, "footnote_id", fn_ref_map, "[Footnote] ")
    _assign_order(en_part_mask, "endnote_id", en_ref_map, "[Endnote] ")

    df = df.sort_values("_sort_key", kind="stable").reset_index(drop=True)
    df = df.drop(columns=["_sort_key"])
    df["order_index"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_runs(
    package: DocxPackage,
    include_headers_footers: bool = True,
    include_notes_comments: bool = True,
    page_label_config: PageLabelPatternConfig | None = None,
) -> pd.DataFrame:
    """
    Extract run-level content from a DOCX package.

    Args:
        package: Parsed DOCX package.
        include_headers_footers: Include word/header*.xml and word/footer*.xml.
        include_notes_comments: Include footnotes, endnotes, and comments.
        page_label_config: Compiled page label patterns; when provided, populates page_label_type.

    Returns:
        DataFrame with one row per text/control/image/field run event.
    """
    rows: list[dict[str, Any]] = []
    counters = _Counters()
    section_tracker = _SectionTracker()
    style_resolver = _StyleResolver(package.get_xml("word/styles.xml"))
    numbering_resolver = _NumberingResolver(package.get_xml("word/numbering.xml"))

    for part_name, part_type, part_item_id in _content_part_specs(package):
        if part_type in {"header", "footer"} and not include_headers_footers:
            continue
        if part_type in {"footnote", "endnote", "comment"} and not include_notes_comments:
            continue

        root = package.get_xml(part_name)
        if root is None:
            continue

        for item_root, item_id in _iter_part_roots(root, part_type):
            ctx = _Context(
                source_part=part_name,
                source_part_id=item_id or part_item_id or part_name,
                header_footer_type=_part_kind(part_name),
                footnote_id=item_id if part_type == "footnote" else None,
                endnote_id=item_id if part_type == "endnote" else None,
                comment_id=item_id if part_type == "comment" else None,
            )
            _walk_container(
                item_root,
                ctx,
                counters,
                rows,
                package,
                style_resolver,
                numbering_resolver,
                section_tracker,
            )

    run_df = pd.DataFrame(rows)
    if not run_df.empty:
        run_df["page_number"] = _assign_page_numbers(run_df)
        run_df = _add_page_labels_from_sections(run_df, package, page_label_config)
        run_df = _inline_footnotes(run_df)
        if "text_orientation" in run_df.columns:
            run_df["text_orientation"] = run_df["text_orientation"].fillna("LTR")
        if "text_align" in run_df.columns:
            run_df["text_align"] = run_df["text_align"].fillna("left")
        if "non_stroking_color" in run_df.columns:
            run_df["non_stroking_color"] = run_df["non_stroking_color"].fillna("#000000")
        leading_cols = [col for col in ("page_number", "page_label") if col in run_df.columns]
        run_df = run_df[leading_cols + [col for col in run_df.columns if col not in leading_cols]]
        run_df = add_calculated_text_features(run_df)

    return run_df

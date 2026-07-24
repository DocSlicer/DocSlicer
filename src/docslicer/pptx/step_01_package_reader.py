# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
PPTX package reader.

Wraps OxmlPackage with PPTX-specific discovery: an ordered slide list and
the resolved layout → master → theme chain for each slide, so downstream
extractors can access style inheritance without re-walking relationships.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from lxml import etree

from .._utils.oxm_package import OxmlPackage, OxmlRelationship, read_oxm_package


# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

_REL_SLIDE = f"{_R_NS}/slide"
_REL_SLIDE_LAYOUT = f"{_R_NS}/slideLayout"
_REL_SLIDE_MASTER = f"{_R_NS}/slideMaster"
_REL_THEME = f"{_R_NS}/theme"

_PRESENTATION_PART = "ppt/presentation.xml"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PptxSlide:
    """One slide with its resolved style chain."""

    slide_index: int           # 0-based position in presentation order
    slide_number: int          # 1-based display number
    part_name: str             # e.g. "ppt/slides/slide1.xml"
    layout_part_name: str | None
    master_part_name: str | None
    theme_part_name: str | None


@dataclass
class PptxPackage:
    """
    Parsed PPTX package with slide-level discovery on top of OxmlPackage.

    Delegates XML and relationship access to the underlying OxmlPackage and
    adds an ordered slide list with pre-resolved layout → master → theme chains.
    """

    package: OxmlPackage
    slides: list[PptxSlide]

    def get_xml(self, part_name: str) -> etree._Element | None:
        return self.package.get_xml(part_name)

    def get_relationship(
        self,
        source_part: str,
        rel_id: str | None,
    ) -> OxmlRelationship | None:
        return self.package.get_relationship(source_part, rel_id)

    def relationship_target(self, source_part: str, rel_id: str | None) -> str | None:
        return self.package.relationship_target(source_part, rel_id)

    @property
    def part_names(self) -> list[str]:
        return self.package.part_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_part(source_part: str, target: str | None) -> str | None:
    """Resolve a relationship target path relative to its source part."""
    if not target:
        return None
    if target.startswith("/"):
        return target.lstrip("/")
    source_dir = posixpath.dirname(source_part)
    return posixpath.normpath(posixpath.join(source_dir, target))


def _first_rel_target_by_type(
    package: OxmlPackage,
    source_part: str,
    rel_type: str,
) -> str | None:
    """Return the resolved part name for the first relationship of a given type."""
    rels = package.relationships.get(source_part, {})
    for rel in rels.values():
        if rel.rel_type == rel_type:
            return _resolve_part(source_part, rel.target)
    return None


def _build_slides(package: OxmlPackage) -> list[PptxSlide]:
    prs = package.get_xml(_PRESENTATION_PART)
    if prs is None:
        return []

    sld_id_lst = prs.find(f"{{{_P_NS}}}sldIdLst")
    if sld_id_lst is None:
        return []

    slides: list[PptxSlide] = []
    for slide_index, sld_id in enumerate(sld_id_lst.findall(f"{{{_P_NS}}}sldId")):
        rel_id = sld_id.get(f"{{{_R_NS}}}id")
        if not rel_id:
            continue

        raw_target = package.relationship_target(_PRESENTATION_PART, rel_id)
        slide_part = _resolve_part(_PRESENTATION_PART, raw_target)
        if not slide_part:
            continue

        layout_part = _first_rel_target_by_type(package, slide_part, _REL_SLIDE_LAYOUT)
        master_part = (
            _first_rel_target_by_type(package, layout_part, _REL_SLIDE_MASTER)
            if layout_part else None
        )
        theme_part = (
            _first_rel_target_by_type(package, master_part, _REL_THEME)
            if master_part else None
        )

        slides.append(PptxSlide(
            slide_index=slide_index,
            slide_number=slide_index + 1,
            part_name=slide_part,
            layout_part_name=layout_part,
            master_part_name=master_part,
            theme_part_name=theme_part,
        ))

    return slides


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_pptx_package(source: str | Path | bytes | BinaryIO) -> PptxPackage:
    """
    Read a .pptx file into a PptxPackage with fully resolved slide metadata.

    Args:
        source: File path, raw .pptx bytes, or a binary file-like object.

    Returns:
        PptxPackage with all XML parts, relationships, and an ordered slide list
        where each slide carries its resolved layout, master, and theme part names.
    """
    package = read_oxm_package(source)
    slides = _build_slides(package)
    return PptxPackage(package=package, slides=slides)

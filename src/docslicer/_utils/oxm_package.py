"""
Open XML package reader.

Reads the zipped Open XML package and exposes parsed XML parts plus relationship
maps. Works for any Open XML format (DOCX, PPTX, XLSX). This module deliberately
stays close to the package format so downstream extractors can decide which
semantics they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from zipfile import ZipFile

from lxml import etree


REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class OxmlRelationship:
    """One relationship from a package part to another part or external URL."""

    rel_id: str
    rel_type: str
    target: str
    target_mode: str | None = None

    @property
    def is_external(self) -> bool:
        return (self.target_mode or "").lower() == "external"


@dataclass
class OxmlPackage:
    """Parsed Open XML package contents."""

    part_names: list[str]
    xml_parts: dict[str, etree._Element] = field(default_factory=dict)
    binary_parts: dict[str, bytes] = field(default_factory=dict)
    relationships: dict[str, dict[str, OxmlRelationship]] = field(default_factory=dict)

    def get_xml(self, part_name: str) -> etree._Element | None:
        return self.xml_parts.get(_normalize_part_name(part_name))

    def get_relationship(
        self,
        source_part: str,
        rel_id: str | None,
    ) -> OxmlRelationship | None:
        if not rel_id:
            return None
        return self.relationships.get(_normalize_part_name(source_part), {}).get(rel_id)

    def relationship_target(self, source_part: str, rel_id: str | None) -> str | None:
        rel = self.get_relationship(source_part, rel_id)
        if rel is None:
            return None
        return rel.target


def _normalize_part_name(part_name: str) -> str:
    return part_name.lstrip("/")


def _source_part_from_rels_name(rels_name: str) -> str:
    """
    Convert a relationship part name to the source part it describes.

    Examples
    --------
    word/_rels/document.xml.rels -> word/document.xml
    _rels/.rels                  -> ""
    """
    rels_name = _normalize_part_name(rels_name)
    if rels_name == "_rels/.rels":
        return ""
    marker = "/_rels/"
    if marker not in rels_name or not rels_name.endswith(".rels"):
        return rels_name
    prefix, rel_file = rels_name.split(marker, 1)
    return f"{prefix}/{rel_file[:-5]}"


def _parse_xml(data: bytes) -> etree._Element | None:
    parser = etree.XMLParser(resolve_entities=False, recover=True, huge_tree=True)
    try:
        return etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError:
        return None


def _parse_relationships(root: etree._Element) -> dict[str, OxmlRelationship]:
    rels: dict[str, OxmlRelationship] = {}
    for rel in root.findall(f"{{{REL_NS}}}Relationship"):
        rel_id = rel.get("Id")
        if not rel_id:
            continue
        rels[rel_id] = OxmlRelationship(
            rel_id=rel_id,
            rel_type=rel.get("Type", ""),
            target=rel.get("Target", ""),
            target_mode=rel.get("TargetMode"),
        )
    return rels


def read_oxm_package(source: str | Path | bytes | BinaryIO) -> OxmlPackage:
    """
    Read an Open XML file (.docx, .pptx, …) into parsed XML parts, binary parts,
    and relationships.

    Args:
        source: File path, raw bytes, or a binary file-like object.

    Returns:
        OxmlPackage with all parseable XML parts and package relationships.
    """
    if isinstance(source, (str, Path)):
        zip_source = Path(source)
    elif isinstance(source, bytes):
        from io import BytesIO

        zip_source = BytesIO(source)
    else:
        zip_source = source

    xml_parts: dict[str, etree._Element] = {}
    binary_parts: dict[str, bytes] = {}
    relationships: dict[str, dict[str, OxmlRelationship]] = {}

    with ZipFile(zip_source) as zf:
        part_names = sorted(zf.namelist())
        for name in part_names:
            data = zf.read(name)
            norm_name = _normalize_part_name(name)
            if norm_name.endswith(".xml") or norm_name.endswith(".rels"):
                root = _parse_xml(data)
                if root is None:
                    binary_parts[norm_name] = data
                    continue
                xml_parts[norm_name] = root
                if norm_name.endswith(".rels"):
                    source_part = _source_part_from_rels_name(norm_name)
                    relationships[source_part] = _parse_relationships(root)
            else:
                binary_parts[norm_name] = data

    return OxmlPackage(
        part_names=part_names,
        xml_parts=xml_parts,
        binary_parts=binary_parts,
        relationships=relationships,
    )

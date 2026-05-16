"""DOCX package reader — thin alias over the shared Open XML package reader."""

from .._utils.oxm_package import OxmlPackage as DocxPackage
from .._utils.oxm_package import OxmlRelationship as DocxRelationship
from .._utils.oxm_package import read_oxm_package as read_docx_package

__all__ = ["DocxPackage", "DocxRelationship", "read_docx_package"]

"""PDF-specific utilities."""
from .struct_tree import (
    StructInfo,
    WidgetLink,
    build_struct_index,
    build_struct_index_with_links,
    has_struct_tree,
)
from .form_fields import FormField, build_form_index, has_acroform
from .form_label_link import build_form_label_index

__all__ = [
    "StructInfo",
    "WidgetLink",
    "build_struct_index",
    "build_struct_index_with_links",
    "has_struct_tree",
    "FormField",
    "build_form_index",
    "has_acroform",
    "build_form_label_index",
]

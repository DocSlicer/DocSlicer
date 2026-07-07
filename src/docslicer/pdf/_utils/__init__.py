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
from ..step_07_word_relationships import (
    add_link_relationships,
    add_rect_relationships,
)

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
    "add_link_relationships",
    "add_rect_relationships",
    "add_horizontal_line_relationships",
]

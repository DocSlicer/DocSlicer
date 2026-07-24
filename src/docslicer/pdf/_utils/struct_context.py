# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Document-level structure context (single pikepdf pass).

All of docslicer's pikepdf-backed enrichment — the logical structure tree, the
AcroForm fields, and the struct-tree widget→label join — is derived from the
*same* ``/StructTreeRoot`` / ``/AcroForm`` objects. This module opens the PDF with
pikepdf exactly **once** and returns every derived index bundled in a
:class:`StructContext`, so the orchestrator can build it up front and hand the
plain dataclass dicts to the pdfium-based extractors (words, images, shapes)
without any of them re-opening the file.

Password handling
-----------------
``build_struct_context`` deliberately does **not** swallow
:class:`pikepdf.PasswordError`. Because it runs before any pdfium call, pikepdf is
the first library to touch the bytes, so a missing/wrong password surfaces here as
a clean typed exception the orchestrator can catch to trigger decryption — rather
than as pdfium's untyped error later. Every *other* failure degrades to an empty
index (an untagged / form-less PDF is normal, not an error).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pikepdf

from .struct_tree import StructInfo, WidgetLink, build_struct_index_with_links
from .form_fields import FormField, build_form_index
from .form_label_link import build_form_label_index


@dataclass
class StructContext:
    """Every pikepdf-derived index for one document, built in a single open.

    Empty fields are the normal degraded state (untagged PDF, no AcroForm); they
    are never ``None`` so callers can index them unconditionally."""
    struct_index: Dict[Tuple[Optional[int], int], StructInfo] = field(default_factory=dict)
    widget_links: Dict[Tuple[int, int], WidgetLink] = field(default_factory=dict)
    form_index: Dict[int, List[FormField]] = field(default_factory=dict)
    form_label_index: Dict[Tuple[Optional[int], int], FormField] = field(default_factory=dict)


def _build_from_pdf(pk: pikepdf.Pdf) -> StructContext:
    ctx = StructContext()
    try:
        ctx.struct_index, ctx.widget_links = build_struct_index_with_links(pk)
    except Exception:
        ctx.struct_index, ctx.widget_links = {}, {}
    try:
        ctx.form_index = build_form_index(pk)
    except Exception:
        ctx.form_index = {}
    try:
        ctx.form_label_index = build_form_label_index(
            ctx.struct_index, ctx.widget_links, ctx.form_index
        )
    except Exception:
        ctx.form_label_index = {}
    return ctx


def build_struct_context(source: Union[str, Path, pikepdf.Pdf]) -> StructContext:
    """Open *source* with pikepdf once and return its :class:`StructContext`.

    Raises :class:`pikepdf.PasswordError` when *source* is an encrypted path that
    needs a password — the caller is expected to decrypt and retry. All other
    parse failures degrade to empty indices inside the returned context."""
    if isinstance(source, pikepdf.Pdf):
        return _build_from_pdf(source)
    # PasswordError from this open is intentionally NOT caught — see module docstring.
    with pikepdf.open(str(Path(source).expanduser())) as pk:
        return _build_from_pdf(pk)

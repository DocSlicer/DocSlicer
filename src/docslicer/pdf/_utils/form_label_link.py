# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Structural form-label linker.

Joins AcroForm fields (:mod:`form_fields`) to their visible label text using the
PDF structure tree (:mod:`struct_tree`) instead of spatial proximity. For tagged
PDFs this is unambiguous and immune to the "two side-by-side Yes/No checkboxes"
failure mode of geometry matching.

How widgets and labels relate in practice
-----------------------------------------
A widget annotation appears in the structure tree as an ``/OBJR`` leaf. Two
layouts occur:

1. **Reading-order siblings** (the common case — Acrobat/USCIS forms). The
   widget's ``/OBJR`` lives in its own ``P``/``Form`` element, a *sibling* of the
   label paragraph, placed in reading order right after the prompt text. The
   label is then the nearest *preceding* text leaf by DFS rank.
2. **Nested containment** (rarer, but 100% reliable when present). The label text
   shares the widget's owning struct element or its parent.

We resolve containment first (priority 0), then fall back to nearest-preceding
reading order (priority 1). When several widgets would claim the same text leaf
(e.g. a question shared across a checkbox group), the rank-closest widget wins.

Output
------
``build_form_label_index`` returns ``{(page_index, mcid): FormField}``. Multiple
MCIDs can map to one field (a multi-line label is several text runs), and step_01
assigns the field's metadata to every word carrying one of those MCIDs. Untagged
widgets (no WidgetLink) are absent and fall back to step_01's spatial matcher.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

from .form_fields import FormField
from .struct_tree import StructInfo, WidgetLink

# (priority, distance) — lower is better. Containment beats reading order; within
# each, the smaller rank gap wins.
_CONTAINMENT = 0
_READING_ORDER = 1


def build_form_label_index(
    struct_index: Dict[Tuple[Optional[int], int], StructInfo],
    widget_links: Dict[Tuple[int, int], WidgetLink],
    form_index: Dict[int, List[FormField]],
) -> Dict[Tuple[Optional[int], int], FormField]:
    """Map ``(page_index, mcid)`` → owning :class:`FormField` via the struct tree.

    Returns an empty dict when the PDF is untagged or no widget resolves a label.
    """
    if not struct_index or not widget_links or not form_index:
        return {}

    field_by_widget: Dict[Tuple[int, int], FormField] = {
        fld.widget_objgen: fld
        for fields in form_index.values()
        for fld in fields
        if fld.widget_objgen is not None
    }
    if not field_by_widget:
        return {}

    # Pre-index leaves: element id → its text-leaf keys, and per-page rank-sorted
    # leaves for the nearest-preceding lookup.
    elem_to_keys: Dict[int, List[Tuple[Optional[int], int]]] = {}
    by_page: Dict[Optional[int], List[Tuple[int, Tuple[Optional[int], int], int]]] = {}
    for key, info in struct_index.items():
        elem_to_keys.setdefault(info.elem_id, []).append(key)
        by_page.setdefault(key[0], []).append((info.rank, key, info.elem_id))
    for leaves in by_page.values():
        leaves.sort()

    # key → (priority, distance, field); keep the best offer per text leaf.
    best: Dict[Tuple[Optional[int], int], Tuple[int, int, FormField]] = {}

    def _offer(key, priority: int, distance: int, fld: FormField) -> None:
        cur = best.get(key)
        if cur is None or (priority, distance) < (cur[0], cur[1]):
            best[key] = (priority, distance, fld)

    for objgen, link in widget_links.items():
        fld = field_by_widget.get(objgen)
        if fld is None:
            continue

        owner = link.elem_id
        parent = link.chain_ids[-2] if len(link.chain_ids) >= 2 else owner

        # Phase A — nested containment (rare, exact). Text under the widget's own
        # element, or a Lbl sharing its parent.
        contained = [
            k for k, info in struct_index.items()
            if owner in info.ancestor_ids
            or (parent in info.ancestor_ids and info.tag == "Lbl")
        ]
        if contained:
            for k in contained:
                _offer(k, _CONTAINMENT, 0, fld)
            continue

        # Phase B — reading-order nearest preceding text leaf on the same page.
        pg = link.page_index
        leaves = by_page.get(pg)
        if not leaves:
            continue
        ranks = [r for r, _, _ in leaves]
        i = bisect.bisect_left(ranks, link.rank) - 1
        if i < 0:
            continue
        prec_rank, _prec_key, prec_elem = leaves[i]
        distance = link.rank - prec_rank
        for k in elem_to_keys.get(prec_elem, []):
            if k[0] == pg:  # same-page leaves of the preceding element
                _offer(k, _READING_ORDER, distance, fld)

    return {k: v[2] for k, v in best.items()}

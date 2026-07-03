"""
Struct-tree field enrichment for per-word DataFrames.

Given the per-word DataFrame produced by step_01 (word extractor), enriched
with struct_ancestors / struct_ancestor_ids by the struct-tree parser, adds:

    1. Extract table structure fields per word:

        table_id          — DFS elem_id of the nearest Table ancestor
        table_row_id      — DFS elem_id of the nearest TR ancestor
        table_cell_id     — DFS elem_id of the nearest TD or TH ancestor
        table_header_flag — True when the word is in a header cell (see below)

        Tables with exactly one cell are rejected: all four columns are left
        null/False for every word in such a table.

        ``table_header_flag`` is True when the ancestor path contains THead,
        or when every cell in the table's first TR is a TH.

    2. Extract ``textbox_id`` — elem_id of the nearest ancestor in
       struct_raw_ancestors whose tag contains "textbox" (case-insensitive),
       null when no such ancestor exists.

    3. Assign ``block_type`` using struct-tree and layout evidence.
       Priority order (first match wins):

       footnote      — "Note" in ancestor chain
       table         — word is inside a non-rejected table cell
       block_quote   — "BlockQuote" in ancestor chain
       chart         — "Chart" in raw ancestor chain (Chart → Sect via RoleMap)
       form_field    — form_widget column is non-blank
       toc_heading   — heading tag (H/H1-H6) in ancestors AND the word is inside
                       a TOC/TOCI element or its text matches a TOC title phrase
       heading       — heading tag (H/H1-H6) in ancestor chain
       vertical_text — text_orientation is BTT or TTB

These fields only depend on raw struct-ancestor columns, not on
struct_group_id, so they can be computed independently of group-ID
assignment.

Public API:
    prefill_styles(df) -> pd.DataFrame
        Adds table fields, textbox_id, and block_type columns, then
        returns *df*.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd


_TABLE_CELL = frozenset({"TD", "TH"})
_HEADINGS   = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_TOC_TAGS   = frozenset({"TOC", "TOCI"})

_TOC_HEADER_TEXTS: frozenset[str] = frozenset({
    "table of contents",
    "table of content",
    "table of figures",
    "table of figure",
    "table of tables",
})


def _as_list(v: Any) -> list:
    """
    Coerce a struct-list cell to a real list.

    ``None`` and float ``NaN`` (how pandas stores missing values in an object
    column) and any other scalar collapse to ``[]`` — critical because ``NaN``
    is truthy, so the ``value or []`` idiom would otherwise leak it into ``in``
    / ``set`` / ``zip`` and raise ``TypeError: 'float' is not iterable``.
    """
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return []


def _assign_table_fields(df: pd.DataFrame) -> None:
    """Add table_id, table_row_id, table_cell_id, table_header_flag to *df* in-place."""
    anc_arr = df["struct_ancestors"].to_numpy(dtype=object)
    aid_arr = df["struct_ancestor_ids"].to_numpy(dtype=object)
    n = len(df)

    t_table   = np.empty(n, dtype=object); t_table[:]   = None
    t_row     = np.empty(n, dtype=object); t_row[:]     = None
    t_cell    = np.empty(n, dtype=object); t_cell[:]    = None
    cell_tags = np.empty(n, dtype=object); cell_tags[:] = None
    thead_arr = np.zeros(n, dtype=bool)

    for i in range(n):
        ancs = _as_list(anc_arr[i])
        aids = _as_list(aid_arr[i])
        if not ancs or not aids:
            continue
        table_id = row_id = cell_id = cell_tag = None
        has_thead = False
        for tag, id_ in zip(ancs, aids):
            if tag == "Table" and table_id is None:
                table_id = id_
            elif tag == "THead":
                has_thead = True
            elif tag == "TR" and row_id is None:
                row_id = id_
            elif tag in _TABLE_CELL and cell_id is None:
                cell_id = id_
                cell_tag = tag
        t_table[i]   = table_id
        t_row[i]     = row_id
        t_cell[i]    = cell_id
        cell_tags[i] = cell_tag
        thead_arr[i] = has_thead

    # Reject single-cell tables.
    cells_per_table: dict[Any, set] = {}
    for i in range(n):
        tid = t_table[i]
        cid = t_cell[i]
        if tid is not None and cid is not None:
            cells_per_table.setdefault(tid, set()).add(cid)
    single_cell = {tid for tid, cells in cells_per_table.items() if len(cells) == 1}

    # Identify first TR (lowest elem_id) per table for the all-TH header check.
    first_row: dict[Any, Any] = {}
    for i in range(n):
        tid = t_table[i]
        rid = t_row[i]
        if tid is not None and rid is not None:
            if tid not in first_row or rid < first_row[tid]:
                first_row[tid] = rid

    first_row_cell_tags: dict[Any, set] = {}
    for i in range(n):
        tid  = t_table[i]
        rid  = t_row[i]
        cid  = t_cell[i]
        ctag = cell_tags[i]
        if (tid is not None and rid is not None
                and cid is not None and ctag is not None
                and rid == first_row.get(tid)):
            first_row_cell_tags.setdefault(tid, set()).add(ctag)

    first_row_all_th = {
        tid for tid, tags in first_row_cell_tags.items()
        if tags and tags <= {"TH"}
    }

    # Build final header-flag array; null out rejected tables.
    header_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        tid = t_table[i]
        if tid is None:
            continue
        if tid in single_cell:
            t_table[i] = None
            t_row[i]   = None
            t_cell[i]  = None
            continue
        header_flags[i] = bool(thead_arr[i]) or (
            tid in first_row_all_th and t_row[i] == first_row.get(tid)
        )

    df["table_id"]          = t_table
    df["table_row_id"]      = t_row
    df["table_cell_id"]     = t_cell
    df["table_header_flag"] = header_flags


def _assign_textbox_id(df: pd.DataFrame) -> None:
    """Add ``textbox_id`` — elem_id of the ancestor whose raw tag contains 'textbox' (case-insensitive)."""
    raw_anc_arr = df["struct_raw_ancestors"].to_numpy(dtype=object) \
        if "struct_raw_ancestors" in df.columns \
        else np.full(len(df), None, dtype=object)
    aid_arr = df["struct_ancestor_ids"].to_numpy(dtype=object) \
        if "struct_ancestor_ids" in df.columns \
        else np.full(len(df), None, dtype=object)
    n = len(df)

    textbox_id = np.empty(n, dtype=object)
    textbox_id[:] = None

    for i in range(n):
        for tag, eid in zip(_as_list(raw_anc_arr[i]), _as_list(aid_arr[i])):
            if isinstance(tag, str) and "textbox" in tag.lower():
                textbox_id[i] = eid
                break

    df["textbox_id"] = textbox_id


def _assign_block_type(df: pd.DataFrame) -> None:
    """Add ``block_type`` column to *df* in-place using struct and layout data."""
    anc_arr     = df["struct_ancestors"].to_numpy(dtype=object)
    raw_anc_arr = df["struct_raw_ancestors"].to_numpy(dtype=object) \
        if "struct_raw_ancestors" in df.columns \
        else np.full(len(df), None, dtype=object)
    cell_id_arr = df["table_cell_id"].to_numpy(dtype=object) \
        if "table_cell_id" in df.columns \
        else np.full(len(df), None, dtype=object)
    fw_arr      = df["form_widget"].to_numpy(dtype=object) \
        if "form_widget" in df.columns \
        else np.full(len(df), None, dtype=object)
    orient_arr  = df["text_orientation"].to_numpy(dtype=object) \
        if "text_orientation" in df.columns \
        else np.full(len(df), None, dtype=object)

    if "text" in df.columns:
        text_arr = df["text"].fillna("").astype(str).str.strip().str.lower().to_numpy()
    else:
        text_arr = np.full(len(df), "", dtype=object)

    n = len(df)
    block_types = np.empty(n, dtype=object)
    block_types[:] = None

    for i in range(n):
        ancs     = _as_list(anc_arr[i])
        ancs_set = set(ancs)
        raw_ancs = _as_list(raw_anc_arr[i])

        # 1. footnote
        if "Note" in ancs_set:
            block_types[i] = "footnote"
            continue

        # 2. table (only for non-rejected cells)
        if cell_id_arr[i] is not None:
            block_types[i] = "table"
            continue

        # 3. block_quote
        if "BlockQuote" in ancs_set:
            block_types[i] = "block_quote"
            continue

        # 4. chart (Chart → Sect via RoleMap; check raw chain)
        if "Chart" in raw_ancs:
            block_types[i] = "chart"
            continue

        # 5. form_field
        fw = fw_arr[i]
        if fw is not None and not (isinstance(fw, float) and fw != fw) and str(fw).strip():
            block_types[i] = "form_field"
            continue

        # 6. toc_heading / heading
        if ancs_set & _HEADINGS:
            is_toc_context = bool(ancs_set & _TOC_TAGS)
            is_toc_text    = text_arr[i] in _TOC_HEADER_TEXTS
            block_types[i] = "toc_heading" if (is_toc_context or is_toc_text) else "heading"
            continue

        # 7. toc: TOC or TOCI in ancestor chain (but not the title heading itself)
        if ancs_set & _TOC_TAGS:
            block_types[i] = "toc"
            continue

        # 8. vertical_text
        if orient_arr[i] in ("BTT", "TTB"):
            block_types[i] = "vertical_text"

    # Post-process: heading immediately before TOC/TOCI content → toc_heading.
    # Handles the common pattern where the TOC title (e.g. "Contents") is tagged
    # as a plain H2 in the struct tree, with the TOC entries starting on the very
    # next text_object_id on the same page (text_object_id is per-page, not global).
    if "text_object_id" in df.columns and "page_number" in df.columns:
        txobj_arr = df["text_object_id"].to_numpy(dtype=object)
        page_arr  = df["page_number"].to_numpy(dtype=object)

        def _safe_int(v) -> Optional[int]:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        # (page, text_object_id) pairs that carry TOC/TOCI in their ancestor chain
        toc_page_txobj: set[tuple] = set()
        for i in range(n):
            ancs = _as_list(anc_arr[i])
            if set(ancs) & _TOC_TAGS:
                v = _safe_int(txobj_arr[i])
                if v is not None:
                    toc_page_txobj.add((page_arr[i], v))

        if toc_page_txobj:
            # Per-page sorted text_object_id lists → next-id map keyed by (page, txobj)
            page_txobjs: dict = defaultdict(set)
            for i in range(n):
                v = _safe_int(txobj_arr[i])
                if v is not None:
                    page_txobjs[page_arr[i]].add(v)

            next_txobj: dict[tuple, int] = {}
            for pg, ids in page_txobjs.items():
                ordered = sorted(ids)
                for a, b in zip(ordered, ordered[1:]):
                    next_txobj[(pg, a)] = b

            for i in range(n):
                if block_types[i] == "heading":
                    v = _safe_int(txobj_arr[i])
                    if v is not None:
                        nxt = next_txobj.get((page_arr[i], v))
                        if nxt is not None and (page_arr[i], nxt) in toc_page_txobj:
                            block_types[i] = "toc_heading"

    df["block_type"] = block_types


def prefill_styles(df: pd.DataFrame) -> pd.DataFrame:
    """Add table fields, textbox_id, and block_type to *df* and return it."""
    _assign_table_fields(df)
    _assign_textbox_id(df)
    _assign_block_type(df)
    return df

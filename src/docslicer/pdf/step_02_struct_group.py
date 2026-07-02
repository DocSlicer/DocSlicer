"""
Step 02 – Struct group ID assignment + struct-based field enrichment

Responsibility:
    Given the per-word DataFrame produced by step_01 (word extractor):

    1. Assign ``struct_group_id`` — an integer that identifies the logical block
       each word belongs to.  Words sharing the same ``struct_group_id`` should
       be treated as a single coherent unit by downstream steps.

    2. Extract table structure fields per word:

       table_id          — DFS elem_id of the nearest Table ancestor
       table_row_id      — DFS elem_id of the nearest TR ancestor
       table_cell_id     — DFS elem_id of the nearest TD or TH ancestor
       table_header_flag — True when the word is in a header cell (see below)

       Tables with exactly one cell are rejected: all four columns are left
       null/False for every word in such a table.

       ``table_header_flag`` is True when the ancestor path contains THead,
       or when every cell in the table's first TR is a TH.

    3. Extract ``textbox_id`` — elem_id of the nearest ancestor in
       struct_raw_ancestors whose tag contains "textbox" (case-insensitive),
       null when no such ancestor exists.

    4. Assign ``block_type`` using struct-tree and layout evidence.
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

Channel priority for struct_group_id (resolved per page, then per word):
    1. Struct — ancestor-based walk through the struct tree.
       Valid only when the page has at least one tag beyond Document/Part.
       A page whose every word is tagged only with Document/Part is structurally
       flat; collapsing all its words into one group would kill downstream ops.

       Grouping anchor searched in priority order (most-specific first):
         TOCI     — one TOC entry per key
         TD / TH  — innermost table cell; takes priority over LI so that a list
                    inside a table cell rolls up to the cell, not list items
         LI       — innermost list item (each LI is its own group; nested lists
                    still resolve to their deepest LI, not the containing L)
         H tags   — outermost heading; ['H4', 'H1'] in root→leaf order → group by H1
         P        — innermost paragraph
         element  — the struct element itself (struct_tag_id), unless it is
                    Document or Part (Sect is allowed as a direct wrapper)

       Document and Part are never used as a grouping key.
       Sect is skipped as an ancestor but allowed as the element's own tag.

    2. MCID + MCID_OFFSET (1 000 000)
       Valid only when the page has more than one distinct MCID value.
       A single MCID = 0 means the renderer dumped all content into one mark.

    3. text_object_id + TXOBJ_OFFSET (2 000 000)   — always-available fallback

Cross-tranche integer ranges prevent collisions between channels.

Public API:
    assign_struct_group_id(df) -> pd.DataFrame
        Adds ``struct_group_id``, table fields, ``textbox_id``, and
        ``block_type`` columns, then returns *df*.  Operates per page for the
        group-id validity gates.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd


MCID_OFFSET  = 1_000_000
TXOBJ_OFFSET = 2_000_000

# Struct fields copied from a healthy word to a blank word that shares the same
# (page, text_object_id).  Words of a single text object are emitted from one
# marked-content run, so they must share identical struct metadata; when the
# producer drops it on a subset of the run's words we can rebuild it losslessly.
_REPAIR_FIELDS = (
    "struct_tag_id", "struct_tag", "struct_raw_tag",
    "struct_col_span", "struct_row_span", "struct_scope", "struct_headers",
    "mcid", "marked_tag", "reading_rank",
    "struct_ancestors", "struct_raw_ancestors", "struct_ancestor_ids",
)

_INVALID_TAGS  = frozenset({"Document", "Part"})
_TABLE_CELL    = frozenset({"TD", "TH"})
_HEADINGS      = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_TOC_TAGS      = frozenset({"TOC", "TOCI"})

_TOC_HEADER_TEXTS: frozenset[str] = frozenset({
    "table of contents",
    "table of content",
    "table of figures",
    "table of figure",
    "table of tables",
})


# ---------------------------------------------------------------------------
# Struct group ID helpers (unchanged)
# ---------------------------------------------------------------------------

def _struct_group_key(
    stag: Optional[str],
    sid: Optional[int],
    ancs: Optional[list[str]],
    aids: Optional[list[int]],
) -> Optional[int]:
    """
    Derive the struct_group_id key for one word from its position in the struct tree.

    Returns None when struct info is absent or the element falls under an
    excluded tag (Document, Part).
    """
    if stag is None or sid is None:
        return None
    try:
        sid = int(sid)
    except (TypeError, ValueError):
        return None

    full_tags = _as_list(ancs) + [stag]
    full_ids  = _as_list(aids) + [sid]

    # 1. innermost TOCI — reversed so nested TOC-in-TOC resolves to the deepest entry
    for tag, eid in zip(reversed(full_tags), reversed(full_ids)):
        if tag == "TOCI":
            return eid

    # 2. innermost TD or TH — must come before LI so that a list inside a
    #    table cell rolls up to the cell, not to individual list items
    for tag, eid in zip(reversed(full_tags), reversed(full_ids)):
        if tag in _TABLE_CELL:
            return eid

    # 3. innermost LI — each list item is its own group; nested lists still
    #    resolve to their deepest LI, not the containing L
    for tag, eid in zip(reversed(full_tags), reversed(full_ids)):
        if tag == "LI":
            return eid

    # 4. outermost H — first match in root→leaf order
    for tag, eid in zip(full_tags, full_ids):
        if tag in _HEADINGS:
            return eid

    # 5. innermost P
    for tag, eid in zip(reversed(full_tags), reversed(full_ids)):
        if tag == "P":
            return eid

    # 6. element itself — Sect allowed, Document/Part excluded
    if stag not in _INVALID_TAGS:
        return int(sid)

    return None


def _is_blank(v: Any) -> bool:
    """True when a struct field carries no information (None, NaN, empty list/str)."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:      # NaN
        return True
    if isinstance(v, (list, tuple, str)) and len(v) == 0:
        return True
    return False


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


def _repair_struct_fields(df: pd.DataFrame) -> None:
    """
    Repair badly formatted struct metadata in-place.

    All words emitted from a single text object belong to the same struct
    element, so they must share identical struct metadata.  Some producers drop
    that metadata on a subset of a text object's words (e.g. a trailing dotted
    leader in a TOC entry), leaving blank ancestors / tags on otherwise-tagged
    objects.  For every such blank word we borrow the fields listed in
    ``_REPAIR_FIELDS`` from a healthy sibling that shares its
    ``(page_number, text_object_id)``.

    Only blank cells are overwritten, so healthy metadata is never clobbered.
    """
    if not {"page_number", "text_object_id", "struct_tag_id"} <= set(df.columns):
        return

    needs = (df["struct_tag_id"].isna() & df["text_object_id"].notna()).to_numpy()
    donor_mask = df["struct_tag_id"].notna() & df["text_object_id"].notna()
    if not needs.any() or not donor_mask.any():
        return

    fields = [c for c in _REPAIR_FIELDS if c in df.columns]

    # First healthy value per (page, text_object_id) for each repairable field.
    donor_first = (
        df.loc[donor_mask, ["page_number", "text_object_id", *fields]]
          .groupby(["page_number", "text_object_id"], sort=False)[fields]
          .first()
    )
    if donor_first.empty:
        return

    # Align the donor table to every row of df by (page, text_object_id).
    mi = pd.MultiIndex.from_arrays(
        [df["page_number"].to_numpy(), df["text_object_id"].to_numpy()]
    )
    aligned = donor_first.reindex(mi)

    # Rows that need repair AND have a donor for their text object.
    has_donor = aligned["struct_tag_id"].notna().to_numpy()
    positions = np.flatnonzero(needs & has_donor)
    if positions.size == 0:
        return

    for field in fields:
        col = df[field].to_numpy(dtype=object).copy()
        donor_vals = aligned[field].to_numpy(dtype=object)
        changed = False
        for i in positions:
            if _is_blank(col[i]):
                col[i] = donor_vals[i]
                changed = True
        if changed:
            df[field] = col


def _assign_page(page_df: pd.DataFrame) -> np.ndarray:
    """Compute struct_group_id for one page's word slice."""
    n = len(page_df)

    stag_arr  = page_df["struct_tag"].to_numpy(dtype=object)
    sid_arr   = page_df["struct_tag_id"].to_numpy(dtype=object)
    anc_arr   = page_df["struct_ancestors"].to_numpy(dtype=object)
    aid_arr   = page_df["struct_ancestor_ids"].to_numpy(dtype=object)
    mcid_arr  = page_df["mcid"].to_numpy(dtype=object)
    txobj_arr = page_df["text_object_id"].to_numpy(dtype=object)

    # Struct validity: at least one word has a tag outside {Document, Part}.
    struct_valid = False
    for j in range(n):
        st = stag_arr[j]
        if st is not None and st not in _INVALID_TAGS:
            struct_valid = True
            break
        for t in _as_list(anc_arr[j]):
            if t not in _INVALID_TAGS:
                struct_valid = True
                break
        if struct_valid:
            break

    # MCID validity: page must have more than one distinct MCID value.
    def _to_int(v) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    mcid_valid = len({mc for mc in mcid_arr if _to_int(mc) is not None}) > 1

    # Pass 1: struct-tree key per word (None when struct channel is invalid).
    struct_keys = np.empty(n, dtype=object)
    struct_keys[:] = None
    if struct_valid:
        for i in range(n):
            struct_keys[i] = _struct_group_key(
                stag_arr[i], sid_arr[i], anc_arr[i], aid_arr[i]
            )

    # Pass 2: reject struct keys whose member text_object_ids are not
    # contiguous.  A well-formed struct group spans a gap-free run of text
    # objects; a gap means an object that should belong here resolved elsewhere
    # (unrepairable blank struct), so the grouping is untrustworthy.  Rejected
    # keys fall back to the text_object_id method below.
    rejected: set = set()
    if struct_valid:
        key_txobjs: dict[Any, set] = defaultdict(set)
        for i in range(n):
            k = struct_keys[i]
            if k is not None:
                tx = _to_int(txobj_arr[i])
                if tx is not None:
                    key_txobjs[k].add(tx)
        for k, txs in key_txobjs.items():
            if txs and (max(txs) - min(txs) + 1) != len(txs):
                rejected.add(k)

    # Pass 3: resolve each word's group id across channels.
    sg = np.empty(n, dtype=object)
    sg[:] = None

    for i in range(n):
        k = struct_keys[i]
        if k is not None:
            if k not in rejected:
                sg[i] = k
                continue
            # Rejected struct group → text_object_id method directly.
            tx = _to_int(txobj_arr[i])
            if tx is not None:
                sg[i] = TXOBJ_OFFSET + tx
            continue
        mc = _to_int(mcid_arr[i])
        if mcid_valid and mc is not None:
            sg[i] = MCID_OFFSET + mc
        else:
            tx = _to_int(txobj_arr[i])
            if tx is not None:
                sg[i] = TXOBJ_OFFSET + tx

    return sg


# ---------------------------------------------------------------------------
# Table field extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Block type assignment
# ---------------------------------------------------------------------------

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
            from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_struct_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``struct_group_id``, table structure fields, and ``block_type`` to *df*
    and return it.

    Doc-level shortcut: if the entire document has no struct_tag_id and no
    mcid, both channels are dead for every page.  Skip the per-page validity
    loop and assign text_object_id + TXOBJ_OFFSET directly (O(n) single pass).

    Otherwise operates per page so validity gates (struct, MCID) are evaluated
    independently for each page's word set.
    """
    if df.empty:
        df["struct_group_id"]  = None
        df["table_id"]         = None
        df["table_row_id"]     = None
        df["table_cell_id"]    = None
        df["table_header_flag"] = False
        df["block_type"]       = None
        df["textbox_id"]       = None
        return df

    # Repair blank struct metadata from healthy siblings sharing a text object
    # before any grouping/enrichment consumes the struct fields.
    _repair_struct_fields(df)

    has_struct = df["struct_tag_id"].notna().any()
    has_mcid   = df["mcid"].notna().any()

    if not has_struct and not has_mcid:
        txobj = df["text_object_id"].to_numpy(dtype=object)
        out = np.empty(len(df), dtype=object)
        out[:] = None
        for i, v in enumerate(txobj):
            try:
                out[i] = TXOBJ_OFFSET + int(v)
            except (TypeError, ValueError):
                pass
        df["struct_group_id"] = out
    else:
        out = np.empty(len(df), dtype=object); out[:] = None

        for _, page_df in df.groupby("page_number", sort=False):
            idx = page_df.index
            pos = df.index.get_indexer(idx)
            out[pos] = _assign_page(page_df)

        df["struct_group_id"] = out

    _assign_table_fields(df)
    _assign_block_type(df)
    _assign_textbox_id(df)

    return df

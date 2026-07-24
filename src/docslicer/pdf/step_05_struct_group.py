# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Assign struct_group_id — the logical block each word belongs to.

df_words → df_words + struct_group_id.

Responsibility:
    Given the per-word DataFrame produced by step_01 (word extractor),
    assign ``struct_group_id`` — an integer that identifies the logical block
    each word belongs to.  Words sharing the same ``struct_group_id`` should
    be treated as a single coherent unit by downstream steps.

    Table fields (table_id, table_row_id, table_cell_id, table_header_flag),
    textbox_id, and block_type are a separate, decoupled concern — see
    ``_utils.struct_fields`` — and are not computed here.

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
        Adds ``struct_group_id`` and returns *df*.  Operates per page for the
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
    "mcid", "bdc_tag", "dfs_position",
    "struct_ancestors", "struct_raw_ancestors", "struct_ancestor_ids",
)

_INVALID_TAGS  = frozenset({"Document", "Part", "Art"})
_TABLE_CELL    = frozenset({"TD", "TH"})
_HEADINGS      = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})


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
# Public API
# ---------------------------------------------------------------------------

def assign_struct_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``struct_group_id`` to *df* and return it.

    Doc-level shortcut: if the entire document has no struct_tag_id and no
    mcid, both channels are dead for every page.  Skip the per-page validity
    loop and assign text_object_id + TXOBJ_OFFSET directly (O(n) single pass).

    Otherwise operates per page so validity gates (struct, MCID) are evaluated
    independently for each page's word set.
    """
    if df.empty:
        df["struct_group_id"]  = None
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

    return df #df_words

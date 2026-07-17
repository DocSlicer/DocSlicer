"""
Shared table column normalization.

Removes sparse "spacer" columns from an extracted table grid and merges their
content into the adjacent real column:

    - blank columns are dropped;
    - rparen columns (only ")", "%", or footnote refs like "(b)") merge their
      text into the nearest keep column to the LEFT  ("17" + "%" -> "17%");
    - lparen columns (only "(" or currency symbols) merge into the nearest
      keep column to the RIGHT ("$" + "252.50" -> "$252.50").

The core function is a behavior-identical copy of
html/step_05_table_extractor._normalize_columns, lifted here so the PDF
pipeline can reuse it. It is representation-agnostic: it only needs a
per-table DataFrame with row_start / col_start / rowspan / colspan / text.

normalize_columns_by_table() is the multi-table entry point used by the PDF
pipeline: it validates and casts the layout columns per table_id group and
skips any table whose grid is incomplete (NA layout values) rather than
guessing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..text_utils import _CURRENCY_SYMBOLS

# ---------------------------------------------------------------------------
# Cell-text classification helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PAREN_INNER_RE = re.compile(r"\(\s+|\s+\)")
_CURRENCY_TOKENS = _CURRENCY_SYMBOLS  # canonical set (single source of truth in text_utils)
_LPAREN_CHARS = frozenset({"(", "[", "{"})
_RPAREN_CHARS = frozenset({")", "]", "}", "%"})
# Matches a fully-parenthesized expression with no nested parens, optional trailing %.
# Used to recognise footnote-ref cells like (1), (n.m.) that sit in an rparen column.
_FULL_PAREN_RE = re.compile(r"^\([^()]*\)%?$")


def _norm_ws(s: str) -> str:
    s = _WS_RE.sub(" ", (s or "").replace("\xa0", " ")).strip()
    s = _PAREN_INNER_RE.sub(lambda m: "(" if m.group()[0] == "(" else ")", s)
    return s


def _is_currency(s: str) -> bool:
    return _norm_ws(s) in _CURRENCY_TOKENS


def _is_lparen(s: str) -> bool:
    s = _norm_ws(s)
    return bool(s) and all(c in _LPAREN_CHARS for c in s)


def _is_rparen(s: str) -> bool:
    s = _norm_ws(s)
    return bool(s) and all(c in _RPAREN_CHARS for c in s)


def _is_rparen_like(s: str) -> bool:
    """Strict rparen chars OR a fully-parenthesized expression like (1) or (n.m.)."""
    s = _norm_ws(s)
    return bool(s) and (_is_rparen(s) or bool(_FULL_PAREN_RE.match(s)))


# ---------------------------------------------------------------------------
# Core: single-table column normalization
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame, debug_cols: bool = True) -> pd.DataFrame:
    """
    Remove blank/paren spacer columns and merge paren values into adjacent cells.

    Expects one table's cells: one row per logical cell with int-valued
    row_start / col_start / rowspan / colspan and a text column. The index
    must be unique (it is used to address cells for text merges).

    Returns the frame with col_start/colspan remapped to the surviving dense
    column indices and spacer cells dropped. With debug_cols=True (default)
    two diagnostic columns are added: max_cols and initial_col_indices
    (post-remap indices per cell); callers that would drop them anyway pass
    False and skip the cost of materialising them per table.
    """
    df = df.copy()
    max_cols = int((df["col_start"] + df["colspan"]).max())
    all_indices_list: list[list[int]] = [
        list(range(cs, cs + sp))
        for cs, sp in zip(df["col_start"].to_numpy(), df["colspan"].to_numpy())
    ]
    if debug_cols:
        df["max_cols"] = max_cols
        df["initial_col_indices"] = all_indices_list
    all_texts: list = df["text"].tolist()
    all_row_starts: list[int] = df["row_start"].tolist()
    all_rowspans: list[int] = df["rowspan"].tolist()
    df_int_idx: list[int] = df.index.tolist()

    # ── Phase 1: classify each column index using only single-index cells ─────
    # Build texts_by_idx in one pass instead of one .loc per index
    texts_by_idx: dict[int, list[str]] = {}
    single_df_int_idx: list[int] = []          # df int indices of single-index cells
    single_col_idx: list[int] = []              # their column index

    for indices, text, di in zip(all_indices_list, all_texts, df_int_idx):
        if len(indices) == 1:
            c = indices[0]
            t = text if isinstance(text, str) else ""
            texts_by_idx.setdefault(c, []).append(t)
            single_df_int_idx.append(di)
            single_col_idx.append(c)

    index_class: dict[int, str] = {}
    for idx in range(max_cols):
        texts = texts_by_idx.get(idx)
        if texts is None:
            index_class[idx] = "multi_only"
            continue
        non_blank = [t for t in texts if t.strip()]
        if not non_blank:
            index_class[idx] = "blank"
        elif all(_is_rparen_like(t) for t in non_blank):
            index_class[idx] = "rparen"
        elif all(_is_lparen(t) or _is_currency(t) for t in non_blank):
            index_class[idx] = "lparen"
        else:
            index_class[idx] = "keep"

    # ── Phase 2: ensure every non-blank multi-index cell retains a keep index ──
    # Iterate raw lists — no iterrows overhead
    is_keep: dict[int, bool] = {idx: cls == "keep" for idx, cls in index_class.items()}

    for indices, text in zip(all_indices_list, all_texts):
        if len(indices) == 1:
            continue
        if not isinstance(text, str) or not text.strip():
            continue  # blank cell — safe to drop; don't promote its indices
        if not any(is_keep.get(i, False) for i in indices):
            # Promote only the first index — the others are redundant padding.
            # Promoting all would inflate colspan for groups where every cell
            # always co-covers the same set of indices (e.g. a 3-wide label col).
            first = indices[0]
            index_class[first] = "keep"
            is_keep[first] = True

    for idx in range(max_cols):
        if index_class[idx] == "multi_only":
            index_class[idx] = "blank"

    # ── Phase 3: paren merge ──────────────────────────────────────────────────
    keep_sorted = sorted(i for i, cls in index_class.items() if cls == "keep")
    rparen_idxs = sorted(i for i, cls in index_class.items() if cls == "rparen")
    lparen_idxs = sorted(i for i, cls in index_class.items() if cls == "lparen")

    if rparen_idxs or lparen_idxs:
        # O(1) reverse lookup: df label index → position in the extracted lists
        pos_of: dict[int, int] = {di: i for i, di in enumerate(df_int_idx)}

        # Compute nearest-left / nearest-right keep for each paren col
        rparen_neighbors: dict[int, int] = {}
        lparen_neighbors: dict[int, int] = {}
        needed_keep_cols: set[int] = set()

        for r_idx in rparen_idxs:
            lk = next((k for k in reversed(keep_sorted) if k < r_idx), None)
            if lk is not None:
                rparen_neighbors[r_idx] = lk
                needed_keep_cols.add(lk)

        for l_idx in lparen_idxs:
            rk = next((k for k in keep_sorted if k > l_idx), None)
            if rk is not None:
                lparen_neighbors[l_idx] = rk
                needed_keep_cols.add(rk)

        # Build paren cell lookup in one pass: col_idx → [(row_start, text), ...]
        paren_idx_set = set(rparen_idxs) | set(lparen_idxs)
        paren_cells: dict[int, list[tuple[int, str]]] = {}
        for c, di in zip(single_col_idx, single_df_int_idx):
            if c in paren_idx_set:
                list_pos = pos_of[di]
                paren_cells.setdefault(c, []).append((all_row_starts[list_pos], all_texts[list_pos]))

        # Build expanded grid only for needed keep cols (sparse — avoids full O(n*cols) grid)
        expanded: dict[tuple[int, int], int] = {}
        for r0, rs, indices, di in zip(all_row_starts, all_rowspans, all_indices_list, df_int_idx):
            for c in indices:
                if c in needed_keep_cols:
                    for r in range(r0, r0 + rs):
                        expanded[(r, c)] = di

        # Collect text updates in a dict; apply to df in one vectorized assignment
        text_updates: dict[int, str] = {}

        for r_idx, left_keep in rparen_neighbors.items():
            for row, ptext in paren_cells.get(r_idx, []):
                t = ptext if isinstance(ptext, str) else ""
                if not t.strip():
                    continue
                target = expanded.get((row, left_keep))
                if target is None:
                    continue
                cur = text_updates.get(target, all_texts[pos_of[target]])
                cur = cur if isinstance(cur, str) else ""
                text_updates[target] = (cur + t).strip() if cur.strip() else t.strip()

        for l_idx, right_keep in lparen_neighbors.items():
            for row, ltext in paren_cells.get(l_idx, []):
                t = ltext if isinstance(ltext, str) else ""
                if not t.strip():
                    continue
                target = expanded.get((row, right_keep))
                if target is None:
                    continue
                cur = text_updates.get(target, all_texts[pos_of[target]])
                cur = cur if isinstance(cur, str) else ""
                text_updates[target] = (t + cur).strip() if cur.strip() else t.strip()

        if text_updates:
            df.loc[list(text_updates.keys()), "text"] = list(text_updates.values())

    # ── Phase 4: remap col_start/colspan and drop removed cells ──────────────
    surviving = sorted(i for i, cls in index_class.items() if cls == "keep")
    if not surviving:
        return df.reset_index(drop=True)

    pos = {idx: new_pos for new_pos, idx in enumerate(surviving)}
    surviving_set = set(surviving)

    # Compute bounds in one list comprehension — no apply() overhead
    new_col_starts: list[int] = []
    new_colspans: list[int] = []
    keep_flags: list[bool] = []

    for indices in all_indices_list:
        kept = [pos[i] for i in indices if i in surviving_set]
        if kept:
            new_col_starts.append(kept[0])
            new_colspans.append(kept[-1] - kept[0] + 1)
            keep_flags.append(True)
        else:
            new_col_starts.append(0)
            new_colspans.append(0)
            keep_flags.append(False)

    keep_arr = np.array(keep_flags)
    df = df[keep_arr].copy()

    if df.empty:
        return df.reset_index(drop=True)

    kept_positions = np.where(keep_arr)[0]
    df["col_start"] = [new_col_starts[i] for i in kept_positions]
    df["colspan"] = [new_colspans[i] for i in kept_positions]
    if debug_cols:
        df["max_cols"] = len(surviving)
        df["initial_col_indices"] = [
            [pos[i] for i in all_indices_list[i] if i in surviving_set]
            for i in kept_positions
        ]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-table entry point (PDF pipeline)
# ---------------------------------------------------------------------------

_LAYOUT_COLS = ("row_start", "col_start", "rowspan", "colspan")


def normalize_columns_by_table(
    df: pd.DataFrame,
    keep_debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Apply normalize_columns() per table_id group of a multi-table cell frame.

    Tables whose layout columns contain NA (unresolved grid) pass through
    unchanged — the normalizer needs a complete int grid to reason about
    columns. Layout columns are cast to plain int for the core call; the
    frame keeps its original row order otherwise (normalized tables come
    back sorted by their internal reset_index, grouped per table).

    Args:
        df:              Cell-level frame with table_id, text, and the four
                         layout columns (row_start, col_start, rowspan, colspan).
        keep_debug_cols: Keep the max_cols / initial_col_indices diagnostic
                         columns added by the core normalizer.
    """
    required = {"table_id", "text", *_LAYOUT_COLS}
    if df.empty or not required.issubset(df.columns):
        return df

    # Cast the layout columns once for the whole frame; per-table casts pay
    # the pandas fixed cost hundreds of times on table-heavy documents.
    layout_all = df[list(_LAYOUT_COLS)].apply(pd.to_numeric, errors="coerce")
    has_na = layout_all.isna().any(axis=1)

    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby("table_id", sort=True, dropna=False):
        if has_na.loc[grp.index].any():
            parts.append(grp)
            continue
        work = grp.copy()
        work[list(_LAYOUT_COLS)] = layout_all.loc[grp.index].astype(int)
        work = normalize_columns(work, debug_cols=keep_debug_cols)
        parts.append(work)

    out = pd.concat(parts, ignore_index=True)
    if not keep_debug_cols:
        out = out.drop(columns=["max_cols", "initial_col_indices"], errors="ignore")
    return out

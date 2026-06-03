"""
HTML table extractor — step 05 of the HTML pipeline.

Runs AFTER merge_boxes_to_lines (step 04) so that table_id values in
df_table_cells are guaranteed to match those in df_lines.

Pipeline:
    1. Build original_table_id → final table_id mapping directly from df_lines
       (using the original_table_id column preserved by _remove_single_row_tables).
       Each HTML <table> carries a data-docslicer-table-id attribute (stamped by
       extract_boxes.js) that equals the JS tableCounter value — immune to nested-
       table ordering differences between JS assignment and BS4 document order.
    2. For each surviving final table, collect the matching HTML <table> element(s).
       Merged tables (multiple single-row HTML tables combined into one) are stacked
       vertically with row_start offsets.
    3. Parse the grid (expand rowspan/colspan), extract iXBRL metadata.
    4. Remove blank rows, normalize columns (blank spacers, currency prefix, paren suffix).
    5. Detect cell roles (shared utility).
    6. Emit one row per logical cell with table_id / table_row_id / page_number /
       page_label all aligned to df_lines.

Output columns (mirrors DOCX step_03_table_cell_builder + ix):
    page_number, page_label, table_id, table_row_id, table_cell_id,
    row_start, col_start, rowspan, colspan, role, text, ix
"""

from __future__ import annotations

import re
import warnings
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .._utils.table_utils import detect_cell_roles

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PAREN_INNER_RE = re.compile(r"\(\s+|\s+\)")
_CURRENCY_TOKENS = frozenset({"$", "€", "£", "¥", "₹", "₽", "₩", "₪", "₺"})
_LPAREN_CHARS = frozenset({"(", "[", "{"})
_RPAREN_CHARS = frozenset({")", "]", "}", "%"})
# Matches a fully-parenthesized expression with no nested parens, optional trailing %.
# Used to recognise footnote-ref cells like (1), (n.m.) that sit in an rparen column.
_FULL_PAREN_RE = re.compile(r"^\([^()]*\)%?$")

_OUTPUT_COLS = [
    "page_number",
    "page_label",
    "table_id",
    "table_row_id",
    "table_cell_id",
    "row_start",
    "col_start",
    "rowspan",
    "colspan",
    "role",
    "text",
    "ix",
    "max_cols",
    "initial_col_indices",
]


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
# iXBRL metadata extraction
# ---------------------------------------------------------------------------

def _extract_ix(cell_tag) -> Optional[str]:
    """Return a compact string of iXBRL attributes if present, else None."""
    ix = cell_tag.find(
        lambda t: getattr(t, "name", "") and str(t.name).lower().startswith("ix:")
    )
    if not ix:
        return None
    parts = {}
    for attr in ("name", "contextref", "unitref", "decimals", "scale", "sign", "id"):
        v = ix.get(attr)
        if v is not None:
            parts[attr] = str(v)
    return str(parts) if parts else None


# ---------------------------------------------------------------------------
# HTML table parsing → flat cell list
# ---------------------------------------------------------------------------

class _Cell(NamedTuple):
    cell_id: int
    row_start: int
    col_start: int
    rowspan: int
    colspan: int
    text: str
    th: bool
    ix: Optional[str]


def _iter_own_trs(table_el):
    """Yield <tr> elements that directly belong to table_el, skipping nested tables."""
    for tr in table_el.find_all("tr"):
        for ancestor in tr.parents:
            if ancestor is table_el:
                yield tr
                break
            if ancestor.name == "table":
                break  # belongs to a nested table, skip


def _parse_html_table(table_el) -> list[_Cell]:
    """
    Parse a single <table> BeautifulSoup element into a list of origin cells.

    Only origin cells are emitted (one per logical cell, not one per grid position).
    rowspan/colspan reflect the original HTML values.
    Nested <table> elements inside cells are ignored — their text folds into the
    outer cell via get_text(), matching how extract_boxes.js attributes those boxes.
    """
    cells: list[_Cell] = []
    cell_id = 0
    # pending[col] = remaining rows this column is still occupied by a rowspan cell
    pending: dict[int, int] = {}

    for r_idx, tr in enumerate(_iter_own_trs(table_el)):
        tds = tr.find_all(["td", "th"], recursive=False)
        if not tds:
            tds = tr.find_all(["td", "th"])

        c = 0
        for td in tds:
            # Advance past columns still occupied by rowspan cells
            while c in pending:
                c += 1

            colspan = max(1, int(td.get("colspan") or 1))
            rowspan = max(1, int(td.get("rowspan") or 1))
            text = _norm_ws(td.get_text(" ", strip=True))
            is_th = td.name == "th"
            ix = _extract_ix(td)

            cells.append(
                _Cell(
                    cell_id=cell_id,
                    row_start=r_idx,
                    col_start=c,
                    rowspan=rowspan,
                    colspan=colspan,
                    text=text,
                    th=is_th,
                    ix=ix,
                )
            )
            cell_id += 1

            # Register occupied columns for remaining rowspan rows
            if rowspan > 1:
                for dc in range(colspan):
                    pending[c + dc] = rowspan - 1

            c += colspan

        # End of row: decrement all pending rowspan counters
        for col in list(pending.keys()):
            pending[col] -= 1
            if pending[col] <= 0:
                del pending[col]

    return cells


def _cells_to_df(cells: list[_Cell]) -> pd.DataFrame:
    if not cells:
        return pd.DataFrame(
            columns=["cell_id", "row_start", "col_start", "rowspan", "colspan", "text", "th", "ix"]
        )
    return pd.DataFrame([c._asdict() for c in cells])


# ---------------------------------------------------------------------------
# Blank-row removal
# ---------------------------------------------------------------------------

def _remove_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where every origin cell has empty text AND no prior rowspan
    cell extends into that row.
    """
    if df.empty:
        return df

    row_start_arr = df["row_start"].values
    rowspan_arr = df["rowspan"].values

    # Rows that contain at least one non-blank cell
    rows_with_text: set[int] = set(
        df.loc[df["text"].str.strip() != "", "row_start"].unique().tolist()
    )

    all_rows = sorted(set(row_start_arr.tolist()))
    row_ends = row_start_arr + rowspan_arr - 1  # inclusive end per cell

    rows_to_remove: set[int] = set()
    for r in all_rows:
        if r in rows_with_text:
            continue
        # Keep if any prior cell spans into this row
        if np.any((row_start_arr < r) & (row_ends >= r)):
            continue
        rows_to_remove.add(r)

    if not rows_to_remove:
        return df

    keep = ~np.isin(row_start_arr, list(rows_to_remove))
    df = df[keep].copy()
    if df.empty:
        return df.reset_index(drop=True)

    rs = df["row_start"].values
    rsp = df["rowspan"].values

    # Build a lookup array for remapping old row indices to new contiguous ones
    surviving = sorted(set(rs.tolist()))
    row_remap = np.full(surviving[-1] + 1, -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(surviving):
        row_remap[old_idx] = new_idx

    # Cumulative-sum trick: surviving[i] = 1 for kept rows, 0 for removed.
    # new_span = sum of kept flags in [r0, r_end].
    max_row_end = int((rs + rsp - 1).max())
    kept_flags = np.ones(max_row_end + 2, dtype=np.int64)
    for r in rows_to_remove:
        if r <= max_row_end:
            kept_flags[r] = 0
    cumkept = np.concatenate([[0], np.cumsum(kept_flags)])

    r_end = np.minimum(rs + rsp - 1, max_row_end)
    new_rowspans = np.maximum(1, cumkept[r_end + 1] - cumkept[rs])

    df["row_start"] = row_remap[rs]
    df["rowspan"] = new_rowspans
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blank/paren spacer columns and merge paren values into adjacent cells."""
    df = df.copy()
    max_cols = int((df["col_start"] + df["colspan"]).max())
    df["max_cols"] = max_cols
    df["initial_col_indices"] = [
        list(range(cs, cs + sp))
        for cs, sp in zip(df["col_start"].to_numpy(), df["colspan"].to_numpy())
    ]

    # Pre-extract columns as plain Python lists/arrays — avoids per-row Series overhead
    all_indices_list: list[list[int]] = df["initial_col_indices"].tolist()
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
    df["max_cols"] = len(surviving)
    df["initial_col_indices"] = [
        [pos[i] for i in all_indices_list[i] if i in surviving_set]
        for i in kept_positions
    ]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# table_id mapping: original HTML table index → final df_lines table_id
# ---------------------------------------------------------------------------

def _build_orig_to_final_map(df_lines: pd.DataFrame) -> dict[int, int]:
    """
    Map original_table_id (= JS 1-indexed table counter = HTML document-order
    index + 1) to the final reindexed table_id in df_lines.

    Uses the original_table_id column preserved by _remove_single_row_tables,
    so no join against df_boxes is needed.

    Returns {orig_table_id: final_table_id} for surviving tables only.
    """
    if "original_table_id" not in df_lines.columns or "table_id" not in df_lines.columns:
        return {}
    valid = df_lines[
        df_lines["original_table_id"].notna() & df_lines["table_id"].notna()
    ][["original_table_id", "table_id"]].drop_duplicates("original_table_id")
    if valid.empty:
        return {}
    return dict(zip(valid["original_table_id"].astype(int), valid["table_id"].astype(int)))


def _build_final_to_origs(orig_to_final: dict[int, int]) -> dict[int, list[int]]:
    """
    Invert orig→final map to final→[sorted list of orig_table_ids].

    Merged tables (multiple single-row HTML <table> elements combined into one
    final table by _remove_single_row_tables) will have len > 1.
    """
    result: dict[int, list[int]] = {}
    for orig, final in orig_to_final.items():
        result.setdefault(final, []).append(orig)
    for final in result:
        result[final].sort()
    return result


# ---------------------------------------------------------------------------
# Per-table page info
# ---------------------------------------------------------------------------

def _table_page_info(df_lines: pd.DataFrame) -> dict[int, tuple]:
    """Return {final_table_id: (page_number, page_label)} from df_lines."""
    info: dict[int, tuple] = {}
    has_label = "page_label" in df_lines.columns
    for tid, grp in df_lines[df_lines["table_id"].notna()].groupby("table_id"):
        pn = grp["page_number"].iloc[0] if "page_number" in grp.columns else None
        pl = grp["page_label"].iloc[0] if has_label else None
        info[int(tid)] = (pn, pl)
    return info


def _table_row_ids(df_lines: pd.DataFrame) -> dict[int, list[int]]:
    """Return {final_table_id: [table_row_id, ...]} sorted by row appearance."""
    row_ids: dict[int, list[int]] = {}
    has_rid = "table_row_id" in df_lines.columns
    if not has_rid:
        return row_ids
    for tid, grp in df_lines[df_lines["table_id"].notna()].groupby("table_id"):
        rids = grp["table_row_id"].dropna().astype(int).tolist()
        row_ids[int(tid)] = rids
    return row_ids


# ---------------------------------------------------------------------------
# Multi-part table assembly (for merged single-row HTML tables)
# ---------------------------------------------------------------------------

def _combine_table_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Stack cell DataFrames from multiple HTML <table> elements vertically,
    adjusting row_start so rows are contiguous.

    Used when _remove_single_row_tables merged N consecutive single-row HTML
    tables into one final table — each part contributes one logical row.
    """
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    combined = []
    offset = 0
    for df in parts:
        if df.empty:
            continue
        df = df.copy()
        max_row_end = int((df["row_start"] + df["rowspan"] - 1).max())
        df["row_start"] = df["row_start"] + offset
        combined.append(df)
        offset += max_row_end + 1
    if not combined:
        return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_table_cells(
    df_lines: pd.DataFrame,
    rendered_html: str,
) -> Optional[pd.DataFrame]:
    """
    Extract and normalize HTML tables into a cell-level DataFrame whose
    table_id, table_row_id, page_number, and page_label match df_lines.

    Each HTML <table> element carries a data-docslicer-table-id attribute set by
    extract_boxes.js (same value as the JS tableCounter for that element). This
    attribute-based lookup is immune to nested-table ordering differences between
    JS assignment order and BS4 document-order indexing.

    Which tables survive is determined entirely by df_lines (via original_table_id →
    table_id). No additional row-count filtering is applied here.

    Args:
        df_lines:      Line-level DataFrame from step 04. Must have
                       original_table_id and table_id columns.
        rendered_html: Rendered HTML string for parsing table structure.
        df_boxes:      Unused; kept for call-site compatibility.

    Returns:
        DataFrame with one row per logical table cell, or None if no tables found.
    """
    if not rendered_html or df_lines is None:
        return None

    # Build original_table_id → final table_id map directly from df_lines
    orig_to_final = _build_orig_to_final_map(df_lines)
    if not orig_to_final:
        return None

    # Invert: final_tid → [orig_tids sorted by document order]
    # Merged tables have multiple orig_tids; normal tables have exactly one.
    final_to_origs = _build_final_to_origs(orig_to_final)

    page_info = _table_page_info(df_lines)
    row_ids_by_table = _table_row_ids(df_lines)

    # Build a lookup: JS table_id → <table> element, using the
    # data-docslicer-table-id attribute stamped by extract_boxes.js.
    soup = BeautifulSoup(rendered_html, "lxml")
    table_by_js_id: dict[int, any] = {}
    for tbl in soup.find_all("table"):
        attr = tbl.get("data-docslicer-table-id")
        if attr is not None:
            try:
                table_by_js_id[int(attr)] = tbl
            except (ValueError, TypeError):
                pass

    all_cells: list[pd.DataFrame] = []
    global_cell_id = 1

    for final_tid, orig_tids in sorted(final_to_origs.items()):
        # Collect parsed cell DataFrames for each HTML table that feeds this final table
        parts: list[pd.DataFrame] = []
        for orig_tid in orig_tids:
            tbl_el = table_by_js_id.get(orig_tid)
            if tbl_el is None:
                continue
            raw_cells = _parse_html_table(tbl_el)
            if raw_cells:
                parts.append(_cells_to_df(raw_cells))

        if not parts:
            continue

        # Stack parts (handles merged single-row tables)
        df = _combine_table_parts(parts)
        if df.empty:
            continue

        # Remove blank rows
        df = _remove_blank_rows(df)
        if df.empty:
            continue

        # Normalize columns
        df = _normalize_columns(df)
        if df.empty:
            continue

        # Detect cell roles
        df = detect_cell_roles(df, with_row_label=True)

        # Assign global cell_ids (sequential across all tables)
        n = len(df)
        df["table_cell_id"] = range(global_cell_id, global_cell_id + n)
        global_cell_id += n

        df["table_id"] = final_tid

        # Map row_start index → table_row_id from df_lines
        row_id_list = row_ids_by_table.get(final_tid, [])
        unique_rows = sorted(df["row_start"].unique())
        row_to_rid: dict[int, Optional[int]] = {
            r: (row_id_list[i] if i < len(row_id_list) else None)
            for i, r in enumerate(unique_rows)
        }
        df["table_row_id"] = df["row_start"].map(row_to_rid)

        pn, pl = page_info.get(final_tid, (None, None))
        df["page_number"] = pn
        df["page_label"] = pl

        present = [c for c in _OUTPUT_COLS if c in df.columns]
        all_cells.append(df[present])

    if not all_cells:
        return None

    result = pd.concat(all_cells, ignore_index=True)

    for col in _OUTPUT_COLS:
        if col not in result.columns:
            result[col] = None

    return result[_OUTPUT_COLS].reset_index(drop=True)

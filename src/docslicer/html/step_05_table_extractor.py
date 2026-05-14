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

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .._utils.table_utils import detect_cell_roles

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
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
]


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").replace("\xa0", " ")).strip()


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


def _parse_html_table(table_el) -> list[_Cell]:
    """
    Parse a single <table> BeautifulSoup element into a list of origin cells.

    Only origin cells are emitted (one per logical cell, not one per grid position).
    rowspan/colspan reflect the original HTML values.
    """
    cells: list[_Cell] = []
    cell_id = 0
    # pending[col] = remaining rows this column is still occupied by a rowspan cell
    pending: dict[int, int] = {}

    for r_idx, tr in enumerate(table_el.find_all("tr", recursive=True)):
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

    rows_to_remove: set[int] = set()
    all_rows = sorted(df["row_start"].unique())

    for r in all_rows:
        origin_cells = df[df["row_start"] == r]
        if origin_cells.empty:
            continue
        if origin_cells["text"].str.strip().ne("").any():
            continue
        # Check that no cell from a previous row spans into r
        spans_into = (
            (df["row_start"] < r) & (df["row_start"] + df["rowspan"] > r)
        ).any()
        if not spans_into:
            rows_to_remove.add(r)

    if not rows_to_remove:
        return df

    df = df[~df["row_start"].isin(rows_to_remove)].copy()

    # Re-index row_start and adjust rowspans
    surviving = sorted(set(df["row_start"].unique()))
    row_remap = {old: new for new, old in enumerate(surviving)}

    def _new_rowspan(row):
        r0 = row["row_start"]
        r_end = r0 + row["rowspan"] - 1
        new_r0 = row_remap[r0]
        # Count surviving rows in [r0, r_end]
        new_span = sum(1 for r in range(r0, r_end + 1) if r not in rows_to_remove)
        return pd.Series({"row_start": new_r0, "rowspan": max(1, new_span)})

    adjusted = df.apply(_new_rowspan, axis=1)
    df["row_start"] = adjusted["row_start"]
    df["rowspan"] = adjusted["rowspan"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _profile_columns(df: pd.DataFrame) -> dict[int, dict]:
    """Profile each column: all_blank / currency_only / lparen_only / rparen_only."""
    profiles: dict[int, dict] = {}
    for col in sorted(df["col_start"].unique()):
        standalone = df[(df["col_start"] == col) & (df["colspan"] == 1)]
        nonblank = standalone[standalone["text"].str.strip() != ""]
        if nonblank.empty:
            profiles[col] = dict(all_blank=True, currency_only=False, lparen_only=False, rparen_only=False)
        else:
            texts = nonblank["text"].tolist()
            profiles[col] = dict(
                all_blank=False,
                currency_only=all(_is_currency(t) for t in texts),
                lparen_only=all(_is_lparen(t) for t in texts),
                rparen_only=all(_is_rparen_like(t) for t in texts),
            )
    return profiles


def _find_next_data_col(profiles: dict[int, dict], from_col: int, direction: int) -> Optional[int]:
    cols = sorted(profiles.keys())
    candidates = [c for c in cols if (c > from_col if direction > 0 else c < from_col)]
    if direction < 0:
        candidates = list(reversed(candidates))
    for c in candidates:
        p = profiles[c]
        if not (p["all_blank"] or p["currency_only"] or p["lparen_only"] or p["rparen_only"]):
            return c
    return None


def _merge_col_text(df: pd.DataFrame, src_col: int, tgt_col: int, prefix: bool) -> pd.DataFrame:
    """Prepend (prefix=True) or append the text of standalone src_col cells into tgt_col cells."""
    df = df.copy()
    src_cells = df[(df["col_start"] == src_col) & (df["colspan"] == 1) & (df["text"].str.strip() != "")]

    for _, src in src_cells.iterrows():
        r = src["row_start"]
        tgt_mask = (
            (df["col_start"] <= tgt_col)
            & (df["col_start"] + df["colspan"] > tgt_col)
            & (df["row_start"] <= r)
            & (df["row_start"] + df["rowspan"] > r)
        )
        if tgt_mask.any():
            if prefix:
                df.loc[tgt_mask, "text"] = src["text"] + df.loc[tgt_mask, "text"]
            else:
                df.loc[tgt_mask, "text"] = df.loc[tgt_mask, "text"] + src["text"]

    return df


def _remove_col(df: pd.DataFrame, col: int) -> pd.DataFrame:
    """
    Remove column col from the cells DataFrame:
    - standalone cells at col are dropped
    - cells spanning through col get colspan - 1
    - cells starting after col get col_start - 1
    """
    df = df.copy()

    # Drop standalone cells at col
    standalone = (df["col_start"] == col) & (df["colspan"] == 1)
    df = df[~standalone].copy()

    # Cells spanning through col: reduce colspan
    covers = (df["col_start"] <= col) & (df["col_start"] + df["colspan"] > col)
    df.loc[covers, "colspan"] -= 1

    # Cells starting strictly after col: shift left
    df.loc[df["col_start"] > col, "col_start"] -= 1

    return df.reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively merge currency/paren columns and remove blank spacer columns."""
    while True:
        profiles = _profile_columns(df)
        changed = False

        # Currency → merge into right neighbor
        for col, p in sorted(profiles.items()):
            if not p["currency_only"]:
                continue
            tgt = _find_next_data_col(profiles, col, direction=1)
            if tgt is None:
                continue
            df = _merge_col_text(df, col, tgt, prefix=True)
            df = _remove_col(df, col)
            changed = True
            break
        if changed:
            continue

        # Left paren → merge into right neighbor
        for col, p in sorted(profiles.items()):
            if not p["lparen_only"]:
                continue
            tgt = _find_next_data_col(profiles, col, direction=1)
            if tgt is None:
                continue
            df = _merge_col_text(df, col, tgt, prefix=True)
            df = _remove_col(df, col)
            changed = True
            break
        if changed:
            continue

        # Right paren → merge into left neighbor
        for col, p in sorted(profiles.items()):
            if not p["rparen_only"]:
                continue
            tgt = _find_next_data_col(profiles, col, direction=-1)
            if tgt is None:
                continue
            df = _merge_col_text(df, col, tgt, prefix=False)
            df = _remove_col(df, col)
            changed = True
            break
        if changed:
            continue

        # Blank spacers → remove
        for col, p in sorted(profiles.items()):
            if not p["all_blank"]:
                continue
            df = _remove_col(df, col)
            changed = True
            break
        if changed:
            continue

        break  # stable

    return df


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
    df_boxes: pd.DataFrame = None,
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

"""
text_merge.py

Shared helpers for aggregating lower-level text fragments into a single text
field. Intended to be reused across the HTML / PDF / DOCX / PPTX line builders
and the shared block / chunk stages, so the joining rules (separators, inline
markup, table rows, de-hyphenation) live in one place instead of being
re-implemented (and quietly diverging) per stage.

Two layers of merge:

  * Horizontal join — sibling fragments on one logical row (words -> cell,
    cells -> line, paragraphs -> table row). See :func:`join_fragments` and
    :func:`join_table_row`.
  * Vertical join — stacked lines into a paragraph/block, where a line that
    begins with a bullet starts on its own new line. See :func:`join_lines`.

Inline markup (sub/superscript, strikethrough) is applied per-fragment *before*
joining via :func:`apply_inline_markup`, which is fully vectorized so it can run
on a whole column ahead of the groupby aggregation (the same pattern PDF's cell
builder already uses with its ``_fmt_text`` column).

Nothing here is wired into the pipeline yet — these are the building blocks.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .text_utils import is_bullet_line

# Script tokens emitted by apply_inline_markup that attach to the previous
# fragment with no intervening space.
_SCRIPT_PREFIXES = ("[^", "[_")


# ==================================================
# INLINE MARKUP  (per-fragment, vectorized)
# ==================================================

def apply_inline_markup(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    script_col: str = "script_type",
    strikethrough_col: str = "is_strikethrough",
) -> pd.Series:
    """
    Return a text Series with inline markup applied per row:

        script_type == "superscript"  ->  [^text]
        script_type == "subscript"    ->  [_text]
        is_strikethrough == True      ->  ~~text~~

    Strikethrough is applied first (innermost), then the script wrap, so a
    fragment that is both becomes ``[^~~text~~]``. Empty / whitespace fragments
    are left untouched. Missing columns are skipped, so the same call works for
    HTML / PDF / DOCX / PPTX regardless of which markup columns they carry.

    Vectorized (boolean masks + string concat) so it can be computed once on the
    full DataFrame before a groupby aggregation::

        df["text"] = apply_inline_markup(df)
        # ... then aggregate "text" with join_fragments ...
    """
    out = df[text_col].astype("string").fillna("")
    # Plain bool (no <NA>): Series.mask treats <NA> conditions as True, which
    # would wrap every row, so every mask below is coerced with fillna(False).
    nonblank = out.str.strip().ne("").fillna(False)

    if strikethrough_col in df.columns:
        strike = df[strikethrough_col].fillna(False).astype(bool) & nonblank
        out = out.mask(strike, "~~" + out + "~~")

    if script_col in df.columns:
        script = df[script_col].astype("string")
        sup = (script == "superscript").fillna(False) & nonblank
        sub = (script == "subscript").fillna(False) & nonblank
        out = out.mask(sup, "[^" + out + "]")
        out = out.mask(sub, "[_" + out + "]")

    return out.astype(str)


# ==================================================
# HORIZONTAL JOIN  (sibling fragments -> one field)
# ==================================================

def join_fragments(
    texts: Iterable[object],
    sep: str = " ",
    *,
    dehyphenate: bool = False,
) -> str:
    """
    Join sibling text fragments left-to-right into a single string.

    Rules:
      * Empty / whitespace-only fragments are skipped.
      * Fragments wrapped as script tokens (``[^...]`` / ``[_...]``) attach to
        the previous fragment with no separator.
      * ``dehyphenate=True``: when the running text ends with a hyphen, the next
        fragment is joined directly with no space — a word split across lines,
        e.g. ``"inter-" + "national" -> "inter-national"``. The hyphen is kept
        (matching the current PDF ``_join_texts``). PDF will set this.

    `texts` is typically a groupby column (Series/list) already in reading order.
    """
    result = ""
    for t in texts:
        if t is None:
            continue
        ts = str(t)
        if not ts.strip():
            continue
        if not result:
            result = ts
        elif dehyphenate and result.endswith("-"):
            result = result + ts
        elif ts[:2] in _SCRIPT_PREFIXES:
            result = result + ts
        else:
            result = result + sep + ts
    return result


# ==================================================
# TABLE ROW JOIN  (cells -> pipe-delimited row)
# ==================================================

def join_table_row(cell_texts: Iterable[object], sep: str = " | ") -> str:
    """
    Join already-assembled cell strings into a pipe-delimited table row, dropping
    empty cells. Replaces the duplicated table-row joining in the HTML / DOCX /
    PPTX line builders. (Build each cell's own text with :func:`join_fragments`
    first, then pass the cell strings here.)
    """
    cells = []
    for c in cell_texts:
        if c is None:
            continue
        cs = str(c).strip()
        if cs:
            cells.append(cs)
    return sep.join(cells)


# ==================================================
# VERTICAL JOIN  (stacked lines -> block, bullet-aware)
# ==================================================

def join_lines(
    texts: Iterable[object],
    sep: str = " ",
    bullet_sep: str = "\n",
) -> str:
    """
    Join stacked line texts into one block.

    A line whose first non-space character is a bullet glyph (or that is a
    standalone bullet) is prefixed with `bullet_sep` (newline) instead of `sep`,
    so each bulleted item starts on its own line. All other lines join with
    `sep`. Empty lines are skipped.

    Bullet detection uses :func:`text_utils.is_bullet_line` (O(1) per line —
    frozenset lookups, no regex), so this stays cheap even on long blocks.
    """
    parts: list[str] = []
    for t in texts:
        if t is None:
            continue
        ts = str(t).strip()
        if not ts:
            continue
        if not parts:
            parts.append(ts)
        elif is_bullet_line(ts):
            parts.append(bullet_sep + ts)
        else:
            parts.append(sep + ts)
    return "".join(parts)


__all__ = [
    "apply_inline_markup",
    "join_fragments",
    "join_table_row",
    "join_lines",
]

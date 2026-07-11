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

Each join exists in two forms with identical semantics:

  * a scalar reference implementation (``join_fragments`` / ``join_table_row``
    / ``join_lines``) that processes one group — readable, and the contract the
    tests check against;
  * a grouped vectorized form (``merge_fragments`` / ``merge_table_rows`` /
    ``merge_lines``) that processes a whole column at once via
    :func:`registry_aggregator.group_join` — no per-group Python, so it is the
    one to use in pipelines. Returns a Series indexed by group key; ``.map`` it
    onto the aggregated frame.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..text_utils import is_bullet_line, is_strict_bullet, strict_bullet_mask
from .registry_aggregator import group_join

# Script tokens emitted by apply_inline_markup that attach to the previous
# fragment with no intervening space.
_SCRIPT_PREFIXES = ("[^", "[_")

# Superscript fragments that are typography, not reference marks — ordinal
# suffixes ("15th", "2nd") and trademark/copyright glyphs ("FINTEPLA®").
# apply_inline_markup leaves these unwrapped so they read "15th" / "FINTEPLA®",
# not "15[^th]" / "FINTEPLA[^®]".
_PLAIN_SUPERSCRIPTS = frozenset({"st", "nd", "rd", "th", "®", "™", "©"})


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

    Ordinal suffixes ("th"/"st"/"nd"/"rd") and trademark/copyright glyphs
    ("®"/"™"/"©") marked as superscript are typography, not a reference mark,
    so they are left unwrapped (plain "th"/"®", not "[^th]"/"[^®]").

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
        is_plain = df[text_col].astype("string").fillna("").str.strip().str.lower().isin(_PLAIN_SUPERSCRIPTS)
        sup = (script == "superscript").fillna(False) & nonblank & ~is_plain
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
    bullet_sep: str | None = None,
) -> str:
    """
    Join sibling text fragments left-to-right into a single string.

    Rules:
      * Empty / whitespace-only fragments are skipped.
      * ``bullet_sep`` (e.g. ``"\\n"``): a fragment starting with an
        *unambiguous* bullet glyph (:func:`text_utils.is_strict_bullet` — ▪ •
        ► …, deliberately narrower than the line-level bullet set) is prefixed
        with ``bullet_sep`` instead of ``sep``. For cells that pack several
        visual lines into one field (tagged-PDF TD/TH); mid-line ``+``/``-``
        stay untouched ("+2y", "A-C").
      * Fragments wrapped as script tokens (``[^...]`` / ``[_...]``) attach to
        the previous fragment with no separator.
      * ``dehyphenate=True``: when the previous fragment is a word ending in a
        hyphen (more than just ``"-"``), the next fragment is joined directly
        with no space — a word split across lines, e.g. ``"inter-" +
        "national" -> "inter-national"``. The hyphen is kept; a standalone
        ``"-"`` dash never glues. The PDF cell builder sets this.

    `texts` is typically a groupby column (Series/list) already in reading order.
    """
    result = ""
    prev = ""
    for t in texts:
        if t is None or t != t:  # skip None and NaN
            continue
        ts = str(t)
        if not ts.strip():
            continue
        if not result:
            result = ts
        elif bullet_sep is not None and is_strict_bullet(ts):
            result = result + bullet_sep + ts
        elif dehyphenate and len(prev) > 1 and prev.endswith("-"):
            result = result + ts
        elif ts[:2] in _SCRIPT_PREFIXES:
            result = result + ts
        else:
            result = result + sep + ts
        prev = ts
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
        if c is None or c != c:  # skip None and NaN
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
        if t is None or t != t:  # skip None and NaN
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


# ==================================================
# GROUPED VECTORIZED FORMS  (whole column at once)
# ==================================================

def _as_str(texts: pd.Series) -> pd.Series:
    """None/NA → "", everything else → str (mirrors the scalar joins)."""
    return texts.astype("string").fillna("").astype(str)


def _fill_missing_groups(result: pd.Series, keys: pd.Series) -> pd.Series:
    """Groups whose fragments were all blank yield "" (as the scalar joins do)."""
    full = pd.Index(keys.dropna().unique(), name=keys.name)
    if len(result) == len(full):
        return result
    return result.reindex(full, fill_value="")


def merge_text_within_line(
    texts: pd.Series,
    keys: pd.Series,
    sep: str = " ",
    *,
    dehyphenate: bool = False,
    bullet_sep: str | None = None,
) -> pd.Series:
    """
    Grouped, vectorized :func:`join_fragments`: one joined string per group key.

    Apply :func:`apply_inline_markup` to the column first if script/strike
    markup is wanted — script tokens then attach with no separator here.
    """
    s = _as_str(texts)
    keep = (s.str.strip() != "").to_numpy(dtype=bool)
    t, k = s[keep], keys[keep]

    attach = t.str.startswith(_SCRIPT_PREFIXES)
    if dehyphenate:
        # Previous kept token within the same group (groups need not be
        # contiguous in the frame); NaN at group starts → False. A standalone
        # "-" never glues — only real words ending in a hyphen do.
        prev = t.groupby(k, sort=False).shift(1)
        attach = attach | (
            prev.str.endswith("-") & (prev.str.len() > 1)
        ).fillna(False).astype(bool)

    seps: object = sep
    if bullet_sep is not None:
        bullet = strict_bullet_mask(t).to_numpy(dtype=bool)
        attach = attach & ~bullet                # bullets always start fresh
        seps = np.where(bullet, bullet_sep, sep)

    return _fill_missing_groups(group_join(t, k, sep=seps, attach_mask=attach), keys)


def merge_table_rows(
    cell_texts: pd.Series,
    keys: pd.Series,
    sep: str = " | ",
) -> pd.Series:
    """Grouped, vectorized :func:`join_table_row`: one pipe-delimited row per key."""
    s = _as_str(cell_texts).str.strip()
    keep = (s != "").to_numpy(dtype=bool)
    return _fill_missing_groups(group_join(s[keep], keys[keep], sep=sep), keys)


def merge_text_across_lines(
    texts: pd.Series,
    keys: pd.Series,
    sep: str = " ",
    bullet_sep: str = "\n",
) -> pd.Series:
    """
    Grouped, vectorized :func:`join_lines`: bullet lines start on a new line,
    everything else joins with ``sep``. is_bullet_line runs once per row (cheap
    frozenset checks), not once per group.
    """
    s = _as_str(texts).str.strip()
    keep = (s != "").to_numpy(dtype=bool)
    t, k = s[keep], keys[keep]

    seps = np.where(t.map(is_bullet_line).to_numpy(dtype=bool), bullet_sep, sep)
    return _fill_missing_groups(group_join(t, k, sep=seps), keys)


__all__ = [
    "apply_inline_markup",
    "join_fragments",
    "join_table_row",
    "join_lines",
    "merge_fragments",
    "merge_table_rows",
    "merge_lines",
]

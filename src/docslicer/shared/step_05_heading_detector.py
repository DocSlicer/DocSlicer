# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
step_05_heading_detector.py

Answers the question: is this line a heading?

Public API: detect_headings(lines_df, compiled_patterns)

Pipeline:
  1. Detect hierarchy markers (numbered, parenthetical, named)
  2. Correct paren alpha/roman misclassification via series context
  3. Validate alpha_heading markers form a consecutive A→B→C series
  4. Add gap context (line_gap_below), on the full frame before prefiltering
  5. Score lines (intrinsic per-line signals, on the prefiltered subset)
  6. Contextual score adjustments (isolated-style bonus, style-run penalty)
  7. Heading decision (threshold)
  8. Numbered section groups + hybrid heading text extraction
  9. Heading fingerprints and fp_ids
 10. Group multi-line headings into heading_id
 11. Suppress repeated / coverpage / magazine-page / per-slide headings
 12. Finalize block roles (blanks → 'paragraph')

heading_id is assigned here rather than in step_06 so that suppression can act on
whole heading units: a named heading and its blank-marker subtitle share an id and
are kept or dropped together. The hierarchy tree itself (weights, parent/level)
still lives in step_06.
"""

from __future__ import annotations
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .._utils.text_utils import _BULLET_PREFIX_CHARS, numeric_line_mask

# ================================================================================
# Pre-filter forbidden heading line formats (text that will never be a heading)
# ================================================================================

_FORBIDDEN_BLOCK_TYPES = {
    # Page furniture
    "image", "hr", "page_label", "navigation", "vertical_text", 
    # Out-of-flow / peripheral regions
    "header", "footer", "footnote", "speaker_notes",
    # Verbatim / non-prose content
    "code", "block_quote", "shape", "form_field", "comment",
    # Structured / reference blocks
    "table", "chart", "toc", "exhibits", 
    }

# ------------------------------------------------------------------------------
# Forbidden content anywhere in the line
#
# Two different questions, deliberately answered two different ways:
#
#   * Signature / online markers are near-zero-false-positive *literals* — a line
#     containing "/s/" or "http://" is never a heading. One hard-kill regex.
#   * Symbol residue is a matter of *degree*. A single Greek letter is ordinary
#     prose ("alpha-Diversity Analysis"); a line that is mostly operators is a
#     formula. Scored over the whole line, not tripped by the first hit.
#
# NOTE: page furniture is deliberately absent. It is already handled upstream by
# block_type (page_label / header / footer in _FORBIDDEN_BLOCK_TYPES) and by the
# dedicated step_11 page-label detector. A bare "page" substring here also killed
# legitimate headings ("Homepage Redesign", "Per-Page Pricing").
# ------------------------------------------------------------------------------

_SIGNATURE_SUBSTRINGS = frozenset({
    "/s/",                              # conformed signature mark
    "signed:",
    "signature of",
})

_ONLINE_SUBSTRINGS = frozenset({
    "http",                             # covers http:// and https:// in one token
    "www.",
    "mailto:",
})

# Shaped patterns rather than bare fragments: a real email/domain needs a label
# on both sides of the separator. Bare "@" killed "Sales@Scale Initiative", and
# a bare ".com" ignored every other TLD.
_EMAIL_PAT = r"[^\s@]+@[^\s@]+\.[a-z]{2,}"
_DOMAIN_PAT = r"\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co\.uk)\b"

# One pre-compiled alternation → a single vectorized pass over the column.
# Pre-compiling (rather than passing a pattern string) also keeps this off
# pandas' ASCII-only RE2 path and its silent fallback.
_FORBIDDEN_CONTENT_RE = re.compile(
    "|".join([
        *(re.escape(t) for t in sorted(_SIGNATURE_SUBSTRINGS)),
        *(re.escape(t) for t in sorted(_ONLINE_SUBSTRINGS)),
        _EMAIL_PAT,
        _DOMAIN_PAT,
    ]),
    re.IGNORECASE,
)

# Math / scientific symbol residue. Greek letters plus common operators —
# principled coverage instead of the previous arbitrary two ("α", "π").
# Note U+2212 MINUS SIGN, which is NOT the ASCII hyphen and is what most PDF
# producers actually emit in formulas.
_SYMBOL_RE = re.compile(
    "["
    "α-ωΑ-Ω"                            # Greek lower / upper
    "∑∏∫√∞∂∇"                           # operators: sum, product, integral, root, …
    "≈≠≤≥≡∝"                            # relations
    "±×÷⋅·−"                            # arithmetic (incl. U+2212 minus)
    "∈∉⊂⊃∪∩∅∀∃"                         # set / logic
    "°µ‰Δ"                              # units & deltas
    "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"              # super / subscripts
    "]"
)

# An equals sign is a much stronger structural signal than density: real
# equations dilute their own symbol ratio with ASCII variable names
# ("G = αS + (1 − α)E" is only 12% symbols), so density alone misses them.
# Excludes ==, <=, >=, != so comparison-style prose isn't caught.
_EQUATION_RE = re.compile(r"(?<![<>!=])=(?!=)")

# A line is formula residue when EITHER:
#   * it clears both density bars — enough symbols that they aren't incidental,
#     and a high enough share of the line that it isn't prose; or
#   * it is equation-shaped — an equals sign plus at least one math symbol.
_SYMBOL_MIN_COUNT = 2
_SYMBOL_MAX_RATIO = 0.15
_EQUATION_MIN_SYMBOLS = 1

# Pipe residue from table/layout flattening. One pipe is a legitimate heading
# separator ("Revenue | EMEA"); several mean a flattened table row.
_PIPE_MAX_COUNT = 1

# Start tokens specific to heading detection. The bullet glyphs come from
# text_utils (_BULLET_PREFIX_CHARS = the non-alphanumeric subset of the canonical
# bullet set, so the OCR bullet "o" can't strip "October"/"Overview").
_FORBIDDEN_START_EXTRA = {
    # Quotes — a line opening on a quote mark is pulled prose, not a heading.
    "'", '"',                           # straight single / double
    "“", "”",                           # curly double  (left / right)
    "‘", "’",                           # curly single  (left / right)
    "„",                                # low-9 double  (German open quote)
    "«", "»",                           # guillemets    (left / right)

    # Reference & legal marks — footnote refs, notices, section callouts.
    "©", "®", "™",                      # copyright / registered / trademark
    "§", "¶",                           # section sign / pilcrow (paragraph mark)
    "†", "‡",                           # dagger / double dagger (footnote refs)
    "‹", "›",                           # single guillemets (left / right)
    "|",                                # pipe — table or layout residue
    "¤",                                # currency sign (mojibake artifact)

    # Signature-block indicators — always a signature line, never a heading.
    "By:", "Name:", "Title:", "Date:",

    # Private-use bullet glyph (U+F09F) — kept as an escape because the literal
    # renders as an invisible/tofu box in most editors. text_utils lists this
    # slot as "" (the glyph was lost from the literal), so the canonical set
    # can't supply it — keep here until that entry is repaired.
    "\uf09f",
}

# Guard against empty/lost entries so a zero-length token can't slip into either
# matcher bucket below.
_FORBIDDEN_START_TEXT = {
    t for t in (_BULLET_PREFIX_CHARS | _FORBIDDEN_START_EXTRA) if t
}

# Split once at import so the row-wise test stays vectorized: single-char tokens
# go through a hash lookup on the first character, multi-char tokens through one
# anchored regex. This keeps the non-ASCII glyphs off pandas' ASCII-only RE2
# path (which silently falls back to Python re on Unicode).
_FORBIDDEN_START_CHARS = frozenset(
    t.lower() for t in _FORBIDDEN_START_TEXT if len(t) == 1
)
_FORBIDDEN_START_MULTI_RE = re.compile(
    r"^(?:%s)" % "|".join(
        re.escape(t)
        for t in sorted({t.lower() for t in _FORBIDDEN_START_TEXT if len(t) > 1})
    )
)

# Parenthesized line patterns:
# fully enclosed in (), [], {} (after trimming whitespace)
_PAREN_FULL_RE = re.compile(r"^\s*(\((?s:.*)\)|\[(?s:.*)\]|\{(?s:.*)\})\s*$")

def _pre_filter_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefilter lines BEFORE heading_score.

    Excludes:
    (1) rows where block_type is in _FORBIDDEN_BLOCK_TYPES
    (2) rows where text starts with any token in _FORBIDDEN_START_TEXT (case-insensitive, after lstrip)
    (3) rows where text is fully parenthesized: (...), [...], {...}
    (4) rows carrying forbidden content anywhere in the line:
        (a) signature marks / URLs / emails  — hard kill on literal match
        (b) formula residue                  — scored on symbol count + density
        (c) flattened table rows             — more than one pipe
    (5) rows where stripped text is fewer than 3 characters (e.g. decorative large first-letters)
    (6) rows that are numeric-only lines (stat callouts / chart axis rows:
        "1,633 1,584 1,258", "2023 2024 2025", "£185k")

    Returns a filtered COPY (subset of rows).
    """
    df = lines_df.copy()

    if "text" not in df.columns:
        return df

    text = df["text"].astype("string").fillna("")
    text_lstrip = text.str.lstrip()
    text_lower = text_lstrip.str.lower()

    keep = pd.Series(True, index=df.index, dtype=bool)

    # (1) forbidden block roles
    if "block_type" in df.columns:
        br = df["block_type"].astype("string").str.strip().str.lower()
        keep &= ~br.isin(_FORBIDDEN_BLOCK_TYPES)

    # (2) forbidden start tokens (text is already lstripped, so ^ is correct)
    # Single glyphs: hash lookup on the first character. Multi-char tokens
    # ("by:", "name:"): one anchored, pre-compiled regex. Both C-level.
    keep &= ~text_lower.str[0].isin(_FORBIDDEN_START_CHARS)
    keep &= ~text_lower.str.match(_FORBIDDEN_START_MULTI_RE)

    # (3) fully parenthesized lines
    keep &= ~text_lstrip.str.match(_PAREN_FULL_RE)

    # (4a) signature / online content — hard kill, one pass over the column
    # (was one full str.contains pass per substring, with re.escape recomputed
    # inside the loop on every call)
    keep &= ~text_lower.str.contains(_FORBIDDEN_CONTENT_RE, na=False)

    # (4b) formula residue — scored over the whole line rather than tripped by
    # the first symbol, so "α-Diversity Analysis" survives and "α = β·σ² + Δπ"
    # does not. Both counts are single vectorized passes.
    # Denominator is raw length including spaces: stripping whitespace first cost
    # ~25x more and only shifts the ratio slightly, well inside the threshold.
    symbol_count = text_lstrip.str.count(_SYMBOL_RE).fillna(0)
    line_len = text_lstrip.str.len().fillna(0)
    symbol_ratio = symbol_count / line_len.where(line_len > 0, 1)
    is_dense = (symbol_count >= _SYMBOL_MIN_COUNT) & (symbol_ratio > _SYMBOL_MAX_RATIO)
    is_equation = (
        text_lstrip.str.contains(_EQUATION_RE, na=False)
        & (symbol_count >= _EQUATION_MIN_SYMBOLS)
    )
    keep &= ~(is_dense | is_equation)

    # (4c) flattened table rows — several pipes, not one used as a separator
    keep &= ~(text_lstrip.str.count(r"\|").fillna(0) > _PIPE_MAX_COUNT)

    # (5) minimum 3 characters — exempt named headings (non-blank hierarchy_marker)
    # so that short markers like "I." are not stripped before the subtitle merge runs
    has_named_marker = (
        df["hierarchy_marker"].astype("string").str.strip().str.len() > 0
        if "hierarchy_marker" in df.columns
        else pd.Series(False, index=df.index)
    )
    if "char_count" in df.columns:
        keep &= (_to_float_series(df["char_count"], default=0.0) >= 3) | has_named_marker
    else:
        keep &= (text.str.strip().str.len() >= 3) | has_named_marker

    # (6) numeric-only lines (stat callouts, chart axis rows) — exempt rows with
    # a hierarchy marker so a standalone "8.1" marker line isn't stripped
    keep &= ~numeric_line_mask(text) | has_named_marker

    return df.loc[keep].copy()


# ================================================================================
# Named heading candidate detection
# ================================================================================


def _detect_marker_candidates(
    lines_df: pd.DataFrame,
    compiled_patterns,
    max_numbered_value: int = 50,
) -> pd.DataFrame:
    """
    Adds hierarchy marker detection columns:
      - hierarchy_marker : str | None   (the matched prefix text)
      - hierarchy_type   : str | None   (e.g. numbered_heading, note, ...)

    Detection logic:
    - For EACH row, test patterns in order; longest match wins
    - Only matches at START of text

    max_numbered_value:
      Rejects numbered_heading matches where any dotted component exceeds this
      value (e.g. 50 blocks "502 Lexington Av" from being a section marker).
    Roman numerals are restricted to i/v/x characters only (values 1-39);
      anything using l/c/d/m is rejected as too large to be a real heading.

    compiled_patterns: HierarchyTypePatternConfig object with .patterns attribute
    """

    out = lines_df.copy()

    out["hierarchy_marker"] = None
    out["hierarchy_type"] = None

    if "text" not in out.columns:
        return out

    texts = out["text"].astype("string").fillna("")

    # Extract patterns from HierarchyTypePatternConfig object
    pattern_list = compiled_patterns.patterns if hasattr(compiled_patterns, 'patterns') else []

    # Track the best match per row: longest match wins so that e.g. "(a)(1)" is
    # classified as double_parens rather than single_parens_single_alpha.
    # Ties are broken by YAML order (lower index = higher priority).
    best_marker: dict = {}   # idx -> marker string
    best_type: dict = {}     # idx -> hierarchy_type
    best_len: dict = {}      # idx -> len of matched marker

    for pattern in pattern_list:
        h_type = pattern.hierarchy_type
        rx = pattern.compiled

        matches = texts.str.match(rx)
        if not matches.any():
            continue

        for idx in texts[matches].index:
            m = rx.match(texts[idx])
            if not m:
                continue
            marker = m.group(0).rstrip('. \t')

            # Value-range guards
            if h_type == "numbered_heading":
                parts = [p for p in marker.replace(' ', '').split('.') if p]
                try:
                    if any(int(p) > max_numbered_value for p in parts):
                        continue
                except ValueError:
                    pass
            elif h_type == "roman_numbered_heading":
                roman_str = marker.rstrip('. \t')
                if not re.fullmatch(r'[ivxIVX]+', roman_str):
                    continue

            marker_len = len(marker)
            if marker_len > best_len.get(idx, -1):
                best_len[idx] = marker_len
                best_marker[idx] = marker
                best_type[idx] = h_type

    if best_marker:
        idx_list = list(best_marker.keys())
        out.loc[idx_list, "hierarchy_marker"] = [best_marker[i] for i in idx_list]
        out.loc[idx_list, "hierarchy_type"] = [best_type[i] for i in idx_list]

    return out

# ================================================================================
# Correct alpha vs roman detection based on series
# ================================================================================

_ROMAN_NUMERAL_VALUES = {
    "i": 1, "v": 5, "x": 10, "l": 50,
    "c": 100, "d": 500, "m": 1000,
}


def _parse_roman_numeral(s: str) -> int:
    result, prev = 0, 0
    for ch in reversed(s.lower()):
        val = _ROMAN_NUMERAL_VALUES.get(ch, 0)
        result += val if val >= prev else -val
        prev = val
    return result


_PAREN_SINGLE_TYPES = frozenset({
    "single_parens_single_alpha",
    "single_parens_double_alpha",
    "single_parens_triple_alpha",
    "single_parens_quadruple_alpha",
    "single_parens_roman",
})

_ALPHA_TYPE_BY_DEPTH = {
    1: "single_parens_single_alpha",
    2: "single_parens_double_alpha",
    3: "single_parens_triple_alpha",
    4: "single_parens_quadruple_alpha",
}

_MAX_ROMAN_SERIES_JUMP = 5  # rejects iii(3) → d(500); allows i(1) → v(5)


def _parse_paren_alpha(inner: str) -> tuple | None:
    """(depth, pos) if every char in inner is the same letter, else None. pos: a=1..z=26."""
    if not inner or not all(c == inner[0] for c in inner) or not inner[0].isalpha():
        return None
    return (len(inner), ord(inner[0]) - ord('a') + 1)


def _is_valid_alpha_step(prev: tuple, curr: tuple) -> bool:
    pd_, pp = prev
    cd, cp = curr
    # same depth, next letter
    if cd == pd_ and cp == pp + 1:
        return True
    # depth increases, previous level ended at z(26) and new level starts at 1 (aa after z)
    if cd == pd_ + 1 and cp == 1 and pp == 26:
        return True
    return False


def _correct_paren_type_by_series(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects hierarchy_type for single-paren alpha/roman markers that were
    misclassified because the regex can't distinguish (c)/(d)/(i)/(l)/(m)/(v)/(x)
    (and their repeated-letter variants) without series context.

    Strategy:
    - Build series using an active/paused state machine (same idea as numbered
      section groups). A series tracks two parallel "last seen" values: alpha
      position and roman value.
    - Mode ("alpha" | "roman") is locked by the first unambiguous member.
      Unambiguous alpha: inner letter is not a roman numeral letter (e.g. b, e, j).
      Unambiguous roman: inner contains mixed letters (iv, vi, vii...).
    - For roman continuation, max jump is _MAX_ROMAN_SERIES_JUMP so that
      iii(3)→d(500) is rejected and d correctly resumes the paused alpha series.
    - After all rows are processed, any series whose mode is still None resolves
      by majority vote of its unambiguous members.
    - Corrects hierarchy_type (and hierarchy_marker outer shape is unchanged).
    """
    out = lines_df.copy()

    target_mask = (
        out.get("hierarchy_type", pd.Series(dtype="object"))
           .fillna("").astype(str)
           .isin(_PAREN_SINGLE_TYPES)
    )
    if not target_mask.any():
        return out

    if "block_type" in out.columns:
        br = out["block_type"].astype("string").str.strip().str.lower()
        target_mask &= ~br.isin(_FORBIDDEN_BLOCK_TYPES)

    cand = out[target_mask].copy()
    if "line_id" in cand.columns:
        cand = cand.sort_values("line_id")

    # --- Parse each marker ---
    def _parse(marker: str):
        m = re.match(r'^\(([^)]+)\)$', str(marker).strip())
        if not m:
            return None, 0, ""
        inner = m.group(1).lower()
        av = _parse_paren_alpha(inner)
        rv = _parse_roman_numeral(inner) if re.fullmatch(r'[ivxlcdm]+', inner) else 0
        return av, rv, inner

    parsed    = cand["hierarchy_marker"].fillna("").apply(_parse)
    alpha_vals = parsed.apply(lambda x: x[0])
    roman_vals = parsed.apply(lambda x: x[1])
    inners     = parsed.apply(lambda x: x[2])

    # --- Classify ambiguity ---
    # "alpha"  : non-roman letter or rv == 0                          (e.g. b, j, bb, jj)
    # "roman"  : mixed roman chars (iv, vi...) or av is None          (e.g. iv, viii)
    # None     : single repeated roman letter (c, cc, i, ii, m, mm…) — ambiguous
    def _ambiguity(inner, av, rv) -> str | None:
        if av is None and rv == 0:
            return "unknown"
        if rv == 0:
            return "alpha"
        if av is None or len(set(inner)) > 1:
            return "roman"
        return None  # single repeated roman letter

    ambiguity = pd.Series(
        [_ambiguity(i, a, r) for i, a, r in zip(inners, alpha_vals, roman_vals)],
        index=cand.index,
    )

    # --- State-machine helpers ---
    series_registry: dict[int, dict] = {}
    gid_counter = 0

    def _new_series(av, rv, ambi) -> dict:
        nonlocal gid_counter
        gid_counter += 1
        s = {
            "gid":         gid_counter,
            "mode":        ambi if ambi in ("alpha", "roman") else None,
            "last_alpha":  av,
            "last_roman":  rv or 0,
            "alpha_votes": 1 if ambi == "alpha" else 0,
            "roman_votes": 1 if ambi == "roman" else 0,
        }
        series_registry[gid_counter] = s
        return s

    def _try_continue(s, av, rv) -> str | None:
        """Returns 'alpha', 'roman', or None (cannot continue)."""
        mode, la, lr = s["mode"], s["last_alpha"], s["last_roman"]
        c_alpha = (av is not None and la is not None
                   and _is_valid_alpha_step(la, av)
                   and mode != "roman")
        c_roman = (rv > 0 and 0 < rv - lr <= _MAX_ROMAN_SERIES_JUMP
                   and mode != "alpha")
        if c_alpha and not c_roman:
            return "alpha"
        if c_roman and not c_alpha:
            return "roman"
        if c_alpha and c_roman:
            # Prefer alpha (resumes existing paused series) when truly tied
            return "alpha"
        return None

    def _accept(s, result, av, rv, ambi) -> None:
        if result == "alpha":
            s["last_alpha"] = av
            if s["mode"] is None:
                s["mode"] = "alpha"
        else:
            s["last_roman"] = rv
            if s["mode"] is None:
                s["mode"] = "roman"
        if ambi == "alpha":
            s["alpha_votes"] += 1
        elif ambi == "roman":
            s["roman_votes"] += 1

    # --- Main pass ---
    active: dict | None = None
    paused: list[dict] = []
    row_gid: dict = {}  # cand.index value → gid

    for idx in cand.index:
        av   = alpha_vals[idx]
        rv   = roman_vals[idx]
        ambi = ambiguity[idx]

        if ambi == "unknown":
            continue

        # Try active
        if active is not None:
            result = _try_continue(active, av, rv)
            if result is not None:
                _accept(active, result, av, rv, ambi)
                row_gid[idx] = active["gid"]
                continue
            paused.append(active)
            active = None

        # Try paused (newest first)
        for i in range(len(paused) - 1, -1, -1):
            result = _try_continue(paused[i], av, rv)
            if result is not None:
                active = paused.pop(i)
                _accept(active, result, av, rv, ambi)
                break

        if active is None:
            active = _new_series(av, rv, ambi)

        row_gid[idx] = active["gid"]

    # --- Resolve mode for series where all members were ambiguous ---
    def _resolve_mode(s) -> str | None:
        if s["mode"] is not None:
            return s["mode"]
        av, rv = s["alpha_votes"], s["roman_votes"]
        if av > rv:
            return "alpha"
        if rv > av:
            return "roman"
        return None

    series_mode = {gid: _resolve_mode(s) for gid, s in series_registry.items()}

    # --- Apply corrections ---
    for idx in cand.index:
        gid = row_gid.get(idx)
        if gid is None:
            continue
        mode = series_mode.get(gid)
        if mode is None:
            continue

        av           = alpha_vals[idx]
        rv           = roman_vals[idx]
        current_type = str(out.at[idx, "hierarchy_type"])

        if mode == "alpha" and av is not None:
            correct_type = _ALPHA_TYPE_BY_DEPTH.get(av[0])
            if correct_type and current_type != correct_type:
                out.at[idx, "hierarchy_type"] = correct_type
        elif mode == "roman" and rv > 0 and current_type != "single_parens_roman":
            out.at[idx, "hierarchy_type"] = "single_parens_roman"

    return out


def _validate_alpha_heading_series(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates alpha_heading markers by requiring them to form a consecutive
    alphabetical series (A→B→C…). A row is valid if at least one adjacent
    alpha_heading neighbour (by line_id order) has the immediately preceding
    or following letter. Singletons have hierarchy_type and hierarchy_marker
    cleared so they fall through as free_form if scored as a heading.
    """
    out = lines_df.copy()

    target_mask = (
        out.get("hierarchy_type", pd.Series(dtype="object"))
           .fillna("").astype(str)
           .eq("alpha_heading")
    )
    if not target_mask.any():
        return out

    cand = out[target_mask].copy()
    if "line_id" in cand.columns:
        cand = cand.sort_values("line_id")

    def _parse_letter(marker) -> str | None:
        s = str(marker).strip().rstrip(". \t").strip()
        return s.upper() if len(s) == 1 and s.isalpha() else None

    letters = cand["hierarchy_marker"].fillna("").apply(_parse_letter)
    valid_letters = letters.dropna()

    if valid_letters.empty:
        return out

    letter_vals = valid_letters.apply(ord)

    # A row is valid if the immediately adjacent alpha_heading (prev or next)
    # has a letter exactly one step away.
    has_valid_prev = letter_vals.diff() == 1
    has_valid_next = letter_vals.diff(-1) == -1
    is_valid = has_valid_prev | has_valid_next

    invalid_idx = valid_letters.index[~is_valid]
    out.loc[invalid_idx, "hierarchy_type"] = None
    out.loc[invalid_idx, "hierarchy_marker"] = None

    return out


# ================================================================================
# Core heading scoring function
# ================================================================================

# ----- Config ----- #

# Colors considered "basic" (black/white/greys) — not treated as special for color rarity scoring
_BASIC_COLORS = frozenset({
    "#000000", "#ffffff", "#fff", "#000",
    "#111111", "#222222", "#333333", "#444444", "#555555", "#666666",
    "#777777", "#888888", "#999999", "#aaaaaa", "#bbbbbb", "#cccccc",
    "#dddddd", "#eeeeee",
    "#0000ff",  # standard hyperlink blue — not a heading signal
})

# Pattern used in coverpage suppression to identify FORM/SCHEDULE headings
_FORM_PATTERN = r"^\s*(?:FORM|SCHEDULE)\b"



def _to_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    """
    Robust boolean parsing for mixed types:
    - True/False
    - "TRUE"/"FALSE"
    - 1/0
    - NaN -> default
    """
    if s is None:
        return pd.Series(default, index=pd.Index([]), dtype=bool)

    if s.dtype == bool:
        return s.fillna(default)

    # strings / objects / numbers
    out = s.copy()

    # numeric-like -> bool
    if pd.api.types.is_numeric_dtype(out):
        return out.fillna(int(default)).astype(int).astype(bool)

    # object -> normalize string
    out = out.astype("string")
    norm = out.str.strip().str.lower()

    true_set = {"true", "t", "1", "yes", "y"}
    false_set = {"false", "f", "0", "no", "n"}

    parsed = pd.Series(np.nan, index=out.index, dtype="float64")
    parsed = parsed.mask(norm.isin(true_set), 1.0)
    parsed = parsed.mask(norm.isin(false_set), 0.0)

    # fallback: keep default for unknowns / missing
    parsed = parsed.fillna(1.0 if default else 0.0)
    return parsed.astype(int).astype(bool)


def _to_float_series(s: pd.Series, default: float = np.nan) -> pd.Series:
    if s is None:
        return pd.Series(default, index=pd.Index([]), dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    num = _to_float_series(num, default=np.nan)
    den = _to_float_series(den, default=np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ----- Core heading scoring function ----- #

def _add_gap_context(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``line_gap_below``: the gap under each line = the *next* line's
    ``line_gap`` (which is the gap above it), within the same page.

    Must run on the full line frame, before _pre_filter_lines: the scorer only
    sees a filtered subset, where "the next line" is not the next line on the page.
    """
    out = lines_df.copy()
    if "line_gap" not in out.columns:
        out["line_gap_below"] = np.nan
        return out

    gap = pd.to_numeric(out["line_gap"], errors="coerce")
    if "page_number" in out.columns:
        out["line_gap_below"] = gap.groupby(out["page_number"], sort=False).shift(-1)
    else:
        out["line_gap_below"] = gap.shift(-1)
    return out


def _add_heading_score(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes lines_df and returns it with two extra columns: `heading_score` and
    `heading_score_debug` (per-row JSON of every non-zero contribution).

    Scoring rules. Bands within a KPI are cumulative unless noted — a 300-char
    line takes the >100, >150 and >250 penalties together (-5.5 total).

    - font_size_ratio (fsr), missing -> 1.0:
      - [1.01, 1.2): +1
      - [1.2, 1.4):  +2
      - >=1.4:       +3
      - <0.95:       -1
      - <0.75:       -5   (cumulative with the <0.95 band -> -6)
    - char_count:
      - <50:   +0.5
      - >100:  -1
      - >150:  -1.5
      - >250:  -3
      - >2000: -2
    - capitalized_word_ratio = capitalized_word_count/word_count
      - >0.75: +0.5
    - is_bold = true:       +2.5
    - is_italic = true:     +1.5
    - is_underlined = true: +1.5
    - is_uppercase = true:  +1
    - text_align = center:  +1
    - non_stroking_color present, not the document's most prevalent color, and
      not in _BASIC_COLORS: +1
    - hierarchy_type detected (any non-blank marker type): +0.5
    - hierarchy_marker with 3+ levels (e.g. '2.1.5'): -0.5
    - has_link = true: -5
    - line_gap (top = gap above, bottom = gap below):
      - top in [-2, 0.5]pt: -5
      - top in (0.5, 2)pt: -2
      - both positive and top < bottom: -0.5
      - both positive, top > bottom and top in [6, 20]pt: +0.5
    - layout_id has 4 or more lines: -2
    """
    out = lines_df.copy()

    score = pd.Series(0.0, index=out.index, dtype="float64")

    # --- font_size_ratio
    fsr = _to_float_series(out.get("font_size_ratio"), default=np.nan).fillna(1.0)
    c_fsr = pd.Series(0.0, index=out.index, dtype="float64")
    c_fsr += np.where((fsr >= 1.01) & (fsr < 1.2), 1.0, 0.0)
    c_fsr += np.where((fsr >= 1.2) & (fsr < 1.4), 2.0, 0.0)
    c_fsr += np.where((fsr >= 1.4), 3.0, 0.0)
    c_fsr += np.where((fsr < 0.95), -1.0, 0.0)
    c_fsr += np.where((fsr < 0.75), -5.0, 0.0)
    score += c_fsr

    # --- char_count
    cc = _to_float_series(out.get("char_count"), default=np.nan)
    c_cc = pd.Series(0.0, index=out.index, dtype="float64")
    c_cc += np.where(cc < 50, 0.5, 0.0)
    c_cc += np.where(cc > 100, -1.0, 0.0)
    c_cc += np.where(cc > 150, -1.5, 0.0)
    c_cc += np.where(cc > 2000, -2.0, 0.0)
    c_cc += np.where(cc > 250, -3.0, 0.0)
    score += c_cc

    # --- capitalized_token_ratio
    cap_ratio = _safe_div(out.get("capitalized_word_count"), out.get("word_count"), fill=0.0)
    if len(cap_ratio) == 0:
        cap_ratio = pd.Series(0.0, index=out.index, dtype="float64")
    c_cap = pd.Series(np.where(cap_ratio > 0.75, 0.5, 0.0), index=out.index, dtype="float64")
    score += c_cap

    # --- styles
    # Note: we must handle missing columns - _to_bool_series returns empty Series if column is None
    is_bold = _to_bool_series(out.get("is_bold"), default=False)
    is_italic = _to_bool_series(out.get("is_italic"), default=False)
    is_underlined = _to_bool_series(out.get("is_underlined"), default=False)
    is_uppercase = _to_bool_series(out.get("is_uppercase"), default=False)

    # _to_bool_series returns an empty Series for missing columns; realign to full index before scoring
    if len(is_bold) == 0:
        is_bold = pd.Series(False, index=out.index, dtype=bool)
    if len(is_italic) == 0:
        is_italic = pd.Series(False, index=out.index, dtype=bool)
    if len(is_underlined) == 0:
        is_underlined = pd.Series(False, index=out.index, dtype=bool)
    if len(is_uppercase) == 0:
        is_uppercase = pd.Series(False, index=out.index, dtype=bool)

    c_bold = is_bold.astype("float64") * 2.5
    c_italic = is_italic.astype("float64") * 1.5
    c_underlined = is_underlined.astype("float64") * 1.5
    c_uppercase = is_uppercase.astype("float64") * 1.0
    score += c_bold + c_italic + c_underlined + c_uppercase

    # --- text_align = center
    c_center = pd.Series(0.0, index=out.index, dtype="float64")
    ta = out.get("text_align")
    if ta is not None:
        ta_norm = ta.astype("string").str.strip().str.lower()
        c_center = pd.Series(np.where(ta_norm.eq("center"), 1.0, 0.0), index=out.index, dtype="float64")
    score += c_center

    # --- non_stroking_color rarity bonus (+1 if not the prevalent color, excluding basic colors)
    c_color = pd.Series(0.0, index=out.index, dtype="float64")
    nsc = out.get("non_stroking_color")
    if nsc is not None:
        nsc_norm = (
            nsc.astype("string")
            .str.strip()
            .str.lower()
            .replace({"": pd.NA, "none": pd.NA, "nan": pd.NA})
        )

        # Most prevalent color (mode) across the rows being scored (non-table slice)
        mode_color = nsc_norm.dropna().mode()
        prevalent = mode_color.iloc[0] if not mode_color.empty else pd.NA

        is_basic = nsc_norm.isin(_BASIC_COLORS)

        # +1 if:
        # - color is present
        # - color != prevalent color
        # - not a basic color
        bonus = (nsc_norm.notna()) & (nsc_norm != prevalent) & (~is_basic)
        c_color = bonus.astype("float64") * 1.0
    score += c_color

    # --- hierarchy_type marker bonus (+0.5 if a structural marker was detected)
    c_hierarchy = pd.Series(0.0, index=out.index, dtype="float64")
    if "hierarchy_type" in out.columns:
        ht = out["hierarchy_type"].fillna("").astype(str).str.strip()
        c_hierarchy = (ht != "").astype("float64") * 0.5
    score += c_hierarchy

    # --- deep numbered marker penalty (-0.5 if 3+ levels, e.g. '2.1.5')
    c_marker_depth = pd.Series(0.0, index=out.index, dtype="float64")
    if "hierarchy_marker" in out.columns:
        marker = out["hierarchy_marker"].fillna("").astype(str).str.strip().str.rstrip(".")
        # split on literal '.' rather than a regex count (RE2/ASCII pitfalls)
        depth = marker.str.split(".").apply(lambda parts: len([p for p in parts if p]))
        c_marker_depth = pd.Series(np.where(depth >= 3, -0.5, 0.0), index=out.index, dtype="float64")
    score += c_marker_depth

    # --- hyperlink penalty (-5)
    c_link = pd.Series(0.0, index=out.index, dtype="float64")
    if "has_link" in out.columns:
        has_link = _to_bool_series(out["has_link"], default=False)
        c_link = pd.Series(np.where(has_link, -5.0, 0.0), index=out.index, dtype="float64")
    score += c_link

    # --- vertical spacing (gap above vs gap below the line)
    # A heading sits under whitespace and hugs the text it introduces, so the gap
    # above should exceed the gap below.
    #
    # The squeeze penalties only apply when there is a real predecessor line above
    # on the same page. The first line of a page has NO line above it, and upstream
    # encodes that as line_gap == 0.0 (a page reset, NOT a large negative). A page
    # top is one of the strongest heading positions there is, so it must be excluded
    # from the "squeezed against the previous line" band: penalise a genuine squeeze
    # (a small *positive* gap) only, i.e. top strictly > 0.
    c_gap = pd.Series(0.0, index=out.index, dtype="float64")
    if "line_gap" in out.columns:
        top = _to_float_series(out.get("line_gap"), default=np.nan)
        bottom = _to_float_series(out.get("line_gap_below"), default=np.nan)

        both_positive = (top > 0) & (bottom > 0)
        # squeezed against the previous line -> almost certainly mid-paragraph.
        # top <= 0 (page top / column reset) is not a squeeze and is left alone.
        c_gap += np.where(top.gt(0.0) & top.le(0.5), -5.0, 0.0)
        c_gap += np.where(top.gt(0.5) & top.lt(2.0), -2.0, 0.0)
        # more air below than above -> reads as a caption, not a heading
        c_gap += np.where(both_positive & (top < bottom), -0.5, 0.0)
        # clear air above, tight below, at a plausible heading spacing
        c_gap += np.where(both_positive & (top > bottom) & top.between(6.0, 20.0), 0.5, 0.0)
    score += c_gap

    # --- layout_id density penalty (-2 if layout_id has 4+ lines)
    c_layout_density = pd.Series(0.0, index=out.index, dtype="float64")
    if "layout_id" in out.columns:
        layout_sizes = out.groupby("layout_id", sort=False)["layout_id"].transform("count")
        c_layout_density = pd.Series(np.where(layout_sizes >= 4, -2.0, 0.0), index=out.index, dtype="float64")
    score += c_layout_density

    out["heading_score"] = score

    # --- build per-row debug JSON with each KPI's contribution
    debug_components = pd.DataFrame({
        "font_size_ratio": c_fsr,
        "char_count": c_cc,
        "cap_ratio": c_cap,
        "is_bold": c_bold,
        "is_italic": c_italic,
        "is_underlined": c_underlined,
        "is_uppercase": c_uppercase,
        "text_align_center": c_center,
        "color_rarity": c_color,
        "hierarchy_type": c_hierarchy,
        "marker_depth": c_marker_depth,
        "has_link": c_link,
        "line_gap": c_gap,
        "layout_density": c_layout_density,
    }, index=out.index)

    # Build each row's debug JSON in a tight loop over the raw numpy block.
    # DataFrame.apply(axis=1) rebuilds a pd.Series per row and is ~10x slower for
    # identical output (same round(float(v), 4), same non-zero filter, same order).
    comp_cols = list(debug_components.columns)
    comp_vals = debug_components.to_numpy(dtype="float64")
    debug_json = np.empty(comp_vals.shape[0], dtype=object)
    for _r in range(comp_vals.shape[0]):
        row = comp_vals[_r]
        debug_json[_r] = json.dumps(
            {comp_cols[_c]: round(float(row[_c]), 4) for _c in range(row.shape[0]) if row[_c] != 0.0}
        )
    out["heading_score_debug"] = debug_json

    return out

# ================================================================================
# Contextual score adjustments
# ================================================================================


def _apply_contextual_score_adjustments(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Window-based score adjustments applied to heading_score.

    Operates on lines sorted by line_id. For each line, three overlapping 3-line
    windows are considered (line is first, middle, and last in the window).

    heading_score: -2  the line is part of any window where all 3 consecutive lines share the
                       same styling key (applied at most once per line)
    heading_score: +1  no line within 2 positions shares the line's styling key
                       (ensures the line is alone in every 3-line window it participates in)
    """
    out = lines_df.copy()

    if "heading_score" not in out.columns:
        return out

    # style key: bold|italic|underlined|font_family|hierarchy_type|font_size_ratio
    # (font_size_ratio bucketed to 1dp). Columns that are absent are skipped.
    # non_stroking_color is deliberately excluded — color_rarity already scores it;
    # is_uppercase caused too many accidental style-run hits in practice.
    style_parts = []
    for col in ["is_bold", "is_italic", "is_underlined", "font_family", "hierarchy_type"]:
        if col in out.columns:
            style_parts.append(out[col].fillna(False).astype(str))
    if "font_size_ratio" in out.columns:
        style_parts.append(out["font_size_ratio"].fillna(1.0).round(1).astype(str))

    if not style_parts:
        return out

    sort_col = "line_id" if "line_id" in out.columns else None
    working_idx = out.sort_values(sort_col).index if sort_col else out.index

    # Vectorized string join — avoids a row-by-row Python loop from .agg()
    style_key = style_parts[0].copy()
    for _part in style_parts[1:]:
        style_key = style_key + "|" + _part
    style_key = style_key.loc[working_idx]

    same_prev2 = (style_key == style_key.shift(2)).fillna(False)
    same_prev  = (style_key == style_key.shift(1)).fillna(False)
    same_next  = (style_key == style_key.shift(-1)).fillna(False)
    same_next2 = (style_key == style_key.shift(-2)).fillna(False)

    # -2: part of any 3-consecutive same-style window
    in_penalty_window = (
        (same_prev & same_prev.shift(1).fillna(False)) |   # window [i-2, i-1, i]
        (same_prev & same_next) |                           # window [i-1, i, i+1]
        (same_next & same_next.shift(-1).fillna(False))    # window [i, i+1, i+2]
    )

    # +1: no line within 2 positions shares the same style key
    # (ensures the line is alone in every 3-line window it participates in)
    isolated = ~same_prev2 & ~same_prev & ~same_next & ~same_next2

    penalty_arr = np.asarray(in_penalty_window).ravel().astype(bool)
    isolated_arr = np.asarray(isolated).ravel().astype(bool)
    adj = pd.Series(
        np.where(penalty_arr, -2.0, 0.0) + np.where(isolated_arr, 1.0, 0.0),
        index=working_idx,
        dtype="float64",
    )

    adj_aligned = adj.reindex(out.index).fillna(0.0)
    out["heading_score"] = (out["heading_score"] + adj_aligned).astype("float64")

    # Append contextual adjustment to debug JSON — only patch rows where adj != 0.
    # Use positional numpy access + a single column write-back; per-row label-based
    # .at[] get/set is ~1000x slower here (repeated index lookups dominate).
    if "heading_score_debug" in out.columns:
        adj_vals = adj_aligned.to_numpy()
        nonzero_pos = np.nonzero(adj_vals != 0.0)[0]
        if len(nonzero_pos):
            debug_vals = out["heading_score_debug"].to_numpy(dtype=object, copy=True)
            for _p in nonzero_pos:
                try:
                    d = json.loads(debug_vals[_p])
                except (ValueError, TypeError):
                    d = {}
                d["contextual_adjustment"] = round(float(adj_vals[_p]), 4)
                debug_vals[_p] = json.dumps(d)
            out["heading_score_debug"] = debug_vals

    return out


# ================================================================================
# Add heading decision
# ================================================================================

_HEADING_SCORE_THRESHOLD = 2.5
_DEFAULT_HEADING_TYPE = "free_form"


def _add_heading_decision(
    lines_df: pd.DataFrame,
    heading_score_threshold: float = _HEADING_SCORE_THRESHOLD,
    default_heading_type: str = _DEFAULT_HEADING_TYPE,
) -> pd.DataFrame:
    """
    Final heading decision. A row passes when heading_score >= threshold
    (inclusive — a score exactly at the threshold is a heading).

    - If a row passes AND block_type is not already set:
        - set block_type = "heading"
        - set heading_source = "score"
    - Otherwise:
        - preserve existing block_type (e.g., toc_heading, page_label, etc.)

    heading_type = hierarchy_type if non-blank, else default_heading_type. It is
    set for every row that passes the threshold AND for every row already carrying
    block_type == "heading" (e.g. from docx), whose score is irrelevant because the
    role was pre-assigned upstream.

    This ensures that special roles like toc_heading are preserved while still
    getting their heading_type populated based on their hierarchy detection.

    Adds/updates:
      - block_type (string)     - only for passing rows with no role yet
      - heading_source (string) - "score", same rows as block_type above
      - heading_type (string)   - passing rows + pre-assigned heading rows
    """
    out = lines_df.copy()

    if "heading_type" not in out.columns:
        out["heading_type"] = pd.NA

    # Ensure block_type exists
    if "block_type" not in out.columns:
        out["block_type"] = pd.NA

    # Need heading_score to decide; if missing, no-op
    if "heading_score" not in out.columns:
        return out

    hs = pd.to_numeric(out["heading_score"], errors="coerce").fillna(-1e9)
    is_heading = hs >= float(heading_score_threshold)

    # Set block_type ONLY for winners that don't already have a role
    # Preserve existing roles like toc_heading, page_label, etc.
    existing_role = out["block_type"].astype("string").str.strip()
    no_role_yet = existing_role.isna() | (existing_role == "") | (existing_role == "nan")
    
    # Only assign "heading" to rows that pass threshold AND don't have a role yet
    should_assign_heading = is_heading & no_role_yet
    out.loc[should_assign_heading, "block_type"] = "heading"

    if "heading_source" not in out.columns:
        out["heading_source"] = pd.NA
    out.loc[should_assign_heading, "heading_source"] = "score"

    # Pick heading_type from hierarchy_type if present, else default
    if "hierarchy_type" in out.columns:
        ht = out["hierarchy_type"].astype("string").str.strip()
        chosen = ht.where(ht.notna() & (ht != ""), default_heading_type)
    else:
        chosen = pd.Series(default_heading_type, index=out.index, dtype="string")

    # Also populate heading_type for rows already marked as heading (e.g. from docx),
    # regardless of score — scoring is irrelevant when the role is pre-assigned.
    already_heading = existing_role.str.lower() == "heading"
    should_set_heading_type = is_heading | already_heading

    out.loc[should_set_heading_type, "heading_type"] = chosen.loc[should_set_heading_type]

    return out


# ================================================================================
# Numbered section group analysis
# ================================================================================

_NUMBERED_HIERARCHY_TYPES = {"numbered_heading", "roman_numbered_heading"}


def _dense_rank_keep_na(values: pd.Series) -> pd.Series:
    """Dense-rank the non-null values of an Int64 column, leaving nulls null.

    Not ``values.rank(method="dense", na_option="keep")``: on nullable (masked)
    dtypes pandas 2 ignores ``na_option`` and ranks NA as if it were a value, so
    every null row comes back as rank 1 — a real id, indistinguishable from a
    detected one. Downstream that reads as "every line belongs to the same
    heading", which collapses whole sections into a single block. pandas 3 keeps
    the nulls, so the bug only shows up on installs that resolve pandas 2 (any
    Python 3.10 environment, since pandas 3 requires 3.11+). Ranking the
    non-null subset gives the same answer on every supported pandas.
    """
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    known = values.notna()
    if known.any():
        out[known] = values[known].astype("int64").rank(method="dense").astype("int64")
    return out


def _parse_numbered_value(marker: str) -> tuple | None:
    """
    Parses a numbered section marker into a comparable int tuple.
      '8.1.2' -> (8, 1, 2)
      '1.'    -> (1,)
      'III.'  -> (3,)
    Returns None if unparseable.
    """
    if not marker:
        return None
    s = marker.strip().rstrip('. \t')
    parts = [p for p in s.split('.') if p]
    if not parts:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        pass
    if len(parts) == 1 and re.fullmatch(r'[ivxlcdmIVXLCDM]+', parts[0]):
        v = _parse_roman_numeral(parts[0])
        return (v,) if v > 0 else None
    return None


def _is_valid_numbered_continuation(prev: tuple, curr: tuple, max_jump: int = 2) -> bool:
    """
    Returns True if curr is a logical next step after prev in a numbered hierarchy.

    Strategy: trim prev to the depth of curr (going shallower is always ok),
    find the first divergence index d, then:
      - at d: curr[d] must advance by 1..max_jump over prev[d]
        (max_jump=2 tolerates exactly one undetected sibling, nothing more)
      - after d: all curr[d+1:] must be 1 (restarting each child level)

    Special case: no divergence (curr is a direct child of prev) — valid only
    if curr goes exactly one level deeper and starts at 1.

    Examples:
      (1,3) → (2,1)    valid  — parent advances, child restarts at 1
      (8,)  → (8,1)    valid  — first child
      (8,2) → (9,)     valid  — go up and advance
      (6,9) → (6,11)   valid  — one sibling (6.10) missed
      (1,3) → (2,2)    invalid — child didn't restart at 1
      (11,) → (1,)     invalid — negative advance (restart → new group)
      (10,) → (24,)    invalid — jump too large
    """
    if not prev or not curr:
        return False

    depth = min(len(prev), len(curr))
    p = prev[:depth]
    c = curr[:depth]

    # Find first divergence
    div = next((i for i in range(depth) if p[i] != c[i]), depth)

    if div == depth:
        # No divergence up to shared depth: curr is strictly deeper than prev
        if len(curr) == len(prev) + 1 and curr[-1] == 1:
            return True
        return False

    diff = c[div] - p[div]
    if not (0 < diff <= max_jump):
        return False

    # Everything after the divergence in curr must be 1 (first child at each level)
    return all(curr[i] == 1 for i in range(div + 1, len(curr)))


def _assign_numbered_heading_groups(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyses rows with hierarchy_type in {numbered_heading, roman_numbered_heading}.

    1. Parses each hierarchy_marker into an int tuple (e.g. '8.1' → (8, 1)).
    2. Groups sequential rows into logical series ordered by line_id, with
       numbered_heading and roman_numbered_heading processed as strictly
       separate pools (a roman marker never continues or interrupts an
       arabic series, and vice versa).
       A new group starts whenever the transition is not a valid continuation,
       and a broken series is closed for good — it can never be resumed later.
       Continuing at a depth the series has already visited additionally
       requires the same font size (|Δ| ≤ 0.25) as the last row seen at that
       depth — same-level items of one scheme share a font, so a size change
       means a different scheme. Depth changes are exempt (child levels may
       legitimately use different sizes); rows without font data skip the check.
       Singleton rows (no valid neighbour on either side) are left ungrouped.
    3. Within each (group, depth-level), if at least one row already has
       block_type = 'heading', every other row at the same depth whose
       block_type is not in {heading, toc_heading, page_label} is promoted to
       'hybrid_heading_paragraph' (and given a heading_type). Skipped entirely
       unless both 'block_type' and 'heading_score' columns are present.

    Adds:
      numbered_heading_group  (Int64, nullable — null means stray/ungrouped)
    Updates:
      block_type  — may set 'hybrid_heading_paragraph'
    """
    out = lines_df.copy()

    target_mask = (
        out.get("hierarchy_type", pd.Series(dtype="object"))
           .fillna("").astype(str)
           .isin(_NUMBERED_HIERARCHY_TYPES)
    )
    if not target_mask.any():
        out["numbered_heading_group"] = pd.NA
        return out

    if "block_type" in out.columns:
        br = out["block_type"].astype("string").str.strip().str.lower()
        target_mask &= ~br.isin(_FORBIDDEN_BLOCK_TYPES)

    cand = out[target_mask].copy()
    if "line_id" in cand.columns:
        cand = cand.sort_values("line_id")

    # Parse markers
    values = cand["hierarchy_marker"].fillna("").astype(str).apply(_parse_numbered_value)

    # Font size per row (NaN → no check). Prefer raw font_size (PDF),
    # fall back to font_size_ratio (HTML).
    font_col = next(
        (c for c in ("font_size", "font_size_ratio") if c in cand.columns), None
    )
    if font_col is not None:
        fonts = pd.to_numeric(cand[font_col], errors="coerce")
    else:
        fonts = pd.Series(np.nan, index=cand.index)

    # Strict outline grouping, one pool per hierarchy_type so arabic and
    # roman series can never mix.
    #
    # Only one series is ever alive per pool. Each parsed value either
    # extends it (sibling +1, one skipped sibling, first child, or ancestor
    # sibling — see _is_valid_numbered_continuation, plus a matching font
    # size at already-visited depths) or breaks it; a broken series is
    # closed permanently and a new group starts. There is no pause/resume:
    # a series can never bridge across another series.
    # Unparseable rows (value=None) are skipped entirely — they don't interrupt state.
    # After assignment, groups of size 1 (no valid neighbour) are marked stray (null).
    _FONT_TOL = 0.25
    group_counter = 0
    row_group_ids: dict = {}   # cand.index → raw gid

    htypes = cand["hierarchy_type"].fillna("").astype(str)
    for _htype in htypes.unique():
        active = None   # {"gid": int, "last": tuple, "fonts": {depth: size}}
        for idx, value in values[htypes == _htype].items():
            if value is None:
                continue  # noise that didn't parse — leave state untouched

            fs = fonts.loc[idx]
            depth = len(value)

            extends = active is not None and _is_valid_numbered_continuation(
                active["last"], value
            )
            if extends and not pd.isna(fs):
                prev_fs = active["fonts"].get(depth)
                if prev_fs is not None and abs(fs - prev_fs) > _FONT_TOL:
                    extends = False

            if extends:
                active["last"] = value
            else:
                group_counter += 1
                active = {"gid": group_counter, "last": value, "fonts": {}}

            if not pd.isna(fs):
                active["fonts"][depth] = float(fs)
            row_group_ids[idx] = active["gid"]

    # Build Series aligned to cand.index (rows with value=None get NA)
    raw_group = pd.Series(row_group_ids, dtype="Int64").reindex(cand.index)

    # Singletons → stray (null group); dense-rank survivors so groups start at 1
    vc = raw_group.value_counts()
    grp_size = raw_group.map(vc)
    final_group = _dense_rank_keep_na(raw_group.where(grp_size > 1))

    out["numbered_heading_group"] = pd.NA
    out.loc[cand.index, "numbered_heading_group"] = final_group

    # ---- Promote to hybrid_heading_paragraph ----
    depth_series = values.apply(lambda t: len(t) if t is not None else pd.NA)
    working = cand.assign(_grp=final_group, _depth=depth_series)
    working = working[working["_grp"].notna()]

    if (
        not working.empty
        and "block_type" in working.columns
        and "heading_score" in working.columns
    ):
        working = working.copy()
        working["_is_heading"] = working["block_type"].fillna("").astype(str) == "heading"

        grp_depth_has_heading = (
            working.groupby(["_grp", "_depth"])["_is_heading"]
            .transform("any")
        )

        _PRESERVE_ROLES = {"heading", "toc_heading", "page_label"}
        promote_mask = grp_depth_has_heading & (
            ~working["block_type"].fillna("").astype(str).isin(_PRESERVE_ROLES)
        )
        promoted_idx = working[promote_mask].index
        out.loc[promoted_idx, "block_type"] = "hybrid_heading_paragraph"

        if "heading_type" not in out.columns:
            out["heading_type"] = pd.NA
        if "hierarchy_type" in out.columns:
            ht = out.loc[promoted_idx, "hierarchy_type"].astype("string").str.strip()
            out.loc[promoted_idx, "heading_type"] = ht.where(ht.notna() & (ht != ""), _DEFAULT_HEADING_TYPE)
        else:
            out.loc[promoted_idx, "heading_type"] = _DEFAULT_HEADING_TYPE

    return out


def _extract_hybrid_heading_texts(
    lines_df: pd.DataFrame,
    max_search: int = 100,
) -> pd.DataFrame:
    """
    For rows with block_type == 'hybrid_heading_paragraph', extracts the heading
    portion from text by finding the first sentence-ending delimiter after the
    hierarchy_marker, searching within max_search characters of body text.

    Delimiters tried in order: . → : → ; → ,

    Adds:
      hybrid_heading_text  (string, nullable)
    """
    out = lines_df.copy()
    out["hybrid_heading_text"] = pd.NA

    mask = (
        out.get("block_type", pd.Series(dtype="object"))
           .fillna("").astype(str)
           .eq("hybrid_heading_paragraph")
    )
    if not mask.any():
        return out

    def _extract(row) -> str | None:
        text   = str(row.get("text",             "") or "")
        marker = str(row.get("hierarchy_marker", "") or "")
        if not text:
            return None

        # Start after the marker, then skip any trailing structural characters
        # (dots, digits, spaces) so that "3.1. Directors..." doesn't match the
        # dot after "1" — we want the first delimiter that follows an alpha char.
        start = len(marker)
        alpha_start = start
        for i in range(start, min(start + 20, len(text))):
            if text[i].isalpha():
                alpha_start = i
                break

        for delim in (".", ":", ";", ","):
            pos = text.find(delim, alpha_start, alpha_start + max_search)
            if pos != -1:
                return text[: pos + 1].strip()
        return None

    out.loc[mask, "hybrid_heading_text"] = (
        out[mask].apply(_extract, axis=1)
    )
    return out


# ================================================================================
# Add heading fingerprints
# ================================================================================

_FINGERPRINT_COLS = [
    "section",
    "heading_type",
    "font_size_ratio",
    "is_bold",
    "is_italic",
    "is_underlined",
    "is_uppercase",
    "text_align",
    "font_family",
    "non_stroking_color",
]

_ID_MAX_PARAMS_DIFF_DEFAULT = 0
_ID_MAX_PARAMS_DIFF_SPECIAL = 2
_SPECIAL_HEADING_TYPES = {
    "item", "part", "note", "annex", "article", "section", "proposal",
    "section_abbreviated", "schedule", "title", "subpart", "chapter",
    "amendment", "rule", "figure", "table", "appendix", "exhibit",
    "numbered_heading", "roman_numbered_heading"
}

# All single_parens_*_alpha variants are one continuous series (a→z→aa→zz…)
# and are normalised to a single fp type before fingerprinting.
_PAREN_ALPHA_TYPES = frozenset({
    "single_parens_single_alpha",
    "single_parens_double_alpha",
    "single_parens_triple_alpha",
    "single_parens_quadruple_alpha",
})

def _normalize_scalar(v):
    """Make values JSON-stable + comparable (incl. pandas / numpy scalars)."""
    if v is None or pd.isna(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        low = s.lower()
        if low in {"true", "t", "1", "yes", "y"}:
            return True
        if low in {"false", "f", "0", "no", "n"}:
            return False
        return s
    if isinstance(v, int):
        return int(v)
    return v


def _canonical_json_hash(obj: dict) -> str:
    """Strict hash: canonical JSON (sorted keys, compact separators), then SHA-256 hex."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add_heading_fingerprints_and_ids(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds heading_fingerprint, heading_hash, and heading_fp_id to heading rows.
    heading_fp_id is stable within a doc: same style → same ID, with tolerance
    for special heading types (_ID_MAX_PARAMS_DIFF_SPECIAL differing params).
    """
    out = lines_df.copy()

    if "heading_fingerprint" not in out.columns:
        out["heading_fingerprint"] = pd.NA
    if "heading_hash" not in out.columns:
        out["heading_hash"] = pd.NA
    if "heading_fp_id" not in out.columns:
        out["heading_fp_id"] = pd.NA

    if "block_type" not in out.columns:
        return out

    br = out["block_type"].astype("string")
    is_heading = br.str.strip().str.lower().isin({"heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph"}).fillna(False).astype(bool)

    if not is_heading.any():
        return out

    # Pre-extract only heading rows, sorted — avoids iterating all rows and
    # repeated out.loc[i] Series construction inside the loop.
    sort_cols = [c for c in ("page_number", "line_id") if c in out.columns]
    fp_cols_present = [c for c in _FINGERPRINT_COLS if c in out.columns]
    extra_cols = [c for c in sort_cols if c not in fp_cols_present]

    heading_sub = out.loc[is_heading, fp_cols_present + extra_cols].copy()
    if sort_cols:
        heading_sub = heading_sub.sort_values(sort_cols, kind="mergesort")
    heading_idx = heading_sub.index

    # Bulk convert to plain dicts once — O(1) field access in the loop below
    records = heading_sub[fp_cols_present].to_dict("index")

    next_id = 1
    seen = []

    for i in heading_idx:
        row_dict = records[i]
        fp = {col: (_normalize_scalar(row_dict[col]) if col in row_dict else None) for col in _FINGERPRINT_COLS}

        ht = fp.get("heading_type")
        if ht is None or (isinstance(ht, str) and ht.strip() == ""):
            fp["heading_type"] = "free_form"
        ht_norm_l = fp["heading_type"].strip().lower() if isinstance(fp["heading_type"], str) else ""

        # All single_parens_*_alpha types are the same series (a→z→aa→zz…), treat as one fp group
        if ht_norm_l in _PAREN_ALPHA_TYPES:
            ht_norm_l = "single_parens_alpha"
            fp["heading_type"] = "single_parens_alpha"

        fsr = fp.get("font_size_ratio")
        try:
            fp["font_size_ratio"] = float(round(float(fsr), 2)) if fsr is not None else None
        except Exception:
            fp["font_size_ratio"] = None

        h = _canonical_json_hash(fp)
        is_special = ht_norm_l in _SPECIAL_HEADING_TYPES
        max_diff = _ID_MAX_PARAMS_DIFF_SPECIAL if is_special else _ID_MAX_PARAMS_DIFF_DEFAULT

        best_match_id = None
        best_match_diff = None
        for prev in seen:
            if prev["section"] != fp.get("section"):
                continue
            if prev["heading_type_norm"] != ht_norm_l:
                continue
            diffs = sum(
                1 for k in _FINGERPRINT_COLS
                if k not in {"section", "heading_type"} and prev["fp"].get(k) != fp.get(k)
            )
            if diffs <= max_diff:
                if best_match_diff is None or diffs < best_match_diff:
                    best_match_diff = diffs
                    best_match_id = prev["heading_fp_id"]
                    if diffs == 0:
                        break

        assigned_id = best_match_id if best_match_id is not None else next_id
        if best_match_id is None:
            next_id += 1

        out.at[i, "heading_fingerprint"] = fp
        out.at[i, "heading_hash"] = h
        out.at[i, "heading_fp_id"] = int(assigned_id)
        seen.append({"heading_fp_id": int(assigned_id), "section": fp.get("section"), "heading_type_norm": ht_norm_l, "fp": fp})

    return out


# ================================================================================
# Assign heading id
# ================================================================================

# block_type values treated as headings from here on. Also imported by step_06.
_HEADING_ROLES = {"heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph"}

# heading types whose hierarchy is determined by their own numeric depth
NUMBERED_HEADING_TYPES = {
    "numbered_heading",
    "roman_numbered_heading",
    "alpha_dotted_numbered_heading",
    "alpha_numbered_heading",
    # TODO add alpha heading
}

_RE_NUMERIC_PREFIX = re.compile(r'^\s*(\d+(?:\.\d+)*)')


def _assign_heading_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups consecutive heading rows that share the same heading_fp_id into a
    single heading_id (multi-line headings). Non-heading rows get NA.

    Numbered headings with different numeric prefixes are never merged,
    even when consecutive and sharing an fp_id.

    Runs BEFORE _suppress_headings so suppression can operate on whole heading
    units rather than individual lines — a named heading and its subtitle share
    one heading_id and are kept or dropped together. Suppression renumbers the
    survivors densely afterwards, so the ids this emits are provisional.
    """
    out = df.copy()
    out["heading_id"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    if "block_type" not in out.columns:
        return out

    is_heading = (
        out["block_type"].astype("string").str.strip().str.lower()
        .isin(_HEADING_ROLES).fillna(False)
    )
    if not is_heading.any():
        return out

    if "heading_fp_id" not in out.columns:
        next_id = 1
        for idx in out.index[is_heading]:
            out.at[idx, "heading_id"] = next_id
            next_id += 1
        return out

    heading_idx = out.index[is_heading]
    fp_vals = out.loc[heading_idx, "heading_fp_id"].values
    line_vals = (pd.to_numeric(out.loc[heading_idx, "line_id"], errors="coerce").values
                 if "line_id" in out.columns else None)
    text_vals = (out.loc[heading_idx, "text"].astype("string").fillna("").values
                 if "text" in out.columns else None)
    ht_vals = (out.loc[heading_idx, "heading_type"].astype("string").fillna("").values
               if "heading_type" in out.columns else None)
    hm_vals = (out.loc[heading_idx, "hierarchy_marker"].astype("string").fillna("").values
               if "hierarchy_marker" in out.columns else None)

    idx_list = list(heading_idx)
    next_id = 1
    i = 0
    while i < len(idx_list):
        cur_fp = fp_vals[i]
        cur_id = next_id
        out.at[idx_list[i], "heading_id"] = cur_id
        j = i + 1
        while j < len(idx_list):
            same_fp = pd.notna(cur_fp) and pd.notna(fp_vals[j]) and fp_vals[j] == cur_fp
            consecutive = (line_vals is None or (
                np.isfinite(line_vals[j - 1]) and np.isfinite(line_vals[j])
                and line_vals[j] == line_vals[j - 1] + 1
            ))
            # Never merge two numbered heading rows that have different numeric prefixes
            diff_numbered_prefix = False
            if same_fp and consecutive and ht_vals is not None and text_vals is not None:
                ht_j = str(ht_vals[j]).strip().lower()
                if ht_j in NUMBERED_HEADING_TYPES:
                    pfx_i = _RE_NUMERIC_PREFIX.match(str(text_vals[i]))
                    pfx_j = _RE_NUMERIC_PREFIX.match(str(text_vals[j]))
                    if pfx_j is not None:
                        pfx_i_str = pfx_i.group(0).strip() if pfx_i else ""
                        if pfx_i_str != pfx_j.group(0).strip():
                            diff_numbered_prefix = True
            # Named heading followed by a blank-marker subtitle on the next line.
            # The subtitle may itself span multiple lines sharing one heading_fp_id,
            # so also absorb further consecutive blank-marker lines that continue the
            # previous subtitle line's fp (where the prior marker is already blank).
            marker_prev = str(hm_vals[j - 1]).strip() if hm_vals is not None else ""
            marker_curr = str(hm_vals[j]).strip() if hm_vals is not None else ""
            same_prev_fp = (
                pd.notna(fp_vals[j - 1]) and pd.notna(fp_vals[j])
                and fp_vals[j - 1] == fp_vals[j]
            )
            named_heading_subtitle = (
                hm_vals is not None
                and consecutive
                and marker_curr == ""
                and (marker_prev != "" or same_prev_fp)
            )
            if (same_fp and consecutive and not diff_numbered_prefix) or named_heading_subtitle:
                out.at[idx_list[j], "heading_id"] = cur_id
                j += 1
            else:
                break
        i = j
        next_id += 1

    return out


# ================================================================================
# Suppress certain headings
# ================================================================================

_SUPPRESS_BLANK_COLS = (
    "heading_type", "heading_id", "heading_fp_id", "heading_fingerprint", "heading_hash",
)

# Magazine-page rule: a page is "magazine-style" (everything bolded/large, so most lines
# score as free_form headings) when it has at least this many free_form headings AND at
# least this fraction of them sit in back-to-back runs of consecutive heading lines.
# Genuine heading-dense pages (e.g. a 10-K "Services" page with 9 headings, each followed
# by a paragraph) stay below the run ratio because their headings are rarely adjacent.
_MAGAZINE_MIN_FREE_FORM = 6
_MAGAZINE_RUN_RATIO = 0.5


def _suppress_headings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rules (1)-(3) never touch headings declared by an upstream parser
    (heading_source in {docx, html}) — those are structural facts, not scores.

    (1) Suppress repeated headings: same (fp_id, text, align) occurring ≥3× (center)
        or ≥5× (other). Only applies to free_form headings — headings with an
        identified hierarchy_type marker are never suppressed. Skipped for pptx.
    (2) Coverpage rule: when 3+ consecutive heading line_ids exist on coverpage/page-1,
        keep only the best one (FORM/SCHEDULE prefix > highest score > lowest line_id).
        Free_form only. Skipped for pptx. Page-1 fallback needs >= 3 pages.
    (3) Magazine-page rule: on pages with >= _MAGAZINE_MIN_FREE_FORM free_form headings
        where >= _MAGAZINE_RUN_RATIO of them are back-to-back with another free_form
        heading, keep only the topmost free_form heading and suppress the rest.
        Requires real pagination (skipped when page_number is 1 everywhere).
    (4) Per-slide rule (pptx): when slide_index is present, keep only the first heading
        per slide (lowest line_id) and suppress all others on that slide. Includes
        hybrid_heading_paragraph rows, and unlike (1)-(3) applies to parser-declared
        headings too.

    Two suppression mechanisms, deliberately different:
      * rule (1) rewrites block_type to the sentinel 'suppressed_repeated_heading',
        which downstream treats as page furniture (excluded from chunks / text).
      * rules (2)-(4) blank block_type, so _finalize_block_types turns the row into
        an ordinary 'paragraph' and its text is retained.
    Both clear the columns in _SUPPRESS_BLANK_COLS.
    """
    out = df.copy()

    if "block_type" not in out.columns:
        return out

    def _is_heading_mask(x):
        return x.astype("string").str.strip().str.lower().eq("heading").fillna(False)

    def _expand_to_units(mask, protect=None):
        """
        Grow a row mask to cover every row sharing a selected row's heading_id, so
        a multi-line heading is suppressed whole rather than decapitated.

        `protect` marks rows a rule has explicitly decided to KEEP. A unit
        containing a protected row is dropped from the mask entirely rather than
        expanded into — otherwise expansion would swallow the very heading the
        rule chose to keep (rules 2-4 all pick a winner and suppress the rest).
        """
        if "heading_id" not in out.columns or mask is None or not mask.any():
            return mask
        hids = set(out.loc[mask, "heading_id"].dropna().unique())
        if not hids:
            return mask

        protected_hids = set()
        if protect is not None and protect.any():
            protected_hids = set(out.loc[protect, "heading_id"].dropna().unique())

        expand_hids = hids - protected_hids
        expanded = mask
        if expand_hids:
            expanded = expanded | out["heading_id"].isin(expand_hids).fillna(False)
        if protected_hids:
            # rescue the whole protected unit, including lines the rule marked
            expanded = expanded & ~out["heading_id"].isin(protected_hids).fillna(False)
        return expanded

    def _suppress_rows(mask, protect=None):
        mask = _expand_to_units(mask, protect)
        if mask is None or not mask.any():
            return
        out.loc[mask, "block_type"] = np.nan
        for c in _SUPPRESS_BLANK_COLS:
            if c in out.columns:
                out.loc[mask, c] = np.nan

    # A heading unit is "named" when ANY of its lines carries a hierarchy_marker.
    # Used by rule (1): the subtitle line of "Chapter 5 / Introduction" is itself
    # free_form and marker-less, so judged on its own it looks like a repeated
    # free_form heading. Judged as part of its unit, it is protected.
    if "heading_id" in out.columns and "hierarchy_marker" in out.columns:
        _has_marker = (
            out["hierarchy_marker"].astype("string").str.strip().str.len().fillna(0) > 0
        )
        _named_hids = out.loc[_has_marker, "heading_id"].dropna().unique()
        in_named_unit = out["heading_id"].isin(_named_hids).fillna(False)
    else:
        in_named_unit = pd.Series(False, index=out.index)

    # Rows declared as headings by an upstream parser (e.g. docx, html) are structural
    # facts — scoring-based suppression does not apply to them.
    docx_heading = (
        out.get("heading_source", pd.Series(dtype="object"))
           .astype("string").str.strip().str.lower()
           .isin(["docx", "html"])
           .fillna(False)
    )

    # (1) Repeated headings — skipped for pptx (rule 3 handles per-slide suppression)
    # Only suppress free_form headings; headings with an identified hierarchy_type marker are structural facts.
    is_free_form = (
        out.get("heading_type", pd.Series(dtype="object"))
           .astype("string").str.strip().str.lower()
           .eq("free_form")
           .fillna(True)  # treat missing heading_type as free_form (conservative)
        & ~in_named_unit  # a free_form subtitle of a named heading is not free-standing
    )
    is_heading = _is_heading_mask(out["block_type"]) & ~docx_heading & is_free_form
    if "heading_fp_id" in out.columns and "text" in out.columns and is_heading.any() and "slide_index" not in out.columns:
        fp = out.loc[is_heading, "heading_fp_id"]
        txt = out.loc[is_heading, "text"].astype("string").fillna("").str.strip()
        align = (out.loc[is_heading, "text_align"].astype("string").fillna("").str.strip().str.lower()
                 if "text_align" in out.columns else pd.Series("", index=out.loc[is_heading].index))
        counts = pd.DataFrame({"fp": fp, "txt": txt, "align": align}).groupby(["fp", "txt", "align"], dropna=False).size()
        bad_pairs = [(fv, tv, av) for (fv, tv, av), cnt in counts.items()
                     if cnt >= (3 if av == "center" else 5)]
        if bad_pairs:
            pair_df = pd.DataFrame({"fp": fp, "txt": txt, "align": align}, index=out.loc[is_heading].index)
            bad_mask = pair_df.set_index(["fp", "txt", "align"]).index.isin(bad_pairs)
            suppress_mask = pd.Series(False, index=out.index)
            suppress_mask.loc[pair_df.index] = bad_mask
            # Deliberately NOT unit-expanded. This rule judges each line by how
            # often its own text repeats, and _assign_heading_id merges any two
            # consecutive same-fp headings — including genuine siblings such as
            # "Key Terms of the Alerian ETNs" followed by "General". Expanding
            # here would let a repeated "General" drag its unrelated neighbour
            # down with it. Multi-line units are already protected from this
            # rule by in_named_unit above.
            if suppress_mask.any():
                out.loc[suppress_mask, "block_type"] = "suppressed_repeated_heading"
                for c in _SUPPRESS_BLANK_COLS:
                    if c in out.columns:
                        out.loc[suppress_mask, c] = np.nan

    # (2) Coverpage / page-1 cluster — skipped for pptx (rule 3 handles per-slide suppression)
    # Only suppresses free_form headings; headings with an identified hierarchy_type marker are kept.
    is_heading = _is_heading_mask(out["block_type"]) & ~docx_heading & is_free_form
    if is_heading.any() and "slide_index" not in out.columns:
        has_coverpage = ("section" in out.columns
                         and out.loc[is_heading, "section"].astype("string")
                         .str.strip().str.lower().eq("coverpage").any())
        if has_coverpage:
            scope_mask = out["section"].astype("string").fillna("").str.strip().str.lower().eq("coverpage")
        elif "page_number" in out.columns and out["page_number"].nunique() >= 3:
            scope_mask = out["page_number"].eq(out["page_number"].min())
        else:
            scope_mask = None

        scope_heading_idx = out.index[is_heading & scope_mask] if scope_mask is not None else pd.Index([])
        if len(scope_heading_idx) >= 3 and "line_id" in out.columns:
            sorted_ids = out.loc[scope_heading_idx, "line_id"].sort_values().values
            has_run = any(sorted_ids[i + 2] - sorted_ids[i] == 2 for i in range(len(sorted_ids) - 2))
            if has_run:
                scope_headings = out.loc[scope_heading_idx].copy()
                kw = (scope_headings["text"].astype("string").fillna("")
                      .str.contains(_FORM_PATTERN, case=False, regex=True)
                      if "text" in scope_headings.columns
                      else pd.Series(False, index=scope_headings.index))
                candidates = scope_headings.loc[scope_headings.index[kw]] if kw.any() else scope_headings
                if "heading_score" in candidates.columns:
                    hs = pd.to_numeric(candidates["heading_score"], errors="coerce").fillna(-1e9)
                    candidates = candidates.loc[hs == hs.max()]
                best_line_id = pd.to_numeric(candidates["line_id"], errors="coerce").min()
                keep_idx = candidates.index[pd.to_numeric(candidates["line_id"], errors="coerce") == best_line_id][0]
                # If the winner heading spans multiple merged lines (same heading_id), keep all of them
                if "heading_id" in out.columns:
                    winner_hid = out.loc[keep_idx, "heading_id"]
                    if pd.notna(winner_hid):
                        keep_set = out.index[out["heading_id"] == winner_hid]
                    else:
                        keep_set = pd.Index([keep_idx])
                else:
                    keep_set = pd.Index([keep_idx])
                suppress_mask = pd.Series(False, index=out.index)
                suppress_mask.loc[scope_heading_idx.difference(keep_set)] = True
                protect = pd.Series(False, index=out.index)
                protect.loc[keep_set] = True
                _suppress_rows(suppress_mask, protect)

    # (3) Magazine-page rule: on pages where most free_form headings follow straight onto
    # another free_form heading (design-heavy pages where everything is bolded/large), keep
    # only the topmost free_form heading as the page title and suppress the rest.
    # Adjacency is free_form-to-free_form only: a free_form under a marker heading (e.g.
    # "Item 1. Business" / "Company Background") is ordinary document structure, not magazine
    # layout. Only the follower of a pair counts, so a back-to-back pair scores 1, not 2.
    # Marker-based headings (numbered/alpha) are structural facts and are never suppressed.
    # Requires real pagination: skipped when page_number is 1 everywhere (formats without
    # page geometry would collapse into a single "page").
    is_heading = _is_heading_mask(out["block_type"]) & ~docx_heading
    is_ff_heading = is_heading & is_free_form
    has_real_pages = (
        "page_number" in out.columns
        and not pd.to_numeric(out["page_number"], errors="coerce").fillna(1).eq(1).all()
    )
    if is_ff_heading.any() and has_real_pages and "line_id" in out.columns:
        ff_page_counts = out.loc[is_ff_heading, "page_number"].value_counts()
        for page in ff_page_counts[ff_page_counts >= _MAGAZINE_MIN_FREE_FORM].index:
            page_order = (out.loc[out["page_number"].eq(page), "line_id"]
                          .pipe(pd.to_numeric, errors="coerce").sort_values().index)
            ff_seq = is_ff_heading.loc[page_order].to_numpy()
            prev_ff = np.roll(ff_seq, 1); prev_ff[0] = False
            follows_ff = ff_seq & prev_ff          # free_form directly under another free_form
            n_ff = int(ff_seq.sum())
            if n_ff and follows_ff.sum() / n_ff >= _MAGAZINE_RUN_RATIO:
                ff_idx = page_order[ff_seq]
                suppress_mask = pd.Series(False, index=out.index)
                suppress_mask.loc[ff_idx[1:]] = True  # keep topmost as page title
                protect = pd.Series(False, index=out.index)
                protect.loc[ff_idx[:1]] = True
                _suppress_rows(suppress_mask, protect)

    # (4) Per-slide rule: keep only the first heading per slide_index
    # Includes hybrid_heading_paragraph — suppressed ones become paragraph via _finalize_block_types
    is_heading = _is_heading_mask(out["block_type"])
    is_hybrid = out["block_type"].astype("string").str.strip().str.lower().eq("hybrid_heading_paragraph").fillna(False)
    is_any_heading = is_heading | is_hybrid
    if "slide_index" in out.columns and is_any_heading.any() and "line_id" in out.columns:
        slide_headings = out.loc[is_any_heading, ["slide_index", "line_id"]].copy()
        keep_idx = (
            slide_headings
            .groupby("slide_index", sort=False)["line_id"]
            .idxmin()
        )
        suppress_mask = pd.Series(False, index=out.index)
        suppress_mask.loc[slide_headings.index.difference(keep_idx)] = True
        protect = pd.Series(False, index=out.index)
        protect.loc[keep_idx] = True
        _suppress_rows(suppress_mask, protect)  # sets block_type to nan → paragraph via _finalize_block_types

    # Renumber survivors densely so heading_id stays a gapless 1..N sequence.
    # Suppression blanks whole units, which would otherwise leave holes in a
    # field that is public API (HierarchyNode.heading_id).
    if "heading_id" in out.columns and out["heading_id"].notna().any():
        out["heading_id"] = _dense_rank_keep_na(out["heading_id"])

    return out


# ================================================================================
# Finalize block roles
# ================================================================================

def _finalize_block_types(df: pd.DataFrame) -> pd.DataFrame:
    """Set all blank/NaN block_type values to 'paragraph'."""
    out = df.copy()
    if "block_type" in out.columns:
        blank_mask = out["block_type"].isna() | (out["block_type"].astype("string").str.strip() == "")
        if blank_mask.any():
            out.loc[blank_mask, "block_type"] = "paragraph"
    return out


# ================================================================================
# Public API
# ================================================================================

def detect_headings(
    lines_df: pd.DataFrame,
    compiled_patterns,  # HierarchyTypePatternConfig
) -> pd.DataFrame:
    """
    Heading detection pipeline. Answers: is this line a heading?

    Steps:
    1. Detect hierarchy markers
    2. Correct paren alpha/roman misclassification
    3. Validate alpha_heading series (singletons cleared to free_form)
    4. Add gap context — runs on the FULL frame, before prefiltering
    5. Score lines (prefiltered subset, merged back onto the full frame)
    6. Contextual score adjustments
    7. Heading decision
    8. Numbered section groups + hybrid heading text
    9. Heading fingerprints and fp_ids
   10. Group multi-line headings into heading_id
   11. Suppress repeated / coverpage / magazine-page / per-slide headings,
       then densely renumber the surviving heading_ids
   12. Finalize block roles (fill blanks → 'paragraph')

    Returns lines_df with columns added:
      hierarchy_marker, hierarchy_type, line_gap_below,
      heading_score, heading_score_debug, block_type, heading_type,
      heading_source, numbered_heading_group, hybrid_heading_text,
      heading_fingerprint, heading_hash, heading_fp_id, heading_id

    Note: when a 'line_id' column is present the scored subset is merged back by
    line_id, which replaces the caller's index with a fresh RangeIndex.
    """
    out = lines_df.copy()

    out = _detect_marker_candidates(out, compiled_patterns)
    out = _correct_paren_type_by_series(out)
    out = _validate_alpha_heading_series(out)

    out = _add_gap_context(out)

    scored_input = _pre_filter_lines(out)
    if "heading_score" not in out.columns:
        out["heading_score"] = 0.0
    if "heading_score_debug" not in out.columns:
        out["heading_score_debug"] = "{}"
    if len(scored_input) > 0:
        scored_slice = _add_heading_score(scored_input)
        if "line_id" in out.columns and "line_id" in scored_slice.columns:
            scored_key = scored_slice.drop_duplicates(subset=["line_id"], keep="first")[["line_id", "heading_score", "heading_score_debug"]]
            out = out.merge(scored_key, on="line_id", how="left", suffixes=("", "_scored"))
            out["heading_score"] = out["heading_score_scored"].fillna(out["heading_score"]).astype("float64")
            out["heading_score_debug"] = out["heading_score_debug_scored"].fillna(out["heading_score_debug"])
            out = out.drop(columns=["heading_score_scored", "heading_score_debug_scored"])
        else:
            idx = scored_slice.index.intersection(out.index)
            out.loc[idx, "heading_score"] = scored_slice.loc[idx, "heading_score"].astype("float64")
            out.loc[idx, "heading_score_debug"] = scored_slice.loc[idx, "heading_score_debug"]

    out = _apply_contextual_score_adjustments(out)
    out = _add_heading_decision(out)
    out = _assign_numbered_heading_groups(out)
    out = _extract_hybrid_heading_texts(out)
    out = _add_heading_fingerprints_and_ids(out)
    out = _assign_heading_id(out)
    out = _suppress_headings(out)
    out = _finalize_block_types(out)

    return out


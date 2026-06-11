"""
step_05_heading_detector.py

Answers the question: is this line a heading?

Public API: detect_headings(lines_df, compiled_patterns)

Pipeline:
  1. Detect hierarchy markers (numbered, parenthetical, named)
  2. Correct paren alpha/roman misclassification via series context
  3. Score lines (intrinsic per-line signals)
  4. Contextual score adjustments (single-line bonus, style-run penalty)
  5. Heading decision (threshold)
  6. Numbered section groups + hybrid heading text extraction
  7. Heading fingerprints and fp_ids
  8. Suppress repeated / coverpage headings
  9. Finalize block roles (blanks → 'paragraph')

The hierarchy tree (weights, heading_id, parent/level) lives in step_06.
"""

from __future__ import annotations
import hashlib
import json
import re

import numpy as np
import pandas as pd

# ================================================================================
# Pre-filter forbidden heading line formats (text that will never be a heading)
# ================================================================================

_FORBIDDEN_BLOCK_TYPES = {"table", "image", "hr", "page_label", "navigation", "toc", "exhibits", "speaker_notes", "shape", "chart"}

_FORBIDDEN_SUBSTRINGS = {
    # signature indicators
    "/s/", "signed:",
    # urls
    "http:", "https", "www.",
    # other
    "page"
}

_FORBIDDEN_START_TEXT = {
    # stars and quotes
    "*", "'",'"', "“", "”", "‘", "’", "„", "«", "»",
    # bullet tokens
    "-", "–", "—", "•", "·", "…", "■", "▪", "",
    "+", "☒", "☐", "○", "◦", "►", "▸", "‣", "⁃",
    "✓", "✔", "✗", "✘", "✖", "✕",
    # other punctuation
    "©", "®", "™", "§", "¶", "†", "‡", "‹", "›",
    # signature indicators
    "By:", "Name:", "Title:", "Date:",
}

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
    (4) rows where text contains any substring in _FORBIDDEN_SUBSTRINGS (case-insensitive)
    (5) rows where stripped text is fewer than 3 characters (e.g. decorative large first-letters)

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

    # (2) forbidden start tokens
    # normalize tokens to lowercase for comparison
    forbidden_tokens = sorted({t.lower() for t in _FORBIDDEN_START_TEXT if t is not None})
    if forbidden_tokens:
        # Build one regex that matches ANY forbidden token at start
        # - uses re.escape for safety
        # - matches after left-trim (we already lstrip, so anchor ^ is correct)
        tok_re = re.compile(r"^(?:%s)" % "|".join(re.escape(t) for t in forbidden_tokens))
        keep &= ~text_lower.str.match(tok_re)

    # (3) fully parenthesized lines
    keep &= ~text_lstrip.str.match(_PAREN_FULL_RE)

    # (4) forbidden substrings (anywhere in text)
    forbidden_substrings = sorted({s.lower() for s in _FORBIDDEN_SUBSTRINGS if s is not None})
    if forbidden_substrings:
        # Check if any forbidden substring appears anywhere in the text
        for substring in forbidden_substrings:
            keep &= ~text_lower.str.contains(re.escape(substring), na=False)

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

def _add_heading_score(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes lines_df and returns it with one extra column: `heading_score`.

    Scoring rules:
    [GENERAL]
    - font_size_ratio:
      - 1.01 to 1.2: +1
      - 1.2 to 1.4: +2
      - >1.4: +3
      - <0.75: -5
    - char_count:
      - < 50: +0.5
      - >100: -1
      - >250: -3
    - capitalized_word_ratio = capitalized_word_count/word_count
      - >0.75: +0.5
    - is_bold = true: +2.5
    - is_italic = true: +1.5
    - is_underlined = true: +1.5
    - text_align = center: +1
    - is_uppercase = true: +1

    [PDF specific]
    - layout_id consists out of only 1 line_id: +1
      (interpreted as: within the same layout_id, the number of rows/lines == 1)
    - layout_id has 4 or more lines: -0.5
    """
    out = lines_df.copy()

    score = pd.Series(0.0, index=out.index, dtype="float64")

    # --- font_size_ratio
    fsr = _to_float_series(out.get("font_size_ratio"), default=np.nan).fillna(1.0)
    c_fsr = pd.Series(0.0, index=out.index, dtype="float64")
    c_fsr += np.where((fsr >= 1.01) & (fsr < 1.2), 1.0, 0.0)
    c_fsr += np.where((fsr >= 1.2) & (fsr < 1.4), 2.0, 0.0)
    c_fsr += np.where((fsr >= 1.4), 3.0, 0.0)
    c_fsr += np.where((fsr < 0.75), -5.0, 0.0)
    score += c_fsr

    # --- char_count
    cc = _to_float_series(out.get("char_count"), default=np.nan)
    c_cc = pd.Series(0.0, index=out.index, dtype="float64")
    c_cc += np.where(cc < 50, 0.5, 0.0)
    c_cc += np.where(cc > 100, -1.0, 0.0)
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

    # --- layout_id density penalty (-0.5 if layout_id has 4+ lines)
    c_layout_density = pd.Series(0.0, index=out.index, dtype="float64")
    if "layout_id" in out.columns:
        layout_sizes = out.groupby("layout_id", sort=False)["layout_id"].transform("count")
        c_layout_density = pd.Series(np.where(layout_sizes >= 4, -0.5, 0.0), index=out.index, dtype="float64")
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
        "layout_density": c_layout_density,
    }, index=out.index)

    def _row_to_debug_json(row: pd.Series) -> str:
        return json.dumps({k: round(float(v), 4) for k, v in row.items() if v != 0.0})

    out["heading_score_debug"] = debug_components.apply(_row_to_debug_json, axis=1)

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

    # style key: bold|italic|underlined|font_family|font_size_ratio (bucketed to 1dp)
    # color_rarity already captures non_stroking_color; hierarchy_type and is_uppercase
    # caused too many accidental style-run hits in practice.
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

    # +2: no line within 2 positions shares the same style key
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

    # Append contextual adjustment to debug JSON — only patch rows where adj != 0
    if "heading_score_debug" in out.columns:
        nonzero_idx = adj_aligned.index[adj_aligned != 0.0]
        for _i in nonzero_idx:
            v = adj_aligned.loc[_i]
            try:
                d = json.loads(out.at[_i, "heading_score_debug"])
            except (ValueError, TypeError):
                d = {}
            d["contextual_adjustment"] = round(float(v), 4)
            out.at[_i, "heading_score_debug"] = json.dumps(d)

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
    Final heading decision:
    - If heading_score > threshold AND block_type is not already set:
        - set block_type = "heading"
        - set heading_type = hierarchy_type if present else default_heading_type
    - Otherwise:
        - preserve existing block_type (e.g., toc_heading, page_label, etc.)
        - heading_type is set for all rows passing threshold (regardless of existing role)

    This ensures that special roles like toc_heading are preserved while still
    getting their heading_type populated based on their hierarchy detection.

    Adds/updates:
      - block_type (string) - only for unassigned rows
      - heading_type (string) - for all rows passing threshold
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


def _is_valid_numbered_continuation(prev: tuple, curr: tuple, max_jump: int = 20) -> bool:
    """
    Returns True if curr is a logical next step after prev in a numbered hierarchy.

    Strategy: trim prev to the depth of curr (going shallower is always ok),
    find the first divergence index d, then:
      - at d: curr[d] must advance by 1..max_jump over prev[d]
      - after d: all curr[d+1:] must be 1 (restarting each child level)

    Special case: no divergence (curr is a direct child of prev) — valid only
    if curr goes exactly one level deeper and starts at 1.

    Examples:
      (1,3) → (2,1)    valid  — parent advances, child restarts at 1
      (8,)  → (8,1)    valid  — first child
      (8,2) → (9,)     valid  — go up and advance
      (1,3) → (2,2)    invalid — child didn't restart at 1
      (11,) → (1,)     invalid — negative advance (restart → new group)
      (11,5)→ (501,)   invalid — jump too large
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
    2. Groups sequential rows into logical series ordered by line_id.
       A new group starts whenever the transition is not a valid continuation.
       Singleton rows (no valid neighbour on either side) are left ungrouped.
    3. Within each (group, depth-level), if at least one row already has
       block_type = 'heading', every other non-special row at the same depth
       is promoted to 'hybrid_heading_paragraph'.

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

    # State-machine grouping: keep a series "alive" across noise interruptions.
    #
    # active  — the series currently being extended {gid, last}
    # paused  — series interrupted by noise, kept alive in case they resume
    #           (most-recently-paused last, so we prefer resuming the freshest)
    #
    # When a value can't continue active:
    #   1. Park active into paused.
    #   2. Walk paused newest→oldest: first one that accepts value is resumed.
    #   3. If none accept: start a new group.
    # Unparseable rows (value=None) are skipped entirely — they don't interrupt state.
    # After assignment, groups of size 1 (no valid neighbour) are marked stray (null).
    active = None       # {"gid": int, "last": tuple}
    paused: list = []   # [{"gid": int, "last": tuple}, ...]
    group_counter = 0
    row_group_ids: dict = {}   # cand.index → raw gid

    for idx, value in values.items():
        if value is None:
            continue  # noise that didn't parse — leave state untouched

        if active and _is_valid_numbered_continuation(active["last"], value):
            active["last"] = value
            row_group_ids[idx] = active["gid"]
            continue

        # Active can't accept → pause it, try to resume a paused series
        if active:
            paused.append(active)
            active = None

        resumed = None
        for i in range(len(paused) - 1, -1, -1):
            if _is_valid_numbered_continuation(paused[i]["last"], value):
                resumed = paused.pop(i)
                break

        if resumed:
            resumed["last"] = value
            active = resumed
        else:
            group_counter += 1
            active = {"gid": group_counter, "last": value}

        row_group_ids[idx] = active["gid"]

    # Build Series aligned to cand.index (rows with value=None get NA)
    raw_group = pd.Series(row_group_ids, dtype="Int64").reindex(cand.index)

    # Singletons → stray (null group); dense-rank survivors so groups start at 1
    vc = raw_group.value_counts()
    grp_size = raw_group.map(vc)
    final_group = (
        raw_group.where(grp_size > 1)
        .rank(method="dense", na_option="keep")
        .astype("Int64")
    )

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
    "x_left_bucket",
    "font_name",
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
# Suppress certain headings
# ================================================================================

_SUPPRESS_BLANK_COLS = (
    "heading_type", "heading_id", "heading_fp_id", "heading_fingerprint", "heading_hash",
)


def _suppress_headings(df: pd.DataFrame) -> pd.DataFrame:
    """
    (1) Suppress repeated headings: same text ≥3× (center) or ≥5× (other) within fp_id.
        Only applies to free_form headings — headings with an identified hierarchy_type marker are never suppressed.
    (2) Coverpage rule: when 3+ consecutive heading line_ids exist on coverpage/page-1,
        keep only the best one (FORM/SCHEDULE prefix > highest score > lowest line_id).
    (3) Per-slide rule (pptx): when slide_index is present, keep only the first heading
        per slide (lowest line_id) and suppress all others on that slide.
    """
    out = df.copy()

    if "block_type" not in out.columns:
        return out

    def _is_heading_mask(x):
        return x.astype("string").str.strip().str.lower().eq("heading").fillna(False)

    def _suppress_rows(mask):
        if mask is None or not mask.any():
            return
        out.loc[mask, "block_type"] = np.nan
        for c in _SUPPRESS_BLANK_COLS:
            if c in out.columns:
                out.loc[mask, c] = np.nan

    # Rows declared as headings by an upstream parser (e.g. docx) are structural facts —
    # scoring-based suppression does not apply to them.
    docx_heading = (
        out.get("heading_source", pd.Series(dtype="object"))
           .astype("string").str.strip().str.lower()
           .eq("docx")
           .fillna(False)
    )

    # (1) Repeated headings — skipped for pptx (rule 3 handles per-slide suppression)
    # Only suppress free_form headings; headings with an identified hierarchy_type marker are structural facts.
    is_free_form = (
        out.get("heading_type", pd.Series(dtype="object"))
           .astype("string").str.strip().str.lower()
           .eq("free_form")
           .fillna(True)  # treat missing heading_type as free_form (conservative)
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
                #TODO at this point heading_id is not in the df, only gets added by step 06
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
                _suppress_rows(suppress_mask)

    # (3) Per-slide rule: keep only the first heading per slide_index
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
        _suppress_rows(suppress_mask)  # sets block_type to nan → paragraph via _finalize_block_types

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
    3. Score lines (prefiltered subset)
    4. Contextual score adjustments
    5. Heading decision
    6. Numbered section groups + hybrid heading text
    7. Heading fingerprints and fp_ids
    8. Suppress repeated / coverpage headings
    9. Finalize block roles (fill blanks → 'paragraph')

    Returns lines_df with columns added:
      hierarchy_marker, hierarchy_type, heading_score, block_type, heading_type,
      numbered_heading_group, hybrid_heading_text,
      heading_fingerprint, heading_hash, heading_fp_id
    """
    out = lines_df.copy()

    out = _detect_marker_candidates(out, compiled_patterns)
    out = _correct_paren_type_by_series(out)
    out = _validate_alpha_heading_series(out)

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
    out = _suppress_headings(out)
    out = _finalize_block_types(out)

    return out


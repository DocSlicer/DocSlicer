from __future__ import annotations
import re
import numpy as np
import pandas as pd


# ==================================================
# BULLET & LIST MARKER DETECTION
# ==================================================

# Symbol bullets — frozenset for O(1) lookup.
# Canonical union of every bullet glyph the pipeline recognised in scattered
# local sets (was duplicated/diverged across block_merger and the line builders).
# Keep this the single source of truth.
_BULLET_TOKENS: frozenset[str] = frozenset({
    "-", "–", "—",                      # hyphen / en-dash / em-dash
    "•", "·", "∙",                      # classic bullets / bullet operator
    "○", "◦",                           # white bullets
    "●",                                # black circle
    "■", "▪", "□",                      # squares
    "◆", "◇",                           # diamonds
    "►", "▸", "▶",                      # triangles
    "➤", "➢",                           # arrows
    "‣", "⁃",                           # triangular / hyphen bullet
    "",                                # private-use bullet glyph
    "…",                                # ellipsis leader
    "+", "*",                           # plus / asterisk markers
    "☒", "☐",                           # ballot boxes
    "✓", "✔", "✗", "✘", "✖", "✕",       # check / cross marks
    "o",                                # lowercase-o OCR bullet (exact match only)
})

# Subset safe to match as a *leading* character (e.g. "•Item", "- item").
# Alphanumeric tokens like "o" are excluded: valid as a standalone bullet ("o"
# on its own) but matching them as a first char would flag ordinary words
# ("October") as bullets. Derived so the glyph list has one source of truth.
_BULLET_PREFIX_CHARS: frozenset[str] = frozenset(
    t for t in _BULLET_TOKENS if not t.isalnum()
)

# Strict roman-numeral core — validates roman *structure*, not just roman letters,
# so English words built from i/v/x/l/c/d/m ("mild", "did", "mill") are rejected.
# The leading lookahead forces ≥1 character, since the value groups are all optional.
_ROMAN = (
    r'(?=[MDCLXVI])'
    r'M?(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
)

# Structured list-token regex — covers the hierarchy types from the yaml
# (numbered, roman, parenthetical, alpha) as *standalone* tokens.
# Pre-compiled once; match is O(n) on token length, typically < 10 chars.
_LIST_MARKER_RE = re.compile(
    r'^(?:'
    r'\d+(?:\.\d+)*\.?(?:\([A-Za-z0-9]{1,4}\))+'  # numbered+paren combo: 4.3(a)  4.3(a)(i)
    r'|\d+(?:\.\d+)*\.?'         # numbered:      1.  1.1.  2.3.1
    r'|\([A-Za-z0-9]{1,4}\)'    # parens alpha/numeric/roman: (a) (iv) (28) (aa)
    r'|\d{1,2}\)'               # half-open numbered:  1)  9)  28)
    r'|[A-Za-z]\)'              # half-open single alpha:  a)  A)
    rf'|{_ROMAN}\)'             # half-open roman:  ii)  iv)  viii)
    r'|\[\d+\]'                 # bracketed numeric: [1]
    rf'|{_ROMAN}\.?'            # roman numeral standalone: iv.  viii  XIV.
    r'|[A-Za-z]\.'              # single alpha with dot: A.  B.
    r')$',
    re.IGNORECASE,
)


def is_bullet_token(text: object) -> bool:
    """True if *text* is a standalone symbol bullet (•, ■, –, etc.)."""
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    return str(text).strip() in _BULLET_TOKENS


def is_list_marker(text: object) -> bool:
    """
    True if *text* is a standalone list/hierarchy marker that should not
    participate in gutter detection as a left-side anchor word.

    Covers:
      - Symbol bullets  (•, ■, –, …)
      - Numbered tokens (1.  1.1.  2.3.1)
      - Numbered+paren combo (4.3(a)  4.3(a)(i))
      - Parenthetical   ((a)  (iv)  (28)  (aa))
      - Half-open paren (1)  a)  iv)  28))
      - Bracketed       ([1])
      - Roman numerals  (iv.  viii  XIV.)
      - Single alpha    (A.  B.)
    """
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    t = str(text).strip()
    if not t:
        return False
    return t in _BULLET_TOKENS or bool(_LIST_MARKER_RE.match(t))


def list_marker_mask(series: pd.Series) -> pd.Series:
    """
    Vectorized :func:`is_list_marker` for a whole text column → boolean Series.
    One isin + one pre-compiled regex match, both C-level; safe on full columns.
    """
    s = series.fillna("").astype(str).str.strip()
    return s.isin(_BULLET_TOKENS) | s.str.match(_LIST_MARKER_RE)


def is_bullet_line(text: object) -> bool:
    """
    True if *text* is a bullet line: either a standalone bullet glyph, or a line
    whose first non-space character is an (unambiguous) bullet glyph.

    O(1): one frozenset membership test for the standalone case and one for the
    leading character — no regex, no per-token loop, so it stays cheap when run
    per fragment inside a join.
    """
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    s = str(text).strip()
    if not s:
        return False
    if s in _BULLET_TOKENS:
        return True
    return s[0] in _BULLET_PREFIX_CHARS


def bullet_line_mask(series: pd.Series) -> pd.Series:
    """
    Vectorized :func:`is_bullet_line` for a whole text column → boolean Series.
    Pure C-level string ops (strip / isin / str[0]); safe on full columns.
    """
    s = series.fillna("").astype(str).str.strip()
    return s.isin(_BULLET_TOKENS) | s.str[0].isin(_BULLET_PREFIX_CHARS)


# Strict subset: glyphs that are practically always bullets even when they
# appear *mid-line*. Excludes the ambiguous markers (- – — + * … ·), which are
# usually math signs, numeric ranges, or dashes inside a sentence ("2-4y",
# "+Evrysdi", "a · b") — those are only trustworthy at the start of a line
# (is_bullet_line), never inside one. Derived from _BULLET_PREFIX_CHARS so the
# glyph list keeps one source of truth.
_AMBIGUOUS_BULLET_CHARS: frozenset[str] = frozenset({"-", "–", "—", "+", "*", "…", "·"})
_STRICT_BULLET_CHARS: frozenset[str] = _BULLET_PREFIX_CHARS - _AMBIGUOUS_BULLET_CHARS


def is_strict_bullet(text: object) -> bool:
    """
    True if *text* begins with an unambiguous bullet glyph (•, ▪, ►, ✓, …) —
    safe to treat as a list marker even mid-line, unlike :func:`is_bullet_line`.
    """
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    s = str(text).strip()
    return bool(s) and s[0] in _STRICT_BULLET_CHARS


def strict_bullet_mask(series: pd.Series) -> pd.Series:
    """Vectorized :func:`is_strict_bullet` for a whole text column → boolean Series."""
    s = series.fillna("").astype(str).str.strip()
    return s.str[0].isin(_STRICT_BULLET_CHARS)


# ==================================================
# CURRENCY SYMBOLS
# ==================================================

# Canonical currency-symbol set — single source of truth, consolidated from the
# per-module copies that had diverged across the repo (table_utils._CUR_SYM,
# gutter_detector._NUMERIC_VALUE_RE, html step_05._CURRENCY_TOKENS, toc_detector).
# Covers the Latin-1 currency signs ($ ¢ £ ¤ ¥) plus the entire Unicode
# "Currency Symbols" block U+20A0–U+20BF (€ ₹ ₽ ₩ ₪ ₺ … and the rest).
_CURRENCY_SYMBOLS_STR: str = "$¢£¤¥" + "".join(chr(c) for c in range(0x20A0, 0x20C0))
_CURRENCY_SYMBOLS: frozenset[str] = frozenset(_CURRENCY_SYMBOLS_STR)

# Same set as a regex character-class body, e.g. rf"{_CURRENCY_SYM_CLASS}?\d+".
# All members are literal inside a character class (none are ^ ] - \), so the
# raw string drops straight in.
_CURRENCY_SYM_CLASS: str = f"[{_CURRENCY_SYMBOLS_STR}]"


def is_currency_symbol(text: object) -> bool:
    """True if *text* is a standalone currency symbol ($, €, £, ¥, ₹, …)."""
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    return str(text).strip() in _CURRENCY_SYMBOLS


# ==================================================
# NUMERIC VALUE TOKENS
# ==================================================

# Dash placeholders used for "no value" cells in financial tables, and reused
# below as the set of valid range separators ("10.3 – 11.2").
_DASH_TOKENS: frozenset[str] = frozenset({"-", "–", "—", "−"})

# Numeric value token: any run of digits, whitespace, and "numeric punctuation"
# (thousands/decimal separators, dash ranges, brackets, currency, %), as long
# as it contains at least one digit — 123 / 1,234.5 / $5 / 12% / (123) /
# 57 (42, 72) / 15.4% (10.6, 20.2) / 10.3 - 11.2. A structured token-by-token
# grammar was tried here before and had multiple overlapping-optional spots
# (e.g. a comma could be "thousands separator" or "list separator", or a
# space could be matched by two adjacent `\s?`s) that made the regex engine
# backtrack combinatorially on non-matching input, taking whole seconds on a
# single cell. A flat character class has no such ambiguity — it's a single
# linear scan — and a unit word like "mg" still correctly fails the match
# since letters aren't in the class.
_NUMERIC_PUNCT = ",.()[]%" + "".join(_DASH_TOKENS) + _CURRENCY_SYMBOLS_STR
_NUMERIC_VALUE_RE = re.compile(
    rf'^(?=.*\d)[\s\d{re.escape(_NUMERIC_PUNCT)}]+$'
)

# NA-style "no value" placeholders, same role as the dashes ("Revenue  NA
# 1,234  5%"). Matched case-insensitively (NA / n/a / N.A.).
_NA_PLACEHOLDER_TOKENS: frozenset[str] = frozenset({"na", "n/a", "n.a."})

# Standalone unit tokens that flank numbers in table value columns: a detached
# percent sign ("17 %" extracted as two words), same role as a detached
# currency symbol.
_UNIT_TOKENS: frozenset[str] = frozenset({"%"})


def is_numeric_value(text: object) -> bool:
    """
    True if *text* is a numeric/currency table value: a numeric value token
    (see _NUMERIC_VALUE_RE), a standalone currency symbol or percent sign, or
    a dash / NA placeholder.
    """
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    t = str(text).strip()
    if not t:
        return False
    return (
        t in _DASH_TOKENS
        or t.lower() in _NA_PLACEHOLDER_TOKENS
        or t in _CURRENCY_SYMBOLS
        or t in _UNIT_TOKENS
        or bool(_NUMERIC_VALUE_RE.match(t))
    )


def numeric_value_mask(series: pd.Series) -> pd.Series:
    """Vectorized :func:`is_numeric_value` for a whole text column → boolean Series."""
    s = series.fillna("").astype(str).str.strip()
    return (
        s.isin(_DASH_TOKENS)
        | s.str.lower().isin(_NA_PLACEHOLDER_TOKENS)
        | s.isin(_CURRENCY_SYMBOLS)
        | s.isin(_UNIT_TOKENS)
        | s.str.match(_NUMERIC_VALUE_RE)
    )


# ==================================================
# FONT FEATURE HELPERS
# ==================================================

ITALIC_RE = re.compile(r"(italic|ital|oblique|slanted|cursive|skew|obl)", re.I)
BOLD_RE   = re.compile(r"(bold|black|semi[- ]?bold|demi|medium|medi|heavy|extra|ultra)", re.I)

# Computer Modern fonts encode style positionally, not as suffixes.
# CMB* → bold (e.g. CMBX10, CMBXTI10, CMBXSL10, CMBSY10)
# CMTI*, CMSL* → italic/slanted (but CMBX is bold, not italic)
_CM_BOLD_RE   = re.compile(r"^(?:[A-Z]{6}\+)?CMB", re.I)
_CM_ITALIC_RE = re.compile(r"^(?:[A-Z]{6}\+)?CM(?:TI|SL)", re.I)

# Japanese fonts (Hiragino, Yu Gothic, etc.) use W-weight suffixes.
# W6+ is bold; W1–W5 is regular/light.
_JP_BOLD_RE = re.compile(r"[- ]W[6-9](?:\D|$)")


def _is_bold_font(font_name: str) -> bool:
    return bool(BOLD_RE.search(font_name) or _CM_BOLD_RE.match(font_name) or _JP_BOLD_RE.search(font_name))


def _is_italic_font(font_name: str) -> bool:
    return bool(ITALIC_RE.search(font_name) or _CM_ITALIC_RE.match(font_name))

FONT_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
FONT_DIGIT_SUFFIX  = re.compile(r"\+\d+$")

# Each atom is a full or abbreviated style word; longer alternatives come first so
# the engine doesn't greedily consume a short prefix (e.g. "Ital" before "Italic").
# The `+` quantifier allows compound suffixes like -BoldItalic or -LightObl.
_STYLE_ATOMS = "|".join([
    "Italic", "Ital",
    "Oblique", "Obl",
    "Regular", "Regu", "Reg",
    "Bold", "Bd",
    "Light",
    "Medium", "Medi", "Med",
    "Heavy", "Hvy",
    "Black", "Blk",
    "Condensed", "Cond",
    "Extended", "Ext",
    "Narrow", "Nr",
    "Thin", "Demi", "Semi", "Ultra", "Extra",
    "Book", "Normal", "Plain", "Roman",
    "Upright", "Wide",
    "It",  # short for Italic; must follow longer Ital/Italic entries
])
HYPHENATED_STYLE_SUFFIX = re.compile(
    r"[-_](?:" + _STYLE_ATOMS + r")+$",
    re.I,
)
POSTSCRIPT_SUFFIX = re.compile(
    r"(psmt|mt|ps|ms|tt|std|pro|cyr|ce|baltic|greek|tur|"
    r"hebrew|arabic|vietnamese|eot|woff|ttf)$",
    re.I,
)


def _extract_font_family(font_name: str) -> str:
    """
    Extract the base font family from a full font name.

    Examples:
        Helvetica-Oblique   → Helvetica
        ABCDEE+Calibri-Bold → Calibri
        TimesNewRomanPSMT   → TimesNewRoman
    """
    if not font_name:
        return ""
    family = FONT_SUBSET_PREFIX.sub("", font_name)
    family = FONT_DIGIT_SUFFIX.sub("", family)
    prev_family = ""
    while prev_family != family:
        prev_family = family
        family = POSTSCRIPT_SUFFIX.sub("", family)
        family = HYPHENATED_STYLE_SUFFIX.sub("", family)
    family = family.rstrip("-_")
    return family if family else font_name


# ==================================================
# MAIN FEATURE FUNCTION
# ==================================================

def add_calculated_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated text features to the DataFrame.
    Vectorized where possible for maximum performance.

    Adds (only if required input columns exist):
        - font_family             [requires: font_name]
        - bold_ratio              [requires: font_name]
        - italic_ratio            [requires: font_name]
        - char_count              [requires: text]
        - alpha_count             [requires: text]
        - digit_count             [requires: text]
        - uppercase_count         [requires: text]
        - word_count              [requires: text]
        - alpha_word_count        [requires: text]
        - capitalized_word_count  [requires: text]
        - has_link                [requires: link_url]
        - link_type               [requires: link_url]
    """
    if df.empty:
        return df

    out = df.copy()

    # ---- Font features ----
    # Compute regex once per unique font name, then broadcast via map.
    # A typical document has O(10) distinct fonts but O(10_000) words.
    if "font_name" in out.columns:
        font_name = out["font_name"].fillna("").astype(str)
        unique_fonts = font_name.unique()
        if "font_family" not in out.columns:
            _fm = {f: _extract_font_family(f) for f in unique_fonts}
            out["font_family"] = font_name.map(_fm)
        if "bold_ratio" not in out.columns:
            _bm = {f: 1.0 if _is_bold_font(f) else 0.0 for f in unique_fonts}
            out["bold_ratio"] = font_name.map(_bm)
        if "italic_ratio" not in out.columns:
            _im = {f: 1.0 if _is_italic_font(f) else 0.0 for f in unique_fonts}
            out["italic_ratio"] = font_name.map(_im)

    # ---- Text features ----
    # Same dedup trick as fonts: word text repeats heavily ("the", digits,
    # boilerplate), so compute each feature once per unique string and
    # broadcast back through the factorize codes.
    if "text" in out.columns:
        text = out["text"].fillna("").astype(str)
        codes, uniques = pd.factorize(text)
        utext = pd.Series(uniques)

        def _bcast(values: pd.Series) -> np.ndarray:
            return values.to_numpy()[codes]

        if "char_count" not in out.columns:
            out["char_count"] = _bcast(utext.str.len())
        if "alpha_count" not in out.columns:
            out["alpha_count"] = _bcast(utext.str.count(r'[^\W\d_]'))
        if "digit_count" not in out.columns:
            out["digit_count"] = _bcast(utext.str.count(r'\d'))
        if "uppercase_count" not in out.columns:
            out["uppercase_count"] = _bcast(utext.str.count(r'[A-Z]'))
        if "word_count" not in out.columns:
            out["word_count"] = _bcast(utext.str.count(r'\S+'))
        # These two stay as Python lambdas: str.isalpha/isupper are
        # Unicode-aware, unlike the RE2 engine behind arrow-backed
        # str.count. Running them on uniques keeps them cheap.
        if "alpha_word_count" not in out.columns:
            out["alpha_word_count"] = _bcast(utext.apply(
                lambda s: sum(1 for w in s.split() if any(map(str.isalpha, w)))
            ))
        if "capitalized_word_count" not in out.columns:
            out["capitalized_word_count"] = _bcast(utext.apply(
                lambda s: sum(1 for w in s.split() if w and w[0].isupper())
            ))

    # ---- Link features ----
    if "link_url" in out.columns:
        stripped = out["link_url"].fillna("").astype(str).str.strip()
        nonempty = stripped.ne("")
        if "has_link" not in out.columns:
            out["has_link"] = nonempty
        if "link_type" not in out.columns:
            out["link_type"] = np.where(
                nonempty,
                np.where(stripped.str.startswith("#"), "internal", "external"),
                None,
            )

    return out

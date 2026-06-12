from __future__ import annotations
import re
import pandas as pd


# ==================================================
# BULLET & LIST MARKER DETECTION
# ==================================================

# Symbol bullets — frozenset for O(1) lookup
_BULLET_TOKENS: frozenset[str] = frozenset({
    "-", "–", "—",          # hyphen / en-dash / em-dash
    "•", "·",               # classic bullets
    "■", "▪", "",          # squares / special bullet glyphs
    "…",                    # ellipsis leader
    "+", "☒", "☐",
    "○", "◦", "►", "▸", "‣", "⁃",
    "✓", "✔", "✗", "✘", "✖", "✕",
})

# Structured list-token regex — covers the hierarchy types from the yaml
# (numbered, roman, parenthetical, alpha) as *standalone* tokens.
# Pre-compiled once; match is O(n) on token length, typically < 10 chars.
_LIST_MARKER_RE = re.compile(
    r'^(?:'
    r'\d+(?:\.\d+)*\.?'         # numbered:      1.  1.1.  2.3.1
    r'|\([A-Za-z0-9]{1,4}\)'    # parens alpha/numeric/roman: (a) (iv) (28) (aa)
    r'|\[\d+\]'                 # bracketed numeric: [1]
    r'|[ivxlcdm]+\.?'           # roman numeral standalone: iv.  viii  XIV.
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
      - Parenthetical   ((a)  (iv)  (28)  (aa))
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
    if "text" in out.columns:
        text = out["text"].fillna("").astype(str)
        if "char_count" not in out.columns:
            out["char_count"] = text.str.len()
        if "alpha_count" not in out.columns:
            out["alpha_count"] = text.str.count(r'[^\W\d_]')
        if "digit_count" not in out.columns:
            out["digit_count"] = text.str.count(r'\d')
        if "uppercase_count" not in out.columns:
            out["uppercase_count"] = text.str.count(r'[A-Z]')
        if "word_count" not in out.columns:
            out["word_count"] = text.str.split().str.len().fillna(0).astype(int)
        if "alpha_word_count" not in out.columns:
            out["alpha_word_count"] = text.apply(
                lambda s: sum(1 for w in s.split() if any(map(str.isalpha, w)))
            )
        if "capitalized_word_count" not in out.columns:
            out["capitalized_word_count"] = text.apply(
                lambda s: sum(1 for w in s.split() if w and w[0].isupper())
            )

    # ---- Link features ----
    if "link_url" in out.columns:
        link_url = out["link_url"].fillna("").astype(str)
        if "has_link" not in out.columns:
            out["has_link"] = link_url.apply(lambda url: bool(url and url.strip()))
        if "link_type" not in out.columns:
            def _link_type(url: str) -> str | None:
                if not url or not url.strip():
                    return None
                return "internal" if url.strip().startswith("#") else "external"
            out["link_type"] = link_url.apply(_link_type)

    return out

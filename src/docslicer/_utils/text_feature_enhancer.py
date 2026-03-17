from __future__ import annotations
import pandas as pd
import re

# ==================================================
# Helper functions
# ==================================================

# Italic and bold detection from fontname (e.g., BCDGEE+Calibri-Bold)
ITALIC_RE = re.compile(r"(italic|ital|oblique|slanted|cursive|skew|obl)", re.I)
BOLD_RE = re.compile(r"(bold|black|semi[- ]?bold|demi|medium|heavy|extra|ultra)", re.I)

# PDF subset prefix pattern (e.g., "ABCDEE+")
FONT_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

# Font style suffixes that come after hyphens/underscores
HYPHENATED_STYLE_SUFFIX = re.compile(
    r"[-_](bold|italic|oblique|regular|light|medium|heavy|black|thin|"
    r"demi|semi|extra|ultra|book|normal|plain)$",
    re.I
)

# PostScript and file format suffixes (no hyphen required)
POSTSCRIPT_SUFFIX = re.compile(
    r"(mt|ps|psmt|ms|tt|std|pro|cyr|ce|baltic|greek|tur|"
    r"hebrew|arabic|vietnamese|eot|woff|ttf)$",
    re.I
)


def _extract_font_family(font_name: str) -> str:
    """
    Extract the base font family from a full font name.
    
    Strategy:
      1. Remove PDF subset prefix (e.g., "ABCDEE+")
      2. Iteratively remove PostScript suffixes AND hyphenated style suffixes
         until no more changes occur (handles multiple suffixes like "CalistoMT-Italic")
      3. Preserve width variants that aren't hyphenated (e.g., "RobotoCondensed")
    
    Examples:
        Helvetica-Oblique → Helvetica
        CourierNew-Italic → CourierNew
        Calibri-Bold → Calibri
        ABCDEE+Calibri-Bold → Calibri
        TimesNewRomanPSMT → TimesNewRoman
        ArialUnicodeMS → ArialUnicode
        RobotoCondensed-Regular → RobotoCondensed
        HelveticaNeue-Light → HelveticaNeue
        CalistoMT-Italic → Calisto
        Arial-BoldMT → Arial
        TimesNewRomanPS-BoldMT → TimesNewRoman
    
    Args:
        font_name: Full font name from PDF
        
    Returns:
        Base font family name with styles/suffixes removed
    """
    if not font_name:
        return ""
    
    # Remove PDF subset prefix (e.g., "ABCDEE+")
    family = FONT_SUBSET_PREFIX.sub("", font_name)
    
    # Iteratively remove both PostScript suffixes and hyphenated style suffixes
    # Keep looping until no more changes occur
    # This handles cases like "CalistoMT-Italic" → "CalistoMT" → "Calisto"
    prev_family = ""
    while prev_family != family:
        prev_family = family
        # Remove PostScript/format suffixes (e.g., "MT", "PSMT", "MS")
        family = POSTSCRIPT_SUFFIX.sub("", family)
        # Remove hyphenated style suffixes (e.g., "-Bold", "-Italic")
        family = HYPHENATED_STYLE_SUFFIX.sub("", family)
    
    # Clean up any trailing hyphens or underscores
    family = family.rstrip("-_")
    
    return family if family else font_name


# ==================================================
# Main function
# ==================================================

def add_calculated_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated text features to the DataFrame.
    Vectorized where possible for maximum performance.
    
    Adds (only if required input columns exist):
        - font_family: str (base font family extracted from font_name) [requires: font_name]
        - bold_ratio: float (0.0 or 1.0 based on fontname) [requires: font_name]
        - italic_ratio: float (0.0 or 1.0 based on fontname) [requires: font_name]
        - char_count: int [requires: text]
        - alpha_count: int [requires: text]
        - digit_count: int [requires: text]
        - uppercase_count: int [requires: text]
        - word_count: int (whitespace-separated words) [requires: text]
        - alpha_word_count: int (words containing at least one alpha char) [requires: text]
        - capitalized_word_count: int (words starting with uppercase) [requires: text]
        - has_link: bool (True if link_url has a value) [requires: link_url]
        - link_type: str ("external" | "internal") [requires: link_url]
    
    Returns:
        DataFrame with additional columns added
    """
    if df.empty:
        return df
    
    out = df.copy()
    
    # =============================
    # Font Family Extraction
    # =============================
    if "font_name" in out.columns:
        font_name = out["font_name"].fillna("").astype(str)
        
        if "font_family" not in out.columns:
            out["font_family"] = font_name.apply(_extract_font_family)
        
        # =============================
        # Bold / Italic Detection (vectorized)
        # =============================
        # These are binary (0.0 or 1.0) to match the schema expectation
        
        # Calculate ratios directly without intermediate boolean columns
        if "bold_ratio" not in out.columns:
            out["bold_ratio"] = font_name.apply(lambda f: 1.0 if BOLD_RE.search(f) else 0.0)
        if "italic_ratio" not in out.columns:
            out["italic_ratio"] = font_name.apply(lambda f: 1.0 if ITALIC_RE.search(f) else 0.0)
    
    # =============================
    # Text-based Features
    # =============================
    if "text" in out.columns:
        # Ensure text is string
        text = out["text"].fillna("").astype(str)
        
        # =============================
        # Character-level Counts (partially vectorized)
        # =============================
        if "char_count" not in out.columns:
            out["char_count"] = text.str.len()
        
        # These require character-by-character checks, but still faster with apply
        if "alpha_count" not in out.columns:
            out["alpha_count"] = text.apply(lambda s: sum(ch.isalpha() for ch in s))
        if "digit_count" not in out.columns:
            out["digit_count"] = text.apply(lambda s: sum(ch.isdigit() for ch in s))
        if "uppercase_count" not in out.columns:
            out["uppercase_count"] = text.apply(lambda s: sum(ch.isupper() for ch in s))
        
        # =============================
        # Word-level Counts
        # =============================
        # A word is a whitespace-separated substring
        
        def count_words(s: str) -> int:
            """Count whitespace-separated words."""
            return len(s.split())
        
        def count_alpha_words(s: str) -> int:
            """Count words containing at least one alphabetic character."""
            words = s.split()
            return sum(1 for w in words if any(ch.isalpha() for ch in w))
        
        def count_capitalized_words(s: str) -> int:
            """Count words starting with an uppercase letter."""
            words = s.split()
            return sum(1 for w in words if w and w[0].isupper())
        
        if "word_count" not in out.columns:
            out["word_count"] = text.apply(count_words)
        if "alpha_word_count" not in out.columns:
            out["alpha_word_count"] = text.apply(count_alpha_words)
        if "capitalized_word_count" not in out.columns:
            out["capitalized_word_count"] = text.apply(count_capitalized_words)
    
    # =============================
    # Link Features
    # =============================
    if "link_url" in out.columns:
        link_url = out["link_url"].fillna("").astype(str)
        
        # has_link: True if link_url has a non-empty value
        if "has_link" not in out.columns:
            out["has_link"] = link_url.apply(lambda url: bool(url and url.strip()))
        
        # link_type: "internal" if starts with "#", "external" otherwise
        # Only set when link_url is non-empty
        if "link_type" not in out.columns:
            def normalize_link_type(url: str) -> str | None:
                """Map link_url to link_type: internal (#) or external."""
                if not url or not url.strip():
                    return None  # No link type when URL is empty
                if url.strip().startswith("#"):
                    return "internal"
                return "external"
            
            out["link_type"] = link_url.apply(normalize_link_type)
    
    return out
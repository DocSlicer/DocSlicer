# ocr/step_04_text_cleaner.py
from __future__ import annotations

import re
import unicodedata
from typing import Set

import pandas as pd


# ==================================================================================================
# BULLET TOKENS
# Raw OCR tokens that, when the entire word matches and is_line_start=1, are bullet misreads.
# Matched against text_raw (pre-normalization) so original OCR artifacts are caught.
# ==================================================================================================

_BULLET_TOKENS: Set[str] = {
    "{",    # OCR of filled bullet
    "=",    # OCR of bullet or dash leader
    "=m",   # compound OCR artifact
    "m=",
    "m",    # OCR of bullet (filled circle)
    "«",    # left guillemet misread as bullet
    "+",    # plus sign
    "—",    # em dash (single — at line start; long sequences are caught as table noise)
    "*",    # asterisk
    "o",    # lowercase o (open bullet)
    "e",    # lowercase e (OCR of open bullet)
            # removed "a" -> "a request for derogation is not left in doubt as to what"
    "s",    # lowercase s (OCR of open bullet) 
    "]",    # right bracket
    "=\u201c",
    "=\"",
    "¢",
    "-—",
    "=»",
    "«=",
    "«*",
    '*"',


}

_BULLET = "•"


# ==================================================================================================
# WORD-LEVEL CLEANING FUNCTIONS
# ==================================================================================================

def _strip_control_chars(text: str) -> str:
    """Strip non-printable control characters. Leaves normal whitespace intact."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]', '', text)


def _normalize_unicode(text: str) -> str:
    """
    NFC normalization + targeted substitutions:
      - Ligatures (ﬁ→fi, ﬀ→ff, etc.)
      - Non-breaking / zero-width spaces
      - Smart quotes → straight ASCII quotes
      - Various Unicode dashes → ASCII hyphen or em/en dash
    """
    text = unicodedata.normalize('NFC', text)

    # Invisible / spacing noise
    text = text.replace('\xa0', ' ')    # non-breaking space
    text = text.replace('\u200b', '')   # zero-width space
    text = text.replace('\u200c', '')   # zero-width non-joiner
    text = text.replace('\u200d', '')   # zero-width joiner

    # Ligatures
    for lig, rep in (
        ('\ufb00', 'ff'), ('\ufb01', 'fi'), ('\ufb02', 'fl'),
        ('\ufb03', 'ffi'), ('\ufb04', 'ffl'), ('\ufb05', 'st'), ('\ufb06', 'st'),
    ):
        text = text.replace(lig, rep)

    # Smart / curly quotes → straight
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # ' '
    text = text.replace('\u201c', '"').replace('\u201d', '"')   # " "
    text = text.replace('\u201a', ',').replace('\u201e', '"')   # ‚ „
    text = text.replace('\u2039', '<').replace('\u203a', '>')   # ‹ ›
    text = text.replace('\u00ab', '"').replace('\u00bb', '"')   # « »

    # Dashes: keep em dash (—) and en dash (–) as distinct; collapse others to hyphen
    text = text.replace('\u2212', '-')   # minus sign
    text = text.replace('\u2010', '-')   # hyphen
    text = text.replace('\u2011', '-')   # non-breaking hyphen
    text = text.replace('\u2012', '-')   # figure dash
    text = text.replace('\u2015', '—')   # horizontal bar → em dash

    return text


def _remove_table_rule_noise(text: str) -> str:
    """
    Remove tokens that are horizontal table rules misread as text.
    Pattern: 3+ characters made up entirely of —, –, -, _, =, or whitespace.
    Returns '' so the word row stays in the dataframe but contributes no text.
    """
    stripped = text.strip()
    if len(stripped) >= 3 and re.fullmatch(r'[—–\-_=\s]+', stripped):
        return ''
    return text


def _normalize_superscript_parens(text: str, line_open_excess: int = 0) -> str:
    """
    Strip unbalanced closing parentheses from superscript / footnote misreads.

    Tesseract often produces artifacts like:
        costs®)   ")   ?)   @)   (")   (')
    where a superscript number or symbol was rendered next to a closing paren.

    Two guards prevent over-stripping:
      1. line_open_excess: number of unmatched '(' from earlier words on the same line.
         Callers compute this so that a closing paren belonging to an opening paren
         elsewhere on the line is not counted as excess.
      2. Only strips ')' that is immediately preceded by a non-alphanumeric character.
         This protects real parentheses like "obligatory)" or "2014)." even when
         line_open_excess cannot fully account for them.
    """
    excess = text.count(')') - text.count('(') - line_open_excess
    if excess <= 0:
        return text

    result = text
    stripped = 0
    while stripped < excess:
        idx = result.rfind(')')
        if idx == -1:
            break
        # Leave ')' alone when it directly follows an alphanumeric char —
        # that is always a real closing paren, never a superscript artifact.
        if idx > 0 and result[idx - 1].isalnum():
            break
        # Walk left over non-alphanumeric, non-opening-paren chars (the junk before the paren)
        start = idx
        while start > 0 and not result[start - 1].isalnum() and result[start - 1] != '(':
            start -= 1
        result = result[:start] + result[idx + 1:]
        stripped += 1

    return result


def _dedupe_punctuation(text: str) -> str:
    """
    Collapse repeated punctuation that is clearly OCR noise.
      - 4+ dots → ... (ellipsis)
      - 2+ commas → ,
      - 2+ question marks → ?
      - 2+ exclamation marks → !
    Double dashes and em dashes are left intact (may be intentional).
    """
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r' {2,}', ' ', text).strip()


# ==================================================================================================
# PUBLIC API
# ==================================================================================================

def clean_words_df(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Word-level OCR text cleaning. Conservative by design — goal is to remove
    noise tokens that would pollute a RAG index, not to "correct" OCR.

    Adds 'text_raw' (original OCR output) for debugging.
    Modifies 'text' in place (on a copy).

    Rules applied (in order):
      1. Control character stripping
      2. Unicode normalization (ligatures, smart quotes, dashes, spaces)
      3. Table rule noise removal  (——————, —___— etc. → '')
      4. Superscript paren normalization  (costs®), ?), ") etc.)
      5. Punctuation deduplication
      6. Whitespace collapse
      7. Line-start bullet normalization (requires is_line_start column)
         Matched against text_raw so pre-normalization OCR tokens are caught.
    """
    if words_df is None or words_df.empty:
        return words_df.copy() if words_df is not None else pd.DataFrame()

    out = words_df.copy()

    # Preserve original for debugging / audit
    out['text_raw'] = out['text']

    # Pre-compute per-line open-paren surplus so _normalize_superscript_parens can
    # account for '(' that appear in earlier words on the same line.
    # line_open_excess[i] = max(0, cumulative open-parens seen before word i on its line).
    line_open_excess: list[int] = [0] * len(out)
    if 'line_id' in out.columns:
        texts = out['text'].astype(str).tolist()
        line_ids = out['line_id'].tolist()
        running: dict[object, int] = {}
        for i, (lid, t) in enumerate(zip(line_ids, texts)):
            surplus = running.get(lid, 0)
            line_open_excess[i] = max(0, surplus)
            running[lid] = surplus + t.count('(') - t.count(')')

    cleaned = []
    for t, open_excess in zip(out['text'].astype(str).tolist(), line_open_excess):
        t = _strip_control_chars(t)
        t = _normalize_unicode(t)
        t = _remove_table_rule_noise(t)
        t = _normalize_superscript_parens(t, line_open_excess=open_excess)
        t = _dedupe_punctuation(t)
        t = _collapse_whitespace(t)
        cleaned.append(t)

    out['text'] = cleaned

    # Bullet normalization: check raw text (pre-normalization) so OCR artifacts
    # like « or — are still in their original form when matched.
    if 'is_line_start' in out.columns:
        bullet_mask = (
            (out['is_line_start'] == 1) &
            (out['text_raw'].str.strip().isin(_BULLET_TOKENS))
        )
        out.loc[bullet_mask, 'text'] = _BULLET

    return out

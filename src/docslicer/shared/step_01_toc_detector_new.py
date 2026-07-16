"""
step_01_toc_detector_new.py

Answers the question: which lines form a Table of Contents (or Index), and
which line is its heading?

New paradigm (replaces the object-juggling detector in step_01_toc_detector.py):
each step is a vectorized pass that ADDS one or more KPI columns to the incoming
lines_df. Nothing is carried in bespoke per-row dataclasses; the DataFrame is the
single source of truth, and a final scorer reads the accumulated KPI columns to
decide what is TOC vs. not. All tunables live in the frozen TocConfig below.

Planned pipeline (built incrementally):
  1. Mask hidden block types                     -> _hidden_block_mask (reused)
  2. Heading candidates    (KPI col)             -> toc_heading_candidate
  3. Row candidates        (KPI col)             -> toc_row_candidate, ...
  4. Layout / run KPIs     (KPI cols)            -> ...
  5. Scorer                                       -> block_type = 'toc' / 'toc_heading'

This module currently implements steps 1-2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .._utils.io.yaml_compilers.page_label_patterns import PageLabelPatternConfig


# ================================================================================
# CONFIG
# ================================================================================


@dataclass(frozen=True)
class TocConfig:
    # ---- step 2: bookmark marking ----
    min_bookmarks: int = 3          # min hyperlinked "table of contents" lines doc-wide
                                    # before any is marked as a navigation bookmark; a lone
                                    # one is usually a bookmark anchor on the real heading
    # ---- step 3: heading candidates ----
    max_heading_chars: int = 80     # a heading line is short; longer lines that merely start
                                    # with "Contents" are prose, not a TOC heading
    # ---- step 5: row candidates ----
    max_row_chars: int = 250        # dot-leaders excluded; longer => prose paragraph, not a TOC row
    max_row_cells: int = 5          # more cells than this => a table row, not a TOC entry
    max_label_tokens: int = 2       # a TOC row carries 1 page number (2 if the title itself has a
                                    # number); more label-like tokens => a numeric table row


# Block types that can never be part of a TOC and are excluded from every KPI
# pass. These are all set by the format-specific pipelines (pdf/docx/pptx/html)
# BEFORE this shared step runs; TOC detection is the first shared step, so
# 'toc'/'toc_heading'/'heading' are not present yet but are listed for safety in
# case this module is ever re-run. Note: 'table' is deliberately NOT hidden — a
# TOC is very often laid out as a table.
_HIDDEN_BLOCK_TYPES = frozenset({
    # Page furniture
    "image", "hr", "page_label", "navigation", "vertical_text",
    # Out-of-flow / peripheral regions
    "header", "footer", "footnote", "endnote", "comment", "speaker_notes",
    # Non-prose content
    "chart", "shape", "math",
    # Already-classified (defensive; not set at this stage)
    "toc", "toc_heading", "heading",
})

# TOC heading anchors — CASE-SENSITIVE on purpose.
#
# The anchor word must be Title-case ("Contents", "Index") or ALL-CAPS
# ("CONTENTS", "INDEX"). A lowercase mid-sentence "contents, ..." or "... index
# of ..." is prose, not a heading, and must NOT match. Requiring the leading
# letter to be uppercase (via explicit case alternation) is what enforces this.
_TOC_HEADING_PATTERNS = (
    # "Table of Contents" / "TABLE OF CONTENTS" (classic)
    re.compile(r"^\s*(?:Table\s+of\s+Contents?|TABLE\s+OF\s+CONTENTS?)\b"),
    # bare "Contents" / "CONTENTS"
    re.compile(r"^\s*(?:Contents?|CONTENTS?)\b"),
    # "Index" / "INDEX" as the leading word ("Index", "Index of Terms")
    re.compile(r"^\s*(?:Index|INDEX)\b"),
    # "Index" / "INDEX" as the trailing word ("Alphabetical Index", "Subject INDEX")
    re.compile(r"\b(?:Index|INDEX)\s*$"),
)

# Dot leaders — the run of "......" / "· · ·" / "………" between a TOC title and its
# page number. A leader glyph followed by >=2 more (whitespace allowed between),
# so a lone period or ellipsis at a sentence end does not match. Unicode glyphs
# match correctly under pandas' str engine (verified).
_DOT_LEADER_CHARS = r".…⋯∙·•‧"
_DOT_LEADERS_RE = re.compile(
    rf"[{re.escape(_DOT_LEADER_CHARS)}](?:\s*[{re.escape(_DOT_LEADER_CHARS)}]){{2,}}"
)

# Currency anywhere in the line disqualifies a TOC row (financial table values).
_CURRENCY_RE = re.compile(r"[$€£¥]")

# Punctuation stripped from a token before page-label classification, so
# "(12)", "12.", "[iv]" classify as their bare token.
_TOKEN_STRIP = "([{.,;:)]}\"'“”‘’"


# ================================================================================
# STEP 1: Mask hidden block types
# ================================================================================

def _hidden_block_mask(df: pd.DataFrame) -> pd.Series:
    """
    Boolean row mask: True where the line's block_type is a hidden furniture type
    that can never be part of a TOC (see _HIDDEN_BLOCK_TYPES).

    Derived on demand rather than persisted — every later KPI pass calls this to
    exclude furniture before writing its column. Returns all-False when there is
    no block_type column.
    """
    if "block_type" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    bt = df["block_type"].astype("string").str.strip().str.lower()
    return bt.isin(_HIDDEN_BLOCK_TYPES).fillna(False).astype(bool)


# ================================================================================
# STEP 2: Mark TOC bookmarks  (KPI col: toc_bookmark)
# ================================================================================

def add_toc_bookmarks(df: pd.DataFrame, config: TocConfig = TocConfig()) -> pd.DataFrame:
    """
    Add the KPI column ``toc_bookmark`` (bool) and hide the marked rows.

    A bookmark is a hyperlinked "table of contents" line — a back-to-TOC
    navigation pointer, not the real TOC. These are marked True and their
    block_type is set to 'navigation' so the hidden-block mask excludes them from
    every later KPI pass (comparable to the old detector's _remove_toc_pointers).

    Guard: nothing is marked unless at least config.min_bookmarks such lines exist —
    a single linked "Table of Contents" is usually just a bookmark anchor on the
    real heading, not a pointer.
    """
    out = df.copy()
    out["toc_bookmark"] = False

    if "text" not in out.columns or "has_link" not in out.columns:
        return out

    text_norm = out["text"].astype("string").str.strip().str.lower()
    has_link = out["has_link"].fillna(False).astype(bool)

    is_bookmark = (text_norm == "table of contents") & has_link
    if int(is_bookmark.sum()) < config.min_bookmarks:
        return out

    out["toc_bookmark"] = is_bookmark.astype(bool)

    if "block_type" not in out.columns:
        out["block_type"] = pd.NA
    out.loc[is_bookmark, "block_type"] = "navigation"
    return out


# ================================================================================
# STEP 3: Heading candidates  (KPI col: toc_heading_candidate)
# ================================================================================

def add_toc_heading_candidates(df: pd.DataFrame, config: TocConfig = TocConfig()) -> pd.DataFrame:
    """
    Add the KPI column ``toc_heading_candidate`` (bool).

    True where the line looks like a TOC/Index heading:
      - text matches a case-sensitive anchor (Title-case or ALL-CAPS "Contents",
        "Table of Contents", or leading/trailing "Index"), AND
      - the line is short (<= config.max_heading_chars), AND
      - the line is not a hidden block type.

    Everything else is False.
    """
    out = df.copy()
    out["toc_heading_candidate"] = False

    if "text" not in out.columns:
        return out

    text = out["text"].astype("string").fillna("")

    match = pd.Series(False, index=out.index, dtype=bool)
    for pattern in _TOC_HEADING_PATTERNS:
        match |= text.str.contains(pattern, regex=True, na=False)

    match &= text.str.len() <= config.max_heading_chars
    match &= ~_hidden_block_mask(out)

    out["toc_heading_candidate"] = match.astype(bool)
    return out


# ================================================================================
# STEP 4: Dot leaders  (KPI col: toc_has_dot_leaders)
# ================================================================================

def add_toc_dot_leaders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the KPI column ``toc_has_dot_leaders`` (bool): True where the line
    contains a dot-leader run ("......", "· · ·", "………") between a title and a
    page number — a strong TOC-row signal.

    A single vectorized regex pass; ~6 ms on 100K lines, so it runs over every
    row (raw text feature, not gated on candidacy or block_type).
    """
    out = df.copy()
    out["toc_has_dot_leaders"] = False

    if "text" not in out.columns:
        return out

    text = out["text"].astype("string").fillna("")
    out["toc_has_dot_leaders"] = text.str.contains(_DOT_LEADERS_RE, regex=True, na=False).astype(bool)
    return out


# ================================================================================
# STEP 5: Row candidates  (KPI cols: toc_row_candidate, toc_row_page_token/type)
# ================================================================================

def _classify_token(token: str, cfg: PageLabelPatternConfig, cache: Dict[str, str]) -> str:
    """Page-label type of a single token ('arabic'/'roman'/... or 'unknown'), memoized."""
    hit = cache.get(token)
    if hit is not None:
        return hit
    t = token.strip().strip(_TOKEN_STRIP)
    res = "unknown"
    if t and len(t) <= cfg.max_length:
        for pat in cfg.patterns:
            if pat.compiled.match(t):
                res = pat.name
                break
    cache[token] = res
    return res


def _has_cased_letter(token: str) -> bool:
    """True if the token contains a cased letter (unicode-safe: lower != upper)."""
    return token.lower() != token.upper()


def _title_has_word(toks: List[str], types: List[str]) -> bool:
    """The title side must carry a real word — a cased-letter token that is not a page label."""
    return any(typ == "unknown" and _has_cased_letter(tok) for tok, typ in zip(toks, types))


def _eval_toc_row(
    toks: List[str],
    cfg: PageLabelPatternConfig,
    cache: Dict[str, str],
    max_label_tokens: int,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Decide whether a tokenized line is a TOC row and, if so, which boundary token
    is its page label. Accepts BOTH layouts:
        title ... page_label      (label at end)
        page_label ... title      (label at start)

    Returns (is_candidate, page_token, page_type).
    """
    n = len(toks)
    if n < 2:
        return (False, None, None)

    types = [_classify_token(t, cfg, cache) for t in toks]
    n_labels = sum(1 for x in types if x != "unknown")
    # Exactly one page number (a title may carry one more); many => numeric table row.
    if n_labels == 0 or n_labels > max_label_tokens:
        return (False, None, None)

    # Label at end: the token before it must NOT also be a label (table columns),
    # and the title side must contain a real word.
    if types[-1] != "unknown" and types[-2] == "unknown" and _title_has_word(toks[:-1], types[:-1]):
        return (True, toks[-1].strip().strip(_TOKEN_STRIP), types[-1])

    # Label at start (the picture's "02  At a glance" layout).
    if types[0] != "unknown" and types[1] == "unknown" and _title_has_word(toks[1:], types[1:]):
        return (True, toks[0].strip().strip(_TOKEN_STRIP), types[0])

    return (False, None, None)


def add_toc_row_candidates(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    config: TocConfig = TocConfig(),
) -> pd.DataFrame:
    """
    Add the KPI columns:
      - ``toc_row_candidate`` (bool)
      - ``toc_row_page_token`` (str | NA)   the page-label token that anchored the row
      - ``toc_row_page_type``  (str | NA)   its page-label type ('arabic', 'roman', ...)

    A line is a TOC row candidate when it starts OR ends with a clean page-label
    token and carries real title text (with or without dot leaders). Rejected:
      - lines that neither start nor end with a page-label token ("US investment plans")
      - currency-bearing lines ("Total net revenue $ 182,447 ...")
      - lines with more than config.max_row_cells cells (table rows)
      - lines with more than config.max_label_tokens label-like tokens — numeric
        tables ("Collaboration Revenue | 93 | (14) | (15) | 11 | (81) | (82)")
      - hidden block types

    The heavy per-row token logic only runs on rows that clear a cheap vectorized
    prefilter (a page-label token at either boundary, under the length/cell/currency
    gates), so most prose lines never reach it.
    """
    out = df.copy()
    out["toc_row_candidate"] = False
    out["toc_row_page_token"] = pd.NA
    out["toc_row_page_type"] = pd.NA

    if "text" not in out.columns:
        return out

    text = out["text"].astype("string").fillna("")

    # Strip dot-leader runs so the page number is a clean trailing/leading token,
    # then collapse whitespace.
    stripped = (
        text.str.replace(_DOT_LEADERS_RE, " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )
    toks_series = stripped.str.split()
    n_tok = toks_series.str.len().fillna(0)

    # ---- cheap vectorized prefilter ----
    first_tok = toks_series.str[0]
    last_tok = toks_series.str[-1]

    cache: Dict[str, str] = {}
    uniq = pd.unique(pd.concat([first_tok, last_tok], ignore_index=True).dropna())
    type_map = {tok: _classify_token(tok, page_label_config, cache) for tok in uniq}

    is_label_first = first_tok.map(type_map).fillna("unknown").ne("unknown")
    is_label_last = last_tok.map(type_map).fillna("unknown").ne("unknown")

    prefilter = (
        (n_tok >= 2)
        & (is_label_first | is_label_last)
        & (stripped.str.len() <= config.max_row_chars)
        & ~text.str.contains(_CURRENCY_RE, regex=True, na=False)
        & ~_hidden_block_mask(out)
    )
    if "cell_count" in out.columns:
        prefilter &= out["cell_count"].fillna(0) <= config.max_row_cells

    # ---- detailed per-row check on the reduced set ----
    toks_list = toks_series.tolist()
    positions = list(out.index[prefilter])
    pos_to_iloc = {idx: i for i, idx in enumerate(out.index)}

    cand_idx: List = []
    tokens_out: List[str] = []
    types_out: List[str] = []
    for idx in positions:
        toks = toks_list[pos_to_iloc[idx]]
        ok, tok, typ = _eval_toc_row(toks, page_label_config, cache, config.max_label_tokens)
        if ok:
            cand_idx.append(idx)
            tokens_out.append(tok)
            types_out.append(typ)

    if cand_idx:
        out.loc[cand_idx, "toc_row_candidate"] = True
        out.loc[cand_idx, "toc_row_page_token"] = tokens_out
        out.loc[cand_idx, "toc_row_page_type"] = types_out

    return out


# ================================================================================
# PUBLIC API
# ================================================================================

def detect_tocs(
    df: pd.DataFrame,
    page_label_config: PageLabelPatternConfig,
    config: TocConfig = TocConfig(),
) -> pd.DataFrame:
    """
    Run the TOC detection pipeline and return the annotated DataFrame.

    Each step adds its KPI column(s) to a copy of the input; the frame is threaded
    through the steps and returned to the caller unchanged apart from the added
    columns. Empty input is returned untouched.

    Currently implements:
      2. toc_bookmark          (also hides marked rows via block_type='navigation')
      3. toc_heading_candidate
      4. toc_has_dot_leaders
      5. toc_row_candidate      (+ toc_row_page_token / toc_row_page_type)

    (Step 1, the hidden-block mask, is applied inside each KPI pass rather than
    persisted as a column.)
    """
    if df.empty:
        return df

    out = add_toc_bookmarks(df, config)
    out = add_toc_heading_candidates(out, config)
    out = add_toc_dot_leaders(out)
    out = add_toc_row_candidates(out, page_label_config, config)
    return out

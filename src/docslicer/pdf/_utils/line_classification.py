"""
Per-line gap-distribution stats and text/table classification.

Takes an arbitrary set of words on one visible line and classifies them as
"text", "table", or "undetermined" -- using only that line's own geometry
and content. No neighboring rows, lookahead, or gridlines are consulted;
callers may feed it a full line or an arbitrary word subset.

Shared by step_10_cell_builder (per-line cell-split decision, full lines)
and step_07_word_relationships (same analysis over the word set above a
candidate horizontal rule).
See step_10_cell_builder's module docstring for the full per-line decision
pipeline this feeds into.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineClassificationConfig:
    # --- em thresholds (gap_em <= T -> merge)---
    em_text:         float = 0.90   # justified prose: permissive, absorbs stretched spaces
    em_undetermined: float = 0.60   # neutral default
    em_table:        float = 0.40   # tight: avoid bridging columns

    # --- Within-line gap distribution ---
    # When consecutive sorted gaps jump by >= this factor, the distribution is
    # considered bimodal (two gap clusters: word-spaces and wide inter-cell gaps).
    ratio_split:        float = 1.8
    # The wider gap in that jump must exceed this (em) to rule out spurious jumps
    # between two slightly different word-spaces.
    min_space_em:       float = 0.30
    # Below this many gaps a line is too short for a stable distribution.
    min_gaps_for_ratio: int   = 2


CONFIG = LineClassificationConfig()

# Words that essentially only occur in running prose, almost never as a
# standalone cell/header token in a table. Matched lowercased (see
# content_stats), so anything that doubles as a table token when cased
# differently stays out: "may" (May 31 date columns), "no" ("No." item
# columns), "a"/"i" (row labels, roman numerals), "per" ("per share" headers).
_STOPWORDS = {
    # articles / determiners / pronouns
    "the", "an", "that", "which", "who", "whose", "its", "this", "these",
    "those", "their", "they", "we", "our", "such", "each", "any", "all",
    "other",
    # conjunctions / subordinators
    "and", "or", "but", "if", "than", "then", "because", "while", "when",
    "where", "whether", "unless", "although",
    # prepositions
    "of", "to", "in", "for", "with", "as", "on", "by", "from", "at", "into",
    "among", "about", "after", "before", "between", "against", "during",
    "since", "under", "over", "through", "without", "within", "upon",
    "including",
    # verbs / auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having", "will", "would", "shall", "should",
    "could", "must", "do", "does", "did", "not",
}
_STRIP_CHARS = ".,;:()[]{}\"'"


# ================================================================================
# 1. GAP DISTRIBUTION  (primary signal, per line)
# ================================================================================

def line_gap_stats(
    x_left:    np.ndarray,
    x_right:   np.ndarray,
    font_size: np.ndarray,
    config: LineClassificationConfig = CONFIG,
) -> dict:
    """
    Em-normalized inter-word gap statistics for ONE line.

    Words must already be sorted left->right. Each gap is divided by the larger
    of the two flanking words' font sizes, so the result is scale-invariant.

    Returns a dict:
        gaps_em    : np.ndarray, len = n_words - 1. Signed: negative for
                     overlapping words; np.nan where a flanking font size is
                     invalid. Only positive finite gaps feed the stats below.
        median_em  : float  -- the line's typical space width, in em
        max_em     : float
        jump_ratio : float  -- ratio between consecutive *sorted* gaps:
                              at the detected valley when is_bimodal, else the
                              largest jump found (~1.0 means uniform spacing)
        split_em   : float  -- gaps strictly above this are the wide cluster
                              (gutters). Only meaningful when is_bimodal.
        n_gaps     : int    -- positive finite gaps feeding the stats (falls
                              back to the non-NaN count when none are positive,
                              i.e. an all-overlap line)
        is_bimodal : bool   -- a clear valley separates spaces from gutters

    The valley is the LOWEST jump that separates word-spaces from wider gaps,
    not the largest jump overall. A line can have several gutter tiers (e.g.
    a 0.5em column gutter AND a 4em section break on the same row); there is
    only ever one space tier. We isolate that space tier, so every gap above
    split_em -- of any tier -- becomes a cell boundary. Picking the largest
    jump instead would split only the widest tier and wrongly merge the rest.

    Detection uses jumps between consecutive *sorted* gaps, not max/median:
    median lands on a gutter when gutters outnumber spaces and the ratio
    collapses to ~1.
    """
    n = len(x_left)
    empty = {
        "gaps_em": np.empty(0), "median_em": 0.0, "max_em": 0.0,
        "jump_ratio": 1.0, "split_em": np.inf, "n_gaps": 0, "is_bimodal": False,
    }
    if n < 2:
        return empty

    gaps_pt = x_left[1:] - x_right[:-1]            # signed; negative = overlap
    fs = np.maximum(font_size[:-1], font_size[1:]).astype(float)  # larger of left/right word
    fs = np.where((fs > 0) & np.isfinite(fs), fs, np.nan)
    gaps_em = gaps_pt / fs

    # Positive, finite gaps only feed the distribution stats.
    valid = np.sort(gaps_em[(gaps_em > 0) & np.isfinite(gaps_em)])
    if valid.size == 0:
        return {**empty, "gaps_em": gaps_em, "n_gaps": int((~np.isnan(gaps_em)).sum())}

    median_em = float(np.median(valid))
    max_em    = float(valid[-1])

    jump_ratio = 1.0
    split_em   = np.inf
    is_bimodal = False
    if valid.size >= config.min_gaps_for_ratio:
        ratios = valid[1:] / (valid[:-1] + 1e-9)     # consecutive sorted ratios
        # Lowest valley: first jump big enough whose upper side is gutter-sized.
        # min_space_em rejects spurious jumps within the space cluster (e.g. a
        # slightly wider space after a period) from being treated as a gutter.
        cand = np.where(
            (ratios >= config.ratio_split) & (valid[1:] > config.min_space_em)
        )[0]
        if cand.size:
            k = int(cand[0])
            jump_ratio = float(ratios[k])
            split_em   = float(np.sqrt(valid[k] * valid[k + 1]))  # geometric midpoint
            is_bimodal = True
        else:
            jump_ratio = float(ratios.max())

    return {
        "gaps_em": gaps_em, "median_em": median_em, "max_em": max_em,
        "jump_ratio": jump_ratio, "split_em": split_em,
        "n_gaps": int(valid.size), "is_bimodal": is_bimodal,
    }


# ================================================================================
# 2. CLASSIFICATION  (prior; resolves unimodal lines only)
# ================================================================================

def content_stats(texts: list[str]) -> dict:
    """Cheap per-line content features computed straight from token text."""
    n = len(texts)
    if n == 0:
        return {"n": 0, "alpha_ratio": 0.0, "numeric_token_ratio": 0.0,
                "stopword_hits": 0, "cap_ratio": 0.0, "has_punct": False}

    alpha = numeric = caps = stop = 0
    has_punct = False
    for t in texts:
        s = str(t).strip()
        if not s:
            continue
        norm = s.lower().strip(_STRIP_CHARS)
        if norm in _STOPWORDS:
            stop += 1
        has_alpha = any(c.isalpha() for c in s)
        if has_alpha:
            alpha += 1
            if s[:1].isupper():   # first letter only: title-case and ALL CAPS both count
                caps += 1
            # Punctuation only counts on alpha tokens, so numeric formatting
            # ("1,000", "(2.5)") never reads as prose punctuation.
            if any(c in ".,;:" for c in s):
                has_punct = True
        # numeric-like token: nothing but digits and numeric punctuation. No
        # digit is required, so bare "%", "-", "()" tokens also count.
        if s and all(c.isdigit() or c in ",.()-+%—– " for c in s):
            numeric += 1

    return {
        "n": n,
        "alpha_ratio": alpha / n,
        "numeric_token_ratio": numeric / n,
        "stopword_hits": stop,
        "cap_ratio": caps / max(alpha, 1),
        "has_punct": has_punct,
    }


def classify_line(
    texts: list[str],
    gap_stats: dict,
) -> tuple[str, float]:
    """
    Classify ONE line, returning ``(label, score)`` with label one of
    "text", "table", or "undetermined".

    ``gap_stats`` is the dict produced by line_gap_stats() for the same words —
    passed in (rather than recomputed here) because every caller also consumes
    the raw stats directly for its own feature columns.

    Lines with <= 2 words (0-1 gaps) are always "undetermined" — too little
    geometry to say anything meaningful.

    For 3+ words: compute a signed score.
        negative  =>  text
        positive  =>  table
    Signals are grouped into sentence evidence (drives score negative) and
    table evidence (drives score positive).
    """
    n = len(texts)
    if n <= 2:
        return "undetermined", 0.0

    c     = content_stats(texts)
    score = 0.0

    # ── Sentence signals (push negative) ──────────────────────────────────
    # Word count: longer lines are more likely prose
    if   n >= 11: score -= 2.0
    elif n >= 8: score -= 1.0

    # Stopwords are a very strong prose indicator
    if   c["stopword_hits"] >= 2: score -= 2.0
    elif c["stopword_hits"] == 1: score -= 1.0

    # Punctuation mid-line (comma, colon, etc.) is a prose indicator
    if c["has_punct"]: score -= 1.0

    # High alpha ratio means mostly real words, not numbers/codes
    if   c["alpha_ratio"] >= 0.75: score -= 2.0
    elif c["alpha_ratio"] >= 0.60: score -= 1.0

    # Mostly lowercase-initial tokens across several alpha words: prose-like
    if n >= 4 and c["alpha_ratio"] > 0 and c["cap_ratio"] <= 0.5:
        score -= 1.0

    # ── Table signals (push positive) ─────────────────────────────────────
    # Numeric-heavy content
    if   c["numeric_token_ratio"] >= 0.35: score += 2.0
    elif c["numeric_token_ratio"] >= 0.20: score += 1.0

    # Nearly every token capitalised (title-case or ALL CAPS — cap_ratio only
    # checks the first letter) with no stopwords: header row or label column
    if c["cap_ratio"] >= 0.8 and c["stopword_hits"] == 0:
        score += 1.0

    # Median em: wide typical spacing is a table signal
    med = gap_stats.get("median_em", 0.0)
    if   med >= 2.0: score += 3.0
    elif med >= 1.5: score += 2.0
    elif med >= 1.0: score += 1.0

    # Gap geometry: jump_ratio measures the bimodal valley strength.
    # is_bimodal=True means a clear word-space / column-gutter split exists.
    jr = gap_stats.get("jump_ratio", 1.0)
    if gap_stats.get("is_bimodal"):
        if   jr >= 5.0: score += 3.0
        elif jr >= 3.0: score += 2.0
        else:           score += 1.0   # bimodal but shallow valley
    else:
        if   jr < 1.2: score -= 2.0   # near-flat: very uniform spacing
        elif jr < 1.5: score -= 1.0   # mildly varied but unimodal

    # ── Decision ──────────────────────────────────────────────────────────
    if score >=  2.0: return "table",        score
    if score <= -2.0: return "text",         score
    return              "undetermined",      score


# ================================================================================
# 3. FALLBACK EM THRESHOLD  (unimodal lines)
# ================================================================================

def em_threshold_for_class(cls: str, config: LineClassificationConfig = CONFIG) -> float:
    """Allowed merge gap, in em, for a uniformly-spaced line of the given class."""
    return {
        "text":  config.em_text,
        "table": config.em_table,
    }.get(cls, config.em_undetermined)

"""
step_08_cell_builder.py  (rewrite)

Words -> Cells, decided strictly per line (no cross-row lookahead).

Per-line decision pipeline
--------------------------
For each line, in order:

  1. Special-case overrides   (bullet -> text, "$" -> number)        [TODO next]
  2. Gap distribution         -> _line_gap_stats()
       The line's inter-word gaps, em-normalized (gap / font_size).
       If the gaps are *bimodal* (a clear valley), the wide cluster are
       column gutters: split there. This is the primary signal and it is
       fully per-line. It is what separates:
         - justified prose : all gaps ~equal  (ratio ~1.0)  -> one texture
         - table header     : tight intra-phrase spaces + wider gutters
                              (ratio >> 1)                  -> split at valley
  3. Classification           -> classify_line()
       "text" | "table" | "undetermined". A *prior*, only consulted to
       resolve UNIMODAL lines (all gaps similar), where the distribution
       alone can't say merge-all vs split-all.
  4. Fallback em threshold    -> em_threshold_for_class()
       Scale-invariant constant (gap_em <= T -> merge). No font interpolation:
       dividing the gap by font_size already removes the size dependence.

Known per-line limitation
-------------------------
A line of single-word columns with uniform wide gaps is indistinguishable
from justified prose without margins or neighbour rows. We classify it by
the content prior and accept the occasional miss. This is deliberate, not a
bug (resolving it requires lookahead, which this step forgoes by design).

Why em, not pt
--------------
A pt is absolute, but "what counts as a word-space" scales with font size.
gap_em = gap_pt / font_size makes one threshold mean the same thing on a
14pt body line (9.6pt gap = 0.69 em) and a 9.6pt table header (4.5pt gap =
0.47 em). Note these two *invert* in magnitude, so no single em cutoff
separates them -- only the within-line ratio does (0.47/0.25 ~= 1.9 vs
0.69/0.69 = 1.0). Hence step 2 before step 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._utils.text_utils import _BULLET_TOKENS


# ================================================================================
# CONFIG
# ================================================================================

@dataclass(frozen=True)
class CellBuildConfig:
    # --- em thresholds (gap_em <= T -> merge)---
    em_text:         float = 0.90   # justified prose: permissive, absorbs stretched spaces
    em_undetermined: float = 0.60   # neutral default
    em_table:        float = 0.40   # tight: avoid bridging columns

    # --- Special-case override thresholds ---
    # Max gap (em) to still merge a bullet/dollar token into the same cell.
    # Derived from old pt constants at reference font size 10:
    #   bullet: 30pt / 10 = 3.0em
    #   dollar: 60pt / 10 = 6.0em
    bullet_max_gap_em: float = 3.0
    dollar_max_gap_em: float = 6.0

    # --- Within-line gap distribution ---
    # When consecutive sorted gaps jump by >= this factor, the distribution is
    # considered bimodal (two gap clusters: word-spaces and wide inter-cell gaps).
    ratio_split:        float = 1.8
    # The wider gap in that jump must exceed this (em) to rule out spurious jumps
    # between two slightly different word-spaces.
    min_space_em:       float = 0.30
    # Below this many gaps a line is too short for a stable distribution.
    min_gaps_for_ratio: int   = 2


CONFIG = CellBuildConfig()

_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "with", "as", "on", "by", "from", "at",
    "into", "among", "including", "that", "which", "who", "whose", "its",
    "is", "are", "was", "were", "be", "been", "being",
}
_STRIP_CHARS = ".,;:()[]{}\"'"


# ================================================================================
# 1. GAP DISTRIBUTION  (primary signal, per line)
# ================================================================================

def _line_gap_stats(
    x_left:    np.ndarray,
    x_right:   np.ndarray,
    font_size: np.ndarray,
    config: CellBuildConfig = CONFIG,
) -> dict:
    """
    Em-normalized inter-word gap statistics for ONE line.

    Words must already be sorted left->right. Each gap is divided by the font
    size of its left-hand (flanking) word, so the result is scale-invariant.

    Returns a dict:
        gaps_em    : np.ndarray, len = n_words - 1   (np.nan for overlaps)
        median_em  : float  -- the line's typical space width, in em
        max_em     : float
        jump_ratio : float  -- largest ratio between consecutive *sorted* gaps
                              (~1.0 uniform, >>1 means two clusters / a valley)
        split_em   : float  -- gaps strictly above this are the wide cluster
                              (gutters). Only meaningful when is_bimodal.
        n_gaps     : int
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
    fs = font_size[:-1].astype(float)             # flanking (left) word's font
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

def _content_stats(texts: list[str]) -> dict:
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
            if s[:1].isupper():
                caps += 1
            if any(c in ".,;:" for c in s):
                has_punct = True
        # numeric-like token: digits + numeric punctuation only
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
) -> str:
    """
    Classify ONE line as "text", "table", or "undetermined".

    Lines with <= 2 words (0-1 gaps) are always "undetermined" — too little
    geometry to say anything meaningful.

    For 3+ words: compute a signed score.
        negative  =>  text
        positive  =>  table
    Signals are grouped into sentence evidence (drives score negative) and
    table evidence (drives score positive). The gap jump_ratio replaces the
    old pt-absolute gap_ratio — it is em-normalised and bimodality-aware.
    """
    n = len(texts)
    if n <= 2:
        return "undetermined", 0.0

    c     = _content_stats(texts)
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

    # Mixed-case (not all-caps) with several alpha tokens: prose-like
    if n >= 4 and c["alpha_ratio"] > 0 and c["cap_ratio"] <= 0.5:
        score -= 1.0

    # ── Table signals (push positive) ─────────────────────────────────────
    # Numeric-heavy content
    if   c["numeric_token_ratio"] >= 0.35: score += 2.0
    elif c["numeric_token_ratio"] >= 0.20: score += 1.0

    # All-caps tokens with no stopwords: header row or label column
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

def em_threshold_for_class(cls: str, config: CellBuildConfig = CONFIG) -> float:
    """Allowed merge gap, in em, for a uniformly-spaced line of the given class."""
    return {
        "text":  config.em_text,
        "table": config.em_table,
    }.get(cls, config.em_undetermined)


# ================================================================================
# 4. ENTRY POINT
# ================================================================================

def build_cells(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    df_links:  pd.DataFrame | None = None,
    config: CellBuildConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Words -> Cells.

    Currently returns df_words annotated with per-line features so the
    classification decisions can be inspected before cell splitting is wired up.

    Added columns (one value per line_id, broadcast to every word in that line):
        line_n_gaps, line_median_em, line_max_em, line_jump_ratio,
        line_is_bimodal, line_split_em, line_class, line_em_threshold
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(), pd.DataFrame()

    feat_rows: list[dict] = []
    gap_em_right_series: dict[int, float | None] = {}  # word index -> gap to next word

    for line_id, grp in df_words.groupby("line_id", sort=False):
        grp_s     = grp.sort_values("x_left")
        x_left    = grp_s["x_left"].to_numpy(float)
        x_right   = grp_s["x_right"].to_numpy(float)
        font_size = grp_s["font_size"].to_numpy(float)
        texts     = grp_s["text"].tolist()
        idx       = grp_s.index.tolist()

        # Exclude super/subscripts from content scoring — footnote markers
        # inflate word count and skew alpha/numeric ratios.
        if "script_type" in grp_s.columns:
            texts_content = grp_s.loc[grp_s["script_type"].isna(), "text"].tolist()
        else:
            texts_content = texts

        gs        = _line_gap_stats(x_left, x_right, font_size, config)
        cs        = _content_stats(texts_content)
        cls, score = classify_line(texts_content, gs)
        thr       = em_threshold_for_class(cls, config)

        feat_rows.append({
            "line_id":                line_id,
            # gap stats
            "line_n_gaps":            gs["n_gaps"],
            "line_median_em":         round(gs["median_em"], 4),
            "line_max_em":            round(gs["max_em"], 4),
            "line_jump_ratio":        round(gs["jump_ratio"], 4),
            "line_is_bimodal":        gs["is_bimodal"],
            "line_split_em":          round(gs["split_em"], 4) if np.isfinite(gs["split_em"]) else None,
            # content stats
            "line_n_words":           cs["n"],
            "line_alpha_ratio":       round(cs["alpha_ratio"], 4),
            "line_numeric_ratio":     round(cs["numeric_token_ratio"], 4),
            "line_stopword_hits":     cs["stopword_hits"],
            "line_cap_ratio":         round(cs["cap_ratio"], 4),
            "line_has_punct":         cs["has_punct"],
            # decision
            "line_score":             round(score, 2),
            "line_class":             cls,
            "line_em_threshold":      thr,
        })

        # per-word gap to the right: gaps_em[i] is between word i and word i+1
        gaps_em = gs["gaps_em"]
        for i, word_idx in enumerate(idx):
            if i < len(gaps_em) and np.isfinite(gaps_em[i]):
                gap_em_right_series[word_idx] = round(float(gaps_em[i]), 4)
            else:
                gap_em_right_series[word_idx] = None

    df_feat  = pd.DataFrame(feat_rows)
    df_words = df_words.merge(df_feat, on="line_id", how="left")
    df_words["gap_em_right"] = df_words.index.map(gap_em_right_series)

    # TODO: split words into cells using gap stats; for now return annotated words as cells stub
    return df_words.copy(), df_words

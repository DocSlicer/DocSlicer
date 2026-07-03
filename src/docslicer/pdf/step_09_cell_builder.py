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

from .._utils.text_utils import _BULLET_TOKENS, _CURRENCY_SYMBOLS, is_list_marker
from .._utils.df_aggregation.hierarchical_aggregator import aggregate_hierarchical, build_standard_agg_spec


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
    # Max gap (em) to still merge a bullet/currency token into the same cell.
    # Derived from old pt constants at reference font size 10:
    #   bullet:   30pt / 10 = 3.0em
    #   currency: 60pt / 10 = 6.0em
    bullet_max_gap_em:   float = 3.0
    currency_max_gap_em: float = 6.0

    # --- Overlap tolerance ---
    # Overlaps up to this many pt are treated as zero (superscript positioning,
    # sub-pixel kerning artefacts). Overlaps beyond this force a cell boundary.
    overlap_split_pt:   float = 2.0

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

# Script detection thresholds (mirror step_01 constants applied at word granularity)
_SCRIPT_DETECT_SIZE_RATIO = 0.82   # word font_size < this * cell ref → candidate
_SCRIPT_DETECT_Y_FACTOR   = 0.20   # baseline shift > this * ref_size → confirmed

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

    Words must already be sorted left->right. Each gap is divided by the larger
    of the two flanking words' font sizes, so the result is scale-invariant.

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
# 4. LINE FEATURE ANNOTATION  (pipeline step 1: combines sections 1-3)
# ================================================================================

def _annotate_line_features(
    df: pd.DataFrame,
    config: CellBuildConfig = CONFIG,
) -> pd.DataFrame:
    """
    Compute per-line gap statistics and classification, broadcast to every word.

    Added columns:
        gap_em_right, line_n_gaps, line_median_em, line_max_em, line_jump_ratio,
        line_is_bimodal, line_split_em, line_class, line_score, line_em_threshold
    """
    # Pre-sort once so per-group rows are in x_left order — avoids re-sorting inside the loop.
    df = df.sort_values(["line_id", "x_left"], kind="mergesort").reset_index(drop=True)

    # Vectorised gap_em_right: for each word, gap to the right neighbour within the same line.
    xl  = df["x_left"].to_numpy(float)
    xr  = df["x_right"].to_numpy(float)
    fs  = df["font_size"].to_numpy(float)
    lid = df["line_id"].to_numpy()

    n = len(df)
    same_next          = np.empty(n, dtype=bool)
    same_next[:-1]     = lid[:-1] == lid[1:]
    same_next[-1]      = False

    gap_pt             = np.empty(n);  gap_pt[-1]  = np.nan
    gap_pt[:-1]        = xl[1:] - xr[:-1]

    fs_max             = np.empty(n);  fs_max[-1]  = np.nan
    fs_max[:-1]        = np.maximum(fs[:-1], fs[1:])
    fs_max             = np.where((fs_max > 0) & np.isfinite(fs_max), fs_max, np.nan)

    raw_gem            = gap_pt / fs_max
    gap_em_col         = np.where(same_next & np.isfinite(raw_gem), np.round(raw_gem, 4), np.nan)
    df["gap_em_right"] = gap_em_col

    # Per-line stats and classification (must remain a Python loop; data is already sorted).
    feat_rows: list[dict] = []
    has_script = "script_type" in df.columns
    has_table  = "table_id" in df.columns

    for line_id, grp in df.groupby("line_id", sort=False):
        x_left    = grp["x_left"].to_numpy(float)
        x_right   = grp["x_right"].to_numpy(float)
        font_size = grp["font_size"].to_numpy(float)
        texts     = grp["text"].tolist()

        if has_script:
            untagged_mask = grp["script_type"].isna().to_numpy()
            texts_content = [t for t, m in zip(texts, untagged_mask) if m]
        else:
            texts_content = texts

        gs = _line_gap_stats(x_left, x_right, font_size, config)
        cs = _content_stats(texts_content)

        # Struct-table lines bypass the classification channel: their cells are cut
        # at TD/TH boundaries downstream, so the em threshold is never consulted.
        if has_table and grp["table_id"].notna().any():
            cls, score = "table", 0.0
            thr        = config.em_table
        else:
            cls, score = classify_line(texts_content, gs)
            thr        = em_threshold_for_class(cls, config)

        feat_rows.append({
            "line_id":            line_id,
            "line_n_gaps":        gs["n_gaps"],
            "line_median_em":     round(gs["median_em"], 4),
            "line_max_em":        round(gs["max_em"], 4),
            "line_jump_ratio":    round(gs["jump_ratio"], 4),
            "line_is_bimodal":    gs["is_bimodal"],
            "line_split_em":      round(gs["split_em"], 4) if np.isfinite(gs["split_em"]) else None,
            "line_n_words":       cs["n"],
            "line_alpha_ratio":   round(cs["alpha_ratio"], 4),
            "line_numeric_ratio": round(cs["numeric_token_ratio"], 4),
            "line_stopword_hits": cs["stopword_hits"],
            "line_cap_ratio":     round(cs["cap_ratio"], 4),
            "line_has_punct":     cs["has_punct"],
            "line_score":         round(score, 2),
            "line_class":         cls,
            "line_em_threshold":  thr,
        })

    df_feat = pd.DataFrame(feat_rows)
    df = df.merge(df_feat, on="line_id", how="left")
    return df


# ================================================================================
# 5. CELL ID ASSIGNMENT  (horizontal words)
# ================================================================================

def _is_numeric_like(text: str) -> bool:
    s = str(text).strip()
    return bool(s) and all(ch.isdigit() or ch in ",.()-+%—– " for ch in s)


def _assign_cell_ids_horiz(df: pd.DataFrame, config: CellBuildConfig = CONFIG) -> tuple[pd.DataFrame, int]:
    """
    Assign cell_id to horizontal words using the per-line em threshold and
    gap_em_right already annotated on df.

    Returns (annotated df, max_cell_id).
    """
    df = df.sort_values(["line_id", "x_left"], kind="mergesort").reset_index(drop=True)

    text_arr    = df["text"].astype(str).to_numpy()
    line_arr    = df["line_id"].to_numpy(dtype=np.int64)
    x_left_arr  = df["x_left"].to_numpy(dtype=float)
    x_right_arr = df["x_right"].to_numpy(dtype=float)
    gap_em_arr  = pd.to_numeric(df["gap_em_right"], errors="coerce").to_numpy(dtype=float)
    thr_arr     = pd.to_numeric(df["line_em_threshold"], errors="coerce").to_numpy(dtype=float)

    stripped       = pd.Series(text_arr).str.strip()
    bullet_flags   = stripped.isin(_BULLET_TOKENS).to_numpy()
    currency_flags = stripped.isin(_CURRENCY_SYMBOLS).to_numpy()
    numeric_flags  = stripped.apply(_is_numeric_like).to_numpy()

    n        = len(df)

    # List markers ((1), 1., (a), [1], iv. …) merge into the following word like
    # bullets, but only when the marker isn't a sub/superscript reference. The
    # regex is only consulted for line-leading words (the merge requires k == i),
    # so restrict the per-word apply to those positions.
    is_line_first = np.zeros(n, dtype=bool)
    if n:
        is_line_first[0]  = True
        is_line_first[1:] = line_arr[1:] != line_arr[:-1]
    list_marker_flags = np.zeros(n, dtype=bool)
    first_idx = np.where(is_line_first)[0]
    if first_idx.size:
        list_marker_flags[first_idx] = stripped.iloc[first_idx].apply(is_list_marker).to_numpy()

    # Script status is resolved later (per-cell), so check it locally here: a real
    # list marker sits at the body baseline and a comparable size; a raised footnote
    # ref is smaller and baseline-shifted relative to the word it precedes.
    has_geom   = {"font_size", "y_bottom"}.issubset(df.columns)
    fs_arr     = pd.to_numeric(df["font_size"], errors="coerce").to_numpy(float) if has_geom else None
    yb_arr     = pd.to_numeric(df["y_bottom"],  errors="coerce").to_numpy(float) if has_geom else None
    pre_script = df["script_type"].notna().to_numpy() if "script_type" in df.columns else np.zeros(n, dtype=bool)

    # Struct-table bypass: words inside a tagged table (nonblank table_id) skip the
    # gap channel entirely. Their cells are cut only at struct_group_id (TD/TH)
    # boundaries so one table cell → one docslicer cell, and no gap heuristic can
    # bridge two identified TDs. _merge_struct_groups later unifies a TD/TH that
    # spans multiple visual lines.
    has_struct_table = {"table_id", "struct_group_id"}.issubset(df.columns)
    if has_struct_table:
        table_word = df["table_id"].notna().to_numpy()
        sg_valid   = df["struct_group_id"].notna().to_numpy()
        sg_arr     = df["struct_group_id"].to_numpy(dtype=object)
    else:
        table_word = np.zeros(n, dtype=bool)
        sg_valid   = np.zeros(n, dtype=bool)
        sg_arr     = np.empty(n, dtype=object)

    def _looks_like_script(a: int, b: int) -> bool:
        """True if word *a* reads as a raised/smaller script relative to body word *b*."""
        if not has_geom:
            return False
        ref = fs_arr[b]
        if not (ref > 0) or not np.isfinite(ref):
            return False
        smaller = fs_arr[a] < _SCRIPT_DETECT_SIZE_RATIO * ref
        raised  = abs(yb_arr[a] - yb_arr[b]) > _SCRIPT_DETECT_Y_FACTOR * ref
        return bool(smaller and raised)

    cell_ids = np.empty(n, dtype=np.int64)

    next_id = 1
    i = 0
    while i < n:
        cur_line = line_arr[i]
        j = i + 1
        while j < n and line_arr[j] == cur_line:
            j += 1

        # Struct-table line: cut cells at TD/TH (struct_group_id) boundaries only,
        # never at gaps. Two adjacent words share a cell iff they carry the same
        # (nonblank) struct_group_id; untagged words each stand alone.
        if table_word[i:j].any():
            cur_cell    = next_id
            cell_ids[i] = cur_cell
            for k in range(i, j - 1):
                same_group = bool(
                    sg_valid[k] and sg_valid[k + 1] and sg_arr[k] == sg_arr[k + 1]
                )
                if same_group:
                    cell_ids[k + 1] = cur_cell
                else:
                    next_id  += 1
                    cur_cell  = next_id
                    cell_ids[k + 1] = cur_cell
            next_id += 1
            i = j
            continue

        cur_cell   = next_id
        cell_ids[i] = cur_cell

        for k in range(i, j - 1):
            raw_gap = x_left_arr[k + 1] - x_right_arr[k]
            gap_em  = gap_em_arr[k]

            if raw_gap < -config.overlap_split_pt:  # significant overlap → split
                next_id   += 1
                cur_cell   = next_id
                cell_ids[k + 1] = cur_cell
                continue

            is_bullet_merge = (
                bullet_flags[k] and k == i
                and bool(text_arr[k + 1].strip())
                and np.isfinite(gap_em)
                and gap_em <= config.bullet_max_gap_em
            )
            is_list_marker_merge = (
                list_marker_flags[k] and k == i
                and bool(text_arr[k + 1].strip())
                and not pre_script[k]                 # already tagged a script upstream
                and not _looks_like_script(k, k + 1)  # raised/smaller ref, not a marker
                and np.isfinite(gap_em)
                and gap_em <= config.bullet_max_gap_em
            )
            is_currency_merge = (
                currency_flags[k]
                and numeric_flags[k + 1]
                and np.isfinite(gap_em)
                and gap_em <= config.currency_max_gap_em
            )
            thr = thr_arr[k] if np.isfinite(thr_arr[k]) else config.em_undetermined
            is_normal_merge = np.isfinite(gap_em) and gap_em <= thr

            if is_bullet_merge or is_list_marker_merge or is_currency_merge or is_normal_merge:
                cell_ids[k + 1] = cur_cell
            else:
                next_id  += 1
                cur_cell  = next_id
                cell_ids[k + 1] = cur_cell

        next_id += 1
        i = j

    df = df.copy()
    df["cell_id"] = cell_ids
    return df, next_id - 1


# ================================================================================
# 5b. POST-MERGE TABLE REFINEMENT  (re-split multi-cell lines at the tight threshold)
# ================================================================================

def _refine_multi_cell_lines(df: pd.DataFrame, config: CellBuildConfig = CONFIG) -> pd.DataFrame:
    """
    Re-split lines that survived the first merge pass holding more than one cell.

    A line that still carries multiple cells after merging is, by construction, a
    table row: real prose always collapses to a single cell. Yet some such lines
    were classified "text"/"undetermined" and merged at the loose em threshold,
    which can bridge two narrow columns into one cell while a third column stays
    separate (the "Cash" | flow" | "Acquisitions" | "Non-cash" header case, where 
    the loose pass fuses "Cash flow" and "Acquisitions").

    Such a line is almost impossible to label "table" in the classify_line pass:
    it is short, all-alpha, and proper-cased, so the only signal pushing it toward
    table is the caps ratio — and bumping that up enough to catch it would also
    drag legitimate proper-cased prose (legal headings, defined terms) into table
    territory. The post-merge cell count is the unambiguous signal classify_line
    lacks, so we defer the decision to here: only for these lines do we drop the
    threshold to em_table and re-run the per-line merge, tightening every gap so
    the bridged columns split apart.

    Cost is one groupby over the page; the actual re-merge touches only the
    suspect rows (typically a handful of header lines), so the common case where
    no line needs refining adds just that single pass.
    """
    if df.empty or "cell_id" not in df.columns:
        return df

    # Lines with >1 surviving cell whose threshold is still above the tight table
    # cutoff. Lines already at em_table (classified table) can't tighten further.
    cells_per_line = df.groupby("line_id")["cell_id"].transform("nunique")
    thr            = pd.to_numeric(df["line_em_threshold"], errors="coerce")
    suspect        = (cells_per_line > 1) & (thr > config.em_table)

    # Struct-table lines already have their cells cut at TD/TH boundaries; the
    # gap-based re-merge inside _assign_cell_ids_horiz must never touch them.
    if "table_id" in df.columns:
        table_line = df.groupby("line_id")["table_id"].transform("count") > 0
        suspect &= ~table_line

    if not suspect.any():
        return df

    df = df.copy()
    mask = df["line_id"].isin(df.loc[suspect, "line_id"].unique())

    df.loc[mask, "line_class"]        = "table"
    df.loc[mask, "line_em_threshold"] = config.em_table

    # Re-merge only the suspect rows at the tight threshold, then map the fresh
    # cell_ids back by word_id (stable across the internal sort). Offset above the
    # current max so re-split lines never collide with the kept lines' ids.
    offset      = int(df["cell_id"].max())
    resplit, _  = _assign_cell_ids_horiz(df.loc[mask].copy(), config)
    id_map      = dict(zip(resplit["word_id"].to_numpy(), resplit["cell_id"].to_numpy() + offset))
    df.loc[mask, "cell_id"] = df.loc[mask, "word_id"].map(id_map).astype(np.int64)

    return df


# ================================================================================
# 6. STRUCT GROUP MERGE  (cross-line merging via PDF logical structure)
# ================================================================================

def _merge_struct_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cell_ids that share a struct_group_id using union-find.

    Each struct_group_id represents a logical PDF element (paragraph, list item, …)
    that may span multiple visual lines. After horizontal cell assignment, words
    in the same struct group but different cell_ids are unified under the lowest
    cell_id in the group.
    """
    if "struct_group_id" not in df.columns:
        return df

    sg_mask = df["struct_group_id"].notna()
    if not sg_mask.any():
        return df

    sg_sub = df.loc[sg_mask, ["struct_group_id", "cell_id"]].drop_duplicates()
    sg_sub = sg_sub.sort_values(["struct_group_id", "cell_id"])
    sg_arr   = sg_sub["struct_group_id"].to_numpy()
    cell_arr = sg_sub["cell_id"].to_numpy()

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    prev_sg   = None
    root_cell = None
    for i in range(len(sg_arr)):
        sg   = sg_arr[i]
        cell = int(cell_arr[i])
        if sg != prev_sg:
            root_cell = cell
            prev_sg   = sg
        else:
            union(root_cell, cell)

    if not parent:
        return df

    df = df.copy()
    df["cell_id"] = df["cell_id"].map(find)
    return df


# ================================================================================
# 7. CELL-LEVEL SCRIPT DETECTION
# ================================================================================

def _detect_cell_level_scripts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each cell, detect sub/superscript words not already tagged by the
    word extractor (they arrived as separate pdfium words).

    A word qualifies if it is untagged and:
      - font_size < _SCRIPT_DETECT_SIZE_RATIO * cell reference size
      - |y_bottom - cell reference baseline| > _SCRIPT_DETECT_Y_FACTOR * ref size

    Reference size     = max font_size in the cell (all words, matching original).
    Reference baseline = median y_bottom of normal-sized untagged words.

    Cells whose words span more than one original line_id (e.g. cells merged
    across visual lines by _merge_struct_groups) are skipped entirely: a size/
    baseline difference there reflects separate stacked lines, not a sub/
    superscript relationship. Words inside a tagged table (nonblank table_id) are
    likewise never tagged, for the same stacked-row reason.
    """
    if "cell_id" not in df.columns or "font_size" not in df.columns:
        return df

    df = df.copy()
    if "script_type" not in df.columns:
        df["script_type"] = None

    untagged = df["script_type"].isna()

    # Struct-table cells often pack several visual rows under one line_id, so a
    # size/baseline difference there is a stacked row, not a sub/superscript.
    # Never script-tag words inside a tagged table.
    if "table_id" in df.columns:
        untagged &= df["table_id"].isna()

    if not untagged.any():
        return df

    font_size = pd.to_numeric(df["font_size"], errors="coerce")
    y_bottom  = pd.to_numeric(df["y_bottom"],  errors="coerce")
    cell_id   = df["cell_id"]

    if "line_id" in df.columns:
        single_line_cell = df["line_id"].groupby(cell_id).transform("nunique") == 1
        untagged = untagged & single_line_cell
        if not untagged.any():
            return df

    # Per-cell max font_size (all words, matching original logic)
    ref_size = font_size.groupby(cell_id).transform("max")
    valid    = ref_size > 0

    # Normal words: untagged and font_size >= 0.88 * ref_size
    normal = untagged & valid & (font_size >= 0.88 * ref_size)

    # Per-cell median y_bottom of normal words
    ref_baseline = y_bottom.where(normal).groupby(cell_id).transform("median")

    threshold = _SCRIPT_DETECT_Y_FACTOR * ref_size
    shift     = ref_baseline - y_bottom
    small     = untagged & valid & (font_size < _SCRIPT_DETECT_SIZE_RATIO * ref_size)
    has_ref   = ref_baseline.notna()

    df.loc[small & has_ref & (shift >  threshold), "script_type"] = "superscript"
    df.loc[small & has_ref & (shift < -threshold), "script_type"] = "subscript"

    return df


# ================================================================================
# 8. CELL AGGREGATION  (words → cells)
# ================================================================================

def _join_texts(texts: list[str]) -> str:
    """Join word texts with a space; script-notation tokens ([^ / [_) attach without space."""
    # TODO: After strikethrough detection, merge in words with ~~word~~
    result = ""
    for t in texts:
        t = str(t)
        if not t.strip():
            continue
        if not result:
            result = t
        elif result.endswith("-"):
            result = result + t
        elif t.startswith(("[^", "[_")):
            result = result + t
        else:
            result = result + " " + t
    return result


def _build_cells_df(df_words: pd.DataFrame) -> pd.DataFrame:
    """Aggregate words into cells via the shared hierarchical aggregator."""
    df = df_words.copy()
    df["_fmt_text"] = df["text"].astype(str)
    if "script_type" in df.columns:
        sup = df["script_type"] == "superscript"
        sub = df["script_type"] == "subscript"
        df.loc[sup, "_fmt_text"] = "[^" + df.loc[sup, "_fmt_text"] + "]"
        df.loc[sub, "_fmt_text"] = "[_" + df.loc[sub, "_fmt_text"] + "]"

    agg_spec = build_standard_agg_spec(
        identity_cols=[
            "page_number",
            "page_width",
            "page_height",
            "reading_column",
            "gutter_id_left",
            "gutter_id_right",
            # line-level features (useful for downstream debug / table detection)
            "line_class",
            "line_score",
            "line_em_threshold",
            "line_is_bimodal",
        ],
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=False,
        include_hierarchy=False,
        include_table=False,
        extra_agg={
            "_fmt_text": lambda s: _join_texts(s.tolist()),
            "word_id":   list,
            "line_id":   lambda s: sorted(s.unique().tolist()),
            "stream_group_id": "first",
            #"stream_group_trigger": "first",
        },
    )

    df_cells = aggregate_hierarchical(
        df,
        group_col="cell_id",
        agg_spec=agg_spec,
        rename_count_col={"word_id": "word_ids", "line_id": "line_ids"},
    )
    df_cells = df_cells.rename(columns={"_fmt_text": "text"})
    # Scalar line_id for downstream consumers that expect a single value per cell.
    # Equals the minimum (first) line_id in the cell.
    if "line_ids" in df_cells.columns:
        df_cells["line_id"] = df_cells["line_ids"].apply(lambda ids: ids[0] if ids else None)
    return df_cells


# ================================================================================
# 9. VERTICAL WORD PROCESSING
# ================================================================================

def _process_vertical_words(
    df_vert: pd.DataFrame,
    cell_id_offset: int,
    config: CellBuildConfig = CONFIG,
    detect_scripts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cells from vertical (TTB/BTT) words via coordinate-swap.

    Step 07 already assigned globally unique line_ids to vertical words (using
    the same x↔y swap trick to group words sharing the same x-band). Here we
    only need the swap to make _assign_cell_ids_horiz operate along the y-axis
    (merging by y-gap within each line_id group). Cell IDs are offset above the
    horizontal ceiling so they remain globally unique.
    """
    if df_vert.empty:
        return pd.DataFrame(), df_vert.copy()

    df = df_vert.copy()

    orig_xl = df["x_left"].to_numpy(dtype=float)
    orig_xr = df["x_right"].to_numpy(dtype=float)
    orig_yt = df["y_top"].to_numpy(dtype=float)
    orig_yb = df["y_bottom"].to_numpy(dtype=float)

    df["y_top"]    = orig_xl
    df["y_bottom"] = orig_xr
    df["x_left"]   = orig_yt
    df["x_right"]  = orig_yb

    # BTT words read bottom→top: negate swapped x so ascending sort gives correct order.
    # Stash pre-negation values so they survive sort+reset_index inside _assign_cell_ids_horiz.
    df["_x_left_pre_neg"]  = df["x_left"]
    df["_x_right_pre_neg"] = df["x_right"]
    if "text_orientation" in df.columns:
        btt_mask = df["text_orientation"] == "BTT"
        if btt_mask.any():
            btt_xl = df.loc[btt_mask, "x_left"].to_numpy(dtype=float)
            btt_xr = df.loc[btt_mask, "x_right"].to_numpy(dtype=float)
            df.loc[btt_mask, "x_left"]  = -btt_xr
            df.loc[btt_mask, "x_right"] = -btt_xl

    # Annotate lines with em features in swapped space (skip classification — use undetermined threshold).
    # Pre-sort once so groups are already in x_left order.
    df = df.sort_values(["line_id", "x_left"], kind="mergesort").reset_index(drop=True)

    # Vectorised gap_em_right in swapped coordinate space.
    xl  = df["x_left"].to_numpy(float)
    xr  = df["x_right"].to_numpy(float)
    fs  = df["font_size"].to_numpy(float)
    lid_arr = df["line_id"].to_numpy()

    nv = len(df)
    same_next_v        = np.empty(nv, dtype=bool)
    same_next_v[:-1]   = lid_arr[:-1] == lid_arr[1:]
    same_next_v[-1]    = False
    gap_pt_v           = np.empty(nv);  gap_pt_v[-1]  = np.nan
    gap_pt_v[:-1]      = xl[1:] - xr[:-1]
    fs_max_v           = np.empty(nv);  fs_max_v[-1]  = np.nan
    fs_max_v[:-1]      = np.maximum(fs[:-1], fs[1:])
    fs_max_v           = np.where((fs_max_v > 0) & np.isfinite(fs_max_v), fs_max_v, np.nan)
    raw_gem_v          = gap_pt_v / fs_max_v
    df["gap_em_right"] = np.where(same_next_v & np.isfinite(raw_gem_v), np.round(raw_gem_v, 4), np.nan)

    feat_rows: list[dict] = []
    for lid, grp in df.groupby("line_id", sort=False):
        gs = _line_gap_stats(
            grp["x_left"].to_numpy(float),
            grp["x_right"].to_numpy(float),
            grp["font_size"].to_numpy(float),
            config,
        )
        feat_rows.append({
            "line_id":           lid,
            "line_class":        "undetermined",
            "line_score":        0.0,
            "line_em_threshold": config.em_undetermined,
            "line_is_bimodal":   gs["is_bimodal"],
        })

    df_feat = pd.DataFrame(feat_rows)
    df = df.merge(df_feat, on="line_id", how="left")

    # Restore pre-negation x before cell assignment
    df["x_left"]  = df["_x_left_pre_neg"]
    df["x_right"] = df["_x_right_pre_neg"]
    df.drop(columns=["_x_left_pre_neg", "_x_right_pre_neg"], inplace=True)

    df, _ = _assign_cell_ids_horiz(df, config)
    if detect_scripts:
        df = _detect_cell_level_scripts(df)

    # Swap back to original coordinates
    cur_xl = df["x_left"].to_numpy(dtype=float)
    cur_xr = df["x_right"].to_numpy(dtype=float)
    cur_yt = df["y_top"].to_numpy(dtype=float)
    cur_yb = df["y_bottom"].to_numpy(dtype=float)
    df["x_left"]   = cur_yt
    df["x_right"]  = cur_yb
    df["y_top"]    = cur_xl
    df["y_bottom"] = cur_xr

    df["cell_id"] = df["cell_id"] + cell_id_offset

    df_vert_cells = _build_cells_df(df)
    df_vert_cells["block_type"] = "vertical_text"

    return df_vert_cells, df


# ================================================================================
# 9b. CELL-ID RENUMBERING  (restore reading-order numbering)
# ================================================================================

def _renumber_cells_reading_order(
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Renumber cell_id to count upward in reading order across the whole document.

    Per-page assignment, the post-merge re-split (which parks fresh ids above the
    page max) and the vertical offset all leave gaps and out-of-order ids. Here we
    collapse them to a dense 1..N sequence ordered by first appearance in
    (page_number, line_id, x_left) — i.e. line_id-ascending, left-to-right within
    a line — so the numbering tracks reading order again. A cell spanning several
    lines takes the slot of its earliest word.

    Both frames are remapped with the same mapping so word↔cell ids stay aligned.
    """
    if df_words.empty or "cell_id" not in df_words.columns:
        return df_cells, df_words

    order  = df_words.sort_values(["page_number", "line_id", "x_left"], kind="mergesort")
    codes  = pd.factorize(order["cell_id"].to_numpy())[0]          # 0-based, by first appearance
    mapping = dict(zip(order["cell_id"].to_numpy(), codes + 1))    # dense 1..N

    df_words = df_words.copy()
    df_words["cell_id"] = df_words["cell_id"].map(mapping).astype(np.int64)

    if not df_cells.empty and "cell_id" in df_cells.columns:
        df_cells = df_cells.copy()
        df_cells["cell_id"] = df_cells["cell_id"].map(mapping).astype(np.int64)

    return df_cells, df_words


# ================================================================================
# 10. ENTRY POINT  (public API)
# ================================================================================

def build_cells(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
    config: CellBuildConfig = CONFIG,
    detect_scripts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Words -> Cells.

    Pipeline:
      1. Annotate words with per-line gap statistics and em threshold.
      2. Assign cell_ids within each line using em-threshold gap merging,
         with bullet, list-marker, and currency special-case overrides.
         Lines inside a tagged table (nonblank table_id) bypass this: their
         cells are cut only at TD/TH (struct_group_id) boundaries, so the gap
         heuristics can never merge two identified table cells.
      3. Merge cell_ids that share a struct_group_id (PDF logical structure).
      4. Aggregate words into cell rows.

    Horizontal and vertical words are processed independently, with vertical
    IDs offset above horizontal so the combined cell_id space is globally unique.

    detect_scripts
        When True (default), run per-cell sub/superscript detection. Callers on
        shaky geometry (e.g. the OCR pipeline, where bboxes and font sizes are
        unreliable) should pass False to suppress false script tags.

    Returns
    -------
    df_cells  : one row per cell
    df_words  : input words annotated with cell_id and line feature columns
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_words.copy()

    if "reading_column" not in df.columns:
        df["reading_column"] = 1
    df["reading_column"] = df["reading_column"].fillna(1).astype(int)

    # Split horizontal vs vertical — vertical words use a coordinate-swap pipeline.
    if "text_orientation" in df.columns:
        vert_mask = df["text_orientation"].isin(["TTB", "BTT"])
    else:
        vert_mask = pd.Series(False, index=df.index)

    df_vert  = df[vert_mask].copy()
    df_horiz = df[~vert_mask].copy()

    # ── Horizontal ────────────────────────────────────────────────────────────
    pages           = sorted(df["page_number"].unique())
    horiz_words_out: list[pd.DataFrame] = []
    cells_out:       list[pd.DataFrame] = []
    vert_cells_out:  list[pd.DataFrame] = []
    vert_words_out:  list[pd.DataFrame] = []
    running_cell = 0

    for page_num in pages:
        df_h = df_horiz[df_horiz["page_number"] == page_num].copy()
        df_v = df_vert[df_vert["page_number"] == page_num].copy()

        # line_id comes from step 07 and is already globally unique — leave it untouched.
        if not df_h.empty:
            df_h = _annotate_line_features(df_h, config)
            df_h, _ = _assign_cell_ids_horiz(df_h, config)
            df_h = _refine_multi_cell_lines(df_h, config)
            df_h = _merge_struct_groups(df_h)
            if detect_scripts:
                df_h = _detect_cell_level_scripts(df_h)
            df_h["cell_id"] += running_cell

        page_cell_max = int(df_h["cell_id"].max()) if not df_h.empty else running_cell

        if not df_v.empty:
            vc, vw = _process_vertical_words(df_v, page_cell_max, config, detect_scripts)
            vert_cells_out.append(vc)
            vert_words_out.append(vw)
            running_cell = int(vw["cell_id"].max()) if not vw.empty else page_cell_max
        else:
            running_cell = page_cell_max

        if not df_h.empty:
            horiz_words_out.append(df_h)
            cells_out.append(_build_cells_df(df_h))

    df_horiz_out = pd.concat(horiz_words_out, ignore_index=True) if horiz_words_out else pd.DataFrame()
    df_cells     = pd.concat(cells_out,        ignore_index=True) if cells_out        else pd.DataFrame()

    if vert_words_out or vert_cells_out:
        df_vert_out = pd.concat([w for w in vert_words_out if not w.empty], ignore_index=True)
        df_vert_cells = pd.concat([c for c in vert_cells_out if not c.empty], ignore_index=True)
        if not df_vert_cells.empty:
            df_cells = pd.concat([df_cells, df_vert_cells], ignore_index=True)
    else:
        df_vert_out = pd.DataFrame()

    df_words_out = (
        pd.concat([df_horiz_out, df_vert_out], ignore_index=True)
        .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Renumber to a dense, reading-order cell_id sequence (the per-page offsets and
    # the post-merge re-split otherwise leave gaps / out-of-order ids).
    df_cells, df_words_out = _renumber_cells_reading_order(df_cells, df_words_out)

    if not df_cells.empty and "cell_id" in df_cells.columns:
        df_cells = df_cells.sort_values("cell_id", kind="mergesort").reset_index(drop=True)

    return df_cells, df_words_out

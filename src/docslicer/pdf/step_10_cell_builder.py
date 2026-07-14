"""
step_10_cell_builder.py

Words -> Cells, decided strictly per line (no cross-row lookahead).

Per-line decision pipeline
--------------------------
For each line, in order (the analysis primitives are shared with step 09 and
the gutter detector; see _utils/line_classification.py):

  1. Gap distribution         -> line_gap_stats()
       The line's inter-word gaps, em-normalized (gap / font_size).
       If the gaps are *bimodal* (a clear valley), the wide cluster are
       column gutters: split there. This is the primary signal and it is
       fully per-line. It is what separates:
         - justified prose : all gaps ~equal  (ratio ~1.0)  -> one texture
         - table header     : tight intra-phrase spaces + wider gutters
                              (ratio >> 1)                  -> split at valley
  2. Classification           -> classify_line()
       "text" | "table" | "undetermined". A *prior*, only consulted to
       resolve UNIMODAL lines (all gaps similar), where the distribution
       alone can't say merge-all vs split-all.
  3. Fallback em threshold    -> em_threshold_for_class()
       Scale-invariant constant (gap_em <= T -> merge). No font interpolation:
       dividing the gap by font_size already removes the size dependence.
  4. Special-case overrides during merging
       Bullets and list markers merge into the following word, and currency
       symbols merge into an adjacent numeric value, at their own (looser)
       gap limits.

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
0.69/0.69 = 1.0). Hence step 1 before step 3.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._utils.cpu import resolve_worker_count
from .._utils.parallel import PARALLEL_PAGE_THRESHOLD, chunk_evenly, warn_pool_fell_back
from .._utils.text_utils import _BULLET_TOKENS, _CURRENCY_SYMBOLS, is_list_marker, numeric_value_mask
from .._utils.df_aggregation.registry_aggregator import Agg, aggregate_to
from .._utils.df_aggregation.text_merge import apply_inline_markup, merge_text_within_line
from ._utils.line_classification import (
    LineClassificationConfig,
    score_lines,
    token_content_features,
)
from ._utils.script_thresholds import SCRIPT_SIZE_DOWN, SCRIPT_SIZE_UP, SCRIPT_Y_FACTOR


# ================================================================================
# CONFIG
# ================================================================================

@dataclass(frozen=True)
class CellBuildConfig:
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

    # --- Shared per-line analysis knobs (em thresholds, bimodality detection) ---
    line: LineClassificationConfig = LineClassificationConfig()


CONFIG = CellBuildConfig()

# Plausible sub/superscript & math tokens: short, and either non-alphabetic,
# a single letter, an ordinal (1st/2nd), or a small roman numeral. Long or
# multi-letter alphabetic runs ("PURSUANT", "TO", "THE") never qualify, so a
# font/baseline anomaly on real words can't be mistaken for a script.
_SCRIPT_TOKEN_RE = re.compile(
    r"""^(?:
        [(\[]?[-+–−]?\d{1,3}[)\].,]?   # small ints, opt sign/bracket:  2  13  -1  (1)  1)
      | \d{1,2}(?:st|nd|rd|th)          # ordinals:  1st 2nd 3rd 4th
      | [(\[]?[A-Za-z][)\].]?           # single letter, opt bracket:  a  (a)  x)
      | [ivxlcIVXLC]{1,3}               # small roman numerals:  i ii iv IX
      | [-+*/=<>~^·•°′″†‡§¶±×÷…()\[\]]+  # math / footnote / bracket symbols
      | (?=[A-Za-z0-9+\-*/=^_·×÷–−]{2,5}$)
        [A-Za-z0-9]+(?:[-+*/=^_·×÷–−][A-Za-z0-9]+)+  # short math expr:  t-1  x+y  n+1
      | [☐☑☒✓✗✔✘]                     # checkbox & cross glyphs
    )$""",
    re.VERBOSE,
)


def _concat_frames(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the non-empty frames; empty DataFrame when none remain."""
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ================================================================================
# 1. LINE FEATURE ANNOTATION  (gap stats + classification, broadcast per word)
# ================================================================================

def _annotate_line_features(
    df: pd.DataFrame,
    config: CellBuildConfig = CONFIG,
    classify: bool = True,
) -> pd.DataFrame:
    """
    Compute per-line gap statistics — and, when ``classify`` is True, content
    stats plus text/table classification — broadcast to every word.

    Fully vectorised equivalent of the shared per-line primitives: the gap
    distribution (line_gap_stats) is computed for all lines at once via sorted
    per-line segments and reduceat, content features (content_stats) via
    token_content_features + per-line sums, and classification via score_lines
    — the same scoring implementation classify_line wraps, so the labels are
    identical by construction.

    Added columns:
        gap_em_right, line_n_gaps, line_median_em, line_max_em, line_jump_ratio,
        line_is_bimodal, line_split_em, line_class, line_score, line_em_threshold
    and with ``classify=True`` additionally:
        line_n_words, line_alpha_ratio, line_numeric_ratio, line_stopword_hits,
        line_cap_ratio, line_has_punct

    ``classify=False`` (the vertical-words path) skips the content channel and
    labels every line "undetermined" at the neutral em threshold.
    """
    if df.empty:
        return df

    df = df.sort_values(["line_id", "x_left"], kind="mergesort").reset_index(drop=True)
    n = len(df)

    xl  = df["x_left"].to_numpy(float)
    xr  = df["x_right"].to_numpy(float)
    fs  = df["font_size"].to_numpy(float)
    lid = df["line_id"].to_numpy()

    # Line index per word and per-line segment bounds.
    is_line_first     = np.empty(n, dtype=bool)
    is_line_first[0]  = True
    is_line_first[1:] = lid[1:] != lid[:-1]
    starts   = np.flatnonzero(is_line_first)
    counts   = np.diff(np.append(starts, n))     # words per line
    L        = len(starts)
    line_idx = np.cumsum(is_line_first) - 1      # 0-based line index per word

    # ── Right-neighbour gap in em (signed; negative = overlap). raw_gem keeps
    # full precision for the distribution stats; the word column is rounded.
    # NaN for line-final words and where a flanking font size is invalid.
    raw_gem = np.full(n, np.nan)
    if n > 1:
        gap_pt = xl[1:] - xr[:-1]
        fs_max = np.maximum(fs[:-1], fs[1:])                 # larger of left/right word
        fs_max = np.where((fs_max > 0) & np.isfinite(fs_max), fs_max, np.nan)
        raw_gem[:-1] = np.where(~is_line_first[1:], gap_pt / fs_max, np.nan)
    df["gap_em_right"] = np.round(raw_gem, 4)

    # ── Gap-distribution stats per line (vectorised line_gap_stats) ─────────
    # Pair p = (word p, word p+1); pairs with NaN raw_gem are cross-line or
    # invalid-font. Only positive finite gaps feed the distribution.
    pair_g    = raw_gem[: n - 1]
    pair_line = line_idx[: n - 1]
    nn_gaps   = np.bincount(pair_line[~np.isnan(pair_g)], minlength=L)  # all-overlap fallback count

    v_mask = np.isfinite(pair_g) & (pair_g > 0)
    vg     = pair_g[v_mask]
    vlin   = pair_line[v_mask]
    c      = np.bincount(vlin, minlength=L)      # valid gaps per line
    off    = np.concatenate(([0], np.cumsum(c)))

    # Sort gaps within each line (vlin is already non-decreasing).
    order = np.lexsort((vg, vlin))
    sv    = vg[order]
    sl    = vlin[order]

    has_valid = c > 0
    median_em = np.zeros(L)
    max_em    = np.zeros(L)
    if sv.size:
        o  = off[:-1]
        lo = o + (c - 1) // 2                    # np.median of a sorted array:
        hi = o + c // 2                          # mean of the two middle values
        median_em[has_valid] = (sv[lo[has_valid]] + sv[hi[has_valid]]) / 2.0
        max_em[has_valid]    = sv[off[1:][has_valid] - 1]

    jump_ratio = np.ones(L)
    split_em   = np.full(L, np.inf)
    is_bimodal = np.zeros(L, dtype=bool)
    if sv.size >= 2:
        rt     = sv[1:] / (sv[:-1] + 1e-9)       # consecutive sorted ratios
        in_seg = sl[1:] == sl[:-1]               # ratio stays within one line

        # Fallback jump_ratio: the largest within-line ratio, for lines with
        # enough gaps but no valley. Cross-line ratio slots are masked to -inf;
        # reduceat over the per-line offsets then takes each line's max (its
        # garbage output for empty/short segments is overwritten below).
        rt_pad  = np.append(np.where(in_seg, rt, -np.inf), -np.inf)
        seg_max = np.maximum.reduceat(rt_pad, np.minimum(off[:-1], rt_pad.size - 1))
        enough  = c >= config.line.min_gaps_for_ratio
        jump_ratio = np.where(enough, seg_max, 1.0)

        # Lowest valley per line: first jump big enough whose upper side is
        # gutter-sized (min_space_em rejects spurious jumps within the space
        # cluster). Gaps are ascending within a line, so the first candidate
        # in document order is the lowest valley.
        cand = (
            in_seg & enough[sl[1:]]
            & (rt >= config.line.ratio_split)
            & (sv[1:] > config.line.min_space_em)
        )
        ci = np.flatnonzero(cand)
        if ci.size:
            valley_lines, first_pos = np.unique(sl[ci], return_index=True)
            k = ci[first_pos]
            jump_ratio[valley_lines] = rt[k]
            split_em[valley_lines]   = np.sqrt(sv[k] * sv[k + 1])  # geometric midpoint
            is_bimodal[valley_lines] = True

    n_gaps = np.where(c > 0, c, nn_gaps)

    line_cols: dict[str, np.ndarray] = {
        "line_n_gaps":     n_gaps,
        "line_median_em":  np.round(median_em, 4),
        "line_max_em":     np.round(max_em, 4),
        "line_jump_ratio": np.round(jump_ratio, 4),
        "line_is_bimodal": is_bimodal,
        "line_split_em":   np.where(np.isfinite(split_em), np.round(split_em, 4), np.nan),
    }

    # ── Content stats + classification per line (vectorised) ────────────────
    if classify:
        def _per_line_sum(mask: np.ndarray) -> np.ndarray:
            return np.add.reduceat(mask.astype(np.int64), starts)

        # Script-tagged words (footnote refs, exponents) don't take part in
        # the content channel — they aren't the line's prose/table texture.
        if "script_type" in df.columns:
            untagged = df["script_type"].isna().to_numpy()
        else:
            untagged = np.ones(n, dtype=bool)

        tok = token_content_features(df["text"].astype(str).to_numpy())

        n_content = _per_line_sum(untagged)
        alpha     = _per_line_sum(untagged & tok["has_alpha"])
        numeric   = _per_line_sum(untagged & tok["is_numeric"])
        caps      = _per_line_sum(untagged & tok["cap_initial"])
        stop      = _per_line_sum(untagged & tok["is_stopword"])
        has_punct = _per_line_sum(untagged & tok["has_punct"]) > 0

        nz            = np.maximum(n_content, 1)
        alpha_ratio   = alpha / nz
        numeric_ratio = numeric / nz
        cap_ratio     = caps / np.maximum(alpha, 1)

        labels, scores = score_lines(
            n_content, stop, has_punct, alpha_ratio, cap_ratio, numeric_ratio,
            median_em, jump_ratio, is_bimodal,
        )
        thr = np.where(labels == "table", config.line.em_table,
              np.where(labels == "text", config.line.em_text,
                       config.line.em_undetermined))

        # Struct-table lines bypass the classification channel: their cells are
        # cut at TD/TH boundaries downstream, so the em threshold is never
        # consulted.
        if "table_id" in df.columns:
            table_line = _per_line_sum(df["table_id"].notna().to_numpy()) > 0
            labels = np.where(table_line, "table", labels).astype(object)
            scores = np.where(table_line, 0.0, scores)
            thr    = np.where(table_line, config.line.em_table, thr)

        line_cols.update({
            "line_n_words":       n_content,
            "line_alpha_ratio":   np.round(alpha_ratio, 4),
            "line_numeric_ratio": np.round(numeric_ratio, 4),
            "line_stopword_hits": stop,
            "line_cap_ratio":     np.round(cap_ratio, 4),
            "line_has_punct":     has_punct,
            "line_score":         np.round(scores, 2),
            "line_class":         labels,
            "line_em_threshold":  thr,
        })
    else:
        line_cols.update({
            "line_score":        np.zeros(L),
            "line_class":        np.full(L, "undetermined", dtype=object),
            "line_em_threshold": np.full(L, config.line.em_undetermined),
        })

    # Broadcast per-line values to words: rows are line-sorted, so np.repeat
    # replaces the old line_id merge.
    for col, values in line_cols.items():
        df[col] = np.repeat(values, counts)

    return df


# ================================================================================
# 2. CELL ID ASSIGNMENT  (horizontal words)
# ================================================================================

def _assign_cell_ids_horiz(df: pd.DataFrame, config: CellBuildConfig = CONFIG) -> tuple[pd.DataFrame, int]:
    """
    Assign cell_id to horizontal words using the per-line em threshold and
    gap_em_right already annotated on df.

    Fully vectorised: each adjacent same-line word pair gets one merge/split
    decision from boolean arrays (pair index k covers words k and k+1), and
    cell_id is the cumulative sum of the new-cell flags — identical numbering
    to a sequential left-to-right scan.

    Returns (annotated df, max_cell_id).
    """
    df = df.sort_values(["line_id", "x_left"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    if n == 0:
        df["cell_id"] = np.empty(0, dtype=np.int64)
        return df, 0

    line_arr    = df["line_id"].to_numpy(dtype=np.int64)
    x_left_arr  = df["x_left"].to_numpy(dtype=float)
    x_right_arr = df["x_right"].to_numpy(dtype=float)
    gap_em_arr  = pd.to_numeric(df["gap_em_right"], errors="coerce").to_numpy(dtype=float)
    thr_arr     = pd.to_numeric(df["line_em_threshold"], errors="coerce").to_numpy(dtype=float)

    stripped       = df["text"].astype(str).str.strip()
    nonblank       = (stripped.str.len() > 0).to_numpy()
    bullet_flags   = stripped.isin(_BULLET_TOKENS).to_numpy()
    currency_flags = stripped.isin(_CURRENCY_SYMBOLS).to_numpy()

    is_line_first     = np.empty(n, dtype=bool)
    is_line_first[0]  = True
    is_line_first[1:] = line_arr[1:] != line_arr[:-1]

    # List markers ((1), 1., (a), [1], iv. …) merge into the following word like
    # bullets, but only when the marker isn't a sub/superscript reference. The
    # regex is only consulted for line-leading words (only they can merge as
    # markers), so restrict the per-word apply to those positions.
    list_marker_flags = np.zeros(n, dtype=bool)
    first_idx = np.flatnonzero(is_line_first)
    list_marker_flags[first_idx] = stripped.iloc[first_idx].apply(is_list_marker).to_numpy()

    # Numeric right-hand side of a currency merge, including dash/NA placeholders
    # ("$  1,234", "$  —"). Only positions directly after a currency symbol are
    # ever consulted, so the mask runs on those rows alone.
    numeric_flags = np.zeros(n, dtype=bool)
    after_currency = np.flatnonzero(currency_flags[:-1]) + 1
    if after_currency.size:
        numeric_flags[after_currency] = numeric_value_mask(df["text"].iloc[after_currency]).to_numpy()

    # Script check for line-leading list markers: a real marker sits at the body
    # baseline and a comparable size; a raised footnote ref is smaller and
    # baseline-shifted relative to the word it precedes. (Script status proper
    # is resolved later, per cell.) NaN geometry compares False on both legs,
    # so invalid references never read as scripts.
    pre_script = df["script_type"].notna().to_numpy() if "script_type" in df.columns else np.zeros(n, dtype=bool)
    if {"font_size", "y_bottom"}.issubset(df.columns):
        fs_arr  = pd.to_numeric(df["font_size"], errors="coerce").to_numpy(float)
        yb_arr  = pd.to_numeric(df["y_bottom"],  errors="coerce").to_numpy(float)
        ref     = fs_arr[1:]
        ref_ok  = (ref > 0) & np.isfinite(ref)
        smaller = fs_arr[:-1] < SCRIPT_SIZE_DOWN * ref
        raised  = np.abs(yb_arr[:-1] - yb_arr[1:]) > SCRIPT_Y_FACTOR * ref
        looks_like_script = ref_ok & smaller & raised
    else:
        looks_like_script = np.zeros(n - 1, dtype=bool)

    # ── Pair decisions (index k = words k and k+1; cross-line pairs are
    #    overridden by is_line_first below) ───────────────────────────────────
    gap_em     = gap_em_arr[:-1]
    finite_gap = np.isfinite(gap_em)
    raw_gap    = x_left_arr[1:] - x_right_arr[:-1]
    thr        = np.where(np.isfinite(thr_arr[:-1]), thr_arr[:-1], config.line.em_undetermined)

    normal_merge = finite_gap & (gap_em <= thr)
    bullet_merge = (
        bullet_flags[:-1] & is_line_first[:-1] & nonblank[1:]
        & finite_gap & (gap_em <= config.bullet_max_gap_em)
    )
    list_marker_merge = (
        list_marker_flags[:-1] & is_line_first[:-1] & nonblank[1:]
        & ~pre_script[:-1]            # already tagged a script upstream
        & ~looks_like_script          # raised/smaller ref, not a marker
        & finite_gap & (gap_em <= config.bullet_max_gap_em)
    )
    currency_merge = (
        currency_flags[:-1] & numeric_flags[1:]
        & finite_gap & (gap_em <= config.currency_max_gap_em)
    )

    # Significant overlap always splits, even when a merge rule fires.
    overlap_split = raw_gap < -config.overlap_split_pt
    gap_split = overlap_split | ~(normal_merge | bullet_merge | list_marker_merge | currency_merge)

    # ── Struct-table bypass: words inside a tagged table (nonblank table_id)
    # skip the gap channel entirely. Their cells are cut only at struct_group_id
    # (TD/TH) boundaries so one table cell → one docslicer cell, and no gap
    # heuristic can bridge two identified TDs; untagged words each stand alone.
    # _merge_struct_groups later unifies a TD/TH spanning multiple visual lines.
    pair_split = gap_split
    if {"table_id", "struct_group_id"}.issubset(df.columns):
        table_word = df["table_id"].notna().to_numpy()
        if table_word.any():
            # A line is a struct-table line when ANY of its words is tagged.
            counts     = np.diff(np.append(first_idx, n))
            table_line = np.repeat(np.logical_or.reduceat(table_word, first_idx), counts)

            sg_valid   = df["struct_group_id"].notna().to_numpy()
            sg_arr     = df["struct_group_id"].to_numpy(dtype=object)
            same_group = sg_valid[:-1] & sg_valid[1:] & (sg_arr[:-1] == sg_arr[1:])
            pair_split = np.where(table_line[:-1], ~same_group, gap_split)

    # ── Cell ids: new cell at every line start and every split pair ─────────
    new_cell = is_line_first.copy()
    new_cell[1:] |= pair_split
    cell_ids = np.cumsum(new_cell)

    df["cell_id"] = cell_ids.astype(np.int64)
    return df, int(cell_ids[-1])


# ================================================================================
# 3. POST-MERGE TABLE REFINEMENT  (re-split multi-cell lines at the tight threshold)
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
    suspect        = (cells_per_line > 1) & (thr > config.line.em_table)

    # Struct-table lines already have their cells cut at TD/TH boundaries; the
    # gap-based re-merge inside _assign_cell_ids_horiz must never touch them.
    if "table_id" in df.columns:
        table_line = df.groupby("line_id")["table_id"].transform("count") > 0
        suspect &= ~table_line

    if not suspect.any():
        return df

    df = df.copy()
    mask = df["line_id"].isin(df.loc[suspect, "line_id"].unique())

    df.loc[mask, "line_class"]        = "table" # NOTE: it may be more useful to keep the original assessment, so we know this was doubtful
    df.loc[mask, "line_em_threshold"] = config.line.em_table

    # Re-merge only the suspect rows at the tight threshold, then map the fresh
    # cell_ids back by word_id (stable across the internal sort). Offset above the
    # current max so re-split lines never collide with the kept lines' ids.
    offset      = int(df["cell_id"].max())
    resplit, _  = _assign_cell_ids_horiz(df.loc[mask].copy(), config)
    id_map      = dict(zip(resplit["word_id"].to_numpy(), resplit["cell_id"].to_numpy() + offset))
    df.loc[mask, "cell_id"] = df.loc[mask, "word_id"].map(id_map).astype(np.int64)

    return df


# ================================================================================
# 4. STRUCT GROUP MERGE  (cross-line merging via PDF logical structure)
# ================================================================================

def _merge_struct_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cell_ids that share a struct_group_id using union-find.

    Each struct_group_id represents a logical PDF element (paragraph, list item, …)
    that may span multiple visual lines. After horizontal cell assignment, words
    in the same struct group but different cell_ids are unified under the lowest
    cell_id in the group.

    The group key is (page_number, struct_group_id): struct_group_id is only
    unique per page — its MCID and text-object fallback channels restart on
    every page — so keying on it alone would merge unrelated cells across
    pages when this runs over a multi-page frame.
    """
    if "struct_group_id" not in df.columns:
        return df

    sg_mask = df["struct_group_id"].notna()
    if not sg_mask.any():
        return df

    sg_sub = df.loc[sg_mask, ["page_number", "struct_group_id", "cell_id"]].drop_duplicates()
    sg_sub = sg_sub.sort_values(["page_number", "struct_group_id", "cell_id"])
    pg_arr   = sg_sub["page_number"].to_numpy()
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

    prev_key  = None
    root_cell = None
    for i in range(len(sg_arr)):
        key  = (pg_arr[i], sg_arr[i])
        cell = int(cell_arr[i])
        if key != prev_key:
            root_cell = cell
            prev_key  = key
        else:
            union(root_cell, cell)

    if not parent:
        return df

    df = df.copy()
    df["cell_id"] = df["cell_id"].map(find)
    return df


# ================================================================================
# 5. CELL-LEVEL SCRIPT DETECTION
# ================================================================================

def _detect_cell_level_scripts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each cell, detect sub/superscript words not already tagged by the
    word extractor (they arrived as separate pdfium words).

    A word qualifies if it is untagged and:
      - font_size < SCRIPT_SIZE_DOWN * cell reference size
      - |y_bottom - cell reference baseline| > SCRIPT_Y_FACTOR * ref size

    Reference size     = max font_size in the cell (all words, matching original).
    Reference baseline = median y_bottom of normal-sized untagged words.

    Cells whose words span more than one original line_id (e.g. cells merged
    across visual lines by _merge_struct_groups) are skipped entirely: a size/
    baseline difference there reflects separate stacked lines, not a sub/
    superscript relationship. Words inside a tagged table (nonblank table_id) are
    likewise never tagged, for the same stacked-row reason.

    An inverted-reference guard also skips any cell where the size-based "small"
    words outnumber the body-size words: that means the reference size (the cell
    max) is pinned to an oversized outlier (checkbox glyph, drop cap, icon) rather
    than the body text, which would otherwise flag the whole run.
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

    # Normal words: untagged and at body size relative to the cell reference
    normal = untagged & valid & (font_size >= SCRIPT_SIZE_UP * ref_size)

    # Per-cell median y_bottom of normal words
    ref_baseline = y_bottom.where(normal).groupby(cell_id).transform("median")

    threshold = SCRIPT_Y_FACTOR * ref_size
    shift     = ref_baseline - y_bottom
    small     = untagged & valid & (font_size < SCRIPT_SIZE_DOWN * ref_size)
    has_ref   = ref_baseline.notna()

    # Inverted-reference guard, on the RAW size-based candidates (before the
    # format gate). ref_size is the cell max, so a single slightly-larger glyph
    # (an 11pt checkbox among 9pt body text, a drop cap, an icon) becomes the
    # reference and makes every body word look "small" and baseline-shifted. A
    # real script is outnumbered by body-size words; when the small words instead
    # outnumber them, the reference is that oversized outlier — skip the cell.
    # This must be measured on the raw mask: the format gate below removes most
    # of those false "small" words, which would otherwise hide the inversion.
    n_small = small.groupby(cell_id).transform("sum")
    n_large = (untagged & ~small).groupby(cell_id).transform("sum")
    ref_ok  = n_small <= n_large

    # Format gate: only tokens shaped like a real sub/superscript or math atom
    # are eligible. Restricted to the small candidates so the regex runs on a
    # fraction of rows. Rejects long/multi-letter words a font anomaly alone
    # would otherwise flag (e.g. "[^PURSUANT]").
    fmt_ok = pd.Series(False, index=df.index)
    if small.any():
        cand = df.loc[small, "text"].astype(str).str.strip()
        # Match via the compiled object: Series.str.match rejects a compiled
        # pattern that carries flags (re.VERBOSE). Runs only on candidates.
        fmt_ok.loc[small] = cand.map(lambda s: _SCRIPT_TOKEN_RE.match(s) is not None)
    small = small & fmt_ok & ref_ok

    df.loc[small & has_ref & (shift >  threshold), "script_type"] = "superscript"
    df.loc[small & has_ref & (shift < -threshold), "script_type"] = "subscript"

    return df


# ================================================================================
# 6. CELL AGGREGATION  (words → cells)
# ================================================================================

def _build_cells_df(df_words: pd.DataFrame) -> pd.DataFrame:
    """Aggregate words into cells via the central column registry."""
    # Join words in content-stream order (text_object_id) when the native PDF
    # provides it — that is the order the text was emitted, a truer reading order
    # than geometry for reordered/overlapping glyphs. (line_id, x_left) tie-breaks
    # equal/null ids back to the geometric order. Absent the column we leave the
    # incoming order alone: horizontal words already arrive sorted (line_id,
    # x_left) from the gap-split, and vertical words arrive in their swap-built
    # top→bottom order where x_left is ~constant, so a geometric re-sort could
    # only scramble them. This reorders only the text/word_ids join — aggregate_to
    # geometry is order-independent, and the upstream x_left gap-split sorts that
    # cell segmentation depends on are untouched.
    # Form-value words borrow the text_object_id of the nearest content-stream
    # word so they slot near their label, but that neighbour is not necessarily
    # the logical last token of the cell. Force word_source == "form_value" to
    # sort last within each cell regardless of the borrowed id, so the filled
    # value trails the label text it belongs to (e.g. "Name: [John]"). A stable
    # sort keeps every other word's relative order intact.
    sort_cols: list[str] = []
    if "word_source" in df_words.columns:
        df_words = df_words.copy()
        df_words["_form_last"] = (df_words["word_source"] == "form_value").astype(int)
        sort_cols.append("_form_last")
    if "text_object_id" in df_words.columns:
        sort_cols += ["text_object_id", "line_id", "x_left"]
    if sort_cols:
        df_words = df_words.sort_values(
            sort_cols,
            kind="mergesort",
            na_position="last",
        )
    if "_form_last" in df_words.columns:
        df_words = df_words.drop(columns="_form_last")

    # Super/subscript markers ("[^…]"/"[_…]") applied per word, then joined
    # per cell with markers attaching to the previous token. dehyphenate keeps
    # words split across fragments ("inter-" + "national") joined directly.
    # bullet_sep breaks on unambiguous bullet glyphs (▪ • ► …) — tagged-PDF
    # TD/TH cells often pack several visual lines into one cell/line_id.
    fmt_text = apply_inline_markup(df_words)
    cell_text = merge_text_within_line(
        fmt_text, df_words["cell_id"], dehyphenate=True, bullet_sep="\n"
    )

    df_cells = aggregate_to(
        df_words,
        by="cell_id",
        overrides={
            "word_id": Agg.LIST,
            "line_id": Agg.SORTED_UNIQUE_LIST,
        },
    )
    df_cells = df_cells.rename(columns={"word_id": "word_ids", "line_id": "line_ids"})
    df_cells["text"] = df_cells["cell_id"].map(cell_text)
    # Scalar line_id for downstream consumers that expect a single value per cell.
    # Equals the minimum (first) line_id in the cell.
    if "line_ids" in df_cells.columns:
        df_cells["line_id"] = df_cells["line_ids"].apply(lambda ids: ids[0] if ids else None)
    return df_cells


# ================================================================================
# 7. VERTICAL WORD PROCESSING
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

    # BTT words read bottom→top: negate swapped x so ascending sort gives correct
    # reading order. Keep the negation all the way through cell assignment (which
    # re-sorts by x_left internally): restoring the positive x here would flip BTT
    # groups back to top→bottom, misaligning each row's precomputed gap_em_right
    # with its real successor (orphaning the last word) and numbering cells in
    # reverse. The swap-back below un-negates.
    if "text_orientation" in df.columns:
        btt_mask = df["text_orientation"] == "BTT"
        if btt_mask.any():
            # Compute both negated columns up front: .to_numpy() can alias df's
            # underlying block, so assigning x_left first would corrupt the value
            # x_right is derived from (leaving x_right un-negated).
            new_xl = -df.loc[btt_mask, "x_right"].to_numpy(dtype=float)
            new_xr = -df.loc[btt_mask, "x_left"].to_numpy(dtype=float)
            df.loc[btt_mask, "x_left"]  = new_xl
            df.loc[btt_mask, "x_right"] = new_xr

    # Annotate lines with em gap features in swapped space; skip classification —
    # every vertical line gets the neutral undetermined threshold.
    df = _annotate_line_features(df, config, classify=False)

    df, _ = _assign_cell_ids_horiz(df, config)
    if detect_scripts:
        df = _detect_cell_level_scripts(df)

    # Swap back to original coordinates (undo the x↔y swap). For BTT the swapped x
    # is still negated: x_left/x_right hold -orig_y_bottom / -orig_y_top, so negate
    # again to recover y_top/y_bottom.
    cur_xl = df["x_left"].to_numpy(dtype=float)
    cur_xr = df["x_right"].to_numpy(dtype=float)
    cur_yt = df["y_top"].to_numpy(dtype=float)
    cur_yb = df["y_bottom"].to_numpy(dtype=float)
    if "text_orientation" in df.columns:
        btt = (df["text_orientation"] == "BTT").to_numpy()
    else:
        btt = np.zeros(len(df), dtype=bool)
    df["x_left"]   = cur_yt
    df["x_right"]  = cur_yb
    df["y_top"]    = np.where(btt, -cur_xr, cur_xl)
    df["y_bottom"] = np.where(btt, -cur_xl, cur_xr)

    df["cell_id"] = df["cell_id"] + cell_id_offset

    df_vert_cells = _build_cells_df(df)
    df_vert_cells["block_type"] = "vertical_text"

    return df_vert_cells, df


# ================================================================================
# 8. CELL-ID RENUMBERING  (restore reading-order numbering)
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

    # Intra-line reading key: left→right (x_left) for horizontal words. Vertical
    # lines share one x-band, so x_left can't order them — read them along y
    # instead (TTB top→bottom = ascending y_top; BTT bottom→top = descending
    # y_bottom, encoded as -y_bottom so ascending sort still gives reading order).
    x_left = pd.to_numeric(df_words["x_left"], errors="coerce").to_numpy(float)
    if "text_orientation" in df_words.columns:
        orient = df_words["text_orientation"].to_numpy()
        yt = pd.to_numeric(df_words["y_top"],    errors="coerce").to_numpy(float)
        yb = pd.to_numeric(df_words["y_bottom"], errors="coerce").to_numpy(float)
        read_key = np.where(orient == "TTB", yt, np.where(orient == "BTT", -yb, x_left))
    else:
        read_key = x_left

    order  = (df_words.assign(_read_key=read_key)
                      .sort_values(["page_number", "line_id", "_read_key"], kind="mergesort"))
    codes  = pd.factorize(order["cell_id"].to_numpy())[0]          # 0-based, by first appearance
    mapping = dict(zip(order["cell_id"].to_numpy(), codes + 1))    # dense 1..N

    df_words = df_words.copy()
    df_words["cell_id"] = df_words["cell_id"].map(mapping).astype(np.int64)

    if not df_cells.empty and "cell_id" in df_cells.columns:
        df_cells = df_cells.copy()
        df_cells["cell_id"] = df_cells["cell_id"].map(mapping).astype(np.int64)

    return df_cells, df_words


# ================================================================================
# 9. PER-PAGE PIPELINE  (chunk worker; runs inline or in a process pool)
# ================================================================================

def _build_cells_pages(
    df_horiz: pd.DataFrame,
    df_vert: pd.DataFrame,
    config: CellBuildConfig = CONFIG,
    detect_scripts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the cell pipeline over one chunk of pages (both frames pre-filtered
    to the same pages).

    Each stage operates per line, per struct group, or per cell — never across
    pages — and line_ids are globally unique (assigned upstream in step 09), so
    the whole chunk runs through every stage in a single pass instead of page
    by page. That pays each stage's fixed pandas cost (merges, groupbys, mask
    filters) once per chunk rather than once per page. The one cross-page
    hazard, struct_group_id colliding between pages, is handled inside
    _merge_struct_groups by keying on (page_number, struct_group_id).

    Returns (horiz_words, vert_words, vert_cells). cell_ids are unique only
    within this call (they restart near 1), so a caller that fans pages out
    over several calls must re-offset each result before concatenating. The
    final _renumber_cells_reading_order pass then produces identical numbering
    regardless of how pages were chunked.

    Module-level (not a closure) so ProcessPoolExecutor can pickle it under
    the spawn start method.
    """
    df_h = df_horiz
    if not df_h.empty:
        df_h = _annotate_line_features(df_h, config)
        df_h, _ = _assign_cell_ids_horiz(df_h, config)
        df_h = _refine_multi_cell_lines(df_h, config)
        df_h = _merge_struct_groups(df_h)
        if detect_scripts:
            df_h = _detect_cell_level_scripts(df_h)
    else:
        df_h = pd.DataFrame()

    # Vertical cell_ids start above the horizontal ceiling so the chunk's
    # combined id space stays unique.
    horiz_cell_max = int(df_h["cell_id"].max()) if not df_h.empty else 0
    if not df_vert.empty:
        df_vert_cells, df_v = _process_vertical_words(
            df_vert, horiz_cell_max, config, detect_scripts
        )
    else:
        df_vert_cells, df_v = pd.DataFrame(), pd.DataFrame()

    return df_h, df_v, df_vert_cells


# ================================================================================
# 10. ENTRY POINT  (public API)
# ================================================================================

def build_cells(
    df_words: pd.DataFrame,
    config: CellBuildConfig = CONFIG,
    detect_scripts: bool = True,
    max_workers: int | None = None,
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

    # ── Per-page pipeline: inline for small documents, process pool above the
    # page threshold. The pipeline is CPU-bound Python that holds the GIL, so
    # threads don't help; pages are independent until the final renumber, so
    # page-chunk processes parallelise cleanly. Chunk results carry chunk-local
    # cell_ids and are re-offset here before concatenation; the reading-order
    # renumber below makes the final ids identical to a serial run.
    pages = sorted(df["page_number"].unique())
    n_workers = 1
    if len(pages) >= PARALLEL_PAGE_THRESHOLD:
        n_workers = resolve_worker_count(max_workers, n_items=len(pages))

    if n_workers > 1:
        h_parts:  list[pd.DataFrame] = []
        vw_parts: list[pd.DataFrame] = []
        vc_parts: list[pd.DataFrame] = []
        chunks = chunk_evenly(pages, n_workers)
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(
                        _build_cells_pages,
                        df_horiz[df_horiz["page_number"].isin(chunk)],
                        df_vert[df_vert["page_number"].isin(chunk)],
                        config, detect_scripts,
                    )
                    for chunk in chunks
                ]
                running_cell = 0
                for f in futures:                   # submit order == page order
                    h, vw, vc = f.result()
                    chunk_max = max(
                        (int(p["cell_id"].max()) for p in (h, vw) if not p.empty),
                        default=0,
                    )
                    for part, sink in ((h, h_parts), (vw, vw_parts), (vc, vc_parts)):
                        if not part.empty:
                            part["cell_id"] += running_cell
                            sink.append(part)
                    running_cell += chunk_max
        except BrokenProcessPool:
            warn_pool_fell_back("cell building")
            df_horiz_out, df_vert_out, df_vert_cells = _build_cells_pages(
                df_horiz, df_vert, config, detect_scripts
            )
        else:
            df_horiz_out  = _concat_frames(h_parts)
            df_vert_out   = _concat_frames(vw_parts)
            df_vert_cells = _concat_frames(vc_parts)
    else:
        df_horiz_out, df_vert_out, df_vert_cells = _build_cells_pages(
            df_horiz, df_vert, config, detect_scripts
        )

    # Aggregate words → cells once over the whole document: cell_ids are already
    # globally unique (offset per page/chunk above), and one aggregate_to call
    # avoids paying its fixed per-call overhead once per page. Keeping this in
    # the parent also pins font_size_ratio to the document median regardless of
    # worker count.
    df_cells = _build_cells_df(df_horiz_out) if not df_horiz_out.empty else pd.DataFrame()

    if not df_vert_cells.empty:
        df_cells = pd.concat([df_cells, df_vert_cells], ignore_index=True)

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

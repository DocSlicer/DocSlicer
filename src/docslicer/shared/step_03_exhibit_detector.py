"""
step_03_exhibit_detector.py

Answers the question: which lines form an Exhibit list (an SEC-filing exhibit
index), and which line is its heading?

Same paradigm as step_02_toc_detector.py (replaces the object-juggling detector
in step_02_exhibit_detector.py): each step is a vectorized pass that ADDS one or
more KPI columns to the incoming lines_df. Nothing is carried in bespoke per-row
dataclasses; the DataFrame is the single source of truth, and a final scorer
reads the accumulated KPI columns to decide what is an exhibit list vs. not.
All tunables live in the frozen ExhibitConfig below; the regexes themselves come
compiled from exhibit_patterns.yaml (ExhibitPatternConfig).

Pipeline:
  1. Mask hidden block types                  -> _hidden_block_mask (reused)
  2. Heading candidates     (KPI col)         -> exhibit_heading_candidate
  3. Row candidates         (KPI cols)        -> exhibit_row_candidate, ...
  4. Run segments           (KPI col)         -> exhibit_segment_id
  5. Scorer                 (KPI col)         -> exhibit_segment_score,
                                                 block_type = 'exhibits' / 'exhibit_heading'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .._utils.io.yaml_compilers.exhibit_patterns import (
    FOOTNOTE_MARKERS,
    ExhibitPatternConfig,
)
from .._utils.text_utils import is_numeric_value


# ================================================================================
# CONFIG
# ================================================================================


@dataclass(frozen=True)
class ExhibitConfig:
    # ---- step 2: heading candidates ----
    max_heading_chars: int = 150    # a heading line is short; longer lines that merely start
                                    # with "Item 15. ... Exhibits" are prose, not a heading
    # ---- step 3: row candidates ----
    max_row_chars: int = 500        # exhibit rows carry long descriptions ("Certification of
                                    # the CEO pursuant to ..."); longer => prose paragraph
    # ---- step 4: run segments ----
    max_bridge_lines: int = 5       # non-candidate, non-link lines a run may skip over between
                                    # two candidate rows and still continue as the same segment
    max_link_bridge_lines: int = 10 # total gap allowed when the extra lines carry links — a
                                    # long exhibit description wraps over many lines, each
                                    # hyperlinked to the filing, before the next exhibit number
    min_table_candidates: int = 2   # candidate rows a standalone table needs to become a segment;
                                    # one accidental hit is not enough
    min_table_candidate_ratio: float = 0.5  # AND at least this fraction of the table's rows must
                                            # be candidates — 2 hits in a 40-row data table don't count
    # ---- step 5: scorer ----
    nearby_lookback: int = 5        # rows above a segment's first line searched for an exhibit
                                    # heading; the heading may also BE the segment's first line
                                    # (a table's own title row, pulled in by whole-table grouping)
    min_score: float = 1.0          # acceptance threshold: currently only the heading signal (+1)
                                    # exists, so no heading nearby => rejected


# Block types that can never be part of an exhibit list and are excluded from
# every KPI pass. TOC detection (step_02) runs BEFORE this step, so 'toc' and
# 'toc_heading' are real exclusions here, not defensive ones — a TOC row like
# "3.1 Business Overview .... 12" would otherwise also match the weak
# numeric_or_dotted exhibit pattern. Note: 'table' is deliberately NOT hidden —
# an exhibit list is very often laid out as a table.
_HIDDEN_BLOCK_TYPES = frozenset({
    # Page furniture
    "image", "hr", "page_label", "navigation", "vertical_text",
    # Out-of-flow / peripheral regions
    "header", "footer", "footnote", "endnote", "comment", "speaker_notes",
    # Non-prose content
    "chart", "shape", "math",
    # Set by earlier shared steps
    "toc", "toc_heading",
    # Already-classified (defensive; not set at this stage)
    "exhibits", "exhibit_heading",
})

# Footnote marker glyphs stripped out of a row's leading token before it is
# reported as the exhibit number ("*10.3" / "10.1†" -> "10.3" / "10.1").
_MARKER_STRIP_RE = re.compile(f"[{re.escape(FOOTNOTE_MARKERS)}]+")

# Any letter (unicode-aware): the cheap prefilter for the numeric-table-row
# guard — a line carrying a letter has title text and is never all-numeric.
_ANY_LETTER_RE = re.compile(r"[^\W\d_]")

# NOTE on regex engine: every pattern is passed to the .str ops as a compiled
# re.Pattern (pattern.compiled), never as a raw string. This forces pandas onto
# Python's re engine, which the exhibit patterns require: exhibit_prefix_row
# uses a lookahead and the _with_markers variants carry non-ASCII glyphs —
# neither survives pandas 3's default RE2/ASCII string engine.


# ================================================================================
# STEP 1: Mask hidden block types
# ================================================================================

def _hidden_block_mask(df: pd.DataFrame) -> pd.Series:
    """
    Boolean row mask: True where the line's block_type can never be part of an
    exhibit list (see _HIDDEN_BLOCK_TYPES) — page furniture plus the TOC lines
    stamped by step_02.

    Derived on demand rather than persisted — every later KPI pass calls this to
    exclude such lines before writing its column. Returns all-False when there
    is no block_type column.
    """
    if "block_type" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    bt = df["block_type"].astype("string").str.strip().str.lower()
    return bt.isin(_HIDDEN_BLOCK_TYPES).fillna(False).astype(bool)


# ================================================================================
# STEP 2: Heading candidates  (KPI col: exhibit_heading_candidate)
# ================================================================================

def add_exhibit_heading_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    config: ExhibitConfig = ExhibitConfig(),
) -> pd.DataFrame:
    """
    Add the KPI column ``exhibit_heading_candidate`` (str | NA): the name of the
    first heading pattern the line matches ("item_any_exhibits",
    "exhibit_index", ...), NA everywhere else.

    A line is a heading candidate when:
      - its text matches one of the compiled ``header_patterns`` from the YAML
        ("Item 15. Exhibits...", "EXHIBIT INDEX", "(d) Exhibits."), AND
      - the line is short (<= config.max_heading_chars), AND
      - the line is not a hidden block type.

    First match wins, in YAML order — one vectorized str.match per pattern
    (~5 patterns), each restricted to the rows still unmatched.
    """
    out = df.copy()
    out["exhibit_heading_candidate"] = pd.NA

    if "text" not in out.columns:
        return out

    stripped = out["text"].astype("string").fillna("").str.strip()
    eligible = (
        (stripped.str.len() > 0)
        & (stripped.str.len() <= config.max_heading_chars)
        & ~_hidden_block_mask(out)
    )

    result = pd.Series(pd.NA, index=out.index, dtype="string")
    for pattern in exhibit_config.header_patterns:
        todo = eligible & result.isna()
        if not todo.any():
            break
        hits = todo & stripped.str.match(pattern.compiled, na=False)
        result[hits] = pattern.name

    out["exhibit_heading_candidate"] = result
    return out


# ================================================================================
# STEP 3: Row candidates  (KPI cols: exhibit_row_candidate, _number, _strength)
# ================================================================================

def add_exhibit_row_candidates(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    config: ExhibitConfig = ExhibitConfig(),
) -> pd.DataFrame:
    """
    Add the KPI columns:
      - ``exhibit_row_candidate`` (str | NA)  name of the matched row pattern
      - ``exhibit_row_number``    (str | NA)  the exhibit number ("10.1", "(c)(1)", "EX-101.PRE")
      - ``exhibit_row_strength``  (str | NA)  the pattern's strength ("strong" / "weak")

    A line is a row candidate when its text matches one of the compiled
    ``row_patterns`` from the YAML ("3.1 Certificate of Incorporation",
    "(c)(1) Opinion of ...", "Exhibit 31.1 ..."), under the usual gates:
    non-empty, <= config.max_row_chars, not a hidden block type.

    Numeric table rows — only for lines inside a detected table (table_id
    set) — are rejected before matching: a letterless line of two or more
    numeric/currency tokens ("150,251 7,330 4.88 133,805 6,239 4.66") is a
    financial data row, not an exhibit row, even though its leading token
    matches numeric_or_dotted or hundred_series_exhibit. A lone number ("104",
    an exhibit-number cell) survives, as does any line carrying a letter.
    Lines without a table_id skip this guard entirely: the table builder
    already ruled them out as table rows.

    First match wins, in compiled order — the pattern compiler emits each base
    pattern's footnote-marker variants ("*10.3", "10.1†") BEFORE the base
    pattern, so marker-bearing rows land on the ``_with_markers`` variant
    (always strong). One vectorized str.match per pattern, each restricted to
    the rows still unmatched.

    The exhibit number is the pattern's ``code`` group where it has one
    (exhibit_prefix_row, hundred_series_exhibit); otherwise the line's first
    whitespace token with footnote markers stripped.
    """
    out = df.copy()
    out["exhibit_row_candidate"] = pd.NA
    out["exhibit_row_number"] = pd.NA
    out["exhibit_row_strength"] = pd.NA

    if "text" not in out.columns:
        return out

    stripped = out["text"].astype("string").fillna("").str.strip()
    eligible = (
        (stripped.str.len() > 0)
        & (stripped.str.len() <= config.max_row_chars)
        & ~_hidden_block_mask(out)
    )

    # ---- numeric-table-row guard (in-table lines only) ----
    # Cheap vectorized prefilter (in a table, no letter anywhere), then the
    # per-token check on the few surviving rows. Pipes are treated as
    # whitespace so piped table text tokenizes into its cells.
    if "table_id" in out.columns:
        tid = out["table_id"]
        in_table = (tid.notna() & tid.astype("string").str.strip().ne("")).fillna(False)
        check = eligible & in_table & ~stripped.str.contains(_ANY_LETTER_RE, na=False)
        if check.any():
            toks = stripped[check].str.replace("|", " ", regex=False).str.split()
            numeric_row = toks.map(
                lambda ts: len(ts) >= 2 and all(is_numeric_value(t) for t in ts)
            ).astype(bool)
            eligible[check] = ~numeric_row

    name = pd.Series(pd.NA, index=out.index, dtype="string")
    strength = pd.Series(pd.NA, index=out.index, dtype="string")
    number = pd.Series(pd.NA, index=out.index, dtype="string")

    for pattern in exhibit_config.row_patterns:
        todo = eligible & name.isna()
        if not todo.any():
            break
        hits = todo & stripped.str.match(pattern.compiled, na=False)
        if not hits.any():
            continue
        name[hits] = pattern.name
        strength[hits] = pattern.strength
        # Patterns with a named 'code' group carry the exhibit number in it
        # ("Exhibit 31.1 ..." -> "31.1"); index-aligned assignment.
        if "code" in pattern.compiled.groupindex:
            number[hits] = stripped[hits].str.extract(pattern.compiled)["code"]

    # Remaining numbers: the line's first whitespace token, footnote markers
    # stripped ("*10.3 Credit Agreement" -> "10.3", "(c)(1) Opinion" -> "(c)(1)").
    fallback = name.notna() & number.isna()
    if fallback.any():
        first_tok = (
            stripped[fallback]
            .str.replace(_MARKER_STRIP_RE, "", regex=True)
            .str.strip()
            .str.split()
            .str[0]
        )
        number[fallback] = first_tok

    out["exhibit_row_candidate"] = name
    out["exhibit_row_number"] = number
    out["exhibit_row_strength"] = strength
    return out


# ================================================================================
# STEP 4: Run segments  (KPI col: exhibit_segment_id)
# ================================================================================

def _truthy_bool(series: pd.Series) -> pd.Series:
    """Robust boolean coercion for flag columns (bool / 0-1 / 'TRUE' strings)."""
    def conv(v: object) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes"}
        try:
            return bool(v)
        except Exception:
            return False
    return series.map(conv, na_action="ignore").fillna(False).astype(bool)


def add_exhibit_run_segments(
    df: pd.DataFrame,
    config: ExhibitConfig = ExhibitConfig(),
) -> pd.DataFrame:
    """
    Add the KPI column ``exhibit_segment_id`` (int | NA): groups exhibit rows
    into contiguous segments, top to bottom.

    Unlike the TOC detector (where runs are held together by a separate layout
    signal), the members here are the ``exhibit_row_candidate`` rows themselves;
    what varies is how far a run may bridge from one candidate to the next.
    Two consecutive candidate rows belong to the same segment when the gap
    between them satisfies BOTH:

      - at most ``config.max_link_bridge_lines`` lines in total, AND
      - at most ``config.max_bridge_lines`` of them WITHOUT a link
        (``has_link`` falsy).

    So 5 plain lines bridge, 10 linked lines bridge (a long exhibit description
    wrapping over many hyperlinked lines before the next exhibit number — the
    dominant layout in linked filings), and a mix passes as long as the plain
    lines stay within their own budget. Hidden block types count like plain
    lines. A segment runs from its first candidate to its last; every line in
    that span (bridged ones included) carries the segment id. Segment ids are
    0, 1, 2, ... in document order.

    Tables are then grouped in whole (the table boundary is trusted as a
    segmentation signal, same as the TOC detector) — but since the members
    here ARE the candidates, any table containing candidates is automatically
    touched by its own run, so absorption must be gated or the count/ratio bar
    would never protect anything. A run absorbs a table's remaining rows only
    when:
      - the table holds at least ``config.min_table_candidates`` candidates
        AND candidates make up at least ``config.min_table_candidate_ratio``
        of its rows (a genuine exhibit table), OR
      - the run extends beyond the table (an exhibit list flowing into/out of
        it — the table is part of a larger list even if mostly description
        rows).
    A run trapped inside a table that fails the bar keeps only its own rows —
    a couple of accidental hits in a big data table stay a tiny naked segment
    for the scorer to discard, instead of dragging the whole table in.
    """
    out = df.copy()
    out["exhibit_segment_id"] = pd.NA

    if out.empty or "exhibit_row_candidate" not in out.columns:
        return out

    cand = (out["exhibit_row_candidate"].notna() & ~_hidden_block_mask(out)).to_numpy()
    cand_pos = cand.nonzero()[0]

    if "has_link" in out.columns:
        has_link = _truthy_bool(out["has_link"]).to_numpy()
    else:
        has_link = np.zeros(len(out), dtype=bool)

    seg_ids = pd.array([pd.NA] * len(out), dtype="Int64")
    next_id = 0

    if cand_pos.size:
        # Per-gap totals between consecutive candidates, via a cumsum of the
        # no-link flag: gap = rows strictly between the two candidates.
        nolink_cum = np.concatenate(([0], np.cumsum(~has_link)))
        gap_total = cand_pos[1:] - cand_pos[:-1] - 1
        gap_nolink = nolink_cum[cand_pos[1:]] - nolink_cum[cand_pos[:-1] + 1]
        bridged = (
            (gap_total <= config.max_link_bridge_lines)
            & (gap_nolink <= config.max_bridge_lines)
        )

        run_starts = [0] + [i + 1 for i in (~bridged).nonzero()[0]]
        for lo, hi in zip(run_starts, run_starts[1:] + [cand_pos.size]):
            start, end = cand_pos[lo], cand_pos[hi - 1]
            seg_ids[start : end + 1] = next_id
            next_id += 1

    seg = pd.Series(seg_ids, index=out.index, dtype="Int64")

    # ---- whole-table grouping ----
    if "table_id" in out.columns:
        tid = out["table_id"].astype("string").str.strip()
        has_tid = (tid.notna() & tid.ne("")).fillna(False)
        if has_tid.any():
            tids = tid[has_tid]

            # First segment touching each table (rows already in a segment
            # always keep theirs; absorption only fills the table's NA rows).
            table_seg = seg[has_tid].groupby(tids).first().dropna()
            if not table_seg.empty:
                # Bar: enough candidates AND a high enough candidate ratio.
                cand_s = pd.Series(cand, index=out.index)
                table_n_cand = cand_s[has_tid].groupby(tids).sum()
                table_n_rows = tids.groupby(tids).size()
                passes_bar = (
                    (table_n_cand >= config.min_table_candidates)
                    & (table_n_cand / table_n_rows >= config.min_table_candidate_ratio)
                )

                # Does the touching segment extend beyond the table? Compare
                # the segment's total row count with how many of its rows sit
                # inside this table.
                seg_sizes = seg[seg.notna()].groupby(seg[seg.notna()]).size()
                touching = tids.map(table_seg)
                rows_of_seg_in_table = (
                    (seg[has_tid] == touching).groupby(tids).sum()
                )
                extends_outside = (
                    table_seg.index.to_series().map(table_seg).map(seg_sizes)
                    > rows_of_seg_in_table[table_seg.index]
                )

                absorb = table_seg.index[
                    passes_bar[table_seg.index] | extends_outside
                ]
                if len(absorb):
                    seg[has_tid] = seg[has_tid].fillna(
                        tids.map(table_seg[absorb])
                    )

    out["exhibit_segment_id"] = seg
    return out


# ================================================================================
# STEP 5: Scorer  (KPI col: exhibit_segment_score; writes block_type)
# ================================================================================

# Scoring weights. Only one signal exists today — the heading is the whole
# ballgame (an exhibit list without its heading is indistinguishable from any
# other numbered list) — but the additive structure leaves room for more.
_SCORE_HEADING_NEARBY = 1.0   # exhibit heading candidate just above (or first line of) the segment


def score_exhibit_segments(
    df: pd.DataFrame,
    config: ExhibitConfig = ExhibitConfig(),
) -> pd.DataFrame:
    """
    Score every ``exhibit_segment_id`` group and stamp the winners:
    ``block_type = 'exhibits'`` on the segment's rows (hidden furniture
    excluded) and ``'exhibit_heading'`` on the heading line(s) that anchored
    it. Adds the KPI columns ``exhibit_segment_score`` (float | NA, broadcast
    to the segment's rows) and ``exhibit_segment_score_detail`` (str | NA,
    which signals fired) for harness inspection.

    Scoring is currently a single signal:

      +1  exhibit heading candidate within ``config.nearby_lookback`` rows
          above the segment's first line, or AS the first line (a table's own
          title row, absorbed by whole-table grouping)

    Accepted when score >= ``config.min_score`` — i.e. today: heading nearby
    -> accept, no heading nearby -> reject.
    """
    out = df.copy()
    out["exhibit_segment_score"] = pd.NA
    out["exhibit_segment_score_detail"] = pd.NA
    if "block_type" not in out.columns:
        out["block_type"] = pd.NA

    if out.empty or "exhibit_segment_id" not in out.columns:
        return out

    seg = out["exhibit_segment_id"]
    has_seg = seg.notna()
    if not has_seg.any():
        return out

    keys = seg[has_seg]
    pos = pd.Series(np.arange(len(out)), index=out.index)
    start_pos = pos[has_seg].groupby(keys).min()

    heading_mask = (
        out["exhibit_heading_candidate"].notna()
        if "exhibit_heading_candidate" in out.columns
        else pd.Series(False, index=out.index)
    )
    heading_pos = np.flatnonzero(heading_mask.to_numpy())

    # Heading in [start - lookback, start]: the +1 upper bound admits the
    # segment's own first line.
    starts = start_pos.to_numpy()
    lo = starts - config.nearby_lookback
    heading_nearby = pd.Series(
        np.searchsorted(heading_pos, lo) < np.searchsorted(heading_pos, starts + 1),
        index=start_pos.index,
    )

    score = _SCORE_HEADING_NEARBY * heading_nearby.astype(float)
    accepted = score >= config.min_score

    detail = heading_nearby.map(
        lambda hit: f"heading{_SCORE_HEADING_NEARBY:+g}" if hit else ""
    )
    out.loc[has_seg, "exhibit_segment_score"] = keys.map(score)
    out.loc[has_seg, "exhibit_segment_score_detail"] = keys.map(detail)

    # ---- stamp winners ----
    hidden = _hidden_block_mask(out)
    accepted_ids = set(accepted.index[accepted])
    if accepted_ids:
        out.loc[has_seg & seg.isin(accepted_ids) & ~hidden, "block_type"] = "exhibits"

        # Heading candidates in each accepted segment's window ('exhibit_heading'
        # wins over the 'exhibits' just stamped on a first-line heading).
        heading_iloc: List[int] = []
        for seg_id in accepted_ids:
            s = int(start_pos.at[seg_id])
            window = heading_pos[(heading_pos >= s - config.nearby_lookback) & (heading_pos <= s)]
            heading_iloc.extend(int(p) for p in window)
        if heading_iloc:
            out.loc[out.index[heading_iloc], "block_type"] = "exhibit_heading"

    return out


# ================================================================================
# PUBLIC API
# ================================================================================

def detect_exhibits(
    df: pd.DataFrame,
    exhibit_config: ExhibitPatternConfig,
    config: ExhibitConfig = ExhibitConfig(),
) -> pd.DataFrame:
    """
    Run the exhibit detection pipeline and return the annotated DataFrame.

    Each step adds its KPI column(s) to a copy of the input; the frame is threaded
    through the steps and returned to the caller unchanged apart from the added
    columns. Empty input is returned untouched.

    Implements:
      2. exhibit_heading_candidate
      3. exhibit_row_candidate      (+ exhibit_row_number / exhibit_row_strength)
      4. exhibit_segment_id         (bridged candidate runs, whole-table grouping)
      5. exhibit_segment_score      + block_type = 'exhibits' / 'exhibit_heading'

    (Step 1, the hidden-block mask, is applied inside each KPI pass rather than
    persisted as a column.)
    """
    if df.empty:
        return df

    out = add_exhibit_heading_candidates(df, exhibit_config, config)
    out = add_exhibit_row_candidates(out, exhibit_config, config)
    out = add_exhibit_run_segments(out, config)
    out = score_exhibit_segments(out, config)
    return out

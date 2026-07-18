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
  3. Heading candidates    (KPI col)             -> toc_heading_candidate
  4. Dot leaders           (KPI col)             -> toc_has_dot_leaders
  5. Row candidates        (KPI cols)            -> toc_row_candidate, ...
  6. Run segments          (KPI col)             -> toc_segment_id
  7. Scorer                                      -> block_type = 'toc' / 'toc_heading'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._utils.io.yaml_compilers.page_label_patterns import PageLabelPatternConfig
from .._utils.text_utils import is_numeric_value


# ================================================================================
# CONFIG
# ================================================================================


@dataclass(frozen=True)
class TocConfig:
    # ---- step 3: heading candidates ----
    max_heading_chars: int = 80     # a heading line is short; longer lines that merely start
                                    # with "Contents" are prose, not a TOC heading
    # ---- step 5: row candidates ----
    max_row_chars: int = 250        # dot-leaders excluded; longer => prose paragraph, not a TOC row
    max_row_cells: int = 5          # more cells than this => a table row, not a TOC entry
    # ---- step 6: run segments ----
    max_bridge_lines: int = 3       # non-member lines a run may skip over (page labels, unlinked
                                    # group titles) and still continue as the same segment
    min_table_candidates: int = 2   # candidate rows a standalone table needs to become a segment;
                                    # one accidental hit ("Total assets 118") is not enough
    min_table_candidate_ratio: float = 0.5  # AND at least this fraction of the table's rows must be
                                            # candidates — 2 hits in a 40-row data table don't count
    # ---- step 7: scorer ----
    min_segment_rows: int = 4       # smaller segments are discarded without scoring
    min_score: float = 2.0          # acceptance threshold: a leader/link majority (+5) passes
                                    # alone; heading nearby (+2) needs one corroborating signal
    nearby_lookback: int = 3        # rows above a segment's first line searched for a TOC
                                    # heading / "Page" column header


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
    "toc", "toc_heading",
})

# TOC heading anchors.
#
# The full phrases "table of contents/figures/tables" are unmistakable, so
# they match in any case, anywhere in the line (the max_heading_chars gate
# still rejects prose).
# The single-word anchors stay CASE-SENSITIVE on purpose: they must be
# Title-case ("Contents", "Index") or ALL-CAPS ("CONTENTS", "INDEX"). A
# lowercase mid-sentence "contents, ..." or "... index of ..." is prose, not a
# heading, and must NOT match. Requiring the leading letter to be uppercase
# (via explicit case alternation) is what enforces this.
_TOC_HEADING_PATTERNS = (
    # "table of contents/figures/tables" in any case, anywhere in the line
    re.compile(r"table\s+of\s+(?:contents?|figures?|tables?)", re.IGNORECASE),
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

# A line that is exactly a page-column header — "Page", "Pages", "Page No.",
# "Page Number(s)", "Page #" — the tell of a page-number listing / TOC-shaped
# table. Nothing else may be on the line.
_PAGE_HEADER_RE = re.compile(r"^\s*pages?\s*(?:no\.?|numbers?|#)?\s*$", re.IGNORECASE)

# Scoring weights (step 7). Deliberately few and uncapped: the majority signals
# are ratios (can't grow with segment size) and the rest are booleans.
_SCORE_LEADER_MAJORITY    = 5.0   # majority of rows carry dot leaders
_SCORE_LINK_MAJORITY      = 5.0   # majority of rows carry an internal link
_SCORE_HEADING_NEARBY     = 2.0   # TOC heading candidate just above the segment
_SCORE_PAGE_HEADER_NEARBY = 1.0   # "Page(s)" line just above / first line of the segment
_SCORE_MIXED_PAGE_TYPES   = -1.0  # candidates mix page-label types ('arabic' + 'roman' + ...)
_SCORE_BIG_TABLE          = 1.0   # more than 5 candidate rows sit inside tables

# Punctuation stripped from a token before page-label classification, so
# "(12)", "12.", "[iv]" classify as their bare token.
_TOKEN_STRIP = "([{.,;:)]}\"'“”‘’"

# Dot-leader runs are replaced by this standalone token (not removed) before
# tokenizing, so the leader still breaks "consecutive numeric items" adjacency:
# in "Sections 1 and 2 ...... 35" it separates the title's trailing "2" from
# the page number, while a table row's "... 7,640 23" stays adjacent and is
# rejected. Private-use codepoint: never occurs in real text, classifies as no
# page label, carries no cased letter.
_LEADER_TOKEN = "\uf8ff"


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


def _is_numeric_like(tok: str, typ: str) -> bool:
    """
    Token-like item of a numeric table row: a page-label token or a
    numeric/currency value ("7,640", "190.9", "(21)", "0.5%", "—").
    """
    return typ != "unknown" or is_numeric_value(tok)


def _leader_page_token(
    toks: List[str],
    cfg: PageLabelPatternConfig,
    cache: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Page label of a leader-fast-tracked row: the boundary token (leader
    stand-ins excluded) that classifies as a page label, trailing side first.
    Returns (page_token, page_type), (None, None) when neither boundary matches.
    """
    core = [t for t in toks if t != _LEADER_TOKEN]
    if not core:
        return (None, None)
    for tok in (core[-1], core[0]):
        typ = _classify_token(tok, cfg, cache)
        if typ != "unknown":
            return (tok.strip().strip(_TOKEN_STRIP), typ)
    return (None, None)


def _eval_toc_row(
    toks: List[str],
    cfg: PageLabelPatternConfig,
    cache: Dict[str, str],
    numeric_guard: bool,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Decide whether a tokenized line is a TOC row and, if so, which boundary token
    is its page label. Accepts BOTH layouts:
        title ... page_label      (label at end)
        page_label ... title      (label at start)

    With ``numeric_guard`` (lines inside a detected table), a numeric table row
    is rejected by adjacency at the label's boundary: its right end is a run of
    consecutive numeric items ("... 7,640 23"), whereas a TOC row's page number
    sits alone next to title text or a dot-leader run (`_LEADER_TOKEN`, which is
    neither a label nor numeric and so breaks the run). Outside a table the
    guard is skipped — trailing numbers there are title text ("Sections 1 and
    2 42"), not columns.

    Returns (is_candidate, page_token, page_type).
    """
    n = len(toks)
    if n < 2:
        return (False, None, None)

    types = [_classify_token(t, cfg, cache) for t in toks]

    # Label at end: inside a table the item before it must NOT also be
    # numeric-like (table columns), and the title side must contain a real word.
    if (
        types[-1] != "unknown"
        and (not numeric_guard or not _is_numeric_like(toks[-2], types[-2]))
        and _title_has_word(toks[:-1], types[:-1])
    ):
        return (True, toks[-1].strip().strip(_TOKEN_STRIP), types[-1])

    # Label at start (the picture's "02  At a glance" layout).
    if (
        types[0] != "unknown"
        and (not numeric_guard or not _is_numeric_like(toks[1], types[1]))
        and _title_has_word(toks[1:], types[1:])
    ):
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

    Dot-leaders fast track: any non-hidden line containing a dot-leader run is a
    candidate outright — the leaders ARE the TOC-row layout, no further gates
    apply. Its page label is whichever boundary token classifies (trailing side
    first); NA when neither does.

    Every other line is a TOC row candidate when it starts OR ends with a clean
    page-label token and carries real title text. Pipes are treated as
    whitespace, so piped table text tokenizes into its cells. Rejected:
      - lines that neither start nor end with a page-label token ("US investment plans")
      - currency-bearing lines ("Total net revenue $ 182,447 ...")
      - lines with more than config.max_row_cells cells (table rows)
      - numeric table rows — only for lines inside a detected table (table_id
        set) — via the numeric item adjacent to the page label
        ("Selling and administrative | 165.4 | 190.9 | 406.3 | 357.6 | 287.4",
        "Income from continuing operations 3,550 4,481 (21) 9,427 7,640 23");
        a dot-leader run between title and page number breaks that adjacency,
        so titles containing numbers ("Sections 1 and 2 ...... 35") survive.
        Lines without a table_id skip this guard entirely: the table builder
        already ruled them out as table rows, so trailing numbers are title text
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

    # Pipes become whitespace (piped table text tokenizes into its cells) and
    # each dot-leader run becomes a standalone _LEADER_TOKEN — kept as a token
    # so the page number is a clean boundary token AND the leader still breaks
    # numeric adjacency in _eval_toc_row. Then collapse whitespace.
    stripped = (
        text.str.replace("|", " ", regex=False)
            .str.replace(_DOT_LEADERS_RE, f" {_LEADER_TOKEN} ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )
    toks_series = stripped.str.split()
    n_tok = toks_series.str.len().fillna(0)

    # Row length is measured without the leader stand-ins (dot-leaders excluded).
    row_chars = stripped.str.replace(_LEADER_TOKEN, "", regex=False).str.len()

    # ---- dot-leaders fast track ----
    # A dot-leader run IS the TOC-row layout: any non-hidden line carrying one
    # is a candidate outright, no further gates. Requires something besides the
    # leaders themselves (a pure "......" line is not a row). Reuses the step-4
    # column when present; recomputed otherwise so this pass stays standalone.
    if "toc_has_dot_leaders" in out.columns:
        has_leaders = out["toc_has_dot_leaders"].fillna(False).astype(bool)
    else:
        has_leaders = text.str.contains(_DOT_LEADERS_RE, regex=True, na=False)
    fast_track = has_leaders & ~_hidden_block_mask(out) & (row_chars > 0)

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
        & (row_chars <= config.max_row_chars)
        & ~text.str.contains(_CURRENCY_RE, regex=True, na=False)
        & ~_hidden_block_mask(out)
        & ~fast_track
    )
    if "cell_count" in out.columns:
        prefilter &= out["cell_count"].fillna(0) <= config.max_row_cells

    # The numeric-adjacency guard only applies inside detected tables: a blank
    # table_id means the table builder already ruled the line out as a table row.
    if "table_id" in out.columns:
        tid = out["table_id"]
        in_table = (tid.notna() & tid.astype("string").str.strip().ne("")).fillna(False)
    else:
        in_table = pd.Series(False, index=out.index, dtype=bool)

    # ---- detailed per-row check on the reduced set ----
    toks_list = toks_series.tolist()
    positions = list(out.index[prefilter])
    pos_to_iloc = {idx: i for i, idx in enumerate(out.index)}

    cand_idx: List = []
    tokens_out: List = []
    types_out: List = []
    for idx in out.index[fast_track]:
        tok, typ = _leader_page_token(toks_list[pos_to_iloc[idx]], page_label_config, cache)
        cand_idx.append(idx)
        tokens_out.append(tok if tok is not None else pd.NA)
        types_out.append(typ if typ is not None else pd.NA)

    for idx in positions:
        toks = toks_list[pos_to_iloc[idx]]
        ok, tok, typ = _eval_toc_row(toks, page_label_config, cache, bool(in_table.at[idx]))
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
# STEP 6: Run segments  (KPI col: toc_segment_id)
# ================================================================================

def add_toc_run_segments(df: pd.DataFrame, config: TocConfig = TocConfig()) -> pd.DataFrame:
    """
    Add the KPI column ``toc_segment_id`` (int | NA): groups TOC-layout rows
    into contiguous segments, top to bottom.

    A run MEMBER is a non-hidden line carrying either TOC-row layout signal:
      - a dot-leader run (``toc_has_dot_leaders``), or
      - an internal hyperlink (``link_type == 'internal'``).

    A segment is seeded by a member that is also a ``toc_row_candidate`` and
    only ever grows DOWNWARD: subsequent member lines join the same segment as
    long as at most ``config.max_bridge_lines`` non-member lines sit between
    consecutive members (bridging page labels, unlinked group titles like
    "Financial Review", and other stray lines that interrupt the run). The
    segment ends at the LAST member line of the run — trailing bridged lines
    are never included. Every line inside the span (bridged ones included)
    carries the segment id; hidden block types don't count as members but can
    be bridged over.

    Runs whose member lines contain no row candidate at all produce no
    segment. Segment ids are 0, 1, 2, ... in document order.

    Tables are grouped in whole (the table boundary is trusted as a
    segmentation signal, like the old detector's table-based segments):
      - a run segment that touches ANY row of a table absorbs the table's
        remaining rows into itself;
      - a table untouched by any run becomes its own segment when at least
        ``config.min_table_candidates`` of its rows are ``toc_row_candidate``
        AND candidates make up at least ``config.min_table_candidate_ratio``
        of the table — this catches TOCs laid out as plain tables with neither
        dot leaders nor internal links, while a couple of accidental
        page-label-shaped rows in a big data table do not promote it. Tables
        below either bar are left alone (they die here, not in the scorer).

    Leftover candidates — claimed by no run and no table — are grouped the
    plain way: strictly consecutive candidate rows share one id, a stray
    single row gets its own. No bridging: one non-candidate line ends the
    group. The scorer separates real TOCs from noise among these.
    """
    out = df.copy()
    out["toc_segment_id"] = pd.NA

    if out.empty or "toc_row_candidate" not in out.columns:
        return out

    member = pd.Series(False, index=out.index, dtype=bool)
    if "toc_has_dot_leaders" in out.columns:
        member |= out["toc_has_dot_leaders"].fillna(False).astype(bool)
    if "link_type" in out.columns:
        link_type = out["link_type"].astype("string").str.strip().str.lower()
        member |= link_type.eq("internal").fillna(False)
    member &= ~_hidden_block_mask(out)

    cand = out["toc_row_candidate"].fillna(False).astype(bool).to_numpy()
    member_pos = member.to_numpy().nonzero()[0]

    # Split member positions into runs: a gap of more than max_bridge_lines
    # non-member lines between consecutive members breaks the run. No members
    # means no runs — but the whole-table grouping below must still happen.
    if member_pos.size:
        breaks = (member_pos[1:] - member_pos[:-1]) > (config.max_bridge_lines + 1)
        run_starts = [0] + [i + 1 for i in breaks.nonzero()[0]]
        run_bounds = list(zip(run_starts, run_starts[1:] + [member_pos.size]))
    else:
        run_bounds = []

    seg_ids = pd.array([pd.NA] * len(out), dtype="Int64")
    next_id = 0
    for lo, hi in run_bounds:
        run = member_pos[lo:hi]
        # The segment starts at the run's first member that is a row candidate
        # (never looks upward past it) and ends at the run's last member.
        seed_hits = run[cand[run]]
        if seed_hits.size == 0:
            continue
        start, end = seed_hits[0], run[-1]
        seg_ids[start : end + 1] = next_id
        next_id += 1

    seg = pd.Series(seg_ids, index=out.index, dtype="Int64")

    # ---- whole-table grouping ----
    if "table_id" in out.columns:
        tid = out["table_id"].astype("string").str.strip()
        has_tid = (tid.notna() & tid.ne("")).fillna(False)
        if has_tid.any():
            tids = tid[has_tid]

            # A run segment that touches any row of a table absorbs the whole
            # table: remaining rows of that table inherit the segment id (first
            # touching segment wins; rows already in a segment keep theirs).
            table_seg = seg[has_tid].groupby(tids).first().dropna()
            if not table_seg.empty:
                seg[has_tid] = seg[has_tid].fillna(tids.map(table_seg))

            # A table untouched by any run becomes its own segment when it
            # contains at least min_table_candidates row candidates AND
            # candidates make up min_table_candidate_ratio of its rows (a few
            # accidental hits in a big data table don't promote it).
            cand_s = pd.Series(cand, index=out.index)
            table_n_cand = cand_s[has_tid].groupby(tids).sum()
            table_n_rows = tids.groupby(tids).size()
            table_cand = (
                (table_n_cand >= config.min_table_candidates)
                & (table_n_cand / table_n_rows >= config.min_table_candidate_ratio)
            )
            standalone = table_cand.index[table_cand & ~table_cand.index.isin(table_seg.index)]
            if len(standalone):
                # Number in document order (first occurrence of each table).
                ordered = [t for t in pd.unique(tids) if t in set(standalone)]
                new_ids = {t: next_id + i for i, t in enumerate(ordered)}
                next_id += len(ordered)
                seg[has_tid] = seg[has_tid].fillna(tids.map(new_ids))

    # ---- leftover candidates ----
    # Row candidates not claimed by any run or table segment are grouped the
    # plain way: strictly consecutive candidate rows share one id, a stray
    # single row gets its own. No bridging here — one non-candidate line ends
    # the group.
    leftover_pos = (cand & seg.isna().to_numpy()).nonzero()[0]
    if leftover_pos.size:
        gaps = leftover_pos[1:] - leftover_pos[:-1]
        group_offsets = np.concatenate(([0], (gaps > 1).cumsum()))
        seg.iloc[leftover_pos] = next_id + group_offsets

    out["toc_segment_id"] = seg
    return out


# ================================================================================
# STEP 7: Scorer  (KPI col: toc_segment_score; writes block_type)
# ================================================================================

def score_toc_segments(df: pd.DataFrame, config: TocConfig = TocConfig()) -> pd.DataFrame:
    """
    Score every ``toc_segment_id`` group and stamp the winners:
    ``block_type = 'toc'`` on the segment's rows (hidden furniture excluded)
    and ``'toc_heading'`` on the heading line(s) just above it. Adds the KPI
    columns ``toc_segment_score`` (float | NA, broadcast to the segment's rows)
    and ``toc_segment_score_detail`` (str | NA, which KPIs fired and their
    contribution, e.g. "leaders+5|heading+2|mixed_types-1") for harness
    inspection.

    Segments with fewer than ``config.min_segment_rows`` rows are discarded
    without scoring (score stays NA). The rest get an additive score:

      +5  majority of rows carry dot leaders
      +5  majority of rows carry an internal link
      +2  TOC heading candidate within ``nearby_lookback`` rows above the start
      +1  "Page(s)" line within that window or as the segment's first line
      +1  more than 5 candidate rows sit inside tables
      -1  candidates mix page-label types ('arabic' + 'roman' + ...)

    Accepted when score >= ``config.min_score``.
    """
    out = df.copy()
    out["toc_segment_score"] = pd.NA
    out["toc_segment_score_detail"] = pd.NA
    if "block_type" not in out.columns:
        out["block_type"] = pd.NA

    if out.empty or "toc_segment_id" not in out.columns:
        return out

    seg = out["toc_segment_id"]
    has_seg = seg.notna()
    if not has_seg.any():
        return out

    keys = seg[has_seg]
    pos = pd.Series(np.arange(len(out)), index=out.index)

    # ---- per-segment stats ----
    n_rows = keys.groupby(keys).size()

    if "toc_has_dot_leaders" in out.columns:
        leaders = out["toc_has_dot_leaders"].fillna(False).astype(bool)
    else:
        leaders = pd.Series(False, index=out.index)
    if "link_type" in out.columns:
        internal = (
            out["link_type"].astype("string").str.strip().str.lower()
            .eq("internal").fillna(False)
        )
    else:
        internal = pd.Series(False, index=out.index)

    leader_majority = leaders[has_seg].groupby(keys).sum() / n_rows > 0.5
    link_majority = internal[has_seg].groupby(keys).sum() / n_rows > 0.5

    cand = out.get("toc_row_candidate", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if "table_id" in out.columns:
        tid = out["table_id"].astype("string").str.strip()
        in_table = (tid.notna() & tid.ne("")).fillna(False)
    else:
        in_table = pd.Series(False, index=out.index)
    big_table = (cand & in_table)[has_seg].groupby(keys).sum() > 5

    if "toc_row_page_type" in out.columns:
        mixed_types = out["toc_row_page_type"][has_seg].groupby(keys).nunique() > 1
    else:
        mixed_types = pd.Series(False, index=n_rows.index)

    # ---- nearby heading / page-header lookback at each segment's start ----
    start_pos = pos[has_seg].groupby(keys).min()

    heading_mask = out.get("toc_heading_candidate", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    heading_pos = np.flatnonzero(heading_mask.to_numpy())

    if "text" in out.columns:
        page_hdr_mask = out["text"].astype("string").fillna("").str.contains(_PAGE_HEADER_RE, regex=True, na=False)
    else:
        page_hdr_mask = pd.Series(False, index=out.index)
    page_hdr_pos = np.flatnonzero(page_hdr_mask.to_numpy())

    starts = start_pos.to_numpy()
    lo = starts - config.nearby_lookback
    heading_nearby = pd.Series(
        np.searchsorted(heading_pos, lo) < np.searchsorted(heading_pos, starts),
        index=start_pos.index,
    )
    # "Page(s)" may also BE the segment's first line (a table's header row).
    page_header_nearby = pd.Series(
        np.searchsorted(page_hdr_pos, lo) < np.searchsorted(page_hdr_pos, starts + 1),
        index=start_pos.index,
    )

    # ---- score ----
    # astype(float): the stats can come out arrow-backed bool (string-dtype
    # comparisons), which refuses float arithmetic.
    score = (
        _SCORE_LEADER_MAJORITY * leader_majority.astype(float)
        + _SCORE_LINK_MAJORITY * link_majority.astype(float)
        + _SCORE_HEADING_NEARBY * heading_nearby.astype(float)
        + _SCORE_PAGE_HEADER_NEARBY * page_header_nearby.astype(float)
        + _SCORE_BIG_TABLE * big_table.astype(float)
        + _SCORE_MIXED_PAGE_TYPES * mixed_types.astype(float)
    )
    scoreable = n_rows >= config.min_segment_rows
    accepted = scoreable & (score >= config.min_score)

    # Per-KPI breakdown, e.g. "leaders+5|heading+2|mixed_types-1" — which
    # signals fired for the segment, with their contribution.
    components = {
        "leaders": (leader_majority, _SCORE_LEADER_MAJORITY),
        "links": (link_majority, _SCORE_LINK_MAJORITY),
        "heading": (heading_nearby, _SCORE_HEADING_NEARBY),
        "page_header": (page_header_nearby, _SCORE_PAGE_HEADER_NEARBY),
        "big_table": (big_table, _SCORE_BIG_TABLE),
        "mixed_types": (mixed_types, _SCORE_MIXED_PAGE_TYPES),
    }
    flags = pd.DataFrame(
        {name: flag.astype(bool) for name, (flag, _) in components.items()},
        index=n_rows.index,
    )
    detail = flags.apply(
        lambda r: "|".join(f"{name}{components[name][1]:+g}" for name in flags.columns if r[name]),
        axis=1,
    )

    out.loc[has_seg, "toc_segment_score"] = keys.map(score.where(scoreable))
    out.loc[has_seg, "toc_segment_score_detail"] = keys.map(detail.where(scoreable))

    # ---- stamp winners ----
    hidden = _hidden_block_mask(out)
    accepted_ids = set(accepted.index[accepted])
    if accepted_ids:
        toc_mask = has_seg & seg.isin(accepted_ids) & ~hidden
        out.loc[toc_mask, "block_type"] = "toc"

        # Heading candidates in the lookback window above each accepted start.
        heading_iloc: List[int] = []
        for seg_id in accepted_ids:
            s = int(start_pos.at[seg_id])
            window = heading_pos[(heading_pos >= s - config.nearby_lookback) & (heading_pos < s)]
            heading_iloc.extend(int(p) for p in window)
        if heading_iloc:
            idx = out.index[heading_iloc]
            out.loc[idx, "block_type"] = "toc_heading"

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
      3. toc_heading_candidate
      4. toc_has_dot_leaders
      5. toc_row_candidate      (+ toc_row_page_token / toc_row_page_type)
      6. toc_segment_id         (dot-leader / internal-link runs, tables, leftovers)
      7. toc_segment_score      + block_type = 'toc' / 'toc_heading'

    (Step 1, the hidden-block mask, is applied inside each KPI pass rather than
    persisted as a column.)
    """
    if df.empty:
        return df

    out = add_toc_heading_candidates(df, config)
    out = add_toc_dot_leaders(out)
    out = add_toc_row_candidates(out, page_label_config, config)
    out = add_toc_run_segments(out, config)
    out = score_toc_segments(out, config)
    return out

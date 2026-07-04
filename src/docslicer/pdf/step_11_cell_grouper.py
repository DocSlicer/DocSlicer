"""
step_11_cell_grouper.py  (option 2 — movement annotation)

# NOTE: this is not used for OCR because it does not have a streaming order

Purpose: Group cells together that are on different visual lines, but form 1 logical table cell
    - Multi-line header
    - Table cells that flex onto another line

Cells -> Cells.  Annotation-only first stage of a new multiline-cell grouper.

Background
----------
step_08 serialises cells in reading order (cell_id). In a multiline table region
a single logical row is spread over several cells, and the reading order walks
one *column* down before jumping to the next column — so the y of consecutive
cells zig-zags down/up as columns are traversed.

Option 1 tried to reconstruct rows directly from per-cell movement. This second
option starts more conservatively: it just *describes* the movement between
consecutive cells (in cell_id order), leaving the actual grouping decision for a
later pass that will be designed after inspecting these signals on real
documents.

What this step adds (nothing destroyed)
---------------------------------------
  y_center     : the cell's vertical center, (y_top + y_bottom) / 2.
  y_movement   : how y_center changes from this cell (t) to the next (t+1),
                 in cell_id (reading) order:
                     DOWN  y_center increases  (larger y == lower on page)
                     UP    y_center decreases
                     NONE  unchanged within tolerance, OR the two cells share a
                           top or bottom edge (a tall multi-line cell beside a
                           short one are top/bottom-aligned siblings of the same
                           row even though their centers differ).
                 Empty on the last cell of a page / document (no successor).
  x_movement   : how the next cell's x-span sits relative to this cell's x-span
                 ([x_left, x_right]):
                     RIGHT next cell lies entirely to the right (new column)
                     LEFT  next cell lies entirely to the left
                     NONE  the spans overlap (even if unequal length)
                 Empty on the last cell of a page / document.

Both movement columns describe the transition t -> t+1 and are stored on cell t
(the source cell). Transitions never cross a page boundary: page-local y resets
each page, so the last cell of every page has empty movement.

Vertical-stack runs
-------------------
A transition that is (y_movement DOWN, x_movement NONE) means "straight down,
same column" — the signature of one line of a multi-line table cell continuing
into the next. Maximal consecutive sequences of cells chained by such
transitions are vertical stacks (candidate multi-line cells). Two more columns
describe them:

  vstack_gap_em: vertical edge gap to the next cell (next.y_top - cur.y_bottom),
                 normalised by the larger of the two font sizes. The signal that
                 breaks an over-long run; empty at boundaries.
  vstack_id    : dense, reading-order id shared by every cell in one vertical
                 stack run. A cell that neither continues nor is continued is its
                 own run of size 1. A (DOWN, NONE) step still breaks into a new id
                 when vstack_gap_em exceeds config.vstack_max_gap_em (the next
                 cell is then a separate block, not a continued line). A change
                 in font styling (font_family, font_size, non_stroking_color)
                 between the two cells also breaks into a new id.
  vstack_width : the run's bounding-box width, max(x_right) - min(x_left) over
                 its cells (i.e. the width of the merged multi-line cell).
  vstack_n_cells: number of cells in the run (all runs, incl. size-1).
  vstack_n_lines: number of visible lines in the run = sum of each cell's
                 line_ids length (mcid: one cell can be several visible lines).
                 This — not the cell count — drives the score_max_lines cap.
  vstack_score : how table-cell-like a run is, scored only for runs with >= 2
                 cells (else NaN). Excluded -> NaN when the run is too tall
                 (> score_max_lines visible lines) or too wide (off the widest
                 score_width_tiers row). Otherwise summed from:
                   width tier (vstack_width vs page content span = max x_right -
                     min x_left, like the gutter extractor): see config
                     score_width_tiers (default < 1/5 -> +3, < 1/4 -> +2,
                     1/4..1/3 -> 0, 1/3..1/2 -> -1);
                   enter (both axes): +2 if entered UP+RIGHT, +1 if DOWN+LEFT;
                   exit  (both axes): +2 if it exits DOWN+LEFT, +2 if NONE+RIGHT.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _count_line_ids(v) -> int:
    """
    Number of visible lines a cell represents = number of entries in its line_ids.

    In-pipeline line_ids is a list; from a re-read CSV it is its string repr. Both
    are handled. A missing/empty value counts as 1 (the cell is one line).
    """
    if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
        return len(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s == "[]":
            return 0
        try:
            parsed = ast.literal_eval(s)
            return len(parsed) if hasattr(parsed, "__len__") else 1
        except (ValueError, SyntaxError):
            return s.count(",") + 1
    return 1


# ================================================================================
# CONFIG
# ================================================================================

@dataclass(frozen=True)
class RowGroupConfig:
    # |Δ y_center| <= this (pt) counts as "same height" -> y_movement NONE.
    # Also the slack for "shares a top/bottom edge". Catches alignment jitter
    # between side-by-side cells that share a row. Keep well below line spacing.
    y_tol: float = 2.0
    # Slack (pt) when testing "entirely left/right". A touch/overlap up to this
    # much is still treated as overlap (NONE) rather than a clean LEFT/RIGHT.
    x_tol: float = 0.0
    # Within a (DOWN, NONE) stack run, break into a new vstack_id when the vertical
    # edge gap between consecutive cells exceeds this many em. The gap is measured
    # as next.y_top - cur.y_bottom, normalised by the larger of the two font sizes
    # (mcid-proof: independent of how many visual lines each cell spans). A normal
    # line continuation is ~0.2-0.3em; lower this later to split tighter.
    vstack_max_gap_em: float = 0.8

    # A change in font styling (font_family, font_size, non_stroking_color)
    # between two otherwise-continuing cells starts a new vstack_id: a restyle
    # marks a new logical block even mid-column. Set False to ignore styling.
    vstack_break_on_style_change: bool = True
    # font_size equality slack (pt) when comparing styling: sizes within this of
    # each other are "the same size", so sub-point extraction jitter alone does
    # not split a run. font_family and non_stroking_color compare exactly.
    vstack_style_font_size_tol: float = 0.1

    # --- vstack scoring (only runs with >= 2 cells are scored) ---
    # A run taller than this many *visible lines* is body text, not a table cell
    # -> excluded. Counts entries across each cell's line_ids (mcid means one cell
    # can already be several visible lines), not the number of cells in the run.
    score_max_lines: int = 6
    # Width score table. Each row is (width_fraction, score), where width_fraction
    # is vstack_width as a fraction of the page *content* span (max x_right -
    # min x_left over the page, like the gutter extractor — NOT page_width).
    # Read it as upper bounds, narrowest first: a run takes the score of the first
    # row it is still narrower than (ratio < width_fraction). A run wider than
    # every row — i.e. >= the widest fraction below — is excluded (no score).
    # To tweak: edit a score next to its fraction, or change the widest fraction
    # to move the exclude cutoff. Keep the rows sorted narrow -> wide.
    score_width_tiers: tuple[tuple[float, float], ...] = (
        (1 / 5, +3.0),   # ratio < 1/5 of the page span
        (1 / 4, +2.0),   # 1/5 <= ratio < 1/4
        (1 / 3,  0.0),   # 1/4 <= ratio < 1/3
        (1 / 2, -3.0),   # 1/3 <= ratio < 1/2
    )                    # ratio >= 1/2  ->  excluded (no score)
    # Movement bonuses. Entering: how the cell before the run moved into it.
    # Exiting: how the run's last cell then moves on. A real column-stack is
    # typically entered from above/right and resumes flow afterwards. Each bonus
    # needs BOTH axes to match (e.g. UP and RIGHT together; UP with LEFT scores 0).
    score_enter_up_right_bonus:  float = 2.0   # preceded by UP and RIGHT
    score_enter_down_left_bonus: float = 1.0   # preceded by DOWN and LEFT
    score_exit_down_left_bonus:  float = 2.0   # followed by DOWN and LEFT
    score_exit_none_right_bonus: float = 2.0   # followed by NONE and RIGHT
    # Reject a run whose merged bbox would be alone in the y-band it spans — no
    # other cell sits beside it (entirely left/right) overlapping that band. That
    # is a centered multi-line title/heading block, not a table cell. A genuine
    # column header shares its band with sibling columns. See _runs_alone_in_y_band.
    score_require_band_siblings: bool = True

    # --- final grouped-row decision (grouped_row_id) ---
    # A vstack *seeds* a logical-row group when its vstack_score is >= this. An
    # unscored (NaN) run never seeds. Seeds are the confident multi-line table
    # cells that survived scoring; grouping grows outward from them.
    grouped_min_score: float = 0.0
    # Two vstacks are "line-adjacent" (a precondition to joining one row) when
    # their line_id ranges are within this many lines of touching: 0 = the ranges
    # must overlap, 1 = also back-to-back reading lines. Keeps a row local — it
    # never reaches across a line-id gap. See assign_grouped_rows.
    grouped_line_gap_max: int = 1
    # Alignment slack (pt) for joining a neighbour into the row: joined when the
    # two vstacks' top edges, bottom edges, OR vertical centers match within this.
    # Bottom-alignment is the usual table-row signal (baselines line up); top/mid
    # catch the other layouts. Keep around a line's worth of slack.
    grouped_align_tol: float = 2.0


CONFIG = RowGroupConfig()


# ================================================================================
# VSTACK ID ASSIGNMENT
# ================================================================================

# Style columns whose change between consecutive cells breaks a vstack run.
_STYLE_COLS = ("font_family", "font_size", "non_stroking_color", "is_bold")


def _norm_style_val(v):
    """Comparable form of a style value (non_stroking_color may be a list/tuple)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(v)
    return v


def _is_na_scalar(v) -> bool:
    """True for a missing scalar; sequences (e.g. an RGB color tuple) are never NA."""
    if isinstance(v, (list, tuple, np.ndarray, dict)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _same_style_next(seq: pd.DataFrame, font_size_tol: float) -> np.ndarray:
    """
    Per position t: do cells t and t+1 share font styling
    (font_family, font_size, non_stroking_color)?

    A change in any *present* style column makes position t a style boundary
    (False) — which breaks a vertical-stack run even on an otherwise-continuing
    (DOWN, NONE) step. Absent columns are ignored. font_size compares within
    font_size_tol to absorb sub-point extraction jitter; font_family and
    non_stroking_color compare exactly, with two missing values counting as the
    same style. The last position has no successor -> False.
    """
    n = len(seq)
    same = np.ones(n, dtype=bool)
    if n:
        same[-1] = False                                   # no successor

    for col in _STYLE_COLS:
        if col not in seq.columns:
            continue
        if col == "font_size":
            a = pd.to_numeric(seq[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(seq[col].shift(-1), errors="coerce").to_numpy(float)
            col_same = np.isclose(a, b, atol=font_size_tol, equal_nan=True)
        else:
            cur = seq[col].to_numpy(object)
            nxt = seq[col].shift(-1).to_numpy(object)
            col_same = np.array(
                [_style_val_eq(x, y) for x, y in zip(cur, nxt)],
                dtype=bool,
            )
        same &= col_same
    return same


def _style_val_eq(x, y) -> bool:
    """Equal styling values, treating two missing values as equal."""
    x_na, y_na = _is_na_scalar(x), _is_na_scalar(y)
    if x_na or y_na:
        return x_na and y_na
    return _norm_style_val(x) == _norm_style_val(y)


def _assign_vstack_ids(
    seq:    pd.DataFrame,
    y_mov:  np.ndarray,
    x_mov:  np.ndarray,
    gap_em: np.ndarray,
    config: RowGroupConfig = CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chain cells into vertical-stack runs and number them densely in reading order.

    A step t -> t+1 continues the current run when it goes straight down in the
    same column (y_movement DOWN, x_movement NONE) AND the cells are close enough
    vertically — vstack_gap_em <= config.vstack_max_gap_em (a larger gap is a
    separate block; a NaN gap never breaks). When config.vstack_break_on_style_change
    is set, a change in font styling (font_family, font_size, non_stroking_color)
    also breaks the run, so a restyle starts a new id even mid-column.

    Returns (new_run, vstack_id_seq): new_run marks each run's first position
    (also consumed by scoring), vstack_id_seq is the dense 1..N id per position.
    """
    n = len(seq)

    # cont[i] is True when the i -> i+1 step goes straight down in the same column
    # AND the cells are close enough vertically. A gap above vstack_max_gap_em
    # means the next cell is a separate block (new paragraph/row), so the run
    # breaks there even though the step is (DOWN, NONE). NaN gap never breaks.
    gap_ok = ~(gap_em > config.vstack_max_gap_em)   # NaN -> True (don't break)
    cont = (y_mov == "DOWN") & (x_mov == "NONE") & gap_ok

    # A restyle between two cells breaks the run even on a continuing step.
    if config.vstack_break_on_style_change:
        cont = cont & _same_style_next(seq, config.vstack_style_font_size_tol)

    # A new run begins at position 0 and wherever the previous step was not a
    # continuation; cumsum then numbers runs densely in reading order.
    new_run = np.ones(n, dtype=bool)
    new_run[1:] = ~cont[:-1]
    vstack_id_seq = np.cumsum(new_run)             # dense 1..N in reading order
    return new_run, vstack_id_seq


# ================================================================================
# VSTACK SCORING
# ================================================================================

def _width_tier_score(
    ratio: np.ndarray,
    tiers: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """
    Map each width ratio to its tier score (see RowGroupConfig.score_width_tiers).

    A ratio takes the score of the narrowest tier it is still below (ratio <
    fraction). Wider than every tier — or a NaN ratio — yields NaN, which the
    caller reads as "excluded". Assigning widest-first lets the narrowest match
    win by overwrite, so tier order in the config is not load-bearing.
    """
    out = np.full(np.shape(ratio), np.nan, dtype=float)
    for fraction, score in sorted(tiers, reverse=True):   # widest -> narrowest
        out[ratio < fraction] = score
    return out


def _runs_alone_in_y_band(
    starts:     np.ndarray, ends:       np.ndarray,
    run_top:    np.ndarray, run_bot:    np.ndarray,
    run_left:   np.ndarray, run_right:  np.ndarray, run_page: np.ndarray,
    cell_top:   np.ndarray, cell_bot:   np.ndarray,
    cell_left:  np.ndarray, cell_right: np.ndarray, cell_page: np.ndarray,
    candidate_mask: np.ndarray,
    x_tol:      float,
) -> np.ndarray:
    """
    For each candidate run, decide whether merging it would leave its bbox *alone*
    in the y-band it spans: True when no other cell that sits beside it (entirely
    to its left or right) overlaps that band.

    A genuine multi-line table cell shares its band with sibling columns (cells
    beside it); a centered multi-line title/heading block spans a band with
    nothing beside it -> alone -> rejected by the caller. Only "beside" cells
    count: a neighbour in the *same* column (above/below) is not a sibling, so we
    require horizontal disjointness, not mere y-overlap.

    Runs are contiguous slices of the sequence, so a run's own cells are the
    [start, end] block and are excluded from the search.
    """
    alone = np.zeros(len(starts), dtype=bool)
    for r in np.where(candidate_mask)[0]:
        y_overlap = (cell_top < run_bot[r]) & (cell_bot > run_top[r])
        beside    = (cell_right <= run_left[r] + x_tol) | (cell_left >= run_right[r] - x_tol)
        others    = (cell_page == run_page[r]) & y_overlap & beside
        others[starts[r]:ends[r] + 1] = False          # drop the run's own cells
        alone[r]  = not others.any()
    return alone


def _score_vstack_runs(
    seq:              pd.DataFrame,
    new_run:          np.ndarray,
    vstack_width_seq: np.ndarray,
    page_span_seq:    np.ndarray,
    y_mov:            np.ndarray,
    x_mov:            np.ndarray,
    config:           RowGroupConfig = CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score each contiguous vstack run. Only runs with >= 2 cells are scored; the
    rest (and runs excluded for being too tall/wide) get NaN.

    All inputs are aligned to the reading-order sequence positions. ``new_run``
    marks each run's first position, so runs are contiguous blocks of the
    sequence — start/end positions and per-run sizes follow directly.

    Score = width tier + enter bonus + exit bonus:
      width tier: config.score_width_tiers via _width_tier_score (vstack_width as
          a fraction of the page content span; too wide -> excluded/NaN).
      enter (movement of the cell *before* the run into it; both axes required):
          UP   AND RIGHT -> +score_enter_up_right_bonus
          DOWN AND LEFT  -> +score_enter_down_left_bonus
      exit (the run's last cell's outgoing movement; both axes required):
          DOWN AND LEFT  -> +score_exit_down_left_bonus
          NONE AND RIGHT -> +score_exit_none_right_bonus

    The tall-run cap is on *visible lines* (summed line_ids entries across the
    run's cells), not the cell count — mcid means one cell can already be several
    visible lines.

    A qualifying run is additionally rejected (NaN) when it would be alone in its
    y-band (see _runs_alone_in_y_band) — a centered title/heading, not a table
    cell. ``seq`` supplies the per-cell bboxes (and line_ids) those checks need.

    Returns (n_cells, n_lines, score) arrays, each broadcast over the run's
    positions and aligned to the sequence (callers map them back by cell_id).
    """
    n = len(new_run)
    starts = np.where(new_run)[0]
    ends   = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1]  = n - 1
    n_cells = ends - starts + 1                       # per run

    # Visible lines per run: sum of each cell's line_ids length over the run.
    if "line_ids" in seq.columns:
        cell_lines = seq["line_ids"].apply(_count_line_ids).to_numpy()
    else:
        cell_lines = np.ones(n, dtype=int)
    n_lines = np.add.reduceat(cell_lines, starts)     # per run

    width = vstack_width_seq[starts]
    span  = page_span_seq[starts]
    ratio = np.where(span > 0, width / span, np.nan)

    # Movement entering the run (from the cell before its first) and leaving it
    # (its last cell's outgoing step). Boundaries read as None -> no bonus.
    prev_y = np.array([y_mov[s - 1] if s > 0 else None for s in starts], dtype=object)
    prev_x = np.array([x_mov[s - 1] if s > 0 else None for s in starts], dtype=object)
    next_y = y_mov[ends]
    next_x = x_mov[ends]

    # Width tier (narrower is more table-cell-like). NaN here == excluded.
    width_score = _width_tier_score(ratio, config.score_width_tiers)
    enter_bonus = (
        np.where((prev_y == "UP")   & (prev_x == "RIGHT"), config.score_enter_up_right_bonus,  0.0)
        + np.where((prev_y == "DOWN") & (prev_x == "LEFT"),  config.score_enter_down_left_bonus, 0.0)
    )
    exit_bonus = (
        np.where((next_y == "DOWN") & (next_x == "LEFT"),  config.score_exit_down_left_bonus,  0.0)
        + np.where((next_y == "NONE") & (next_x == "RIGHT"), config.score_exit_none_right_bonus, 0.0)
    )
    raw_score = width_score + enter_bonus + exit_bonus

    # Qualify: >= 2 cells, within the visible-line cap, and a usable width tier
    # (NaN width_score = no/over-wide span -> excluded).
    scored = (
        (n_cells >= 2)
        & (n_lines <= config.score_max_lines)
        & np.isfinite(width_score)
    )

    # Band-sibling exclusion: drop runs whose merged bbox would be alone in its
    # y-band (a centered title/heading block, not a table cell). Only checked for
    # runs that otherwise qualify, so it never resurrects an excluded run.
    if config.score_require_band_siblings:
        cell_top   = seq["y_top"].to_numpy(float)
        cell_bot   = seq["y_bottom"].to_numpy(float)
        cell_left  = seq["x_left"].to_numpy(float)
        cell_right = seq["x_right"].to_numpy(float)
        cell_page  = (seq["page_number"].to_numpy() if "page_number" in seq.columns
                      else np.zeros(n))
        # Merged run band/span via reduceat over the contiguous run slices.
        run_top   = np.minimum.reduceat(cell_top,   starts)
        run_bot   = np.maximum.reduceat(cell_bot,   starts)
        run_left  = np.minimum.reduceat(cell_left,  starts)
        run_right = np.maximum.reduceat(cell_right, starts)
        run_page  = cell_page[starts]
        alone = _runs_alone_in_y_band(
            starts, ends, run_top, run_bot, run_left, run_right, run_page,
            cell_top, cell_bot, cell_left, cell_right, cell_page,
            candidate_mask=scored, x_tol=config.x_tol,
        )
        scored = scored & ~alone

    run_score = np.where(scored, raw_score, np.nan)

    # Broadcast per-run values back over the sequence (runs are contiguous and
    # ordered, so a repeat by run size reconstructs the sequence layout).
    return (
        np.repeat(n_cells, n_cells),
        np.repeat(n_lines, n_cells),
        np.repeat(run_score, n_cells),
    )


# ================================================================================
# GROUPED-ROW DECISION (final)
# ================================================================================

def _parse_line_ids(v) -> list:
    """Line ids of a cell as a plain list (handles a list or its CSV-repr string)."""
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s == "[]":
            return []
        try:
            parsed = ast.literal_eval(s)
            return list(parsed) if hasattr(parsed, "__iter__") else [parsed]
        except (ValueError, SyntaxError):
            return []
    return []


def _cell_line_bounds(df: pd.DataFrame):
    """
    Per-cell (min line id, max line id). A cell is normally one visible line
    (``line_id``); with mcid it may carry several (``line_ids`` list). Returns
    None when neither column is present — grouping needs a line axis to test
    adjacency.
    """
    if "line_id" in df.columns:
        v = pd.to_numeric(df["line_id"], errors="coerce").to_numpy(float)
        return v, v
    if "line_ids" in df.columns:
        lo = np.full(len(df), np.nan)
        hi = np.full(len(df), np.nan)
        for i, val in enumerate(df["line_ids"].to_numpy(object)):
            ids = pd.to_numeric(pd.Series(_parse_line_ids(val)), errors="coerce").dropna()
            if len(ids):
                lo[i], hi[i] = float(ids.min()), float(ids.max())
        return lo, hi
    return None


class _Union:
    """Minimal union-find over range(n) for connecting vstacks into rows."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]          # path halving
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_grouped_rows(
    df_cells: pd.DataFrame,
    config: RowGroupConfig = CONFIG,
) -> pd.DataFrame:
    """
    Final decision: merge confident vstacks into logical rows (grouped_row_id).

    Seeds are the vstacks that survived scoring (vstack_score >=
    config.grouped_min_score; an unscored NaN run never seeds). Starting from
    those, two vstacks are joined into one logical row when they are BOTH:

      line-adjacent  their line_id ranges overlap, or sit within
                     config.grouped_line_gap_max lines of touching; and
      aligned        their top edges, bottom edges, OR vertical centers match
                     within config.grouped_align_tol.

    Adjacency + alignment is transitive (union-find), so a whole row of
    side-by-side columns collapses into one group with one merged bbox — a
    multi-line column header (all bottom-aligned), or a header stub beside its
    single-line year columns (line-id overlapping, bottom-aligned).

    A joined-in vstack need not itself be a survivor: a single-line first column
    that aligns with the scored columns beside it is pulled in. A component
    becomes a logical row only if it holds a seed AND actually merged (>= 2
    vstacks); a lone survivor that grouped with nothing is rejected (left blank),
    as is any vstack outside a seeded component (an ordinary single-line data row,
    whose logical row is already its shared line_id).

    Adds, one value per cell (shared within a group, NaN when the cell's vstack
    is not in a seeded group): grouped_row_id (dense, reading order),
    grouped_row_n_vstacks, and the merged bbox grouped_row_x_left / _x_right /
    _y_top / _y_bottom. Returns df_cells unchanged when the inputs it needs
    (vstack_id, vstack_score, a line axis, bboxes) are missing.
    """
    required = {"cell_id", "vstack_id", "x_left", "x_right", "y_top", "y_bottom"}
    if (df_cells is None or df_cells.empty
            or not required.issubset(df_cells.columns)
            or "vstack_score" not in df_cells.columns):
        return df_cells
    bounds = _cell_line_bounds(df_cells)
    if bounds is None:
        return df_cells
    line_lo, line_hi = bounds

    df = df_cells.copy()
    page = (df["page_number"].to_numpy() if "page_number" in df.columns
            else np.zeros(len(df), dtype=int))

    cell = pd.DataFrame({
        "page":      page,
        "vstack_id": df["vstack_id"].to_numpy(),
        "x_left":    df["x_left"].to_numpy(float),
        "x_right":   df["x_right"].to_numpy(float),
        "y_top":     df["y_top"].to_numpy(float),
        "y_bottom":  df["y_bottom"].to_numpy(float),
        "line_lo":   line_lo,
        "line_hi":   line_hi,
        "score":     pd.to_numeric(df["vstack_score"], errors="coerce").to_numpy(float),
    })

    # Collapse to one row per (page, vstack): its bbox, line span, and score
    # (score is uniform within a vstack; max just ignores the NaN duplicates).
    vs = (cell.groupby(["page", "vstack_id"], sort=True)
              .agg(x_left=("x_left", "min"), x_right=("x_right", "max"),
                   y_top=("y_top", "min"),   y_bottom=("y_bottom", "max"),
                   line_lo=("line_lo", "min"), line_hi=("line_hi", "max"),
                   score=("score", "max"))
              .reset_index())
    n = len(vs)
    y_top = vs["y_top"].to_numpy(float)
    y_bot = vs["y_bottom"].to_numpy(float)
    y_cen = (y_top + y_bot) / 2.0
    l_lo  = vs["line_lo"].to_numpy(float)
    l_hi  = vs["line_hi"].to_numpy(float)
    pg    = vs["page"].to_numpy()
    score = vs["score"].to_numpy(float)
    vsid  = vs["vstack_id"].to_numpy()

    tol = config.grouped_align_tol
    gap = config.grouped_line_gap_max
    uf  = _Union(n)

    # Link line-adjacent, edge-aligned vstacks (never across pages). Sorting by
    # (page, line_lo) lets the inner scan stop as soon as a candidate starts past
    # the current vstack's adjacency window — the pair with the smaller line_lo is
    # always the outer one, so the forward window covers every real pair.
    order = np.lexsort((vsid, l_lo, pg))
    for oi in range(n):
        i = order[oi]
        for oj in range(oi + 1, n):
            j = order[oj]
            if pg[j] != pg[i] or l_lo[j] > l_hi[i] + gap:
                break
            aligned = (abs(y_top[i] - y_top[j]) <= tol
                       or abs(y_bot[i] - y_bot[j]) <= tol
                       or abs(y_cen[i] - y_cen[j]) <= tol)
            if aligned:
                uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(n)])

    # A component becomes a logical row only if it holds a seed (survivor) AND
    # actually merged (>= 2 vstacks). A lone seed that grouped with no neighbour,
    # and any vstack outside a seeded component, is left blank (rejected here).
    is_seed   = np.isfinite(score) & (score >= config.grouped_min_score)
    seed_root = set(roots[is_seed].tolist())
    comp_size = np.bincount(roots, minlength=n)
    in_group  = np.array(
        [roots[i] in seed_root and comp_size[roots[i]] >= 2 for i in range(n)]
    )

    # Dense reading-order ids over seeded components only, numbered by first
    # appearance in (page, line, id); every other vstack gets NaN.
    dense: dict[int, int] = {}
    for i in order:
        if in_group[i]:
            r = int(roots[i])
            if r not in dense:
                dense[r] = len(dense) + 1
    vs["grouped_row_id"] = [
        float(dense[int(roots[i])]) if in_group[i] else np.nan for i in range(n)
    ]

    # NaN group keys drop out of groupby, so the bbox/count columns are NaN for
    # the blank vstacks too.
    grp = vs.groupby("grouped_row_id")
    vs["grouped_row_x_left"]    = grp["x_left"].transform("min")
    vs["grouped_row_x_right"]   = grp["x_right"].transform("max")
    vs["grouped_row_y_top"]     = grp["y_top"].transform("min")
    vs["grouped_row_y_bottom"]  = grp["y_bottom"].transform("max")
    vs["grouped_row_n_vstacks"] = grp["vstack_id"].transform("size")

    # Map per-vstack results back onto every cell by (page, vstack_id).
    out_cols = ["grouped_row_id", "grouped_row_n_vstacks",
                "grouped_row_x_left", "grouped_row_x_right",
                "grouped_row_y_top", "grouped_row_y_bottom"]
    lut = vs.set_index(["page", "vstack_id"])[out_cols]
    keyed = pd.MultiIndex.from_arrays([page, df["vstack_id"].to_numpy()])
    for col in out_cols:
        df[col] = lut[col].reindex(keyed).to_numpy()

    return df


# ================================================================================
# ENTRY POINT
# ================================================================================

def group_multiline_cells(
    df_cells: pd.DataFrame,
    config: RowGroupConfig = CONFIG,
) -> pd.DataFrame:
    """
    Annotate cells with their vertical center and cell-to-cell movement.

    Adds y_center, y_movement, x_movement (see module docstring). Pure
    annotation: row/column membership is decided in a later pass.
    """
    required = {"cell_id", "x_left", "x_right", "y_top", "y_bottom"}
    if df_cells is None or df_cells.empty or not required.issubset(df_cells.columns):
        return df_cells

    df = df_cells.copy()

    # ── Per-cell vertical center ───────────────────────────────────────────────
    df["y_center"] = (df["y_top"].to_numpy(float) + df["y_bottom"].to_numpy(float)) / 2.0

    # ── Walk cells in reading order (page first, then cell_id) ─────────────────
    # df_cells is one row per cell, so cell_id order *is* the sequence — no
    # grouping needed. Sort a working copy; movement is mapped back by cell_id.
    page_col = "page_number" if "page_number" in df.columns else None
    sort_cols = (["page_number", "cell_id"] if page_col else ["cell_id"])
    seq = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Next cell's geometry (the t+1 partner of each transition).
    nyc = seq["y_center"].shift(-1).to_numpy(float)
    nxl = seq["x_left"].shift(-1).to_numpy(float)
    nxr = seq["x_right"].shift(-1).to_numpy(float)
    nyt = seq["y_top"].shift(-1).to_numpy(float)
    nyb = seq["y_bottom"].shift(-1).to_numpy(float)

    cur_yc = seq["y_center"].to_numpy(float)
    cur_xl = seq["x_left"].to_numpy(float)
    cur_xr = seq["x_right"].to_numpy(float)
    cur_yt = seq["y_top"].to_numpy(float)
    cur_yb = seq["y_bottom"].to_numpy(float)

    # A transition exists only between two same-page cells. The last cell of each
    # page (and of the doc) has no valid successor -> movement stays empty.
    has_next = np.isfinite(nyc)
    if page_col:
        npg = seq["page_number"].shift(-1).to_numpy()
        has_next &= (npg == seq["page_number"].to_numpy())

    # ── y_movement: increase => DOWN, decrease => UP, ~equal => NONE ───────────
    # A shared top or bottom edge overrides to NONE: two cells that align on an
    # edge are top/bottom-aligned siblings of one row (e.g. a tall multi-line
    # cell beside a short one), even though their centers differ.
    dy = nyc - cur_yc
    edge_shared = (np.abs(nyt - cur_yt) <= config.y_tol) | (np.abs(nyb - cur_yb) <= config.y_tol)
    y_mov = np.full(len(seq), None, dtype=object)
    y_mov[has_next & (dy >  config.y_tol)]              = "DOWN"
    y_mov[has_next & (dy < -config.y_tol)]              = "UP"
    y_mov[has_next & (np.abs(dy) <= config.y_tol)]      = "NONE"
    y_mov[has_next & edge_shared]                       = "NONE"

    # ── x_movement: next span entirely right/left of current => RIGHT/LEFT ─────
    x_mov = np.full(len(seq), None, dtype=object)
    entirely_right = nxl >= cur_xr - config.x_tol
    entirely_left  = nxr <= cur_xl + config.x_tol
    x_mov[has_next]                  = "NONE"          # default: spans overlap
    x_mov[has_next & entirely_right] = "RIGHT"
    x_mov[has_next & entirely_left]  = "LEFT"

    # ── Vertical-stack runs: chain cells joined by (DOWN, NONE) transitions ────
    # Edge gap to the next cell, em-normalised: next.y_top - cur.y_bottom over the
    # larger of the two font sizes. mcid-proof — independent of either cell's
    # height (a cell can already span several visual lines). NaN at boundaries and
    # where font size is unusable.
    if "font_size" in seq.columns:
        cur_fs = pd.to_numeric(seq["font_size"], errors="coerce").to_numpy(float)
        nfs    = pd.to_numeric(seq["font_size"], errors="coerce").shift(-1).to_numpy(float)
        fs_max = np.maximum(cur_fs, nfs)
        fs_max = np.where((fs_max > 0) & np.isfinite(fs_max), fs_max, np.nan)
        gap_em = (nyt - cur_yb) / fs_max
    else:
        gap_em = np.full(len(seq), np.nan)

    new_run, vstack_id_seq = _assign_vstack_ids(seq, y_mov, x_mov, gap_em, config)
    seq["_vstack_id"] = vstack_id_seq

    # Run bounding-box width = max right edge - min left edge over the run's cells.
    grp = seq.groupby("_vstack_id")
    vstack_width_seq = (grp["x_right"].transform("max") - grp["x_left"].transform("min")).to_numpy()

    # ── Score each vstack run with >= 2 cells ──────────────────────────────────
    # Per-page content span (text extent, like the gutter extractor: max right -
    # min left over the page, NOT page_width). Each run sits on one page.
    if page_col:
        page_span_seq = (
            seq.groupby("page_number")["x_right"].transform("max")
            - seq.groupby("page_number")["x_left"].transform("min")
        ).to_numpy(float)
    else:
        page_span_seq = np.full(len(seq), cur_xr.max() - cur_xl.min(), dtype=float)

    vstack_n_cells_seq, vstack_n_lines_seq, vstack_score_seq = _score_vstack_runs(
        seq, new_run, vstack_width_seq, page_span_seq, y_mov, x_mov, config
    )

    # ── Map movement + stack annotations back onto cells by cell_id ────────────
    cell_ids  = seq["cell_id"].to_numpy()
    df["y_movement"]     = df["cell_id"].map(dict(zip(cell_ids, y_mov)))
    df["x_movement"]     = df["cell_id"].map(dict(zip(cell_ids, x_mov)))
    df["vstack_gap_em"]  = df["cell_id"].map(dict(zip(cell_ids, gap_em)))
    df["vstack_id"]      = df["cell_id"].map(dict(zip(cell_ids, vstack_id_seq)))
    df["vstack_width"]   = df["cell_id"].map(dict(zip(cell_ids, vstack_width_seq)))
    df["vstack_n_cells"] = df["cell_id"].map(dict(zip(cell_ids, vstack_n_cells_seq)))
    df["vstack_n_lines"] = df["cell_id"].map(dict(zip(cell_ids, vstack_n_lines_seq)))
    df["vstack_score"]   = df["cell_id"].map(dict(zip(cell_ids, vstack_score_seq)))

    # ── Final decision: merge scored vstacks into logical rows (grouped_row_id) ─
    df = assign_grouped_rows(df, config)

    return df

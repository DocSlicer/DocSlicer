# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Group cells across visual lines that form one logical table cell (annotation stage).

df_cells → df_cells + movement / vstack annotations.

Not used for OCR, which has no streaming order.

Purpose: group cells on different visual lines that form one logical table
cell — multi-line headers, and table cells that flex onto another line.
Annotation-only first stage of a multiline-cell grouper (nothing destroyed).

Background:
step_08 serialises cells in reading order (cell_id). In a multiline table region
a single logical row is spread over several cells, and the reading order walks
one *column* down before jumping to the next column — so the y of consecutive
cells zig-zags down/up as columns are traversed.

Option 1 tried to reconstruct rows directly from per-cell movement. This second
option starts more conservatively: it just *describes* the movement between
consecutive cells (in cell_id order), leaving the actual grouping decision for a
later pass that will be designed after inspecting these signals on real
documents.

What this step adds:
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

Vertical-stack runs:
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
                 between the two cells also breaks into a new id, as does a cell
                 whose text ends in a subheading colon ("Race/Ethnicity:",
                 "Race/Ethnicity:[^(a)]") — the next cell then starts a new id.
  vstack_width : the run's bounding-box width, max(x_right) - min(x_left) over
                 its cells (i.e. the width of the merged multi-line cell).
  vstack_n_cells: number of cells in the run (all runs, incl. size-1).
  vstack_n_lines: number of visible lines in the run = sum of each cell's
                 line_ids length (mcid: one cell can be several visible lines).
                 This — not the cell count — feeds the score_line_tiers band.
  vstack_score : how table-cell-like a run is, scored only for runs with >= 2
                 cells (else NaN). The ONLY thing that excludes a real stack
                 (-> NaN) is the alone-in-y-band hard gate (see
                 vstack_alone_in_band); every band below is a pure score.
                 Summed from:
                   width tier (vstack_width vs page content span = max x_right -
                     min x_left, like the gutter extractor): see config
                     score_width_tiers as lower bounds (ratio >= min): < 1/5 -> +3,
                     >= 1/5 -> +2, >= 1/4 -> 0, >= 1/3 -> -3, >= 1/2 -> -6; a
                     score, never an exclusion);
                   line tier (vstack_n_lines): score_line_tiers — a table cell is
                     a few lines (2-3 -> +3, 4 -> +2) and tall runs are penalised
                     rather than excluded (>= 7 -> -2, >= 10 -> -3, >= 15 -> -5);
                   grid: +score_grid_cell_match_bonus when every cell shares one
                     non-null grid_cell_id (all in the same detected grid cell);
                   table-rule: every cell sharing one non-null shape_id_tr_above
                     scores +score_shape_tr_above_bonus, shape_id_tr_below the
                     same, and both together score_shape_tr_both_bonus; but a
                     shape_id that is below one cell and above another in the run
                     is a divider cutting through it -> score_shape_tr_internal_penalty;
                   enter (both axes): +2 if entered UP+RIGHT, +1 if DOWN+LEFT;
                   exit  (both axes): +2 if it exits DOWN+LEFT, +2 if NONE+RIGHT.
  vstack_alone_in_band: bool — the alone-in-y-band hard gate's verdict. True when
                 the run is a >= 2-cell stack with no sibling cell beside it in the
                 y-band it spans (a centered title/heading), which is exactly the
                 case that blanks vstack_score to NaN. False otherwise, including
                 every size-1 run. Makes the one exclusion auditable rather than a
                 silent NaN. Gated by config.score_require_band_siblings.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ================================================================================
# CONFIG
# ================================================================================

@dataclass(frozen=True)
class RowGroupConfig:

    # --------------------
    # V-stack assignment
    # --------------------

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
    # Relaxed gap limit for a "bonded" step: when the two consecutive cells share
    # strong table evidence — one non-blank grid_cell_id, OR the same non-blank
    # shape_id_tr_above AND shape_id_tr_below together (both rules — one alone is
    # not enough) — they sit inside the same detected cell / rule-bounded row band,
    # so a wider vertical gap (e.g. a tall table cell) still continues the run.
    # Applied only when it exceeds vstack_max_gap_em; a NaN gap never breaks.
    vstack_bonded_gap_em: float = 2.0

    # A change in font styling (font_family, font_size, non_stroking_color)
    # between two otherwise-continuing cells starts a new vstack_id: a restyle
    # marks a new logical block even mid-column. Set False to ignore styling.
    vstack_break_on_style_change: bool = True
    # font_size equality slack (pt) when comparing styling: sizes within this of
    # each other are "the same size", so sub-point extraction jitter alone does
    # not split a run. font_family and non_stroking_color compare exactly.
    vstack_style_font_size_tol: float = 0.1

    # A cell whose text ends in a colon — optionally trailed by a footnote /
    # reference marker like "[^(a)]" or "(a)" — is a table subheading (e.g.
    # "Race/Ethnicity:"). The next cell then starts a new vstack_id even on an
    # otherwise-continuing (DOWN, NONE) step. Fires ONLY when the colon cell is the
    # first of its run (a subheading is a left-column start; a colon on a cell that
    # continued from above is mid-value, not a boundary). Set False to ignore.
    vstack_break_on_colon_subheading: bool = True

    # --------------------
    # V-stack scoring
    # --------------------

    # HARD GATE (not a score): reject a run whose merged bbox would be alone in the
    # y-band it spans — no other cell sits beside it (entirely left/right)
    # overlapping that band. That is a centered multi-line title/heading block, not
    # a table cell; a genuine column header shares its band with sibling columns.
    # This is the ONLY thing that blanks vstack_score to NaN for a real (>= 2 cell)
    # stack; it is recorded per cell in the vstack_alone_in_band column so the
    # rejection is auditable. Set False to disable the gate (nothing excluded).
    # See _runs_alone_in_y_band.
    score_require_band_siblings: bool = True

    # --- vstack scoring (only runs with >= 2 cells are scored) ---
    # Visible-line score table (vstack_n_lines: summed line_ids entries across the
    # run's cells, mcid-aware — one cell can already be several visible lines). Each
    # row is (min_lines, score), read as lower bounds tallest-first: a run takes the
    # score of the first row whose min_lines it still reaches (n_lines >= min_lines).
    # A table cell is a few lines (+); a tall run is body text and scored down (-)
    # rather than hard-excluded, so height is now a signal, not a gate. Keep sorted.
    score_line_tiers: tuple[tuple[int, float], ...] = (
        (30, -15.0),  # n_lines >= 30
        (20, -10.0),   # n_lines >= 20
        (15, -8.0),   # n_lines >= 15
        (12, -6.0),   # n_lines >= 15
        (10, -5.0),   # 10..14
        (7,  -3.0),   # 7..9
        (5,  +0.0),   # 5..6
        (4,  +2.0),   # 4
        (2,  +5.0),   # 2..3
    )                 # n_lines < 2 -> 0 (never happens: scored runs have >= 2 cells)
    # Width score table, same lower-bound shape as score_line_tiers. width_ratio is
    # vstack_width as a fraction of the page *content* span (max x_right - min x_left
    # over the page, like the gutter extractor — NOT page_width). Each row is
    # (min_ratio, score): a run takes the score of the WIDEST tier it still reaches
    # (ratio >= min_ratio). Width is a SCORE, not a gate — the widest tier just
    # carries the most negative points, so a wide run is penalised, not excluded,
    # and can still survive on other evidence (a wide table cell sharing one
    # grid_cell_id, say). The (0.0, ...) row is the base for the narrowest runs; a
    # NaN ratio (no page span) scores 0. Keep sorted narrow -> wide.
    score_width_tiers: tuple[tuple[float, float], ...] = (
        (0.0,   +3.0),   # ratio < 1/5   (narrowest — base)
        (1 / 5, +2.0),   # ratio >= 1/5
        (1 / 4,  0.0),   # ratio >= 1/4
        (1 / 3, -3.0),   # ratio >= 1/3
        (1 / 2, -4.0),   # ratio >= 1/2  (widest)
        (2 / 3, -5.0),   # ratio >= 1/2  (widest)
        (4 / 5, -6.0),   # ratio >= 1/2  (widest)
    )
    # Movement bonuses. Entering: how the cell before the run moved into it.
    # Exiting: how the run's last cell then moves on. A real column-stack is
    # typically entered from above/right and resumes flow afterwards. Each bonus
    # needs BOTH axes to match (e.g. UP and RIGHT together; UP with LEFT scores 0).
    score_enter_up_right_bonus:  float = 2.0   # preceded by UP and RIGHT
    score_enter_down_left_bonus: float = 1.0   # preceded by DOWN and LEFT
    score_exit_down_left_bonus:  float = 2.0   # followed by DOWN and LEFT
    score_exit_none_right_bonus: float = 2.0   # followed by NONE and RIGHT
    # Grid / table-rule bonuses. Each fires only when EVERY cell in the run shares
    # one non-null value of the column (a cell missing the id, or two cells
    # disagreeing, earns nothing). grid_cell_id: all cells fall in the same detected
    # grid cell — a near-certain table cell. shape_id_tr_above / _below: all cells
    # share the same table-rule line above / below; both rules together bound a real
    # table row, so that is scored much higher than either edge alone (replaces, not
    # adds to, the two single-side bonuses).
    score_grid_cell_match_bonus: float = 20.0  # all cells share one grid_cell_id
    score_shape_tr_above_bonus:  float = 2.0   # all cells share one shape_id_tr_above
    score_shape_tr_below_bonus:  float = 2.0   # all cells share one shape_id_tr_below
    score_shape_tr_both_bonus:   float = 10.0  # above AND below both matched
    # Internal table-rule penalty. When one shape_id is a cell's shape_id_tr_below
    # AND another cell's shape_id_tr_above within the SAME run, that rule line sits
    # between two of the run's cells — a horizontal divider cutting through the
    # stack, so the cells below it are a separate row and should not have merged.
    score_shape_tr_internal_penalty: float = -10.0  # a rule line splits the run

    # --------------------
    # Group winning V-stacks into rows
    # --------------------

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
    # Merge-eligibility gate. When True, only *winning* (seed) and *un-annotated*
    # (size-1, NaN-score) vstacks may join a row; a *losing* vstack (finite score
    # below grouped_min_score — a real >= 2-cell stack that scored badly) and an
    # *alone* vstack (vstack_alone_in_band, a centered title) are barred from every
    # union. Stops a stray winning run from dragging in the losing column or title
    # beside it — the seed is then alone and correctly rejected. False = no gate
    # (any aligned, line-adjacent pair may merge, the old behaviour).
    grouped_gate_ineligible: bool = True


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
    

def _style_val_eq(x, y) -> bool:
    """Equal styling values, treating two missing values as equal."""
    x_na, y_na = _is_na_scalar(x), _is_na_scalar(y)
    if x_na or y_na:
        return x_na and y_na
    return _norm_style_val(x) == _norm_style_val(y)


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


# A subheading colon: the text ends in ':' — optionally followed by a footnote /
# reference marker ('[^...]' or '(...)') and trailing whitespace — e.g.
# "Race/Ethnicity:" or "Race/Ethnicity:[^(a)]".
def _same_nonnull_next(seq: pd.DataFrame, col: str) -> np.ndarray:
    """
    Per position t: do cells t and t+1 carry the same *non-blank* value of ``col``?

    True only when both cells have the column present (non-NA) and equal — a shared
    non-null id spanning the pair (e.g. one grid_cell_id, or one table-rule shape
    id). Either side missing, or the two disagreeing, is False; so is the last
    position (no successor). Absent column -> all False.
    """
    n = len(seq)
    if col not in seq.columns:
        return np.zeros(n, dtype=bool)
    cur = seq[col].to_numpy(object)
    nxt = seq[col].shift(-1).to_numpy(object)
    return np.array(
        [not _is_na_scalar(a) and not _is_na_scalar(b)
         and _norm_style_val(a) == _norm_style_val(b)
         for a, b in zip(cur, nxt)],
        dtype=bool,
    )


_COLON_SUBHEADING_RE = re.compile(r":\s*(?:\[\^[^\]]*\]|\([^)]*\))?\s*$")


def _ends_in_colon_subheading(seq: pd.DataFrame) -> np.ndarray:
    """
    Per position t: does cell t's text end in a subheading colon?

    True when the text ends in ':' — optionally trailed by a footnote/reference
    marker ('[^...]' or '(...)') and whitespace (see _COLON_SUBHEADING_RE). Such a
    cell is a table subheading, so the run breaks *after* it: the next cell starts
    a new vstack_id even on an otherwise-continuing step. Missing text (or no
    'text' column) counts as not a subheading.
    """
    n = len(seq)
    if "text" not in seq.columns:
        return np.zeros(n, dtype=bool)
    return np.array(
        [isinstance(t, str) and _COLON_SUBHEADING_RE.search(t) is not None
         for t in seq["text"].to_numpy(object)],
        dtype=bool,
    )


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
    vertically — vstack_gap_em <= its gap limit (a larger gap is a separate block;
    a NaN gap never breaks). The limit is config.vstack_max_gap_em, raised to
    config.vstack_bonded_gap_em for a "bonded" step whose two cells share strong
    table evidence — one non-blank grid_cell_id, or the same non-blank
    shape_id_tr_above AND shape_id_tr_below together — since such cells sit inside
    one detected cell / rule-bounded band. When config.vstack_break_on_style_change
    is set, a change in font styling (font_family, font_size, non_stroking_color)
    also breaks the run, so a restyle starts a new id even mid-column. When
    config.vstack_break_on_colon_subheading is set, a cell whose text ends in a
    subheading colon (see _ends_in_colon_subheading) breaks the run after it —
    but only when that colon cell is itself the first of its run (a subheading is
    a left-column start; a colon on a cell continued from above is not a boundary).

    Returns (new_run, vstack_id_seq): new_run marks each run's first position
    (also consumed by scoring), vstack_id_seq is the dense 1..N id per position.
    """
    n = len(seq)

    # cont[i] is True when the i -> i+1 step goes straight down in the same column
    # AND the cells are close enough vertically. A gap above vstack_max_gap_em
    # means the next cell is a separate block (new paragraph/row), so the run
    # breaks there even though the step is (DOWN, NONE). NaN gap never breaks.
    # A step's gap limit is normally vstack_max_gap_em, but rises to
    # vstack_bonded_gap_em when the two cells share strong table evidence: one
    # non-blank grid_cell_id, OR the same non-blank shape_id_tr_above AND
    # shape_id_tr_below together (both rules — a rule-bounded row band). Such a
    # pair is inside one detected cell, so a wider gap still continues the run.
    gap_limit = np.full(n, config.vstack_max_gap_em, dtype=float)
    if config.vstack_bonded_gap_em > config.vstack_max_gap_em:
        bonded = (
            _same_nonnull_next(seq, "grid_cell_id")
            | (_same_nonnull_next(seq, "shape_id_tr_above")
               & _same_nonnull_next(seq, "shape_id_tr_below"))
        )
        gap_limit[bonded] = config.vstack_bonded_gap_em
    gap_ok = ~(gap_em > gap_limit)                  # NaN -> True (don't break)
    cont = (y_mov == "DOWN") & (x_mov == "NONE") & gap_ok

    # A restyle between two cells breaks the run even on a continuing step.
    if config.vstack_break_on_style_change:
        cont = cont & _same_style_next(seq, config.vstack_style_font_size_tol)

    # A subheading colon ("Race/Ethnicity:") breaks the run after that cell, so
    # the values below it start a fresh vstack_id even on a continuing step — but
    # only when the colon cell is itself the FIRST of its run. A subheading is
    # normally a left-column start: the step into it was a column jump, not a
    # continuation. A colon on a cell that physically continued from above (an
    # ongoing vstack) is not a subheading boundary and must not split the run.
    # `cont` here already reflects the gap/style breaks, so a colon cell right
    # after such a break still counts as a run-start.
    if config.vstack_break_on_colon_subheading:
        is_run_start = np.ones(n, dtype=bool)
        is_run_start[1:] = ~cont[:-1]
        cont = cont & ~(_ends_in_colon_subheading(seq) & is_run_start)

    # A new run begins at position 0 and wherever the previous step was not a
    # continuation; cumsum then numbers runs densely in reading order.
    new_run = np.ones(n, dtype=bool)
    new_run[1:] = ~cont[:-1]
    vstack_id_seq = np.cumsum(new_run)             # dense 1..N in reading order
    return new_run, vstack_id_seq


# ================================================================================
# VSTACK EXCLUSION
# ================================================================================

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


# ================================================================================
# VSTACK SCORING
# ================================================================================

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


def _tier_score(
    value: np.ndarray,
    tiers: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """
    Map each value to its tier score against a lower-bound table.

    Tiers are (min_value, score): a value takes the score of the HIGHEST tier it
    still reaches (value >= min_value). Assigning smallest-first lets the largest
    match win by overwrite, so tier order in the config is not load-bearing. A
    value below every tier — or NaN (comparisons are all False) — scores 0.

    Shared by the width and line bands (see RowGroupConfig.score_width_tiers /
    score_line_tiers). Both are pure scores, never exclusions: the widest / tallest
    tier just carries the most negative points, and the config's smallest tier is
    the base for everything below it (e.g. a (0.0, +3) row rewards the narrowest
    runs).
    """
    out = np.zeros(np.shape(value), dtype=float)
    for min_value, score in sorted(tiers):                # smallest -> largest
        out[value >= min_value] = score
    return out


def _run_all_same_nonnull(
    seq:        pd.DataFrame,
    col:        str,
    run_of_pos: np.ndarray,
    n_runs:     int,
) -> np.ndarray:
    """
    Per run: do all its cells share one and the same non-null value of ``col``?

    True only when every cell in the run carries the column and they all agree —
    a single distinct non-null value spanning the whole run. A missing value on
    any cell, or two cells disagreeing, is False. A run whose cells all share one
    grid_cell_id (or one table-rule shape id) is strong table evidence.

    ``run_of_pos`` is the run index (0..n_runs-1) of each sequence position; runs
    are contiguous, so this is cumsum(new_run) - 1. Absent column -> all False.
    """
    size = np.bincount(run_of_pos, minlength=n_runs)
    if col not in seq.columns:
        return np.zeros(n_runs, dtype=bool)
    grp = pd.Series(seq[col].to_numpy(object)).groupby(run_of_pos)
    idx = range(n_runs)
    n_distinct = grp.nunique(dropna=True).reindex(idx, fill_value=0).to_numpy()
    n_present  = grp.count().reindex(idx, fill_value=0).to_numpy()   # excludes NA
    return (n_distinct == 1) & (n_present == size)


def _run_has_internal_rule(
    seq:        pd.DataFrame,
    above_col:  str,
    below_col:  str,
    run_of_pos: np.ndarray,
    n_runs:     int,
) -> np.ndarray:
    """
    Per run: does one shape_id sit *below* one cell and *above* another within the
    same run — i.e. a table-rule line crossing between the run's cells?

    A horizontal rule at some y is the shape_id_tr_below of the cell above it and
    the shape_id_tr_above of the cell below it. So a shape_id present in BOTH the
    run's below values and its above values is sandwiched between two of the run's
    cells: the divider splits the stack into separate rows and the run should not
    have merged across it. (The two edges of any single cell are different lines,
    so a match always spans two distinct cells.) Missing values are ignored; either
    column absent -> all False.
    """
    out = np.zeros(n_runs, dtype=bool)
    if above_col not in seq.columns or below_col not in seq.columns:
        return out
    frame = pd.DataFrame({
        "run":   run_of_pos,
        "above": seq[above_col].to_numpy(object),
        "below": seq[below_col].to_numpy(object),
    })
    for run, g in frame.groupby("run", sort=False):
        above = {v for v in g["above"] if not _is_na_scalar(v)}
        if above and above & {v for v in g["below"] if not _is_na_scalar(v)}:
            out[run] = True
    return out


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
    Score each contiguous vstack run. A run is a candidate only if it is a real
    stack (>= 2 cells); a size-1 run gets NaN. Every candidate's raw score is
    computed from the bands below — width and height are pure scores, never gates.

    All inputs are aligned to the reading-order sequence positions. ``new_run``
    marks each run's first position, so runs are contiguous blocks of the
    sequence — start/end positions and per-run sizes follow directly.

    Score = width tier + line tier + grid/shape bonus + enter bonus + exit bonus:
      width tier: config.score_width_tiers via _tier_score (vstack_width as a
          fraction of the page content span). A pure score: the widest tier just
          carries the most negative points, it does NOT exclude.
      line tier: config.score_line_tiers via _tier_score (vstack_n_lines).
          Never excludes — a tall run is scored down, not dropped.
      grid: +score_grid_cell_match_bonus when every cell shares one non-null
          grid_cell_id (all in the same detected grid cell).
      table-rule: every cell sharing one non-null shape_id_tr_above scores
          +score_shape_tr_above_bonus, shape_id_tr_below the same; when BOTH
          match the run scores score_shape_tr_both_bonus instead of their sum.
      internal rule: +score_shape_tr_internal_penalty when one shape_id is a
          cell's shape_id_tr_below and another cell's shape_id_tr_above in the
          same run (a divider line cuts through the stack; see
          _run_has_internal_rule).
      enter (movement of the cell *before* the run into it; both axes required):
          UP   AND RIGHT -> +score_enter_up_right_bonus
          DOWN AND LEFT  -> +score_enter_down_left_bonus
      exit (the run's last cell's outgoing movement; both axes required):
          DOWN AND LEFT  -> +score_exit_down_left_bonus
          NONE AND RIGHT -> +score_exit_none_right_bonus

    Visible-line and grid/shape signals count *visible lines* / per-cell ids over
    the run — mcid means one cell can already be several visible lines.

    The ONE hard gate (config.score_require_band_siblings) then blanks the score
    to NaN for any candidate that would be alone in its y-band (see
    _runs_alone_in_y_band) — a centered title/heading, not a table cell. That
    rejection is returned separately as ``alone`` so the caller can surface it as
    vstack_alone_in_band. ``seq`` supplies the per-cell bboxes those checks need.

    Returns (n_cells, n_lines, score, alone) arrays, each broadcast over the run's
    positions and aligned to the sequence (callers map them back by cell_id).
    ``alone`` is the band-sibling gate verdict (False for non-candidate runs).
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

    # Width tier (narrower is more table-cell-like). A pure score: wide runs take
    # the widest tier's negative points, never excluded.
    width_score = _tier_score(ratio, config.score_width_tiers)
    # Line tier (a few lines is table-cell-like; tall runs scored down, not cut).
    line_score = _tier_score(n_lines, config.score_line_tiers)

    # Grid / table-rule bonuses: every cell of the run sharing one non-null id.
    run_of_pos = np.cumsum(new_run) - 1                   # run index per position
    grid_match = _run_all_same_nonnull(seq, "grid_cell_id", run_of_pos, len(starts))
    above_match = _run_all_same_nonnull(seq, "shape_id_tr_above", run_of_pos, len(starts))
    below_match = _run_all_same_nonnull(seq, "shape_id_tr_below", run_of_pos, len(starts))
    grid_bonus = np.where(grid_match, config.score_grid_cell_match_bonus, 0.0)
    # Both rules together bound a real row -> flat _both_ bonus, not the sum.
    shape_bonus = np.where(
        above_match & below_match, config.score_shape_tr_both_bonus,
        np.where(above_match, config.score_shape_tr_above_bonus, 0.0)
        + np.where(below_match, config.score_shape_tr_below_bonus, 0.0),
    )
    # A rule line running between the run's cells (one shape_id both below one
    # cell and above another) splits the stack into separate rows -> penalty.
    internal_rule = _run_has_internal_rule(
        seq, "shape_id_tr_above", "shape_id_tr_below", run_of_pos, len(starts)
    )
    internal_penalty = np.where(internal_rule, config.score_shape_tr_internal_penalty, 0.0)

    enter_bonus = (
        np.where((prev_y == "UP")   & (prev_x == "RIGHT"), config.score_enter_up_right_bonus,  0.0)
        + np.where((prev_y == "DOWN") & (prev_x == "LEFT"),  config.score_enter_down_left_bonus, 0.0)
    )
    exit_bonus = (
        np.where((next_y == "DOWN") & (next_x == "LEFT"),  config.score_exit_down_left_bonus,  0.0)
        + np.where((next_y == "NONE") & (next_x == "RIGHT"), config.score_exit_none_right_bonus, 0.0)
    )
    raw_score = (
        width_score + line_score + grid_bonus + shape_bonus
        + internal_penalty + enter_bonus + exit_bonus
    )

    # A run is a scoring CANDIDATE when it is an actual stack (>= 2 cells). Width
    # and height feed the raw score above but never gate — a size-1 run is the
    # only thing the score bands themselves exclude.
    candidate = n_cells >= 2

    # ── Hard gate: alone in its y-band ─────────────────────────────────────────
    # The one exclusion. A candidate whose merged bbox has no sibling cell beside
    # it (a centered title/heading, not a table cell) is gated out. Computed for
    # every candidate so `alone` is a full per-run verdict the caller surfaces as
    # vstack_alone_in_band; a non-candidate run is never alone (False).
    alone = np.zeros(len(starts), dtype=bool)
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
            candidate_mask=candidate, x_tol=config.x_tol,
        )

    scored = candidate & ~alone
    run_score = np.where(scored, raw_score, np.nan)

    # Broadcast per-run values back over the sequence (runs are contiguous and
    # ordered, so a repeat by run size reconstructs the sequence layout).
    return (
        np.repeat(n_cells, n_cells),
        np.repeat(n_lines, n_cells),
        np.repeat(run_score, n_cells),
        np.repeat(alone, n_cells),
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

    ...and both are merge-ELIGIBLE (config.grouped_gate_ineligible). Only winning
    (seed) and un-annotated (size-1, NaN-score — a lone single-line cell) vstacks
    are eligible; a losing vstack (finite score below grouped_min_score, a real
    >= 2-cell stack that scored badly) and an alone vstack (vstack_alone_in_band,
    a centered title) never join a row. This is the gate that stops a stray
    winning run from dragging in the losing column or centered title beside it —
    the seed is then alone in its component and rejected below.

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

    # vstack_alone_in_band marks a centered title/heading (the alone-in-band gate);
    # absent when scoring's band gate was disabled -> nothing is alone.
    if "vstack_alone_in_band" in df.columns:
        alone_col = df["vstack_alone_in_band"].fillna(False).to_numpy(bool)
    else:
        alone_col = np.zeros(len(df), dtype=bool)

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
        "alone":     alone_col,
    })

    # Collapse to one row per (page, vstack): its bbox, line span, score, and
    # alone flag (all uniform within a vstack; max just ignores the NaN dups).
    vs = (cell.groupby(["page", "vstack_id"], sort=True)
              .agg(x_left=("x_left", "min"), x_right=("x_right", "max"),
                   y_top=("y_top", "min"),   y_bottom=("y_bottom", "max"),
                   line_lo=("line_lo", "min"), line_hi=("line_hi", "max"),
                   score=("score", "max"), alone=("alone", "max"))
              .reset_index())
    n = len(vs)
    y_top = vs["y_top"].to_numpy(float)
    y_bot = vs["y_bottom"].to_numpy(float)
    y_cen = (y_top + y_bot) / 2.0
    l_lo  = vs["line_lo"].to_numpy(float)
    l_hi  = vs["line_hi"].to_numpy(float)
    pg    = vs["page"].to_numpy()
    score = vs["score"].to_numpy(float)
    alone = vs["alone"].to_numpy(bool)
    vsid  = vs["vstack_id"].to_numpy()

    # Merge eligibility (the gate). A vstack falls in one of four buckets by its
    # score: winning (finite, >= grouped_min_score — the seeds), losing (finite,
    # < min — a real >= 2-cell stack that scored badly, e.g. a body-text column),
    # un-annotated (NaN because it is a size-1 run: a lone single-line cell like a
    # stub first column), and alone (vstack_alone_in_band — a centered title whose
    # score was blanked by scoring's band gate). Only winning and un-annotated may
    # merge; losing and alone vstacks never join a row. This stops a stray winning
    # run from dragging in the losing column or centered title beside it — the seed
    # is then alone in its component and correctly rejected below.
    is_winning = np.isfinite(score) & (score >= config.grouped_min_score)
    is_losing  = np.isfinite(score) & (score <  config.grouped_min_score)
    eligible   = ~is_losing & ~alone if config.grouped_gate_ineligible else np.ones(n, bool)

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
            # Losing / alone vstacks never merge (see `eligible`). `continue`, not
            # `break`: an ineligible vstack mid-window must not stop the scan from
            # reaching an eligible pair further on.
            if not (eligible[i] and eligible[j]):
                continue
            aligned = (abs(y_top[i] - y_top[j]) <= tol
                       or abs(y_bot[i] - y_bot[j]) <= tol
                       or abs(y_cen[i] - y_cen[j]) <= tol)
            if aligned:
                uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(n)])

    # A component becomes a logical row only if it holds a seed (survivor) AND
    # actually merged (>= 2 vstacks). A lone seed that grouped with no neighbour,
    # and any vstack outside a seeded component, is left blank (rejected here).
    seed_root = set(roots[is_winning].tolist())
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
# FINAL REINDEX (cell_id / line_id)
# ================================================================================

def reindex_grouped_ids(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass: fold the grouped structures back into the cell/line id space.

    A vstack that was linked into a logical row (grouped_row_id set) is one
    logical table cell -> all its cells share one cell_id (keyed by page +
    vstack_id). The whole group of linked neighbours is one logical row -> all
    its cells share one line_id (keyed by page + grouped_row_id). Ungrouped
    cells keep their own cell and line identity.

    Both ids are then renumbered densely (1..N) by first appearance — cells in
    reading order (page, old cell_id), lines in line order (page, earliest old
    line, old cell_id) — so a merged row sits at the earliest line it swallowed
    and downstream steps see an ordinary cells table again. The originals are
    preserved in cell_id_orig / line_id_orig. line_ids (when present) is
    rewritten to the single new line id: the old per-visual-line numbering no
    longer exists after the renumber.

    Returns df_cells unchanged when grouping never ran (no grouped_row_id) or
    the id columns it needs are missing.
    """
    required = {"cell_id", "vstack_id", "grouped_row_id"}
    if df_cells is None or df_cells.empty or not required.issubset(df_cells.columns):
        return df_cells
    bounds = _cell_line_bounds(df_cells)
    if bounds is None:
        return df_cells
    line_lo, _ = bounds

    df = df_cells.copy()
    n = len(df)
    page = (df["page_number"].to_numpy() if "page_number" in df.columns
            else np.zeros(n, dtype=int))
    grouped  = df["grouped_row_id"].notna().to_numpy()
    old_cell = df["cell_id"].to_numpy()
    vstack   = df["vstack_id"].to_numpy()
    grow     = df["grouped_row_id"].to_numpy()
    old_line = df["line_id"].to_numpy() if "line_id" in df.columns else line_lo

    # Identity keys: a grouped vstack is one cell; a grouped row is one line.
    # The tag keeps grouped and ungrouped keys in disjoint namespaces.
    cell_key = [
        ("v", page[i], vstack[i]) if grouped[i] else ("c", old_cell[i])
        for i in range(n)
    ]
    line_key = [
        ("g", page[i], grow[i]) if grouped[i] else ("l", page[i], old_line[i])
        for i in range(n)
    ]

    # Dense 1..N by first appearance in the respective order (lexsort keys are
    # listed minor -> major).
    cell_order = np.lexsort((old_cell, page))
    line_order = np.lexsort((old_cell, line_lo, page))
    new_cell_of: dict = {}
    for i in cell_order:
        new_cell_of.setdefault(cell_key[i], len(new_cell_of) + 1)
    new_line_of: dict = {}
    for i in line_order:
        new_line_of.setdefault(line_key[i], len(new_line_of) + 1)

    df["cell_id_orig"] = df["cell_id"]
    df["cell_id"]      = [new_cell_of[k] for k in cell_key]
    if "line_id" in df.columns:
        df["line_id_orig"] = df["line_id"]
    df["line_id"] = [new_line_of[k] for k in line_key]
    if "line_ids" in df.columns:
        df["line_ids"] = [[int(v)] for v in df["line_id"].to_numpy()]

    return df


def sync_word_ids(df_words: pd.DataFrame, df_reindexed: pd.DataFrame) -> pd.DataFrame:
    """
    Carry the reindex_grouped_ids renumbering onto df_words so word↔cell ids
    stay aligned.

    ``df_reindexed`` is the cells frame straight out of :func:`reindex_grouped_ids`
    (one row per original cell, still carrying ``cell_id_orig``): its ``cell_id``
    holds the new dense id and ``cell_id_orig`` the old one, so old->new is a plain
    per-cell lookup. Each word follows its cell — its new ``cell_id`` and (when both
    frames have it) ``line_id`` are taken from that cell's reindexed row, since a
    merged multi-line cell now sits on a single new line whose number does not
    follow from the word's old line_id alone.

    Words whose cell_id is absent from the mapping keep their existing ids. Returns
    df_words unchanged when the reindex never ran (no ``cell_id_orig``) or df_words
    lacks ``cell_id``.
    """
    if (df_words is None or df_words.empty
            or "cell_id" not in df_words.columns
            or df_reindexed is None or "cell_id_orig" not in df_reindexed.columns):
        return df_words

    old = df_reindexed["cell_id_orig"].to_numpy()
    cell_map = dict(zip(old, df_reindexed["cell_id"].to_numpy()))

    df_words = df_words.copy()
    orig_cell = df_words["cell_id"]                         # keyed on the old id
    df_words["cell_id"] = (
        orig_cell.map(cell_map).fillna(orig_cell).astype(np.int64)
    )
    if "line_id" in df_words.columns and "line_id" in df_reindexed.columns:
        line_map = dict(zip(old, df_reindexed["line_id"].to_numpy()))
        df_words["line_id"] = (
            orig_cell.map(line_map).fillna(df_words["line_id"]).astype(np.int64)
        )
    return df_words


def rebuild_merged_cells(
    df_cells: pd.DataFrame,
    text_sep: str = " ",
) -> pd.DataFrame:
    """
    Physically collapse the cells that reindex_grouped_ids merged, so cell_id is
    unique again — one df row per cell instead of N stacked rows sharing an id.

    reindex_grouped_ids gave every cell of a merged vstack the same cell_id but
    left them as separate rows. The merged row is rebuilt by running just those
    rows back through the central registry aggregator (:func:`aggregate_to`,
    grouped by cell_id) — the same rules the words -> cells step uses — so the
    metadata is correct, not just copied off the top row:

      counts (char_count, alpha_count, word_count, ...) : summed
      ratios (bold_ratio, italic_ratio, ...)            : char_count-weighted mean
      style  (font_size, font_family, colors, align)    : dominant (most-alpha row)
      bbox / width / height                             : merged + recomputed
      is_bold / is_italic / is_uppercase                : recomputed from the above

    Call-site exceptions:
      text            : the rows' text joined top-to-bottom (y_top, x_left order)
                        with ``text_sep`` (default a space). This does NOT
                        dehyphenate a word split across the seam between two
                        stacked lines ("inter-" / "national") — that needs the
                        word table; within each original cell it already happened.
      word_ids        : flattened into one ordered list (UNIQUE_LIST).
      line_id         : kept (FIRST) — the registry drops it as a group key, but
                        here it carries the reindexed value (uniform in the cell).
      font_size_ratio : kept (FIRST). Cells that merge into one are almost
                        certainly the same font, so the top row's ratio holds;
                        recomputing it here would use only this handful of cells
                        as the median reference and corrupt it.

    Annotation columns the registry doesn't know (vstack_*, grouped_row_*,
    cell_id_orig, line_ids, y_center, movement/gap) fall through to FIRST; the
    movement/gap ones describe the pre-merge transitions and are stale on a
    rebuilt row. Any column aggregate_to would otherwise drop is backfilled from
    the cell's first row so nothing is silently lost.

    Cells that did not merge (a unique cell_id) pass through untouched, so the
    aggregation only runs on the handful of merged cells. Returns df_cells
    unchanged when it never went through the reindex (no cell_id_orig) or lacks
    cell_id.
    """
    if (df_cells is None or df_cells.empty
            or "cell_id" not in df_cells.columns
            or "cell_id_orig" not in df_cells.columns):
        return df_cells

    dup = df_cells["cell_id"].duplicated(keep=False)
    merged = df_cells[dup]
    if merged.empty:
        return df_cells.reset_index(drop=True)

    from .._utils.df_aggregation.registry_aggregator import (
        Agg, aggregate_to, _compute_derived,
    )

    # Row order within a cell drives the text join and FIRST picks: top to bottom.
    sort_cols = ["cell_id"] + [c for c in ("y_top", "x_left") if c in merged.columns]
    merged = merged.sort_values(sort_cols, kind="mergesort")

    overrides = {
        "text":            lambda s: text_sep.join(
            str(t) for t in s if t is not None and not (isinstance(t, float) and np.isnan(t))
        ),
        "word_ids":        Agg.UNIQUE_LIST,
        "line_id":         Agg.FIRST,
        "font_size_ratio": Agg.FIRST,   # merged cells share a font; keep top row's
    }
    grouped = aggregate_to(
        merged, by="cell_id", overrides=overrides,
        derived=False, on_unknown="silent",
    )

    # Recompute geometry + style flags at the merged level, but preserve the
    # FIRST font_size_ratio (derived would recompute it against this tiny subset).
    saved_fsr = grouped["font_size_ratio"].copy() if "font_size_ratio" in grouped.columns else None
    grouped = _compute_derived(grouped)
    if saved_fsr is not None:
        grouped["font_size_ratio"] = saved_fsr

    # Backfill any original column the aggregator dropped (unknown/DROP columns
    # not otherwise reconstructed) from the cell's first row, so nothing is lost.
    missing = [c for c in df_cells.columns if c not in grouped.columns]
    if missing:
        first = merged.drop_duplicates("cell_id", keep="first")[["cell_id"] + missing]
        grouped = grouped.merge(first, on="cell_id", how="left")

    grouped = grouped[df_cells.columns.tolist()]
    out = pd.concat([df_cells[~dup], grouped], ignore_index=True)
    out = out.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
    return out


# ================================================================================
# ENTRY POINT
# ================================================================================

def group_multiline_cells(
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame | None = None,
    config: RowGroupConfig = CONFIG,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annotate cells with their vertical center and cell-to-cell movement.

    Adds y_center, y_movement, x_movement (see module docstring). The later passes
    reindex cell_id/line_id when they merge multi-line cells into logical rows; pass
    ``df_words`` to have that renumbering carried onto the word rows too, so the two
    frames stay in sync (see :func:`sync_word_ids`).

    Returns df_cells alone when called without df_words (back-compat), or the
    (df_cells, df_words) pair when df_words is given.
    """
    required = {"cell_id", "x_left", "x_right", "y_top", "y_bottom"}
    if df_cells is None or df_cells.empty or not required.issubset(df_cells.columns):
        return df_cells if df_words is None else (df_cells, df_words)

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

    vstack_n_cells_seq, vstack_n_lines_seq, vstack_score_seq, vstack_alone_seq = _score_vstack_runs(
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
    df["vstack_alone_in_band"] = df["cell_id"].map(dict(zip(cell_ids, vstack_alone_seq)))

    # ── Final decision: merge scored vstacks into logical rows (grouped_row_id) ─
    df = assign_grouped_rows(df, config)

    # ── Fold groups into the id space: 1 vstack -> 1 cell_id, 1 group -> 1
    # line_id, both renumbered densely (originals in cell_id_orig/line_id_orig).
    df = reindex_grouped_ids(df)

    # ── Carry the cell_id/line_id renumbering onto the word rows (before the
    # rebuild collapses the per-original-cell rows the mapping is read from).
    if df_words is not None:
        df_words = sync_word_ids(df_words, df)

    # ── Physically collapse each merged cell_id back to a single row.
    df = rebuild_merged_cells(df)

    return df if df_words is None else (df, df_words)

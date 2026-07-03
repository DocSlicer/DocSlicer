"""
step_07b_stream_group_order.py

# ==============================================================================
# STREAM-GROUP READING-ORDER SHUFFLE
# ==============================================================================
#
# Goal
# ----
# step_07 assigns every word a ``stream_group_id`` — a contiguous run of the PDF
# content stream that behaves like one logical reading segment. Content-stream
# order approximates reading order but *between* groups it can be wrong (a tall
# right-hand callout emitted early, a title emitted after its own column, etc.).
#
# This stage takes those groups as opaque bounding boxes and re-sequences them
# into human reading order, writing a per-word ``reading_order`` rank. It does
# NOT look at text — only at each group's union bbox.
#
# It then sorts the words into that reading order and assigns ``line_id`` (via
# the shared line merger), with one rule on top: all words sharing a
# ``table_row_id`` land on a single line, even when the row wraps onto several
# visual lines.
#
# The algorithm (per page)
# ------------------------
# Repeatedly pick the current "top-left" group, then emit it once everything
# that must be read before it has been emitted:
#
#   pick_top_left(S):
#       1. L = the group in S with the smallest x_left (the leftmost).
#       2. Among the groups in S whose x-range overlaps L's x-range, return the
#          one with the smallest y_top.
#      Anchoring on the leftmost group's x-column (step 2) is what stops us from
#      grabbing a box that merely has the globally lowest y_top but lives in a
#      different column (e.g. a right-hand table on the McKinsey contents page).
#
#   process(G):
#       a. LEFT-BLOCKERS. While some still-unresolved group sits inside G's
#          y-range AND starts to the left of G, G cannot be read yet. Mask G
#          (put it aside) and recursively process the top-left of those blockers
#          first. This is what defers a very tall right-hand box until the
#          left/middle columns nested inside its y-range are done.
#       b. EMIT G.
#       c. FOLLOWERS. G has now "reserved" its y-band: resolve every group that
#          falls inside it, picking the top-left each time and recursing (each
#          follower runs the same process). A follower whose own y-range reaches
#          further down pulls in whatever sits inside it, so a tall first column
#          sweeps its sibling columns before the page moves on.
#
# It is the same recursive step everywhere — top level, left-blockers, and
# followers all just "pick the top-left and process it".
#
# Masking (a group on the recursion stack is invisible to the scans below it) is
# what keeps a deferred box out of the way while its nested content — including
# content the box does not strictly block — is resolved around it.
#
# Out of scope
# ------------
# Vertical (TTB / BTT) groups are masked out up front and re-attached at the end
# of their page; placing them precisely is a separate concern.
#
# See ``assign_stream_group_order`` for the public entry point.
# ==============================================================================
"""

# NOTE: This algo is left to right biased. For slides that have a columnar layout, a new algo should be added

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .._utils.layout.line_merger import assign_line_id

# ================================================================================
# TOLERANCES
# ================================================================================

# Two groups "share a y-band" only when their vertical overlap exceeds this, so a
# grazing 1-pt touch does not chain unrelated groups together.
_Y_OVERLAP_TOL: float = 2.0

# A group is "to the left of" G when its left edge is at least this far left of
# G's left edge — a small dead-zone so near-aligned columns do not jitter.
_X_LEFT_TOL: float = 3.0

# Two groups' x-ranges "overlap" (for the top-left anchor) when they share more
# than this much horizontal extent.
_X_OVERLAP_TOL: float = 1.0

_GROUP_KEY: list[str] = ["page_number", "stream_group_id"]


# ================================================================================
# GROUP-LEVEL BOXES
# ================================================================================

def _build_group_boxes(df_words: pd.DataFrame) -> pd.DataFrame:
    """Collapse words to one union-bbox row per ``(page_number, stream_group_id)``.

    Returns a frame with ``page_number, stream_group_id, x_left, y_top, x_right,
    y_bottom, is_vertical``. ``is_vertical`` is True when most of the group's
    words are TTB/BTT — such groups are excluded from the ordering walk.
    """
    required = ["page_number", "stream_group_id", "x_left", "y_top", "x_right", "y_bottom"]
    missing = [c for c in required if c not in df_words.columns]
    if missing:
        raise KeyError(f"_build_group_boxes: missing required columns: {missing}")

    g = df_words.groupby(_GROUP_KEY, sort=False, dropna=False)
    boxes = g.agg(
        x_left=("x_left", "min"),
        y_top=("y_top", "min"),
        x_right=("x_right", "max"),
        y_bottom=("y_bottom", "max"),
    )

    if "text_orientation" in df_words.columns:
        vert = df_words["text_orientation"].isin(["TTB", "BTT"])
        frac = vert.groupby([df_words["page_number"], df_words["stream_group_id"]]).mean()
        boxes["is_vertical"] = frac.reindex(boxes.index).fillna(0.0).to_numpy() > 0.5
    else:
        boxes["is_vertical"] = False

    return boxes.reset_index()


# ================================================================================
# PER-PAGE SOLVER
# ================================================================================

class _PageSolver:
    """Order the (horizontal) stream groups of a single page into reading order.

    Works purely on the group bboxes, indexed by their position 0..m-1 in the
    arrays passed to the constructor. ``solve`` returns those positions in
    reading order.
    """

    def __init__(
        self,
        xl: np.ndarray,
        yt: np.ndarray,
        xr: np.ndarray,
        yb: np.ndarray,
    ) -> None:
        self.xl, self.yt, self.xr, self.yb = xl, yt, xr, yb
        self.n = len(xl)
        self._resolved = np.zeros(self.n, dtype=bool)
        self._masked = np.zeros(self.n, dtype=bool)
        self._order: list[int] = []

    # --- geometry -------------------------------------------------------------

    def _y_overlaps(self, i: int, g: int) -> bool:
        return min(self.yb[i], self.yb[g]) - max(self.yt[i], self.yt[g]) > _Y_OVERLAP_TOL

    def _x_overlaps(self, i: int, j: int) -> bool:
        return min(self.xr[i], self.xr[j]) - max(self.xl[i], self.xl[j]) > _X_OVERLAP_TOL

    def _is_left_of(self, i: int, g: int) -> bool:
        return self.xl[i] < self.xl[g] - _X_LEFT_TOL

    # --- candidate sets -------------------------------------------------------

    def _active(self) -> list[int]:
        return [i for i in range(self.n) if not self._resolved[i] and not self._masked[i]]

    def _blocked_by_masked(self, h: int) -> bool:
        """True if a masked (deferred) group sits entirely to h's left within h's
        y-band.

        A masked group is one we have set aside to resolve its own left-blockers
        (step a). It still reserves its slot: anything in a separate column to its
        right must read *after* it, so we must not resolve ``h`` while such a
        group is pending. Only a strictly-left, x-disjoint masked group counts —
        a masked box that h sits *inside* (x-overlapping, e.g. a tall enclosing
        box) does not defer h, which is what keeps nested content resolving ahead
        of the box that encloses it.
        """
        for m in range(self.n):
            if (self._masked[m]
                    and self._y_overlaps(m, h)
                    and self.xr[m] <= self.xl[h] + _X_OVERLAP_TOL):
                return True
        return False

    def _pick_top_left(self, cands: list[int]) -> int:
        """Pick the group to read next from ``cands``.

        Base rule: the leftmost group's x-column, then the topmost group within
        it. Then one correction: if a group floats entirely *above* that pick's
        horizontal band, it reads first (recurse on the groups above the band).

        The band is the pick ``g`` plus every group it y-overlaps, so a box that
        sits *beside* ``g`` (same band, e.g. a taller neighbouring column) raises
        the band's top edge and is NOT treated as floating above — only a group
        clear of the whole band counts. This is what makes a centred heading
        above a row of left-aligned stats read before the leftmost stat, while a
        bottom-left title whose band still reaches the columns beside it stays
        first (contents page).
        """
        if len(cands) == 1:
            return cands[0]

        # 1. leftmost group (x_left, then y_top as a deterministic tie-break)
        anchor = min(cands, key=lambda i: (self.xl[i], self.yt[i]))
        # 2. topmost group whose x-range overlaps the anchor's column
        column = [i for i in cands if self._x_overlaps(i, anchor)]
        g = min(column, key=lambda i: (self.yt[i], self.xl[i]))

        # 3. top edge of g's horizontal band (g + everything it y-overlaps)
        band_top = self.yt[g]
        for i in cands:
            if i != g and self._y_overlaps(i, g):
                band_top = min(band_top, self.yt[i])

        # 4. groups floating entirely above that band read first
        above = [
            i for i in cands
            if i != g
            and not self._y_overlaps(i, g)
            and self.yb[i] <= band_top + _Y_OVERLAP_TOL
        ]
        if above:
            return self._pick_top_left(above)
        return g

    # --- walk -----------------------------------------------------------------

    def _process(self, g: int) -> None:
        # (a) Resolve everything nested to the left of G within G's y-band first.
        #     G is masked while we do so, so its deferral also hides it from the
        #     content resolved around it.
        while True:
            blockers = [
                i for i in self._active()
                if i != g and self._y_overlaps(i, g) and self._is_left_of(i, g)
            ]
            if not blockers:
                break
            self._masked[g] = True
            self._process(self._pick_top_left(blockers))
            self._masked[g] = False

        # (b) Emit G.
        self._resolved[g] = True
        self._order.append(g)

        # (c) G has reserved its y-band: resolve every group that falls inside
        #     it, picking the top-left each time (recursively — each follower
        #     runs the same process, so its own left-blockers and y-band are
        #     handled in turn). A group whose y-range extends below G's is
        #     picked up here and, in its own step (a)/(c), pulls in whatever sits
        #     inside its larger band — this is what lets a tall first column
        #     sweep its sibling columns before the page's lower content.
        #
        #     Followers to the right of a currently-masked group are skipped:
        #     that masked group is a deferred column reserving its slot, so its
        #     right-hand neighbours must wait for it (they surface again once it
        #     is unmasked and emitted). Without this, resolving a masked box's
        #     left-blocker could sweep in columns sitting to the box's right and
        #     emit them ahead of it.
        while True:
            in_band = [
                h for h in self._active()
                if self._y_overlaps(h, g) and not self._blocked_by_masked(h)
            ]
            if not in_band:
                break
            self._process(self._pick_top_left(in_band))

    def solve(self) -> list[int]:
        if self.n == 0:
            return []
        # The walk recurses at most once per group; give deep pages headroom.
        if self.n + 50 > sys.getrecursionlimit():
            sys.setrecursionlimit(self.n * 3 + 100)
        while True:
            act = self._active()
            if not act:
                break
            self._process(self._pick_top_left(act))
        return self._order


# ================================================================================
# PUBLIC API
# ================================================================================

def order_group_boxes(boxes: pd.DataFrame) -> pd.DataFrame:
    """Assign a per-page ``reading_order`` rank to every group box.

    ``boxes`` is one row per group (see :func:`_build_group_boxes`). Horizontal
    groups are ordered by the reading-order walk; vertical groups are appended
    after them on their page, sorted top-to-bottom / left-to-right. Returns
    ``boxes`` with an added int ``reading_order`` that is globally sequential
    (pages in ascending order, groups within a page in reading order).
    """
    out = boxes.copy()
    out["reading_order"] = -1
    rank = 0

    for _page, page_boxes in out.groupby("page_number", sort=True):
        idx = page_boxes.index.to_numpy()
        is_vert = page_boxes["is_vertical"].to_numpy()

        horiz_pos = np.flatnonzero(~is_vert)
        vert_pos = np.flatnonzero(is_vert)

        if len(horiz_pos):
            solver = _PageSolver(
                page_boxes["x_left"].to_numpy(float)[horiz_pos],
                page_boxes["y_top"].to_numpy(float)[horiz_pos],
                page_boxes["x_right"].to_numpy(float)[horiz_pos],
                page_boxes["y_bottom"].to_numpy(float)[horiz_pos],
            )
            for local in solver.solve():
                out.at[idx[horiz_pos[local]], "reading_order"] = rank
                rank += 1

        # Vertical groups are out of scope for the walk: park them at the end of
        # the page in a stable top-down / left-right order.
        if len(vert_pos):
            vp = sorted(
                vert_pos.tolist(),
                key=lambda p: (page_boxes["y_top"].to_numpy(float)[p],
                               page_boxes["x_left"].to_numpy(float)[p]),
            )
            for p in vp:
                out.at[idx[p], "reading_order"] = rank
                rank += 1

    return out


def _sort_into_reading_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort words into reading order: by ``reading_order`` (the stream-group
    rank), then by ``text_object_id`` (content-stream order) within each group.

    A stable sort preserves the original left-to-right word order inside a single
    text object. Null ``text_object_id``s sort to the end of their group.
    """
    sort_cols = ["reading_order"]
    if "text_object_id" in df.columns:
        sort_cols.append("text_object_id")
    return df.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def assign_stream_group_order(df_words: pd.DataFrame) -> pd.DataFrame:
    """Reshuffle words into reading order and assign ``reading_order`` + ``line_id``.

    Requires ``stream_group_id`` (from step_07). Steps:

    1. Collapse words to group boxes and run the per-page reading-order walk,
       broadcasting each group's rank back onto its words as ``reading_order``
       (Int64, globally sequential — all words in a group share it).
    2. Sort words into reading order: ``reading_order`` then ``text_object_id``
       within each group.
    3. Assign ``line_id`` with :func:`assign_line_id`, then guarantee that words
       sharing a ``table_row_id`` land on one line even across visual lines.

    The returned frame is sorted in reading order and carries ``reading_order``,
    ``line_id`` and ``center_bucket``.
    """
    if df_words is None or df_words.empty:
        out = df_words.copy() if df_words is not None else pd.DataFrame()
        out["reading_order"] = pd.Series(dtype="Int64")
        out["line_id"] = pd.Series(dtype="Int64")
        return out

    boxes = _build_group_boxes(df_words)
    boxes = order_group_boxes(boxes)

    ranks = boxes[[*_GROUP_KEY, "reading_order"]]
    # Drop any stale reading_order so re-runs are idempotent (and so the merge
    # cannot suffix-rename an existing one).
    out = df_words.drop(columns=["reading_order"], errors="ignore").merge(
        ranks, on=_GROUP_KEY, how="left"
    )
    out["reading_order"] = out["reading_order"].astype("Int64")

    out = _sort_into_reading_order(out)
    out = assign_line_id(out)

    return out

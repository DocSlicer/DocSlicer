"""
Shared reading-order walk over opaque group bounding boxes.

Generalized from the PDF pipeline's step_08_reading_order (which currently
still carries its own copy — see the note there): given one union bbox per
"group" (a stream group, a PPTX reading group, ...), assign each group a
per-page ``reading_order`` rank in human reading order. The walk never looks
at text, only at the boxes.

The algorithm (per page)
------------------------
Repeatedly pick the current "top-left" group, then emit it once everything
that must be read before it has been emitted:

  pick_top_left(S):
      1. L = the group in S with the smallest x_left (the leftmost).
      2. Among the groups in S whose x-range overlaps L's x-range, return the
         one with the smallest y_top.
     Anchoring on the leftmost group's x-column (step 2) is what stops us from
     grabbing a box that merely has the globally lowest y_top but lives in a
     different column.

  process(G):
      a. LEFT-BLOCKERS. While some still-unresolved group sits inside G's
         y-range AND starts to the left of G, G cannot be read yet. Mask G
         (put it aside) and recursively process the top-left of those blockers
         first. This is what defers a very tall right-hand box until the
         left/middle columns nested inside its y-range are done.
      b. EMIT G.
      c. FOLLOWERS. G has now "reserved" its y-band: resolve every group that
         falls inside it, picking the top-left each time and recursing. A
         follower whose own y-range reaches further down pulls in whatever
         sits inside it, so a tall first column sweeps its sibling columns
         before the page moves on.

Masking (a group on the recursion stack is invisible to the scans below it)
is what keeps a deferred box out of the way while its nested content is
resolved around it.

NOTE: the walk is left-to-right biased — it reserves a y-band and sweeps it
left to right, which reads side-by-side columns row-wise. For column-first
layouts (typical slides) use ``method="xy"`` in :func:`order_group_boxes`:
a recursive XY-cut that splits on clear horizontal gaps first (title band /
body / footer), then on vertical gutters (columns), recursing until no gap
remains; irreducible overlapping clusters fall back to the walk.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# ================================================================================
# TOLERANCES
# ================================================================================

# Two groups "share a y-band" only when their vertical overlap exceeds this, so a
# grazing 1-pt touch does not chain unrelated groups together.
Y_OVERLAP_TOL: float = 2.0

# A group is "to the left of" G when its left edge is at least this far left of
# G's left edge — a small dead-zone so near-aligned columns do not jitter.
X_LEFT_TOL: float = 3.0

# Two groups' x-ranges "overlap" (for the top-left anchor) when they share more
# than this much horizontal extent.
X_OVERLAP_TOL: float = 1.0

# XY-cut: two projected intervals merge into the same slice/column only when
# they overlap by more than this, so a grazing touch across a gutter does not
# fuse two columns.
XY_MERGE_TOL: float = 2.0


# ================================================================================
# PER-PAGE SOLVER
# ================================================================================

class PageSolver:
    """Order the group boxes of a single page into reading order.

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
        return min(self.yb[i], self.yb[g]) - max(self.yt[i], self.yt[g]) > Y_OVERLAP_TOL

    def _x_overlaps(self, i: int, j: int) -> bool:
        return min(self.xr[i], self.xr[j]) - max(self.xl[i], self.xl[j]) > X_OVERLAP_TOL

    def _is_left_of(self, i: int, g: int) -> bool:
        return self.xl[i] < self.xl[g] - X_LEFT_TOL

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
                    and self.xr[m] <= self.xl[h] + X_OVERLAP_TOL):
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
        first.
        """
        if len(cands) == 1:
            return cands[0]

        # 1. leftmost group (x_left, then y_top as a deterministic tie-break)
        anchor = min(cands, key=lambda i: (self.xl[i], self.yt[i]))
        # 2. topmost group whose x-range overlaps the anchor's column
        #    (a group always "overlaps" itself, even if narrower than the
        #    tolerance, so the anchor is never filtered out of its own column)
        column = [i for i in cands if i == anchor or self._x_overlaps(i, anchor)]
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
            and self.yb[i] <= band_top + Y_OVERLAP_TOL
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
# XY-CUT
# ================================================================================

def _interval_segments(
    starts: np.ndarray, ends: np.ndarray, positions: list[int], tol: float
) -> list[list[int]]:
    """Partition ``positions`` into maximal runs of transitively overlapping
    intervals along one axis, ordered by start coordinate. More than one run
    means a clean gap (a cut) exists between them."""
    order = sorted(positions, key=lambda p: starts[p])
    segments: list[list[int]] = [[order[0]]]
    seg_end = ends[order[0]]
    for p in order[1:]:
        if starts[p] < seg_end - tol:
            segments[-1].append(p)
            seg_end = max(seg_end, ends[p])
        else:
            segments.append([p])
            seg_end = ends[p]
    return segments


def xy_cut_order(
    xl: np.ndarray, yt: np.ndarray, xr: np.ndarray, yb: np.ndarray
) -> list[int]:
    """Order group boxes by recursive XY-cut, column-first.

    Split on horizontal gaps into top-to-bottom slices, which peels off
    full-width bands (title, footer). Then, before emitting the slices one by
    one, greedily merge maximal runs of consecutive slices whose *union* still
    admits a vertical gutter: such a run is a multi-column region (e.g.
    subtitle-above-chart next to subtitle-above-panels) and is read
    column-first — each column in full, left to right — instead of slice by
    slice, which would interleave the columns. Everything recurses; a cluster
    that can be cut on neither axis is ordered by the PageSolver walk.
    """

    def x_segments(positions: list[int]) -> list[list[int]]:
        return _interval_segments(xl, xr, positions, XY_MERGE_TOL)

    def recurse(positions: list[int]) -> list[int]:
        if len(positions) <= 1:
            return list(positions)
        y_segments = _interval_segments(yt, yb, positions, XY_MERGE_TOL)
        if len(y_segments) == 1:
            return cut_slice(positions)

        out: list[int] = []
        i = 0
        while i < len(y_segments):
            j = i
            union = list(y_segments[i])
            while j + 1 < len(y_segments) and len(x_segments(union + y_segments[j + 1])) > 1:
                j += 1
                union += y_segments[j]
            if j > i:
                # Multi-column run: read each column in full, left to right.
                for col in x_segments(union):
                    out.extend(recurse(col))
            else:
                out.extend(cut_slice(y_segments[i]))
            i = j + 1
        return out

    def cut_slice(positions: list[int]) -> list[int]:
        """One y-slice: only a vertical gutter (or the walk) can split it."""
        if len(positions) <= 1:
            return list(positions)
        segments = x_segments(positions)
        if len(segments) > 1:
            return [p for seg in segments for p in recurse(seg)]
        solver = PageSolver(xl[positions], yt[positions], xr[positions], yb[positions])
        return [positions[local] for local in solver.solve()]

    return recurse(list(range(len(xl))))


# ================================================================================
# PUBLIC API
# ================================================================================

def order_group_boxes(
    boxes: pd.DataFrame,
    page_col: str = "page_number",
    method: str = "walk",
) -> pd.DataFrame:
    """Assign a per-page ``reading_order`` rank to every group box.

    ``boxes`` is one row per group with ``x_left, y_top, x_right, y_bottom``
    plus ``page_col`` and any identifying columns the caller needs to merge
    the rank back. An optional boolean ``is_vertical`` column excludes
    vertical-text groups from the ordering; they are appended after the
    horizontal groups of their page, sorted top-to-bottom / left-to-right.

    ``method`` selects the per-page algorithm: ``"walk"`` (band-reserving
    top-left walk, left-to-right biased -- suits dense print layouts) or
    ``"xy"`` (recursive XY-cut, column-first -- suits slide layouts).

    Returns ``boxes`` with an added int ``reading_order`` that is globally
    sequential (pages in ascending order, groups within a page in reading
    order).
    """
    if method not in ("walk", "xy"):
        raise ValueError(f"order_group_boxes: unknown method {method!r}")

    out = boxes.copy()
    out["reading_order"] = -1
    rank = 0

    for _page, page_boxes in out.groupby(page_col, sort=True, dropna=False):
        idx = page_boxes.index.to_numpy()
        if "is_vertical" in page_boxes.columns:
            is_vert = page_boxes["is_vertical"].to_numpy(bool)
        else:
            is_vert = np.zeros(len(page_boxes), dtype=bool)

        horiz_pos = np.flatnonzero(~is_vert)
        vert_pos = np.flatnonzero(is_vert)

        if len(horiz_pos):
            coords = (
                page_boxes["x_left"].to_numpy(float)[horiz_pos],
                page_boxes["y_top"].to_numpy(float)[horiz_pos],
                page_boxes["x_right"].to_numpy(float)[horiz_pos],
                page_boxes["y_bottom"].to_numpy(float)[horiz_pos],
            )
            if method == "xy":
                ordered = xy_cut_order(*coords)
            else:
                ordered = PageSolver(*coords).solve()
            for local in ordered:
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


__all__ = [
    "PageSolver",
    "xy_cut_order",
    "order_group_boxes",
    "Y_OVERLAP_TOL",
    "X_LEFT_TOL",
    "X_OVERLAP_TOL",
    "XY_MERGE_TOL",
]

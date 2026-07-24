# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
PPTX visual container grouping.

Runs after the paragraph builder and before the line builder. Assigns each
paragraph a `container_shape_ids` list: the shape_ids of textless shapes
(decorative rectangles, pills, cards, ...) whose bounding box visually
contains that paragraph's shape, ordered largest container first.

This is a heuristic *visual* grouping derived from raw bbox geometry --
distinct from `group_ids`, which reflects actual PowerPoint p:grpSp shape
groups. Slides frequently nest a text box inside a white "pill" shape inside
a grey card, with no p:grpSp relationship between any of them at all.
container_shape_ids recovers that visual nesting so it is available as a
grouping signal ahead of the line builder's spatial band/column heuristic.

A second column, `container_group_ids`, carries the real p:grpSp group_ids
that those containers themselves belong to. Authors sometimes group the
decorative shape rather than the text box it visually encloses (the text box
sits alone, un-grouped, on top of a grouped rectangle) -- in that case the
paragraph's own `group_ids` is empty even though it is visually part of a
PowerPoint group, and container_group_ids recovers it via the container.

build_paragraphs (step_05) drops any paragraph whose shape carries no text,
image, or chart content (see its _SELF_SUFFICIENT_REF_TYPES), so the geometry
of purely decorative/textless shapes never reaches paragraph_df. This module
reads it back from run_df, the only place that geometry survives.

Containment rule: a paragraph's shape counts as "inside" a candidate
container when at least CONTAINMENT_RATIO of the paragraph shape's own bbox
area overlaps the container's bbox. Grazing or slight overlap does not
count -- hence the high default threshold.

Building on those columns, `assign_reading_groups` derives a reading-order
grouping. Every grouping id a paragraph carries (real p:grpSp groups,
container shapes, its own shape) becomes a cluster whose member set is the
paragraphs carrying it; identical member sets are merged. Clusters nest by
strict member-set containment into a forest, and within each tree siblings
are ordered top-to-bottom by their inferred bboxes -- falling back to
left-to-right when two siblings' vertical overlap exceeds Y_OVERLAP_RATIO of
the shorter one. Paragraphs within a shape keep their original DataFrame
order. The DataFrame itself is NOT reshuffled; the result is carried in
columns (`reading_group_key`, `reading_group_order`, `reading_group_path`,
`reading_group_bboxes`) so a later step can reorder; ungrouped shapes simply
become their own top-level group.

Finally, `assign_group_order` orders the top-level groups *between* each
other: each reading_group_key's union bbox is fed to a per-page recursive
XY-cut walk (`_xy_cut_order`, inlined below) yielding `reading_group_rank`,
and `order_index` combines that rank with the within-group order into one
global paragraph sequence -- again as a column only, without reshuffling the
DataFrame.

`assign_reading_order` is the single entry point running all three stages.
"""

from __future__ import annotations

import pandas as pd

# Minimum fraction of a shape's own bbox area that must overlap a candidate
# container's bbox for that container to count as enclosing it.
CONTAINMENT_RATIO = 0.90

# Two sibling clusters count as the same horizontal band (and therefore sort
# left-to-right instead of top-to-bottom) only when their vertical overlap is
# at least this fraction of the shorter cluster's height. A heading that
# merely grazes the top of a taller block below it stays above that block.
Y_OVERLAP_RATIO = 0.5

# Cluster-id namespaces: real p:grpSp groups ("g"), shapes ("s"), and bare
# paragraphs without a shape_id ("p"). When merged clusters have identical
# member sets, the canonical id is picked in this priority order.
_PREFIX_PRIORITY = {"g": 0, "s": 1, "p": 2}

# run_type values that mark a shape as carrying no text/image/chart content of
# its own -- exactly the shapes build_paragraphs drops entirely (see step_05's
# _SELF_SUFFICIENT_REF_TYPES comment). These are the only eligible containers.
_CONTAINER_ONLY_RUN_TYPES = frozenset({"shape_ref", "graphic_ref"})
_CONTENT_RUN_TYPES = frozenset({"text", "math", "image_ref", "chart_ref"})

_GEOM_COLS = ["x_left", "x_right", "y_top", "y_bottom"]


def _shape_boxes(df: pd.DataFrame, group_cols: list[str], shape_ids) -> pd.DataFrame:
    """One row per shape_id (+ group_cols) with its bbox and area."""
    geom = (
        df[df["shape_id"].isin(shape_ids)]
        .dropna(subset=_GEOM_COLS)
        .groupby(group_cols + ["shape_id"], sort=False)[_GEOM_COLS]
        .first()
        .reset_index()
    )
    geom["area"] = (geom["x_right"] - geom["x_left"]) * (geom["y_bottom"] - geom["y_top"])
    return geom[geom["area"] > 0]


def _container_shape_ids_for_slide(
    content: pd.DataFrame,
    containers: pd.DataFrame,
    ratio: float,
) -> dict[int, list[int]]:
    """For each content shape_id, the containing container shape_ids, largest first."""
    result: dict[int, list[int]] = {}
    for s in content.itertuples(index=False):
        matches: list[tuple[float, int]] = []
        for c in containers.itertuples(index=False):
            if c.shape_id == s.shape_id or c.area <= s.area:
                continue
            ix_left = max(s.x_left, c.x_left)
            ix_top = max(s.y_top, c.y_top)
            ix_right = min(s.x_right, c.x_right)
            ix_bottom = min(s.y_bottom, c.y_bottom)
            if ix_right <= ix_left or ix_bottom <= ix_top:
                continue
            overlap = (ix_right - ix_left) * (ix_bottom - ix_top)
            if overlap / s.area >= ratio:
                matches.append((c.area, int(c.shape_id)))
        matches.sort(key=lambda t: t[0], reverse=True)
        result[int(s.shape_id)] = [sid for _, sid in matches]
    return result


def _shape_group_ids_map(run_df: pd.DataFrame) -> dict[int, list]:
    """shape_id -> its own p:grpSp group_ids (identical across all its runs)."""
    if "group_ids" not in run_df.columns:
        return {}
    first_group_ids = run_df.groupby("shape_id")["group_ids"].first()
    return {
        sid: (gids if isinstance(gids, list) else [])
        for sid, gids in first_group_ids.items()
    }


def _container_group_ids(container_shape_ids: list[int], shape_group_ids: dict[int, list]) -> list:
    """Union (order-preserved) of the real group_ids of a paragraph's containers."""
    result: list = []
    for cid in container_shape_ids:
        for gid in shape_group_ids.get(cid, []):
            if gid not in result:
                result.append(gid)
    return result


def assign_container_shape_ids(
    paragraph_df: pd.DataFrame,
    run_df: pd.DataFrame,
    containment_ratio: float = CONTAINMENT_RATIO,
) -> pd.DataFrame:
    """
    Add `container_shape_ids` and `container_group_ids` columns to paragraph_df.

    Args:
        paragraph_df: Output of build_paragraphs.
        run_df: Output of extract_runs for the same document -- the source of
            textless shape geometry that paragraph_df no longer carries.
        containment_ratio: Minimum fraction of a shape's own bbox area that
            must overlap a candidate container's bbox to count as contained.

    Returns:
        paragraph_df with two added columns:
          - container_shape_ids: shape_ids (largest area first) whose bbox
            encloses that paragraph's shape, or an empty list when none match.
          - container_group_ids: the real p:grpSp group_ids those containers
            belong to (order-preserved union, largest container first).
    """
    out = paragraph_df.copy()
    out["container_shape_ids"] = [[] for _ in range(len(out))]
    out["container_group_ids"] = [[] for _ in range(len(out))]

    if out.empty or run_df.empty or "shape_id" not in out.columns:
        return out
    if not set(_GEOM_COLS).issubset(out.columns) or not set(_GEOM_COLS).issubset(run_df.columns):
        return out

    group_cols = [
        c for c in ("page_number", "slide_index")
        if c in out.columns and c in run_df.columns
    ]

    shape_run_types = run_df.groupby("shape_id")["run_type"].agg(set)
    container_shape_ids = shape_run_types[
        shape_run_types.map(
            lambda types: bool(types & _CONTAINER_ONLY_RUN_TYPES) and not (types & _CONTENT_RUN_TYPES)
        )
    ].index

    containers = _shape_boxes(run_df, group_cols, container_shape_ids)
    if containers.empty:
        return out

    content = _shape_boxes(out, group_cols, out["shape_id"].dropna().unique())
    if content.empty:
        return out

    mapping: dict[int, list[int]] = {}
    if group_cols:
        containers_by_slide = dict(list(containers.groupby(group_cols, sort=False)))
        for key, content_slide in content.groupby(group_cols, sort=False):
            containers_slide = containers_by_slide.get(key)
            if containers_slide is None or containers_slide.empty:
                continue
            mapping.update(
                _container_shape_ids_for_slide(content_slide, containers_slide, containment_ratio)
            )
    else:
        mapping.update(_container_shape_ids_for_slide(content, containers, containment_ratio))

    out["container_shape_ids"] = out["shape_id"].map(mapping)
    out["container_shape_ids"] = out["container_shape_ids"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    shape_group_ids = _shape_group_ids_map(run_df)
    out["container_group_ids"] = out["container_shape_ids"].apply(
        lambda ids: _container_group_ids(ids, shape_group_ids)
    )
    return out


def _cluster_ids_for_row(row) -> list[str]:
    """Prefixed cluster ids a paragraph belongs to, own shape (or self) last."""
    ids: list[str] = []
    for col, prefix in (
        ("group_ids", "g"),
        ("container_group_ids", "g"),
        ("container_shape_ids", "s"),
    ):
        val = getattr(row, col, None)
        if isinstance(val, list):
            for raw in val:
                cid = f"{prefix}:{raw}"
                if cid not in ids:
                    ids.append(cid)
    shape_id = getattr(row, "shape_id", None)
    if pd.notna(shape_id):
        leaf = f"s:{int(shape_id)}"
    else:
        leaf = f"p:{getattr(row, 'paragraph_id', row.Index)}"
    if leaf not in ids:
        ids.append(leaf)
    return ids


def _cluster_bbox(df: pd.DataFrame, member_idx) -> tuple[float, float, float, float] | None:
    """Union bbox (x_left, y_top, x_right, y_bottom) of the members' geometry."""
    geom = df.loc[list(member_idx), _GEOM_COLS].dropna()
    if geom.empty:
        return None
    return (
        float(geom["x_left"].min()),
        float(geom["y_top"].min()),
        float(geom["x_right"].max()),
        float(geom["y_bottom"].max()),
    )


def _spatial_order(items: list[tuple[str, tuple | None]]) -> list[str]:
    """
    Order sibling clusters top-to-bottom, breaking into left-to-right only
    within a shared Y band (vertical overlap >= Y_OVERLAP_RATIO of the shorter
    cluster). Clusters without geometry sort last, in given order.
    """
    boxed = [(cid, box) for cid, box in items if box is not None]
    unboxed = [cid for cid, box in items if box is None]
    boxed.sort(key=lambda cb: (cb[1][1], cb[1][0]))

    # bands: [y_top, y_bottom, [(cid, box), ...]] built by sweep over y_top.
    bands: list[list] = []
    for cid, box in boxed:
        height = box[3] - box[1]
        target = None
        best_overlap = 0.0
        for band in bands:
            overlap = min(box[3], band[1]) - max(box[1], band[0])
            min_height = min(height, band[1] - band[0])
            if min_height > 0 and overlap >= Y_OVERLAP_RATIO * min_height and overlap > best_overlap:
                target = band
                best_overlap = overlap
        if target is None:
            bands.append([box[1], box[3], [(cid, box)]])
        else:
            target[0] = min(target[0], box[1])
            target[1] = max(target[1], box[3])
            target[2].append((cid, box))

    ordered: list[str] = []
    for band in sorted(bands, key=lambda b: b[0]):
        ordered.extend(cid for cid, _ in sorted(band[2], key=lambda cb: (cb[1][0], cb[1][1])))
    return ordered + unboxed


def _format_bbox(cid: str, box: tuple | None) -> str:
    if box is None:
        return f"{cid} [no bbox]"
    return f"{cid} [x:{box[0]:.1f}-{box[2]:.1f} y:{box[1]:.1f}-{box[3]:.1f}]"


def _reading_groups_for_slide(
    slide_df: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    """
    Per-paragraph (by df index): root cluster id, order within root, cluster
    path root->leaf, and formatted path bboxes for debugging.
    """
    position = {idx: i for i, idx in enumerate(slide_df.index)}

    members: dict[str, set] = {}
    leaf_of: dict = {}
    for row in slide_df.itertuples():
        ids = _cluster_ids_for_row(row)
        for cid in ids:
            members.setdefault(cid, set()).add(row.Index)
        leaf_of[row.Index] = ids[-1]

    # Merge clusters with identical member sets: they are the same grouping
    # seen through different id namespaces (e.g. a shape and the p:grpSp
    # group that contains exactly that shape).
    by_members: dict[frozenset, list[str]] = {}
    for cid, m in members.items():
        by_members.setdefault(frozenset(m), []).append(cid)
    canon: dict[str, str] = {}
    clusters: dict[str, set] = {}
    for member_set, ids in by_members.items():
        ids.sort(key=lambda c: (_PREFIX_PRIORITY.get(c.split(":", 1)[0], 9), c))
        for cid in ids:
            canon[cid] = ids[0]
        clusters[ids[0]] = set(member_set)

    bboxes = {cid: _cluster_bbox(slide_df, m) for cid, m in clusters.items()}

    # Forest by strict member-set containment: parent is the smallest strict
    # superset. Bbox size never decides nesting -- membership does.
    parent: dict[str, str | None] = {}
    for cid, m in clusters.items():
        supersets = [
            (len(m2), cid2)
            for cid2, m2 in clusters.items()
            if cid2 != cid and m < m2
        ]
        parent[cid] = min(supersets)[1] if supersets else None

    children: dict[str, list[str]] = {cid: [] for cid in clusters}
    roots: list[str] = []
    for cid, par in parent.items():
        if par is None:
            roots.append(cid)
        else:
            children[par].append(cid)

    paras_at: dict[str, list] = {cid: [] for cid in clusters}
    for idx, leaf in leaf_of.items():
        paras_at[canon[leaf]].append(idx)

    def emit(cid: str) -> list:
        out = list(sorted(paras_at[cid], key=position.get))
        for child in _spatial_order([(c, bboxes[c]) for c in children[cid]]):
            out.extend(emit(child))
        return out

    keys: dict = {}
    orders: dict = {}
    paths: dict = {}
    debugs: dict = {}
    for root in roots:
        for i, idx in enumerate(emit(root)):
            keys[idx] = root
            orders[idx] = i
    for idx, leaf in leaf_of.items():
        path = [canon[leaf]]
        while (par := parent[path[-1]]) is not None:
            path.append(par)
        path.reverse()
        paths[idx] = path
        debugs[idx] = [_format_bbox(cid, bboxes[cid]) for cid in path]
    return keys, orders, paths, debugs


def assign_reading_groups(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reading-group columns to paragraph_df without reordering it.

    Runs after assign_container_shape_ids. Added columns:
      - reading_group_key: id of the paragraph's top-level cluster, prefixed
        with its page so keys stay unique across slides (e.g. "3/g:28694").
      - reading_group_order: 0-based position of the paragraph within its
        top-level cluster's depth-first, spatially sorted emission.
      - reading_group_path: cluster ids from top-level cluster down to the
        paragraph's own shape.
      - reading_group_bboxes: same path with each cluster's inferred bbox
        formatted for inspection.

    Ordering between top-level groups is out of scope here (part 2).
    """
    out = paragraph_df.copy()
    out["reading_group_key"] = None
    out["reading_group_order"] = None
    out["reading_group_path"] = [[] for _ in range(len(out))]
    out["reading_group_bboxes"] = [[] for _ in range(len(out))]
    if out.empty:
        return out

    group_cols = [c for c in ("page_number", "slide_index") if c in out.columns]
    slides = out.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, out)]

    for key, slide_df in slides:
        keys, orders, paths, debugs = _reading_groups_for_slide(slide_df)
        page = key[0] if isinstance(key, tuple) else key
        prefix = f"{page}/" if page is not None else ""
        for idx in slide_df.index:
            out.at[idx, "reading_group_key"] = f"{prefix}{keys[idx]}"
            out.at[idx, "reading_group_order"] = orders[idx]
            out.at[idx, "reading_group_path"] = paths[idx]
            out.at[idx, "reading_group_bboxes"] = debugs[idx]
    return out


# XY-cut: two projected intervals merge into the same slice/column only when
# they overlap by more than this, so a grazing touch across a gutter does not
# fuse two columns.
_XY_MERGE_TOL = 2.0


def _interval_segments(starts, ends, positions: list[int], tol: float) -> list[list[int]]:
    """Partition `positions` into maximal runs of transitively overlapping
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


def _xy_cut_order(xl, yt, xr, yb) -> list[int]:
    """Order group boxes by recursive XY-cut, column-first.

    Split on horizontal gaps into top-to-bottom slices, which peels off
    full-width bands (title, footer). Then, before emitting the slices one by
    one, greedily merge maximal runs of consecutive slices whose *union* still
    admits a vertical gutter: such a run is a multi-column region and is read
    column-first -- each column in full, left to right -- instead of slice by
    slice, which would interleave the columns. Everything recurses; a cluster
    that can be cut on neither axis (irreducible overlap) falls back to a plain
    top-to-bottom / left-to-right sort. Returns box positions 0..n-1 in order.
    """

    def x_segments(positions: list[int]) -> list[list[int]]:
        return _interval_segments(xl, xr, positions, _XY_MERGE_TOL)

    def cut_slice(positions: list[int]) -> list[int]:
        """One y-slice: a vertical gutter splits it, else fall back to a sort."""
        if len(positions) <= 1:
            return list(positions)
        segments = x_segments(positions)
        if len(segments) > 1:
            return [p for seg in segments for p in recurse(seg)]
        return sorted(positions, key=lambda p: (yt[p], xl[p]))

    def recurse(positions: list[int]) -> list[int]:
        if len(positions) <= 1:
            return list(positions)
        y_segments = _interval_segments(yt, yb, positions, _XY_MERGE_TOL)
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

    return recurse(list(range(len(xl))))


def _order_group_boxes_xy(boxes: pd.DataFrame, page_col: str) -> pd.DataFrame:
    """Assign a globally sequential `reading_order` rank per group box, running
    the recursive XY-cut per page (pages ascending, groups within a page in
    reading order)."""
    out = boxes.copy()
    out["reading_order"] = -1
    rank = 0
    for _page, page_boxes in out.groupby(page_col, sort=True, dropna=False):
        idx = page_boxes.index.to_numpy()
        ordered = _xy_cut_order(
            page_boxes["x_left"].to_numpy(float),
            page_boxes["y_top"].to_numpy(float),
            page_boxes["x_right"].to_numpy(float),
            page_boxes["y_bottom"].to_numpy(float),
        )
        for local in ordered:
            out.at[idx[local], "reading_order"] = rank
            rank += 1
    return out


def assign_group_order(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order the top-level reading groups between each other.

    Runs after assign_reading_groups. Builds one union bbox per
    reading_group_key and per page, runs the shared reading-order walk on
    those boxes, and adds two columns without reordering paragraph_df:
      - reading_group_rank: the group's rank in the per-page walk (globally
        sequential across pages). Groups without geometry get no rank and
        sort last.
      - order_index: 0-based global paragraph sequence combining
        reading_group_rank with reading_group_order (original row order as
        final tiebreaker) -- sort by this to get final reading order.
    """
    out = paragraph_df.copy()
    out["reading_group_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["order_index"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if out.empty or "reading_group_key" not in out.columns:
        return out

    page_col = next((c for c in ("page_number", "slide_index") if c in out.columns), None)

    valid = out["reading_group_key"].notna()
    geom = out[valid].dropna(subset=[c for c in _GEOM_COLS if c in out.columns])
    if not set(_GEOM_COLS).issubset(out.columns) or geom.empty:
        return out

    boxes = (
        geom.groupby("reading_group_key", sort=False)
        .agg(
            x_left=("x_left", "min"),
            y_top=("y_top", "min"),
            x_right=("x_right", "max"),
            y_bottom=("y_bottom", "max"),
            **({"page": (page_col, "first")} if page_col else {}),
        )
        .reset_index()
    )
    if not page_col:
        boxes["page"] = 0

    # Recursive XY-cut reads side-by-side panels column-first, which is how
    # slides are meant to be read; a left-to-right band walk would sweep shared
    # y-bands and interleave the columns.
    boxes = _order_group_boxes_xy(boxes, page_col="page")
    rank_map = dict(zip(boxes["reading_group_key"], boxes["reading_order"]))
    out["reading_group_rank"] = out["reading_group_key"].map(rank_map).astype("Int64")

    # Global paragraph sequence: group rank (already page-sequential), order
    # within the group, original row position as tiebreaker. Unranked rows
    # (no geometry) go last.
    sort_df = pd.DataFrame(
        {
            "rank": out["reading_group_rank"],
            "within": out["reading_group_order"],
            "pos": range(len(out)),
        },
        index=out.index,
    )
    ordered_idx = sort_df.sort_values(["rank", "within", "pos"], na_position="last").index
    out.loc[ordered_idx, "order_index"] = range(len(out))
    return out


def assign_reading_order(
    paragraph_df: pd.DataFrame,
    run_df: pd.DataFrame,
    containment_ratio: float = CONTAINMENT_RATIO,
) -> pd.DataFrame:
    """
    Single entry point for this step.

    Runs assign_container_shape_ids (visual bbox-containment grouping),
    assign_reading_groups (containment forest + within-group ordering), and
    assign_group_order (between-group ordering + final order_index). Adds
    columns only; never reorders paragraph_df.

    Args:
        paragraph_df: Output of build_paragraphs.
        run_df: Output of extract_runs for the same document.
        containment_ratio: See assign_container_shape_ids.

    Returns:
        paragraph_df with container_shape_ids, container_group_ids,
        reading_group_key, reading_group_order, reading_group_path,
        reading_group_bboxes, reading_group_rank, and order_index columns
        added.
    """
    out = assign_container_shape_ids(paragraph_df, run_df, containment_ratio)
    out = assign_reading_groups(out)
    return assign_group_order(out)


__all__ = [
    "assign_reading_order",
    "assign_container_shape_ids",
    "assign_reading_groups",
    "assign_group_order",
    "CONTAINMENT_RATIO",
    "Y_OVERLAP_RATIO",
]

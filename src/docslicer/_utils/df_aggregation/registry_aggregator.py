"""
registry_aggregator.py

Registry-driven hierarchical aggregation (words → cells → lines → blocks → chunks).

Instead of every builder enumerating which columns survive aggregation (pull),
this module declares — once, centrally — how each known column aggregates
(push). The spec for any given aggregation is derived from the columns that are
actually present in the input DataFrame, so pipelines never have to opt in.

    from docslicer._utils.df_aggregation.registry_aggregator import Agg, aggregate_to

    df_lines = aggregate_to(
        df_cells,
        by="line_id",
        overrides={"text": lambda s: " ".join(s.astype(str))},
        size_as="cell_count",
    )

Resolution order for each column (first match wins):

    1. ``overrides``        — call-site exceptions
    2. ``COLUMN_REGISTRY``  — the central registry below
    3. leading underscore   — internal helper columns are dropped silently
    4. ``PREFIX_RULES`` / ``SUFFIX_RULES`` — naming conventions
    5. ``DEFAULT_POLICY``   — ``Agg.FIRST``, with a logged warning

Adding a new column to the pipeline therefore takes one line in
``COLUMN_REGISTRY`` (or zero, if its name matches a convention). A column that
reaches step 5 is never silently lost — it is carried through with "first" and
flagged in the logs so it can be promoted to the registry.

Use :func:`explain` to see exactly how a DataFrame would be aggregated.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "Agg",
    "COLUMN_REGISTRY",
    "PREFIX_RULES",
    "SUFFIX_RULES",
    "DEFAULT_POLICY",
    "aggregate_to",
    "explain",
    "group_join",
]


# =============================================================================
# POLICY VOCABULARY
# =============================================================================

class Agg(str, Enum):
    """How a column collapses when child rows are grouped into a parent row."""

    FIRST = "first"                    # first non-null value in the group
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    ANY = "any"                        # boolean OR across the group (via max)
    DOMINANT = "dominant"              # value from the row with the most alphabetic
                                       # text (alpha_count, else char_count, else first)
    LIST = "list"                      # all values in row order (vectorised; much
                                       # faster than a per-group `list` override)
    SORTED_UNIQUE_LIST = "sorted_unique_list"  # unique non-null values, ascending
    UNIQUE_LIST = "unique_list"        # ordered unique non-null values, flattened
    MERGE_DICTS = "merge_dicts"        # dict union, later rows win on duplicate keys
    WEIGHTED_RATIO = "weighted_ratio"  # char_count-weighted mean (exact ratio recompute)
    DROP = "drop"                      # intentionally not propagated


# A policy is an Agg member, or anything pandas ``groupby().agg`` accepts
# (string, callable, list) for one-off call-site overrides.
Policy = Union[Agg, str, Callable[..., Any], list]

# Weight columns for DOMINANT, in priority order.
_DOMINANT_WEIGHT_COLS = ("alpha_count", "char_count")

# Weight column for WEIGHTED_RATIO (falls back to plain mean when absent).
_RATIO_WEIGHT_COL = "char_count"

# Thresholds for the derived flag columns recomputed after aggregation.
BOLD_THRESHOLD = 0.75
ITALIC_THRESHOLD = 0.75
UNDERLINED_THRESHOLD = 0.75
UPPERCASE_THRESHOLD = 0.90


# =============================================================================
# CENTRAL COLUMN REGISTRY
# =============================================================================
# One line per known column. This is the single place to tweak aggregation
# behaviour; call sites should only override for genuinely local exceptions
# (e.g. a custom text joiner, or collecting child IDs into a list).

COLUMN_REGISTRY: Dict[str, Agg] = {
    # --- document / page identity -------------------------------------------
    "doc_name": Agg.FIRST,
    "page_number": Agg.FIRST,
    "page_width": Agg.FIRST,
    "page_height": Agg.FIRST,
    "page_format": Agg.FIRST,
    "page_label": Agg.FIRST,
    "page_label_type": Agg.FIRST,
    "page_label_value": Agg.FIRST,
    "slide_index": Agg.FIRST,
    "section": Agg.FIRST,
    "section_id": Agg.FIRST,

    # --- layout / reading order ----------------------------------------------
    "layout_id": Agg.FIRST,
    "layout_type": Agg.FIRST,
    "block_type": Agg.FIRST,
    "chart_id": Agg.FIRST,
    "reading_column": Agg.FIRST,
    "gutter_id_left": Agg.FIRST,
    "gutter_id_right": Agg.FIRST,
    "stream_group_id": Agg.FIRST,
    "sentence_score": Agg.FIRST,
    "line_class": Agg.FIRST,
    "line_score": Agg.FIRST,
    "line_em_threshold": Agg.FIRST,
    "line_is_bimodal": Agg.FIRST,

    # --- heading hierarchy -----------------------------------------------------
    "heading_id": Agg.FIRST,
    "parent_heading_id": Agg.FIRST,
    "heading_level": Agg.FIRST,
    "heading_type": Agg.FIRST,
    "heading_fp_id": Agg.FIRST,
    "heading_fingerprint": Agg.FIRST,
    "heading_hash": Agg.FIRST,
    "hybrid_heading_text": Agg.FIRST,
    "active_heading_id": Agg.FIRST,
    "chunk_index": Agg.FIRST,

    # --- geometry ---------------------------------------------------------------
    "x_left": Agg.MIN,
    "x_right": Agg.MAX,
    "y_top": Agg.MIN,
    "y_bottom": Agg.MAX,
    "width": Agg.DROP,             # recomputed from the aggregated bbox
    "height": Agg.DROP,            # recomputed from the aggregated bbox
    "layout_align": Agg.DOMINANT,
    "text_align": Agg.DOMINANT,

    # --- style -------------------------------------------------------------------
    "font_size": Agg.DOMINANT,
    "font_weight": Agg.DOMINANT,
    "font_name": Agg.DOMINANT,
    "font_family": Agg.DOMINANT,
    "text_orientation": Agg.DOMINANT,
    "non_stroking_color": Agg.DOMINANT,
    "stroking_color": Agg.DOMINANT,
    "background_non_stroking_color": Agg.DOMINANT,
    "background_stroking_color": Agg.DOMINANT,
    "has_vertical_line": Agg.ANY,
    "inside_rect_shape": Agg.ANY,

    # --- counts (sums) --------------------------------------------------------------
    "char_count": Agg.SUM,
    "alpha_count": Agg.SUM,
    "digit_count": Agg.SUM,
    "uppercase_count": Agg.SUM,
    "word_count": Agg.SUM,
    "alpha_word_count": Agg.SUM,
    "capitalized_word_count": Agg.SUM,

    # --- ratios & flags ---------------------------------------------------------------
    # Ratios are recomputed exactly: sum(ratio * char_count) / sum(char_count).
    "bold_ratio": Agg.WEIGHTED_RATIO,
    "italic_ratio": Agg.WEIGHTED_RATIO,
    "underlined_ratio": Agg.WEIGHTED_RATIO,
    "font_size_ratio": Agg.DROP,   # recomputed vs. the aggregated level's median
    "is_bold": Agg.DROP,           # recomputed from bold_ratio
    "is_italic": Agg.DROP,         # recomputed from italic_ratio
    "is_uppercase": Agg.DROP,      # recomputed from uppercase_count / alpha_count
    "is_underlined": Agg.ANY,      # OR'd; overwritten by underlined_ratio when present

    # --- links / metadata ----------------------------------------------------------------
    "has_link": Agg.ANY,
    "link_url": Agg.UNIQUE_LIST,
    "link_dest": Agg.DOMINANT,
    "link_type": Agg.DOMINANT,
    "hyperlink_url": Agg.UNIQUE_LIST,
    "ixbrl_id": Agg.UNIQUE_LIST,
    "html_data_attrs": Agg.MERGE_DICTS,

    # --- tables ------------------------------------------------------------------------
    "table_id": Agg.FIRST,
    "table_row_id": Agg.FIRST,
    "table_cell_id": Agg.FIRST,    # line builders override to UNIQUE_LIST (lines span cells)
    "table_header_flag": Agg.FIRST,
    "table_cell_index": Agg.FIRST,
    "table_row_cell_count": Agg.FIRST,
    "table_row_count": Agg.FIRST,
    "row_start": Agg.FIRST,
    "col_start": Agg.FIRST,
    "nested_table_depth": Agg.FIRST,

    # --- HTML provenance -----------------------------------------------------------------
    "structure_tag_id": Agg.FIRST,
    "structure_tag": Agg.FIRST,
    "wrapping_tag": Agg.FIRST,
    "split_reason": Agg.FIRST,
    "dom_id": Agg.FIRST,
    "dom_class": Agg.FIRST,
    "ancestor_ids": Agg.FIRST,
    "ancestor_classes": Agg.FIRST,
    "ancestor_tags": Agg.FIRST,
    "ancestor_aria_roles": Agg.FIRST,
    "struct_ancestors": Agg.FIRST,
    "img_alt": Agg.FIRST,
    "img_src": Agg.FIRST,
    "shape_id_vertical_grid_line": Agg.UNIQUE_LIST,

    # --- PDF structure tree / content-stream provenance ----------------------------------------
    "struct_group_id": Agg.FIRST,
    "struct_tag": Agg.FIRST,
    "struct_tag_id": Agg.FIRST,
    "struct_raw_tag": Agg.FIRST,
    "struct_ancestor_ids": Agg.FIRST,
    "struct_raw_ancestors": Agg.FIRST,
    "struct_scope": Agg.FIRST,
    "struct_headers": Agg.FIRST,
    "struct_col_span": Agg.FIRST,
    "struct_row_span": Agg.FIRST,
    "bdc_tag": Agg.FIRST,
    "dfs_position": Agg.MIN,       # struct-tree traversal order key
    "reading_order": Agg.MIN,      # reading-order key
    "textbox_id": Agg.FIRST,
    "raw_shape_id": Agg.FIRST,
    "word_source": Agg.FIRST,
    "mcid": Agg.DROP,              # marked-content id; word-level detail
    "text_object_id": Agg.DROP,    # content-stream object id; word-level detail
    "gap_em_right": Agg.DROP,      # inter-word gap, consumed by cell splitting

    # --- PDF form fields ------------------------------------------------------------------------
    "form_widget": Agg.FIRST,
    "form_field_name": Agg.FIRST,
    "form_value": Agg.FIRST,
    "form_tooltip": Agg.FIRST,
    "form_is_empty": Agg.FIRST,

    # --- DOCX / PPTX structure ----------------------------------------------------------------
    "header_footer_type": Agg.FIRST,
    "source_part": Agg.FIRST,
    "source_part_id": Agg.FIRST,
    "list_num_id": Agg.FIRST,
    "list_level": Agg.FIRST,
    "list_label": Agg.FIRST,
    "outline_level": Agg.FIRST,
    "page_break_before": Agg.FIRST,
    "section_break_type": Agg.FIRST,
    "section_break_after": Agg.ANY,
    "bookmark_ids": Agg.FIRST,
    "bookmark_names": Agg.FIRST,
    "comment_id": Agg.FIRST,
    "footnote_id": Agg.FIRST,
    "endnote_id": Agg.FIRST,
    "style_id": Agg.FIRST,
    "style_name": Agg.FIRST,
    "paragraph_style_id": Agg.FIRST,
    "paragraph_style_name": Agg.FIRST,
    "effective_paragraph_style_id": Agg.FIRST,
    "effective_paragraph_style_name": Agg.FIRST,
    "character_style_id": Agg.DOMINANT,
    "character_style_name": Agg.DOMINANT,
    "effective_character_style_id": Agg.DOMINANT,
    "effective_character_style_name": Agg.DOMINANT,
    "shape_id": Agg.FIRST,
    "shape_name": Agg.FIRST,
    "shape_type": Agg.FIRST,
    "placeholder_type": Agg.FIRST,

    # --- child-level detail: consumed before aggregation, not propagated -----------------------
    # Override at the call site to collect them (e.g. {"word_id": list}).
    "text": Agg.DROP,              # rebuilt per level with a level-specific joiner
    "word_id": Agg.DROP,
    "run_id": Agg.DROP,
    "run_type": Agg.DROP,
    "box_id": Agg.DROP,
    "cell_id": Agg.DROP,
    "line_id": Agg.DROP,
    "block_id": Agg.DROP,
    "paragraph_id": Agg.UNIQUE_LIST,
    "order_index": Agg.DROP,
    "script_type": Agg.DROP,
}

# Naming conventions for columns not in the registry. Checked in order.
PREFIX_RULES: List[Tuple[str, Agg]] = [
    ("is_", Agg.ANY),
    ("has_", Agg.ANY),
    ("inside_", Agg.ANY),
    ("page_", Agg.FIRST),
]

SUFFIX_RULES: List[Tuple[str, Agg]] = [
    ("_count", Agg.SUM),
    ("_ratio", Agg.WEIGHTED_RATIO),
]

DEFAULT_POLICY: Agg = Agg.FIRST

# Columns already reported as missing from the registry — each unknown column
# is warned about once per process, not once per aggregate_to() call.
_WARNED_DEFAULT_COLS: set = set()


# =============================================================================
# POLICY RESOLUTION
# =============================================================================

def _resolve_policy(
    col: str,
    overrides: Mapping[str, Policy],
) -> Tuple[Policy, str]:
    """Resolve a column's aggregation policy. Returns (policy, source)."""
    if col in overrides:
        return overrides[col], "override"
    if col in COLUMN_REGISTRY:
        return COLUMN_REGISTRY[col], "registry"
    if col.startswith("_"):
        return Agg.DROP, "internal"
    for prefix, policy in PREFIX_RULES:
        if col.startswith(prefix):
            return policy, f"prefix:{prefix}"
    for suffix, policy in SUFFIX_RULES:
        if col.endswith(suffix):
            return policy, f"suffix:{suffix}"
    return DEFAULT_POLICY, "default"


def explain(
    df: pd.DataFrame,
    by: str,
    overrides: Optional[Mapping[str, Policy]] = None,
    drop: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Show how :func:`aggregate_to` would treat every column of ``df``.

    Returns a DataFrame with columns ``[column, policy, source]`` — useful for
    debugging and for spotting columns that fall through to the default.
    """
    overrides = dict(overrides or {})
    drop_set = set(drop)
    rows = []
    for col in df.columns:
        if col == by:
            rows.append((col, "group_key", "group_key"))
            continue
        if col in drop_set:
            rows.append((col, str(Agg.DROP.value), "drop-param"))
            continue
        policy, source = _resolve_policy(col, overrides)
        name = policy.value if isinstance(policy, Agg) else repr(policy)
        rows.append((col, name, source))
    return pd.DataFrame(rows, columns=["column", "policy", "source"])


# =============================================================================
# VECTORISED AGGREGATORS
# =============================================================================

def _agg_list(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    """
    Per-group list of ``col`` values in row order, without per-group Python calls:
    factorise the keys, stable-argsort the codes, then ``np.split`` the value
    array at the group boundaries.
    """
    codes, uniques = pd.factorize(df[by])
    mask = codes >= 0                       # drop NaN group keys (groupby default)
    codes = codes[mask]
    if codes.size == 0:
        return pd.Series(dtype=object, name=col)
    values = df[col].to_numpy()[mask]

    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniques))
    chunks = np.split(values[order], np.cumsum(counts)[:-1])
    return pd.Series(
        [c.tolist() for c in chunks],
        index=pd.Index(uniques, name=by),
        name=col,
    )


def _agg_sorted_unique_list(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    """Per-group sorted unique non-null values of ``col`` (one global dedupe+sort)."""
    sub = df[[by, col]].dropna(subset=[col]).drop_duplicates()
    if sub.empty:
        return pd.Series(dtype=object, name=col)
    return _agg_list(sub.sort_values(col, kind="stable"), by, col)


def group_join(
    tokens: pd.Series,
    keys: pd.Series,
    sep: Union[str, "np.ndarray", pd.Series] = " ",
    attach_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Join per-group strings in row order — a fast ``groupby().agg(sep.join)``.

    Bypasses pandas' pure-Python per-group path (one frame slice per group, very
    slow on Arrow-backed strings) with factorize + stable argsort + np.split.

    Args:
        tokens: String values to join (must be aligned with ``keys``).
        keys: Group keys; rows with a null key are dropped (groupby's default).
        sep: Separator placed before each token (never before the first token of
            a group). Either a single string (``"\\n"`` for line-per-row joins,
            ``""`` to concatenate) or an array-like aligned with ``tokens``
            giving a per-token separator (e.g. newline before bullet lines).
        attach_mask: Optional boolean mask (aligned with ``tokens``); True marks
            tokens that attach directly to the previous token with no separator
            (e.g. superscript/subscript markers).

    Returns:
        Series of joined strings indexed by group key.
    """
    codes, uniques = pd.factorize(keys)
    mask = codes >= 0
    codes = codes[mask]
    if codes.size == 0:
        return pd.Series(dtype=object)
    values = tokens.to_numpy()[mask]

    order = np.argsort(codes, kind="stable")
    values = values[order]
    counts = np.bincount(codes, minlength=len(uniques))
    bounds = np.cumsum(counts)

    if isinstance(sep, str):
        prefixes = np.full(len(values), sep, dtype=object) if sep else None
    else:
        prefixes = np.asarray(sep, dtype=object)[mask][order]

    if prefixes is not None:
        if attach_mask is not None:
            prefixes[np.asarray(attach_mask, dtype=bool)[mask][order]] = ""
        prefixes[np.r_[0, bounds[:-1]]] = ""      # no separator at group starts
        values = prefixes + values

    chunks = np.split(values, bounds[:-1])
    return pd.Series(
        ["".join(c) for c in chunks],
        index=pd.Index(uniques, name=keys.name),
    )


def _agg_unique_list(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    """
    Per-group ordered unique non-null values of ``col``, flattening values that
    are already lists (from a prior aggregation pass). NA filtering happens
    once at the column level; grouping uses pandas' C-level ``agg(list)``.
    """
    sub = df[[by, col]].dropna(subset=[col])
    if sub.empty:
        return pd.Series(dtype=object, name=col)

    if sub[col].map(lambda v: isinstance(v, list)).any():
        sub = sub.explode(col).dropna(subset=[col])

    sub = sub[sub[col] != ""]
    if sub.empty:
        return pd.Series(dtype=object, name=col)

    raw = sub.groupby(by, sort=False, observed=True)[col].agg(list)
    return raw.map(lambda lst: list(dict.fromkeys(lst)) or None)


def _agg_merge_dicts(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    """Per-group dict union of ``col`` (later rows win on duplicate keys)."""
    sub = df[[by, col]].dropna(subset=[col])
    if sub.empty:
        return pd.Series(dtype=object, name=col)

    sub = sub[sub[col].map(lambda v: isinstance(v, dict))]
    if sub.empty:
        return pd.Series(dtype=object, name=col)

    return sub.groupby(by, sort=False, observed=True)[col].agg(
        lambda dicts: {k: v for d in dicts for k, v in d.items()} or None
    )


def _merge_on_key(grouped: pd.DataFrame, by: str, series: pd.Series, col: str) -> pd.DataFrame:
    """Left-join a per-group Series onto the aggregated frame (None for missing groups)."""
    if series.empty:
        grouped[col] = None
        return grouped
    grouped = grouped.merge(series.rename(col).reset_index(), on=by, how="left")
    grouped[col] = grouped[col].where(grouped[col].notna(), None)
    return grouped


# =============================================================================
# PUBLIC API
# =============================================================================

def aggregate_to(
    df: pd.DataFrame,
    by: str,
    *,
    overrides: Optional[Mapping[str, Policy]] = None,
    drop: Iterable[str] = (),
    size_as: Optional[str] = None,
    rename_by: Optional[str] = None,
    derived: bool = True,
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """
    Aggregate ``df`` one level up the hierarchy, grouped by ``by``.

    Every column's aggregation rule comes from :data:`COLUMN_REGISTRY` (see the
    module docstring for the full resolution order); the call site only supplies
    what is genuinely local to that step.

    Args:
        df: Input frame with the grouping column already assigned.
        by: Column to group by (e.g. ``"cell_id"``, ``"line_id"``).
        overrides: Per-column policy exceptions for this call, e.g.
            ``{"text": my_joiner, "word_id": list}``. Values may be Agg members
            or anything pandas ``agg`` accepts.
        drop: Columns to discard for this call (clearer than a DROP override).
        size_as: If set, adds a column with the group size under this name
            (e.g. ``size_as="cell_count"``).
        rename_by: Rename the group column in the output
            (e.g. ``by="_block_id", rename_by="block_id"``).
        derived: Recompute derived columns after aggregation: width/height,
            is_bold/is_italic/is_underlined/is_uppercase, font_size_ratio.
        on_unknown: What to do with columns that fall through to the default
            policy: ``"warn"`` (default), ``"raise"``, or ``"silent"``.

    Returns:
        One row per group, columns renamed per ``rename_by``.
    """
    if on_unknown not in ("warn", "raise", "silent"):
        raise ValueError(f"on_unknown must be 'warn', 'raise' or 'silent', got {on_unknown!r}")
    if by not in df.columns:
        raise KeyError(f"Group column {by!r} not in DataFrame columns")
    if df.empty:
        return pd.DataFrame()

    overrides = dict(overrides or {})
    drop_set = set(drop)

    # -------------------------------------------------------------------------
    # 1. Resolve a policy for every column
    # -------------------------------------------------------------------------
    spec: Dict[str, Any] = {}          # plain pandas aggregations
    dominant_cols: List[str] = []      # Agg.DOMINANT
    list_cols: List[str] = []          # Agg.LIST
    sorted_unique_cols: List[str] = [] # Agg.SORTED_UNIQUE_LIST
    unique_list_cols: List[str] = []   # Agg.UNIQUE_LIST
    merge_dict_cols: List[str] = []    # Agg.MERGE_DICTS
    ratio_cols: List[str] = []         # Agg.WEIGHTED_RATIO
    defaulted_cols: List[str] = []

    dominant_weight = next((c for c in _DOMINANT_WEIGHT_COLS if c in df.columns), None)
    has_ratio_weight = _RATIO_WEIGHT_COL in df.columns

    for col in df.columns:
        if col == by or col in drop_set:
            continue
        policy, source = _resolve_policy(col, overrides)
        if source == "default":
            defaulted_cols.append(col)

        if not isinstance(policy, Agg):
            spec[col] = policy                       # raw pandas agg from an override
        elif policy is Agg.DROP:
            continue
        elif policy is Agg.DOMINANT:
            if dominant_weight is not None:
                dominant_cols.append(col)
            else:
                spec[col] = "first"
        elif policy is Agg.LIST:
            list_cols.append(col)
        elif policy is Agg.SORTED_UNIQUE_LIST:
            sorted_unique_cols.append(col)
        elif policy is Agg.UNIQUE_LIST:
            unique_list_cols.append(col)
        elif policy is Agg.MERGE_DICTS:
            merge_dict_cols.append(col)
        elif policy is Agg.WEIGHTED_RATIO:
            if has_ratio_weight:
                ratio_cols.append(col)
            else:
                spec[col] = "mean"
        elif policy is Agg.ANY:
            spec[col] = "max"
        else:
            spec[col] = policy.value                 # FIRST / MIN / MAX / SUM

    if defaulted_cols and on_unknown == "raise":
        raise ValueError(
            f"No aggregation policy for columns {sorted(defaulted_cols)}. "
            f"Add them to COLUMN_REGISTRY in {__name__}."
        )
    if defaulted_cols and on_unknown == "warn":
        unseen = sorted(set(defaulted_cols) - _WARNED_DEFAULT_COLS)
        if unseen:
            _WARNED_DEFAULT_COLS.update(unseen)
            logger.warning(
                "aggregate_to(by=%r): no policy for columns %s — defaulting to 'first'. "
                "Add them to COLUMN_REGISTRY to silence this warning. "
                "(Each column is only reported once per process.)",
                by, unseen,
            )

    # -------------------------------------------------------------------------
    # 2. Prepare weighted-ratio numerators (exact recompute after grouping)
    # -------------------------------------------------------------------------
    work = df
    if ratio_cols:
        weight = pd.to_numeric(df[_RATIO_WEIGHT_COL], errors="coerce").fillna(0.0)
        numerators = {
            f"_wnum_{col}": pd.to_numeric(df[col], errors="coerce").fillna(0.0) * weight
            for col in ratio_cols
        }
        work = df.assign(**numerators)
        for helper in numerators:
            spec[helper] = "sum"

    # -------------------------------------------------------------------------
    # 3. Aggregate
    # -------------------------------------------------------------------------
    gb = work.groupby(by, sort=False, observed=True)
    if spec:
        grouped = gb.agg(spec).reset_index()
    else:
        grouped = work[[by]].drop_duplicates().reset_index(drop=True)

    # DOMINANT: one vectorised idxmax on the weight column picks, per group, the
    # row with the most alphabetic text; its values are taken for all dominant
    # columns at once. No per-group Python calls.
    if dominant_cols:
        weights = pd.to_numeric(work[dominant_weight], errors="coerce").fillna(0.0)
        pick = weights.groupby(work[by], sort=False, observed=True).idxmax()
        dominant_rows = (
            work.loc[pick.values, [by] + dominant_cols].reset_index(drop=True)
        )
        grouped = grouped.merge(dominant_rows, on=by, how="left")

    for col in list_cols:
        grouped = _merge_on_key(grouped, by, _agg_list(work, by, col), col)

    for col in sorted_unique_cols:
        grouped = _merge_on_key(grouped, by, _agg_sorted_unique_list(work, by, col), col)

    for col in unique_list_cols:
        grouped = _merge_on_key(grouped, by, _agg_unique_list(work, by, col), col)

    for col in merge_dict_cols:
        grouped = _merge_on_key(grouped, by, _agg_merge_dicts(work, by, col), col)

    if size_as:
        grouped[size_as] = grouped[by].map(gb.size())

    # -------------------------------------------------------------------------
    # 4. Finalise weighted ratios: sum(ratio * weight) / sum(weight)
    # -------------------------------------------------------------------------
    if ratio_cols:
        # The weight column is usually already summed in the main agg — reuse it
        # instead of running a second groupby over the input.
        if spec.get(_RATIO_WEIGHT_COL) == "sum":
            totals = pd.to_numeric(grouped[_RATIO_WEIGHT_COL], errors="coerce")
        else:
            totals = grouped[by].map(
                pd.to_numeric(work[_RATIO_WEIGHT_COL], errors="coerce").fillna(0.0)
                .groupby(work[by], sort=False, observed=True).sum()
            )
        totals = totals.replace(0, np.nan)
        helpers = [f"_wnum_{col}" for col in ratio_cols]
        for col, helper in zip(ratio_cols, helpers):
            grouped[col] = (grouped[helper] / totals).fillna(0.0)
        grouped = grouped.drop(columns=helpers)

    # -------------------------------------------------------------------------
    # 5. Derived columns
    # -------------------------------------------------------------------------
    if derived:
        grouped = _compute_derived(grouped)

    if rename_by:
        grouped = grouped.rename(columns={by: rename_by})

    return grouped


def _compute_derived(grouped: pd.DataFrame) -> pd.DataFrame:
    """Recompute geometry, style flags and font_size_ratio at the new level."""
    cols = grouped.columns

    if "x_left" in cols and "x_right" in cols:
        grouped["width"] = grouped["x_right"] - grouped["x_left"]
    if "y_top" in cols and "y_bottom" in cols:
        grouped["height"] = grouped["y_bottom"] - grouped["y_top"]

    if "bold_ratio" in cols:
        grouped["is_bold"] = grouped["bold_ratio"] > BOLD_THRESHOLD
    if "italic_ratio" in cols:
        grouped["is_italic"] = grouped["italic_ratio"] > ITALIC_THRESHOLD
    if "underlined_ratio" in cols:
        grouped["is_underlined"] = grouped["underlined_ratio"] > UNDERLINED_THRESHOLD

    if "uppercase_count" in cols and "alpha_count" in cols:
        alpha_safe = grouped["alpha_count"].replace(0, np.nan)
        grouped["is_uppercase"] = (
            (grouped["uppercase_count"] / alpha_safe).fillna(0.0) > UPPERCASE_THRESHOLD
        )

    if "font_size" in cols:
        fs = pd.to_numeric(grouped["font_size"], errors="coerce")
        median_fs = float(fs.dropna().median()) if fs.notna().any() else np.nan
        if np.isfinite(median_fs) and median_fs > 0:
            grouped["font_size_ratio"] = fs / median_fs
        else:
            grouped["font_size_ratio"] = 1.0

    return grouped

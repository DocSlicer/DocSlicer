# step_02_box_cleaner.py
# Cleans raw extracted boxes by (1) dropping boilerplate elements based on HTML ancestor
# context (nav/header/footer, cookie banners, social widgets, ads, etc.), (2) normalising
# font sizes, (3) merging fragments that share a struct_tag_id, and (4) re-ordering
# boxes by DOM position so that downstream steps receive a clean, sorted box list.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from docslicer._utils.text_utils import add_calculated_text_features
from docslicer._utils.df_aggregation.registry_aggregator import Agg, aggregate_to, group_join
from docslicer._utils.df_aggregation.text_merge import (
    apply_inline_markup,
    merge_text_within_line,
)


# =======================================================================================================================
# STEP 1: Drop Boilerplate by Ancestors
# =======================================================================================================================

# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class AncestorDropConfig:
    col_ids: str = "ancestor_ids"
    col_classes: str = "ancestor_classes"
    col_tags: str = "struct_ancestors"
    col_roles: str = "ancestor_aria_roles"
    col_dom_id: str = "dom_id"
    col_dom_class: str = "dom_class"

    # High-precision tag / role exact drops
    # HTML tags
    drop_tags_exact: tuple[str, ...] = ("header", "footer", "nav", "aside", "menu")
    # ARIA roles
    drop_aria_roles_exact: tuple[str, ...] = (
        "navigation",
        "banner",
        "contentinfo",
        "search",
        "dialog",
        "alertdialog",
        "menu",
        "menubar",
    )

    # High-precision keyword tokens matched as substrings against ancestor ids/classes/tags/roles
    # and the element's own dom_id/dom_class.  Avoid overly generic tokens (e.g. "header").
    drop_keywords: tuple[str, ...] = (
        # cookie / consent / CMP
        "cookie", "cookies", "consent", "gdpr", "ccpa", "cmp", "onetrust", "trustarc", "quantcast",
        # auth / account / subscription
        "login", "log-in", "signin", "sign-in", "signup", "sign-up", "register", "createaccount",
        "subscribe", "subscription", "newsletter",
        # ads / sponsored
        "adslot", "advert", "advertise", "advertisement", "sponsor", "sponsored", "promoted",
        "outbrain", "taboola", "doubleclick", "adsense", "amazon-ads", "pubmatic", "criteo", 
        "tracking", "analytics", "gtm", "tag-manager", "affiliate", "referral",
        # social / share
        "share", "social", "facebook", "linkedin", "twitter", "instagram", "youtube", "pinterest",
        # navigation chrome
        "navbar", "topbar", "breadcrumb", "pagination", "sidebar", "drawer", "hamburger", "toolbar", 
        "subnav", "utility-nav", "mega-menu", "site-map", "footer", "banner", "sitenotice",
        "back-to-top", "scroll-to-top",
        # hidden / accessibility-only elements
        "sr-only", "visually-hidden", "hidden-xs", "hidden-sm", "skipnav", "skip-to-content", 
        "skip-link", "accessibility",
        # modal / overlay chrome
        "modal", "overlay", "lightbox", "popup", "tooltip", "popover",
        # misc UI chrome
        "button", "createfreeaccount", "backlink", "loggedout", "noprint", "no-print",
        # legal boilerplate
        "terms-of-service", "disclaimer", "copyright", "credits",
    )

    # If a keyword appears in >= this fraction of rows it is suppressed to avoid
    # false-positive drops on sites that use these tokens in every wrapper div.
    common_token_max_frac: float = 0.80


CFG = AncestorDropConfig()


# -------------------------
# Helper Functions
# -------------------------

def _to_list(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None and not (isinstance(x, float) and pd.isna(x))]
    s = str(v).strip()
    return [s] if s else []


def _build_row_strings(df: pd.DataFrame, cfg: AncestorDropConfig) -> pd.Series:
    """
    Build a per-row search string by concatenating all ancestor and own-element fields.

    Operates column-by-column instead of row-by-row to avoid the overhead of creating
    a ``pd.Series`` object for every row (as ``DataFrame.apply(axis=1)`` would do).
    """
    cols = [cfg.col_ids, cfg.col_classes, cfg.col_tags, cfg.col_roles, cfg.col_dom_id, cfg.col_dom_class]
    parts = [df[col].apply(lambda v: " ".join(_to_list(v))) for col in cols if col in df.columns]
    if not parts:
        return pd.Series("", index=df.index)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p
    return combined.str.lower()


def _any_exact_match(tokens: Iterable[str], exact_set: set[str]) -> bool:
    for t in tokens:
        if str(t).strip().lower() in exact_set:
            return True
    return False


def _compute_common_keywords(row_strings: pd.Series, keywords: set[str], max_frac: float) -> set[str]:
    """
    Return the subset of ``keywords`` that appear as a substring in >= ``max_frac`` of rows.
    These keywords will be suppressed so that site-wide wrapper patterns do not cause
    false-positive drops.
    """
    n = len(row_strings)
    if n == 0:
        return set()

    counts: dict[str, int] = {}
    for row_str in row_strings:
        if not row_str:
            continue
        for kw in keywords:
            if kw in row_str:
                counts[kw] = counts.get(kw, 0) + 1

    threshold = int(max_frac * n)
    return {kw for kw, c in counts.items() if c >= threshold}


def _normalize_font_size(val) -> float | None:
    """Convert a font-size value to a plain float, stripping CSS units (e.g. ``"18.6667px"`` → ``18.6667``)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    match = re.match(r'^([-+]?\d+\.?\d*)', str(val).strip())
    return float(match.group(1)) if match else None


# -------------------------
# Core Boilerplate Dropping Function
# -------------------------

def _drop_boilerplate_by_ancestors(
    df: pd.DataFrame,
    cfg: AncestorDropConfig = CFG,
    dry_run: bool = False,
    keep_debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Drop rows that look like boilerplate based on their HTML ancestor context.

    Three rules are applied in order:
    - **Tag exact match** – ancestor tags contain a structural chrome tag (``<nav>``, ``<footer>``, …).
    - **ARIA role exact match** – ancestor ARIA roles contain a landmark role (``navigation``, ``banner``, …).
    - **Keyword substring match** – ancestor ids/classes or the element's own ``dom_id``/``dom_class``
      contain a known boilerplate keyword.  Keywords that appear on >= ``cfg.common_token_max_frac``
      of all rows are suppressed to avoid site-specific false positives.

    ``<h1>`` elements are never dropped regardless of their ancestor context.

    Args:
        df: DataFrame of extracted boxes.
        cfg: Drop configuration (tags, roles, keywords, thresholds).
        dry_run: If ``True``, annotate rows with debug columns instead of removing them.
        keep_debug_cols: Attach ``_drop_*`` diagnostic columns to the returned DataFrame.

    Returns:
        Cleaned DataFrame (or annotated copy when ``dry_run=True``).
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    tags_exact = {x.lower() for x in cfg.drop_tags_exact}
    aria_roles_exact = {x.lower() for x in cfg.drop_aria_roles_exact}
    drop_kw = {x.lower() for x in cfg.drop_keywords}

    # Rule 1 / 2: exact tag and ARIA role drops
    tags_list = out.get(cfg.col_tags, pd.Series([[]] * len(out))).apply(_to_list).apply(
        lambda lst: [x.lower() for x in lst]
    )
    roles_list = out.get(cfg.col_roles, pd.Series([[]] * len(out))).apply(_to_list).apply(
        lambda lst: [x.lower() for x in lst]
    )

    hit_tag_exact = tags_list.apply(lambda toks: _any_exact_match(toks, tags_exact))
    hit_role_exact = roles_list.apply(lambda toks: _any_exact_match(toks, aria_roles_exact))

    # Rule 3: keyword drops via substring matching (ancestors + own dom_id/dom_class)
    row_strings = _build_row_strings(out, cfg)

    common_keywords = _compute_common_keywords(row_strings, drop_kw, cfg.common_token_max_frac)
    active_keywords = drop_kw - common_keywords

    if active_keywords:
        # str.contains with a compiled regex runs at C level — much faster than Series.apply
        pattern = "|".join(re.escape(kw) for kw in sorted(active_keywords))
        hit_kw = row_strings.str.contains(pattern, regex=True, na=False)
    else:
        hit_kw = pd.Series(False, index=out.index)

    # Exception: never drop <h1> elements, even when nested inside chrome
    is_h1 = pd.Series([False] * len(out), index=out.index)
    if "struct_tag" in out.columns:
        is_h1 = out["struct_tag"].fillna("").str.lower() == "h1"
    elif "wrapping_tag" in out.columns:
        is_h1 = out["wrapping_tag"].fillna("").str.lower() == "h1"

    drop_mask = (hit_tag_exact | hit_role_exact | hit_kw) & ~is_h1

    if keep_debug_cols:
        out["_drop_hit_tag_exact"] = hit_tag_exact
        out["_drop_hit_role_exact"] = hit_role_exact
        out["_drop_hit_keyword"] = hit_kw
        out["_drop_reason"] = ""
        out.loc[hit_tag_exact, "_drop_reason"] = "tag_exact"
        out.loc[hit_role_exact & out["_drop_reason"].eq(""), "_drop_reason"] = "role_exact"
        out.loc[hit_role_exact & out["_drop_reason"].ne(""), "_drop_reason"] += "|role_exact"
        out.loc[hit_kw & out["_drop_reason"].eq(""), "_drop_reason"] = "keyword"
        out.loc[hit_kw & out["_drop_reason"].ne(""), "_drop_reason"] += "|keyword"
        out["_debug_common_keywords_suppressed"] = " ".join(sorted(common_keywords)[:50])
        out["_will_be_dropped"] = drop_mask

    if dry_run:
        return out

    return out.loc[~drop_mask].copy()


# =======================================================================================================================
# STEP 2: Merge Boxes by Structure Tag ID
# =======================================================================================================================

def _collapse_consecutive_script_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive fragments that share a ``struct_tag_id`` and
    ``script_type`` into a single row (text concatenated, no separator), so
    :func:`apply_inline_markup` wraps each run once instead of once per
    fragment — three single-char superscript boxes "[", "15", "]" become one
    "[15]" row -> "[^[15]]", instead of "[^[][^15][^]]".

    ``df`` must already be sorted in join (box_id) order. Uses pandas'
    vectorized ``groupby().first()`` plus :func:`group_join` for the text
    concatenation, so no per-row Python loop.
    """
    if df.empty:
        return df

    script = df["script_type"].astype("string").fillna("").to_numpy()
    key = df["struct_tag_id"].to_numpy()
    is_script = script != ""
    # Plain-text rows never join a run (they keep the normal word-spaced join
    # downstream); only consecutive fragments of the *same* non-empty script
    # collapse together.
    boundary = np.ones(len(df), dtype=bool)
    boundary[1:] = ~(is_script[1:] & (script[1:] == script[:-1]) & (key[1:] == key[:-1]))
    run_id = pd.Series(np.cumsum(boundary), index=df.index)

    out = df.groupby(run_id, sort=False).first()
    out["text"] = group_join(df["text"], run_id, sep="").to_numpy()
    return out.reset_index(drop=True)


def _merge_boxes_by_structure_tag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge boxes that share the same ``struct_tag_id``, EXCEPT when ``split_reason`` is
    ``'br_tag'`` or ``'code_line'``.

    Boxes split by a ``<br>`` tag remain separate because they represent intentional line
    breaks; ``code_line`` boxes (one per line of a ``<pre>`` block) remain separate for the
    same reason.
    The merged box inherits the minimum ``box_id`` of its group so that downstream DOM ordering
    is preserved.

    Args:
        df: DataFrame with box data including ``struct_tag_id`` and ``split_reason``.

    Returns:
        DataFrame with merged boxes.
    """
    if df.empty or "struct_tag_id" not in df.columns:
        return df

    # Boxes that must NOT be merged: intentional line splits (<br>, <pre> code
    # lines) or no valid struct_tag_id
    br_mask = df.get("split_reason", pd.Series([None] * len(df), index=df.index)).isin(["br_tag", "code_line"])
    no_struct_id_mask = df["struct_tag_id"].isna() | (df["struct_tag_id"] < 0)
    no_merge_mask = br_mask | no_struct_id_mask

    # Passed-through boxes (<br> splits / orphans): bake inline markup straight into
    # their text so they match the merged boxes and script_type can be dropped.
    df_no_merge = df[no_merge_mask].copy()
    df_no_merge["text"] = apply_inline_markup(df_no_merge)
    df_no_merge = df_no_merge.drop(columns=["script_type"], errors="ignore")

    df_to_merge = df[~no_merge_mask].copy()
    if df_to_merge.empty:
        return df_no_merge.sort_values("box_id").reset_index(drop=True)

    # Fragments of one element must join in DOM (reading) order; box_id is that order.
    df_to_merge = df_to_merge.sort_values("box_id", kind="mergesort")

    # Consecutive same-script fragments (e.g. several adjacent <sup> reference marks,
    # often single-char boxes) collapse into one run so apply_inline_markup wraps them
    # once instead of once per fragment ("[^[15][16]]" rather than "[^[15]][^[16]]").
    # Ordinal superscripts ("15th", "2nd") are handled by apply_inline_markup itself,
    # which leaves them unwrapped as plain typography.
    df_to_merge = _collapse_consecutive_script_runs(df_to_merge)

    fmt_text = apply_inline_markup(df_to_merge)
    merged_text = merge_text_within_line(fmt_text, df_to_merge["struct_tag_id"])

    df_merged = aggregate_to(
        df_to_merge,
        by="struct_tag_id",
        overrides={
            "box_id": Agg.MIN,        # inherit the min box_id → preserves DOM ordering
            "struct_tag": Agg.FIRST,  # element tag is shared by all fragments
        },
    )
    df_merged["text"] = df_merged["struct_tag_id"].map(merged_text)

    result = pd.concat([df_merged, df_no_merge], ignore_index=True)
    result = result.sort_values("box_id").reset_index(drop=True)

    return result


# =======================================================================================================================
# Table Header Parsing
# =======================================================================================================================


_TABLE_CELL_TAGS = frozenset({"td", "th"})


def _deepest_cell_is_th(ancestors) -> bool:
    """True when the nearest (deepest) <td>/<th> ancestor is a <th>.

    struct_ancestors runs root -> leaf, so the last cell tag in the chain is the
    box's own cell. Non-table boxes (no td/th) return False.
    """
    if not isinstance(ancestors, (list, tuple, np.ndarray)):
        return False
    deepest = None
    for tag in ancestors:
        if tag in _TABLE_CELL_TAGS:
            deepest = tag
    return deepest == "th"


def _assign_table_header_flag(df: pd.DataFrame) -> None:
    """Add ``table_header_flag`` (bool) to *df* in-place, one value per box."""
    n = len(df)
    if not {"table_id", "table_row_id", "struct_ancestors"} <= set(df.columns):
        df["table_header_flag"] = np.zeros(n, dtype=bool)
        return

    is_table = df["table_id"].notna()
    cell_is_th = df["struct_ancestors"].map(_deepest_cell_is_th)
    has_thead = df["struct_ancestors"].map(lambda a: "thead" in a if isinstance(a, (list, tuple, np.ndarray)) else False)

    # Row is entirely <th>: all cells sharing (table_id, table_row_id) are <th>.
    row_all_th = cell_is_th.groupby([df["table_id"], df["table_row_id"]]).transform("min").astype(bool)
    # First row of each table (smallest table_row_id).
    first_row = df["table_row_id"].groupby(df["table_id"]).transform("min")
    is_first_row = df["table_row_id"].eq(first_row)

    header = is_table & (has_thead | (row_all_th & is_first_row))
    df["table_header_flag"] = header.fillna(False).astype(bool)


# =======================================================================================================================
# Public API
# =======================================================================================================================

def clean_boxes(
    df_boxes: pd.DataFrame,
    keep_debug_cols: bool = False,
    dry_run: bool = False,
    reorder_by_coordinates: bool = True,
) -> pd.DataFrame:
    """
    Full cleaning pipeline for a raw box DataFrame extracted from an HTML document.

    Steps applied in order:

    1. **Drop boilerplate** – remove navigation, cookie banners, ads, social widgets, etc.
       based on HTML ancestor context (see :func:`drop_boilerplate_by_ancestors`).
    2. **Text features** – compute derived text metrics (character counts, whitespace ratios, …).
    3. **Font size normalisation** – strip CSS units so ``"18.6667px"`` becomes ``18.6667``.
    4. **Merge fragments** – collapse boxes that share a ``struct_tag_id`` into a single row,
       preserving ``<br>``-split boxes as separate entries.
    5. **DOM ordering** – reassign ``box_id`` in reading order: regular content sorted by
       ``struct_tag_id``; ``<hr>`` and ``<img>`` elements inserted by ``y_top`` position
       (skipped when ``reorder_by_coordinates=False`` — use for static extraction where
       y_top is always 0 and DOM order is already correct).
    6. **Block role** – populate ``block_type`` for structural elements (``"hr"``, ``"image"``).

    Args:
        df_boxes: Raw box DataFrame as produced by the box extractor step.
        keep_debug_cols: Attach ``_drop_*`` diagnostic columns from the boilerplate filter.
        dry_run: Skip actual row removal in the boilerplate step (useful for inspection).
        reorder_by_coordinates: When True (default), hr/img elements are slotted into the
            regular content stream by y_top. Set to False for statically extracted boxes
            where y_top is always 0 — boxes are already in DOM order and slotting by y_top
            would push all hr/img elements to the top.

    Returns:
        Cleaned, ordered box DataFrame ready for downstream processing.
    """
    # 1) Drop boilerplate by ancestors
    df_clean = _drop_boilerplate_by_ancestors(df_boxes, keep_debug_cols=keep_debug_cols, dry_run=dry_run)

    # 2) Add calculated text features
    df_clean = add_calculated_text_features(df_clean)

    # 3) Normalise font_size to a plain float
    if "font_size" in df_clean.columns:
        df_clean["font_size"] = df_clean["font_size"].apply(_normalize_font_size)

    # 4) Merge boxes that share a struct_tag_id (br_tag splits are kept separate)
    df_clean = _merge_boxes_by_structure_tag(df_clean)

    # 5) Reassign box_id in DOM order: regular content by struct_tag_id,
    #    hr/img elements inserted by y_top (skipped for static extraction)
    if "box_id" in df_clean.columns and "struct_tag" in df_clean.columns:
        is_special = df_clean["struct_tag"].isin(["hr", "img"])
        special_rows = df_clean[is_special].copy()
        regular_rows = df_clean[~is_special].copy()

        if reorder_by_coordinates and not special_rows.empty and not regular_rows.empty and "y_top" in df_clean.columns:
            if "struct_tag_id" in regular_rows.columns:
                # box_id tiebreaker: boxes sharing a struct_tag_id (br_tag / code_line
                # splits) must keep extraction order — a bare single-key sort_values
                # uses an unstable quicksort and scrambles them.
                regular_rows = regular_rows.sort_values(["struct_tag_id", "box_id"]).reset_index(drop=True)
            special_rows = special_rows.sort_values("y_top").reset_index(drop=True)

            # Assign float sort keys so special rows slot in before the first regular row
            # whose y_top exceeds theirs — no Python loop needed.
            reg_y = regular_rows["y_top"].to_numpy(dtype=float, na_value=0.0)
            spec_y = special_rows["y_top"].to_numpy(dtype=float, na_value=0.0)
            ins = np.searchsorted(reg_y, spec_y, side="left")

            regular_rows["_sort_key"] = np.arange(len(regular_rows)) * 2.0
            special_rows["_sort_key"] = ins * 2.0 - 1.0  # sits just before its insertion point

            df_clean = (
                pd.concat([regular_rows, special_rows], ignore_index=True)
                .sort_values(["_sort_key", "y_top"])
                .drop(columns=["_sort_key"])
                .reset_index(drop=True)
            )
        else:
            # Static extraction: DOM order is already correct — sort everything by struct_tag_id
            # (box_id tiebreaker keeps br_tag / code_line splits in extraction order)
            if "struct_tag_id" in df_clean.columns:
                df_clean = df_clean.sort_values(["struct_tag_id", "box_id"]).reset_index(drop=True)
            else:
                df_clean = df_clean.reset_index(drop=True)

        df_clean["box_id"] = range(1, len(df_clean) + 1)  # 1-based

    # 6) Assign block_type for structural element types
    if "struct_tag" in df_clean.columns:
        df_clean["block_type"] = None
        df_clean.loc[df_clean["struct_tag"] == "hr", "block_type"] = "hr"
        df_clean.loc[df_clean["struct_tag"] == "img", "block_type"] = "image"

    # 7) Table header rows (mirrors the PDF rule in pdf/step_06_style_prefiller):
    #    a row is a header when a <thead> sits in its ancestor chain, OR the row
    #    is entirely <th> AND it is the table's first row. Computed here, per box,
    #    while the <th>/<td> distinction is still available — row cells later merge
    #    into one line, so the flag is stamped on every cell of the row and carried
    #    through the merge by the aggregator's "first" policy. A lone <th> row-label
    #    therefore never flips a whole body row to a header.
    _assign_table_header_flag(df_clean)

    return df_clean




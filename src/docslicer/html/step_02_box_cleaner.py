# step_02_box_cleaner.py
# Cleans raw extracted boxes by (1) dropping boilerplate elements based on HTML ancestor
# context (nav/header/footer, cookie banners, social widgets, ads, etc.), (2) normalising
# font sizes, (3) merging fragments that share a structure_tag_id, and (4) re-ordering
# boxes by DOM position so that downstream steps receive a clean, sorted box list.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from docslicer._utils.text_utils import add_calculated_text_features
from docslicer._utils.hierarchical_aggregator import (
    build_standard_agg_spec,
    aggregate_hierarchical,
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
    col_tags: str = "ancestor_tags"
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
    if "structure_tag" in out.columns:
        is_h1 = out["structure_tag"].fillna("").str.lower() == "h1"
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

def _merge_boxes_by_structure_tag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge boxes that share the same ``structure_tag_id``, EXCEPT when ``split_reason = 'br_tag'``.

    Boxes split by a ``<br>`` tag remain separate because they represent intentional line breaks.
    The merged box inherits the minimum ``box_id`` of its group so that downstream DOM ordering
    is preserved.

    Args:
        df: DataFrame with box data including ``structure_tag_id`` and ``split_reason``.

    Returns:
        DataFrame with merged boxes.
    """
    if df.empty or "structure_tag_id" not in df.columns:
        return df

    # Boxes that must NOT be merged: intentional <br> splits or no valid structure_tag_id
    br_mask = df.get("split_reason", pd.Series([None] * len(df))) == "br_tag"
    no_struct_id_mask = df["structure_tag_id"].isna() | (df["structure_tag_id"] < 0)
    no_merge_mask = br_mask | no_struct_id_mask

    df_no_merge = df[no_merge_mask].copy()
    df_to_merge = df[~no_merge_mask].copy()

    if df_to_merge.empty:
        return df

    df_to_merge["_min_box_id"] = df_to_merge.groupby("structure_tag_id")["box_id"].transform("min")

    agg_spec = build_standard_agg_spec(
        identity_cols=["page_number", "page_width", "page_height", "page_format"],
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,
        include_html_provenance=True,
        extra_first=[
            "_min_box_id",
            "img_alt",
            "img_src",
            "underlined_ratio",
        ],
        extra_agg={
            "text": lambda s: " ".join(str(t) for t in s if t and str(t).strip()),
        },
    )

    df_merged = aggregate_hierarchical(
        df_to_merge,
        group_col="structure_tag_id",
        agg_spec=agg_spec,
        compute_derived=True,
    )

    df_merged["box_id"] = df_merged["_min_box_id"]
    df_merged = df_merged.drop(columns=["_min_box_id"])

    result = pd.concat([df_merged, df_no_merge], ignore_index=True)
    result = result.sort_values("box_id").reset_index(drop=True)

    return result


# =======================================================================================================================
# Public API
# =======================================================================================================================

def clean_boxes(
    df_boxes: pd.DataFrame,
    keep_debug_cols: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Full cleaning pipeline for a raw box DataFrame extracted from an HTML document.

    Steps applied in order:

    1. **Drop boilerplate** – remove navigation, cookie banners, ads, social widgets, etc.
       based on HTML ancestor context (see :func:`drop_boilerplate_by_ancestors`).
    2. **Text features** – compute derived text metrics (character counts, whitespace ratios, …).
    3. **Font size normalisation** – strip CSS units so ``"18.6667px"`` becomes ``18.6667``.
    4. **Merge fragments** – collapse boxes that share a ``structure_tag_id`` into a single row,
       preserving ``<br>``-split boxes as separate entries.
    5. **DOM ordering** – reassign ``box_id`` in reading order: regular content sorted by
       ``structure_tag_id``; ``<hr>`` and ``<img>`` elements inserted by ``y_top`` position.
    6. **Block role** – populate ``block_type`` for structural elements (``"hr"``, ``"image"``).

    Args:
        df_boxes: Raw box DataFrame as produced by the box extractor step.
        keep_debug_cols: Attach ``_drop_*`` diagnostic columns from the boilerplate filter.
        dry_run: Skip actual row removal in the boilerplate step (useful for inspection).

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

    # 4) Merge boxes that share a structure_tag_id (br_tag splits are kept separate)
    df_clean = _merge_boxes_by_structure_tag(df_clean)

    # 5) Reassign box_id in DOM order: regular content by structure_tag_id,
    #    hr/img elements inserted by y_top
    if "box_id" in df_clean.columns and "structure_tag" in df_clean.columns:
        is_special = df_clean["structure_tag"].isin(["hr", "img"])
        special_rows = df_clean[is_special].copy()
        regular_rows = df_clean[~is_special].copy()

        if not special_rows.empty and not regular_rows.empty and "y_top" in df_clean.columns:
            if "structure_tag_id" in regular_rows.columns:
                regular_rows = regular_rows.sort_values("structure_tag_id").reset_index(drop=True)
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
            if "structure_tag_id" in df_clean.columns:
                df_clean = df_clean.sort_values("structure_tag_id").reset_index(drop=True)
            else:
                df_clean = df_clean.reset_index(drop=True)

        df_clean["box_id"] = range(1, len(df_clean) + 1)  # 1-based

    # 6) Assign block_type for structural element types
    if "structure_tag" in df_clean.columns:
        df_clean["block_type"] = None
        df_clean.loc[df_clean["structure_tag"] == "hr", "block_type"] = "hr"
        df_clean.loc[df_clean["structure_tag"] == "img", "block_type"] = "image"

    return df_clean

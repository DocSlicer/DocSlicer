# step_0X_drop_boilerplate_by_ancestors.py
# Drops rows that look like boilerplate (cookie banners, nav/header/footer, social/share, etc.)
# based on keywords found in ancestor_ids / ancestor_classes / ancestor_tags / ancestor_aria_roles.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from docslicer._utils.text_feature_enhancer import add_calculated_text_features
from docslicer._utils.hierarchical_aggregator import (
    build_standard_agg_spec,
    aggregate_hierarchical,
)

# =======================================================================================================================
# STEP 1: Drop Boilerplate by Ancestors
# =======================================================================================================================

# TODO: Potentially add a text filter. If its < 50 chars stuff like: Subscribe via RSS, Download PDF, Javascript, ...
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

    # High-precision keyword tokens (avoid ambiguous ones like "dialog")
    drop_keywords: tuple[str, ...] = (
        # cookie / consent / CMP
        "cookie", "cookies", "consent", "gdpr", "ccpa", "cmp", "onetrust", "trustarc", "quantcast",
        # auth / account / subscription
        "login", "log-in", "signin", "sign-in", "signup", "sign-up", "register", "createaccount",
        "subscribe", "subscription", "newsletter",
        # ads / sponsored
        "adslot", "advert", "advertise", "advertisement", "sponsor", "sponsored", "promoted",
        "outbrain", "taboola", "doubleclick",
        # social / share
        "share", "social", "facebook", "linkedin", "twitter", "instagram", "youtube", "pinterest",
        # chrome-ish
        "navbar", "topbar", "breadcrumb", "pagination", "sidebar",
        "drawer", "hamburger", "toolbar", "subnav", "utility-nav", "mega-menu", "site-map",
        # Hidden/Accessibility
        "sr-only", "visually-hidden", "hidden-xs", "hidden-sm",
        # modal-ish (keep these, but not plain "dialog")
        "modal", "overlay", "lightbox", "popup", "tooltip", "popover",
        # New
         "footer", "banner", "sitenotice", "loggedout", "noprint","no-print",
        "button", "createaccount", "createfreeaccount", "backlink", "accessibility",
        # Legal/Technical
        "terms-of-service", "disclaimer", "copyright", "credits", "skipnav", "skip-to-content", "skip-link",
        #Ads
        "adsense", "amazon-ads", "pubmatic", "criteo", "tracking", "analytics", "gtm", "tag-manager","affiliate", "referral",
        "back-to-top", "scroll-to-top",

        #"header", --> problematic

    )

    # If a token appears in >= this fraction of rows, ignore it for keyword dropping
    common_token_max_frac: float = 0.80

    keep_debug_cols: bool = True


CFG = AncestorDropConfig()


# -------------------------
# Helper Functions
# -------------------------

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def _to_list(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None and not (isinstance(x, float) and pd.isna(x))]
    s = str(v).strip()
    return [s] if s else []


def _tokenize(s: str) -> list[str]:
    if not s:
        return []
    return [t for t in _SPLIT_RE.split(s.lower()) if t]


def _row_tokens(r: pd.Series, cfg: AncestorDropConfig) -> list[str]:
    parts: list[str] = []
    parts.extend(_to_list(r.get(cfg.col_ids)))
    parts.extend(_to_list(r.get(cfg.col_classes)))
    parts.extend(_to_list(r.get(cfg.col_tags)))
    parts.extend(_to_list(r.get(cfg.col_roles)))
    tokens: list[str] = []
    for p in parts:
        tokens.extend(_tokenize(p))
    return tokens


def _row_strings_for_keyword_matching(r: pd.Series, cfg: AncestorDropConfig) -> str:
    """
    Concatenate all relevant fields (ancestors + dom_id + dom_class) into a single
    lowercase string for substring matching.
    """
    parts: list[str] = []
    parts.extend(_to_list(r.get(cfg.col_ids)))
    parts.extend(_to_list(r.get(cfg.col_classes)))
    parts.extend(_to_list(r.get(cfg.col_tags)))
    parts.extend(_to_list(r.get(cfg.col_roles)))
    parts.extend(_to_list(r.get(cfg.col_dom_id)))
    parts.extend(_to_list(r.get(cfg.col_dom_class)))
    return " ".join(parts).lower()


def _any_exact_match(tokens: Iterable[str], exact_set: set[str]) -> bool:
    for t in tokens:
        if str(t).strip().lower() in exact_set:
            return True
    return False


def _compute_common_tokens(token_lists: pd.Series, max_frac: float) -> set[str]:
    """
    token_lists: Series[list[str]]
    Return tokens that appear in >= max_frac of rows.
    """
    n = len(token_lists)
    if n == 0:
        return set()

    # count presence per row (set) to avoid long ancestor chains bias
    counts: dict[str, int] = {}
    for toks in token_lists:
        seen = set(toks or [])
        for t in seen:
            counts[t] = counts.get(t, 0) + 1

    threshold = int(max_frac * n)
    return {t for t, c in counts.items() if c >= threshold}


def _compute_common_keywords(row_strings: pd.Series, keywords: set[str], max_frac: float) -> set[str]:
    """
    For substring matching: return keywords that appear in >= max_frac of rows.
    These keywords will be suppressed to avoid false positives.
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


# -------------------------
# Core Boilerplate Dropping Function
# -------------------------

def drop_boilerplate_by_ancestors(df: pd.DataFrame, 
                                   cfg: AncestorDropConfig = CFG,
                                   dry_run: bool = False,
                                   keep_debug_cols: bool = False
) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    tags_exact = set(x.lower() for x in cfg.drop_tags_exact)
    aria_roles_exact = set(x.lower() for x in cfg.drop_aria_roles_exact)
    drop_kw = set(x.lower() for x in cfg.drop_keywords)

    # Build token lists per row from ancestor context (for tag/role matching)
    tok_lists = out.apply(lambda r: _row_tokens(r, cfg), axis=1)

    # Rule 1/2: exact tag/role drops
    tags_list = out.get(cfg.col_tags, pd.Series([[]] * len(out))).apply(_to_list).apply(
        lambda lst: [x.lower() for x in lst]
    )
    roles_list = out.get(cfg.col_roles, pd.Series([[]] * len(out))).apply(_to_list).apply(
        lambda lst: [x.lower() for x in lst]
    )

    hit_tag_exact = tags_list.apply(lambda toks: _any_exact_match(toks, tags_exact))
    hit_role_exact = roles_list.apply(lambda toks: _any_exact_match(toks, aria_roles_exact))

    # Rule 3: keyword drops using substring matching (no tokenization)
    # Build concatenated strings per row including dom_id and dom_class
    row_strings = out.apply(lambda r: _row_strings_for_keyword_matching(r, cfg), axis=1)

    # Suppress keywords that appear in too many rows (e.g., common wrapper patterns)
    common_keywords = _compute_common_keywords(row_strings, drop_kw, cfg.common_token_max_frac)
    active_keywords = drop_kw - common_keywords

    def hit_keyword(row_str: str) -> bool:
        if not row_str:
            return False
        # Check if any non-common keyword appears as substring
        return any(kw in row_str for kw in active_keywords)

    hit_kw = row_strings.apply(hit_keyword)

    # Exception: Never drop <h1> tags, even if they're in a header/nav/etc.
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

        # Show what keywords got suppressed for being too common
        out["_debug_common_keywords_suppressed"] = " ".join(sorted(list(common_keywords))[:50])
        out["_will_be_dropped"] = drop_mask

    if dry_run:
        # Do not remove anything, just annotate
        return out

    # Normal mode: actually drop
    return out.loc[~drop_mask].copy()


# =======================================================================================================================
# STEP 2: Merge Boxes by Structure Tag ID
# =======================================================================================================================

def _merge_boxes_by_structure_tag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge boxes that share the same structure_tag_id, EXCEPT when split_reason = 'br_tag'.
    
    Boxes split by br_tag should remain separate as they represent intentional line breaks.
    
    Args:
        df: DataFrame with box data including structure_tag_id and split_reason
        
    Returns:
        DataFrame with merged boxes (final DOM order determined by y_top/x_left in step 5)
    """
    if df.empty or "structure_tag_id" not in df.columns:
        return df
    
    # Separate boxes that should NOT be merged (br_tag splits or missing structure_tag_id)
    br_mask = df.get("split_reason", pd.Series([None] * len(df))) == "br_tag"
    no_struct_id_mask = df["structure_tag_id"].isna() | (df["structure_tag_id"] < 0)
    no_merge_mask = br_mask | no_struct_id_mask
    
    df_no_merge = df[no_merge_mask].copy()
    df_to_merge = df[~no_merge_mask].copy()
    
    # If nothing to merge, return original
    if df_to_merge.empty:
        return df
    
    # Track the minimum box_id for each structure_tag_id to preserve relative order
    # (Final DOM order will be determined by y_top/x_left sorting in step 5)
    df_to_merge["_min_box_id"] = df_to_merge.groupby("structure_tag_id")["box_id"].transform("min")
    
    # Build aggregation spec
    agg_spec = build_standard_agg_spec(
        identity_cols=["page_number", "page_width", "page_height", "page_format"],
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,  # Include table_id, table_row_id, table_header_flag, etc.
        include_html_provenance=True,
        extra_first=[
            "_min_box_id",      # Track minimum box_id for DOM ordering
            "img_alt",          # Image alt text
            "img_src",          # Image source URL
            "underlined_ratio", # Underline styling ratio
        ],
        extra_agg={
            "text": lambda s: " ".join(str(t) for t in s if t and str(t).strip()),
        },
    )
    
    # Aggregate by structure_tag_id
    df_merged = aggregate_hierarchical(
        df_to_merge,
        group_col="structure_tag_id",
        agg_spec=agg_spec,
        compute_derived=True,
    )
    
    # Use the minimum box_id from the group to maintain DOM order
    df_merged["box_id"] = df_merged["_min_box_id"]
    df_merged = df_merged.drop(columns=["_min_box_id"])
    
    # Combine merged and non-merged boxes, sort by box_id for now
    # (Final DOM order will be determined by y_top/x_left sorting in step 5)
    result = pd.concat([df_merged, df_no_merge], ignore_index=True)
    result = result.sort_values("box_id").reset_index(drop=True)
    
    return result


# =======================================================================================================================
# Public API
# =======================================================================================================================

def clean_boxes(df_boxes: pd.DataFrame, keep_debug_cols: bool = False, dry_run: bool = False) -> pd.DataFrame:

    # 1) Drop Boilerplate by Ancestors
    df_clean = drop_boilerplate_by_ancestors(df_boxes, keep_debug_cols=keep_debug_cols, dry_run=dry_run)

    # 2) Add text enhancements
    df_clean = add_calculated_text_features(df_clean)

    # 3) Font size normalization (convert "18.6667px" -> 18.6667)
    if "font_size" in df_clean.columns:
        def normalize_font_size(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            if isinstance(val, (int, float)):
                return float(val)
            # Extract numeric part from string like "18.6667px"
            s = str(val).strip()
            match = re.match(r'^([-+]?\d+\.?\d*)', s)
            if match:
                return float(match.group(1))
            return None
        
        df_clean["font_size"] = df_clean["font_size"].apply(normalize_font_size)

    # 4) Merge boxes by structure_tag_id (except br_tag splits)
    df_clean = _merge_boxes_by_structure_tag(df_clean)
    
    # 5) Reindex box_id by DOM order (y_top first, then x_left)
    #if "box_id" in df_clean.columns:
    #    # Sort by y_top (top to bottom), then x_left (left to right)
    #    sort_cols = []
    #    if "y_top" in df_clean.columns:
    #        sort_cols.append("y_top")
    #    if "x_left" in df_clean.columns:
    #        sort_cols.append("x_left")
    #    if sort_cols:
    #        df_clean = df_clean.sort_values(sort_cols).reset_index(drop=True)
    #    else:
    #        df_clean = df_clean.reset_index(drop=True)
    #    df_clean["box_id"] = range(1, len(df_clean) + 1) # 1-based box_id
    
    # Alternative: Reindex by structure_tag_id, but insert hr/img by y_top
    if "box_id" in df_clean.columns and "structure_tag" in df_clean.columns:
        # Separate special tags (hr, img) from regular content
        is_special = df_clean["structure_tag"].isin(["hr", "img"])
        special_rows = df_clean[is_special].copy()
        regular_rows = df_clean[~is_special].copy()
        
        if not special_rows.empty and not regular_rows.empty and "y_top" in df_clean.columns:
            # Sort regular rows by structure_tag_id (already in order)
            if "structure_tag_id" in regular_rows.columns:
                regular_rows = regular_rows.sort_values("structure_tag_id")
            
            # Insert special rows based on y_top position
            result_rows = []
            special_rows = special_rows.sort_values("y_top")
            special_idx = 0
            
            for idx, regular_row in regular_rows.iterrows():
                regular_y_top = regular_row["y_top"]
                
                # Insert all special rows that come before this regular row
                while special_idx < len(special_rows):
                    special_row = special_rows.iloc[special_idx]
                    if special_row["y_top"] < regular_y_top:
                        result_rows.append(special_row)
                        special_idx += 1
                    else:
                        break
                
                # Add the regular row
                result_rows.append(regular_row)
            
            # Add any remaining special rows at the end
            while special_idx < len(special_rows):
                result_rows.append(special_rows.iloc[special_idx])
                special_idx += 1
            
            # Reconstruct dataframe
            df_clean = pd.DataFrame(result_rows).reset_index(drop=True)
        else:
            # Fallback: just sort by structure_tag_id if available
            if "structure_tag_id" in df_clean.columns:
                df_clean = df_clean.sort_values("structure_tag_id").reset_index(drop=True)
            else:
                df_clean = df_clean.reset_index(drop=True)
        
        # Reassign box_id based on new order
        df_clean["box_id"] = range(1, len(df_clean) + 1)  # 1-based box_id
    
    # 6) Add block_role based on structure_tag
    if "structure_tag" in df_clean.columns:
        df_clean["block_role"] = None
        df_clean.loc[df_clean["structure_tag"] == "hr", "block_role"] = "hr"
        df_clean.loc[df_clean["structure_tag"] == "img", "block_role"] = "image"
        # TODO: Potentially add table, but not sure because of the fake 1 row tables

    return df_clean


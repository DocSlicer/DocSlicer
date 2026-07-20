"""
step_06_hierarchy_builder.py

Builds the document heading hierarchy from annotated lines_df.

Receives lines_df from step_05 (which already has block_type, heading_fp_id and
heading_id) and produces the heading tree:
  - heading_weight_static / heading_weight_dynamic
  - parent_heading_id / parent_heading_text / heading_level

heading_id itself is assigned in step_05, before heading suppression, so that
suppression can act on whole multi-line heading units. This module only reads it.

Public API: assign_doc_hierarchy(lines_df)
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

from .step_05_heading_detector import (
    _HEADING_ROLES,
    NUMBERED_HEADING_TYPES,
    _RE_NUMERIC_PREFIX,
)


# ================================================================================
# Add heading weights
# ================================================================================

# static feature weights
STATIC_WEIGHTS = {
    "is_bold": 1.0,
    "is_uppercase": 1.0,
    "text_align_center": 1.0,
    "is_italic": -0.5,
}

# heading type rank (lower = stronger / higher in hierarchy)
_HEADING_TYPE_RANK_TIERS = {
    0: ["exhibit", "volume", "appendix", "annex", "schedule", "attachment",
        "form", "book", "amendment", "endnote"], # doc-level: always resets the stack
    1: ["title"],
    2: ["subtitle"],
    3: ["chapter", "part"], # Chapter sometimes higher, sometimes lower than part
    4: ["subchapter", "subpart", "item", "division"],
    5: ["subdivision"],
    6: ["note", "section", "proposal"],
    7: ["subsection", "section_symbol", "section_abbreviated"],
    8: ["article", "rule"],
    9: ["free_form"],
    10: ["table", "figure"], # These are usually captions
}
HEADING_TYPE_RANK = {t: rank for rank, types in _HEADING_TYPE_RANK_TIERS.items() for t in types}

# tier-0 heading types restart the hierarchy at level 1 regardless of what precedes them
STACK_RESET_TYPES = frozenset(_HEADING_TYPE_RANK_TIERS[0])

# _HEADING_ROLES, NUMBERED_HEADING_TYPES and _RE_NUMERIC_PREFIX are imported from step_05.
_RE_ALPHA_DOTTED_PREFIX = re.compile(r'^\s*[A-Za-z]\.?\d+(?:\.\d+)*')


def _numeric_depth(text: str, heading_type: str) -> int:
    """Return the hierarchical depth from a numbered heading's leading prefix.

    Examples: "1." -> 1,  "1.2." -> 2,  "10.2 Revenue" -> 2,  "1.2.3" -> 3
    Roman and plain-alpha types are always depth 1 (they don't nest by number).
    """
    if heading_type in ("roman_numbered_heading", "alpha_numbered_heading"):
        return 1
    m = _RE_NUMERIC_PREFIX.match(text)
    if m:
        return len(m.group(1).split("."))
    m = _RE_ALPHA_DOTTED_PREFIX.match(text)
    if m:
        return max(1, m.group(0).strip().count("."))
    return 1


def _add_heading_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds heading_weight_static and heading_weight_dynamic to heading rows.
    Non-heading rows receive NaN.
    """
    out = df.copy()
    out["heading_weight_static"] = np.nan
    out["heading_weight_dynamic"] = np.nan

    if "block_type" not in out.columns:
        return out

    is_heading = out["block_type"].astype("string").str.strip().str.lower().isin(_HEADING_ROLES).fillna(False)
    if not is_heading.any():
        return out

    # Static score
    static = pd.Series(0.0, index=out.index, dtype="float32")
    for col, weight in [("is_bold", "is_bold"), ("is_uppercase", "is_uppercase"), ("is_italic", "is_italic")]:
        if col in out.columns:
            static.loc[is_heading] += out.loc[is_heading, col].fillna(False).astype(int) * STATIC_WEIGHTS[weight]
    if "text_align" in out.columns:
        is_center = out.loc[is_heading, "text_align"].astype("string").str.strip().str.lower().eq("center").fillna(False)
        static.loc[is_heading] += is_center.astype(int) * STATIC_WEIGHTS["text_align_center"]
    static = static.clip(lower=0.0)
    out.loc[is_heading, "heading_weight_static"] = static.loc[is_heading].astype("float32")

    # Dynamic score (relative to previous heading within same region)
    region_key = (out["section"].astype("string").fillna("")
                  if "section" in out.columns
                  else pd.Series("", index=out.index, dtype="string"))
    idx = out.index[is_heading]
    groups = region_key.loc[idx]
    dynamic = pd.Series(0, index=idx, dtype="int16")

    if "font_size_ratio" in out.columns:
        font = pd.to_numeric(out.loc[idx, "font_size_ratio"], errors="coerce")
        font_prev = font.groupby(groups).shift(1)
        dynamic.loc[font > font_prev] += 1
        dynamic.loc[font < font_prev] -= 1

    if "heading_type" in out.columns:
        rank = out.loc[idx, "heading_type"].map(HEADING_TYPE_RANK)
        rank_prev = rank.groupby(groups).shift(1)
        dynamic.loc[rank < rank_prev] += 1
        dynamic.loc[rank > rank_prev] -= 1

    out.loc[idx, "heading_weight_dynamic"] = dynamic
    return out


# ================================================================================
# Infer heading hierarchy
# ================================================================================

def _infer_heading_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds parent_heading_id, parent_heading_text, and heading_level to heading rows.
    Processes block_type in {heading, toc_heading, exhibit_heading}.
    Assumes df is already in reading order.
    """
    out = df.copy()

    for col, dtype in [("parent_heading_id", "Int64"), ("heading_level", "Int64")]:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), index=out.index, dtype=dtype)
    if "parent_heading_text" not in out.columns:
        out["parent_heading_text"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")

    if "block_type" not in out.columns or "heading_id" not in out.columns:
        return out

    heading_mask = (
        out["block_type"].astype("string").str.strip().str.lower()
        .isin(_HEADING_ROLES).fillna(False)
    ) & out["heading_id"].notna()

    if not heading_mask.any():
        return out

    region = (out["section"].astype("string").fillna("")
              if "section" in out.columns
              else pd.Series("", index=out.index, dtype="string"))

    def sf(v) -> float:
        try:
            return 0.0 if pd.isna(v) else float(v)
        except Exception:
            return 0.0

    for region_name in region[heading_mask].unique():
        region_mask = heading_mask & (region == region_name)
        region_rows = out[region_mask].copy()
        if region_rows.empty:
            continue

        unique_headings = []
        for hid, group in region_rows.groupby("heading_id", sort=False):
            first, last = group.iloc[0], group.iloc[-1]
            unique_headings.append({
                "heading_id": int(hid),
                "indices": group.index.tolist(),
                "fp": first["heading_fp_id"] if "heading_fp_id" in first.index and pd.notna(first["heading_fp_id"]) else pd.NA,
                "text": first["text"] if "text" in first.index and pd.notna(first["text"]) else "",
                "static": sf(first["heading_weight_static"]) if "heading_weight_static" in first.index else 0.0,
                "dynamic": sf(first["heading_weight_dynamic"]) if "heading_weight_dynamic" in first.index else 0.0,
                "heading_type": (str(first["heading_type"]).strip().lower()
                                 if "heading_type" in first.index and pd.notna(first["heading_type"])
                                 else "free_form"),
                "end_line_id": last.get("line_id") if "line_id" in last.index else None,
                "block_type": (str(first["block_type"]).strip().lower()
                               if "block_type" in first.index and pd.notna(first["block_type"])
                               else ""),
                "numeric_depth": _numeric_depth(
                    str(first["text"]) if "text" in first.index and pd.notna(first["text"]) else "",
                    str(first["heading_type"]).strip().lower()
                    if "heading_type" in first.index and pd.notna(first["heading_type"]) else "free_form",
                ),
            })

        path_stack = []
        relationship_memory = {}

        for heading in unique_headings:
            curr_heading_id = heading["heading_id"]
            curr_fp        = heading["fp"]
            curr_text      = heading["text"]
            curr_static    = heading["static"]
            curr_total     = curr_static + heading["dynamic"]
            curr_type      = heading["heading_type"]
            curr_line_id   = heading["end_line_id"]
            curr_nd        = heading["numeric_depth"]
            indices        = heading["indices"]

            if not path_stack or (
                curr_type in STACK_RESET_TYPES
                and heading["block_type"] != "toc_heading"
            ):
                for idx in indices:
                    out.at[idx, "parent_heading_id"] = pd.NA
                    out.at[idx, "parent_heading_text"] = pd.NA
                    out.at[idx, "heading_level"] = 1
                path_stack.clear()
                path_stack.append({"heading_id": curr_heading_id, "fp": curr_fp, "text": curr_text,
                                   "level": 1, "static_weight": curr_static, "heading_type": curr_type,
                                   "end_line_id": curr_line_id, "block_type": heading["block_type"],
                                   "numeric_depth": curr_nd})
                continue

            prior = path_stack[-1]
            prior_fp = prior["fp"]
            fp_path = [n["fp"] for n in path_stack]

            # Decision
            if pd.notna(curr_fp) and pd.notna(prior_fp) and curr_fp == prior_fp:
                if curr_type in NUMBERED_HEADING_TYPES and prior.get("heading_type") in NUMBERED_HEADING_TYPES:
                    prior_nd = prior["numeric_depth"]
                    if curr_nd > prior_nd:
                        decision = "child"
                    elif curr_nd == prior_nd:
                        decision = "sibling"
                    else:
                        decision = "numbered_pop"
                else:
                    decision = "sibling"
            elif (prior.get("block_type") == "heading"
                  and curr_type not in NUMBERED_HEADING_TYPES
                  and curr_line_id is not None and prior["end_line_id"] is not None
                  and pd.notna(curr_line_id) and pd.notna(prior["end_line_id"])
                  and curr_line_id == prior["end_line_id"] + 1):
                decision = "child"
            elif pd.notna(curr_fp) and curr_fp in fp_path:
                decision = "numbered_reattach" if curr_type in NUMBERED_HEADING_TYPES else "reattach"
            elif curr_total < prior["static_weight"]:
                decision = "child"
            elif curr_total > prior["static_weight"]:
                decision = "pop_up"
            else:
                decision = "pop_up" if any(
                    child_fp == prior_fp and parent_fp == curr_fp
                    for (parent_fp, child_fp) in relationship_memory
                ) else "child"

            def _node(level):
                return {"heading_id": curr_heading_id, "fp": curr_fp, "text": curr_text,
                        "level": level, "static_weight": curr_static, "heading_type": curr_type,
                        "end_line_id": curr_line_id, "block_type": heading["block_type"],
                        "numeric_depth": curr_nd}

            def _set(indices, pid, ptxt, level):
                for idx in indices:
                    out.at[idx, "parent_heading_id"] = pid
                    out.at[idx, "parent_heading_text"] = ptxt
                    out.at[idx, "heading_level"] = level

            def _remember(parent_fp, child_fp):
                if pd.notna(parent_fp) and pd.notna(child_fp):
                    relationship_memory[(parent_fp, child_fp)] = relationship_memory.get((parent_fp, child_fp), 0) + 1

            if decision == "sibling":
                if len(path_stack) > 1:
                    parent = path_stack[-2]
                    _set(indices, parent["heading_id"], parent["text"], prior["level"])
                    _remember(parent["fp"], curr_fp)
                else:
                    _set(indices, pd.NA, pd.NA, 1)
                path_stack[-1] = _node(prior["level"])

            elif decision == "child":
                _set(indices, prior["heading_id"], prior["text"], prior["level"] + 1)
                _remember(prior_fp, curr_fp)
                path_stack.append(_node(prior["level"] + 1))

            elif decision == "reattach":
                target_idx = next((i for i in range(len(path_stack) - 1, -1, -1)
                                   if pd.notna(path_stack[i]["fp"]) and path_stack[i]["fp"] == curr_fp), None)
                if target_idx is not None:
                    path_stack = path_stack[:target_idx + 1]
                    if target_idx > 0:
                        parent = path_stack[target_idx - 1]
                        _set(indices, parent["heading_id"], parent["text"], path_stack[target_idx]["level"])
                    else:
                        _set(indices, pd.NA, pd.NA, 1)
                    path_stack[-1] = _node(path_stack[-1]["level"])

            elif decision == "pop_up":
                special_idx = next((i for i in range(len(path_stack) - 1, -1, -1)
                                    if path_stack[i]["heading_type"] not in ("free_form", "table", "figure")), None)
                if special_idx is not None:
                    sp = path_stack[special_idx]
                    path_stack = path_stack[:special_idx + 1]
                    _set(indices, sp["heading_id"], sp["text"], sp["level"] + 1)
                    _remember(sp["fp"], curr_fp)
                    path_stack.append(_node(sp["level"] + 1))
                else:
                    path_stack.pop()
                    is_high = HEADING_TYPE_RANK.get(curr_type, 999) <= 2
                    if is_high or not path_stack:
                        _set(indices, pd.NA, pd.NA, 1)
                        path_stack.append(_node(1))
                    else:
                        parent = path_stack[-1]
                        _set(indices, parent["heading_id"], parent["text"], parent["level"] + 1)
                        _remember(parent["fp"], curr_fp)
                        path_stack.append(_node(parent["level"] + 1))

            elif decision == "numbered_pop":
                # Shallower depth than prior: walk back to the deepest numbered node
                # whose depth <= curr_nd, then attach as sibling (equal) or child (shallower).
                target_idx = next(
                    (i for i in range(len(path_stack) - 1, -1, -1)
                     if path_stack[i].get("heading_type") in NUMBERED_HEADING_TYPES
                     and path_stack[i]["numeric_depth"] <= curr_nd),
                    None,
                )
                if target_idx is not None:
                    node_at = path_stack[target_idx]
                    path_stack = path_stack[:target_idx + 1]
                    if node_at["numeric_depth"] == curr_nd:
                        if target_idx > 0:
                            parent = path_stack[target_idx - 1]
                            _set(indices, parent["heading_id"], parent["text"], node_at["level"])
                            _remember(parent["fp"], curr_fp)
                        else:
                            _set(indices, pd.NA, pd.NA, 1)
                        path_stack[-1] = _node(node_at["level"])
                    else:
                        _set(indices, node_at["heading_id"], node_at["text"], node_at["level"] + 1)
                        _remember(node_at["fp"], curr_fp)
                        path_stack.append(_node(node_at["level"] + 1))
                else:
                    _set(indices, pd.NA, pd.NA, 1)
                    path_stack.clear()
                    path_stack.append(_node(1))

            elif decision == "numbered_reattach":
                # fp seen earlier but prior is not numbered: find the deepest stack node
                # with the same fp whose numeric_depth <= curr_nd.
                target_idx = next(
                    (i for i in range(len(path_stack) - 1, -1, -1)
                     if pd.notna(path_stack[i].get("fp")) and path_stack[i]["fp"] == curr_fp
                     and path_stack[i].get("heading_type") in NUMBERED_HEADING_TYPES
                     and path_stack[i]["numeric_depth"] <= curr_nd),
                    None,
                )
                if target_idx is None:
                    target_idx = next(
                        (i for i in range(len(path_stack) - 1, -1, -1)
                         if pd.notna(path_stack[i].get("fp")) and path_stack[i]["fp"] == curr_fp),
                        None,
                    )
                if target_idx is not None:
                    node_at = path_stack[target_idx]
                    path_stack = path_stack[:target_idx + 1]
                    if node_at.get("numeric_depth") == curr_nd:
                        if target_idx > 0:
                            parent = path_stack[target_idx - 1]
                            _set(indices, parent["heading_id"], parent["text"], node_at["level"])
                            _remember(parent["fp"], curr_fp)
                        else:
                            _set(indices, pd.NA, pd.NA, 1)
                        path_stack[-1] = _node(node_at["level"])
                    else:
                        _set(indices, node_at["heading_id"], node_at["text"], node_at["level"] + 1)
                        _remember(node_at["fp"], curr_fp)
                        path_stack.append(_node(node_at["level"] + 1))

    return out


# ================================================================================
# Public API
# ================================================================================

def assign_doc_hierarchy(
    lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full document hierarchy pipeline: weights → hierarchy tree.
    heading_id is expected to already exist (assigned in step_05).

    Returns lines_df with all heading columns added.
    """
    out = _add_heading_weights(lines_df)
    out = _infer_heading_hierarchy(out)
    return out

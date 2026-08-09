# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Partition blocks into size-bounded chunks (greedy/DP) and join their text."""

# step_06_chunk_builder.py

from __future__ import annotations

import functools
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from .._utils.df_aggregation.registry_aggregator import Agg, aggregate_to, group_join

_log = logging.getLogger(__name__)

# =======================================================================================================================
# CONFIG
# =======================================================================================================================

# Default chunking parameters
_DEFAULT_MAX_CHUNK_CHARS = 3200       # Only split if active_heading_id content exceeds this
_DEFAULT_OPTIMAL_CHUNK_CHARS = 1600   # Used to split when content exceeds this
_DEFAULT_SOFTMIN_CHUNK_CHARS = 700    # Chunks below this size become undesirable
_DEFAULT_MIN_CHUNK_CHARS = 400        # Min boundary for chunking
#_DEFAULT_OVERLAP_CHUNK_CHARS = 200    # Overlap between chunks if splitting is necessary

# Bounds for chunking parameters (validation limits)
_MAX_CHUNK_CHARS_MIN = 800
_MAX_CHUNK_CHARS_MAX = 8000
_OPTIMAL_CHUNK_CHARS_MIN = 400
_OPTIMAL_CHUNK_CHARS_MAX = 4000
_SOFTMIN_CHUNK_CHARS_MIN = 200
_SOFTMIN_CHUNK_CHARS_MAX = 2000
_MIN_CHUNK_CHARS_MIN = 100
_MIN_CHUNK_CHARS_MAX = 1000

# DP is the globally optimal solution for chunk partitioning: it finds the exact split of n blocks
# into k contiguous groups that minimizes a scoring function (penalizing deviation from the target
# chunk size). Without strict constraints, the worst-case cost scales cubically to O(n³) — a single 
# group of 332 blocks with a high character count was measured taking ~30s on real production data. 
# 
# Beyond the threshold below, we fall back to a greedy O(n) packer targeting optimal_chunk_chars. 
# The visual and retrieval quality remains near-identical for large groups because the optimal split 
# at scale naturally converges to a simple "pack to target size" behavior.
_DP_MAX_BLOCKS = 100

# Block roles that should be removed from chunk content
_NOISE_BLOCK_TYPES= {"hr", "page_label", "image", "suppressed_repeated_heading", "navigation", "vertical_text"}

# Block roles that should be treated as headings (for chunk_heading)
_HEADING_BLOCK_TYPES = {"heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph"}


# =======================================================================================================================
# CHUNK SIZE CONFIG
# =======================================================================================================================

@dataclass(frozen=True)
class ChunkSizeConfig:
    """
    Chunk size targets, clamped to bounds and ordered on construction.

    The invariant min <= softmin <= optimal <= max is enforced in __post_init__,
    so any constructed instance is guaranteed valid — internal functions don't
    have to re-validate or trust that a separate validator was called first.
    """
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS
    optimal_chunk_chars: int = _DEFAULT_OPTIMAL_CHUNK_CHARS
    softmin_chunk_chars: int = _DEFAULT_SOFTMIN_CHUNK_CHARS
    min_chunk_chars: int = _DEFAULT_MIN_CHUNK_CHARS

    def __post_init__(self) -> None:
        # frozen=True blocks normal attribute assignment, so use object.__setattr__
        # Clamp to absolute bounds
        max_c = max(_MAX_CHUNK_CHARS_MIN, min(int(self.max_chunk_chars), _MAX_CHUNK_CHARS_MAX))
        optimal_c = max(_OPTIMAL_CHUNK_CHARS_MIN, min(int(self.optimal_chunk_chars), _OPTIMAL_CHUNK_CHARS_MAX))
        softmin_c = max(_SOFTMIN_CHUNK_CHARS_MIN, min(int(self.softmin_chunk_chars), _SOFTMIN_CHUNK_CHARS_MAX))
        min_c = max(_MIN_CHUNK_CHARS_MIN, min(int(self.min_chunk_chars), _MIN_CHUNK_CHARS_MAX))

        # Ensure logical ordering: min <= softmin <= optimal <= max
        if min_c > softmin_c:
            min_c = softmin_c
        if softmin_c > optimal_c:
            softmin_c = optimal_c
        if optimal_c > max_c:
            optimal_c = max_c

        object.__setattr__(self, "max_chunk_chars", max_c)
        object.__setattr__(self, "optimal_chunk_chars", optimal_c)
        object.__setattr__(self, "softmin_chunk_chars", softmin_c)
        object.__setattr__(self, "min_chunk_chars", min_c)


# =======================================================================================================================
# STEP 1: Prepare DataFrame
# =======================================================================================================================

def _prepare_blocks_df(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the blocks dataframe for chunking:
    - Add 'active_heading_id' column that propagates heading_id values downward
    - Remove blocks with noise block roles
    
    Args:
        blocks_df: DataFrame from block merger step
        
    Returns:
        Prepared DataFrame ready for chunk building
    """
    df = blocks_df.copy()
    
    # Add active_heading_id column - forward fill heading_id values
    # This propagates each heading_id to all blocks below it until the next heading
    if 'heading_id' in df.columns:
        # Convert to string to handle Int64 dtype properly
        heading_id_str = df['heading_id'].astype(str).replace(['<NA>', 'nan', 'None'], '')
        df['active_heading_id'] = heading_id_str.replace('', np.nan).ffill().fillna('')
    else:
        # If heading_id doesn't exist, create empty active_heading_id
        df['active_heading_id'] = ''
    
    # Remove noise blocks
    if 'block_type' in df.columns:
        df = df[~df['block_type'].isin(_NOISE_BLOCK_TYPES)].reset_index(drop=True)
    
    return df


# =======================================================================================================================
# STEP 2: Assign Chunk IDs (Decision Logic)
# =======================================================================================================================

# ------------------------------
# Oversize block pre-split (Decision-layer only; creates virtual rows)
# ------------------------------

def _find_best_split_point(text: str, target_pos: int, max_search_range: int = 200) -> int:
    """
    Find the best position to split text near target_pos.
    
    Priority:
    1. Newline (\n) within search range
    2. Sentence boundary (. ! ?) followed by space/newline within search range
    3. Comma (,) followed by space within search range
    4. Word boundary (space) within search range
    5. Exact target_pos as fallback
    
    Args:
        text: Text to split
        target_pos: Ideal split position
        max_search_range: Maximum distance to search for better break point
        
    Returns:
        Best split position
    """
    n = len(text)
    if target_pos >= n:
        return n
    if target_pos <= 0:
        return 0
    
    # Define search window
    search_start = max(0, target_pos - max_search_range)
    search_end = min(n, target_pos + max_search_range)
    
    # Look for newlines first (highest priority)
    # Search backward first (prefer breaking earlier to keep chunks smaller)
    for i in range(target_pos, search_start - 1, -1):
        if text[i] == '\n':
            return i + 1  # Split after the newline
    
    # Search forward for newline
    for i in range(target_pos + 1, search_end):
        if text[i] == '\n':
            return i + 1
    
    # Look for sentence boundaries (. ! ?) followed by space or newline
    # Search backward first
    for i in range(target_pos, search_start - 1, -1):
        if i > 0 and text[i - 1] in '.!?' and (i >= n or text[i] in ' \n\t'):
            return i  # Split after the punctuation + space
    
    # Search forward for sentence boundary
    for i in range(target_pos + 1, search_end):
        if i > 0 and text[i - 1] in '.!?' and (i >= n or text[i] in ' \n\t'):
            return i
    
    # Look for comma followed by space
    # Search backward first
    for i in range(target_pos, search_start - 1, -1):
        if i > 0 and text[i - 1] == ',' and (i >= n or text[i] in ' \n\t'):
            return i  # Split after the comma + space
    
    # Search forward for comma
    for i in range(target_pos + 1, search_end):
        if i > 0 and text[i - 1] == ',' and (i >= n or text[i] in ' \n\t'):
            return i
    
    # Look for word boundaries (spaces)
    # Search backward first
    for i in range(target_pos, search_start - 1, -1):
        if text[i] == ' ':
            return i + 1  # Split after the space
    
    # Search forward for word boundary
    for i in range(target_pos + 1, search_end):
        if text[i] == ' ':
            return i + 1
    
    # Fallback: use exact target position
    return target_pos


def _split_text_into_parts(
    text: str,
    max_chars: int,
    target_chars: int,
    softmin_chars: int,
) -> List[str]:
    """
    Split text into parts:
      1) Prefer paragraph boundaries (\n\n)
      2) Then line boundaries (\n)
      3) Finally hard-slice

    Goal: parts roughly near target_chars, all <= max_chars when possible.
    No overlap by default.
    """
    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph/line boundaries as "units"
    units: List[str] = []
    if "\n\n" in text:
        units = [u for u in text.split("\n\n") if u != ""]
        joiner = "\n\n"
    elif "\n" in text:
        units = [u for u in text.split("\n") if u != ""]
        joiner = "\n"
    else:
        units = []
        joiner = ""

    # If we have units, pack them into parts near target_chars, respecting max_chars
    parts: List[str] = []
    if units:
        buf: List[str] = []
        buf_len = 0

        def flush():
            nonlocal buf, buf_len
            if not buf:
                return
            parts.append(joiner.join(buf))
            buf = []
            buf_len = 0

        for u in units:
            u_len = len(u)
            # if a single unit is too large, fall back to hard slicing that unit
            if u_len > max_chars:
                flush()
                parts.extend(_split_text_into_parts(u, max_chars=max_chars, target_chars=target_chars, softmin_chars=softmin_chars))
                continue

            sep = 0 if not buf else len(joiner)
            next_len = buf_len + sep + u_len

            # if adding would exceed max, flush first
            if buf and next_len > max_chars:
                flush()

            # if buffer is already "good enough" and adding would push far past target, flush to keep balance
            if buf and buf_len >= softmin_chars and (buf_len >= target_chars) and (buf_len + sep + u_len) > (target_chars + (target_chars // 2)):
                flush()

            # add to buffer
            sep = 0 if not buf else len(joiner)
            buf_len = buf_len + sep + u_len
            buf.append(u)

        flush()

        # If last part is tiny, rebalance with previous if possible
        if len(parts) >= 2 and len(parts[-1]) < softmin_chars:
            last = parts.pop()
            prev = parts.pop()
            candidate = prev + joiner + last if joiner else prev + last
            if len(candidate) <= max_chars:
                parts.append(candidate)
            else:
                # put back
                parts.append(prev)
                parts.append(last)

        return parts

    # Hard slice fallback: make near-target pieces with smart break points
    # This handles cases where text has no paragraph/line boundaries,
    # or when individual units are too large
    parts = []
    i = 0
    n = len(text)
    step = min(max_chars, max(1, target_chars))
    
    while i < n:
        # Calculate initial target end position
        initial_j = min(n, i + step)
        
        # If we're at the end, just take the rest
        if initial_j == n:
            parts.append(text[i:n])
            break
        
        # Find smart break point near the target position
        # Search range: allow going back/forward up to 200 chars to find good break
        smart_j = _find_best_split_point(text, initial_j, max_search_range=200)
        
        # Safety check: ensure we're making progress
        if smart_j <= i:
            smart_j = initial_j
        
        # Ensure we don't exceed max_chars (unless unavoidable)
        if smart_j - i > max_chars:
            # If smart break point would create too large chunk, use max_chars boundary
            smart_j = i + max_chars
            # But try to find a better break within this constrained range
            smart_j = _find_best_split_point(text, smart_j, max_search_range=50)
            if smart_j <= i:
                smart_j = i + max_chars
        
        parts.append(text[i:smart_j])
        i = smart_j

    # Rebalance tail
    if len(parts) >= 2 and len(parts[-1]) < softmin_chars:
        tail = parts.pop()
        prev = parts.pop()
        # borrow from prev if possible
        need = softmin_chars - len(tail)
        if need > 0 and len(prev) - need >= softmin_chars:
            parts.append(prev[:-need])
            parts.append(prev[-need:] + tail)
        else:
            parts.append(prev)
            parts.append(tail)

    return parts


def _heading_block_mask(g: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows whose block_type counts as a heading."""
    block_types = (
        g.get("block_type", pd.Series([""] * len(g), index=g.index))
        .astype("string")
        .str.strip()
        .str.lower()
    )
    return block_types.isin(_HEADING_BLOCK_TYPES).fillna(False)


def _group_heading_chars(g: pd.DataFrame, heading_mask: Optional[pd.Series] = None) -> int:
    """Chars contributed by the heading blocks — prepended to EVERY chunk of the group."""
    if heading_mask is None:
        heading_mask = _heading_block_mask(g)
    if not bool(heading_mask.any()):
        return 0
    return int(g.loc[heading_mask, "embed_char_count"].astype(int).sum())


def _content_budget_chars(heading_chars: int, cfg: ChunkSizeConfig) -> int:
    """
    Room left for content once the heading is accounted for.

    Every chunk in a heading group carries the heading text, so the ceiling that
    content blocks must respect is max - heading_chars, never max. Floored at
    min_chunk_chars so a pathologically long heading still yields a usable budget.
    """
    return max(int(cfg.min_chunk_chars), int(cfg.max_chunk_chars) - int(heading_chars))


def _explode_oversize_blocks_within_group(
    group_df: pd.DataFrame,
    cfg: ChunkSizeConfig,
    budget_chars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Replace any content row with embed_char_count > budget_chars by multiple "virtual" rows.

    budget_chars defaults to max_chunk_chars, but callers that know the group's
    heading overhead must pass max_chunk_chars - heading_chars instead. A block
    that slips under max yet over the budget stays un-split and then makes every
    candidate partition infeasible in _partition_blocks_dp.

    Heading rows are never split: their chars land in heading_chars either way, so
    splitting them buys nothing and only fabricates extra rows.

    Adds:
      - sub_block_index (0 for original non-split blocks, 1..N for split parts)
      - is_virtual_sub_block (bool)
      - orig_row_id (stable id within this function)
    Keeps order by original sort + sub_block_index.
    """
    max_chunk_chars = int(budget_chars) if budget_chars is not None else cfg.max_chunk_chars
    optimal_chunk_chars = min(cfg.optimal_chunk_chars, max_chunk_chars)
    softmin_chunk_chars = min(cfg.softmin_chunk_chars, optimal_chunk_chars)

    g = group_df.copy()

    if "embed_char_count" not in g.columns:
        g["embed_char_count"] = g["text"].astype("string").fillna("").str.len().astype(int)

    # provide a stable original ordering key within the group
    g = g.reset_index(drop=False).rename(columns={"index": "_orig_index"})
    g["sub_block_index"] = 0
    g["is_virtual_sub_block"] = False

    is_heading = _heading_block_mask(g)

    rows_out: List[pd.Series] = []

    for pos, (_, row) in enumerate(g.iterrows()):
        L = int(row["embed_char_count"])
        if L <= max_chunk_chars or bool(is_heading.iloc[pos]):
            rows_out.append(row)
            continue

        # Oversize: split into ~optimal-sized parts (DP-friendly atoms)
        text = str(row.get("text", "") or "")
        parts = _split_text_into_parts(
            text=text,
            max_chars=max_chunk_chars,
            target_chars=optimal_chunk_chars,
            softmin_chars=softmin_chunk_chars,
        )

        # Emit one row per part (virtual rows)
        for si, part in enumerate(parts, start=1):
            r2 = row.copy()
            r2["text"] = part
            r2["embed_char_count"] = int(len(part))
            r2["sub_block_index"] = int(si)
            r2["is_virtual_sub_block"] = True
            rows_out.append(r2)

    out = pd.DataFrame(rows_out)

    # Order: original order, then sub_block_index (0 comes before 1..N but oversize rows only have 1..N)
    out = out.sort_values(["_orig_index", "sub_block_index"], kind="mergesort").reset_index(drop=True)

    return out


# ------------------------------
# Main Decision Engine
# ------------------------------

# Dynamic Programming Algorithm (DP)
def _score_chunk_len_vec(
    L_arr: np.ndarray,
    max_chunk_chars: int,
    optimal_chunk_chars: int,
    softmin_chunk_chars: int,
    min_chunk_chars: int,
) -> np.ndarray:
    """Vectorized version of _score_chunk_len operating on a numpy array of lengths."""
    L = L_arr.astype(np.float64)
    cost = (L - optimal_chunk_chars) ** 2
    cost = np.where(L > max_chunk_chars, np.inf, cost)
    hard_min = L < min_chunk_chars
    soft_min = (~hard_min) & (L < softmin_chunk_chars)
    cost = np.where(hard_min, cost + 50.0 * (min_chunk_chars - L) ** 2, cost)
    cost = np.where(soft_min, cost + 5.0 * (softmin_chunk_chars - L) ** 2, cost)
    return cost


def _partition_blocks_greedy(
    block_lens: List[int],
    heading_chars: int,
    max_chunk_chars: int,
    optimal_chunk_chars: int,
) -> List[Tuple[int, int]]:
    """
    O(n) greedy partition: accumulate blocks until we reach optimal_chunk_chars,
    then start a new chunk. Respects max_chunk_chars as a hard ceiling.
    """
    n = len(block_lens)
    if n == 0:
        return []

    cuts: List[Tuple[int, int]] = []
    start = 0
    acc = heading_chars

    for i, L in enumerate(block_lens):
        next_acc = acc + L
        # Flush before adding this block if it would push past optimal AND we have content
        if acc > heading_chars and next_acc > optimal_chunk_chars:
            cuts.append((start, i - 1))
            start = i
            acc = heading_chars + L
        else:
            acc = next_acc
            # Hard ceiling: flush immediately if even a single block exceeds max
            if acc > max_chunk_chars and i > start:
                cuts.append((start, i - 1))
                start = i
                acc = heading_chars + L

    cuts.append((start, n - 1))
    return cuts


def _partition_blocks_dp(
    block_lens: List[int],
    heading_chars: int,
    cfg: ChunkSizeConfig,
) -> List[Tuple[int, int]]:
    """
    Partition ordered blocks into k contiguous groups using DP, where
    each group's effective length is heading_chars + sum(block_lens[group]).

    Falls back to a greedy O(n) approach when the group has too many blocks
    for DP to be practical.

    Returns list of (start_idx, end_idx) inclusive indices into block_lens.
    """
    # Pull config into locals: these are read inside the vectorized inner loop,
    # so avoid repeated attribute lookups on the hot path.
    max_chunk_chars = cfg.max_chunk_chars
    optimal_chunk_chars = cfg.optimal_chunk_chars
    softmin_chunk_chars = cfg.softmin_chunk_chars
    min_chunk_chars = cfg.min_chunk_chars

    n = len(block_lens)
    if n == 0:
        return []

    prefix = np.zeros(n + 1, dtype=np.int64)
    for i, v in enumerate(block_lens, start=1):
        prefix[i] = prefix[i - 1] + int(v)

    total = heading_chars + int(prefix[n])

    k_min = int(np.ceil(total / max_chunk_chars))
    k_target = int(np.ceil(total / optimal_chunk_chars))

    # Only try a small range of k values (balanced vs feasible)
    k_lo = max(1, k_min)
    k_hi = max(k_lo, min(n, k_target + 1))  # +1 gives a bit of flexibility

    # Greedy fallback when the DP work estimate exceeds the budget.
    # DP cost scales as k_hi² × n; empirically k_hi²×n ≈ 8M → ~30s, so cap at 800K (~3s).
    if n > _DP_MAX_BLOCKS or k_hi * k_hi * n > 800_000:
        return _partition_blocks_greedy(block_lens, heading_chars, max_chunk_chars, optimal_chunk_chars)

    best_cost = float("inf")
    best_cuts: Optional[List[Tuple[int, int]]] = None

    # DP for each k candidate — inner j-loop vectorized over numpy
    for k in range(k_lo, k_hi + 1):
        # dp[c][i] = min cost to partition first i blocks into c groups
        dp = np.full((k + 1, n + 1), np.inf, dtype=np.float64)
        prev = np.full((k + 1, n + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0

        for c in range(1, k + 1):
            j_start = c - 1
            for i in range(c, n + 1):
                # j ranges from j_start to i-1 — vectorize this whole slice
                j_vals = np.arange(j_start, i, dtype=np.int64)

                # L[j] = heading_chars + prefix[i] - prefix[j]
                L_vals = (heading_chars + int(prefix[i])) - prefix[j_vals].astype(np.float64)
                cost_vals = _score_chunk_len_vec(
                    L_vals, max_chunk_chars, optimal_chunk_chars, softmin_chunk_chars, min_chunk_chars
                )

                prev_dp = dp[c - 1, j_vals]
                cand_vals = prev_dp + cost_vals

                # Only consider candidates where the predecessor state was reachable
                cand_vals = np.where(np.isfinite(prev_dp), cand_vals, np.inf)

                best_local = int(np.argmin(cand_vals))
                best_val = cand_vals[best_local]

                if best_val < dp[c, i]:
                    dp[c, i] = best_val
                    prev[c, i] = j_vals[best_local]

        final_cost = float(dp[k, n])
        if final_cost < best_cost:
            # reconstruct
            cuts: List[Tuple[int, int]] = []
            c = k
            i = n
            while c > 0 and i > 0:
                j = int(prev[c, i])
                if j < 0:
                    break
                cuts.append((j, i - 1))
                i = j
                c -= 1
            cuts.reverse()

            if cuts and len(cuts) == k and cuts[0][0] == 0 and cuts[-1][1] == n - 1:
                best_cost = final_cost
                best_cuts = cuts

    # Fallback: greedy packing.
    # Reached when no partition scores finite — i.e. at least one atom is longer
    # than the ceiling once heading_chars is added, so _score_chunk_len_vec returns
    # inf for every candidate group containing it and `final_cost < best_cost` never
    # fires. Emitting [(0, n-1)] here (the previous behaviour) collapsed the ENTIRE
    # heading group into one chunk, turning a block that overshot by a few chars
    # into a chunk many times over max. Greedy still cuts at the ceiling, so the
    # damage stays confined to the offending block.
    if best_cuts is None:
        return _partition_blocks_greedy(block_lens, heading_chars, max_chunk_chars, optimal_chunk_chars)
    return best_cuts


# ------------------------------
# Assign Chunk Indices
# ------------------------------

def _assign_chunk_indices(
    blocks_df: pd.DataFrame,
    cfg: ChunkSizeConfig,
) -> pd.DataFrame:
    """
    Assign chunk_index to each block based on chunk partitioning strategy.
    
    This is the pure decision logic that determines which blocks belong together.
    chunk_index is a global sequential counter (1, 2, 3...) across the entire document.
    
    Strategy:
      1. Group by active_heading_id
      2. Within each group:
         - Identify heading block(s) and content blocks
         - If total chars (heading + content) <= max_chunk_chars => all blocks get same chunk_index
         - If total > max_chunk_chars => use DP to partition content blocks optimally
         - Each partition gets a unique chunk_index
         - Heading blocks get the chunk_index of the first content partition
      3. chunk_index counts up globally across all active_heading_id groups
    
    Args:
        blocks_df: Blocks dataframe with active_heading_id
        max_chunk_chars: Maximum chunk size in characters
        optimal_chunk_chars: Target chunk size in characters
        softmin_chunk_chars: Soft minimum chunk size (discouraged but allowed)
        min_chunk_chars: Hard minimum chunk size (strongly discouraged)
        
    Returns:
        Same df with "chunk_index" column added
    
    Adds:
      - chunk_index (global, 1..N)
      - needs_block_split (flag on original rows that exceeded max before explosion)
      - sub_block_index (0 for normal rows, 1..N for virtual sub-blocks)
      - is_virtual_sub_block (bool)

    Important:
      - If an oversize block is exploded into virtual rows, chunk_index is assigned to those virtual rows.
      - You can later aggregate chunks using (active_heading_id, chunk_index) and ignore sub-block metadata if desired.
    """
    if blocks_df is None or blocks_df.empty:
        out = blocks_df.copy()
        out["chunk_index"] = 0
        return out

    max_chunk_chars = cfg.max_chunk_chars

    df0 = blocks_df.copy()

    # Ensure embed_char_count exists
    if "embed_char_count" not in df0.columns:
        df0["embed_char_count"] = df0["text"].astype("string").fillna("").str.len().astype(int)

    # Marked per heading group below, against that group's content budget
    df0["needs_block_split"] = False

    # Stable sort
    sort_cols = [c for c in ["page_number", "block_id"] if c in df0.columns]
    if sort_cols:
        df0 = df0.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df0 = df0.reset_index(drop=True)

    # We'll build an expanded working df for decisioning,
    # then merge chunk_index back to original df0 if you don't want to keep virtual rows.
    expanded_rows: List[pd.DataFrame] = []

    for active_hid, g in df0.groupby("active_heading_id", sort=False):
        g2 = g.copy().reset_index(drop=True)

        # Content must fit under max MINUS the heading, since every chunk of this
        # group is prefixed with the heading text.
        heading_mask = _heading_block_mask(g2)
        budget = _content_budget_chars(_group_heading_chars(g2, heading_mask), cfg)

        oversize = (g2["embed_char_count"].astype(int) > budget) & ~heading_mask
        g2["needs_block_split"] = oversize

        # If any oversize in this heading group, explode them into virtual rows (DP-friendly atoms)
        if bool(oversize.any()):
            g2 = _explode_oversize_blocks_within_group(g2, cfg, budget_chars=budget)
        else:
            # ensure expected columns exist
            if "sub_block_index" not in g2.columns:
                g2["sub_block_index"] = 0
            if "is_virtual_sub_block" not in g2.columns:
                g2["is_virtual_sub_block"] = False

        expanded_rows.append(g2)

    df = pd.concat(expanded_rows, ignore_index=True)

    # Initialize chunk_index on expanded df
    df["chunk_index"] = 0

    # Global counter
    global_chunk_idx = 1

    # Now do DP per heading group on expanded rows
    for active_hid, g_indices in df.groupby("active_heading_id", sort=False).groups.items():
        g = df.loc[g_indices].copy()

        heading_mask = _heading_block_mask(g)

        heading_indices = g.index[heading_mask].tolist()
        content_indices = g.index[~heading_mask].tolist()

        heading_chars = _group_heading_chars(g, heading_mask)

        if not content_indices:
            df.loc[g.index, "chunk_index"] = global_chunk_idx
            global_chunk_idx += 1
            continue

        block_lens = [int(x) for x in df.loc[content_indices, "embed_char_count"].tolist()]
        total_len = heading_chars + int(np.sum(block_lens))

        if total_len <= max_chunk_chars:
            df.loc[g.index, "chunk_index"] = global_chunk_idx
            global_chunk_idx += 1
            continue

        _t0_dp = time.perf_counter()
        cuts = _partition_blocks_dp(
            block_lens,
            heading_chars=heading_chars,
            cfg=cfg,
        )
        # Logged, not printed: stdout belongs to the caller. Under the MCP
        # stdio transport it is the JSON-RPC channel itself.
        _dp_elapsed = time.perf_counter() - _t0_dp
        if _dp_elapsed > 0.5:
            _log.warning(
                "slow chunk partition: active_heading_id=%r n=%d total_chars=%d cuts=%d time=%.2fs",
                active_hid, len(block_lens), total_len, len(cuts), _dp_elapsed,
            )

        for local_idx, (a, b) in enumerate(cuts, start=1):
            segment_indices = content_indices[a : b + 1]
            df.loc[segment_indices, "chunk_index"] = global_chunk_idx

            if local_idx == 1 and heading_indices:
                df.loc[heading_indices, "chunk_index"] = global_chunk_idx

            global_chunk_idx += 1

    return df

# ------------------------------
# Join Chunk Text
# ------------------------------

def _join_chunk_text(blocks_with_chunk_index: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build text for each chunk by joining block texts.
    
    Includes headings inside the chunk text with ## prefix for non-merged chunks.
    Also extracts chunk_heading metadata from heading blocks based on active_heading_id.
    
    Args:
        blocks_with_chunk_index: Blocks df with chunk_index column
        chunks_df: Aggregated chunks df (from registry_aggregator)
        
    Returns:
        chunks_df with "text", "chunk_heading", and "contains_table" columns added
    """
    if blocks_with_chunk_index.empty or chunks_df.empty:
        chunks_df["text"] = ""
        chunks_df["chunk_heading"] = ""
        chunks_df["contains_table"] = False
        chunks_df["embed_char_count"] = 0
        return chunks_df
    
    # First, extract heading metadata per active_heading_id
    # All chunks from the same active_heading_id should have the same heading metadata
    heading_cols_to_map = [
        "heading_type",
        "heading_id", 
        "heading_fp_id",
        "heading_fingerprint",
        "heading_hash",
        "heading_level",
        "parent_heading_id",
    ]
    
    heading_metadata = {}
    if "active_heading_id" in blocks_with_chunk_index.columns:
        for active_hid, g in blocks_with_chunk_index.groupby("active_heading_id", sort=False):
            # Treat heading, toc_heading, and exhibit_heading as heading blocks
            block_types = g.get("block_type", pd.Series([""] * len(g))).astype("string").str.strip().str.lower()
            heading_mask = block_types.isin(_HEADING_BLOCK_TYPES)
            
            metadata = {"chunk_heading": ""}
            
            if heading_mask.any():
                heading_row = g.loc[heading_mask].iloc[0]
                # For hybrid_heading_paragraph, use hybrid_heading_text (not the full paragraph text)
                if str(heading_row.get("block_type", "")).strip().lower() == "hybrid_heading_paragraph":
                    heading_text = heading_row.get("hybrid_heading_text", "") or heading_row.get("text", "")
                else:
                    heading_text = heading_row.get("text", "")
                metadata["chunk_heading"] = str(heading_text) if pd.notna(heading_text) else ""
                
                # Extract all heading metadata columns
                for col in heading_cols_to_map:
                    if col in heading_row.index:
                        val = heading_row[col]
                        metadata[col] = val if pd.notna(val) else None
                    else:
                        metadata[col] = None
            else:
                # No heading block - set all to None/empty
                for col in heading_cols_to_map:
                    metadata[col] = None
            
            heading_metadata[active_hid] = metadata
    
    # Build chunk text per chunk_index, including heading with ## prefix
    # We need to pass active_heading_id to look up the heading for multi-chunk sections
    # ------------------------------------------------------------------
    # Build chunk text — vectorized, no per-chunk groupby.apply.
    #
    # Each chunk's text is:  "## {heading}\n\n{content}"  where
    #   * content is the "\n\n"-join of the chunk's non-heading block texts
    #     (via group_join, the shared text primitive), blanks skipped;
    #   * heading is the chunk's heading block text, or — for continuation
    #     chunks with no heading block — the heading looked up from
    #     heading_metadata via active_heading_id;
    #   * a hybrid_heading_paragraph contributes its paragraph body as the
    #     first content part.
    # Empty heading / content collapse exactly as the scalar form did.
    # ------------------------------------------------------------------
    blocks = blocks_with_chunk_index
    all_chunks = pd.Index(blocks["chunk_index"].unique(), name="chunk_index")

    block_types_all = (
        blocks.get("block_type", pd.Series([""] * len(blocks), index=blocks.index))
        .astype("string").str.strip().str.lower()
    )
    is_heading = block_types_all.isin(_HEADING_BLOCK_TYPES).to_numpy()

    # --- content: join non-heading block texts per chunk, skipping blanks ---
    content_txt = blocks["text"].astype("string").fillna("")
    keep = (~is_heading) & content_txt.str.strip().ne("").fillna(False).to_numpy()
    content_series = group_join(
        content_txt[keep], blocks["chunk_index"][keep], sep="\n\n"
    ).reindex(all_chunks, fill_value="")

    # --- heading text + hybrid body from the first heading block per chunk ---
    # (heading_rows is ~one row per heading section, far smaller than all blocks)
    heading_text_by_chunk = pd.Series("", index=all_chunks, dtype=object)
    hybrid_body_by_chunk = pd.Series("", index=all_chunks, dtype=object)
    heading_rows = blocks.loc[is_heading].drop_duplicates(subset="chunk_index", keep="first")
    has_heading_block = pd.Series(False, index=all_chunks)
    if not heading_rows.empty:
        has_heading_block.loc[heading_rows["chunk_index"].to_numpy()] = True
        bt_arr = heading_rows["block_type"].astype("string").str.strip().str.lower().fillna("").to_numpy()
        txt_arr = heading_rows["text"].astype("string").fillna("").to_numpy()
        hyb_arr = (
            heading_rows["hybrid_heading_text"].astype("string").fillna("").to_numpy()
            if "hybrid_heading_text" in heading_rows.columns
            else np.full(len(heading_rows), "", dtype=object)
        )
        h_text, h_body = [], []
        for bt, txt, hyb in zip(bt_arr, txt_arr, hyb_arr):
            if bt == "hybrid_heading_paragraph":
                raw = hyb if hyb else txt          # hybrid_heading_text, else the paragraph text
                prefix = raw.strip()
                if prefix and txt.startswith(prefix):
                    body = txt[len(prefix):].strip()
                elif txt and not prefix:
                    body = txt
                else:
                    body = ""
            else:
                raw, body = txt, ""
            h_text.append(raw)
            h_body.append(body)
        ci = heading_rows["chunk_index"].to_numpy()
        heading_text_by_chunk.loc[ci] = h_text
        hybrid_body_by_chunk.loc[ci] = h_body

    # --- continuation chunks (no heading block): heading from metadata ---
    active_by_chunk = (
        blocks.drop_duplicates("chunk_index").set_index("chunk_index")["active_heading_id"]
        .reindex(all_chunks)
        if "active_heading_id" in blocks.columns
        else pd.Series("", index=all_chunks)
    )
    meta_heading = active_by_chunk.map(
        lambda a: heading_metadata.get(a, {}).get("chunk_heading", "") if pd.notna(a) else ""
    ).fillna("")
    heading_final = heading_text_by_chunk.where(has_heading_block, meta_heading)

    # --- assemble: prepend hybrid body to content, then the "## heading" ---
    mid_sep = np.where(
        (hybrid_body_by_chunk != "").to_numpy() & (content_series != "").to_numpy(), "\n\n", ""
    )
    content_with_body = hybrid_body_by_chunk + pd.Series(mid_sep, index=all_chunks) + content_series

    has_h = (heading_final != "").to_numpy()
    has_c = (content_with_body != "").to_numpy()
    final_text = np.where(
        has_h & has_c,
        ("## " + heading_final + "\n\n" + content_with_body).to_numpy(),
        np.where(has_h, ("## " + heading_final).to_numpy(), content_with_body.to_numpy()),
    )
    text_df = pd.DataFrame({"chunk_index": all_chunks.to_numpy(), "text": final_text})

    # --- contains_table: any table block per chunk (vectorized) ---
    if "block_type" in blocks.columns:
        is_table = block_types_all.eq("table")
        ct = is_table.groupby(blocks["chunk_index"], sort=False, observed=True).any()
        contains_table_df = (
            ct.reindex(all_chunks, fill_value=False).rename("contains_table").reset_index()
        )
    else:
        contains_table_df = pd.DataFrame(
            {"chunk_index": all_chunks.to_numpy(), "contains_table": False}
        )

    # Merge text and contains_table into chunks_df
    chunks_df = chunks_df.merge(text_df, on="chunk_index", how="left")
    chunks_df = chunks_df.merge(contains_table_df, on="chunk_index", how="left")
    chunks_df["text"] = chunks_df["text"].fillna("")
    chunks_df["contains_table"] = chunks_df["contains_table"].fillna(False)
    
    # Add chunk_heading and all heading metadata based on active_heading_id
    # (repeats for all chunks from same heading)
    if "active_heading_id" in chunks_df.columns:
        # Map chunk_heading
        chunks_df["chunk_heading"] = chunks_df["active_heading_id"].map(
            lambda aid: heading_metadata.get(aid, {}).get("chunk_heading", "")
        )
        
        # Map all heading metadata columns
        for col in heading_cols_to_map:
            chunks_df[col] = chunks_df["active_heading_id"].map(
                lambda aid: heading_metadata.get(aid, {}).get(col, None)
            )
    else:
        chunks_df["chunk_heading"] = ""
        for col in heading_cols_to_map:
            chunks_df[col] = None
    
    # Calculate embed_char_count = length of the text field
    chunks_df["embed_char_count"] = chunks_df["text"].astype("string").str.len()
    
    return chunks_df

# =======================================================================================================================
# STEP 3: Merge Chunk Candidates (where necessary)
# =======================================================================================================================

# ------------------------------
# Small Chunk Merge Decision Engine
# ------------------------------

@dataclass(frozen=True)
class MergePlanConfig:
    max_chunk_chars: int
    softmin_chunk_chars: int
    min_children_to_merge: int = 2  # require at least 2 tiny siblings to merge
    # If you want to allow merging a single tiny child with its neighbor, set to 1.


def _norm_id(v: Any) -> str:
    """
    Normalize heading ids so that:
      23, 23.0, "23", "23.0", np.int64(23) -> "23"
    Keeps non-numeric ids (UUIDs etc.) as stripped strings.
    """
    if v is None:
        return ""
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        if np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return str(v).strip()

    s = str(v).strip()
    if not s:
        return ""

    # handle "23.0" style strings
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def assign_merged_chunk_id(chunks_df: pd.DataFrame, cfg: MergePlanConfig) -> pd.DataFrame:
    """
    Pure decision engine.

    Path A (parent_plus_children):
      Merge parent chunk row (heading_id == parent_id) with ALL its children if:
        - parent chunk exists
        - all children are tiny (< softmin)
        - all children are leaf headings (do NOT absorb subtrees)
        - parent + children total <= max

    Path B (children_tail):
      Merge only consecutive tiny LEAF siblings under the same parent_heading_id,
      contiguity enforced by chunk_index adjacency (no jumping).

    Output cols:
      - merged_chunk_id (int): pd.NA = no merge
      - merge_mode (str): "", "parent_plus_children", "children_tail"
      - merge_group_parent_heading_id (str)
      - merge_member_heading_ids (list[str])  # debug
    """
    if chunks_df is None or chunks_df.empty:
        out = (chunks_df.copy() if chunks_df is not None else pd.DataFrame())
        out["merged_chunk_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["merge_mode"] = ""
        out["merge_group_parent_heading_id"] = ""
        out["merge_member_heading_ids"] = [[] for _ in range(len(out))]
        return out

    df = chunks_df.copy()

    required = ["embed_char_count", "heading_id", "parent_heading_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"assign_merged_chunk_id missing required columns: {missing}")

    # stable doc order
    sort_cols = [c for c in ["page_number", "chunk_index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # normalize ids into helper columns
    df["_hid_norm"] = df["heading_id"].map(_norm_id)
    df["_pid_norm"] = df["parent_heading_id"].map(_norm_id)

    # init outputs
    df["merged_chunk_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["merge_mode"] = ""
    df["merge_group_parent_heading_id"] = ""
    df["merge_member_heading_ids"] = [[] for _ in range(len(df))]

    # detect parents globally (normed)
    parent_id_set: Set[str] = set([p for p in df["_pid_norm"].tolist() if p])
    df["_is_parent_heading"] = df["_hid_norm"].isin(parent_id_set)
    df["_is_leaf_heading"] = ~df["_is_parent_heading"]

    # tiny + leaf eligibility
    df["_is_tiny"] = df["embed_char_count"].fillna(0).astype(int) < int(cfg.softmin_chunk_chars)
    df["_is_tiny_leaf"] = df["_is_tiny"] & df["_is_leaf_heading"]

    # map norm heading_id -> row index (first occurrence)
    hid_to_idx: Dict[str, int] = {}
    for i, hid in enumerate(df["_hid_norm"].tolist()):
        if hid and hid not in hid_to_idx:
            hid_to_idx[hid] = i

    # parent_norm -> children indices
    by_parent: Dict[str, List[int]] = {}
    for i, pid in enumerate(df["_pid_norm"].tolist()):
        if not pid:
            continue
        by_parent.setdefault(pid, []).append(i)

    next_merge_id = 1

    def _assign_group(member_idxs: List[int], mode: str, parent_norm: str) -> None:
        nonlocal next_merge_id
        if not member_idxs:
            return
        if not df.loc[member_idxs, "merged_chunk_id"].isna().all():
            return

        member_heading_ids = df.loc[member_idxs, "_hid_norm"].tolist()

        df.loc[member_idxs, "merged_chunk_id"] = next_merge_id
        df.loc[member_idxs, "merge_mode"] = mode
        df.loc[member_idxs, "merge_group_parent_heading_id"] = parent_norm
        for m in member_idxs:
            df.at[m, "merge_member_heading_ids"] = member_heading_ids

        next_merge_id += 1

    def _try_assign_children_run(run_idxs: List[int], parent_norm: str) -> None:
        if len(run_idxs) < int(cfg.min_children_to_merge):
            return
        if not df.loc[run_idxs, "merged_chunk_id"].isna().all():
            return
        run_sum = int(df.loc[run_idxs, "embed_char_count"].sum())
        if run_sum > int(cfg.max_chunk_chars):
            return
        _assign_group(run_idxs, mode="children_tail", parent_norm=parent_norm)

    for parent_norm, child_idxs in by_parent.items():
        # -------------------------
        # Path A: parent + ALL children (safe only if children are tiny AND leaf)
        # -------------------------
        parent_row_idx = hid_to_idx.get(parent_norm)

        if parent_row_idx is not None and pd.isna(df.at[parent_row_idx, "merged_chunk_id"]):
            all_children_tiny = bool(df.loc[child_idxs, "_is_tiny"].all())
            all_children_leaf = bool(df.loc[child_idxs, "_is_leaf_heading"].all())
            total_chars = int(df.at[parent_row_idx, "embed_char_count"]) + int(df.loc[child_idxs, "embed_char_count"].sum())

            if (
                all_children_tiny
                and all_children_leaf
                and total_chars <= int(cfg.max_chunk_chars)
                and df.loc[child_idxs, "merged_chunk_id"].isna().all()
            ):
                _assign_group([parent_row_idx] + child_idxs, mode="parent_plus_children", parent_norm=parent_norm)
                continue  # don't do tail merges inside this parent

        # -------------------------
        # Path B: consecutive tiny LEAF children runs, contiguous by chunk_index (no jumping)
        # -------------------------
        run: List[int] = []

        for idx in child_idxs:
            if not bool(df.at[idx, "_is_tiny_leaf"]):
                if run:
                    _try_assign_children_run(run, parent_norm)
                    run = []
                continue

            if not run:
                run = [idx]
                continue

            prev_ci = int(df.at[run[-1], "chunk_index"]) if "chunk_index" in df.columns else run[-1]
            curr_ci = int(df.at[idx, "chunk_index"]) if "chunk_index" in df.columns else idx

            if curr_ci == prev_ci + 1:
                run.append(idx)
            else:
                _try_assign_children_run(run, parent_norm)
                run = [idx]

        if run:
            _try_assign_children_run(run, parent_norm)

    # drop helpers
    df = df.drop(
        columns=[
            "_hid_norm",
            "_pid_norm",
            "_is_parent_heading",
            "_is_leaf_heading",
            "_is_tiny",
            "_is_tiny_leaf",
        ],
        errors="ignore",
    )

    return df


# ------------------------------
# Rebuild Merged Chunks
# ------------------------------

def _aggregate_merged_fields_by_id(merged_chunks: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the non-text fields of every merged group in ONE registry-driven
    pass, keyed by merged_chunk_id — bbox, counts, styles, flags, etc.

    Runs the aggregation machinery once for all merged groups instead of once per
    group (the previous per-group call was a bottleneck). The caller rebuilds
    text / chunk_heading per merge mode and overwrites those afterwards.

    Local overrides (everything else resolves via COLUMN_REGISTRY, where
    chunk_index/heading_id/parent_heading_id/heading_level/heading_type are already
    "first"):
      - table_id/chart_id collect the (already list-valued) child ids, flattened
        and de-duplicated across the merged chunks;
      - contains_table is True if ANY merged chunk holds a table (the registry has
        no entry for this chunk-level column, so it would otherwise default to
        first).

    Returns a frame indexed by merged_chunk_id (one row per merged group).
    """
    return aggregate_to(
        merged_chunks,
        by="merged_chunk_id",
        overrides={
            "table_id": Agg.UNIQUE_LIST,
            "chart_id": Agg.UNIQUE_LIST,
            "contains_table": Agg.ANY,
        },
        # merge bookkeeping (merge_mode / merge_group_parent_heading_id /
        # merge_member_heading_ids) and chunk_heading are local to this stage, not
        # part of the shared column registry; "first" is fine (chunk_heading is
        # overwritten by the caller). Silenced so they don't nag to be registered.
        on_unknown="silent",
        derived=True,
    ).set_index("merged_chunk_id")


def _rebuild_merged_chunks(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild the complete dataframe based on merging of chunks.
    
    For merged chunks (where merged_chunk_id is not NA):
    - merge_mode = parent_plus_children:
      - chunk_heading becomes the parent followed by the absorbed children's
        headings, pipe delimited (parent first)
      - text includes the children's texts (which already have ## headings inside)
    - merge_mode = children_tail:
      - chunk_heading becomes the children's headings pipe delimited
      - text includes the children's texts (which already have ## headings inside)
    
    Args:
        chunks_df: DataFrame with merged_chunk_id, merge_mode, chunk_heading, text columns
        
    Returns:
        Rebuilt DataFrame with merged chunks consolidated
    """
    if chunks_df is None or chunks_df.empty:
        return chunks_df
    
    df = chunks_df.copy()
    
    # Check if there are any merged chunks
    if "merged_chunk_id" not in df.columns or df["merged_chunk_id"].isna().all():
        return df
    
    # Separate merged and non-merged chunks
    merged_mask = df["merged_chunk_id"].notna()
    merged_chunks = df[merged_mask].copy()
    non_merged_chunks = df[~merged_mask].copy()
    
    if merged_chunks.empty:
        return df

    # Aggregate every merged group's fields in a single pass, keyed by
    # merged_chunk_id (bbox/counts/styles/flags). The per-group loop below only
    # rebuilds text + chunk_heading and pulls the pre-aggregated row by id.
    agg_by_id = _aggregate_merged_fields_by_id(merged_chunks)

    # Group merged chunks by merged_chunk_id
    merged_groups = []

    for merge_id, group in merged_chunks.groupby("merged_chunk_id", sort=False):
        # Get the merge mode (should be same for all in group)
        merge_mode = group["merge_mode"].iloc[0] if "merge_mode" in group.columns else ""
        parent_heading_id = group["merge_group_parent_heading_id"].iloc[0] if "merge_group_parent_heading_id" in group.columns else ""
        
        # Sort by chunk_index to maintain order
        if "chunk_index" in group.columns:
            group = group.sort_values("chunk_index", kind="mergesort")
        
        if merge_mode == "parent_plus_children":
            # Find parent chunk (heading_id matches parent_heading_id)
            # Use the same normalization as assign_merged_chunk_id
            group_heading_ids_norm = group["heading_id"].map(_norm_id)
            parent_heading_id_norm = _norm_id(parent_heading_id)
            
            parent_mask = group_heading_ids_norm == parent_heading_id_norm
            parent_chunk = group[parent_mask]
            children_chunks = group[~parent_mask]
            
            if not parent_chunk.empty:
                # Use parent's chunk_heading
                new_chunk_heading = parent_chunk["chunk_heading"].iloc[0] if "chunk_heading" in parent_chunk.columns else ""
                
                # Build text: Start with parent heading, then add parent content (if any), then children
                text_parts = []
                
                # Get parent's full text (should be "## Parent Title\n\nContent" or just "## Parent Title")
                parent_text = str(parent_chunk["text"].iloc[0]) if "text" in parent_chunk.columns else ""
                parent_text = parent_text.strip()
                
                # If parent text exists, use it as the base
                if parent_text:
                    text_parts.append(parent_text)
                elif new_chunk_heading:
                    # If parent text is empty but we have a heading, add it
                    text_parts.append(f"## {new_chunk_heading}")
                
                # Then add ALL children's texts (already have ## headings inside)
                # Sort children by chunk_index to maintain order
                if "chunk_index" in children_chunks.columns and not children_chunks.empty:
                    children_chunks = children_chunks.sort_values("chunk_index", kind="mergesort")

                # The absorbed children's headings must survive into chunk_heading,
                # exactly as in children_tail. The merged row keeps the parent's
                # heading_id (registry "first"), so a child folded in here is named
                # nowhere else: its own heading node ends up with no chunk_ids, and
                # consumers that split chunk_heading on " | " to attribute a merged
                # chunk across its headings would report zero content beneath it.
                child_headings: List[str] = []
                for _, child_row in children_chunks.iterrows():
                    child_heading = str(child_row.get("chunk_heading", "") or "").strip()
                    if child_heading:
                        child_headings.append(child_heading)

                    child_text = str(child_row.get("text", "") or "").strip()
                    if child_text:
                        text_parts.append(child_text)

                if child_headings:
                    parent_heading_text = str(new_chunk_heading or "").strip()
                    new_chunk_heading = " | ".join(
                        ([parent_heading_text] if parent_heading_text else []) + child_headings
                    )

                new_text = "\n\n".join(text_parts)
                
                # Aggregate all fields across the merged chunks (bbox, counts, styles, etc.)
                merged_row = agg_by_id.loc[merge_id].copy()
                
                # Override with merge-specific values
                merged_row["chunk_heading"] = new_chunk_heading
                merged_row["text"] = new_text
                merged_row["embed_char_count"] = len(new_text)
                
                merged_groups.append(merged_row)
            else:
                # Fallback: if no parent found, aggregate all chunks
                merged_row = agg_by_id.loc[merge_id].copy()
                merged_groups.append(merged_row)
        
        elif merge_mode == "children_tail":
            # chunk_heading = children's headings pipe delimited
            headings = []
            text_parts = []
            
            # Sort by chunk_index to maintain order
            if "chunk_index" in group.columns:
                group = group.sort_values("chunk_index", kind="mergesort")
            
            for _, child_row in group.iterrows():
                child_heading = str(child_row.get("chunk_heading", "") or "").strip()
                child_text = str(child_row.get("text", "") or "").strip()
                
                if child_heading:
                    headings.append(child_heading)
                
                # Add text (already has ## heading inside)
                if child_text:
                    text_parts.append(child_text)
            
            new_chunk_heading = " | ".join(headings)
            new_text = "\n\n".join(text_parts)
            
            # Aggregate all fields across the merged chunks (bbox, counts, styles, etc.)
            merged_row = agg_by_id.loc[merge_id].copy()
            
            # Override with merge-specific values
            merged_row["chunk_heading"] = new_chunk_heading
            merged_row["text"] = new_text
            merged_row["embed_char_count"] = len(new_text)
            
            merged_groups.append(merged_row)
        
        else:
            # Unknown merge mode, aggregate all chunks
            merged_row = agg_by_id.loc[merge_id].copy()
            merged_groups.append(merged_row)
    
    # Combine merged groups back into dataframe
    if merged_groups:
        merged_df = pd.DataFrame(merged_groups)
        
        # Combine with non-merged chunks
        result_df = pd.concat([non_merged_chunks, merged_df], ignore_index=True)
        
        # Sort by original order (page_number, chunk_index)
        sort_cols = [c for c in ["page_number", "chunk_index"] if c in result_df.columns]
        if sort_cols:
            result_df = result_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            result_df = result_df.reset_index(drop=True)
        
        return result_df
    
    return df


# =======================================================================================================================
# Post-condition: enforce the hard character ceiling
# =======================================================================================================================

def _enforce_max_chunk_chars(chunks_df: pd.DataFrame, cfg: ChunkSizeConfig) -> pd.DataFrame:
    """
    Last line of defence: no chunk may leave this module longer than max_chunk_chars.

    The partitioner and the small-chunk merger both aim to respect the ceiling, but
    neither re-validates its own output, and an embedder rejects an over-length input
    outright — so the invariant is enforced here rather than assumed. Any oversize
    chunk is split in place into as many rows as needed; metadata is copied to each
    part and chunk_index is left duplicated on purpose, since
    _add_chunk_ids_and_reindex renumbers sequentially afterwards (the sort ahead of it
    is stable, so part order holds).

    Runs after merging so it also covers chunks that merging made too large.
    """
    if chunks_df is None or chunks_df.empty or "text" not in chunks_df.columns:
        return chunks_df

    max_chunk_chars = int(cfg.max_chunk_chars)

    lengths = chunks_df["text"].astype("string").fillna("").str.len()
    oversize = lengths > max_chunk_chars
    if not bool(oversize.any()):
        return chunks_df

    # Never print: stdout is the JSON-RPC wire when we run under the MCP stdio server.
    _log.warning(
        "%d chunk(s) exceeded max_chunk_chars=%d (largest=%d); splitting to enforce the ceiling.",
        int(oversize.sum()), max_chunk_chars, int(lengths.max()),
    )

    rows_out: List[pd.Series] = []
    for pos, (_, row) in enumerate(chunks_df.iterrows()):
        if not bool(oversize.iloc[pos]):
            rows_out.append(row)
            continue

        parts = _split_text_into_parts(
            text=str(row.get("text", "") or ""),
            max_chars=max_chunk_chars,
            target_chars=cfg.optimal_chunk_chars,
            softmin_chars=cfg.softmin_chunk_chars,
        )
        for part in parts:
            r2 = row.copy()
            r2["text"] = part
            r2["embed_char_count"] = int(len(part))
            rows_out.append(r2)

    return pd.DataFrame(rows_out).reset_index(drop=True)


# =======================================================================================================================
# STEP 4: Add Token Count & Chunk IDs
# =======================================================================================================================

@functools.lru_cache(maxsize=1)
def token_encoder():
    """The cl100k_base encoder, or None if exact counting is unavailable here.

    Two ways this comes back None: tiktoken is not installed, or it is but the
    first ``get_encoding`` cannot fetch its BPE vocabulary (offline, sandboxed,
    no cache). Both are survivable — callers fall back to estimation — so the
    result is cached to keep one failed network attempt from repeating per
    document, and to keep every caller's answer consistent within a process.
    """
    try:
        import tiktoken as _tiktoken
        return _tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        _log.info("Exact token counting unavailable (%s) — estimating as chars/4.", e)
        return None


def _add_token_count(chunks_df: pd.DataFrame, exact_tokens: bool = False) -> pd.DataFrame:
    """
    Add token_count to each chunk.

    If exact_tokens=True, uses tiktoken (cl100k_base) and falls back to
    char-count estimation when it is unavailable.
    If exact_tokens=False, always uses embed_char_count // 4.
    """
    if chunks_df is None or chunks_df.empty:
        chunks_df = chunks_df.copy() if chunks_df is not None else pd.DataFrame()
        chunks_df["token_count"] = 0
        return chunks_df

    df = chunks_df.copy()

    encoding = token_encoder() if exact_tokens else None

    if encoding is not None:
        df["token_count"] = df["text"].apply(
            lambda t: len(encoding.encode(str(t) if pd.notna(t) else ""))
        )
    else:
        char_counts = df["embed_char_count"] if "embed_char_count" in df.columns else df["text"].str.len().fillna(0)
        df["token_count"] = (char_counts.fillna(0).astype(int) // 4).astype(int)

    return df


def _add_chunk_ids_and_reindex(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chunk_id (UUID v4), parent_chunk_id (UUID v4), and reindex chunk_index to remove gaps.
    
    Logic:
    - Generate UUID v4 for each chunk as chunk_id
    - For each chunk, find parent chunk where heading_id == parent_heading_id
    - Set parent_chunk_id to parent chunk's chunk_id (or None if no parent)
    - Reindex chunk_index to be sequential (1, 2, 3, ...) with no gaps
    
    Args:
        chunks_df: DataFrame with heading_id, parent_heading_id, and chunk_index columns
        
    Returns:
        DataFrame with chunk_id, parent_chunk_id added and chunk_index reindexed
    """
    if chunks_df is None or chunks_df.empty:
        chunks_df = chunks_df.copy() if chunks_df is not None else pd.DataFrame()
        chunks_df["chunk_id"] = ""
        chunks_df["parent_chunk_id"] = None
        return chunks_df
    
    df = chunks_df.copy()
    
    # Sort by chunk_index first to maintain order (before generating IDs)
    if "chunk_index" in df.columns:
        sort_cols = [c for c in ["page_number", "chunk_index"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    
    # Generate UUID v4 for each chunk
    df["chunk_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Initialize parent_chunk_id column
    df["parent_chunk_id"] = None
    
    # Build parent_chunk_id lookup if heading columns exist
    if "heading_id" in df.columns and "parent_heading_id" in df.columns:
        # Normalize heading_id and parent_heading_id for matching (vectorized)
        # Handles int, float, and string IDs - converts all to normalized strings
        # e.g., 1 -> "1", 1.0 -> "1", "h-1" -> "h-1", None -> ""
        def _normalize_id_series(series: pd.Series) -> pd.Series:
            """Normalize heading IDs to strings for consistent matching."""
            # Convert to string
            normalized = series.astype(str)
            # Replace string representations of None/NaN with empty string
            normalized = normalized.replace(["None", "nan", "NaN", "<NA>"], "")
            # Remove trailing .0 from float-like strings (e.g., "1.0" -> "1")
            normalized = normalized.str.replace(r"\.0+$", "", regex=True)
            # Strip whitespace
            normalized = normalized.str.strip()
            return normalized
        
        df["_heading_id_norm"] = _normalize_id_series(df["heading_id"])
        df["_parent_heading_id_norm"] = _normalize_id_series(df["parent_heading_id"])
        
        # Create a lookup: heading_id -> chunk_id (vectorized)
        # For each unique heading_id, map it to the chunk_id of the FIRST chunk with that heading_id
        # Filter out blank heading_ids, then drop duplicates keeping first occurrence
        valid_headings = df["_heading_id_norm"] != ""
        lookup_df = df.loc[valid_headings, ["_heading_id_norm", "chunk_id"]].drop_duplicates(
            subset=["_heading_id_norm"], keep="first"
        )
        heading_to_chunk_id = dict(zip(lookup_df["_heading_id_norm"], lookup_df["chunk_id"]))
        
        # Set parent_chunk_id using vectorized map (much faster than apply)
        # Map parent_heading_id_norm -> chunk_id, returns NaN for missing keys
        df["parent_chunk_id"] = df["_parent_heading_id_norm"].map(heading_to_chunk_id)
        
        # Convert empty parent_heading_id_norm to None (root level chunks)
        df.loc[df["_parent_heading_id_norm"] == "", "parent_chunk_id"] = None
        
        # Clean up helper columns
        df = df.drop(columns=["_heading_id_norm", "_parent_heading_id_norm"])
    
    # Reindex chunk_index to be sequential (1, 2, 3, ...) with no gaps
    # (df is already sorted from earlier)
    if "chunk_index" in df.columns:
        # Create new sequential chunk_index
        df["chunk_index"] = range(1, len(df) + 1)
    
    return df


def _add_chunk_path(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chunk_path column showing the hierarchical path from root to current chunk.
    
    The path is formatted as markdown headings with increasing levels:
    # Part I
    ## Item 1. Business
    ### Our Company
    
    Algorithm:
    1. For each chunk, look up its parent_heading_id
    2. Find the first row in df where heading_id == parent_heading_id
    3. Collect that chunk's heading and repeat until no parent (root)
    4. Build path by joining all headings from root to current chunk
    
    Args:
        chunks_df: DataFrame with heading_id, parent_heading_id, chunk_heading, heading_level columns
        
    Returns:
        DataFrame with chunk_path column added
    """
    if chunks_df is None or chunks_df.empty:
        chunks_df = chunks_df.copy() if chunks_df is not None else pd.DataFrame()
        chunks_df["chunk_path"] = ""
        return chunks_df
    
    df = chunks_df.copy()
    
    # Check if required columns exist
    required = {"heading_id", "parent_heading_id", "chunk_heading"}
    if not required.issubset(df.columns):
        df["chunk_path"] = ""
        return df
    
    # Normalize heading IDs for consistent lookup
    def _normalize_id(val):
        """Normalize heading ID to string for matching."""
        if pd.isna(val) or val == "" or val == "None":
            return None
        s = str(val).strip()
        # Remove trailing .0 from float-like strings
        s = s.replace(".0", "")
        return s if s else None
    
    # Build lookup: heading_id -> (chunk_heading, parent_heading_id, heading_level)
    # Use first occurrence of each heading_id
    heading_info = {}
    for _, row in df.iterrows():
        heading_id = _normalize_id(row.get("heading_id"))
        if heading_id is None or heading_id in heading_info:
            continue  # Skip if already processed (keep first occurrence)
        
        chunk_heading = row.get("chunk_heading", "")
        parent_heading_id = _normalize_id(row.get("parent_heading_id"))
        heading_level = row.get("heading_level", 1)
        
        # Normalize heading and level
        if pd.isna(chunk_heading):
            chunk_heading = ""
        if pd.isna(heading_level):
            heading_level = 1
        else:
            heading_level = int(heading_level)
        
        heading_info[heading_id] = (chunk_heading, parent_heading_id, heading_level)
    
    # Cache for computed paths: heading_id -> path string
    # This avoids recomputing the same parent paths multiple times
    path_cache = {}
    
    def _get_path_for_heading(heading_id: str) -> str:
        """Get the full path for a heading_id, with caching."""
        # Check cache first
        if heading_id in path_cache:
            return path_cache[heading_id]
        
        # Get heading info
        if heading_id not in heading_info:
            path_cache[heading_id] = ""
            return ""
        
        chunk_heading, parent_heading_id, level = heading_info[heading_id]
        
        # Collect parent headings by walking up the hierarchy
        parent_headings = []
        visited = {heading_id}  # Prevent infinite loops
        current_parent = parent_heading_id
        
        while current_parent is not None:
            # Check for cycles
            if current_parent in visited:
                break
            visited.add(current_parent)
            
            # Look up parent info
            if current_parent not in heading_info:
                break
            
            parent_chunk_heading, next_parent_id, parent_level = heading_info[current_parent]
            
            # Add parent heading to list (if not empty)
            if parent_chunk_heading:
                parent_headings.append((parent_chunk_heading, parent_level))
            
            # Move to next parent
            current_parent = next_parent_id
        
        # Reverse to get path from root to current
        parent_headings.reverse()
        
        # Build markdown path
        path_parts = []
        for heading_text, heading_level in parent_headings:
            formatted = "#" * max(1, heading_level) + " " + heading_text
            path_parts.append(formatted)
        
        # Add current heading if not empty
        if chunk_heading:
            formatted = "#" * max(1, level) + " " + chunk_heading
            path_parts.append(formatted)
        
        path = "\n".join(path_parts)
        path_cache[heading_id] = path
        return path
    
    # Build paths for all chunks using the cached lookup
    def _build_path_for_chunk(row) -> str:
        """Build path by looking up the heading_id in the cache."""
        heading_id = _normalize_id(row.get("heading_id"))
        if heading_id is None:
            return ""
        return _get_path_for_heading(heading_id)
    
    # Build paths for all chunks
    df["chunk_path"] = df.apply(_build_path_for_chunk, axis=1)
    
    return df


# =======================================================================================================================
# Public API
# =======================================================================================================================

def build_chunks(
    blocks_df: pd.DataFrame,
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    optimal_chunk_chars: int = _DEFAULT_OPTIMAL_CHUNK_CHARS,
    softmin_chunk_chars: int = _DEFAULT_SOFTMIN_CHUNK_CHARS,
    min_chunk_chars: int = _DEFAULT_MIN_CHUNK_CHARS,
    merge_small_chunks: bool = True,
    exact_tokens: bool = False,
) -> pd.DataFrame:
    """
    Build semantic chunks from blocks with heading hierarchy.
    
    Process:
      1. Prepare blocks (add active_heading_id, remove noise blocks)
      2. Assign chunk indices (decision logic using DP partitioning)
      3. Aggregate to chunk level (using registry_aggregator)
      4. Join text and extract chunk headings
      5. Sort by document order
      6. Merge chunk candidates (if merge_small_chunks=True)
      7. Add token count
      8. Add chunk IDs and parent_chunk_id
      9. Add chunk_path (hierarchical heading path)
    
    Args:
        blocks_df: Blocks-level DataFrame from block merger
        max_chunk_chars: Maximum chunk size in characters (bounded: 800-8000, default: 3200)
        optimal_chunk_chars: Target chunk size in characters (bounded: 400-4000, default: 1200)
        softmin_chunk_chars: Soft minimum chunk size - discouraged but allowed (bounded: 200-2000, default: 700)
        min_chunk_chars: Hard minimum chunk size - strongly discouraged (bounded: 100-1000, default: 400)
        
    Returns:
        Chunks-level DataFrame with semantic chunks
    """
    if blocks_df is None or blocks_df.empty:
        return pd.DataFrame()
    
    # -------------------------
    # VALIDATE PARAMETERS
    # -------------------------
    # ChunkSizeConfig clamps to bounds and enforces min <= softmin <= optimal <= max
    # on construction, so the clamped values are read back from the instance.
    size_cfg = ChunkSizeConfig(
        max_chunk_chars=max_chunk_chars,
        optimal_chunk_chars=optimal_chunk_chars,
        softmin_chunk_chars=softmin_chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )
    max_chunk_chars = size_cfg.max_chunk_chars
    softmin_chunk_chars = size_cfg.softmin_chunk_chars

    # -------------------------
    # STEP 1: PREPARE
    # -------------------------
    df = _prepare_blocks_df(blocks_df)
    
    if df.empty:
        return pd.DataFrame()
    
    # Ensure required columns exist for aggregation DEPRECATED
    #df = prepare_for_aggregation(
    #    df,
    #    required_cols=["page_number", "block_id", "text", "active_heading_id"],
    #    optional_cols=STANDARD_OPTIONAL_COLUMNS,
    #)
    
    # -------------------------
    # STEP 2: ASSIGN CHUNK INDICES
    # -------------------------
    df = _assign_chunk_indices(df, size_cfg)
    
    # -------------------------
    # STEP 3: AGGREGATE
    # -------------------------
    # Registry-driven: every column's roll-up rule lives in COLUMN_REGISTRY, so
    # the chunk level picks up new block columns automatically. Only the local
    # exceptions are supplied here — table_id/chart_id collect the child ids into
    # a list (a chunk spans many blocks/tables). The decision-layer helper columns
    # from _assign_chunk_indices are dropped; they are internal to chunking and not
    # part of the chunk contract. block_count == group size (every row has a block).
    chunks_df = aggregate_to(
        df,
        by="chunk_index",  # kept as-is (registry "first"); preserved as the group key
        overrides={"table_id": Agg.UNIQUE_LIST, "chart_id": Agg.UNIQUE_LIST},
        drop=["needs_block_split", "sub_block_index", "is_virtual_sub_block"],
        size_as="block_count",
        derived=True,
    )
    
    # -------------------------
    # STEP 4: JOIN TEXT
    # -------------------------
    chunks_df = _join_chunk_text(df, chunks_df)
    
    # -------------------------
    # STEP 5: SORT
    # -------------------------
    sort_cols = [c for c in ["page_number", "chunk_index"] if c in chunks_df.columns]
    if sort_cols:
        chunks_df = chunks_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # -------------------------
    # STEP 6: MERGE CHUNK CANDIDATES
    # -------------------------
    if merge_small_chunks:
        cfg = MergePlanConfig(
            max_chunk_chars=max_chunk_chars,
            softmin_chunk_chars=softmin_chunk_chars,
            min_children_to_merge=2,
        )

        chunks_df = assign_merged_chunk_id(chunks_df, cfg)
        
        # Rebuild the complete df based on the merging of chunks
        chunks_df = _rebuild_merged_chunks(chunks_df)
    
    # -------------------------
    # STEP 6b: ENFORCE HARD CEILING
    # -------------------------
    chunks_df = _enforce_max_chunk_chars(chunks_df, size_cfg)

    # -------------------------
    # STEP 7: ADD TOKEN COUNT
    # -------------------------
    chunks_df = _add_token_count(chunks_df, exact_tokens=exact_tokens)
    
    # -------------------------
    # STEP 8: ADD CHUNK IDS AND REINDEX
    # -------------------------
    chunks_df = _add_chunk_ids_and_reindex(chunks_df)
    
    # -------------------------
    # STEP 9: ADD CHUNK PATH
    # -------------------------
    try:
        chunks_df = _add_chunk_path(chunks_df)
    except RecursionError:
        # If there's a recursion error (e.g., circular parent references),
        # skip adding the chunk_path column and continue without it
        pass
    
    return chunks_df
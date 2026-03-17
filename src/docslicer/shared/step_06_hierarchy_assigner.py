"""
step_01_doc_hierarchy_assigner.py

Assess document hierarchy signals on a per-line basis.

Current version:
- Adds `heading_score` to lines_df (vectorized scoring).
- Public API: `assign_doc_hierarchy(lines_df)` filters out tables and applies scoring.

Notes:
- Designed for HTML and PDF-derived `lines_df` (but gracefully handles missing columns).
- Expects typical columns shown in your example.
"""

from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
import re

# ================================================================================
# Pre-filter forbidden heading line formats (text that will never be a heading)
# ================================================================================

_FORBIDDEN_BLOCK_ROLES = {"table", "image", "hr", "page_label", "navigation", "watermark", "toc", "exhibits"}

_FORBIDDEN_SUBSTRINGS = {
    # signature indicators
    "/s/", "signed:", "page"
}

_FORBIDDEN_START_TEXT = {
    # stars and quotes
    "*", "'",'"', "“", "”", "‘", "’", "„", "«", "»",
    # Bullet tokens
    "-", "–", "—", "•", "·", "…", "■", "▪", "",
    "+", "☒", "☐", "○", "◦", "►", "▸", "‣", "⁃",
    "✓", "✔", "✗", "✘", "✖", "✕",
    # Other
    "©", "®", "™", "§", "¶", "†", "‡", "•", "…", "‹", "›", "“", "”", "‘", "’", "„", "«", "»",
}

# Parenthesized line patterns:
# fully enclosed in (), [], {} (after trimming whitespace)
_PAREN_FULL_RE = re.compile(r"^\s*(\((?s:.*)\)|\[(?s:.*)\]|\{(?s:.*)\})\s*$")

def pre_filter_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefilter lines BEFORE heading_score.

    Excludes:
    (1) rows where block_role is in _FORBIDDEN_BLOCK_ROLES
    (2) rows where text starts with any token in _FORBIDDEN_START_TEXT (case-insensitive, after lstrip)
    (3) rows where text is fully parenthesized: (...), [...], {...}
    (4) rows where text contains any substring in _FORBIDDEN_SUBSTRINGS (case-insensitive)

    Returns a filtered COPY (subset of rows).
    """
    df = lines_df.copy()

    if "text" not in df.columns:
        return df

    text = df["text"].astype("string").fillna("")
    text_lstrip = text.str.lstrip()
    text_lower = text_lstrip.str.lower()

    keep = pd.Series(True, index=df.index, dtype=bool)

    # (1) forbidden block roles
    if "block_role" in df.columns:
        br = df["block_role"].astype("string").str.strip().str.lower()
        keep &= ~br.isin(_FORBIDDEN_BLOCK_ROLES)

    # (2) forbidden start tokens
    # normalize tokens to lowercase for comparison
    forbidden_tokens = sorted({t.lower() for t in _FORBIDDEN_START_TEXT if t is not None})
    if forbidden_tokens:
        # Build one regex that matches ANY forbidden token at start
        # - uses re.escape for safety
        # - matches after left-trim (we already lstrip, so anchor ^ is correct)
        tok_re = re.compile(r"^(?:%s)" % "|".join(re.escape(t) for t in forbidden_tokens))
        keep &= ~text_lower.str.match(tok_re)

    # (3) fully parenthesized lines
    keep &= ~text_lstrip.str.match(_PAREN_FULL_RE)

    # (4) forbidden substrings (anywhere in text)
    forbidden_substrings = sorted({s.lower() for s in _FORBIDDEN_SUBSTRINGS if s is not None})
    if forbidden_substrings:
        # Check if any forbidden substring appears anywhere in the text
        for substring in forbidden_substrings:
            keep &= ~text_lower.str.contains(re.escape(substring), na=False)

    return df.loc[keep].copy()


# ================================================================================
# Core heading scoring function
# ================================================================================

# ----- Helpers ----- #

def _to_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    """
    Robust boolean parsing for mixed types:
    - True/False
    - "TRUE"/"FALSE"
    - 1/0
    - NaN -> default
    """
    if s is None:
        return pd.Series(default, index=pd.Index([]), dtype=bool)

    if s.dtype == bool:
        return s.fillna(default)

    # strings / objects / numbers
    out = s.copy()

    # numeric-like -> bool
    if pd.api.types.is_numeric_dtype(out):
        return out.fillna(int(default)).astype(int).astype(bool)

    # object -> normalize string
    out = out.astype("string")
    norm = out.str.strip().str.lower()

    true_set = {"true", "t", "1", "yes", "y"}
    false_set = {"false", "f", "0", "no", "n"}

    parsed = pd.Series(np.nan, index=out.index, dtype="float64")
    parsed = parsed.mask(norm.isin(true_set), 1.0)
    parsed = parsed.mask(norm.isin(false_set), 0.0)

    # fallback: keep default for unknowns / missing
    parsed = parsed.fillna(1.0 if default else 0.0)
    return parsed.astype(int).astype(bool)


def _to_float_series(s: pd.Series, default: float = np.nan) -> pd.Series:
    if s is None:
        return pd.Series(default, index=pd.Index([]), dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    num = _to_float_series(num, default=np.nan)
    den = _to_float_series(den, default=np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ----- Core heading scoring function ----- #

# TODO: Lower the score of items that are ubiquitous in their layout_id (bold, ratio > 1, but the whole block is like that, then it's not a heading)

def add_heading_score(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes lines_df and returns it with one extra column: `heading_score`.

    Scoring rules (as requested):
    [GENERAL]
    - font_size_ratio:
      - 1.01 to 1.2: +1
      - 1.2 to 1.4: +2
      - >1.4: +3
      - <1: -10
    - char_count
      - < 50: +0.5
      - >100: -1
      - >250: -5
    - capitalized_word_ratio = capitalized_word_count/word_count
      - >0.75: +0.5
    - is_bold = true: +1.5
    - is_italic = true: +1.5
    - is_underlined = true: +1.5
    - text_align = center: +1
    - is_uppercase = true: +1

    [PDF specific]
    - layout_id consists out of only 1 line_id: +1
      (interpreted as: within the same layout_id, the number of rows/lines == 1)
    """
    out = lines_df.copy()

    n = len(out)
    score = pd.Series(0.0, index=out.index, dtype="float64")

    # --- font_size_ratio
    fsr = _to_float_series(out.get("font_size_ratio"), default=np.nan).fillna(1.0)
    score += np.where((fsr >= 1.01) & (fsr < 1.2), 1.0, 0.0)
    score += np.where((fsr >= 1.2) & (fsr < 1.4), 2.0, 0.0)
    score += np.where((fsr >= 1.4), 3.0, 0.0)
    score += np.where((fsr < 1.0), -1.0, 0.0)
    score += np.where((fsr < 0.8), -3.0, 0.0)

    # --- char_count
    cc = _to_float_series(out.get("char_count"), default=np.nan)
    score += np.where(cc < 50, 0.5, 0.0)
    score += np.where(cc > 100, -1.0, 0.0)
    score += np.where(cc > 250, -3.0, 0.0)

    # --- capitalized_token_ratio
    cap_ratio = _safe_div(out.get("capitalized_word_count"), out.get("word_count"), fill=0.0)
    score += np.where(cap_ratio > 0.75, 0.5, 0.0)

    # --- styles
    # Note: we must handle missing columns - _to_bool_series returns empty Series if column is None
    is_bold = _to_bool_series(out.get("is_bold"), default=False)
    is_italic = _to_bool_series(out.get("is_italic"), default=False)
    is_underlined = _to_bool_series(out.get("is_underlined"), default=False)
    is_uppercase = _to_bool_series(out.get("is_uppercase"), default=False)
    
    # Fix: Ensure all boolean series have the same index as score
    # If a column is missing, _to_bool_series returns an empty Series, which causes NaN when adding
    if len(is_bold) == 0:
        is_bold = pd.Series(False, index=out.index, dtype=bool)
    if len(is_italic) == 0:
        is_italic = pd.Series(False, index=out.index, dtype=bool)
    if len(is_underlined) == 0:
        is_underlined = pd.Series(False, index=out.index, dtype=bool)
    if len(is_uppercase) == 0:
        is_uppercase = pd.Series(False, index=out.index, dtype=bool)

    score += is_bold.astype("float64") * 2.5
    score += is_italic.astype("float64") * 1.5
    score += is_underlined.astype("float64") * 1.5
    score += is_uppercase.astype("float64") * 1.0

    # --- text_align = center
    ta = out.get("text_align")
    if ta is not None:
        ta_norm = ta.astype("string").str.strip().str.lower()
        score += np.where(ta_norm.eq("center"), 1.0, 0.0)

    # --- non_stroking_color rarity bonus (+1 if not the prevalent color, excluding basic colors)
    nsc = out.get("non_stroking_color")
    if nsc is not None:
        nsc_norm = (
            nsc.astype("string")
            .str.strip()
            .str.lower()
            .replace({"": pd.NA, "none": pd.NA, "nan": pd.NA})
        )

        # Most prevalent color (mode) across the rows being scored (non-table slice)
        mode_color = nsc_norm.dropna().mode()
        prevalent = mode_color.iloc[0] if not mode_color.empty else pd.NA

        # Exclude "basic" colors from being considered "special"
        BASIC_COLORS = {
            "#000000",  # black
            "#ffffff",  # white
            "#fff",     # shorthand white (just in case)
            "#000",     # shorthand black
            "#111111", "#222222", "#333333", "#444444", "#555555", "#666666",
            "#777777", "#888888", "#999999", "#aaaaaa", "#bbbbbb", "#cccccc",
            "#dddddd", "#eeeeee",
        }

        is_basic = nsc_norm.isin(BASIC_COLORS)

        # +1 if:
        # - color is present
        # - color != prevalent color
        # - not a basic color
        bonus = (nsc_norm.notna()) & (nsc_norm != prevalent) & (~is_basic)
        score += bonus.astype("float64") * 1.0

    # =====================
    # only for pdf's
    # =====================

    # --- pdf-specific: layout_id has only 1 line (row) in that layout
    layout_id = out.get("layout_id")
    if layout_id is not None:
        # If layout_id is missing for some rows, treat as not single-line.
        grp_size = out.groupby("layout_id")["layout_id"].transform("size")
        score += np.where(grp_size == 1, 1.0, 0.0)

    out["heading_score"] = score
    return out

# ================================================================================
# Named heading candidate detection
# ================================================================================


def _detect_marker_candidates(
    lines_df: pd.DataFrame,
    compiled_patterns,
) -> pd.DataFrame:
    """
    Adds hierarchy marker detection columns:
      - hierarchy_marker : str | None   (the matched prefix text)
      - hierarchy_type   : str | None   (e.g. numbered_section, note, ...)

    Detection logic:
    - For EACH row, test patterns in order
    - First matching hierarchy_type wins
    - Only matches at START of text
    - Works row-wise but vector-friendly enough for current scale

    compiled_patterns: HierarchyTypePatternConfig object with .patterns attribute
    """

    out = lines_df.copy()

    out["hierarchy_marker"] = None
    out["hierarchy_type"] = None

    if "text" not in out.columns:
        return out

    texts = out["text"].astype("string").fillna("")

    # Extract patterns from HierarchyTypePatternConfig object
    pattern_list = compiled_patterns.patterns if hasattr(compiled_patterns, 'patterns') else []

    for pattern in pattern_list:
        h_type = pattern.hierarchy_type
        rx = pattern.compiled

        # vectorized startswith-style regex match
        matches = texts.str.match(rx)

        # only fill where not already matched by a higher-priority rule
        assign_mask = matches & out["hierarchy_type"].isna()

        if not assign_mask.any():
            continue

        # extract the actual marker substring - we want the FULL match, not just capture groups
        # Use apply with match.group(0) to get the complete matched string
        def extract_full_match(text):
            m = rx.match(text)
            if m:
                # group(0) is the entire match, strip trailing dots/spaces
                return m.group(0).rstrip('. \t')
            return None
        
        extracted = texts[assign_mask].apply(extract_full_match)

        out.loc[assign_mask, "hierarchy_marker"] = extracted
        out.loc[assign_mask, "hierarchy_type"] = h_type

    return out


# ================================================================================
# Add heading decision
# ================================================================================

_HEADING_SCORE_THRESHOLD = 2.5
_DEFAULT_HEADING_TYPE = "free_form"


def _add_heading_decision(
    lines_df: pd.DataFrame,
    heading_score_threshold: float = _HEADING_SCORE_THRESHOLD,
    default_heading_type: str = _DEFAULT_HEADING_TYPE,
) -> pd.DataFrame:
    """
    Final heading decision:
    - If heading_score > threshold AND block_role is not already set:
        - set block_role = "heading"
        - set heading_type = hierarchy_type if present else default_heading_type
    - Otherwise:
        - preserve existing block_role (e.g., toc_header, page_label, etc.)
        - heading_type is set for all rows passing threshold (regardless of existing role)

    This ensures that special roles like toc_header are preserved while still
    getting their heading_type populated based on their hierarchy detection.

    Adds/updates:
      - block_role (string) - only for unassigned rows
      - heading_type (string) - for all rows passing threshold
    """
    out = lines_df.copy()

    if "heading_type" not in out.columns:
        out["heading_type"] = pd.NA

    # Ensure block_role exists
    if "block_role" not in out.columns:
        out["block_role"] = pd.NA

    # Need heading_score to decide; if missing, no-op
    if "heading_score" not in out.columns:
        return out

    hs = pd.to_numeric(out["heading_score"], errors="coerce").fillna(-1e9)
    is_heading = hs >= float(heading_score_threshold)

    # Set block_role ONLY for winners that don't already have a role
    # Preserve existing roles like toc_header, page_label, etc.
    existing_role = out["block_role"].astype("string").str.strip()
    no_role_yet = existing_role.isna() | (existing_role == "") | (existing_role == "nan")
    
    # Only assign "heading" to rows that pass threshold AND don't have a role yet
    should_assign_heading = is_heading & no_role_yet
    out.loc[should_assign_heading, "block_role"] = "heading"

    # Pick heading_type from hierarchy_type if present, else default
    if "hierarchy_type" in out.columns:
        ht = out["hierarchy_type"].astype("string").str.strip()
        chosen = ht.where(ht.notna() & (ht != ""), default_heading_type)
    else:
        chosen = pd.Series(default_heading_type, index=out.index, dtype="string")

    out.loc[is_heading, "heading_type"] = chosen.loc[is_heading]

    return out


# ================================================================================
# Add heading fingerprints
# ================================================================================

_FINGERPRINT_COLS = [
    "document_region",
    "heading_type",
    "font_size_ratio",
    "is_bold",
    "is_italic",
    "is_underlined",
    "is_uppercase",
    "text_align",
    "x_left_bucket",
    "font_name",
    "font_family",
    "non_stroking_color",
    #"background_non_stroking_color", -- May result in too much splitting
]

_ID_MAX_PARAMS_DIFF_DEFAULT = 0
_ID_MAX_PARAMS_DIFF_SPECIAL = 2
_SPECIAL_HEADING_TYPES = {
    "item", "part", "note", "annex", "article", "section", "proposal",
    "section_abbreviated", "schedule", "title", "subpart", "chapter", 
    "amendment", "rule", "figure", "table", "appendix", "exhibit",
}

def _normalize_scalar(v):
    """Make values JSON-stable + comparable (incl. pandas / numpy scalars)."""
    # pandas / numpy missing
    if v is None or pd.isna(v):
        return None
    # numpy scalar → python scalar
    if isinstance(v, np.generic):
        v = v.item()
    # bool (now guaranteed to be python bool)
    if isinstance(v, bool):
        return bool(v)
    # floats
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    # strings
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        low = s.lower()
        if low in {"true", "t", "1", "yes", "y"}:
            return True
        if low in {"false", "f", "0", "no", "n"}:
            return False
        return s
    # ints
    if isinstance(v, int):
        return int(v)
    # fallback (last resort, but JSON-safe)
    return v


def _canonical_json_hash(obj: dict) -> str:
    """
    Strict hash: canonical JSON (sorted keys, compact separators), then SHA-256 hex.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_heading_fingerprints_and_ids(
    lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds:
      - heading_fingerprint : dict (JSONB-ready)
      - heading_hash        : str (sha256 of canonical JSON)
      - heading_fp_id          : int (stable-ish within doc; tolerance for special heading types)

    Fingerprint fields (strict, used for heading_hash):
      doc_region, heading_type, font_size_ratio,
      is_bold, is_italic, is_underlined, is_uppercase,
      text_align, x_left_bucket, font_family,
      non_stroking_color, background_non_stroking_color

    ID assignment:
      - Only assigns IDs to rows where block_role == "heading"
      - For a new heading, tries to match to a prior heading in the SAME (doc_region, heading_type)
      - If heading_type (aka heading_type category) is “special”, allow up to
        _ID_MAX_PARAMS_DIFF_SPECIAL differing params (across the remaining fingerprint fields).
      - Otherwise allow up to _ID_MAX_PARAMS_DIFF_DEFAULT (default 0 → exact match required).

    Notes:
      - heading_hash is always strict (different params => different hash)
      - heading_id can reuse earlier IDs under the special-type tolerance rule
    """
    out = lines_df.copy()

    # Ensure columns exist
    if "heading_fingerprint" not in out.columns:
        out["heading_fingerprint"] = pd.NA
    if "heading_hash" not in out.columns:
        out["heading_hash"] = pd.NA
    if "heading_fp_id" not in out.columns:
        out["heading_fp_id"] = pd.NA

    # Only headings get these filled
    if "block_role" not in out.columns:
        return out

    # ensure pure bool (no pd.NA)
    br = out["block_role"].astype("string")
    is_heading = br.str.strip().str.lower().isin({"heading", "toc_header", "exhibit_header"}).fillna(False).astype(bool)

    if not is_heading.any():
        return out

    groups = [(None, out.index)]

    # Precompute order within group for determinism (page_number, line_id if available)
    def _sorted_idx(idx):
        sub = out.loc[idx]
        sort_cols = []
        if "page_number" in sub.columns:
            sort_cols.append("page_number")
        if "line_id" in sub.columns:
            sort_cols.append("line_id")
        if sort_cols:
            return sub.sort_values(sort_cols, kind="mergesort").index
        return sub.index

    for _, idx in groups:
        idx = _sorted_idx(idx)

        # per-doc state
        next_id = 1
        # store previous headings as list of dicts:
        # {"heading_fp_id": int, "doc_region": ..., "heading_type": ..., "fp": {...}}
        seen = []

        for i in idx:
            if not is_heading.loc[i]:
                continue

            row = out.loc[i]

            # Build fingerprint dict (strict)
            fp = {}
            for col in _FINGERPRINT_COLS:
                if col in out.columns:
                    fp[col] = _normalize_scalar(row[col])
                else:
                    fp[col] = None

            # -----------------------------
            # UPDATE 1: ensure heading_type in fingerprint is never None
            # (use default free_form if missing/empty)
            # -----------------------------
            ht = fp.get("heading_type")
            if ht is None or (isinstance(ht, str) and ht.strip() == ""):
                fp["heading_type"] = "free_form"

            # Normalize heading_type for matching logic
            ht_norm = fp.get("heading_type")
            if isinstance(ht_norm, str):
                ht_norm_l = ht_norm.strip().lower()
            else:
                ht_norm_l = ""

            # -----------------------------
            # UPDATE 2: round font_size_ratio to 2 digits (e.g., 2.50)
            # -----------------------------
            fsr = fp.get("font_size_ratio")
            if fsr is None:
                fp["font_size_ratio"] = None
            else:
                # fsr should be float-like by now; round strictly to 2 decimals
                try:
                    fp["font_size_ratio"] = float(round(float(fsr), 2))
                except Exception:
                    fp["font_size_ratio"] = None

            # Strict hash
            h = _canonical_json_hash(fp)

            # Decide allowed param diffs for ID matching
            is_special = ht_norm_l in _SPECIAL_HEADING_TYPES
            max_diff = _ID_MAX_PARAMS_DIFF_SPECIAL if is_special else _ID_MAX_PARAMS_DIFF_DEFAULT

            # Try to match a prior heading_id (same doc_region + heading_type)
            best_match_id = None
            best_match_diff = None

            for prev in seen:
                if prev["doc_region"] != fp.get("doc_region"):
                    continue
                if prev["heading_type_norm"] != ht_norm_l:
                    continue

                # Count diffs across remaining params (excluding doc_region + heading_type)
                diffs = 0
                prev_fp = prev["fp"]

                for k in _FINGERPRINT_COLS:
                    if k in {"doc_region", "heading_type"}:
                        continue
                    if prev_fp.get(k) != fp.get(k):
                        diffs += 1
                        if diffs > max_diff:
                            break

                if diffs <= max_diff:
                    if best_match_diff is None or diffs < best_match_diff:
                        best_match_diff = diffs
                        best_match_id = prev["heading_fp_id"]
                        if diffs == 0:
                            break  # can't do better

            if best_match_id is None:
                assigned_id = next_id
                next_id += 1
            else:
                assigned_id = best_match_id

            # write results
            out.at[i, "heading_fingerprint"] = fp
            out.at[i, "heading_hash"] = h
            out.at[i, "heading_fp_id"] = int(assigned_id)

            # store as candidate for future matches
            seen.append(
                {
                    "heading_fp_id": int(assigned_id),
                    "doc_region": fp.get("doc_region"),
                    "heading_type_norm": ht_norm_l,
                    "fp": fp,
                }
            )

    return out


# ================================================================================
# Suppress certain headings
# ================================================================================

# NOTE: Coverpage, TODO(Signatures) and repeated headings.
# Multi-line headings (consecutive headings with same fp_id) are handled in heading_id assignment.

def suppress_and_merge_headings(
    df: pd.DataFrame,
    *,
    block_role_col: str = "block_role",
    heading_role_value: str = "heading",
    text_col: str = "text",
    fp_col: str = "heading_fp_id",
    line_id_col: str = "line_id",
    document_region_col: str = "document_region",
    coverpage_value: str = "coverpage",
    # columns to blank when suppressing a heading
    blank_cols: tuple[str, ...] = (
        "heading_type",
        "heading_id",
        "heading_fp_id",
        "heading_fingerprint",
        "heading_hash",
    ),
) -> pd.DataFrame:
    """
    Pre-step before hierarchy inference.

    Does 2 things:

    (1) Suppress repeated headings:
        Within each heading_fp_id, if there are >=3 heading rows with the same text,
        those rows are suppressed (will not be headings anymore).
        -> block_role set to NaN and blank_cols set to NaN.

    (2) Coverpage rule:
        For document_region == "coverpage", keep ONLY the first heading,
        suppress all remaining headings in coverpage.

    Note: Multi-line headings (consecutive headings with same fp_id) are handled
    in the heading_id assignment step (infer_heading_hierarchy), where they get
    the same heading_id and can be merged later without duplication.

    Assumes df is already in reading order.
    """

    out = df.copy()

    if block_role_col not in out.columns:
        return out

    # --- helpers ---
    def _is_heading_mask(x: pd.Series) -> pd.Series:
        return (
            x.astype("string")
            .str.strip()
            .str.lower()
            .eq(heading_role_value)
            .fillna(False)
        )

    def _suppress_rows(mask: pd.Series) -> None:
        if mask is None or not mask.any():
            return
        out.loc[mask, block_role_col] = np.nan
        for c in blank_cols:
            if c in out.columns:
                out.loc[mask, c] = np.nan

    # =========================================================================
    # (1) Suppress "same text appears >=3 times within same fp" (headings only)
    #     >= 3 if text_align = center, otherwise >= 5
    # =========================================================================
    is_heading = _is_heading_mask(out[block_role_col])
    if fp_col in out.columns and text_col in out.columns and is_heading.any():
        fp = out.loc[is_heading, fp_col]
        txt = out.loc[is_heading, text_col].astype("string").fillna("").str.strip()
        
        # Get text_align for headings (check if column exists)
        if "text_align" in out.columns:
            align = out.loc[is_heading, "text_align"].astype("string").fillna("").str.strip().str.lower()
        else:
            align = pd.Series("", index=out.loc[is_heading].index)

        # count occurrences per (fp, text, text_align)
        counts = (
            pd.DataFrame({"fp": fp, "txt": txt, "align": align})
            .groupby(["fp", "txt", "align"], dropna=False)
            .size()
        )

        # Apply different thresholds based on text_align
        bad_pairs = []
        for (fp_val, txt_val, align_val), count in counts.items():
            threshold = 3 if align_val == "center" else 5
            if count >= threshold:
                bad_pairs.append((fp_val, txt_val, align_val))
        
        if len(bad_pairs) > 0:
            pair_df = pd.DataFrame({"fp": fp, "txt": txt, "align": align}, index=out.loc[is_heading].index)
            bad_mask = pair_df.set_index(["fp", "txt", "align"]).index.isin(bad_pairs)
            suppress_mask = pd.Series(False, index=out.index)
            suppress_mask.loc[pair_df.index] = bad_mask
            # Set block_role to "suppressed_repeated_heading" instead of blanking it
            if suppress_mask.any():
                out.loc[suppress_mask, block_role_col] = "suppressed_repeated_heading"
                for c in blank_cols:
                    if c in out.columns:
                        out.loc[suppress_mask, c] = np.nan

    # =========================================================================
    # (2) Coverpage: keep only first heading per page_number (which may already be merged)
    # =========================================================================
    """ --> TODO: Hold off on this until a better algorithm is developed, it trims too much
    # Enable largest font, or if starts with FORM, SCHEDULE, COPY, COPIES
    # Recompute headings after merging
    is_heading = _is_heading_mask(out[block_role_col])

    if (
        document_region_col in out.columns
        and is_heading.any()
    ):
        is_cover = out[document_region_col].astype("string").fillna("").str.strip().str.lower().eq(coverpage_value)
        cover_heading_idx = out.index[is_heading & is_cover]

        if len(cover_heading_idx) > 0:
            suppress_mask = pd.Series(False, index=out.index)
            
            if "page_number" in out.columns:
                # Group by page_number and keep first heading per page
                cover_headings = out.loc[cover_heading_idx]
                # Group by page_number and keep first in each group
                for page_num, group_idx in cover_headings.groupby("page_number", dropna=False).groups.items():
                    page_heading_indices = list(group_idx)
                    if len(page_heading_indices) > 1:
                        # Keep first, suppress rest for this page
                        suppress_mask.loc[page_heading_indices[1:]] = True
            else:
                # Fallback: keep first heading across whole coverpage if page_number not available
                suppress_mask.loc[cover_heading_idx[1:]] = True
            
            _suppress_rows(suppress_mask)
    """ 

    return out

# ================================================================================
# Finalize block roles
# ================================================================================

def finalize_block_roles(
    df: pd.DataFrame,
    *,
    block_role_col: str = "block_role",
    default_role: str = "paragraph",
) -> pd.DataFrame:
    """
    Set all blank/NaN block_role values to the default role (typically "paragraph").
    """
    out = df.copy()
    if block_role_col in out.columns:
        # Find rows where block_role is blank/NaN
        blank_mask = out[block_role_col].isna() | (out[block_role_col].astype("string").str.strip() == "")
        if blank_mask.any():
            out.loc[blank_mask, block_role_col] = default_role
    return out


# ================================================================================
# Add heading weights
# ================================================================================

# ===============
# CONFIG
# ===============

# static feature weights
STATIC_WEIGHTS = {
    "is_bold": 1.0,
    "is_uppercase": 1.0,
    "text_align_center": 1.0,   # derived from text_align == "center"
    "is_italic": -0.5,
}

# heading type order (low → high)
HEADING_TYPE_RANK = {
    # Highest priority
    "sec_chapter": 0, "exhibit": 0,
    # Rank 1
    "part": 1,
    # Rank 2
    "item": 2, "appendix": 2, "annex": 2, "subpart": 2,
    # Rank 3
    "note": 3,
    # Rank 4
    "free_form": 4,
    # Rank 5
    "table": 5, "figure": 5,
}

# ================
# MAIN FUNCTION
# ================

def add_heading_weights(
    df: pd.DataFrame,
    *,
    block_role_col: str = "block_role",
    heading_role_value: str = "heading",
    document_region_col: str = "document_region",
) -> pd.DataFrame:
    """
    Adds:
      - heading_weight_static: computed ONLY where block_role == "heading"
      - heading_weight_dynamic: computed ONLY where block_role == "heading",
        and scored vs prior heading WITHIN each document_region.
    Non-heading rows get 0.0 for both columns.

    Assumes df is already in reading order.
    """
    out = df.copy()

    out["heading_weight_static"] = np.nan
    out["heading_weight_dynamic"] = np.nan

    if block_role_col not in out.columns:
        return out

    is_heading = out[block_role_col].astype("string").str.strip().str.lower().eq(heading_role_value).fillna(False)
    if not is_heading.any():
        return out

    # -------------------------
    # 1) STATIC (only headings)
    # -------------------------
    static = pd.Series(0.0, index=out.index, dtype="float32")

    if "is_bold" in out.columns:
        static.loc[is_heading] += out.loc[is_heading, "is_bold"].fillna(False).astype(int) * STATIC_WEIGHTS["is_bold"]

    if "is_uppercase" in out.columns:
        static.loc[is_heading] += out.loc[is_heading, "is_uppercase"].fillna(False).astype(int) * STATIC_WEIGHTS["is_uppercase"]

    if "text_align" in out.columns:
        is_center = (
            out.loc[is_heading, "text_align"]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("center")
            .fillna(False)
        )
        static.loc[is_heading] += is_center.astype(int) * STATIC_WEIGHTS["text_align_center"]

    if "is_italic" in out.columns:
        static.loc[is_heading] += out.loc[is_heading, "is_italic"].fillna(False).astype(int) * STATIC_WEIGHTS["is_italic"]

    static = static.clip(lower=0.0)  # optional: keep non-negative
    out.loc[is_heading, "heading_weight_static"] = static.loc[is_heading].astype("float32")

    # -------------------------
    # 2) DYNAMIC (signed margin)
    # -------------------------
    if document_region_col in out.columns:
        region_key = out[document_region_col].astype("string").fillna("")
    else:
        region_key = pd.Series("", index=out.index, dtype="string")

    idx = out.index[is_heading]
    groups = region_key.loc[idx]

    dynamic = pd.Series(0, index=idx, dtype="int16")

    # vote 1: font size
    if "font_size_ratio" in out.columns:
        font = pd.to_numeric(out.loc[idx, "font_size_ratio"], errors="coerce")
        font_prev = font.groupby(groups).shift(1)

        dynamic.loc[font > font_prev] += 1
        dynamic.loc[font < font_prev] -= 1

    # vote 2: heading type strength
    if "heading_type" in out.columns:
        rank = out.loc[idx, "heading_type"].map(HEADING_TYPE_RANK)
        rank_prev = rank.groupby(groups).shift(1)

        # lower rank = stronger
        dynamic.loc[(rank < rank_prev)] += 1
        dynamic.loc[(rank > rank_prev)] -= 1

    out.loc[idx, "heading_weight_dynamic"] = dynamic

    return out


# ================================================================================
# Assign heading id
# ================================================================================

def assign_heading_id(
    df: pd.DataFrame,
    *,
    block_role_col: str = "block_role",
    fp_col: str = "heading_fp_id",
    line_id_col: str = "line_id",
    out_col: str = "heading_id",
) -> pd.DataFrame:
    """
    Assign heading_id to heading-like rows by grouping *consecutive* rows that:
      - are heading-like (heading / toc_header / exhibit_header)
      - share the same heading_fp_id
      - are consecutive by line_id (line_id increments by 1)

    This preserves multi-line headings: adjacent lines with the same fp are given the same heading_id.

    Non-heading rows get NA in out_col.
    """
    out = df.copy()

    out[out_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    if block_role_col not in out.columns:
        return out

    is_heading = (
        out[block_role_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"heading", "toc_header", "exhibit_header"})
        .fillna(False)
    )
    if not is_heading.any():
        return out

    if fp_col not in out.columns:
        # fallback: sequential ids for heading rows
        next_id = 1
        for idx in out.index[is_heading]:
            out.at[idx, out_col] = next_id
            next_id += 1
        return out

    heading_idx = out.index[is_heading]
    fp_vals = out.loc[heading_idx, fp_col].values

    line_vals = None
    if line_id_col in out.columns:
        line_vals = pd.to_numeric(out.loc[heading_idx, line_id_col], errors="coerce").values

    idx_list = list(heading_idx)
    next_id = 1
    i = 0

    while i < len(idx_list):
        cur_idx = idx_list[i]
        cur_fp = fp_vals[i]
        cur_id = next_id

        out.at[cur_idx, out_col] = cur_id

        j = i + 1
        while j < len(idx_list):
            nxt_fp = fp_vals[j]

            same_fp = pd.notna(cur_fp) and pd.notna(nxt_fp) and (nxt_fp == cur_fp)

            consecutive = True
            if line_vals is not None:
                prev_line = line_vals[j - 1]
                this_line = line_vals[j]
                consecutive = (
                    np.isfinite(prev_line)
                    and np.isfinite(this_line)
                    and (this_line == prev_line + 1)
                )

            if same_fp and consecutive:
                out.at[idx_list[j], out_col] = cur_id
                j += 1
            else:
                break

        i = j
        next_id += 1

    return out


# ================================================================================
# Make final heading hierarchy
# ================================================================================

# IMPORTANT TODO!!! 
# NOTE: + also think about numbered headings (1.1, 1.1.1, etc.)

"""
Adds (only for heading-like rows; others stay NA):
    - parent_heading_id (Int64)
    - parent_heading_text (string)
    - heading_level (Int64)
    - heading_fp_path (string)  e.g. "10 > 13 > 14 > 19"

General:
    - Walks per heading_id (multi-row heading supported via line_id start/end).
    - Per document_region, first heading is always level 1.
    - heading_fp_path shows the fp chain root->...->current.

Algorithm (in priority order):
    1) same fp as previous => sibling of prior
    2) if current heading starts directly underneath prior heading (start_line_id == prior_end_line_id + 1)
        => pop down (child) regardless
    3) if current fp is in prior fp_path => attach as sibling of the last occurrence of that fp in the path
        (i.e., parent becomes the node above it, or root if it was root)
    4) else compare prior_static vs current_total(static+dynamic):
        - current < prior_static => pop down (child)
        - current > prior_static => pop up (see special-parent override below)
        - current == prior_static => use relationship memory:
            if we have evidence that current_fp is an ancestor of prev_fp elsewhere, pop up (resume under last seen)
            else sibling of prior
    5) Pop-up special-parent override:
        when the decision is "pop up", if the current chain (root..prior) contains a non-free_form heading_type
        (excluding table/figure), then hang the current under the *last* such special node; otherwise sibling of prior.

Assumptions:
    - df is already in reading order.
    - heading_id_col exists for heading rows (multi-row headings share same heading_id).
    - line_id_col exists for "directly underneath" rule; if missing/invalid, that rule is skipped.
"""

"""
Can you give me a new function that does:  
General: 
- it walks per heading_id: A heading_id can consist out of multiple line_id's (multi-row heading) 
- per document_region, first heading is always level 1 
- make a column heading_fp_path which shows the path of heading_fp_id's that have been followed to come to a heading  
Algorithm: 
- if the heading_fp_id stays equal, always a sibling of the prior 
- if the current heading is directly underneath another one (line_id +1) then always pop down, regardless of what happened before 
- check if the current heading is in the heading_fp_path of the prior heading, if yes attach it to the last occurence of the parent heading_fp_id (if any, may be root as well) 
- if not: score prior (heading_weight_static) vs current (heading_weight_static + heading_weight_dynamic) 
-- if current < prior: pop down 
-- if current > prior: pop up 
-- if current == prior: check if there existed a relationship before in other paths (for example if now we have a tie between 13 and 19 and we knew before: "10 > 13 > 14 > 19", then we know 13 pops up) 
- When popping up, check if the current chain from root contains a non-heading_type= free_form heading (excl table, figure). If yes, hang it underneath there, if no, just make it a sibling of the prior heading
"""


#_NON_PARENT_TYPES = {"table", "figure"} # Remove this


def infer_heading_hierarchy(
    df: pd.DataFrame,
    *,
    block_role_col: str = "block_role",
    document_region_col: str = "document_region",
    text_col: str = "text",
    fp_col: str = "heading_fp_id",
    heading_type_col: str = "heading_type",
    static_col: str = "heading_weight_static",
    dynamic_col: str = "heading_weight_dynamic",
) -> pd.DataFrame:
    """
    Adds (only for headings; others stay NaN):
      - heading_id (Int64)
      - parent_heading_id (Int64)
      - parent_heading_text (string)
      - heading_level (Int64)

    Processes rows with block_role in: heading, toc_header, exhibit_header

    Assumes df is already in reading order.
    """

    # Make a copy to avoid modifying original
    out = df.copy()
    
    # Initialize output columns (heading_id should already exist from assign_heading_id)
    if "parent_heading_id" not in out.columns:
        out["parent_heading_id"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    if "parent_heading_text" not in out.columns:
        out["parent_heading_text"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
    if "heading_level" not in out.columns:
        out["heading_level"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    
    if block_role_col not in out.columns or "heading_id" not in out.columns:
        return out
    
    # Filter to heading-like rows that have a heading_id
    heading_mask = (
        out[block_role_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"heading", "toc_header", "exhibit_header"})
        .fillna(False)
    ) & out["heading_id"].notna()
    
    if not heading_mask.any():
        return out
    
    # Get region column
    if document_region_col in out.columns:
        region = out[document_region_col].astype("string").fillna("")
    else:
        region = pd.Series("", index=out.index, dtype="string")
    
    # Helper: safe float
    def sf(v) -> float:
        try:
            if pd.isna(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0
    
    # Process per document_region
    for region_name in region[heading_mask].unique():
        region_heading_mask = heading_mask & (region == region_name)
        region_heading_rows = out[region_heading_mask].copy()
        
        if region_heading_rows.empty:
            continue
        
        # Group by heading_id to get unique headings (multi-line headings have same heading_id)
        # For each heading_id, take the first row's properties and last row's line_id
        grouped = region_heading_rows.groupby("heading_id", sort=False)
        
        unique_headings = []
        for hid, group in grouped:
            first_row = group.iloc[0]
            last_row = group.iloc[-1]
            unique_headings.append({
                "heading_id": int(hid),
                "indices": group.index.tolist(),  # All row indices with this heading_id
                "fp": first_row[fp_col] if fp_col in first_row.index and pd.notna(first_row[fp_col]) else pd.NA,
                "text": first_row[text_col] if text_col in first_row.index and pd.notna(first_row[text_col]) else "",
                "static": sf(first_row[static_col]) if static_col in first_row.index else 0.0,
                "dynamic": sf(first_row[dynamic_col]) if dynamic_col in first_row.index else 0.0,
                "heading_type": str(first_row[heading_type_col]).strip().lower() if heading_type_col in first_row.index and pd.notna(first_row[heading_type_col]) else "free_form",
                "end_line_id": last_row.get("line_id") if "line_id" in last_row.index else None,
            })
        
        # Track state
        path_stack = []  # List of dicts: {heading_id, fp, text, level, static_weight, heading_type, end_line_id}
        relationship_memory = {}  # {(parent_fp, child_fp): count}
        
        for heading in unique_headings:
            curr_heading_id = heading["heading_id"]
            curr_fp = heading["fp"]
            curr_text = heading["text"]
            curr_static = heading["static"]
            curr_dynamic = heading["dynamic"]
            curr_total = curr_static + curr_dynamic
            curr_type = heading["heading_type"]
            curr_line_id = heading["end_line_id"]
            heading_indices = heading["indices"]  # All row indices for this heading_id
            
            # First heading in region: always root level 1
            if not path_stack:
                for idx in heading_indices:
                    out.at[idx, "parent_heading_id"] = pd.NA
                    out.at[idx, "parent_heading_text"] = pd.NA
                    out.at[idx, "heading_level"] = 1
                
                path_stack.append({
                    "heading_id": curr_heading_id,
                    "fp": curr_fp,
                    "text": curr_text,
                    "level": 1,
                    "static_weight": curr_static,
                    "heading_type": curr_type,
                    "end_line_id": curr_line_id
                })
                continue
            
            # Get prior heading
            prior = path_stack[-1]
            prior_fp = prior["fp"]
            prior_static = prior["static_weight"]
            prior_end_line = prior.get("end_line_id")
            
            # Build current fp_path
            fp_path = [node["fp"] for node in path_stack]
            
            # Decision logic
            decision = None  # "sibling", "child", "reattach", or "pop_up"
            
            # Rule 1: Same fp as prior → sibling
            if pd.notna(curr_fp) and pd.notna(prior_fp) and curr_fp == prior_fp:
                decision = "sibling"
            
            # Rule 2: Directly underneath (line_id + 1) → pop down
            elif (curr_line_id is not None and prior_end_line is not None and 
                  pd.notna(curr_line_id) and pd.notna(prior_end_line) and
                  curr_line_id == prior_end_line + 1):
                decision = "child"
            
            # Rule 3: Current fp in prior fp_path → attach to last occurrence
            elif pd.notna(curr_fp) and curr_fp in fp_path:
                decision = "reattach"
            
            # Rule 4: Compare weights
            else:
                if curr_total < prior_static:
                    decision = "child"
                elif curr_total > prior_static:
                    decision = "pop_up"
                else:  # curr_total == prior_static
                    # Check relationship memory
                    found_relationship = False
                    for (parent_fp, child_fp) in relationship_memory:
                        if child_fp == prior_fp and parent_fp == curr_fp:
                            # We know curr_fp is ancestor of prior_fp
                            decision = "pop_up"
                            found_relationship = True
                            break
                    
                    if not found_relationship:
                        # Default to child when undecided
                        decision = "child"
            
            # Execute decision
            if decision == "sibling":
                # Same level as prior, same parent
                if len(path_stack) > 1:
                    parent = path_stack[-2]
                    for idx in heading_indices:
                        out.at[idx, "parent_heading_id"] = parent["heading_id"]
                        out.at[idx, "parent_heading_text"] = parent["text"]
                        out.at[idx, "heading_level"] = prior["level"]
                    
                    # Replace prior with current in path
                    path_stack[-1] = {
                        "heading_id": curr_heading_id,
                        "fp": curr_fp,
                        "text": curr_text,
                        "level": prior["level"],
                        "static_weight": curr_static,
                        "heading_type": curr_type,
                        "end_line_id": curr_line_id
                    }
                else:
                    # Prior is root, current is also root sibling
                    for idx in heading_indices:
                        out.at[idx, "parent_heading_id"] = pd.NA
                        out.at[idx, "parent_heading_text"] = pd.NA
                        out.at[idx, "heading_level"] = 1
                    
                    path_stack[-1] = {
                        "heading_id": curr_heading_id,
                        "fp": curr_fp,
                        "text": curr_text,
                        "level": 1,
                        "static_weight": curr_static,
                        "heading_type": curr_type,
                        "end_line_id": curr_line_id
                    }
                
                # Update relationship memory
                if len(path_stack) > 1:
                    parent_fp = path_stack[-2]["fp"]
                    if pd.notna(parent_fp) and pd.notna(curr_fp):
                        relationship_memory[(parent_fp, curr_fp)] = relationship_memory.get((parent_fp, curr_fp), 0) + 1
            
            elif decision == "child":
                # Child of prior
                for idx in heading_indices:
                    out.at[idx, "parent_heading_id"] = prior["heading_id"]
                    out.at[idx, "parent_heading_text"] = prior["text"]
                    out.at[idx, "heading_level"] = prior["level"] + 1
                
                path_stack.append({
                    "heading_id": curr_heading_id,
                    "fp": curr_fp,
                    "text": curr_text,
                    "level": prior["level"] + 1,
                    "static_weight": curr_static,
                    "heading_type": curr_type,
                    "end_line_id": curr_line_id
                })
                
                # Update relationship memory
                if pd.notna(prior_fp) and pd.notna(curr_fp):
                    relationship_memory[(prior_fp, curr_fp)] = relationship_memory.get((prior_fp, curr_fp), 0) + 1
            
            elif decision == "reattach":
                # Find last occurrence of curr_fp in path
                target_idx = None
                for i in range(len(path_stack) - 1, -1, -1):
                    if pd.notna(path_stack[i]["fp"]) and pd.notna(curr_fp) and path_stack[i]["fp"] == curr_fp:
                        target_idx = i
                        break
                
                if target_idx is not None:
                    # Pop stack to target level
                    path_stack = path_stack[:target_idx + 1]
                    
                    # Make sibling of target
                    if target_idx > 0:
                        parent = path_stack[target_idx - 1]
                        for idx in heading_indices:
                            out.at[idx, "parent_heading_id"] = parent["heading_id"]
                            out.at[idx, "parent_heading_text"] = parent["text"]
                            out.at[idx, "heading_level"] = path_stack[target_idx]["level"]
                        
                        path_stack[-1] = {
                            "heading_id": curr_heading_id,
                            "fp": curr_fp,
                            "text": curr_text,
                            "level": path_stack[target_idx]["level"],
                            "static_weight": curr_static,
                            "heading_type": curr_type,
                            "end_line_id": curr_line_id
                        }
                    else:
                        # Target is root
                        for idx in heading_indices:
                            out.at[idx, "parent_heading_id"] = pd.NA
                            out.at[idx, "parent_heading_text"] = pd.NA
                            out.at[idx, "heading_level"] = 1
                        
                        path_stack[-1] = {
                            "heading_id": curr_heading_id,
                            "fp": curr_fp,
                            "text": curr_text,
                            "level": 1,
                            "static_weight": curr_static,
                            "heading_type": curr_type,
                            "end_line_id": curr_line_id
                        }
            
            elif decision == "pop_up":
                # Rule 5: Check for special-parent override
                # Find last non-free_form heading (excl table/figure) in stack
                special_parent_idx = None
                for i in range(len(path_stack) - 1, -1, -1):
                    node_type = path_stack[i]["heading_type"]
                    if node_type not in ["free_form", "table", "figure"]:
                        special_parent_idx = i
                        break
                
                if special_parent_idx is not None:
                    # Hang under last special heading
                    special_parent = path_stack[special_parent_idx]
                    path_stack = path_stack[:special_parent_idx + 1]
                    
                    for idx in heading_indices:
                        out.at[idx, "parent_heading_id"] = special_parent["heading_id"]
                        out.at[idx, "parent_heading_text"] = special_parent["text"]
                        out.at[idx, "heading_level"] = special_parent["level"] + 1
                    
                    path_stack.append({
                        "heading_id": curr_heading_id,
                        "fp": curr_fp,
                        "text": curr_text,
                        "level": special_parent["level"] + 1,
                        "static_weight": curr_static,
                        "heading_type": curr_type,
                        "end_line_id": curr_line_id
                    })
                    
                    # Update relationship memory
                    if pd.notna(special_parent["fp"]) and pd.notna(curr_fp):
                        relationship_memory[(special_parent["fp"], curr_fp)] = relationship_memory.get((special_parent["fp"], curr_fp), 0) + 1
                else:
                    # No special parent
                    # Check if this is a high-priority heading type (rank ≤ 1) that should become root
                    curr_type_rank = HEADING_TYPE_RANK.get(curr_type, 999)
                    is_high_priority = curr_type_rank <= 1  # sec_chapter, exhibit, part
                    
                    path_stack.pop()
                    
                    if is_high_priority or len(path_stack) == 0:
                        # High-priority headings become root level (Part, Exhibit, SEC Chapter)
                        # OR stack is empty, so becomes new root
                        for idx in heading_indices:
                            out.at[idx, "parent_heading_id"] = pd.NA
                            out.at[idx, "parent_heading_text"] = pd.NA
                            out.at[idx, "heading_level"] = 1
                        
                        path_stack.append({
                            "heading_id": curr_heading_id,
                            "fp": curr_fp,
                            "text": curr_text,
                            "level": 1,
                            "static_weight": curr_static,
                            "heading_type": curr_type,
                            "end_line_id": curr_line_id
                        })
                    else:
                        # Not high-priority and stack not empty, make sibling of prior
                        parent = path_stack[-1]
                        for idx in heading_indices:
                            out.at[idx, "parent_heading_id"] = parent["heading_id"]
                            out.at[idx, "parent_heading_text"] = parent["text"]
                            out.at[idx, "heading_level"] = parent["level"] + 1
                        
                        path_stack.append({
                            "heading_id": curr_heading_id,
                            "fp": curr_fp,
                            "text": curr_text,
                            "level": parent["level"] + 1,
                            "static_weight": curr_static,
                            "heading_type": curr_type,
                            "end_line_id": curr_line_id
                        })
                        
                        # Update relationship memory
                        if pd.notna(parent["fp"]) and pd.notna(curr_fp):
                            relationship_memory[(parent["fp"], curr_fp)] = relationship_memory.get((parent["fp"], curr_fp), 0) + 1
    
    return out

# ================================================================================
# Public API
# ================================================================================

def assign_doc_hierarchy(
    lines_df: pd.DataFrame,
    compiled_patterns,
) -> pd.DataFrame:
    """
    Orchestrator entrypoint.

    Pipeline:
    0. Prefilter lines (ONLY for step 1 scoring)
    1. Calculate heading_score (ONLY on prefiltered df)
    2. Detect hierarchy markers (full df)
    3. Make heading decision (full df)
    4. Heading fingerprints + IDs (full df)

    Returns: full dataframe with new columns added
    """
    out = lines_df.copy()

    # ------- Step 0: Prefilter for scoring only ------- #
    scored_input = pre_filter_lines(out)

    # ------- Step 1: Heading score (only on prefiltered df) ------- #

    # Initialize heading_score column on FULL df
    if "heading_score" not in out.columns:
        out["heading_score"] = 0.0

    # If nothing to score, skip
    if len(scored_input) > 0:
        scored_slice = add_heading_score(scored_input)

        # Merge scores back using line_id (preferred) or index (fallback)
        if "line_id" in out.columns and "line_id" in scored_slice.columns:
            scored_key = (
                scored_slice
                .drop_duplicates(subset=["line_id"], keep="first")[["line_id", "heading_score"]]
            )
            out = out.merge(scored_key, on="line_id", how="left", suffixes=("", "_scored"))
            out["heading_score"] = out["heading_score_scored"].fillna(out["heading_score"]).astype("float64")
            out = out.drop(columns=["heading_score_scored"])
        else:
            # Index-based merge (fallback)
            idx = scored_slice.index.intersection(out.index)
            out.loc[idx, "heading_score"] = scored_slice.loc[idx, "heading_score"].astype("float64")

    # ------- Step 2: Hierarchy type detection (full df) ------- #
    out = _detect_marker_candidates(out, compiled_patterns)

    # ------- Step 3: Heading decision (full df) ------- #
    out = _add_heading_decision(out)

    # ------- Step 4: Heading fingerprints and IDs (full df) ------- #
    out = add_heading_fingerprints_and_ids(out)

    # ------- Step 5: Suppress and merge headings (full df) ------- #
    out = suppress_and_merge_headings(out)

    # ------- Step 6: Finalize block roles (full df) ------- #
    out = finalize_block_roles(out)

    # ------- Step 6: Heading weights (full df) ------- #
    out = add_heading_weights(out)

    # ------- Step 7: Assign heading id (full df) ------- #
    out = assign_heading_id(out)

    # ------- Step 7: Make final heading hierarchy (full df) ------- #
    out = infer_heading_hierarchy(out)

    return out



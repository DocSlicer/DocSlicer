# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Compile exhibit and footnote regex patterns from YAML."""

# utils/exhibit_patterns.py
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# =========================
# Footnote marker handling
# =========================

# Footnote markers that may appear before/after exhibit numbers
# Includes: asterisk, dagger, double dagger, section, pilcrow, hash, plus, caret,
# various geometric shapes, and other common footnote symbols
FOOTNOTE_MARKERS = "*†‡§¶#+^■●▲▼◆◇○□△▽◊~"

# =========================
# Shared compiled pattern type
# =========================

@dataclass(frozen=True)
class CompiledPattern:
    name: str
    regex: str
    flags: int
    compiled: re.Pattern
    strength: str = "strong"  # "strong" or "weak"
    has_footnote_markers: bool = False  # True if pattern was auto-generated with markers


@dataclass(frozen=True)
class ExhibitPatternConfig:
    row_patterns: Tuple[CompiledPattern, ...]
    header_patterns: Tuple[CompiledPattern, ...]
    anti_patterns: Tuple[CompiledPattern, ...]


# module-level cache
_PATTERN_CACHE: Dict[str, ExhibitPatternConfig] = {}


def _stable_yaml_fingerprint(yaml_obj: Dict[str, Any]) -> str:
    """
    Deterministic fingerprint for caching.
    Assumes yaml_obj is JSON-serializable (plain dict/list/str/int).
    """
    payload = json.dumps(
        yaml_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _augment_pattern_with_footnotes(
    base_regex: str,
    flags: int,
) -> List[Tuple[str, bool]]:
    """
    Generate variations of a regex pattern with optional footnote markers.
    
    Returns list of (regex_string, has_markers) tuples.
    Marker variations come FIRST (checked before base pattern).
    
    Variations:
    1. Embedded after first capture: e.g., "4.25 †Rest" or "4.25† Rest"
    2. Leading markers: [markers]* <pattern>
    3. Trailing markers: <pattern> [markers]*
    4. Both leading and trailing: [markers]* <pattern> [markers]*
    5. Base pattern (no markers) - LAST
    
    Strategy: Insert optional marker groups at key positions in the regex.
    Marker variations are checked first to ensure "4.25 †" matches marker pattern, not base.
    """
    marker_class = f"[{re.escape(FOOTNOTE_MARKERS)}]"
    marker_optional = f"{marker_class}*"
    marker_with_space = rf"(?:\s*{marker_class}+\s*)"
    
    variations = []
    
    # 1. Embedded after first number/code (REQUIRES at least one marker)
    # For numeric patterns like ^\s*\d{1,3}(?:\.\d+)*\s
    # This is the MOST SPECIFIC pattern - check it first
    # Pattern: ^\s* <number> [REQUIRED markers with spaces] <rest>
    # Matches: "4.25 †Description", "4.25† Description", "4.25 * Description"
    # Does NOT match: "4.25 Description" (no markers - will match base pattern)
    if r'\d' in base_regex and r'\s' in base_regex:
        # For patterns like: ^\s*\d{1,3}(?:\.\d+)*\s
        # Convert to: ^\s*\d{1,3}(?:\.\d+)*\s*[REQUIRED markers]+\s*
        if r'(?:\.\d+)*\s' in base_regex:
            # REQUIRES at least one marker (NOT optional)
            embedded_pattern = base_regex.replace(
                r'(?:\.\d+)*\s',
                rf'(?:\.\d+)*{marker_with_space}',
                1
            )
            variations.append((embedded_pattern, True))
    
    # 2. Leading markers (REQUIRES at least one marker): ^\s* [markers]+ <rest>
    if base_regex.startswith(r'^\s*'):
        # Insert after ^\s* - REQUIRES at least one marker (NOT optional)
        leading_pattern = base_regex.replace(
            r'^\s*',
            rf'^\s*{marker_with_space}',
            1
        )
        variations.append((leading_pattern, True))
    
    # 3. Trailing markers (REQUIRES at least one marker): <pattern> [markers]+
    # Add before any end-of-line anchor or end of pattern
    if base_regex.endswith('$'):
        trailing_pattern = base_regex[:-1] + rf'{marker_with_space}$'
    else:
        trailing_pattern = base_regex + rf'{marker_with_space}'
    variations.append((trailing_pattern, True))
    
    # 4. Both leading and trailing (REQUIRES markers in at least one place)
    if base_regex.startswith(r'^\s*'):
        if base_regex.endswith('$'):
            both_pattern = base_regex.replace(
                r'^\s*',
                rf'^\s*{marker_with_space}',
                1
            )[:-1] + rf'{marker_with_space}$'
        else:
            both_pattern = base_regex.replace(
                r'^\s*',
                rf'^\s*{marker_with_space}',
                1
            ) + rf'{marker_with_space}'
        variations.append((both_pattern, True))
    
    # 5. Base pattern (no markers) - checked LAST
    variations.append((base_regex, False))
    
    return variations


def _compile_group(raw_patterns: List[Dict[str, Any]], augment_with_footnotes: bool = False) -> Tuple[CompiledPattern, ...]:
    """
    Compile a group of patterns from YAML.
    
    Args:
        raw_patterns: List of pattern dicts from YAML
        augment_with_footnotes: If True, auto-generate footnote marker variations
    """
    compiled: List[CompiledPattern] = []

    for p in raw_patterns or []:
        name = str(p.get("name", "unknown")).strip()
        regex = str(p.get("regex", "")).strip()
        flags_s = str(p.get("flags", "") or "").strip().lower()
        strength = str(p.get("strength", "strong")).strip().lower()

        flags = 0
        if "i" in flags_s:
            flags |= re.IGNORECASE
        if "m" in flags_s:
            flags |= re.MULTILINE
        if "s" in flags_s:
            flags |= re.DOTALL
        if "x" in flags_s:
            flags |= re.VERBOSE

        if not regex:
            continue

        if augment_with_footnotes:
            # Generate base + footnote variations
            variations = _augment_pattern_with_footnotes(regex, flags)
            
            for var_regex, has_markers in variations:
                # Patterns with footnote markers are automatically strong
                var_strength = "strong" if has_markers else strength
                var_name = f"{name}_with_markers" if has_markers else name
                
                try:
                    compiled.append(
                        CompiledPattern(
                            name=var_name,
                            regex=var_regex,
                            flags=flags,
                            compiled=re.compile(var_regex, flags),
                            strength=var_strength,
                            has_footnote_markers=has_markers,
                        )
                    )
                except re.error as e:
                    # Skip invalid patterns (edge cases in augmentation)
                    continue
        else:
            # No augmentation - compile as-is
            compiled.append(
                CompiledPattern(
                    name=name,
                    regex=regex,
                    flags=flags,
                    compiled=re.compile(regex, flags),
                    strength=strength,
                    has_footnote_markers=False,
                )
            )

    return tuple(compiled)


def load_and_compile_exhibit_patterns(yaml_obj: Dict[str, Any]) -> ExhibitPatternConfig:
    """
    Load and compile exhibit patterns from YAML.
    
    - Row patterns are augmented with footnote marker variations
    - Header patterns are NOT augmented (footnotes don't appear in headers)
    - Anti-patterns are NOT augmented
    
    Args:
        yaml_obj: Parsed YAML dict
        
    Returns:
        Compiled pattern config with row_patterns, header_patterns, anti_patterns
    """
    key = _stable_yaml_fingerprint(yaml_obj)
    cached = _PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    row_raw = yaml_obj.get("exhibit_row_patterns", []) or []
    header_raw = yaml_obj.get("exhibit_heading_patterns", []) or []
    anti_raw = yaml_obj.get("exhibit_row_anti_patterns", []) or []

    # Augment row patterns with footnote marker variations
    row_patterns = _compile_group(row_raw, augment_with_footnotes=True)
    
    # Header patterns: no augmentation (footnotes don't appear in headers)
    header_patterns = _compile_group(header_raw, augment_with_footnotes=False)
    
    # Anti-patterns: no augmentation
    anti_patterns = _compile_group(anti_raw, augment_with_footnotes=False)

    cfg = ExhibitPatternConfig(
        row_patterns=row_patterns,
        header_patterns=header_patterns,
        anti_patterns=anti_patterns,
    )
    _PATTERN_CACHE[key] = cfg
    return cfg

# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Compile hierarchy-marker regex patterns from YAML into profile-keyed configs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Tuple


# =========================
# Dataclasses
# =========================

@dataclass(frozen=True)
class CompiledPattern:
    hierarchy_type: str
    regex: str
    flags: int
    compiled: re.Pattern
    applies_to_profiles: FrozenSet[str]
    index: int  # preserves YAML order for deterministic matching


@dataclass(frozen=True)
class HierarchyTypePatternConfig:
    version: str
    defaults: Mapping[str, Any]
    patterns: Tuple[CompiledPattern, ...]
    patterns_by_profile: Mapping[str, Tuple[CompiledPattern, ...]]


# =========================
# Cache
# =========================

_CACHE: Dict[str, HierarchyTypePatternConfig] = {}


# =========================
# Helpers
# =========================

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


def _parse_flags(flags_s: str) -> int:
    flags_s = (flags_s or "").strip().lower()
    flags = 0
    if "i" in flags_s:
        flags |= re.IGNORECASE
    if "m" in flags_s:
        flags |= re.MULTILINE
    if "s" in flags_s:
        flags |= re.DOTALL
    return flags


# =========================
# Public API
# =========================

def load_and_compile_hierarchy_type_patterns(yaml_obj: Dict[str, Any]) -> HierarchyTypePatternConfig:
    """
    yaml_obj is parsed YAML (dict) passed in by the orchestrator.

    Expected shape (simplified):

      version: "1.0.0"
      defaults: {...}

      hierarchy_type_rules:
        - hierarchy_type: item
          applies_to_profiles: [finance, legal, government]
          patterns:
            - regex: '^\s*item\s+'
              flags: 'i'
    """
    key = _stable_yaml_fingerprint(yaml_obj)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    version = str(yaml_obj.get("version", "") or "")
    defaults = dict(yaml_obj.get("defaults") or {})

    raw_rules = yaml_obj.get("hierarchy_rules") or []
    compiled_patterns: List[CompiledPattern] = []

    pattern_index = 0  # global index respecting YAML order

    for rule in raw_rules:
        hierarchy_type = str(rule.get("hierarchy_type", "")).strip()
        if not hierarchy_type:
            continue

        applies_raw = rule.get("applies_to_profiles") or []
        applies_to_profiles: FrozenSet[str] = frozenset(
            p.strip().lower()
            for p in applies_raw
            if isinstance(p, str) and p.strip()
        )

        patterns_cfg = rule.get("patterns") or []
        for p in patterns_cfg:
            regex = str(p.get("regex", "")).strip()
            if not regex:
                continue

            flags = _parse_flags(str(p.get("flags", "") or ""))
            compiled = re.compile(regex, flags)

            compiled_patterns.append(
                CompiledPattern(
                    hierarchy_type=hierarchy_type,
                    regex=regex,
                    flags=flags,
                    compiled=compiled,
                    applies_to_profiles=applies_to_profiles,
                    index=pattern_index,
                )
            )
            pattern_index += 1

    # Sort once by index to fix order
    compiled_patterns.sort(key=lambda cp: cp.index)

    # Build per-profile index
    patterns_by_profile: Dict[str, List[CompiledPattern]] = {}
    for cp in compiled_patterns:
        targets = cp.applies_to_profiles or frozenset(["generic"])
        for profile in targets:
            patterns_by_profile.setdefault(profile, []).append(cp)

    frozen_patterns_by_profile: Dict[str, Tuple[CompiledPattern, ...]] = {
        profile: tuple(sorted(pats, key=lambda cp: cp.index))
        for profile, pats in patterns_by_profile.items()
    }

    cfg = HierarchyTypePatternConfig(
        version=version,
        defaults=defaults,
        patterns=tuple(compiled_patterns),
        patterns_by_profile=frozen_patterns_by_profile,
    )

    _CACHE[key] = cfg
    return cfg


def get_patterns_for_profile(cfg: HierarchyTypePatternConfig, profile: str) -> Tuple[CompiledPattern, ...]:
    """
    Returns the compiled patterns for a given DocumentProfile.

    Falls back to:
      - 'generic' if present
      - else all patterns
    """
    profile = (profile or "").strip().lower()
    if profile in cfg.patterns_by_profile:
        return cfg.patterns_by_profile[profile]
    return cfg.patterns_by_profile.get("generic", cfg.patterns)

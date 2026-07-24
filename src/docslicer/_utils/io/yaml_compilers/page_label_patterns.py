# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Compile page-label regex patterns from YAML into a cached config."""

# utils/page_label_patterns.py
from __future__ import annotations

import re
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# =========================
# Compile Page Label YAML Cache
# =========================

@dataclass(frozen=True)
class CompiledPattern:
    name: str
    regex: str
    flags: int
    compiled: re.Pattern


@dataclass(frozen=True)
class PageLabelPatternConfig:
    max_length: int
    patterns: Tuple[CompiledPattern, ...]


# module-level cache
_PATTERN_CACHE: Dict[str, PageLabelPatternConfig] = {}


def _stable_yaml_fingerprint(yaml_obj: Dict[str, Any]) -> str:
    """
    Deterministic fingerprint for caching.
    Assumes yaml_obj is JSON-serializable (plain dict/list/str/int).
    """
    payload = json.dumps(yaml_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_and_compile_patterns(yaml_obj: Dict[str, Any]) -> PageLabelPatternConfig:
    """
    yaml_obj is already parsed in orchestrator (dict) and passed in.
    Compiles regexes once per unique yaml content.
    """
    key = _stable_yaml_fingerprint(yaml_obj)
    cached = _PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    max_length = int(yaml_obj.get("max_length", 8))
    raw_patterns = yaml_obj.get("patterns", []) or []

    compiled: List[CompiledPattern] = []
    for p in raw_patterns:
        name = str(p.get("name", "unknown")).strip()
        regex = str(p.get("regex", "")).strip()
        flags_s = str(p.get("flags", "") or "").strip().lower()

        flags = 0
        if "i" in flags_s:
            flags |= re.IGNORECASE
        if "m" in flags_s:
            flags |= re.MULTILINE
        if "s" in flags_s:
            flags |= re.DOTALL

        compiled.append(CompiledPattern(name=name, regex=regex, flags=flags, compiled=re.compile(regex, flags)))

    cfg = PageLabelPatternConfig(max_length=max_length, patterns=tuple(compiled))
    _PATTERN_CACHE[key] = cfg
    return cfg
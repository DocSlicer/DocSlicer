# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Centralized YAML configuration loader for parsing orchestrators.

This module provides a single function to load and compile all YAML pattern 
configurations used by both PDF and HTML parsing pipelines.
"""

from importlib.resources import files
from typing import Dict, Tuple, Any
import yaml

from .yaml_compilers.page_label_patterns import load_and_compile_patterns, PageLabelPatternConfig
from .yaml_compilers.hierarchy_type_patterns import load_and_compile_hierarchy_type_patterns
from .yaml_compilers.exhibit_patterns import load_and_compile_exhibit_patterns

_CONFIG = files("docslicer") / "config"


def load_page_label_config() -> PageLabelPatternConfig:
    """Load and compile just the page label patterns (used by the docx pipeline,
    which doesn't need the hierarchy_type/exhibit patterns loaded by load_yamls)."""
    with (_CONFIG / "page_label_patterns.yaml").open("r", encoding="utf-8") as f:
        page_label_dict = yaml.safe_load(f)
    return load_and_compile_patterns(page_label_dict)


def load_yamls() -> Tuple[Dict[str, Any], Any, Any, Any]:
    """
    Load and compile all YAML configs once at startup.

    Returns:
        Tuple containing:
        - page_label_dict: Raw dict for JS generation in box extractor
        - page_label_config: Compiled patterns for Python use (assign_page_labels)
        - hierarchy_type_pattern_config: Compiled hierarchy type patterns
        - exhibit_pattern_config: Compiled exhibit patterns
    """
    # ==== Page Label Patterns ====
    with (_CONFIG / "page_label_patterns.yaml").open("r", encoding="utf-8") as f:
        page_label_dict = yaml.safe_load(f)
    page_label_config = load_and_compile_patterns(page_label_dict)

    # ==== Hierarchy Type Patterns ====
    with (_CONFIG / "hierarchy_type_patterns.yaml").open("r", encoding="utf-8") as f:
        hierarchy_type_dict = yaml.safe_load(f)
    hierarchy_type_pattern_config = load_and_compile_hierarchy_type_patterns(hierarchy_type_dict)

    # ==== Exhibit Patterns ====
    with (_CONFIG / "exhibit_patterns.yaml").open("r", encoding="utf-8") as f:
        exhibit_dict = yaml.safe_load(f)
    exhibit_pattern_config = load_and_compile_exhibit_patterns(exhibit_dict)

    return page_label_dict, page_label_config, hierarchy_type_pattern_config, exhibit_pattern_config

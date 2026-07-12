# config.py - Pipeline configuration loader
from pathlib import Path
from typing import Dict, Any
import yaml

from .yaml_compilers.page_label_patterns import load_and_compile_patterns, PageLabelPatternConfig

# Config directory contains YAML pattern files (src/docslicer/config)
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_page_label_config() -> Dict[str, Any]:
    """Load page label patterns from YAML config (raw dict)."""
    config_path = CONFIG_DIR / "page_label_patterns.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_compiled_page_label_config() -> PageLabelPatternConfig:
    """Load and compile page label patterns for use in step_02."""
    raw_config = load_page_label_config()
    return load_and_compile_patterns(raw_config)


def load_chunk_tag_rules() -> list:
    """Load chunk tagging rules from YAML config."""
    config_path = CONFIG_DIR / "chunk_tags.yaml"
    if not config_path.exists():
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("chunk_tags", cfg) if isinstance(cfg, dict) else cfg


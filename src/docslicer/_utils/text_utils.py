from __future__ import annotations
import pandas as pd


#==================================================================================================
# BULLET TOKEN DETECTION
#==================================================================================================

_BULLET_TOKENS = {
    "-", "–", "—",          # en/em dash / hyphen – you can remove "-" if too aggressive
    "•", "·",               # classic bullets
    "■", "▪", "",          # squares / special bullet glyphs
    "…",                    # ellipsis, if used as leader
    "+", "☒", "☐",
    "○", "◦", "►", "▸", "‣", "⁃",
    "✓", "✔", "✗", "✘", "✖", "✕",
}

def is_bullet_token(text: object) -> bool:
    if text is None:
        return False
    if isinstance(text, float) and pd.isna(text):
        return False
    t = str(text).strip()
    return t in _BULLET_TOKENS


#==================================================================================================
# BULLET TOKEN DETECTION
#==================================================================================================
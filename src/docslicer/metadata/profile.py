"""Document profile detection — keyword-based classification."""
from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd


def _load_profile_keywords() -> pd.DataFrame:
    """
    Load document profile keywords from CSV.

    Returns:
        DataFrame with columns: profile, type, keyword, score, language
    """
    from importlib.resources import files

    try:
        with (files("docslicer") / "config" / "doc_profile_keywords.csv").open("rb") as f:
            df = pd.read_csv(f)
        # Ensure keyword column is lowercase for case-insensitive matching
        if 'keyword' in df.columns:
            df['keyword_lower'] = df['keyword'].str.lower().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def detect_document_profile(df_lines: pd.DataFrame, language: Optional[str] = None) -> Optional[str]:
    """
    Detect document profile based on keyword matching.

    Algorithm:
    1. Load profile keywords CSV
    2. For each line in document, check if it contains any keywords (substring match)
    3. Count hits per profile (weighted by score if available)
    4. Return profile with most hits

    Args:
        df_lines: DataFrame with 'text' column
        language: Document language (e.g., "en") - filters keywords by language if provided

    Returns:
        Profile name (e.g., "finance", "legal", "academic", "government") or None
    """
    if df_lines.empty or 'text' not in df_lines.columns:
        return None

    # Load keywords
    keywords_df = _load_profile_keywords()
    if keywords_df.empty or 'keyword_lower' not in keywords_df.columns:
        return None

    # Filter by language if provided
    if language and 'language' in keywords_df.columns:
        keywords_df = keywords_df[
            (keywords_df['language'] == language) |
            (keywords_df['language'].isna())
        ]

    if keywords_df.empty:
        return None

    # Concatenate all document text (convert to lowercase for matching)
    all_text = ' '.join(df_lines['text'].dropna().astype(str)).lower()

    if not all_text.strip():
        return None

    # Count hits per profile
    profile_scores = {}

    for idx, row in keywords_df.iterrows():
        keyword = row['keyword_lower']
        profile = row['profile']
        score = row.get('score', 1) if pd.notna(row.get('score')) else 1

        if not keyword or not profile:
            continue

        # Substring match (case-insensitive, already lowercase)
        if keyword in all_text:
            if profile not in profile_scores:
                profile_scores[profile] = 0
            profile_scores[profile] += score

    # Return profile with highest score
    if profile_scores:
        max_profile = max(profile_scores.items(), key=lambda x: x[1])
        return max_profile[0]

    return None


def add_profile_info(
    doc_meta: Dict[str, Any],
    df_lines: Optional[pd.DataFrame] = None,
) -> None:
    """
    Detect and add document profile information.

    Populates:
    - profile: Document profile (e.g., "finance", "legal", "academic", "government")

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        df_lines: DataFrame with text lines

    Modifies:
        doc_meta dictionary in-place
    """
    profile = None

    if df_lines is not None:
        # Use resolved language if available
        language = doc_meta.get("language")
        if language == "unknown":
            language = None

        profile = detect_document_profile(df_lines, language=language)

    doc_meta["profile"] = profile

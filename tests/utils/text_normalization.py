from __future__ import annotations

import re

__all__ = ["normalize_en", "normalize_ja", "normalize_text"]


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_en(text: str, *, keep_apostrophes: bool = False) -> str:
    """
    Normalize English transcripts:
    - lower-case
    - drop punctuation (optionally keep apostrophes, which are replaced with spaces)
    - collapse whitespace
    """

    cleaned = text.lower().strip()
    if keep_apostrophes:
        cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
        cleaned = cleaned.replace("'", " ")
    else:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return _collapse_spaces(cleaned)


def normalize_ja(text: str, *, strip_punctuation: bool = True) -> str:
    """
    Normalize Japanese transcripts:
    - trim surrounding whitespace
    - optionally strip common punctuation
    - remove all whitespace (half/full-width)
    """

    cleaned = text.strip()
    if strip_punctuation:
        cleaned = (
            cleaned.replace("。", "")
            .replace("、", "")
            .replace("．", "")
            .replace("，", "")
        )
    # Remove any spacing (half- or full-width)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def normalize_text(text: str, *, lang: str) -> str:
    """Dispatch to language-specific normalization."""

    if lang == "en":
        return normalize_en(text)
    if lang == "ja":
        return normalize_ja(text)
    raise ValueError(f"Unsupported language for normalization: {lang}")

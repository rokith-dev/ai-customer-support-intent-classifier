from __future__ import annotations

import re
from typing import Optional


_whitespace_re = re.compile(r"\s+")
_non_word_re = re.compile(r"[^\w\s]")


def clean_text(text: Optional[str]) -> str:
    """Normalize support ticket text for downstream modeling."""
    if not text:
        return ""

    normalized = text.lower().strip()
    normalized = _non_word_re.sub(" ", normalized)
    normalized = _whitespace_re.sub(" ", normalized)
    return normalized.strip()

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for ticket text."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

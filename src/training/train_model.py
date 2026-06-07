from __future__ import annotations

import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.tfidf_features import build_vectorizer


def build_pipeline() -> Pipeline:
    """Build the baseline text classification pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def save_pipeline(model: Pipeline, output_path: str | Path) -> None:
    """Persist a trained pipeline to disk."""
    joblib.dump(model, output_path)

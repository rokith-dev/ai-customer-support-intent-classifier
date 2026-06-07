from __future__ import annotations

import joblib
from pathlib import Path
from typing import Iterable


def load_model(model_path: str | Path):
    """Load a trained intent classifier pipeline."""
    return joblib.load(model_path)


def predict_intent(model, tickets: Iterable[str]):
    """Predict intents for a collection of ticket texts."""
    return model.predict(list(tickets))

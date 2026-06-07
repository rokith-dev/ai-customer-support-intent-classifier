from __future__ import annotations

from sklearn.metrics import classification_report, accuracy_score


def evaluate_predictions(y_true, y_pred) -> dict:
    """Return common classification metrics for intent models."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }

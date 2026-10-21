"""Metrics, threshold selection, and cross-validation helpers."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score


def regression_report(y_true, y_pred) -> dict:
    """Return common regression metrics as plain Python floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_dict(y_true, y_pred, y_proba) -> dict:
    """Return classification metrics, including ROC-AUC."""
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Choose the best threshold on a grid from 0.05 to 0.95."""
    scorers = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    if metric not in scorers:
        raise ValueError("metric must be f1, precision, or recall")
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    best_t, best_score = 0.05, -1.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        pred = (y_proba >= threshold).astype(int)
        score = scorers[metric](y_true, pred, zero_division=0)
        if score > best_score:
            best_t, best_score = float(round(threshold, 2)), float(score)
    return best_t


def cross_validate_model(pipe, X, y, scoring: str, cv: int = 5) -> dict:
    """Cross-validate and return mean, standard deviation, and fold scores."""
    scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
    # sklearn uses negative scores for losses so that higher is always better.
    # Convert them back to positive error values for a report people can read.
    if scoring.startswith("neg_"):
        scores = -scores
    return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": [float(x) for x in scores]}

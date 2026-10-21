"""Small scikit-learn-compatible models and model factories."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression trained with plain gradient descent."""
    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y")
        weights = np.zeros(X.shape[1], dtype=float)
        bias = 0.0
        self.loss_history_ = []
        count = len(y)
        for _ in range(self.n_iter):
            errors = X @ weights + bias - y
            self.loss_history_.append(float(np.mean(errors ** 2) / 2))
            weights -= self.lr * (X.T @ errors / count)
            bias -= self.lr * float(errors.mean())
        self.coef_ = weights
        self.intercept_ = float(bias)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("Call fit before predict")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Create a preprocessing + regression pipeline."""
    if kind not in {"dummy", "linear", "ridge", "poly"}:
        raise ValueError("kind must be dummy, linear, ridge, or poly")
    prep = build_preprocessor(degree=degree if kind == "poly" else 1)
    if kind == "dummy":
        estimator = DummyRegressor(strategy="mean")
    elif kind == "linear":
        estimator = LinearRegression()
    else:
        estimator = Ridge(alpha=alpha)
    return Pipeline([("prep", prep), ("model", estimator)])


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Create a preprocessing + classifier pipeline with probabilities."""
    if kind not in {"dummy", "logistic"}:
        raise ValueError("kind must be dummy or logistic")
    prep = build_preprocessor()
    if kind == "dummy":
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        estimator = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    return Pipeline([("prep", prep), ("model", estimator)])

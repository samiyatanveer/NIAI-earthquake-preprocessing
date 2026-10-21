"""Train and evaluate an earthquake regression or classification model."""
import argparse
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split

from . import config
from .evaluate import best_threshold, classification_report_dict, cross_validate_model, regression_report
from .etl import DB_PATH
from .models import make_classifier, make_regressor


def load_data(db_path=None):
    """Load the ETL output from SQLite."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query('SELECT * FROM quakes', connection)


def main() -> None:
    """Parse arguments, train baseline and requested model, and save reports."""
    parser = argparse.ArgumentParser(description="Train earthquake models")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    data = load_data(args.db)
    if data.empty:
        raise RuntimeError(f"No rows found in {args.db}. Run python -m quakes.etl first.")
    target = "mag" if args.task == "regression" else config.TARGET
    if target not in data:
        raise ValueError(f"Required target column {target!r} is missing")
    data = data.dropna(subset=[target]).copy()
    X = data.drop(columns=["mag", config.TARGET], errors="ignore")
    y = data[target]
    stratify = y if args.task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=stratify
    )

    config.ROOT.joinpath("models").mkdir(parents=True, exist_ok=True)
    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    report = {"task": args.task, "model": args.model or ("linear" if args.task == "regression" else "logistic"), "train_rows": len(X_train), "test_rows": len(X_test)}

    if args.task == "regression":
        kind = args.model or "linear"
        baseline = make_regressor("dummy")
        model = make_regressor(kind, degree=args.degree, alpha=args.alpha)
        baseline.fit(X_train, y_train)
        model.fit(X_train, y_train)
        report["baseline_test"] = regression_report(y_test, baseline.predict(X_test))
        report["model_test"] = regression_report(y_test, model.predict(X_test))
        report["cross_validation"] = cross_validate_model(model, X_train, y_train, "neg_root_mean_squared_error")
    else:
        kind = args.model or "logistic"
        weights = "balanced" if args.balanced else None
        baseline = make_classifier("dummy", class_weight=weights)
        model = make_classifier(kind, class_weight=weights)
        baseline.fit(X_train, y_train)
        model.fit(X_train, y_train)
        report["baseline_test"] = classification_report_dict(y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1])
        # Threshold is selected only from out-of-fold predictions on training data.
        cv_proba = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
        threshold = best_threshold(y_train, cv_proba, metric="f1")
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        report["threshold"] = threshold
        report["model_test"] = classification_report_dict(y_test, test_pred, test_proba)
        report["cross_validation"] = cross_validate_model(model, X_train, y_train, "roc_auc")

    model_path = config.ROOT / "models" / f"{args.task}_{kind}.joblib"
    report_path = config.ROOT / "reports" / f"{args.task}_{kind}_report.json"
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

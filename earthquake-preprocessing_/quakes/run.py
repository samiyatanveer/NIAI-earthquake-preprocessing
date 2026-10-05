"""Task 5: run the whole pipeline end to end.

Steps:
1. Fetch all_week, falling back to config.FALLBACK_PATH if needed.
2. Merge data/stream/*.jsonl rows, then dedupe_latest.
3. Clean, engineer features, and drop leaky columns.
4. Split data, fit_transform on train only, transform test.
5. Print the report and save the processed outputs.
"""

import glob
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from . import config
from .cleaning import clean, dedupe_latest
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data


def to_dense(matrix) -> np.ndarray:
    """Return a dense NumPy array from a dense or sparse matrix."""

    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())

    return np.asarray(matrix)


def load_stream_rows() -> pd.DataFrame:
    """Load all JSONL files from the stream directory."""

    paths = glob.glob(str(config.STREAM_DIR / "*.jsonl"))

    if not paths:
        return pd.DataFrame()

    frames = [pd.read_json(path, lines=True) for path in paths]

    return pd.concat(frames, ignore_index=True)


def load_raw_data() -> pd.DataFrame:
    """Fetch all_week or load the fallback GeoJSON."""

    try:
        payload = fetch_feed("all_week")
        return geojson_to_df(payload)

    except Exception as exc:
        print(f"Network fetch failed: {exc}")
        print(f"Using fallback: {config.FALLBACK_PATH}")

        with open(config.FALLBACK_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)

        return geojson_to_df(payload)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the required features and target."""

    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)

    df["region"] = group_rare(df["region"], top_k=15)

    df = add_target(df)
    df = drop_leaky_columns(df)

    return df


def main() -> None:
    """Run the complete earthquake preprocessing pipeline."""

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch raw earthquake data
    raw_df = load_raw_data()
    raw_rows = len(raw_df)

    # 2. Load stream data
    stream_df = load_stream_rows()
    stream_rows = len(stream_df)

    updated_changed = 0

    if not stream_df.empty:
        raw_df = pd.concat([raw_df, stream_df], ignore_index=True)

        rows_before = len(raw_df)
        raw_df = dedupe_latest(raw_df)
        updated_changed = rows_before - len(raw_df)

    # 3. Clean and engineer features
    cleaned_df = clean(raw_df)
    cleaned_rows = len(cleaned_df)

    df = engineer_features(cleaned_df)

    # 4. Split before preprocessing
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Fit only on training data, transform test data only
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    # 6. Convert to dense arrays (handles sparse or dense output)
    X_train_array = to_dense(X_train_processed)
    X_test_array = to_dense(X_test_processed)

    train_df = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index,
    )
    train_df[config.TARGET] = y_train

    test_df = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index,
    )
    test_df[config.TARGET] = y_test

    # 7. Save outputs
    train_path = config.PROCESSED_DIR / "train.csv"
    test_path = config.PROCESSED_DIR / "test.csv"
    preprocessor_path = config.PROCESSED_DIR / "preprocessor.joblib"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    # 8. Print report
    print()
    print("=== Earthquake preprocessing report ===")
    print(f"Raw rows: {raw_rows}")
    print(f"Rows after cleaning: {cleaned_rows}")
    print(f"Stream rows merged: {stream_rows}")
    print(f"Events whose updated changed: {updated_changed}")
    print(f"Train shape: {X_train_array.shape}")
    print(f"Test shape: {X_test_array.shape}")
    print(f"Train positive-class rate: {y_train.mean():.4f}")
    print(f"Test positive-class rate: {y_test.mean():.4f}")
    print()
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {preprocessor_path}")


if __name__ == "__main__":
    main()
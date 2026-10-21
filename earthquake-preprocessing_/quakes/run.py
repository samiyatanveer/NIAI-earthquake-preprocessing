"""Run the full earthquake preprocessing pipeline and save model-ready files."""
import glob
import json
import joblib
import numpy as np
import pandas as pd
from . import config
from .cleaning import clean, dedupe_latest
from .features import add_location_features, add_quality_features, add_target, add_time_features, drop_leaky_columns, group_rare
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data


def load_stream_rows() -> pd.DataFrame:
    """Read all stream JSONL files; return empty data when none exist."""
    paths = sorted(glob.glob(str(config.STREAM_DIR / "*.jsonl")))
    frames = [pd.read_json(path, lines=True) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_raw_data() -> pd.DataFrame:
    """Fetch all_week, using the instructor fallback file if fetch fails."""
    try:
        return geojson_to_df(fetch_feed("all_week"))
    except Exception as exc:
        if not config.FALLBACK_PATH.exists():
            raise RuntimeError(f"USGS fetch failed and fallback file is missing: {config.FALLBACK_PATH}") from exc
        print(f"Network fetch failed ({exc}); loading fallback data.")
        with open(config.FALLBACK_PATH, encoding="utf-8") as file:
            return geojson_to_df(json.load(file))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and target, group rare regions, and remove leakage."""
    out = add_time_features(df)
    out = add_quality_features(out)
    out = add_location_features(out)
    out["region"] = group_rare(out["region"], top_k=15)
    out = add_target(out)
    return drop_leaky_columns(out)


def main() -> None:
    """Run each stage in order and report saved outputs."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_data()
    raw_rows = len(raw_df)

    stream_df = load_stream_rows()
    stream_rows = len(stream_df)
    events_updated = 0
    if not stream_df.empty:
        combined = pd.concat([raw_df, stream_df], ignore_index=True)
        # Count IDs whose latest timestamp differs from the original batch timestamp.
        before = combined.sort_values("updated").drop_duplicates("id", keep="first").set_index("id")["updated"]
        latest = combined.sort_values("updated").drop_duplicates("id", keep="last").set_index("id")["updated"]
        common_ids = before.index.intersection(latest.index)
        events_updated = int((before.loc[common_ids] != latest.loc[common_ids]).sum())
        raw_df = dedupe_latest(combined)

    cleaned = clean(raw_df)
    cleaned_rows = len(cleaned)
    data = engineer_features(cleaned)
    X_train, X_test, y_train, y_test = split_data(data)

    preprocessor = build_preprocessor()
    train_array = preprocessor.fit_transform(X_train)
    test_array = preprocessor.transform(X_test)
    # The preprocessor is configured to produce dense arrays for simple CSV output.
    train_array = np.asarray(train_array)
    test_array = np.asarray(test_array)
    names = preprocessor.get_feature_names_out()
    train_out = pd.DataFrame(train_array, columns=names, index=X_train.index)
    test_out = pd.DataFrame(test_array, columns=names, index=X_test.index)
    train_out[config.TARGET] = y_train
    test_out[config.TARGET] = y_test

    train_path = config.PROCESSED_DIR / "train.csv"
    test_path = config.PROCESSED_DIR / "test.csv"
    model_path = config.PROCESSED_DIR / "preprocessor.joblib"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    joblib.dump(preprocessor, model_path)

    print("\n=== Earthquake preprocessing report ===")
    print(f"Raw rows: {raw_rows}")
    print(f"Rows after cleaning: {cleaned_rows}")
    print(f"Stream rows merged: {stream_rows}")
    print(f"Events whose updated changed: {events_updated}")
    print(f"Train shape: {train_array.shape}")
    print(f"Test shape: {test_array.shape}")
    print(f"Train positive-class rate: {y_train.mean():.4f}")
    print(f"Test positive-class rate: {y_test.mean():.4f}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()

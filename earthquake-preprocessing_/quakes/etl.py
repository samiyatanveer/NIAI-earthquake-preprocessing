"""Extract, transform, and load earthquake data with simple logging."""
import argparse
import json
import logging
import sqlite3

import pandas as pd

from . import config
from .cleaning import clean
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch a USGS feed, falling back to the saved GeoJSON when offline."""
    try:
        return geojson_to_df(fetch_feed(feed))
    except Exception as exc:
        log.warning("Could not fetch USGS feed %s: %s", feed, exc)
        if not config.FALLBACK_PATH.exists():
            raise RuntimeError(
                f"Fetch failed and fallback file is missing: {config.FALLBACK_PATH}"
            ) from exc
        with config.FALLBACK_PATH.open(encoding="utf-8") as file:
            return geojson_to_df(json.load(file))


def read_stream_files() -> pd.DataFrame:
    """Read and concatenate all JSONL files in data/stream."""
    paths = sorted(config.STREAM_DIR.glob("*.jsonl"))
    frames = [pd.read_json(path, lines=True) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and feature steps as the Day 1 pipeline."""
    data = clean(raw)
    data = add_time_features(data)
    data = add_quality_features(data)
    data = add_location_features(data)
    data["region"] = group_rare(data["region"], top_k=15)
    data = add_target(data)
    return drop_leaky_columns(data)


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Insert rows by ID, replacing old versions without creating duplicates."""
    if not table.replace("_", "").isalnum():
        raise ValueError("table must contain only letters, numbers, and underscores")
    path = str(db_path)
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype("string")
    with sqlite3.connect(path) as connection:
        if data.empty:
            return
        if "id" not in data.columns:
            raise ValueError("DataFrame must contain an 'id' column for upsert")
        # Delete only IDs present in this batch, then append their latest rows.
        try:
            existing = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 0', connection)
            table_exists = True
        except Exception:
            table_exists = False
        if table_exists:
            ids = data["id"].dropna().astype(str).unique().tolist()
            if ids:
                connection.execute(f'CREATE TEMP TABLE incoming_ids (id TEXT PRIMARY KEY)')
                connection.executemany("INSERT OR IGNORE INTO incoming_ids(id) VALUES (?)", [(x,) for x in ids])
                connection.execute(f'DELETE FROM "{table}" WHERE CAST(id AS TEXT) IN (SELECT id FROM incoming_ids)')
                connection.execute("DROP TABLE incoming_ids")
        data.to_sql(table, connection, if_exists="append", index=False)


def run_etl(mode: str = "batch") -> None:
    """Run extract, optional stream merge, transform, and SQLite load."""
    if mode not in {"batch", "merge"}:
        raise ValueError("mode must be 'batch' or 'merge'")
    raw = extract()
    log.info("Extracted %d feed rows", len(raw))
    if mode == "merge":
        stream = read_stream_files()
        log.info("Read %d stream rows", len(stream))
        if not stream.empty:
            raw = pd.concat([raw, stream], ignore_index=True)
    log.info("Rows before transform: %d", len(raw))
    data = transform(raw)
    log.info("Rows after transform: %d", len(data))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load_sqlite(data, DB_PATH)
    log.info("Loaded %d rows into %s", len(data), DB_PATH)


def main() -> None:
    """Set up file/console logging and run the selected mode."""
    parser = argparse.ArgumentParser(description="Earthquake ETL")
    parser.add_argument("--mode", choices=["batch", "merge"], default="batch")
    args = parser.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    run_etl(args.mode)


if __name__ == "__main__":
    main()

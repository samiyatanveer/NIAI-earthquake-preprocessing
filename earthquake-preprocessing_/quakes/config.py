from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

NUMERIC = [
    "lon",
    "lat",
    "depth_km",
    "hour",
    "dayofweek",
    "update_lag_hours",
    "is_reviewed",
    "nst_missing",
    "abs_lat",
    "is_shallow",
]

NOMINAL = [
    "region",
    "magType",
    "type",
]

LEAKY = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
]
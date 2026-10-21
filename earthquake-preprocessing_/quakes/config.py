"""Shared paths and feature lists for the earthquake pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# Columns measured before prediction and safe to use as numeric features.
NUMERIC = [
    "lon", "lat", "depth_km", "hour", "dayofweek",
    "update_lag_hours", "is_reviewed", "nst_missing",
    "abs_lat", "is_shallow",
]

# Categorical features that will be one-hot encoded.
NOMINAL = ["region", "magType", "type"]

# These columns reveal or are derived from earthquake magnitude/impact.
LEAKY = ["title", "sig", "mmi", "cdi", "felt", "alert"]

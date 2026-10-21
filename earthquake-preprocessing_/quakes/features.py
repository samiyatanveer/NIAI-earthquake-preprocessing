"""Feature engineering functions; each returns a new DataFrame."""
import pandas as pd
from . import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add UTC hour and weekday features."""
    out = df.copy()
    times = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["hour"] = times.dt.hour
    out["dayofweek"] = times.dt.dayofweek
    return out


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add update delay, review status, and missing-station indicator."""
    out = df.copy()
    time = pd.to_datetime(out["time"], utc=True, errors="coerce")
    updated = pd.to_datetime(out["updated"], utc=True, errors="coerce")
    out["update_lag_hours"] = (updated - time).dt.total_seconds() / 3600
    out["is_reviewed"] = out["status"].astype("string").str.strip().str.lower().eq("reviewed").astype(int)
    out["nst_missing"] = out["nst"].isna().astype(int)
    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute latitude and shallow-earthquake indicator."""
    out = df.copy()
    out["abs_lat"] = pd.to_numeric(out["lat"], errors="coerce").abs()
    out["is_shallow"] = (pd.to_numeric(out["depth_km"], errors="coerce") < 70).astype(int)
    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top-k common values and map all other values to Other."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    common = s.value_counts(dropna=False).head(top_k).index
    return s.where(s.isin(common), "Other").fillna("Unknown")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: magnitude at least 4.5 is a big quake."""
    out = df.copy()
    out[config.TARGET] = (pd.to_numeric(out["mag"], errors="coerce") >= 4.5).astype(int)
    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove configured columns that leak magnitude/impact information."""
    return df.drop(columns=config.LEAKY, errors="ignore").copy()

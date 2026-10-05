"""Task 4: feature engineering."""

import pandas as pd

from . import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour and dayofweek from UTC time."""

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    return df


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add update lag, review status, and missing nst indicator."""

    df = df.copy()

    df["update_lag_hours"] = (
        df["updated"] - df["time"]
    ).dt.total_seconds() / 3600

    df["is_reviewed"] = (
        df["status"]
        .astype("string")
        .str.lower()
        .eq("reviewed")
        .astype(int)
    )

    df["nst_missing"] = df["nst"].isna().astype(int)

    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute latitude and shallow-earthquake indicator."""

    df = df.copy()

    df["abs_lat"] = df["lat"].abs()
    df["is_shallow"] = (df["depth_km"] < 70).astype(int)

    return df


def group_rare(
    s: pd.Series,
    top_k: int = 15,
) -> pd.Series:
    """Keep top_k frequent values and replace others with Other."""

    top_values = s.value_counts().nlargest(top_k).index

    return s.where(s.isin(top_values), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add big_quake target."""

    df = df.copy()

    df["big_quake"] = (df["mag"] >= 4.5).astype(int)

    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns listed in config.LEAKY."""

    return df.drop(
        columns=config.LEAKY,
        errors="ignore",
    )
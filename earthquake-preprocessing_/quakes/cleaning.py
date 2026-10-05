"""Task 3: cleaning."""

import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["updated"] = pd.to_datetime(
        df["updated"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""

    df = df.copy()

    return (
        df.sort_values("updated")
        .drop_duplicates("id", keep="last")
        .reset_index(drop=True)
    )


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' and keep only 'earthquake'."""

    df = df.copy()

    df["type"] = (
        df["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return df[df["type"] == "earthquake"].copy()


def extract_region(place: pd.Series) -> pd.Series:
    """Extract text after the last comma."""

    return (
        place.fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[-1]
        .str.strip()
    )


def iqr_outlier_mask(
    s: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """True where a value lies outside Q1 - k*IQR and Q3 + k*IQR."""

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (s < lower) | (s > upper)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""

    return df.dropna(subset=["mag"]).copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps and add a region column."""

    df = epoch_ms_to_datetime(df)
    df = dedupe_latest(df)
    df = keep_earthquakes(df)
    df = drop_missing_target(df)

    df["region"] = extract_region(df["place"])

    return df
"""Small, explicit cleaning functions for USGS earthquake records."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time and updated from epoch milliseconds to UTC datetimes."""
    out = df.copy()
    for column in ("time", "updated"):
        if column in out.columns:
            # Already-converted timestamps are accepted as well.
            if pd.api.types.is_numeric_dtype(out[column]):
                out[column] = pd.to_datetime(out[column], unit="ms", utc=True, errors="coerce")
            else:
                out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest updated row for each event id."""
    if df.empty or "id" not in df.columns or "updated" not in df.columns:
        return df.copy().reset_index(drop=True)
    return df.sort_values("updated").drop_duplicates("id", keep="last").reset_index(drop=True)


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose normalised type is earthquake."""
    out = df.copy()
    if "type" not in out.columns:
        return out.iloc[0:0].copy()
    out["type"] = out["type"].astype("string").str.strip().str.lower()
    return out.loc[out["type"] == "earthquake"].copy()


def extract_region(place: pd.Series) -> pd.Series:
    """Return the text after the final comma; keep whole place if no comma."""
    values = place.fillna("Unknown").astype(str).str.strip()
    return values.str.split(",").str[-1].str.strip().replace("", "Unknown")


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Mark values outside the IQR fences; missing values are not outliers."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    spread = q3 - q1
    return ((s < q1 - k * spread) | (s > q3 + k * spread)).fillna(False)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records without magnitude, which is needed to make the target."""
    return df.dropna(subset=["mag"]).copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps, keep latest earthquakes, and add region."""
    out = epoch_ms_to_datetime(df)
    out = dedupe_latest(out)
    out = keep_earthquakes(out)
    out = drop_missing_target(out)
    if "place" in out.columns:
        out["region"] = extract_region(out["place"])
    else:
        out["region"] = "Unknown"
    return out.reset_index(drop=True)

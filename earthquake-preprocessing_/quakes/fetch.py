"""Fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """Download a feed and return parsed JSON, with clear errors."""
    import requests

    response = requests.get(f"{BASE}/{feed}.geojson", timeout=15)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"USGS feed '{feed}' did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"USGS feed '{feed}' returned an unexpected JSON value")
    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Create one row per feature with properties and coordinates."""
    rows = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        rows.append({
            "id": feature.get("id"),
            **properties,
            "lon": coordinates[0] if len(coordinates) > 0 else None,
            "lat": coordinates[1] if len(coordinates) > 1 else None,
            "depth_km": coordinates[2] if len(coordinates) > 2 else None,
        })
    return pd.DataFrame(rows)

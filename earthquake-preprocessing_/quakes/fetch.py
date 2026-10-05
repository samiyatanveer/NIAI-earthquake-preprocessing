"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    ...
    import requests

    url = f"{BASE}/{feed}.geojson"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(
            f"USGS feed did not return valid JSON for feed '{feed}'."
        ) from exc

def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """
    rows = []

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        coordinates = feature.get("geometry", {}).get("coordinates", [])

        lon = coordinates[0] if len(coordinates) > 0 else None
        lat = coordinates[1] if len(coordinates) > 1 else None
        depth = coordinates[2] if len(coordinates) > 2 else None

        row = {
            "id": feature.get("id"),
            **properties,
            "lon": lon,
            "lat": lat,
            "depth_km": depth,
        }

        rows.append(row)

    return pd.DataFrame(rows)
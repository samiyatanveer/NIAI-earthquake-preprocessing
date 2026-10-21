"""Poll a USGS feed and append newly observed event versions as JSONL."""
import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd

from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Keep rows with a new (id, updated) pair and update seen in place."""
    keep = []
    for index, row in df.iterrows():
        key = (row.get("id"), row.get("updated"))
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return df.loc[keep].copy().reset_index(drop=True)


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll until duration expires; a failed poll does not stop the loop."""
    if interval_s < 0 or duration_s < 0:
        raise ValueError("interval_s and duration_s must be non-negative")
    seen = set()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        try:
            df = geojson_to_df(fetch_feed(feed))
            new_rows = filter_unseen(df, seen)
            if not new_rows.empty:
                with open(out_path, "a", encoding="utf-8") as file:
                    for record in new_rows.to_dict(orient="records"):
                        file.write(json.dumps(record, default=str) + "\n")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(new_rows)} new / {len(df)} fetched")
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Poll failed: {exc}")
        remaining = duration_s - (time.monotonic() - start)
        if remaining <= 0:
            break
        time.sleep(min(interval_s, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect new USGS earthquake event versions")
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default="data/stream/stream.jsonl")
    args = parser.parse_args()
    run_stream(args.feed, args.interval, args.duration, args.out)


if __name__ == "__main__":
    main()

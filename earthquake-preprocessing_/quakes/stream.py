"""Task 2: simulate streaming by polling a feed on a timer."""

import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd

from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in seen.

    Add the new pairs to seen (modify the set in place).
    """
    unseen_rows = []

    for _, row in df.iterrows():
        key = (row["id"], row["updated"])

        if key not in seen:
            unseen_rows.append(row)
            seen.add(key)

    return pd.DataFrame(unseen_rows, columns=df.columns)


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll feed, keep unseen rows, and append them as JSONL."""

    seen = set()
    start_time = time.time()

    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    while time.time() - start_time < duration_s:
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)

            new_rows = filter_unseen(df, seen)

            if not new_rows.empty:
                with open(out_path, "a", encoding="utf-8") as file:
                    for record in new_rows.to_dict(orient="records"):
                        file.write(json.dumps(record, default=str) + "\n")

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] "
                f"{len(new_rows)} new / {len(df)} fetched"
            )

        except Exception as exc:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Poll failed: {exc}")

        elapsed = time.time() - start_time
        remaining = duration_s - elapsed

        if remaining <= 0:
            break

        time.sleep(min(interval_s, remaining))


def main() -> None:
    """Parse command-line arguments and run the stream."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument(
        "--out",
        default="data/stream/stream.jsonl",
    )

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
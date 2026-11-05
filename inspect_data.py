
import json
import io
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import zstandard as zstd

DATA_DIR = Path(__file__).resolve().parent / "reddit" / "subreddits24"

FILES = [
    "MachineLearning_submissions.zst",
    "learnmachinelearning_submissions.zst",
    "deeplearning_submissions.zst",
    "LocalLLaMA_submissions.zst",
]

for filename in FILES:
    path = DATA_DIR / filename
    years = Counter()
    total = 0
    usable = 0
    earliest = None
    latest = None

    print(f"\n{'=' * 55}")
    print(filename)

    with path.open("rb") as file:
        with zstd.ZstdDecompressor().stream_reader(file) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as lines:
                for line in lines:
                    try:
                        post = json.loads(line)
                        timestamp = float(post["created_utc"])
                        date = datetime.fromtimestamp(
                            timestamp, tz=timezone.utc
                        )
                    except (ValueError, KeyError, TypeError, OverflowError):
                        continue

                    total += 1
                    years[date.year] += 1

                    earliest = min(earliest or date, date)
                    latest = max(latest or date, date)

                    title = post.get("title") or ""
                    body = post.get("selftext") or ""

                    if body in ("[removed]", "[deleted]"):
                        body = ""

                    if len((title + " " + body).strip()) >= 50:
                        usable += 1

    print(f"Total posts: {total:,}")
    print(f"Posts with >= 50 characters: {usable:,}")
    print(f"Earliest: {earliest}")
    print(f"Latest: {latest}")
    print("Posts per year:")

    for year, count in sorted(years.items()):
        print(f"  {year}: {count:,}")

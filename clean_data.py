
import csv
import html
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd


# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "reddit" / "subreddits24"
OUTPUT_DIR = BASE_DIR / "processed"

FILES = {
    "MachineLearning": "MachineLearning_submissions.zst",
    "learnmachinelearning": "learnmachinelearning_submissions.zst",
    "deeplearning": "deeplearning_submissions.zst",
    "LocalLLaMA": "LocalLLaMA_submissions.zst",
}

START_YEAR = 2021
END_YEAR = 2024
MIN_CHARS = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)
SPACE_RE = re.compile(r"\s+")

REMOVED_VALUES = {
    "[deleted]",
    "[removed]",
    "[unavailable]",
    "[removed by reddit]",
}


# ---------------- CLEANING ----------------

def normalize_text(value):
    """Basic cleaning without destroying linguistic information."""
    if not isinstance(value, str):
        return ""

    value = value.strip()

    if value.lower() in REMOVED_VALUES:
        return ""

    value = html.unescape(value)
    value = value.replace("\u200b", "")
    value = value.replace("\ufeff", "")

    return SPACE_RE.sub(" ", value).strip()


def prepare_nlp_text(title, body):
    """Keep words, punctuation and Reddit-specific markers."""
    combined = f"{title} {body}".strip()

    # Avoid letting random URLs dominate NLP features.
    combined = URL_RE.sub(" URL ", combined)

    return SPACE_RE.sub(" ", combined).strip()


# ---------------- PROCESSING ----------------

stats = defaultdict(Counter)
seen_ids = set()
invalid_records = Counter()

dataset_path = OUTPUT_DIR / "cleaned_posts.csv"
report_path = OUTPUT_DIR / "cleaning_report.csv"

columns = [
    "post_id",
    "subreddit",
    "created_utc",
    "year",
    "month",
    "title",
    "body",
    "text",
    "has_body",
    "is_self",
    "eligible_50",
    "score",
    "num_comments",
    "flair",
    "url",
]

with dataset_path.open(
    "w", encoding="utf-8", newline=""
) as output_file:

    writer = csv.DictWriter(output_file, fieldnames=columns)
    writer.writeheader()

    for subreddit, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        print(f"\nProcessing r/{subreddit}...")

        with path.open("rb") as compressed_file:
            decompressor = zstd.ZstdDecompressor()

            with decompressor.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(
                    reader, encoding="utf-8"
                ) as lines:

                    for line in lines:
                        try:
                            post = json.loads(line)

                            if not isinstance(post, dict):
                                raise ValueError("Not a JSON object")

                            timestamp = float(post["created_utc"])
                            date = datetime.fromtimestamp(
                                timestamp, tz=timezone.utc
                            )

                        except (
                            ValueError,
                            KeyError,
                            TypeError,
                            OverflowError,
                            OSError,
                        ):
                            invalid_records[subreddit] += 1
                            continue

                        year = date.year

                        if not START_YEAR <= year <= END_YEAR:
                            continue

                        key = (subreddit, year)
                        stats[key]["raw"] += 1

                        post_id = str(post.get("id") or "").strip()

                        if not post_id:
                            stats[key]["missing_id"] += 1
                            continue

                        if post_id in seen_ids:
                            stats[key]["duplicate_id"] += 1
                            continue

                        seen_ids.add(post_id)

                        title = normalize_text(post.get("title"))
                        body = normalize_text(post.get("selftext"))

                        # Text excluding URLs, used to assess content.
                        meaningful = URL_RE.sub(
                            " ", f"{title} {body}"
                        )
                        meaningful = SPACE_RE.sub(
                            " ", meaningful
                        ).strip()

                        if not meaningful:
                            stats[key]["empty_or_url_only"] += 1
                            continue

                        text = prepare_nlp_text(title, body)

                        eligible_50 = int(
                            len(meaningful) >= MIN_CHARS
                        )

                        stats[key]["kept"] += 1

                        if eligible_50:
                            stats[key]["eligible_50"] += 1

                        if not body:
                            stats[key]["title_only"] += 1

                        writer.writerow({
                            "post_id": post_id,
                            "subreddit": subreddit,
                            "created_utc": int(timestamp),
                            "year": year,
                            "month": date.month,
                            "title": title,
                            "body": body,
                            "text": text,
                            "has_body": int(bool(body)),
                            "is_self": post.get("is_self", ""),
                            "eligible_50": eligible_50,
                            "score": post.get("score", ""),
                            "num_comments": post.get(
                                "num_comments", ""
                            ),
                            "flair": post.get(
                                "link_flair_text", ""
                            ),
                            "url": post.get("url") or "",
                        })

        total_kept = sum(
            stats[(subreddit, year)]["kept"]
            for year in range(START_YEAR, END_YEAR + 1)
        )

        print(f"  Kept: {total_kept:,}")


# ---------------- REPORT ----------------

report_columns = [
    "subreddit",
    "year",
    "raw",
    "kept",
    "eligible_50",
    "title_only",
    "empty_or_url_only",
    "duplicate_id",
    "missing_id",
    "retention_pct",
]

with report_path.open(
    "w", encoding="utf-8", newline=""
) as report_file:

    writer = csv.DictWriter(
        report_file, fieldnames=report_columns
    )
    writer.writeheader()

    for subreddit in FILES:
        for year in range(START_YEAR, END_YEAR + 1):
            s = stats[(subreddit, year)]

            if s["raw"] == 0:
                continue

            writer.writerow({
                "subreddit": subreddit,
                "year": year,
                "raw": s["raw"],
                "kept": s["kept"],
                "eligible_50": s["eligible_50"],
                "title_only": s["title_only"],
                "empty_or_url_only": s["empty_or_url_only"],
                "duplicate_id": s["duplicate_id"],
                "missing_id": s["missing_id"],
                "retention_pct": round(
                    100 * s["kept"] / s["raw"], 2
                ),
            })


total_raw = sum(s["raw"] for s in stats.values())
total_kept = sum(s["kept"] for s in stats.values())
total_eligible = sum(
    s["eligible_50"] for s in stats.values()
)

print("\n" + "=" * 55)
print("CLEANING COMPLETE")
print(f"Raw posts (2021-2024): {total_raw:,}")
print(f"Kept posts: {total_kept:,}")
print(f"Posts with >= 50 chars: {total_eligible:,}")
print(f"Invalid records: {sum(invalid_records.values()):,}")
print(f"\nDataset: {dataset_path}")
print(f"Report:  {report_path}")


import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "cleaned_posts.csv"
OUTPUT = BASE / "processed" / "audit_report.csv"

stats = defaultdict(Counter)
lengths = defaultdict(list)
seen_texts = set()

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

with INPUT.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        key = (row["subreddit"], row["year"])
        s = stats[key]

        title = row["title"]
        body = row["body"]
        text = row["text"]

        s["total"] += 1
        lengths[key].append(len(text))

        if not body.strip():
            s["title_only"] += 1

        if row["eligible_50"] == "1":
            s["eligible_50"] += 1

        if row["is_self"].lower() == "false":
            s["link_post"] += 1

        # Diagnostic only: identical title + body.
        # Do not automatically delete these posts.
        fingerprint = normalize(title) + "\0" + normalize(body)

        if fingerprint in seen_texts:
            s["repeated_text"] += 1
        else:
            seen_texts.add(fingerprint)

columns = [
    "subreddit", "year", "total",
    "title_only_pct", "eligible_50_pct",
    "link_post_pct", "repeated_text",
    "median_text_length"
]

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for key in sorted(stats):
        subreddit, year = key
        s = stats[key]
        n = s["total"]

        result = {
            "subreddit": subreddit,
            "year": year,
            "total": n,
            "title_only_pct": round(100 * s["title_only"] / n, 2),
            "eligible_50_pct": round(100 * s["eligible_50"] / n, 2),
            "link_post_pct": round(100 * s["link_post"] / n, 2),
            "repeated_text": s["repeated_text"],
            "median_text_length": median(lengths[key]),
        }

        writer.writerow(result)
        print(result)

print(f"\nAudit saved to: {OUTPUT}")

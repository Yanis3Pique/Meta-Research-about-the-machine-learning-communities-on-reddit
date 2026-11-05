
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "processed" / "cleaned_posts.csv"


def fingerprint(row):
    title = re.sub(r"\s+", " ", row["title"].casefold()).strip()
    body = re.sub(r"\s+", " ", row["body"].casefold()).strip()
    content = f"{title}\0{body}".encode("utf-8")
    return hashlib.blake2b(content, digest_size=16).digest()


counts = Counter()

# First pass: count identical title + body combinations.
with DATA.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        counts[fingerprint(row)] += 1

repeated = {key: n for key, n in counts.items() if n > 1}

print(f"Repeated text groups: {len(repeated):,}")
print(f"Extra repeated posts: {sum(n - 1 for n in repeated.values()):,}")

top = [key for key, n in counts.most_common(15) if n > 1]
top_set = set(top)

examples = {}
locations = defaultdict(Counter)

# Second pass: inspect the most frequent repeated texts.
with DATA.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        key = fingerprint(row)

        if key not in top_set:
            continue

        locations[key][
            f"{row['subreddit']} / {row['year']}"
        ] += 1

        if key not in examples:
            examples[key] = {
                "title": row["title"][:180],
                "body": row["body"][:120],
                "is_self": row["is_self"],
            }

for index, key in enumerate(top, start=1):
    example = examples[key]

    print(f"\n{'=' * 60}")
    print(f"#{index} | Occurrences: {counts[key]}")
    print(f"Title: {example['title']}")
    print(f"Body: {example['body']}")
    print(f"Is self post: {example['is_self']}")
    print("Where:", dict(locations[key]))

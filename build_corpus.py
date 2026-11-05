
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent / "processed"
INPUT = BASE / "cleaned_posts.csv"
OUTPUT = BASE / "modeling_corpus.csv"
REPORT = BASE / "corpus_filter_report.csv"

PLACEHOLDERS = {
    "[deleted by user]",
    "[deleted]",
    "[removed]",
    "[image processing failed]",
    "[unavailable]",
    "[removed by reddit]",
}

GENERIC_EMPTY = {
    "help",
    "need help",
    "test",
    "[d]",
    "[p]",
    "[r]",
}

# Patterns manually identified in the duplicate audit.
SPAM_FRAGMENTS = (
    "kid and his mom cctv video",
    "susanna gibson",
    "ais from ai dungeon 2 to sexy to funny and one based wholly on reddit!",
    "notas de az",
    "notasdeaz.blogspot.com",
)

MEGATHREAD = re.compile(
    r"^\[d\]\s*simple questions thread\b",
    re.IGNORECASE,
)

def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def rejection_reason(row):
    title = normalize(row["title"])
    body = normalize(row["body"])

    if title in PLACEHOLDERS and not body:
        return "placeholder"

    if title in GENERIC_EMPTY and not body:
        return "generic_empty"

    if MEGATHREAD.search(title):
        return "recurring_megathread"

    if any(fragment in title for fragment in SPAM_FRAGMENTS):
        return "identified_spam"

    if not title and not body:
        return "empty"

    return None


def duplicate_key(row):
    subreddit = row["subreddit"]
    year = row["year"]
    title = normalize(row["title"])
    body = normalize(row["body"])
    url = row["url"].strip()

    is_self = row["is_self"].strip().lower()

    # Self-post: exact text duplication within a community/year.
    if is_self == "true":
        if body:
            return (subreddit, year, "self", title, body)

        # Identical titles alone do not prove duplication.
        return None

    # Link-post: same title AND same URL.
    if is_self == "false" and url:
        return (subreddit, year, "link", title, url)

    # Without a reliable URL/type, don't assume duplication.
    return None


seen = set()
stats = defaultdict(Counter)

with INPUT.open(encoding="utf-8", newline="") as source:
    reader = csv.DictReader(source)

    if "url" not in reader.fieldnames:
        raise ValueError(
            "Missing 'url' column. Update and rerun clean_data.py."
        )

    with OUTPUT.open(
        "w", encoding="utf-8", newline=""
    ) as target:

        writer = csv.DictWriter(
            target, fieldnames=reader.fieldnames
        )
        writer.writeheader()

        for row in reader:
            group = (row["subreddit"], row["year"])
            stats[group]["input"] += 1

            reason = rejection_reason(row)

            if reason:
                stats[group][reason] += 1
                continue

            key = duplicate_key(row)

            if key is not None:
                if key in seen:
                    stats[group]["duplicate"] += 1
                    continue

                seen.add(key)

            writer.writerow(row)
            stats[group]["kept"] += 1


report_columns = [
    "subreddit", "year", "input", "kept",
    "placeholder", "generic_empty",
    "recurring_megathread", "identified_spam",
    "empty", "duplicate",
]

with REPORT.open(
    "w", encoding="utf-8", newline=""
) as file:

    writer = csv.DictWriter(
        file, fieldnames=report_columns
    )
    writer.writeheader()

    for (subreddit, year), counts in sorted(stats.items()):
        writer.writerow({
            "subreddit": subreddit,
            "year": year,
            **{
                column: counts[column]
                for column in report_columns[2:]
            },
        })


totals = Counter()

for counts in stats.values():
    totals.update(counts)

print("\n" + "=" * 55)
print("FINAL CORPUS")
print(f"Input posts: {totals['input']:,}")
print(f"Kept posts: {totals['kept']:,}")
print(f"Removed posts: {totals['input'] - totals['kept']:,}")

for reason in report_columns[4:]:
    print(f"  {reason}: {totals[reason]:,}")

print(f"\nCorpus: {OUTPUT}")
print(f"Report: {REPORT}")

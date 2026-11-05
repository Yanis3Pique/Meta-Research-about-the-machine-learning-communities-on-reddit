
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "modeling_corpus.csv"

df = pd.read_csv(
    INPUT,
    usecols=["subreddit", "year", "title", "body", "url"],
    keep_default_na=False,
)

patterns = {
    "AI Dungeon": r"ai dungeon",
    "Portuguese terms": r"\b(?:leia|notas)\b",
    "Suspicious AZ": r"\baz\b",
}

for label, pattern in patterns.items():
    matches = df[
        df["title"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    ]

    print(f"\n{'=' * 60}")
    print(f"{label}: {len(matches):,} posts")

    print("\nBy subreddit and year:")
    print(
        matches.groupby(["subreddit", "year"])
        .size()
        .to_string()
    )

    print("\nMost repeated titles:")
    for title, count in (
        matches["title"].value_counts().head(10).items()
    ):
        print(f"  {count:>5}x  {title[:160]}")

    print("\nExample URLs:")
    for url in matches["url"].head(5):
        print(f"  {url}")

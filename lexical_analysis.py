
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "modeling_corpus.csv"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

# Read the fields needed for the first NLP experiment.
df = pd.read_csv(
    INPUT,
    usecols=["subreddit", "year", "title"],
    keep_default_na=False,
)

df = df[df["title"].str.strip().ne("")].reset_index(drop=True)

# Remove Reddit post-type tags only for this experiment.
# Original titles remain unchanged in modeling_corpus.csv.
titles = df["title"].str.replace(
    r"^\s*(?:\[[A-Za-z]{1,3}\]\s*)+",
    "",
    regex=True,
)
titles = titles.str.replace(
    r"https?://\S+|www\.\S+",
    " ",
    regex=True,
)

print(f"Documents: {len(df):,}")
print("Building TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=30,
    max_df=0.90,
    max_features=30000,
    sublinear_tf=True,
    dtype=np.float32,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*\b",
)

X = vectorizer.fit_transform(titles)
terms = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(terms):,}")
print(f"TF-IDF matrix shape: {X.shape}")

top_rows = []
shift_rows = []

communities = [
    "MachineLearning",
    "learnmachinelearning",
    "deeplearning",
    "LocalLLaMA",
]

for community in communities:
    print(f"\n{'=' * 60}")
    print(f"r/{community}")

    if community == "LocalLLaMA":
        periods = [
            ("2023", [2023]),
            ("2024", [2024]),
        ]
    else:
        periods = [
            ("2021-2022", [2021, 2022]),
            ("2023-2024", [2023, 2024]),
        ]

    period_data = []

    for period_name, years in periods:
        mask = (
            df["subreddit"].eq(community)
            & df["year"].isin(years)
        ).to_numpy()

        subset = X[mask]
        n = subset.shape[0]

        # Average TF-IDF score.
        mean_tfidf = np.asarray(
            subset.mean(axis=0)
        ).ravel()

        # Document frequency: proportion of posts
        # containing each term at least once.
        doc_counts = np.asarray(
            subset.getnnz(axis=0)
        ).ravel()

        prevalence = doc_counts / n

        period_data.append(
            (period_name, prevalence, doc_counts, n)
        )

        top_indices = np.argsort(mean_tfidf)[-15:][::-1]

        print(f"\n{period_name} ({n:,} posts)")
        print("Top TF-IDF terms:")

        for idx in top_indices:
            term = terms[idx]
            score = float(mean_tfidf[idx])

            print(f"  {term:<30} {score:.5f}")

            top_rows.append({
                "subreddit": community,
                "period": period_name,
                "term": term,
                "mean_tfidf": score,
                "document_frequency_pct": round(
                    100 * prevalence[idx], 3
                ),
            })

    before_name, before, before_counts, n_before = period_data[0]
    after_name, after, after_counts, n_after = period_data[1]

    # Change in percentage points of document prevalence.
    change = (after - before) * 100

    # Avoid ranking terms that are extremely rare
    # within this particular community.
    eligible = (before_counts + after_counts) >= 25
    eligible_indices = np.flatnonzero(eligible)

    rising = eligible_indices[
        np.argsort(change[eligible_indices])[-15:][::-1]
    ]
    falling = eligible_indices[
        np.argsort(change[eligible_indices])[:15]
    ]

    print("\nLargest increases in document prevalence:")

    for idx in rising:
        print(f"  {terms[idx]:<30} {change[idx]:+.2f} pp")

    print("\nLargest decreases in document prevalence:")

    for idx in falling:
        print(f"  {terms[idx]:<30} {change[idx]:+.2f} pp")

    for direction, indices in [
        ("increase", rising),
        ("decrease", falling),
    ]:
        for idx in indices:
            shift_rows.append({
                "subreddit": community,
                "period_before": before_name,
                "period_after": after_name,
                "direction": direction,
                "term": terms[idx],
                "before_pct": round(100 * before[idx], 3),
                "after_pct": round(100 * after[idx], 3),
                "change_pp": round(float(change[idx]), 3),
            })

pd.DataFrame(top_rows).to_csv(
    OUTPUT / "tfidf_top_terms.csv",
    index=False,
)

pd.DataFrame(shift_rows).to_csv(
    OUTPUT / "term_shifts.csv",
    index=False,
)

print("\nAnalysis complete.")
print(f"Results saved in: {OUTPUT}")

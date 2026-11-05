
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "modeling_corpus.csv"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_TOPICS = 15
MAX_TRAIN_PER_GROUP = 3000

print("Loading corpus...")

df = pd.read_csv(
    INPUT,
    usecols=["subreddit", "year", "title"],
    keep_default_na=False,
)

df = df[df["title"].str.strip().ne("")].reset_index(drop=True)

titles = (
    df["title"]
    .str.replace(
        r"^\s*(?:\[[A-Za-z]{1,3}\]\s*)+",
        "",
        regex=True,
    )
    .str.replace(
        r"https?://\S+|www\.\S+",
        " ",
        regex=True,
    )
)

print(f"Total documents: {len(df):,}")

# --------------------------------------------------
# 1. TF-IDF representation
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=20,
    max_df=0.90,
    max_features=20000,
    sublinear_tf=True,
    dtype=np.float32,
    token_pattern=(
        r"(?u)\b[a-zA-Z][a-zA-Z0-9]+"
        r"(?:[-.][a-zA-Z0-9]+)*\b"
    ),
)

print("Building TF-IDF matrix...")

X = vectorizer.fit_transform(titles)
terms = vectorizer.get_feature_names_out()

print(f"TF-IDF shape: {X.shape}")

# --------------------------------------------------
# 2. Balanced training sample
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)
train_indices = []

for (_, _), indices in df.groupby(
    ["subreddit", "year"]
).indices.items():

    sample_size = min(
        MAX_TRAIN_PER_GROUP,
        len(indices),
    )

    selected = rng.choice(
        indices,
        size=sample_size,
        replace=False,
    )

    train_indices.extend(selected)

train_indices = np.array(train_indices)

print(f"Training sample: {len(train_indices):,} documents")

# --------------------------------------------------
# 3. Train NMF
# --------------------------------------------------

model = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    random_state=RANDOM_STATE,
    max_iter=300,
    solver="cd",
)

print("Training NMF...")

model.fit(X[train_indices])

print(f"Reconstruction error: {model.reconstruction_err_:.4f}")

# --------------------------------------------------
# 4. Extract topic keywords
# --------------------------------------------------

topic_rows = []

print("\nDISCOVERED TOPICS")
print("=" * 65)

for topic_id, weights in enumerate(model.components_):
    top_indices = np.argsort(weights)[-12:][::-1]
    keywords = [terms[i] for i in top_indices]

    print(f"Topic {topic_id:02d}: {', '.join(keywords)}")

    topic_rows.append({
        "topic_id": topic_id,
        "keywords": ", ".join(keywords),
    })

pd.DataFrame(topic_rows).to_csv(
    OUTPUT / "nmf_topics.csv",
    index=False,
)

# --------------------------------------------------
# 5. Assign topics to ALL documents
# --------------------------------------------------

print("\nAssigning topics to full corpus...")

W = model.transform(X)

topic_ids = np.argmax(W, axis=1)

# Documents with no recognized vocabulary
# must not be assigned an arbitrary topic.
zero_weight = W.sum(axis=1) == 0
topic_ids[zero_weight] = -1

df["topic_id"] = topic_ids

print(f"Unassigned documents: {zero_weight.sum():,}")

# --------------------------------------------------
# 6. Define comparison periods
# --------------------------------------------------

df["period"] = np.where(
    df["subreddit"] == "LocalLLaMA",
    df["year"].astype(str),
    np.where(
        df["year"] <= 2022,
        "2021-2022",
        "2023-2024",
    ),
)

valid = df[df["topic_id"] >= 0].copy()

shares = (
    valid.groupby(["subreddit", "period"])["topic_id"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("share_pct")
    .reset_index()
)

shares.to_csv(
    OUTPUT / "nmf_topic_shares.csv",
    index=False,
)

# --------------------------------------------------
# 7. Compare topic prevalence across periods
# --------------------------------------------------

shift_rows = []

for community in sorted(df["subreddit"].unique()):

    if community == "LocalLLaMA":
        before_period = "2023"
        after_period = "2024"
    else:
        before_period = "2021-2022"
        after_period = "2023-2024"

    community_shares = shares[
        shares["subreddit"] == community
    ]

    pivot = community_shares.pivot(
        index="topic_id",
        columns="period",
        values="share_pct",
    ).fillna(0)

    before = pivot[before_period]
    after = pivot[after_period]
    change = after - before

    print(f"\n{'=' * 65}")
    print(f"r/{community}")
    print(f"{before_period} -> {after_period}")

    print("\nLargest increases:")
    for topic_id in change.nlargest(5).index:
        print(
            f"  Topic {topic_id:02d}: "
            f"{change[topic_id]:+.2f} pp"
        )

    print("\nLargest decreases:")
    for topic_id in change.nsmallest(5).index:
        print(
            f"  Topic {topic_id:02d}: "
            f"{change[topic_id]:+.2f} pp"
        )

    for topic_id in pivot.index:
        shift_rows.append({
            "subreddit": community,
            "topic_id": int(topic_id),
            "before_period": before_period,
            "after_period": after_period,
            "before_pct": round(float(before[topic_id]), 4),
            "after_pct": round(float(after[topic_id]), 4),
            "change_pp": round(float(change[topic_id]), 4),
        })

pd.DataFrame(shift_rows).to_csv(
    OUTPUT / "nmf_topic_shifts.csv",
    index=False,
)

print("\nNMF analysis complete.")
print(f"Results saved to: {OUTPUT}")

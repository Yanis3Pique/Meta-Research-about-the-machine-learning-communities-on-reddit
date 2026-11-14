"""
A §2.5 — BERTopic as a second topic model (robustness for NMF).

Fits on the same balanced training sample as NMF (10_topics_select.training_sample,
seed 0), then assigns every post. Uses precomputed embeddings from 50_embed.py.

Usage: python 12_topics_bertopic.py [MODEL]
Outputs:
  results/a_bertopic_topics.csv          topic id, size, top terms (aggregate)
  processed/bertopic_doc_topics.parquet  post_id, topic (after outlier reduction), raw topic
  processed/bertopic_model/              saved model (safetensors)
"""

import importlib
import sys
import time

import numpy as np
import pandas as pd

import common

embed_mod = importlib.import_module("50_embed")

MIN_CLUSTER_SIZE = 100
TARGET_TOPICS = 40          # reduce the HDBSCAN topics to roughly this many


def training_sample(corpus, seed=0, per_cell=3000):
    rng = np.random.default_rng(seed)
    idx = []
    for _, group in corpus.groupby(["subreddit", "year"]):
        idx.extend(rng.choice(group.index.values, size=min(per_cell, len(group)), replace=False))
    return np.sort(np.array(idx))


def main():
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    model_name = sys.argv[1] if len(sys.argv) > 1 else embed_mod.DEFAULT_MODEL
    df = pd.read_parquet(embed_mod.EXPORT)
    emb = np.load(embed_mod.embeddings_path(model_name)).astype(np.float32)
    assert len(df) == len(emb)
    texts = df["text"].tolist()

    train = training_sample(df)
    print(f"Training BERTopic on {len(train):,} titles; assigning {len(df):,}")

    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                random_state=common.RANDOM_STATE, low_memory=False)
    hdb = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    vec = CountVectorizer(stop_words=common.stopwords(), ngram_range=(1, 2), min_df=10)

    topic_model = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec,
                           calculate_probabilities=False, verbose=True)
    t0 = time.time()
    topic_model.fit([texts[i] for i in train], embeddings=emb[train])
    n_raw = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
    print(f"Fitted in {time.time() - t0:.0f}s: {n_raw} topics before reduction")

    if n_raw > TARGET_TOPICS:
        topic_model.reduce_topics([texts[i] for i in train], nr_topics=TARGET_TOPICS + 1)

    raw, _ = topic_model.transform(texts, embeddings=emb)
    raw = np.array(raw)
    print(f"Outliers before reduction: {100 * (raw == -1).mean():.1f}%")
    reduced = np.array(topic_model.reduce_outliers(texts, raw, strategy="embeddings",
                                                   embeddings=emb, threshold=0.0))
    print(f"Outliers after reduction:  {100 * (reduced == -1).mean():.1f}%")

    pd.DataFrame({"post_id": df["post_id"], "topic": reduced, "topic_raw": raw}).to_parquet(
        common.PROCESSED / "bertopic_doc_topics.parquet", index=False)

    info = topic_model.get_topic_info()
    sizes = pd.Series(reduced).value_counts()
    rows = []
    for t in sorted(t for t in info["Topic"] if t != -1):
        terms = [w for w, _ in topic_model.get_topic(t)][:15]
        rows.append({"topic_id": t, "posts": int(sizes.get(t, 0)),
                     "share_pct": 100 * sizes.get(t, 0) / len(reduced),
                     "terms": ", ".join(terms)})
    pd.DataFrame(rows).round(3).to_csv(common.RESULTS / "a_bertopic_topics.csv", index=False)

    topic_model.save(str(common.PROCESSED / "bertopic_model"), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=False)
    print(pd.DataFrame(rows)[["topic_id", "posts", "terms"]].to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()

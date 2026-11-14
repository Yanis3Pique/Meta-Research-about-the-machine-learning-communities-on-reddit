"""
A §2.3 — choose the number of NMF topics.

For each k in K_GRID and each of N_SEEDS balanced training samples:
  - fit NMF on lemmatized titles
  - NPMI coherence of each topic's top-10 terms (reference corpus: all titles)
  - topic diversity (unique share of top-25 terms across topics)
Stability per k: Hungarian matching of topics between every pair of runs on
cosine similarity of topic-term vectors; mean matched cosine.

NMF with nndsvda init is deterministic, so variation between runs comes from
drawing a different balanced training sample per seed.

Outputs:
  results/a_model_selection.csv        one row per (k, seed)
  results/a_model_selection_summary.csv one row per k
  processed/nmf_vectorizer.pkl          fitted TF-IDF vectorizer (reused by 11_)
"""

import itertools
import pickle
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import common

K_GRID = [10, 12, 15, 20, 25, 30]
N_SEEDS = 5
PER_CELL = 3000          # max posts per subreddit-year in a training sample
TOP_COHERENCE = 10
TOP_DIVERSITY = 25
STABILITY_MIN = 0.80


def build_matrix():
    corpus = common.load_corpus(["year"])
    lemmas = common.load_lemmatized_titles()
    assert (corpus["post_id"].values == lemmas["post_id"].values).all()

    vectorizer = TfidfVectorizer(
        stop_words=common.stopwords(),
        ngram_range=(1, 2),
        min_df=20,
        max_df=0.5,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w.]*\w\w[\w.]*\b",
    )
    X = vectorizer.fit_transform(lemmas["lemma"])
    return corpus, vectorizer, X


def training_sample(corpus, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for _, group in corpus.groupby(["subreddit", "year"]):
        size = min(PER_CELL, len(group))
        idx.extend(rng.choice(group.index.values, size=size, replace=False))
    return np.sort(np.array(idx))


def npmi_coherence(top_terms_idx, B, n_docs, doc_freq, terms):
    """Mean NPMI over term pairs; pairs where one term contains the other skipped."""
    scores = []
    for topic in top_terms_idx:
        pair_scores = []
        for i, j in itertools.combinations(topic, 2):
            ti, tj = terms[i], terms[j]
            if set(ti.split()) & set(tj.split()):
                continue
            co = B[:, i].multiply(B[:, j]).sum()
            if co == 0:
                pair_scores.append(-1.0)
                continue
            p_ij = co / n_docs
            p_i, p_j = doc_freq[i] / n_docs, doc_freq[j] / n_docs
            pair_scores.append(np.log(p_ij / (p_i * p_j)) / -np.log(p_ij))
        scores.append(np.mean(pair_scores) if pair_scores else np.nan)
    return np.array(scores)


def stability(components_list):
    """Mean Hungarian-matched cosine between all pairs of runs."""
    normed = [c / np.linalg.norm(c, axis=1, keepdims=True) for c in components_list]
    pair_means, per_topic = [], []
    for a, b in itertools.combinations(range(len(normed)), 2):
        sim = normed[a] @ normed[b].T
        rows, cols = linear_sum_assignment(-sim)
        pair_means.append(sim[rows, cols].mean())
        per_topic.append(sim[rows, cols])
    return float(np.mean(pair_means)), np.vstack(per_topic)


def main():
    t0 = time.time()
    corpus, vectorizer, X = build_matrix()
    terms = vectorizer.get_feature_names_out()
    print(f"Matrix: {X.shape[0]:,} docs x {X.shape[1]:,} terms")

    with open(common.PROCESSED / "nmf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    B = (X > 0).astype(np.float32).tocsc()
    doc_freq = np.asarray(B.sum(axis=0)).ravel()
    n_docs = X.shape[0]

    samples = {seed: training_sample(corpus, seed) for seed in range(N_SEEDS)}

    rows, summary = [], []
    for k in K_GRID:
        comps = []
        for seed, idx in samples.items():
            model = NMF(n_components=k, init="nndsvda", solver="cd",
                        max_iter=400, random_state=common.RANDOM_STATE)
            model.fit(X[idx])
            comps.append(model.components_)

            top_c = np.argsort(model.components_, axis=1)[:, ::-1][:, :TOP_COHERENCE]
            top_d = np.argsort(model.components_, axis=1)[:, ::-1][:, :TOP_DIVERSITY]
            coh = npmi_coherence(top_c, B, n_docs, doc_freq, terms)
            div = len(np.unique(top_d)) / top_d.size

            rows.append({"k": k, "seed": seed, "train_docs": len(idx),
                         "npmi": np.nanmean(coh), "npmi_min": np.nanmin(coh),
                         "diversity": div,
                         "recon_err": model.reconstruction_err_})
            print(f"k={k:2d} seed={seed} npmi={np.nanmean(coh):.3f} "
                  f"div={div:.3f}  ({time.time() - t0:.0f}s)")

        stab, per_topic = stability(comps)
        # Share of topics that reappear (cos >= 0.7) in the average pair of runs.
        reappear = float((per_topic >= 0.7).mean())
        sub = pd.DataFrame(rows)
        sub = sub[sub["k"] == k]
        summary.append({"k": k,
                        "npmi_mean": sub["npmi"].mean(),
                        "npmi_sd": sub["npmi"].std(),
                        "diversity_mean": sub["diversity"].mean(),
                        "stability": stab,
                        "share_topics_reappearing": reappear})
        print(f"  -> k={k}: stability={stab:.3f}, reappearing={reappear:.2f}")

    runs = pd.DataFrame(rows)
    summary = pd.DataFrame(summary)
    summary["score"] = summary["npmi_mean"].rank() + summary["diversity_mean"].rank()
    summary["eligible"] = summary["stability"] >= STABILITY_MIN

    common.RESULTS.mkdir(exist_ok=True)
    runs.round(4).to_csv(common.RESULTS / "a_model_selection.csv", index=False)
    summary.round(4).to_csv(common.RESULTS / "a_model_selection_summary.csv", index=False)

    print("\nSUMMARY")
    print(summary.round(3).to_string(index=False))
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

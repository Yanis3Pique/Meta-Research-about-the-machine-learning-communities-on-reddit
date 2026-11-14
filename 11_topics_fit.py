"""
A §2.3–2.4 — fit the final NMF model and prepare topic labelling.

Usage:  python 11_topics_fit.py K [SEED]
  K     number of topics chosen from results/a_model_selection_summary.csv
  SEED  training sample to use (default: run with median NPMI at that K)

Outputs:
  results/a_topics.csv                 top-15 terms per topic (aggregate, tracked)
  results/a_topic_shares_overall.csv   mean topic weight per subreddit
  processed/doc_topics.parquet         post-level normalised topic weights (not tracked)
  processed/nmf_model.pkl
  processed/topic_labeling.md          20 example titles per topic, for labelling
                                       (post-level text: do NOT commit)
  config/topic_labels.csv              template to fill in (label, family, coherent)
"""

import importlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

import common

select = importlib.import_module("10_topics_select")

N_EXAMPLES = 20
EXAMPLE_MIN_WEIGHT = 0.5


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python 11_topics_fit.py K [SEED]")
    k = int(sys.argv[1])

    runs = pd.read_csv(common.RESULTS / "a_model_selection.csv")
    runs_k = runs[runs["k"] == k].sort_values("npmi")
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    else:
        seed = int(runs_k.iloc[len(runs_k) // 2]["seed"])
    print(f"Fitting k={k} on training sample seed={seed}")

    corpus, vectorizer, X = select.build_matrix()
    terms = vectorizer.get_feature_names_out()
    idx = select.training_sample(corpus, seed)

    model = NMF(n_components=k, init="nndsvda", solver="cd",
                max_iter=400, random_state=common.RANDOM_STATE)
    model.fit(X[idx])
    W = model.transform(X)
    Wn = common.normalize_rows(W)
    unassigned = (W.sum(axis=1) == 0)
    print(f"Unassigned (all-zero) posts: {unassigned.sum():,} "
          f"({100 * unassigned.mean():.2f}%)")

    with open(common.PROCESSED / "nmf_model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "k": k, "seed": seed}, f)

    cols = [f"t{i:02d}" for i in range(k)]
    doc_topics = pd.DataFrame(Wn.astype(np.float32), columns=cols)
    doc_topics.insert(0, "post_id", corpus["post_id"].values)
    doc_topics["unassigned"] = unassigned
    doc_topics["top_weight"] = Wn.max(axis=1)
    doc_topics.to_parquet(common.PROCESSED / "doc_topics.parquet", index=False)

    # Topic terms
    top = np.argsort(model.components_, axis=1)[:, ::-1][:, :15]
    topics = pd.DataFrame({
        "topic_id": range(k),
        "terms": [", ".join(terms[row]) for row in top],
        "mean_weight_all": Wn.mean(axis=0),
    })
    topics.round(5).to_csv(common.RESULTS / "a_topics.csv", index=False)

    shares = (pd.DataFrame(Wn, columns=cols)
              .assign(subreddit=corpus["subreddit"].values)
              .groupby("subreddit").mean().T * 100)
    shares.index.name = "topic"
    shares.round(2).to_csv(common.RESULTS / "a_topic_shares_overall.csv")

    # Labelling sheet: examples stratified across subreddits and years.
    titles = pd.read_csv(common.CORPUS, usecols=["title"], low_memory=False)["title"]
    rng = np.random.default_rng(common.RANDOM_STATE)
    lines = [f"# Topic labelling sheet (k={k}, seed={seed})\n",
             "Post-level text: keep local, do not commit.\n",
             "Fill config/topic_labels.csv: label, family, coherent (yes/partly/no).\n"]
    for t in range(k):
        strong = np.where(Wn[:, t] >= EXAMPLE_MIN_WEIGHT)[0]
        lines.append(f"\n## Topic {t:02d} — {len(strong):,} posts with weight ≥ {EXAMPLE_MIN_WEIGHT}\n")
        lines.append(f"**Terms:** {topics.loc[t, 'terms']}\n")
        share_line = ", ".join(f"{s}: {shares.loc[cols[t], s]:.1f}%" for s in shares.columns)
        lines.append(f"**Share by subreddit:** {share_line}\n")
        if len(strong) == 0:
            continue
        groups = (pd.DataFrame({"i": strong,
                                "g": corpus["subreddit"].values[strong] + corpus["year"].astype(str).values[strong]})
                  .groupby("g")["i"].apply(list))
        # Round-robin over subreddit-year groups, each shuffled once.
        queues = [list(rng.permutation(members)) for members in groups]
        picked = []
        while len(picked) < min(N_EXAMPLES, len(strong)):
            for queue in queues:
                if queue and len(picked) < N_EXAMPLES:
                    picked.append(queue.pop())
        for i in picked:
            lines.append(f"- [{corpus.at[i, 'subreddit']} {corpus.at[i, 'year']}] {titles.iat[i]}")
    (common.PROCESSED / "topic_labeling.md").write_text("\n".join(lines) + "\n")

    labels_path = common.CONFIG / "topic_labels.csv"
    template = pd.DataFrame({"topic_id": range(k), "terms": topics["terms"],
                             "label": "", "family": "", "coherent": ""})
    if labels_path.exists():
        print(f"{labels_path} exists — not overwritten. Template printed only.")
    else:
        template.to_csv(labels_path, index=False)

    print(topics[["topic_id", "terms"]].to_string(index=False))


if __name__ == "__main__":
    main()

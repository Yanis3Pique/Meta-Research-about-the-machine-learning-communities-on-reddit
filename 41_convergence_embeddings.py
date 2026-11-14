"""
D robustness — convergence measured with sentence embeddings (50_embed.py).

Per quarter, equal-size samples of the three historical subreddits:
  1. Distinctiveness: logistic regression on title embeddings predicting the
     subreddit, 5-fold CV macro-F1 (chance 1/3). Stronger than TF-IDF on short
     titles, so a flat line here is stronger evidence of "no convergence".
  2. Centroid similarity: cosine similarity between subreddit mean embeddings
     (bootstrap), minus a shuffled-label baseline.
Both are run on all posts and on posts excluding LLM-heavy titles
(LLM-family NMF weight >= LLM_CUTOFF), the embedding analogue of word masking.

Usage: python 41_convergence_embeddings.py [MODEL]
Outputs: results/d_embed_classifier.csv, results/d_embed_centroids.csv,
         results/figures/fig_d_embeddings.png
"""

import importlib
import re
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import common

embed_mod = importlib.import_module("50_embed")

LLM_CUTOFF = 0.3
N_BOOT = 100
SHORT = {"MachineLearning": "ML", "learnmachinelearning": "learnML", "deeplearning": "DL"}
PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load(model_name):
    corpus = common.load_corpus()
    emb = np.load(embed_mod.embeddings_path(model_name)).astype(np.float32)
    topics = pd.read_parquet(common.PROCESSED / "doc_topics.parquet")
    labels = pd.read_csv(common.CONFIG / "topic_labels.csv")
    llm_ids = labels.loc[labels["family"] == "llm_genai", "topic_id"]
    corpus["llm_weight"] = topics[[f"t{i:02d}" for i in llm_ids]].sum(axis=1).values
    assert len(corpus) == len(emb)
    return corpus, emb


def run(corpus, emb, rng):
    clf_rows, cen_rows = [], []
    hist = corpus[corpus["subreddit"].isin(common.HISTORICAL)]
    for condition, data in [("all posts", hist),
                            ("LLM-heavy posts excluded", hist[hist["llm_weight"] < LLM_CUTOFF])]:
        for q, g in data.groupby("quarter"):
            n = g["subreddit"].value_counts().min()
            sample = g.groupby("subreddit").sample(n, random_state=int(rng.integers(1e9)))
            X, y = emb[sample.index.to_numpy()], sample["subreddit"].to_numpy(dtype=str)
            cv = StratifiedKFold(5, shuffle=True, random_state=common.RANDOM_STATE)
            f1 = cross_val_score(LogisticRegression(max_iter=3000, C=1.0), X, y,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
            clf_rows.append({"condition": condition, "quarter": q, "n_per_sub": n,
                             "macro_f1": f1.mean(), "f1_sd": f1.std()})

            idx = {s: g.index[g["subreddit"] == s].values for s in common.HISTORICAL}
            for a, b in combinations(common.HISTORICAL, 2):
                real, null = [], []
                for _ in range(N_BOOT):
                    ia, ib = rng.choice(idx[a], n), rng.choice(idx[b], n)
                    ca, cb = emb[ia].mean(0), emb[ib].mean(0)
                    real.append(ca @ cb / np.linalg.norm(ca) / np.linalg.norm(cb))
                    pooled = rng.permutation(np.concatenate([ia, ib]))
                    pa, pb = emb[pooled[:n]].mean(0), emb[pooled[n:]].mean(0)
                    null.append(pa @ pb / np.linalg.norm(pa) / np.linalg.norm(pb))
                # Distance above baseline: (1 - cos) - (1 - cos_null)
                cen_rows.append({"condition": condition, "quarter": q,
                                 "pair": f"{SHORT[a]}–{SHORT[b]}", "n_per_sub": n,
                                 "cos": np.mean(real),
                                 "excess_distance": np.mean(null) - np.mean(real)})
    return pd.DataFrame(clf_rows), pd.DataFrame(cen_rows)


def figure(clf, cen, model_name):
    quarters = sorted(clf["quarter"].unique())
    x = {q: i for i, q in enumerate(quarters)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for color, cond in zip(["#2a78d6", "#55554f"], ["all posts", "LLM-heavy posts excluded"]):
        g = clf[clf["condition"] == cond]
        a1.plot(g["quarter"].map(x), g["macro_f1"], color=color, linewidth=2,
                linestyle="-" if cond == "all posts" else ":", label=cond)
    a1.axhline(1 / 3, color="#b5b4ad", linewidth=0.8)
    a1.text(0, 1 / 3, "chance", fontsize=7, color="#8a8a85", va="bottom")
    a1.set_title("Embedding classifier: telling the 3 subreddits apart", fontsize=10, loc="left")
    a1.set_ylabel("Macro-F1 (5-fold CV)", fontsize=8, color="#55554f")
    a1.legend(frameon=False, fontsize=8)

    g = cen[cen["condition"] == "all posts"]
    for color, (pair, gp) in zip(PAIR_COLORS, g.groupby("pair", sort=False)):
        a2.plot(gp["quarter"].map(x), gp["excess_distance"], color=color, linewidth=2, label=pair)
    a2.set_ylim(0, None)
    a2.set_title("Distance between subreddit centroids", fontsize=10, loc="left")
    a2.set_ylabel("Cosine distance above shuffled baseline", fontsize=8, color="#55554f")
    a2.legend(frameon=False, fontsize=8)

    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8a8a85")
        ax.tick_params(colors="#55554f", labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        ticks = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
        ax.set_xticks(ticks, [quarters[i][:4] for i in ticks])
        ax.axvline(quarters.index("2022Q4") + 0.67, color="#55554f", linestyle="--",
                   linewidth=1, zorder=0)
    fig.text(0.99, 0.01, f"Embeddings: {model_name}", ha="right", fontsize=6, color="#8a8a85")
    fig.tight_layout()
    fig.savefig(common.FIGURES / f"fig_d_embeddings_{embed_mod.model_slug(model_name)}.png", dpi=200)
    plt.close(fig)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else embed_mod.DEFAULT_MODEL
    slug = embed_mod.model_slug(model_name)
    rng = np.random.default_rng(common.RANDOM_STATE)
    corpus, emb = load(model_name)
    clf, cen = run(corpus, emb, rng)
    clf.insert(0, "model", slug)
    cen.insert(0, "model", slug)

    def upsert(df, path):
        if path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[old["model"] != slug], df])
        df.round(5).to_csv(path, index=False)

    upsert(clf, common.RESULTS / "d_embed_classifier.csv")
    upsert(cen, common.RESULTS / "d_embed_centroids.csv")
    figure(clf, cen, model_name)

    print(f"Model: {model_name}")
    for cond, g in clf.groupby("condition"):
        tau, p = kendalltau(range(len(g)), g["macro_f1"])
        pre, post = g[g["quarter"] < "2023Q1"]["macro_f1"], g[g["quarter"] >= "2023Q1"]["macro_f1"]
        print(f"  classifier [{cond}]: F1 pre={pre.mean():.3f} post={post.mean():.3f} "
              f"tau={tau:.2f} p={p:.3f}")
    for (cond, pair), g in cen.groupby(["condition", "pair"]):
        tau, p = kendalltau(range(len(g)), g["excess_distance"])
        print(f"  centroids [{cond}] {pair}: tau={tau:.2f} p={p:.3f}")


if __name__ == "__main__":
    main()

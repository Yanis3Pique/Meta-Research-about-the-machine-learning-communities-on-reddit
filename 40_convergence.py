"""
D — cross-community convergence (docs/research_design.md §5).

Per quarter, with equal-size samples per subreddit (n = smallest of the three
historical subreddits that quarter):
  1. Topic divergence: Jensen-Shannon divergence (base 2) between subreddit mean
     topic vectors; bootstrap mean + 95% CI; null = labels shuffled within the
     pair; excess = real - null.
  2. Distinctiveness: TF-IDF + logistic regression predicting subreddit from the
     title, 5-fold CV macro-F1 (chance = 1/3). Repeated with LLM-family
     vocabulary masked: if convergence disappears, it is driven by everyone
     talking about LLMs.
  3. Distance to LocalLLaMA's 2023 profile, and drift from each subreddit's own
     2021H1 profile.

Outputs (aggregate only):
  results/d_jsd_quarterly.csv, results/d_classifier.csv,
  results/d_reference_distance.csv, results/d_trends.csv,
  results/figures/fig_d_convergence.png, results/figures/fig_d_reference.png
"""

import pickle
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

import common

N_BOOT = 200
N_FOLDS = 5
LLM_MASK_TOP = 40          # top terms per LLM-family topic to mask
SAMPLE = "submitted"       # "submitted" or "visible"
PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SUB_COLORS = dict(zip(common.HISTORICAL, PAIR_COLORS))
SHORT = {"MachineLearning": "ML", "learnmachinelearning": "learnML", "deeplearning": "DL"}

LEAK_RE = re.compile(r"\S*(machinelearning|deeplearning|localllama|learnml)\S*")


def jsd(p, q):
    return jensenshannon(p, q, base=2) ** 2


def load():
    corpus = common.load_corpus(["is_removed", "stickied"])
    topics = pd.read_parquet(common.PROCESSED / "doc_topics.parquet")
    lemmas = common.load_lemmatized_titles()
    assert (corpus["post_id"].values == topics["post_id"].values).all()

    corpus["lemma"] = lemmas["lemma"].str.replace(LEAK_RE, " ", regex=True).values
    keep = ~topics["unassigned"].values
    if SAMPLE == "visible":
        keep &= (corpus["is_removed"].eq(0)
                 & pd.to_numeric(corpus["stickied"], errors="coerce").fillna(0).eq(0)).values
    W = topics[[c for c in topics.columns if re.fullmatch(r"t\d+", c)]].values
    return corpus[keep].reset_index(drop=True), W[keep]


def llm_mask_terms():
    with open(common.PROCESSED / "nmf_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    labels = pd.read_csv(common.CONFIG / "topic_labels.csv")
    ids = labels.loc[labels["family"] == "llm_genai", "topic_id"]
    terms = bundle["vectorizer"].get_feature_names_out()
    words = set()
    for t in ids:
        top = np.argsort(bundle["model"].components_[t])[::-1][:LLM_MASK_TOP]
        for term in terms[top]:
            words.update(term.split())
    return sorted(words)


def mask_text(texts, words):
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")
    return texts.str.replace(pattern, " ", regex=True)


# --------------------------------------------------

def topic_divergence(corpus, W, rng):
    rows = []
    for q, g in corpus[corpus["subreddit"].isin(common.HISTORICAL)].groupby("quarter"):
        idx = {s: g.index[g["subreddit"] == s].values for s in common.HISTORICAL}
        n = min(len(v) for v in idx.values())
        for a, b in combinations(common.HISTORICAL, 2):
            real, null = [], []
            for _ in range(N_BOOT):
                ia = rng.choice(idx[a], n, replace=True)
                ib = rng.choice(idx[b], n, replace=True)
                real.append(jsd(W[ia].mean(0), W[ib].mean(0)))
                pooled = rng.permutation(np.concatenate([ia, ib]))
                null.append(jsd(W[pooled[:n]].mean(0), W[pooled[n:]].mean(0)))
            real, null = np.array(real), np.array(null)
            rows.append({"quarter": q, "pair": f"{SHORT[a]}–{SHORT[b]}", "n_per_sub": n,
                         "jsd": real.mean(), "jsd_lo": np.percentile(real, 2.5),
                         "jsd_hi": np.percentile(real, 97.5), "null": null.mean(),
                         "excess": real.mean() - null.mean()})
    return pd.DataFrame(rows)


def distinctiveness(corpus, mask_words, rng):
    stop = common.stopwords()
    rows = []
    hist = corpus[corpus["subreddit"].isin(common.HISTORICAL)]
    for q, g in hist.groupby("quarter"):
        n = g["subreddit"].value_counts().min()
        sample = g.groupby("subreddit").sample(n, random_state=int(rng.integers(1e9)))
        for condition, texts in [("all vocabulary", sample["lemma"]),
                                 ("LLM vocabulary masked", mask_text(sample["lemma"], mask_words))]:
            pipe = make_pipeline(
                TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=3,
                                sublinear_tf=True),
                LogisticRegression(max_iter=2000, C=1.0),
            )
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=common.RANDOM_STATE)
            f1 = cross_val_score(pipe, texts, sample["subreddit"], cv=cv, scoring="f1_macro")
            rows.append({"quarter": q, "condition": condition, "n_per_sub": n,
                         "macro_f1": f1.mean(), "f1_sd": f1.std()})
    return pd.DataFrame(rows)


def reference_distance(corpus, W, rng):
    ref_mask = (corpus["subreddit"] == "LocalLLaMA") & corpus["quarter"].str.startswith("2023")
    ref = W[ref_mask.values].mean(0)
    rows = []
    for s in common.HISTORICAL:
        own_start = W[((corpus["subreddit"] == s)
                       & corpus["quarter"].isin(["2021Q1", "2021Q2"])).values].mean(0)
        for q, g in corpus[corpus["subreddit"] == s].groupby("quarter"):
            n = min(len(g), 1000)
            to_ref, drift = [], []
            for _ in range(N_BOOT):
                i = rng.choice(g.index.values, n, replace=True)
                m = W[i].mean(0)
                to_ref.append(jsd(m, ref))
                drift.append(jsd(m, own_start))
            rows.append({"subreddit": s, "quarter": q,
                         "jsd_to_localllama2023": np.mean(to_ref),
                         "jsd_to_localllama2023_lo": np.percentile(to_ref, 2.5),
                         "jsd_to_localllama2023_hi": np.percentile(to_ref, 97.5),
                         "drift_from_2021h1": np.mean(drift)})
    return pd.DataFrame(rows)


def trends(jsd_q, clf, ref):
    """Kendall's tau over quarters + mean before/after ChatGPT (2022Q4 counted before)."""
    out = []

    def add(series_name, group, quarters, values):
        post = np.array([q >= "2023Q1" for q in quarters])
        tau, p = kendalltau(range(len(values)), values)
        out.append({"measure": series_name, "group": group, "kendall_tau": tau, "p": p,
                    "mean_pre": np.mean(np.array(values)[~post]),
                    "mean_post": np.mean(np.array(values)[post])})

    for pair, g in jsd_q.groupby("pair"):
        add("excess_jsd", pair, g["quarter"].tolist(), g["excess"].tolist())
    for cond, g in clf.groupby("condition"):
        add("macro_f1", cond, g["quarter"].tolist(), g["macro_f1"].tolist())
    for s, g in ref.groupby("subreddit"):
        g = g[g["quarter"] >= "2021Q3"]
        add("jsd_to_localllama2023", s, g["quarter"].tolist(), g["jsd_to_localllama2023"].tolist())
        add("drift_from_2021h1", s, g["quarter"].tolist(), g["drift_from_2021h1"].tolist())
    return pd.DataFrame(out)


# --------------------------------------------------

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def quarter_axis(ax, quarters):
    ticks = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
    ax.set_xticks(ticks, [quarters[i][:4] for i in ticks])
    chatgpt = quarters.index("2022Q4") + 0.67   # 30 Nov within Q4
    ax.axvline(chatgpt, color="#55554f", linestyle="--", linewidth=1, zorder=0)
    return chatgpt


def figures(jsd_q, clf, ref):
    quarters = sorted(jsd_q["quarter"].unique())
    x = {q: i for i, q in enumerate(quarters)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for color, (pair, g) in zip(PAIR_COLORS, jsd_q.groupby("pair", sort=False)):
        xs = g["quarter"].map(x)
        a1.fill_between(xs, g["jsd_lo"] - g["null"], g["jsd_hi"] - g["null"],
                        color=color, alpha=0.15, linewidth=0)
        a1.plot(xs, g["excess"], color=color, linewidth=2, label=pair)
    style(a1)
    c = quarter_axis(a1, quarters)
    a1.text(c, a1.get_ylim()[1], " ChatGPT", fontsize=7, color="#55554f", va="top")
    a1.set_ylim(0, None)
    a1.set_title("Topic divergence between subreddits", fontsize=10, loc="left")
    a1.set_ylabel("JSD above shuffled baseline", fontsize=8, color="#55554f")
    a1.legend(frameon=False, fontsize=8)

    for color, cond in zip(["#2a78d6", "#55554f"], ["all vocabulary", "LLM vocabulary masked"]):
        g = clf[clf["condition"] == cond]
        a2.plot(g["quarter"].map(x), g["macro_f1"], color=color, linewidth=2,
                linestyle="-" if cond == "all vocabulary" else ":", label=cond)
    a2.axhline(1 / 3, color="#b5b4ad", linewidth=0.8)
    a2.text(0, 1 / 3, "chance", fontsize=7, color="#8a8a85", va="bottom")
    style(a2)
    quarter_axis(a2, quarters)
    a2.set_title("How well a classifier tells the 3 subreddits apart", fontsize=10, loc="left")
    a2.set_ylabel("Macro-F1 (5-fold CV)", fontsize=8, color="#55554f")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig_d_convergence.png", dpi=200)
    plt.close(fig)

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for s, g in ref.groupby("subreddit", sort=False):
        xs = g["quarter"].map(x)
        b1.fill_between(xs, g["jsd_to_localllama2023_lo"], g["jsd_to_localllama2023_hi"],
                        color=SUB_COLORS[s], alpha=0.15, linewidth=0)
        b1.plot(xs, g["jsd_to_localllama2023"], color=SUB_COLORS[s], linewidth=2,
                label=f"r/{s}")
        g2 = g[g["quarter"] >= "2021Q3"]
        b2.plot(g2["quarter"].map(x), g2["drift_from_2021h1"], color=SUB_COLORS[s],
                linewidth=2, label=f"r/{s}")
    for ax, title, ylabel in [
        (b1, "Distance to r/LocalLLaMA's 2023 topic profile", "JSD"),
        (b2, "Drift from own 2021 H1 topic profile", "JSD"),
    ]:
        style(ax)
        quarter_axis(ax, quarters)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel, fontsize=8, color="#55554f")
    b1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig_d_reference.png", dpi=200)
    plt.close(fig)


def main():
    common.FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(common.RANDOM_STATE)
    corpus, W = load()
    mask_words = llm_mask_terms()
    print(f"Masked LLM words ({len(mask_words)}): {', '.join(mask_words[:30])} ...")

    jsd_q = topic_divergence(corpus, W, rng)
    clf = distinctiveness(corpus, mask_words, rng)
    ref = reference_distance(corpus, W, rng)
    tr = trends(jsd_q, clf, ref)

    jsd_q.round(5).to_csv(common.RESULTS / "d_jsd_quarterly.csv", index=False)
    clf.round(4).to_csv(common.RESULTS / "d_classifier.csv", index=False)
    ref.round(5).to_csv(common.RESULTS / "d_reference_distance.csv", index=False)
    tr.round(4).to_csv(common.RESULTS / "d_trends.csv", index=False)
    figures(jsd_q, clf, ref)

    pd.set_option("display.width", 200)
    print(tr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

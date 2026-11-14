"""
Distil the LLM annotator (60_llm_annotate.py) into a cheap post-level classifier
and apply it to the whole corpus: a third, post-level measure of LLM/genAI content.

Logistic regression on Qwen3-Embedding-4B title embeddings, trained on the
~2k LLM-labelled titles. 5-fold CV metrics are reported against the annotator;
the threshold is the CV-F1-optimal one (slightly optimistic, report as such).

Outputs:
  processed/llm_pred.parquet            post_id, llm_prob, llm_pred (post-level, not tracked)
  results/a_llm_classifier_cv.csv       CV metrics
  results/a_llm_classifier_monthly.csv  monthly predicted LLM share per subreddit
  results/a_its_three_measures.csv      ITS + placebo for NMF, BERTopic, classifier
  results/figures/fig_a_llm_three_measures.png
"""

import importlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import common

tt = importlib.import_module("13_topics_time")
EMB = "qwen3-embedding-4b"


def main():
    s = pd.read_parquet(common.PROCESSED / "llm_sample.parquet")
    lab = pd.read_json(common.PROCESSED / "llm_labels.jsonl", lines=True)
    df = s.merge(lab[["post_id", "llm_genai"]], on="post_id")
    y = df["llm_genai"].astype(int).values

    E = np.load(common.PROCESSED / f"embeddings_{EMB}.npy", mmap_mode="r")
    X = E[df["row"].values].astype(np.float32)

    clf = LogisticRegressionCV(Cs=8, cv=5, max_iter=3000, scoring="f1", class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=common.RANDOM_STATE)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    thr = max(np.round(np.arange(0.2, 0.81, 0.05), 2), key=lambda t: f1_score(y, prob >= t))
    pred = prob >= thr
    cvm = pd.DataFrame([{"embeddings": EMB, "n_train": len(y), "threshold": thr,
                         "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                         "f1": f1_score(y, pred), "kappa": cohen_kappa_score(y, pred),
                         "pred_share": pred.mean(), "annotator_share": y.mean()}])
    cvm.round(4).to_csv(common.RESULTS / "a_llm_classifier_cv.csv", index=False)
    print(cvm.round(3).to_string(index=False))

    # Fit on all labelled titles, apply to the corpus in chunks
    clf.fit(X, y)
    probs = np.concatenate([clf.predict_proba(E[i:i + 50_000].astype(np.float32))[:, 1]
                            for i in range(0, len(E), 50_000)])
    corpus = common.load_corpus()
    out = pd.DataFrame({"post_id": corpus["post_id"], "llm_prob": probs.astype(np.float32),
                        "llm_pred": probs >= thr})
    out.to_parquet(common.PROCESSED / "llm_pred.parquet", index=False)

    # Monthly share in the same format as 13_topics_time.monthly_prevalence
    corpus["assigned"] = True
    corpus["visible"] = True
    w = pd.DataFrame({"llm_genai": out["llm_pred"].astype(float)})
    clf_prev = tt.monthly_prevalence(corpus, w)
    clf_prev = clf_prev[clf_prev["sample"] == "submitted"]
    clf_prev.round(5).to_csv(common.RESULTS / "a_llm_classifier_monthly.csv", index=False)

    # Three measures side by side
    nmf_prev = pd.read_csv(common.RESULTS / "a_prevalence_monthly.csv")
    nmf_prev = nmf_prev[nmf_prev["sample"] == "submitted"]
    bt_monthly = common.RESULTS / "a_bertopic_prevalence_monthly.csv"
    measures = [("NMF topic family", nmf_prev, "#2a78d6"),
                ("Embedding classifier (LLM-annotator labels)", clf_prev, "#1baf7a")]
    if bt_monthly.exists():
        measures.insert(1, ("BERTopic family", pd.read_csv(bt_monthly), "#eb6834"))

    rows = []
    for name, prev, _ in measures:
        its, placebo = tt.run_its(prev, ["llm_genai"])
        mx = (placebo.assign(a=placebo["beta_level"].abs())
              .groupby("subreddit")["a"].max().rename("max_placebo_abs_level"))
        its = its.set_index("subreddit").join(mx).reset_index()
        its["beats_all_placebos"] = its["beta_level"].abs() > its["max_placebo_abs_level"]
        rows.append(its.assign(measure=name))
    its_all = pd.concat(rows)
    its_all.round(5).to_csv(common.RESULTS / "a_its_three_measures.csv", index=False)

    # Historical subreddits only: the annotator sample contains no r/LocalLLaMA titles.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, sub in zip(axes, common.HISTORICAL):
        for name, prev, color in measures:
            d = prev[(prev["subreddit"] == sub) & (prev["family"] == "llm_genai")
                     & (prev["posts"] >= tt.MIN_POSTS)].sort_values("month")
            ax.plot(pd.to_datetime(d["month"] + "-01"), 100 * d["prevalence"], color=color,
                    linewidth=2, label=name)
        tt.style(ax)
        tt.mark_events(ax)
        ax.set_ylim(0, None)
        ax.set_title(f"r/{sub}", fontsize=10, loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("LLMs & generative AI (%)", fontsize=8, color="#55554f")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(common.FIGURES / "fig_a_llm_three_measures.png", dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(its_all[["measure", "subreddit", "pre_mean_pct", "fitted_last_pct",
                   "counterfactual_last_pct", "gap_pp", "beta_level", "max_placebo_abs_level",
                   "beats_all_placebos"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

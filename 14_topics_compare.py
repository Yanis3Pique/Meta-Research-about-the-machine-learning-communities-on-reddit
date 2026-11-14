"""
A §2.5 — does the BERTopic model tell the same story as NMF?

1. Map each BERTopic topic to the NMF family its posts most often belong to
   (dominant NMF family per post); report the overlap cross-tab.
2. Monthly family shares under BERTopic (share of posts per family).
3. Same ITS + placebo procedure as 13_topics_time.py on both models, side by side.

Inputs: processed/bertopic_doc_topics.parquet, results/a_bertopic_topics.csv,
        processed/doc_topics.parquet, config/topic_labels.csv
Outputs: results/a_bertopic_family_map.csv, results/a_model_agreement.csv,
         results/a_its_by_model.csv, results/figures/fig_a_nmf_vs_bertopic.png
"""

import importlib
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

tt = importlib.import_module("13_topics_time")


def main():
    corpus = common.load_corpus()
    nmf = pd.read_parquet(common.PROCESSED / "doc_topics.parquet")
    bt = pd.read_parquet(common.PROCESSED / "bertopic_doc_topics.parquet")
    bt_info = pd.read_csv(common.RESULTS / "a_bertopic_topics.csv")
    labels = pd.read_csv(common.CONFIG / "topic_labels.csv")
    assert (corpus["post_id"].values == nmf["post_id"].values).all()
    assert (corpus["post_id"].values == bt["post_id"].values).all()

    topic_cols = [c for c in nmf.columns if re.fullmatch(r"t\d+", c)]
    family_of = dict(zip(labels["topic_id"], labels["family"]))
    nmf_dom = pd.Series(np.argmax(nmf[topic_cols].values, axis=1)).map(family_of)
    nmf_dom[nmf["unassigned"].values] = "unassigned"

    # 1. Map BERTopic topics -> NMF family by majority overlap
    ct = pd.crosstab(bt["topic"], nmf_dom)
    mapping = ct.drop(columns="unassigned", errors="ignore").idxmax(axis=1).drop(-1, errors="ignore")
    purity = (ct.max(axis=1) / ct.sum(axis=1))
    fam_map = (bt_info.set_index("topic_id")
               .assign(family=mapping, purity=purity)
               .reset_index()[["topic_id", "posts", "family", "purity", "terms"]])
    fam_map.round(3).to_csv(common.RESULTS / "a_bertopic_family_map.csv", index=False)

    bt_family = bt["topic"].map(mapping).fillna("outlier")
    agree = (bt_family.values == nmf_dom.values)
    valid = (bt_family != "outlier").values & (nmf_dom != "unassigned").values
    agreement = pd.DataFrame({
        "metric": ["post-level family agreement (dominant NMF family vs mapped BERTopic family)",
                   "BERTopic outliers after reduction (%)"],
        "value": [agree[valid].mean(), 100 * (bt["topic"] == -1).mean()],
    })
    agreement.round(4).to_csv(common.RESULTS / "a_model_agreement.csv", index=False)

    # 2. Monthly shares under BERTopic (one-hot, so prevalence = share of posts)
    onehot = pd.get_dummies(bt_family).reindex(columns=tt.FAMILY_ORDER, fill_value=0).astype(float)
    corpus["assigned"] = (bt_family != "outlier").values
    corpus["visible"] = True
    bt_prev = tt.monthly_prevalence(corpus, onehot)
    bt_prev = bt_prev[bt_prev["sample"] == "submitted"].assign(model="BERTopic")

    nmf_prev = pd.read_csv(common.RESULTS / "a_prevalence_monthly.csv")
    nmf_prev = nmf_prev[nmf_prev["sample"] == "submitted"].assign(model="NMF")

    # 3. ITS + placebo for both
    rows = []
    for model, prev in [("NMF", nmf_prev), ("BERTopic", bt_prev)]:
        its, placebo = tt.run_its(prev, tt.FAMILY_ORDER)
        max_pl = (placebo.assign(a=placebo["beta_level"].abs())
                  .groupby(["subreddit", "family"])["a"].max().rename("max_placebo_abs_level"))
        its = its.set_index(["subreddit", "family"]).join(max_pl).reset_index()
        its["beats_all_placebos"] = its["beta_level"].abs() > its["max_placebo_abs_level"]
        rows.append(its.assign(model=model))
    its_all = pd.concat(rows)
    its_all.round(5).to_csv(common.RESULTS / "a_its_by_model.csv", index=False)

    # Figure: LLM family share, NMF vs BERTopic
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, sub in zip(axes, common.HISTORICAL):
        for model, prev, color in [("NMF (mean topic weight)", nmf_prev, "#2a78d6"),
                                   ("BERTopic (share of posts)", bt_prev, "#eb6834")]:
            d = prev[(prev["subreddit"] == sub) & (prev["family"] == "llm_genai")
                     & (prev["posts"] >= tt.MIN_POSTS)].sort_values("month")
            ax.plot(pd.to_datetime(d["month"] + "-01"), 100 * d["prevalence"], color=color,
                    linewidth=2, label=model)
        tt.style(ax)
        tt.mark_events(ax)
        ax.set_ylim(0, None)
        ax.set_title(f"r/{sub}", fontsize=10, loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("LLMs & generative AI (%)", fontsize=8, color="#55554f")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(common.FIGURES / "fig_a_nmf_vs_bertopic.png", dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(agreement.to_string(index=False))
    print("\nBERTopic topics mapped to llm_genai:")
    print(fam_map[fam_map["family"] == "llm_genai"][["topic_id", "posts", "purity", "terms"]]
          .to_string(index=False, max_colwidth=80))
    print("\nITS, both models (gap_pp = Dec 2024 vs counterfactual)")
    show = its_all.pivot_table(index=["subreddit", "family"], columns="model",
                               values=["gap_pp", "beats_all_placebos"], aggfunc="first")
    print(show.round(2).to_string())


if __name__ == "__main__":
    main()

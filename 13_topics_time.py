"""
A §2.6–2.9 — topic prevalence over time, interrupted time series, placebo
tests, r/MachineLearning post-type tags, emerging vocabulary.

Inputs: processed/doc_topics.parquet (11_topics_fit.py), config/topic_labels.csv
Outputs (aggregate only):
  results/a_prevalence_monthly.csv      sample x subreddit x month x family
  results/a_prevalence_monthly_topics.csv
  results/a_its.csv                     ITS coefficients + counterfactual gap
  results/a_placebo.csv                 ITS at fake dates (pre-period only)
  results/a_tags_monthly.csv            r/ML [D]/[R]/[P]/[N] shares (visible)
  results/a_emerging_terms.csv          monthly title share of new terms
  results/figures/fig_a_family_prevalence.png
  results/figures/fig_a_llm_counterfactual.png
  results/figures/fig_a_term_births.png
"""

import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import common

MIN_POSTS = 100
HAC_LAGS = 3
PLACEBO_MONTHS = ["2021-06", "2021-12", "2022-06"]

FAMILY_ORDER = ["llm_genai", "compute_tooling", "classic_dl_cv", "data_applied",
                "learning_career", "research", "generic"]
FAMILY_NAMES = {
    "llm_genai": "LLMs & generative AI",
    "compute_tooling": "Compute & tooling",
    "classic_dl_cv": "Classic DL & vision",
    "data_applied": "Data & applied ML",
    "learning_career": "Learning & career",
    "research": "Research & papers",
    "generic": "Generic model/AI talk",
}
# Reference categorical palette (dataviz skill), fixed order = FAMILY_ORDER.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
FAMILY_COLOR = dict(zip(FAMILY_ORDER, COLORS))

TAGS = ["tag_research", "tag_project", "tag_discussion", "tag_news"]

TERMS = ["chatgpt", "gpt4", "llama", "lora", "rag", "gguf", "ollama", "agent",
         "prompt", "finetune", "stable diffusion", "transformer"]


# --------------------------------------------------
# Data
# --------------------------------------------------

def load():
    corpus = common.load_corpus(["is_removed", "stickied", "title_tag"])
    topics = pd.read_parquet(common.PROCESSED / "doc_topics.parquet")
    assert (corpus["post_id"].values == topics["post_id"].values).all()

    labels = pd.read_csv(common.CONFIG / "topic_labels.csv")
    missing = labels["family"].isna() | labels["family"].eq("")
    if missing.any():
        raise SystemExit("config/topic_labels.csv: every topic needs a family")

    topic_cols = [c for c in topics.columns if re.fullmatch(r"t\d+", c)]
    fam = pd.DataFrame(index=topics.index)
    for family in FAMILY_ORDER:
        ids = labels.loc[labels["family"] == family, "topic_id"]
        fam[family] = topics[[f"t{i:02d}" for i in ids]].sum(axis=1)

    corpus["visible"] = (corpus["is_removed"].eq(0)
                         & pd.to_numeric(corpus["stickied"], errors="coerce").fillna(0).eq(0))
    corpus["assigned"] = ~topics["unassigned"].values
    return corpus, topics[topic_cols], fam


def monthly_prevalence(corpus, weights):
    """Mean normalised weight per (sample, subreddit, month), assigned posts only."""
    out = []
    for sample, mask in [("submitted", corpus["assigned"]),
                         ("visible", corpus["assigned"] & corpus["visible"])]:
        sub = weights[mask.values].copy()
        sub["subreddit"] = corpus.loc[mask, "subreddit"].values
        sub["month"] = corpus.loc[mask, "month"].values
        g = sub.groupby(["subreddit", "month"])
        prev = g.mean().stack().rename("prevalence").reset_index()
        prev.columns = ["subreddit", "month", "family", "prevalence"]
        prev = prev.merge(g.size().rename("posts").reset_index(), on=["subreddit", "month"])
        prev.insert(0, "sample", sample)
        out.append(prev)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------
# Interrupted time series
# --------------------------------------------------

def design(months, break_month):
    idx = pd.PeriodIndex(months, freq="M")
    month_idx = (idx.year - 2021) * 12 + idx.month - 1
    b = pd.Period(break_month, freq="M")
    b_idx = (b.year - 2021) * 12 + b.month - 1
    X = pd.DataFrame({
        "trend": month_idx,
        "post": (month_idx >= b_idx).astype(int),
        "t_since": np.maximum(0, month_idx - b_idx) * (month_idx >= b_idx),
        "academic_term": np.isin(idx.month, [2, 3, 4, 5, 9, 10, 11, 12]).astype(int),
        "blackout": (idx.strftime("%Y-%m") == "2023-06").astype(int),
    }, index=months)
    if X["blackout"].sum() == 0:
        X = X.drop(columns="blackout")
    return sm.add_constant(X, has_constant="add")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_its(series, break_month):
    """series: prevalence indexed by 'YYYY-MM'. Returns dict of results."""
    X = design(series.index, break_month)
    y = logit(series.values)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    # Counterfactual at the last month: same model without post terms.
    last = X.iloc[[-1]]
    cf = last.copy()
    cf[["post", "t_since"]] = 0
    inv = lambda z: 1 / (1 + np.exp(-z))
    fitted_last = inv(float(fit.predict(last).iloc[0]))
    cf_last = inv(float(fit.predict(cf).iloc[0]))

    return {
        "n_months": len(series),
        "beta_trend": fit.params["trend"],
        "beta_level": fit.params["post"], "p_level": fit.pvalues["post"],
        "beta_slope": fit.params["t_since"], "p_slope": fit.pvalues["t_since"],
        "pre_mean_pct": 100 * series[series.index < break_month].mean(),
        "last_month": series.index[-1],
        "fitted_last_pct": 100 * fitted_last,
        "counterfactual_last_pct": 100 * cf_last,
        "gap_pp": 100 * (fitted_last - cf_last),
        "r2": fit.rsquared,
    }, fit


def run_its(prev, families):
    rows, placebo = [], []
    for (sample, sub, fam), g in prev.groupby(["sample", "subreddit", "family"]):
        if sub not in common.HISTORICAL or fam not in families:
            continue
        s = g[g["posts"] >= MIN_POSTS].set_index("month")["prevalence"].sort_index()
        res, _ = fit_its(s, common.INTERVENTION_MONTH)
        rows.append({"sample": sample, "subreddit": sub, "family": fam, **res})

        pre = s[s.index < common.INTERVENTION_MONTH]
        for pm in PLACEBO_MONTHS:
            pres, _ = fit_its(pre, pm)
            placebo.append({"sample": sample, "subreddit": sub, "family": fam,
                            "placebo_month": pm, "beta_level": pres["beta_level"],
                            "p_level": pres["p_level"], "beta_slope": pres["beta_slope"],
                            "p_slope": pres["p_slope"]})

    its = pd.DataFrame(rows)
    for sample in its["sample"].unique():
        m = its["sample"] == sample
        pvals = np.concatenate([its.loc[m, "p_level"], its.loc[m, "p_slope"]])
        q = multipletests(pvals, method="fdr_bh")[1]
        its.loc[m, "q_level"] = q[: m.sum()]
        its.loc[m, "q_slope"] = q[m.sum():]
    return its, pd.DataFrame(placebo)


# --------------------------------------------------
# Tags and emerging terms
# --------------------------------------------------

def tag_shares(corpus):
    ml = corpus[(corpus["subreddit"] == "MachineLearning") & corpus["visible"]]
    shares = pd.crosstab(ml["month"], ml["title_tag"], normalize="index")
    shares = shares.stack().rename("prevalence").reset_index()
    shares.columns = ["month", "family", "prevalence"]
    shares["family"] = "tag_" + shares["family"]
    counts = ml.groupby("month").size().rename("posts").reset_index()
    shares = shares.merge(counts, on="month")
    shares.insert(0, "subreddit", "MachineLearning")
    shares.insert(0, "sample", "visible")
    return shares


def emerging_terms(corpus):
    lemmas = common.load_lemmatized_titles()["lemma"]
    df = pd.DataFrame({"subreddit": corpus["subreddit"], "month": corpus["month"]})
    for term in TERMS:
        df[term] = lemmas.str.contains(rf"\b{re.escape(term)}\b", regex=True).values
    out = df.groupby(["subreddit", "month"])[TERMS].mean().mul(100)
    out = out.stack().rename("share_pct").reset_index()
    out.columns = ["subreddit", "month", "term", "share_pct"]
    return out


# --------------------------------------------------
# Figures
# --------------------------------------------------

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def mark_events(ax, label=False):
    for month, name in common.EVENTS.items():
        x = pd.Timestamp(month + "-01")
        main = month == common.INTERVENTION_MONTH
        ax.axvline(x, color="#55554f" if main else "#b5b4ad",
                   linewidth=1 if main else 0.7, linestyle="--", zorder=0)
        if label and main:
            ax.text(x, ax.get_ylim()[1], " ChatGPT", fontsize=7, color="#55554f",
                    va="top", ha="left")


def fig_family_prevalence(prev):
    p = prev[prev["sample"] == "submitted"].copy()
    p["date"] = pd.to_datetime(p["month"] + "-01")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for ax, sub in zip(axes.flat, common.SUBREDDITS):
        d = p[(p["subreddit"] == sub) & (p["posts"] >= MIN_POSTS)]
        for fam in FAMILY_ORDER:
            f = d[d["family"] == fam]
            ax.plot(f["date"], 100 * f["prevalence"], color=FAMILY_COLOR[fam],
                    linewidth=2, label=FAMILY_NAMES[fam], solid_capstyle="round")
        style(ax)
        ax.set_title(f"r/{sub}", fontsize=10, loc="left", color="#1a1a19")
        ax.set_ylim(0, None)
        mark_events(ax, label=(sub == "MachineLearning"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean topic weight (%)", fontsize=8, color="#55554f")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Topic families per month (all submitted posts)", x=0.06, ha="left",
                 fontsize=11, color="#1a1a19")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(common.FIGURES / "fig_a_family_prevalence.png", dpi=200)
    plt.close(fig)


def fig_llm_counterfactual(prev):
    p = prev[(prev["sample"] == "submitted") & (prev["family"] == "llm_genai")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, sub in zip(axes, common.HISTORICAL):
        s = (p[(p["subreddit"] == sub) & (p["posts"] >= MIN_POSTS)]
             .set_index("month")["prevalence"].sort_index())
        _, fit = fit_its(s, common.INTERVENTION_MONTH)
        X = design(s.index, common.INTERVENTION_MONTH)
        # Plot without the one-month blackout dummy so lines stay smooth.
        if "blackout" in X:
            X["blackout"] = 0
        cf = X.copy()
        cf[["post", "t_since"]] = 0
        inv = lambda z: 100 / (1 + np.exp(-z))
        dates = pd.to_datetime(s.index + "-01")
        post = s.index >= common.INTERVENTION_MONTH
        ax.plot(dates, 100 * s.values, color="#2a78d6", linewidth=1, alpha=0.45,
                label="Observed")
        ax.plot(dates, inv(fit.predict(X)), color="#2a78d6", linewidth=2,
                label="Fitted (ITS)")
        ax.plot(dates[post], inv(fit.predict(cf))[post], color="#55554f", linewidth=2,
                linestyle=":", label="Counterfactual: pre-trend continued")
        style(ax)
        mark_events(ax)
        ax.set_title(f"r/{sub}", fontsize=10, loc="left", color="#1a1a19")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("LLMs & generative AI (%)", fontsize=8, color="#55554f")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(common.FIGURES / "fig_a_llm_counterfactual.png", dpi=200)
    plt.close(fig)


def fig_term_births(terms):
    """Heat strip: monthly share of titles with each term, all subreddits pooled
    except LocalLLaMA (so the strip shows diffusion into the historical communities)."""
    t = terms[terms["subreddit"].isin(common.HISTORICAL)]
    pivot = t.groupby(["term", "month"])["share_pct"].mean().unstack()
    first = pivot.gt(0.1).idxmax(axis=1).where(pivot.gt(0.1).any(axis=1), "9999")
    pivot = pivot.loc[first.sort_values().index]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0,
                   vmax=np.nanpercentile(pivot.values, 98))
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=8)
    ticks = [i for i, m in enumerate(pivot.columns) if m.endswith("-01")]
    ax.set_xticks(ticks, [pivot.columns[i][:4] for i in ticks], fontsize=8)
    chatgpt = list(pivot.columns).index(common.INTERVENTION_MONTH)
    ax.axvline(chatgpt - 0.5, color="#eb6834", linewidth=1.2)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of titles (mean of 3 historical subreddits)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("When new vocabulary arrived (orange line: ChatGPT)", fontsize=10,
                 loc="left", color="#1a1a19")
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig_a_term_births.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------

def main():
    common.FIGURES.mkdir(parents=True, exist_ok=True)
    corpus, topics, fam = load()

    prev = monthly_prevalence(corpus, fam)
    prev.round(5).to_csv(common.RESULTS / "a_prevalence_monthly.csv", index=False)
    monthly_prevalence(corpus, topics).round(5).to_csv(
        common.RESULTS / "a_prevalence_monthly_topics.csv", index=False)

    tags = tag_shares(corpus)
    tags.round(5).to_csv(common.RESULTS / "a_tags_monthly.csv", index=False)

    its, placebo = run_its(pd.concat([prev, tags]), FAMILY_ORDER + TAGS)
    its.round(5).to_csv(common.RESULTS / "a_its.csv", index=False)
    placebo.round(5).to_csv(common.RESULTS / "a_placebo.csv", index=False)

    terms = emerging_terms(corpus)
    terms.round(4).to_csv(common.RESULTS / "a_emerging_terms.csv", index=False)

    fig_family_prevalence(prev)
    fig_llm_counterfactual(prev)
    fig_term_births(terms)

    pd.set_option("display.width", 200)
    show = its[its["sample"] == "submitted"][[
        "subreddit", "family", "pre_mean_pct", "fitted_last_pct",
        "counterfactual_last_pct", "gap_pp", "beta_level", "q_level",
        "beta_slope", "q_slope"]]
    print("\nITS (submitted posts) — gap = fitted minus counterfactual in Dec 2024")
    print(show.round(3).to_string(index=False))
    print("\nTag ITS (r/ML visible)")
    print(its[its["sample"] == "visible"].query("family in @TAGS")[
        ["family", "pre_mean_pct", "fitted_last_pct", "gap_pp", "q_level", "q_slope"]
    ].round(3).to_string(index=False))
    sig = (placebo["p_level"] < 0.05) | (placebo["p_slope"] < 0.05)
    print(f"\nPlacebo tests significant at p<0.05 (uncorrected): {sig.sum()}/{len(placebo)}")


if __name__ == "__main__":
    main()

"""
Task F / H4: Research versus applied content.

Primary:
  - Academic vs code/model link presence among external-link posts.
  - [R] vs [P] title tags among visible r/MachineLearning posts.
  - Monthly interrupted time series, including direct research-minus-applied contrasts.

Exploratory:
  - All-post denominators.
  - NMF topic-family crossovers.
  - Title-tag and domain crossovers.
  - Fully mapped-link sensitivity analysis.

No causal interpretation is implied.
"""

import importlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import common

topic_time = importlib.import_module("13_topics_time")
BREAK = common.INTERVENTION_MONTH

common.RESULTS.mkdir(parents=True, exist_ok=True)
common.FIGURES.mkdir(parents=True, exist_ok=True)


def boolean_column(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )


def safe_rate(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def load_data():
    print("Loading corpus, link features, events and topics...")

    corpus = pd.read_csv(
        common.CORPUS,
        usecols=[
            "post_id", "subreddit", "title_tag",
            "is_removed", "stickied"
        ],
        low_memory=False
    )

    features = pd.read_parquet(
        common.PROCESSED / "link_features.parquet",
        columns=[
            "post_id", "subreddit", "month",
            "has_external_link", "fully_mapped"
        ]
    )

    events = pd.read_parquet(
        common.PROCESSED / "link_events.parquet",
        columns=[
            "post_id", "subreddit",
            "canonical_domain", "category"
        ]
    )

    topics = pd.read_parquet(
        common.PROCESSED / "doc_topics.parquet"
    )

    labels = pd.read_csv(
        common.CONFIG / "topic_labels.csv"
    )

    topic_cols = sorted(
        c for c in topics.columns
        if re.fullmatch(r"t\d+", c)
    )

    if not labels["topic_id"].is_unique:
        raise ValueError("Duplicate topic IDs in topic_labels.csv")

    family_map = {
        f"t{int(row.topic_id):02d}": str(row.family)
        for row in labels.itertuples()
    }

    if set(topic_cols) != set(family_map):
        raise ValueError(
            "Topic columns and topic-label IDs do not match"
        )

    research_cols = [
        c for c in topic_cols
        if family_map[c] == "research"
    ]

    applied_cols = [
        c for c in topic_cols
        if family_map[c] in {
            "compute_tooling", "data_applied"
        }
    ]

    if not research_cols or not applied_cols:
        raise ValueError(
            "Research or applied topic families are missing"
        )

    topics["research_weight"] = topics[
        research_cols
    ].sum(axis=1)

    topics["applied_weight"] = topics[
        applied_cols
    ].sum(axis=1)

    topics["assigned"] = (
        ~topics["unassigned"].astype(bool)
    ).astype(int)

    topics["dominant_family"] = (
        topics[topic_cols]
        .idxmax(axis=1)
        .map(family_map)
    )

    topics.loc[
        topics["top_weight"] < 0.4,
        "dominant_family"
    ] = "mixed"

    topics.loc[
        topics["assigned"] == 0,
        "dominant_family"
    ] = "unassigned"

    flag_rules = {
        "academic": events["category"].eq("academic"),
        "code": events["category"].eq("code"),
        "models": events["category"].eq("models/hubs"),
        "applied_link": events["category"].isin(
            ["code", "models/hubs"]
        ),
        "arxiv": events["canonical_domain"].eq("arxiv.org"),
        "github": events["canonical_domain"].eq("github.com"),
        "huggingface": events["canonical_domain"].eq(
            "huggingface.co"
        )
    }

    flag_cols = list(flag_rules)

    for name, condition in flag_rules.items():
        events[name] = condition.astype("int8")

    flags = (
        events.groupby(["post_id", "subreddit"], as_index=False)
        [flag_cols]
        .max()
    )

    df = features.merge(
        corpus,
        on=["post_id", "subreddit"],
        how="left",
        validate="one_to_one"
    )

    df = df.merge(
        topics[
            [
                "post_id", "research_weight",
                "applied_weight", "assigned",
                "dominant_family"
            ]
        ],
        on="post_id",
        how="left",
        validate="one_to_one"
    )

    if df["assigned"].isna().any():
        raise ValueError("Some posts have no topic assignment row")

    df = df.merge(
        flags,
        on=["post_id", "subreddit"],
        how="left",
        validate="one_to_one"
    )

    df[flag_cols] = (
        df[flag_cols].fillna(0).astype("int8")
    )

    df["removed"] = boolean_column(df["is_removed"])
    df["stickied_bool"] = boolean_column(df["stickied"])

    df["visible"] = (
        ~df["removed"] & ~df["stickied_bool"]
    )

    df["tag"] = (
        df["title_tag"]
        .fillna("none")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["research_tag"] = df["tag"].eq("research").astype(int)
    df["project_tag"] = df["tag"].eq("project").astype(int)

    df["period"] = np.where(
        df["month"] < BREAK, "pre", "post"
    )

    inconsistent = (
        (df["academic"] == 1)
        | (df["applied_link"] == 1)
    ) & (df["has_external_link"] == 0)

    if inconsistent.any():
        raise ValueError(
            f"{inconsistent.sum()} posts have external categories "
            "but has_external_link=0. Fix classification first."
        )

    if len(df) != len(features):
        raise ValueError("Join changed the post count")

    print(f"Joined {len(df):,} posts successfully")
    print("Research topics:", research_cols)
    print("Applied topics:", applied_cols)

    return df


AGG = {
    "posts": ("post_id", "size"),
    "external_posts": ("has_external_link", "sum"),
    "academic_posts": ("academic", "sum"),
    "applied_posts": ("applied_link", "sum"),
    "research_tag_posts": ("research_tag", "sum"),
    "project_tag_posts": ("project_tag", "sum"),
    "assigned_posts": ("assigned", "sum"),
    "research_weight_sum": ("research_weight", "sum"),
    "applied_weight_sum": ("applied_weight", "sum")
}


def add_rates(table):
    pairs = {
        "academic_ext": (
            "academic_posts", "external_posts"
        ),
        "applied_ext": (
            "applied_posts", "external_posts"
        ),
        "academic_all": (
            "academic_posts", "posts"
        ),
        "applied_all": (
            "applied_posts", "posts"
        ),
        "research_tag_all": (
            "research_tag_posts", "posts"
        ),
        "project_tag_all": (
            "project_tag_posts", "posts"
        ),
        "research_topic": (
            "research_weight_sum", "assigned_posts"
        ),
        "applied_topic": (
            "applied_weight_sum", "assigned_posts"
        )
    }

    for metric, (numerator, denominator) in pairs.items():
        table[metric] = safe_rate(
            table[numerator], table[denominator]
        )

    rp_total = (
        table["research_tag_posts"]
        + table["project_tag_posts"]
    )

    table["research_tag_rp"] = safe_rate(
        table["research_tag_posts"], rp_total
    )

    table["project_tag_rp"] = safe_rate(
        table["project_tag_posts"], rp_total
    )

    table["link_balance_ext"] = (
        table["academic_ext"] - table["applied_ext"]
    )

    table["tag_balance_all"] = (
        table["research_tag_all"]
        - table["project_tag_all"]
    )

    return table


def make_aggregates(df):
    monthly_parts = []
    prepost_parts = []

    for sample, part in [
        ("submitted", df),
        ("visible", df[df["visible"]])
    ]:
        m = (
            part.groupby(["subreddit", "month"])
            .agg(**AGG)
            .reset_index()
        )
        m.insert(0, "sample", sample)
        monthly_parts.append(add_rates(m))

        p = (
            part.groupby(["subreddit", "period"])
            .agg(**AGG)
            .reset_index()
        )
        p.insert(0, "sample", sample)
        prepost_parts.append(add_rates(p))

    monthly = pd.concat(
        monthly_parts, ignore_index=True
    )

    prepost = pd.concat(
        prepost_parts, ignore_index=True
    )

    monthly.to_csv(
        common.RESULTS / "f_h4_monthly.csv",
        index=False
    )

    prepost.to_csv(
        common.RESULTS / "f_h4_prepost.csv",
        index=False
    )

    return monthly, prepost


def fit_linear_balance(series):
    """HAC segmented regression for signed share differences."""
    X = topic_time.design(series.index, BREAK)

    fit = sm.OLS(
        series.to_numpy(dtype=float), X
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": topic_time.HAC_LAGS}
    )

    last = X.iloc[[-1]]
    counterfactual = last.copy()
    counterfactual[["post", "t_since"]] = 0

    gap = float(
        fit.predict(last).iloc[0]
        - fit.predict(counterfactual).iloc[0]
    )

    return {
        "beta_level": float(fit.params["post"]),
        "p_level": float(fit.pvalues["post"]),
        "beta_slope": float(fit.params["t_since"]),
        "p_slope": float(fit.pvalues["t_since"]),
        "gap_pp": 100 * gap,
        "scale": "proportion_difference"
    }


def run_h4_its(monthly):
    specifications = [
        ("submitted", "academic_ext", "external_posts", 30),
        ("submitted", "applied_ext", "external_posts", 30),
        ("submitted", "link_balance_ext", "external_posts", 30),
        ("visible", "research_tag_all", "posts", 100),
        ("visible", "project_tag_all", "posts", 100),
        ("visible", "tag_balance_all", "posts", 100)
    ]

    rows = []

    for subreddit in common.HISTORICAL:
        for sample, metric, denom, minimum in specifications:

            if "tag" in metric and subreddit != "MachineLearning":
                continue

            subset = monthly[
                (monthly["subreddit"] == subreddit)
                & (monthly["sample"] == sample)
                & (monthly[denom] >= minimum)
                & monthly[metric].notna()
            ].sort_values("month")

            series = subset.set_index("month")[metric]

            pre = series[series.index < BREAK]
            post = series[series.index >= BREAK]

            if len(pre) < 12 or len(post) < 12:
                print(
                    "Skipping insufficient series:",
                    subreddit, sample, metric
                )
                continue

            is_balance = "balance" in metric

            if is_balance:
                result = fit_linear_balance(series)
            else:
                result, _ = topic_time.fit_its(series, BREAK)
                result["scale"] = "log_odds"

            rows.append({
                "subreddit": subreddit,
                "sample": sample,
                "metric": metric,
                "n_pre_months": len(pre),
                "n_post_months": len(post),
                "boundary_months": int(
                    series.isin([0.0, 1.0]).sum()
                ),
                **result
            })

    its = pd.DataFrame(rows)

    if its.empty:
        raise RuntimeError("No estimable H4 ITS series")

    n = len(its)

    pvals = np.concatenate([
        its["p_level"].to_numpy(),
        its["p_slope"].to_numpy()
    ])

    pvals = np.nan_to_num(
        pvals, nan=1.0, posinf=1.0, neginf=1.0
    )

    qvals = multipletests(
        pvals, method="fdr_bh"
    )[1]

    its["q_level"] = qvals[:n]
    its["q_slope"] = qvals[n:]

    its.to_csv(
        common.RESULTS / "f_h4_its.csv",
        index=False
    )

    print("\n=== H4 ITS ===")
    print(
        its[
            [
                "subreddit", "metric", "scale",
                "beta_level", "q_level",
                "beta_slope", "q_slope",
                "gap_pp", "boundary_months"
            ]
        ].round(4).to_string(index=False)
    )

    return its


def make_tag_crossovers(df):
    ml = df[
        (df["subreddit"] == "MachineLearning")
        & df["tag"].isin(["research", "project"])
    ]

    parts = []

    for sample, part in [
        ("submitted", ml),
        ("visible", ml[ml["visible"]])
    ]:
        out = (
            part.groupby(["period", "tag"])
            .agg(
                posts=("post_id", "size"),
                external_posts=("has_external_link", "sum"),
                academic_posts=("academic", "sum"),
                applied_posts=("applied_link", "sum"),
                arxiv_posts=("arxiv", "sum"),
                github_posts=("github", "sum"),
                huggingface_posts=("huggingface", "sum")
            )
            .reset_index()
        )

        out.insert(0, "sample", sample)

        for name in [
            "academic", "applied", "arxiv",
            "github", "huggingface"
        ]:
            out[name + "_all_pct"] = (
                100 * safe_rate(
                    out[name + "_posts"], out["posts"]
                )
            )
            out[name + "_ext_pct"] = (
                100 * safe_rate(
                    out[name + "_posts"], out["external_posts"]
                )
            )

        parts.append(out)

    result = pd.concat(parts, ignore_index=True)

    result.to_csv(
        common.RESULTS / "f_h4_tag_links.csv",
        index=False
    )

    print("\n=== [R]/[P] LINK CROSSOVER: VISIBLE POSTS ===")
    print(
        result[result["sample"] == "visible"][
            [
                "period", "tag", "posts",
                "academic_all_pct", "applied_all_pct",
                "arxiv_all_pct", "github_all_pct",
                "huggingface_all_pct"
            ]
        ].round(2).to_string(index=False)
    )


def make_topic_crossovers(df):
    parts = []

    for sample, part in [
        ("submitted", df),
        ("visible", df[df["visible"]])
    ]:
        out = (
            part.groupby(
                ["subreddit", "period", "dominant_family"]
            )
            .agg(
                posts=("post_id", "size"),
                external_posts=("has_external_link", "sum"),
                academic_posts=("academic", "sum"),
                applied_posts=("applied_link", "sum"),
                arxiv_posts=("arxiv", "sum"),
                github_posts=("github", "sum"),
                huggingface_posts=("huggingface", "sum")
            )
            .reset_index()
        )

        out.insert(0, "sample", sample)

        out["academic_ext_pct"] = (
            100 * safe_rate(
                out["academic_posts"], out["external_posts"]
            )
        )

        out["applied_ext_pct"] = (
            100 * safe_rate(
                out["applied_posts"], out["external_posts"]
            )
        )

        out["academic_all_pct"] = (
            100 * safe_rate(
                out["academic_posts"], out["posts"]
            )
        )

        out["applied_all_pct"] = (
            100 * safe_rate(
                out["applied_posts"], out["posts"]
            )
        )

        out["cell_ge_200"] = out["posts"] >= 200

        parts.append(out)

    result = pd.concat(parts, ignore_index=True)

    result.to_csv(
        common.RESULTS / "f_h4_topic_link_cross.csv",
        index=False
    )


def make_coverage_sensitivity(df):
    parts = []

    for sample, part in [
        ("submitted", df),
        ("visible", df[df["visible"]])
    ]:
        external = part[part["has_external_link"] == 1]

        for scope, subset in [
            ("all_external", external),
            (
                "fully_mapped_external",
                external[external["fully_mapped"] == 1]
            )
        ]:
            out = (
                subset.groupby(["subreddit", "period"])
                .agg(
                    posts=("post_id", "size"),
                    academic_posts=("academic", "sum"),
                    applied_posts=("applied_link", "sum")
                )
                .reset_index()
            )

            out["sample"] = sample
            out["scope"] = scope

            out["academic_pct"] = (
                100 * safe_rate(
                    out["academic_posts"], out["posts"]
                )
            )

            out["applied_pct"] = (
                100 * safe_rate(
                    out["applied_posts"], out["posts"]
                )
            )

            parts.append(out)

    result = pd.concat(parts, ignore_index=True)

    result.to_csv(
        common.RESULTS / "f_h4_coverage_sensitivity.csv",
        index=False
    )


def make_figure(monthly):
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True
    )

    for ax, subreddit in zip(
        axes, common.HISTORICAL
    ):
        g = monthly[
            (monthly["sample"] == "submitted")
            & (monthly["subreddit"] == subreddit)
        ].sort_values("month")

        dates = pd.to_datetime(
            g["month"] + "-01"
        )

        ax.plot(
            dates,
            100 * g["academic_ext"],
            label="Academic",
            color="#2474b5",
            linewidth=2
        )

        ax.plot(
            dates,
            100 * g["applied_ext"],
            label="Code / model hubs",
            color="#e1772b",
            linewidth=2
        )

        ax.axvline(
            pd.Timestamp("2022-12-01"),
            color="black",
            linestyle="--",
            alpha=0.65
        )

        ax.set_title("r/" + subreddit)
        ax.set_ylabel("% external-link posts")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Month")

    fig.suptitle(
        "H4: Academic versus applied link presence",
        fontsize=14
    )

    fig.tight_layout()

    path = (
        common.FIGURES
        / "fig_f_h4_research_vs_applied.png"
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    print("Saved figure:", path)


def main():
    df = load_data()

    monthly, prepost = make_aggregates(df)

    run_h4_its(monthly)
    make_tag_crossovers(df)
    make_topic_crossovers(df)
    make_coverage_sensitivity(df)
    make_figure(monthly)

    print("\n=== H4 PRE/POST LINK SUMMARY ===")

    summary = prepost[
        (prepost["sample"] == "submitted")
        & prepost["subreddit"].isin(common.HISTORICAL)
    ].copy()

    for col in [
        "academic_ext", "applied_ext",
        "academic_all", "applied_all",
        "research_topic", "applied_topic"
    ]:
        summary[col] *= 100

    print(
        summary[
            [
                "subreddit", "period",
                "academic_ext", "applied_ext",
                "academic_all", "applied_all",
                "research_topic", "applied_topic"
            ]
        ].round(2).to_string(index=False)
    )

    print("\nDone. H4 results saved to results/.")


if __name__ == "__main__":
    main()

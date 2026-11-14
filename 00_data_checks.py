"""
Step 0 data-validity checks (docs/research_design.md, section 1.5).

Reads processed/modeling_corpus.csv and writes aggregate-only tables:
  results/data_checks_monthly.csv   one row per subreddit-month
  results/data_checks_yearly.csv    one row per subreddit-year
and prints a short summary with the decision-rule flags.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "modeling_corpus.csv"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

MIN_MONTHLY_POSTS = 100

columns = [
    "subreddit", "created_utc", "year", "month",
    "is_self", "is_removed", "removed_by_category",
    "retrieved_on", "updated_on", "subreddit_subscribers",
    "score", "num_comments", "upvote_ratio", "stickied",
    "title_tag", "author_hash", "url",
]

df = pd.read_csv(INPUT, usecols=columns, low_memory=False)

for col in ["retrieved_on", "updated_on", "score", "num_comments",
            "upvote_ratio", "subreddit_subscribers", "stickied"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["removed_by_category"] = df["removed_by_category"].fillna("none")
df["period_month"] = pd.to_datetime(
    df["created_utc"], unit="s", utc=True
).dt.strftime("%Y-%m")
df["capture_lag_h"] = (df["retrieved_on"] - df["created_utc"]) / 3600
df["update_lag_h"] = (df["updated_on"] - df["created_utc"]) / 3600
df["has_url"] = df["url"].fillna("").str.strip().ne("")

visible = df["is_removed"].eq(0) & df["stickied"].fillna(0).eq(0)


def summarize(group):
    vis = group[visible.loc[group.index]]

    row = {
        "posts": len(group),
        "visible_posts": len(vis),
        "unique_authors": group["author_hash"].nunique(),
        "removed_pct": 100 * group["is_removed"].mean(),
    }

    for category in ["moderator", "automod_filtered", "deleted", "reddit"]:
        row[f"removed_{category}_pct"] = 100 * group[
            "removed_by_category"
        ].eq(category).mean()

    row.update({
        "stickied_pct": 100 * group["stickied"].fillna(0).mean(),
        "capture_lag_h_median": group["capture_lag_h"].median(),
        "has_updated_on_pct": 100 * group["updated_on"].notna().mean(),
        "update_lag_h_median": group["update_lag_h"].median(),
        "subscribers_median": group["subreddit_subscribers"].median(),
        # Engagement on visible posts only.
        "score_median": vis["score"].median(),
        "score_p90": vis["score"].quantile(0.9),
        "score_le1_pct": 100 * vis["score"].le(1).mean(),
        "comments_median": vis["num_comments"].median(),
        "zero_comments_pct": 100 * vis["num_comments"].eq(0).mean(),
        "upvote_ratio_median": vis["upvote_ratio"].median(),
        "title_tag_pct": 100 * group["title_tag"].ne("none").mean(),
        "missing_score_pct": 100 * group["score"].isna().mean(),
        "missing_url_pct": 100 * (~group["has_url"]).mean(),
    })
    return pd.Series(row)


monthly = (
    df.groupby(["subreddit", "period_month"])
    .apply(summarize, include_groups=False)
    .reset_index()
)
monthly["low_volume"] = monthly["posts"] < MIN_MONTHLY_POSTS

yearly = (
    df.groupby(["subreddit", "year"])
    .apply(summarize, include_groups=False)
    .reset_index()
)

monthly.round(3).to_csv(OUTPUT / "data_checks_monthly.csv", index=False)
yearly.round(3).to_csv(OUTPUT / "data_checks_yearly.csv", index=False)


# --------------------------------------------------
# Summary and decision-rule flags
# --------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print("\nYEARLY OVERVIEW")
print(yearly[[
    "subreddit", "year", "posts", "visible_posts", "removed_pct",
    "removed_moderator_pct", "capture_lag_h_median",
    "has_updated_on_pct", "score_median", "score_le1_pct",
    "comments_median", "zero_comments_pct", "title_tag_pct",
]].round(1).to_string(index=False))

print("\nFLAGS")

# Score capture: jump in share of visible posts with score <= 1.
for sub, g in monthly.sort_values("period_month").groupby("subreddit"):
    jumps = g["score_le1_pct"].diff().abs()
    if jumps.max() > 20:
        month = g.loc[jumps.idxmax(), "period_month"]
        print(f"- r/{sub}: score<=1 share jumps {jumps.max():.1f} pp "
              f"at {month} -> check score comparability (B gate)")

    lag = g["capture_lag_h_median"].replace(0, np.nan).dropna()
    if len(lag) and lag.max() / max(lag.min(), 1e-6) > 10:
        print(f"- r/{sub}: median capture lag ranges "
              f"{lag.min():.3g}h to {lag.max():.3g}h (>10x)")

    removed_jump = g["removed_pct"].diff().abs()
    if removed_jump.max() > 10:
        month = g.loc[removed_jump.idxmax(), "period_month"]
        print(f"- r/{sub}: removal rate moves {removed_jump.max():.1f} pp "
              f"at {month} -> moderation confound for A/D")

    low = g.loc[g["low_volume"], "period_month"].tolist()
    if low:
        print(f"- r/{sub}: {len(low)} months with < {MIN_MONTHLY_POSTS} "
              f"posts: {', '.join(low[:6])}{' ...' if len(low) > 6 else ''}")

ml = df[df["subreddit"].eq("MachineLearning")]
coverage = 100 * ml["title_tag"].ne("none").mean()
print(f"- r/MachineLearning title-tag coverage: {coverage:.1f}% "
      f"({'OK' if coverage >= 80 else 'below 80% -> descriptive only'})")

print(f"\nSaved: {OUTPUT / 'data_checks_monthly.csv'}")
print(f"Saved: {OUTPUT / 'data_checks_yearly.csv'}")

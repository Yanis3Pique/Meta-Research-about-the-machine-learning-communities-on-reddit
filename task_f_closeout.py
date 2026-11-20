#!/usr/bin/env python3
"""Task F closeout, read-only against corpus/taxonomy.

From repository root:
    python task_f_closeout.py --prepare
    # Two reviewers complete results/f_closeout/top100_reviewer_A.csv and _B.csv
    python task_f_closeout.py --agreement

The script never edits config/domain_categories.csv, research outputs, or corpus.
Generated CSV files are placed under results/f_closeout/.
"""
import argparse
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

SITES = ("arxiv.org", "github.com", "huggingface.co")
VALID = {"academic", "code", "models/hubs", "blog/tutorial", "company/product",
         "news/media", "social", "video", "reddit-internal", "reddit-media", "other"}


def load_project():
    import common
    link = import_module("20_links")
    for attr in ("post_urls", "classify_url", "load_mapping"):
        if not hasattr(link, attr):
            raise RuntimeError("Your 20_links.py is missing " + attr + ". Use the version that produced f_coverage.csv.")
    return common, link


def fraction(num, den):
    return 100.0 * num / den if den else np.nan


def prepare(root):
    import os, sys
    os.chdir(root)
    sys.path.insert(0, str(root))
    common, link = load_project()
    out = root / "results" / "f_closeout"
    out.mkdir(parents=True, exist_ok=True)
    required = [root / "results" / file for file in (
        "f_domains_top.csv", "f_category_monthly.csv", "f_coverage.csv")]
    required += [root / "processed" / "link_features.parquet", root / "processed" / "modeling_corpus.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing current pipeline outputs: " + ", ".join(missing))
    print("Reading current pipeline outputs and corpus...")
    domains = pd.read_csv(root / "results" / "f_domains_top.csv")
    if not {"domain", "total_posts", "category"}.issubset(domains.columns):
        raise ValueError("Expected f_domains_top.csv from updated 20_links.py")
    global_domains = (domains.groupby("domain", as_index=False, dropna=False)
                      .agg(total_posts=("total_posts", "max"), category=("category", "first")))
    global_domains = global_domains.sort_values(["total_posts", "domain"], ascending=[False, True])
    if len(global_domains) < 100:
        raise ValueError("Fewer than 100 domains: confirm the current domain export")

    mapping = link.load_mapping()
    columns = ["is_self", "url", "body", "domain", "title_tag", "is_removed", "stickied"]
    corpus = common.load_corpus(columns)
    month_counts = Counter()
    domain_month_counts = Counter()
    domain_month_weights = Counter()
    category_month_weights = Counter()
    site_period_counts = Counter()
    site_period_den = Counter()
    examples = defaultdict(list)
    linked_rows = []
    totals = Counter()
    print(f"Scanning {len(corpus):,} posts with the actual 20_links.py extraction logic...")
    for row in corpus.itertuples(index=False):
        origin, urls = link.post_urls(row)
        items = []
        for url in urls:
            item = link.classify_url(url, mapping)
            if item is not None:
                items.append((item[0], item[1], bool(item[2]), url))
        if not items:
            continue
        sub, month = str(row.subreddit), str(row.month)
        period = "pre" if month < common.INTERVENTION_MONTH else "post"
        uniq = {dom: category for dom, category, _, _ in items}
        classes = set(uniq.values())
        totals["posts"] += 1
        totals["links"] += len(items)
        totals["mapped_links"] += sum(mapped for _, _, mapped, _ in items)
        totals["post_domains"] += len(uniq)
        totals["mapped_domains"] += sum(any(z[0] == dom and z[2] for z in items) for dom in uniq)
        totals["fully_mapped_posts"] += int(all(mapped for _, _, mapped, _ in items))
        if origin == "primary":
            totals["primary_links"] += len(items)
            totals["mapped_primary"] += sum(mapped for _, _, mapped, _ in items)
        month_counts[sub, month] += 1
        site_period_den[sub, period] += 1
        for domain in uniq:
            domain_month_counts[sub, month, domain] += 1
            if domain in SITES:
                site_period_counts[sub, period, domain] += 1
        for category in classes:
            members = [dom for dom, cat in uniq.items() if cat == category]
            cat_weight = 1 / len(classes)
            category_month_weights[sub, month, category] += cat_weight
            for dom in members:
                domain_month_weights[sub, month, category, dom] += cat_weight / len(members)
        for domain, _, _, url in items:
            if url not in examples[domain] and len(examples[domain]) < 3:
                examples[domain].append(url)
        linked_rows.append({"post_id": row.post_id, "subreddit": sub, "month": month,
                            "period": period, "title_tag": row.title_tag,
                            "is_removed": row.is_removed, "stickied": row.stickied,
                            "categories": classes, "domains": set(uniq)})

    cov = pd.read_csv(root / "results" / "f_coverage.csv")
    overall = cov[(cov["subreddit"] == "ALL") & (cov["period"] == "all")].set_index("metric")
    checks = {
        "individual_links": (totals["mapped_links"], totals["links"]),
        "primary_links": (totals["mapped_primary"], totals["primary_links"]),
        "post_domains": (totals["mapped_domains"], totals["post_domains"]),
        "fully_mapped_posts": (totals["fully_mapped_posts"], totals["posts"]),
    }
    for name, (mapped, total) in checks.items():
        expected_mapped, expected_total = int(overall.loc[name, "mapped"]), int(overall.loc[name, "total"])
        if (mapped, total) != (expected_mapped, expected_total):
            raise RuntimeError(f"Current corpus and f_coverage.csv disagree for {name}: "
                               f"scanned {mapped}/{total}, reported {expected_mapped}/{expected_total}. "
                               "Do not use these outputs until reconciled.")
    print("All four coverage measures reproduce exactly from the current corpus.")

    # Blind review sheets: intentionally NO existing model/category column.
    top = global_domains.head(100).copy()
    base = pd.DataFrame({
        "rank": range(1, 101), "domain": top["domain"].tolist(),
        "posts": top["total_posts"].astype(int).tolist(),
        "sample_url_1": [examples[d][0] if len(examples[d]) > 0 else "" for d in top["domain"]],
        "sample_url_2": [examples[d][1] if len(examples[d]) > 1 else "" for d in top["domain"]],
        "sample_url_3": [examples[d][2] if len(examples[d]) > 2 else "" for d in top["domain"]],
        "reviewer_category": "", "evidence_note": "",
    })
    for who in ("A", "B"):
        target = out / f"top100_reviewer_{who}.csv"
        if not target.exists():
            base.to_csv(target, index=False, encoding="utf-8-sig")
            print("Created blind review sheet:", target)
        else:
            existing = pd.read_csv(target, dtype=str).fillna("")
            if existing["domain"].tolist() != base["domain"].tolist():
                raise RuntimeError(f"Review sheet {who} does not match current top100; do not overwrite filled labels.")
            print("Preserved existing review sheet:", target)

    pd.DataFrame([{**{"metric": metric}, "mapped": num, "total": den,
                   "coverage_pct": fraction(num, den)} for metric, (num, den) in checks.items()])\
      .to_csv(out / "coverage_reproduced.csv", index=False)
    (cov[(cov["metric"] == "fully_mapped_posts") & (cov["subreddit"] != "ALL")]
     .sort_values(["subreddit", "period"]))\
       .to_csv(out / "coverage_by_community_period.csv", index=False)

    # Actual site-domain presence shares, not same as broad category shares.
    site_month = []
    for (sub, month), den in sorted(month_counts.items()):
        for domain in SITES:
            n = domain_month_counts[sub, month, domain]
            site_month.append({"subreddit": sub, "month": month, "domain": domain,
                               "posts_with_domain": n, "link_posts": den,
                               "share_pct": fraction(n, den)})
    pd.DataFrame(site_month).to_csv(out / "h5_domain_monthly.csv", index=False)
    site_period = []
    for (sub, period), den in sorted(site_period_den.items()):
        for domain in SITES:
            n = site_period_counts[sub, period, domain]
            site_period.append({"subreddit": sub, "period": period, "domain": domain,
                                "posts_with_domain": n, "link_posts": den,
                                "share_pct": fraction(n, den)})
    pd.DataFrame(site_period).to_csv(out / "h5_domain_prepost.csv", index=False)

    # H5 site-level 2x2 association tests and A-style ITS, with pre-intervention placebos.
    historical = list(common.HISTORICAL)
    tests = []
    for sub in historical:
        for domain in SITES:
            pre_yes = site_period_counts[sub, "pre", domain]
            post_yes = site_period_counts[sub, "post", domain]
            pre_n = site_period_den[sub, "pre"]
            post_n = site_period_den[sub, "post"]
            matrix = np.array([[pre_yes, pre_n - pre_yes], [post_yes, post_n - post_yes]])
            if matrix.min() < 0 or matrix.sum() == 0 or (matrix.sum(axis=0) == 0).any():
                continue
            chi2, p, _, expected = chi2_contingency(matrix, correction=False)
            tests.append({"subreddit": sub, "domain": domain, "pre_yes": pre_yes, "pre_n": pre_n,
                          "post_yes": post_yes, "post_n": post_n, "pre_pct": fraction(pre_yes, pre_n),
                          "post_pct": fraction(post_yes, post_n), "delta_pp": fraction(post_yes, post_n) - fraction(pre_yes, pre_n),
                          "chi2": chi2, "p_value": p, "cramers_v": np.sqrt(chi2/matrix.sum()),
                          "min_expected": expected.min()})
    if tests:
        t = pd.DataFrame(tests)
        t["q_value"] = multipletests(t["p_value"], method="fdr_bh")[1]
        t.to_csv(out / "h5_domain_chi2.csv", index=False)
    topic_time = import_module("13_topics_time")
    tests = []
    monthly_frame = pd.DataFrame(site_month)
    for sub in historical:
        for domain in SITES:
            subset = monthly_frame[(monthly_frame.subreddit == sub) &
                                   (monthly_frame.domain == domain) &
                                   (monthly_frame.link_posts >= 100)].sort_values("month")
            series = subset.set_index("month")["share_pct"] / 100.0
            pre = series[series.index < common.INTERVENTION_MONTH]
            post = series[series.index >= common.INTERVENTION_MONTH]
            if len(pre) < 12 or len(post) < 12 or series.nunique() < 2:
                continue
            try:
                actual, _ = topic_time.fit_its(series, common.INTERVENTION_MONTH)
                fake_levels = [abs(topic_time.fit_its(pre, fm)[0]["beta_level"])
                               for fm in ("2021-06", "2021-12", "2022-06")]
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"Skipped H5 ITS {sub}/{domain}: {exc}")
                continue
            tests.append({"subreddit": sub, "domain": domain, **actual,
                          "max_placebo_abs_level": max(fake_levels),
                          "beats_all_placebos": bool(abs(actual["beta_level"]) > max(fake_levels))})
    if tests:
        t = pd.DataFrame(tests)
        q = multipletests(pd.concat([t["p_level"].fillna(1), t["p_slope"].fillna(1)]), method="fdr_bh")[1]
        t["q_level"], t["q_slope"] = q[:len(t)], q[len(t):]
        t.to_csv(out / "h5_domain_its.csv", index=False)

    # Category-spike audit: top three absolute adjacent-month shifts per subreddit/category.
    monthly = pd.read_csv(root / "results" / "f_category_monthly.csv")
    spikes = []
    for (sub, category), group in monthly.groupby(["subreddit", "category"]):
        group = group.sort_values("month").reset_index(drop=True)
        good = (group["link_posts"] >= 100) & (group["link_posts"].shift(1) >= 100)
        group["delta_pp"] = group["share_pct"].diff()
        shifts = group.loc[good & group["delta_pp"].notna()].copy()
        shifts["magnitude"] = shifts["delta_pp"].abs()
        for _, spike in shifts.nlargest(3, "magnitude").iterrows():
            month = str(spike["month"])
            contributors = sorted(
                [(d, value) for (s, m, cat, d), value in domain_month_weights.items()
                 if s == sub and m == month and cat == category],
                key=lambda x: -x[1])
            for rank, (dom, mass) in enumerate(contributors[:3], start=1):
                total_mass = category_month_weights[sub, month, category]
                spikes.append({"subreddit": sub, "category": category, "month": month,
                               "delta_pp_from_previous_month": spike["delta_pp"],
                               "category_share_pct": spike["share_pct"],
                               "link_posts": spike["link_posts"],
                               "domain_rank": rank, "domain": dom,
                               "contribution_to_category_pct": fraction(mass, total_mass),
                               "domain_weight": mass})
    pd.DataFrame(spikes).to_csv(out / "category_spikes_top_domains.csv", index=False)

    # F x r/MachineLearning author-provided title tags (visible, non-sticky only).
    linked_df = pd.DataFrame(linked_rows)
    if not linked_df.empty:
        visible = (pd.to_numeric(linked_df["is_removed"], errors="coerce").fillna(0).eq(0)
                   & pd.to_numeric(linked_df["stickied"], errors="coerce").fillna(0).eq(0))
        sample = linked_df[visible & linked_df["subreddit"].eq("MachineLearning") &
                           linked_df["title_tag"].isin(["research", "project", "discussion", "news"])].copy()
        rows = []
        for (period, tag), g in sample.groupby(["period", "title_tag"]):
            row = {"period": period, "title_tag": tag, "link_posts": len(g)}
            for c in ("academic", "code", "models/hubs"):
                row[f"pct_has_{c.replace('/', '_')}"] = 100.0 * g["categories"].apply(lambda xs: c in xs).mean()
            for dom in SITES:
                row[f"pct_has_{dom.replace('.', '_')}"] = 100.0 * g["domains"].apply(lambda xs: dom in xs).mean()
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "h4_tags_visible_machinelearning.csv", index=False)

        # Cross with A's preferred post-level LLM classifier if available.
        pred_path = root / "processed" / "llm_pred.parquet"
        if pred_path.exists():
            preds = pd.read_parquet(pred_path)
            if {"post_id", "llm_pred"}.issubset(preds.columns):
                joined = linked_df.merge(preds[["post_id", "llm_pred"]], on="post_id", validate="one_to_one")
                rows = []
                for (sub, period, is_llm), g in joined.groupby(["subreddit", "period", "llm_pred"]):
                    row = {"subreddit": sub, "period": period, "is_llm": is_llm, "link_posts": len(g)}
                    for category in ("academic", "code", "models/hubs"):
                        row[f"pct_has_{category.replace('/', '_')}"] = 100.0 * g["categories"].apply(lambda xs: category in xs).mean()
                    rows.append(row)
                pd.DataFrame(rows).to_csv(out / "h4_llm_vs_other_links.csv", index=False)
            else:
                print("Skipping LLM cross-analysis: llm_pred.parquet lacks post_id/llm_pred")
        else:
            print("Optional H4 LLM cross-analysis skipped: processed/llm_pred.parquet not found")

    # Explicitly record QC exceptions without modifying any category mapping.
    audit_path = root / "results" / "f_domain_work_4635_MERGED_ACCEPTED_with_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, low_memory=False).fillna("")
        if "link_validity" in audit.columns:
            exceptions = audit[(audit["link_validity"] == "suspected_false_positive") |
                               (audit.get("needs_url_review", pd.Series("", index=audit.index)).astype(str).str.lower().isin(["true", "1"]))].copy()
            exceptions = exceptions.merge(global_domains[["domain", "total_posts"]], on="domain", how="left")
            exceptions.to_csv(out / "quality_exceptions_with_impact.csv", index=False)
            print("Quality exception rows:", len(exceptions))
    print("Prepared Task F closeout outputs in", out)
    print("Next: two INDEPENDENT reviewers fill the last two columns of their own CSVs.")
    print("Then run: python task_f_closeout.py --agreement")


def agreement(root):
    out = root / "results" / "f_closeout"
    a_path, b_path = out / "top100_reviewer_A.csv", out / "top100_reviewer_B.csv"
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Run --prepare first; both review templates must exist")
    a = pd.read_csv(a_path, dtype=str).fillna("")
    b = pd.read_csv(b_path, dtype=str).fillna("")
    if a["domain"].tolist() != b["domain"].tolist() or len(a) != 100:
        raise ValueError("Reviewer sheets have different domains, order or wrong row count")
    labs_a = a["reviewer_category"].str.strip()
    labs_b = b["reviewer_category"].str.strip()
    for name, labs in [("A", labs_a), ("B", labs_b)]:
        invalid = labs[~labs.isin(VALID)]
        if not invalid.empty:
            raise ValueError(f"Reviewer {name}: {len(invalid)} missing/invalid labels; "
                             f"first rows: {list(invalid.index[:10]+2)}. Complete both independently first.")
    observed = float((labs_a == labs_b).mean())
    pa, pb = labs_a.value_counts(normalize=True), labs_b.value_counts(normalize=True)
    expected = sum(pa.get(c, 0) * pb.get(c, 0) for c in VALID)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    differences = a[["rank", "domain", "posts", "sample_url_1"]].copy()
    differences["reviewer_A"] = labs_a
    differences["reviewer_B"] = labs_b
    disagreements = differences[labs_a.ne(labs_b)].copy()
    disagreements["adjudicated_category"] = ""
    disagreements["adjudication_note"] = ""
    disagreements.to_csv(out / "top100_disagreements_to_adjudicate.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{"n": len(a), "agree_n": int((labs_a == labs_b).sum()),
                             "agreement_pct": observed * 100, "cohens_kappa": kappa,
                             "disagreements": len(disagreements)}])
    summary.to_csv(out / "top100_interannotator_agreement.csv", index=False)
    print(f"Independent top100 exact agreement: {observed*100:.2f}%")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Disagreements to reconcile: {len(disagreements)}")
    print("Results:", out / "top100_interannotator_agreement.csv")
    print("Review/adjudicate disagreements; compare adjudicated decisions with taxonomy, then re-run F if edits are necessary.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--agreement", action="store_true")
    parser.add_argument("--project", type=Path, default=Path("."))
    opts = parser.parse_args()
    root = opts.project.expanduser().resolve()
    if opts.prepare:
        prepare(root)
    else:
        agreement(root)


if __name__ == "__main__":
    main()

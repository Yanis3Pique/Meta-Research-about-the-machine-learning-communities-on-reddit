"""
Angle F — sources linked by ML Reddit posts around December 2022.

Run from the project with common.py, 13_topics_time.py, the project requirements
and tldextract (the packaged suffix snapshot is used offline). Input is
processed/modeling_corpus.csv and config/domain_categories.csv. A taxonomy
beside this script is also accepted if the config file is absent.

Outputs: processed/link_features.parquet (one row per corpus post for Angle B),
results/f_{domains_top,coverage,category_monthly,chi2,its}.csv and
results/figures/fig_f_link_categories.png.

Counting rules:
* Non-self posts contribute their primary URL only. Self posts contribute HTTP(S)
  or www links in the original body, in written order. Only documented Google
  AMP paths are unwrapped; no redirects are followed over the network. Reddit links retain the project's two explicit source classes.
* Individual links count occurrences (including repeated body destinations).
  Each canonical domain counts once per post for post-domain coverage. The
  primary audit covers non-self URL fields only. A fully mapped post must have
  at least one eligible link and every one must have a reviewed classification.
  Reviewed includes explicit 'other' entries and the built-in Reddit rules.
  This is host-classification coverage. A reviewed shortener or document host
  need not reveal the content type of its ultimate target.
* In f_domains_top, 'posts' is the within-subreddit count; 'total_posts' is the
  corpus-wide count for that domain, repeated on its subreddit rows. Take one
  row per domain when ranking globally. Posts can link to several domains.
* 'none' means no eligible URL. 'unclassified' means a valid source for which
  no review rule exists. A post with one known and two unknown domains splits
  its descriptive weight equally between the known class and 'unclassified'.
  Repeating a URL or adding another link in the same class gives no extra weight.
* Monthly shares and the main ITS use ALL eligible link-bearing posts. These
  measure the observed taxonomy mix. f_category_monthly also reports shares
  conditional on fully mapped posts to diagnose sensitivity to coverage.
  Chi-square uses the non-self primary or first eligible self-body link; an
  unreviewed representative is excluded even when a later URL is reviewed.
  Inspect exclusion counts and pre/post coverage before interpreting changes.
  No time comparison identifies a causal effect of ChatGPT.
"""

import argparse
import html
import importlib
import ipaddress
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

try:
    import tldextract
except ModuleNotFoundError as exc:
    raise ImportError("Angle F requires tldextract in addition to requirements.txt "
                      "(python -m pip install tldextract).") from exc

import common


CATEGORIES = [
    "academic", "code", "models/hubs", "video", "blog/tutorial", "social",
    "news/media", "company/product", "reddit-media", "reddit-internal", "other",
]
UNCLASSIFIED = "unclassified"
PLOT_CATEGORIES = [*CATEGORIES, UNCLASSIFIED]
TARGET_ITS = ["academic", "code", "models/hubs"]
MIN_MONTHLY_LINK_POSTS = 100
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
ALIASES = {"youtu.be": "youtube.com", "hf.co": "huggingface.co", "twitter.com": "x.com"}

# Google viewer domains from https://www.google.com/supported_domains .
# Check exact domains. A host merely containing "google" is never an AMP viewer.
GOOGLE_VIEWERS = frozenset("google." + suffix for suffix in """
ad ae al am as at az ba be bf bg bi bj bs bt by ca cat cd cf cg ch ci cl cm cn co.ao co.bw
co.ck co.cr co.id co.il co.in co.jp co.ke co.kr co.ls co.ma co.mz co.nz co.th co.tz co.ug
co.uk co.uz co.ve co.vi co.za co.zm co.zw com com.af com.ag com.ar com.au com.bd com.bh
com.bn com.bo com.br com.bz com.co com.cu com.cy com.do com.ec com.eg com.et com.fj com.gh
com.gi com.gt com.hk com.jm com.kh com.kw com.lb com.ly com.mm com.mt com.mx com.my com.na
com.ng com.ni com.np com.om com.pa com.pe com.pg com.ph com.pk com.pr com.py com.qa com.sa
com.sb com.sg com.sl com.sv com.tj com.tr com.tw com.ua com.uy com.vc com.vn cv cz de dj dk
dm dz ee es fi fm fr ga ge gg gl gm gr gy hn hr ht hu ie im iq is it je jo kg ki kz la li lk
lt lu lv md me mg mk ml mn mu mv mw ne nl no nr nu pl pn ps pt ro rs ru rw sc se sh si sk sm
sn so sr st td tg tl tm tn to tt vu ws
""".split())
CACHE_HOST = "cdn.ampproject.org"
AMP_VIEWER = re.compile(r"^/amp/(?:s/)?([^/]+)(/.*)?$")
AMP_CACHE = re.compile(r"^/(?:c|v|wp|i|ii)/(?:s/)?([^/]+)(/.*)?$")
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
NUMERIC_HOST = re.compile(r"(?:0x[\da-f]+|\d+)(?:\.(?:0x[\da-f]+|\d+))*")
URL_RE = re.compile(r"""(?i)(?<![\w./@=?&#%])(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>"'`]+""")
INLINE_LABEL = re.compile(r"!?(?<!\\)\[(?:\\.|[^\[\]\\\n])*\]\(\s*")
LOCAL_SUFFIXES = (".localhost", ".local", ".localdomain", ".internal", ".lan", ".home.arpa")
REDDIT_MEDIA = {"i.redd.it", "v.redd.it", "preview.redd.it", "external-preview.redd.it"}
PRESERVE_HOSTS = {"news.ycombinator.com", "proceedings.mlr.press"}


def load_mapping():
    """The reviewed CSV is authoritative. Never silently substitute a starter list."""
    path = common.CONFIG / "domain_categories.csv"
    if not path.exists():
        sibling = Path(__file__).with_name(path.name)
        if not sibling.is_file():
            raise FileNotFoundError(f"Missing reviewed taxonomy: {path} (or {sibling})")
        path = sibling
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"domain", "category"}.issubset(table.columns):
        raise ValueError(f"{path} needs domain,category columns")
    mapping = {}
    for row in table.itertuples(index=False):
        key = row.domain.strip().lower().rstrip(".").removeprefix("www.")
        key = ALIASES.get(key, key)
        category = row.category.strip()
        if not key or validated_host(key) != key or category not in CATEGORIES:
            raise ValueError(f"Invalid taxonomy row: {row.domain!r}, {category!r}")
        if key in mapping:
            raise ValueError(f"Duplicate taxonomy domain: {key}")
        mapping[key] = category
    return mapping


def dns_name(host):
    return ("." in host and len(host) <= 253
            and all(DNS_LABEL.fullmatch(label) for label in host.split(".")))


@lru_cache(maxsize=65536)
def validated_host(host):
    """Canonical public HTTP hostname or IP literal, without network lookups."""
    try:
        host = host.encode("idna").decode("ascii").lower().removesuffix(".")
    except UnicodeError:
        return None
    host = host.removeprefix("www.")
    if not host or host in {"localhost", "home.arpa"} or host.endswith(LOCAL_SUFFIXES):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not dns_name(host) or NUMERIC_HOST.fullmatch(host):
            return None
        return host
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast or "%" in host:
        return None
    return str(address)


def split_url(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = html.unescape(raw.strip())
    if any(mark in value for mark in ("\r", "\n", "\t")):
        return None
    if value.startswith("//"):
        value = "https:" + value
    elif not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        if re.match(r"^[a-z][a-z0-9+.-]*:", value, re.I) and not re.match(
                r"^[^/?#@:]+:[0-9]+(?:[/?#]|$)", value):
            return None
        # A bare email address in a URL field supplies no HTTP authority.
        if "@" in re.split(r"[/?#]", value, maxsplit=1)[0]:
            return None
        value = "https://" + value
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or re.search(r"[\\\s]", parts.netloc)):
            return None
        # urllib validates port syntax and range on attribute access.
        _ = parts.port
    except ValueError:
        return None
    return parts


@lru_cache(maxsize=65536)
def registered_domain(host):
    extracted = EXTRACTOR(host)
    if extracted.suffix and extracted.domain:
        return f"{extracted.domain}.{extracted.suffix}"
    return host


def normalize_domain(raw, mapping=None):
    """Return (canonical source, built-in class) or None for ineligible URLs."""
    parts = split_url(raw)
    if parts is None:
        return None
    host = validated_host(parts.hostname)
    if host is None:
        return None
    path = parts.path
    amp = None
    if host in GOOGLE_VIEWERS:
        amp = AMP_VIEWER.fullmatch(path)
    elif host == CACHE_HOST or host.endswith("." + CACHE_HOST):
        amp = AMP_CACHE.fullmatch(path)
    if amp:
        authority, article_path = amp.groups()
        # Only the literal publisher authority in a documented AMP route is
        # eligible. Malformed/opaque routes remain attributed to the proxy.
        if not re.search(r"[@%\\\s?#]", authority):
            dest = split_url("https://" + authority)
            if dest:
                candidate = dest.hostname.rstrip(".")
                publisher = validated_host(candidate)
                if publisher:
                    host, path = publisher, article_path or "/"
                elif (candidate == "localhost" or candidate.endswith(LOCAL_SUFFIXES)
                      or ":" in candidate or NUMERIC_HOST.fullmatch(candidate)
                      or dns_name(candidate)):
                    # A syntactically valid but local/non-global destination.
                    return None

    if host in REDDIT_MEDIA:
        return "reddit-media", "reddit-media"
    if host == "reddit.com" or host.endswith(".reddit.com"):
        if path.lower().startswith("/gallery/"):
            return "reddit-media", "reddit-media"
        return "reddit.com", "reddit-internal"
    if host == "redd.it" or host.endswith(".redd.it"):
        return "reddit.com", "reddit-internal"

    domain = registered_domain(host)
    if (host.endswith(".github.io") or domain in GOOGLE_VIEWERS or host in PRESERVE_HOSTS
            or (mapping is not None and host in mapping)):
        return ALIASES.get(host, host), None
    return ALIASES.get(domain, domain), None


def classify_url(url, mapping):
    normalized = normalize_domain(url, mapping)
    if normalized is None:
        return None
    domain, special = normalized
    category = special or mapping.get(domain, UNCLASSIFIED)
    return domain, category, category != UNCLASSIFIED


def body_urls(body):
    """Plain/autolink URLs and Markdown inline destinations in source order.

    Drop inline link labels (they can themselves look like URLs). Trim prose
    punctuation and unmatched closing delimiters, preserving IPv6 brackets
    and balanced parentheses in paths.
    """
    if not isinstance(body, str):
        return []
    text = INLINE_LABEL.sub(" ", html.unescape(body))
    urls = []
    for match in URL_RE.finditer(text):
        token = match.group().rstrip(".,;:!?")
        for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
            excess = max(0, token.count(closer) - token.count(opener))
            if excess:
                trailing = len(token) - len(token.rstrip(closer))
                token = token[:len(token) - min(trailing, excess)]
        urls.append(token)
    return urls


def is_self_post(value):
    flag = str(value).strip().lower()
    if flag in {"true", "1", "1.0", "yes", "t"}:
        return True
    if flag in {"false", "0", "0.0", "no", "f"}:
        return False
    raise ValueError(f"Missing or invalid is_self flag: {value!r}")


def post_urls(row):
    if is_self_post(row.is_self):
        return "body", body_urls(row.body)
    return "primary", [row.url]


def extract_features(corpus, mapping):
    """Build the Angle B features, domain audit and Reddit hostname cross-check."""
    if corpus["post_id"].isna().any() or corpus["post_id"].duplicated().any():
        raise ValueError("Expected one row per nonmissing, unique post_id")
    if corpus[["subreddit", "month"]].isna().any().any():
        raise ValueError("Missing subreddit or month would drop posts from denominators")

    records = []
    domain_counts = Counter()
    primary_counts = Counter()
    link_counts = Counter()
    domain_categories = {}
    audit_total = audit_mismatch = 0
    for row in corpus.itertuples(index=False):
        source, urls = post_urls(row)
        links = []
        for url in urls:
            item = classify_url(url, mapping)
            if item is not None:
                links.append(item)
        domains = {d for d, _, _ in links}
        categories = sorted({c for _, c, _ in links}, key=PLOT_CATEGORIES.index)
        n_reviewed = sum(reviewed for _, _, reviewed in links)
        n_reviewed_domains = len({d for d, _, reviewed in links if reviewed})
        is_primary = bool(links) and source == "primary"
        for domain, category, _ in links:
            domain_categories[domain] = category
            link_counts[row.subreddit, domain] += 1
        for domain in domains:
            domain_counts[row.subreddit, domain] += 1
            if is_primary:
                primary_counts[row.subreddit, domain] += 1

        if is_primary and isinstance(row.domain, str) and not row.domain.lower().startswith("self."):
            # The Reddit field identifies the transport host. Compare like
            # with like before AMP unwrapping and Reddit gallery reclassification.
            transport = split_url(row.url)._replace(path="", query="", fragment="")
            transport_source = normalize_domain(transport.geturl(), mapping)
            observed = normalize_domain(row.domain, mapping)
            if observed and transport_source:
                audit_total += 1
                audit_mismatch += int(observed[0] != transport_source[0])

        records.append({
            "post_id": row.post_id, "subreddit": row.subreddit, "month": row.month,
            "has_link": int(bool(links)), "link_category": links[0][1] if links else "none",
            "categories": categories, "n_categories": len(categories), "n_domains": len(domains),
            "fully_mapped": int(bool(links) and n_reviewed == len(links)),
            "n_links": len(links), "n_mapped_links": n_reviewed,
            "n_primary": int(is_primary), "n_mapped_primary": n_reviewed if is_primary else 0,
            "n_mapped_domains": n_reviewed_domains,
        })

    domain_columns = ["subreddit", "domain", "posts", "primary_posts", "body_posts",
                      "category", "mapped", "link_occurrences", "total_posts"]
    totals = Counter()
    for (_, domain), count in domain_counts.items():
        totals[domain] += count
    domain_rows = []
    for (subreddit, domain), posts in domain_counts.items():
        primary = primary_counts[subreddit, domain]
        category = domain_categories[domain]
        domain_rows.append([subreddit, domain, posts, primary, posts - primary, category,
                            int(category != UNCLASSIFIED), link_counts[subreddit, domain], totals[domain]])
    domains = pd.DataFrame(domain_rows, columns=domain_columns).sort_values(
        ["subreddit", "posts", "domain"], ascending=[True, False, True])
    audit = {"compared_primary_domains": audit_total, "domain_mismatches": audit_mismatch,
             "mismatch_pct": 100 * audit_mismatch / audit_total if audit_total else np.nan}
    columns = ["post_id", "subreddit", "month", "has_link", "link_category", "categories", "n_categories",
               "n_domains", "fully_mapped", "n_links", "n_mapped_links", "n_primary", "n_mapped_primary",
               "n_mapped_domains"]
    return pd.DataFrame.from_records(records, columns=columns), domains, audit


def coverage_table(features):
    """Four ratios of pooled counts (never averages of subgroup percentages)."""
    metrics = {
        "individual_links": ("n_mapped_links", "n_links"),
        "primary_links": ("n_mapped_primary", "n_primary"),
        "post_domains": ("n_mapped_domains", "n_domains"),
        "fully_mapped_posts": ("fully_mapped", "has_link"),
    }
    rows = []
    groups = [("ALL", features), *features.groupby("subreddit", sort=True)]
    for subreddit, group in groups:
        pre = group["month"] < common.INTERVENTION_MONTH
        for period, sample in [("all", group), ("pre", group[pre]), ("post", group[~pre])]:
            for name, (num, den) in metrics.items():
                mapped, total = int(sample[num].sum()), int(sample[den].sum())
                rows.append({"subreddit": subreddit, "period": period, "metric": name,
                             "mapped": mapped, "unmapped": total - mapped, "total": total,
                             "coverage_pct": 100 * mapped / total if total else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------
# Monthly category shares
# --------------------------------------------------

def monthly_shares(features):
    totals = (
        features.groupby(["subreddit", "month"])
        .agg(
            posts=("post_id", "size"),
            link_posts=("has_link", "sum"),
            fully_mapped_posts=("fully_mapped", "sum"),
        )
        .reset_index()
    )

    linked = features.loc[
        features["has_link"].eq(1),
        ["post_id", "subreddit", "month",
         "categories", "n_categories", "fully_mapped"],
    ].copy()

    # Multi-category posts contribute a total weight of 1.
    expanded = linked.explode("categories")
    expanded["weight"] = 1.0 / expanded["n_categories"]

    expanded["fully_mapped_weight"] = expanded["weight"] * expanded["fully_mapped"]

    counts = (
        expanded.groupby(["subreddit", "month", "categories"])
        .agg(
            category_weight=("weight", "sum"),
            fully_mapped_weight=("fully_mapped_weight", "sum"),
            posts_with_category=("post_id", "size"),
        )
        .reset_index()
        .rename(columns={"categories": "category"})
    )

    # Include zero-share categories in every observed month.
    grid = totals.merge(
        pd.DataFrame({"category": PLOT_CATEGORIES}),
        how="cross",
    )

    monthly = grid.merge(
        counts,
        on=["subreddit", "month", "category"],
        how="left",
    )

    monthly[["category_weight", "posts_with_category", "fully_mapped_weight"]] = (
        monthly[["category_weight", "posts_with_category", "fully_mapped_weight"]]
        .fillna(0)
    )

    denominator = monthly["link_posts"].replace(0, np.nan)

    monthly["share_pct"] = (
        100 * monthly["category_weight"] / denominator
    )

    monthly["fully_mapped_share_pct"] = (
        100 * monthly["fully_mapped_weight"] / monthly["fully_mapped_posts"].replace(0, np.nan)
    )
    monthly["fully_mapped_pct"] = 100 * monthly["fully_mapped_posts"] / denominator

    # Inclusive measure: a post can count in multiple categories.
    # These percentages need not sum to 100%.
    monthly["presence_pct"] = (
        100 * monthly["posts_with_category"] / denominator
    )

    monthly["link_rate_pct"] = (
        100 * monthly["link_posts"] / monthly["posts"]
    )

    return monthly


# --------------------------------------------------
# Pre/post chi-square tests
# --------------------------------------------------

def run_chi2(features):
    rows = []

    for subreddit in common.HISTORICAL:
        subset = features[
            features["subreddit"].eq(subreddit)
            & features["has_link"].eq(1)
        ].copy()

        subset["period"] = np.where(
            subset["month"] < common.INTERVENTION_MONTH,
            "pre",
            "post",
        )

        unknown = subset.loc[subset["link_category"].eq(UNCLASSIFIED), "period"].value_counts()
        reviewed = subset.loc[subset["link_category"].ne(UNCLASSIFIED)]
        table = pd.crosstab(
            reviewed["period"], reviewed["link_category"]
        ).reindex(index=["pre", "post"], fill_value=0)

        # Remove categories with zero observations.
        table = table.loc[:, table.sum(axis=0) > 0]

        if (
            table.shape[1] < 2
            or (table.sum(axis=1) == 0).any()
        ):
            continue

        chi2, p, dof, expected = chi2_contingency(
            table, correction=False
        )

        n = table.to_numpy().sum()

        # For a 2 x k table, min(r-1, c-1) = 1.
        cramers_v = np.sqrt(chi2 / n)

        rows.append({
            "subreddit": subreddit,
            "n_link_posts": n,
            "pre_posts": int(table.loc["pre"].sum()),
            "post_posts": int(table.loc["post"].sum()),
            "n_all_link_posts": len(subset),
            "pre_unclassified": int(unknown.get("pre", 0)),
            "post_unclassified": int(unknown.get("post", 0)),
            "min_expected": expected.min(),
            "cells_expected_lt5": int((expected < 5).sum()),
            "n_cells": expected.size,
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "cramers_v": cramers_v,
        })

    columns = ["subreddit", "n_link_posts", "pre_posts", "post_posts", "chi2", "dof",
               "p_value", "cramers_v", "n_all_link_posts", "pre_unclassified",
               "post_unclassified", "min_expected", "cells_expected_lt5", "n_cells"]
    tests = pd.DataFrame(rows, columns=columns)
    tests["q_value"] = (multipletests(tests["p_value"], method="fdr_bh")[1]
                        if not tests.empty else np.nan)
    return tests


# --------------------------------------------------
# Interrupted time series
# --------------------------------------------------

def estimable_its(topic_time, series, break_month):
    if series.nunique() < 2:
        return False
    design = topic_time.design(series.index, break_month)
    return len(series) > design.shape[1] and np.linalg.matrix_rank(design) == design.shape[1]


def run_its(monthly):
    """
    Reuse Angle A's ITS specification for consistency.

    The A module is imported without running its main pipeline,
    so doc_topics.parquet is NOT required.
    """
    topic_time = importlib.import_module("13_topics_time")

    rows = []

    for subreddit in common.HISTORICAL:
        for category in TARGET_ITS:
            subset = monthly[
                monthly["subreddit"].eq(subreddit)
                & monthly["category"].eq(category)
                & (monthly["link_posts"] >= MIN_MONTHLY_LINK_POSTS)
            ].sort_values("month")

            series = (
                subset.set_index("month")["share_pct"] / 100
            )

            pre = series[
                series.index < common.INTERVENTION_MONTH
            ]
            post = series[
                series.index >= common.INTERVENTION_MONTH
            ]

            if (len(pre) < 12 or len(post) < 12
                    or not estimable_its(topic_time, series, common.INTERVENTION_MONTH)):
                print(
                    f"Skipping ITS: {subreddit} / {category}"
                    " (insufficient months or variation)"
                )
                continue

            result, _ = topic_time.fit_its(
                series, common.INTERVENTION_MONTH
            )

            # Compare the actual level jump to placebo breaks.
            placebo_levels = []

            for fake_month in topic_time.PLACEBO_MONTHS:
                if not estimable_its(topic_time, pre, fake_month):
                    continue
                placebo, _ = topic_time.fit_its(
                    pre, fake_month
                )
                placebo_levels.append(
                    abs(placebo["beta_level"])
                )

            all_placebos = len(placebo_levels) == len(topic_time.PLACEBO_MONTHS)
            max_placebo = max(placebo_levels) if all_placebos else np.nan

            rows.append({
                "subreddit": subreddit,
                "category": category,
                **result,
                "n_placebos": len(placebo_levels),
                "max_placebo_abs_level": max_placebo,
                "beats_all_placebos": (
                    abs(result["beta_level"]) > max_placebo if all_placebos else pd.NA
                ),
            })

    columns = ["subreddit", "category", "n_months", "beta_trend", "beta_level", "p_level",
               "beta_slope", "p_slope", "pre_mean_pct", "last_month", "fitted_last_pct",
               "counterfactual_last_pct", "gap_pp", "r2", "max_placebo_abs_level",
               "beats_all_placebos", "n_placebos"]
    its = pd.DataFrame(rows, columns=columns)

    if not its.empty:
        pvals = np.concatenate([
            its["p_level"].fillna(1).to_numpy(),
            its["p_slope"].fillna(1).to_numpy(),
        ])

        qvals = multipletests(
            pvals, method="fdr_bh"
        )[1]

        n = len(its)
        its["q_level"] = qvals[:n]
        its["q_slope"] = qvals[n:]

    else:
        its["q_level"] = np.nan
        its["q_slope"] = np.nan
    return its


# --------------------------------------------------
# Figure
# --------------------------------------------------

def plot_categories(monthly):
    colors = [
        "#2563eb", "#0d9488", "#8b5cf6",
        "#eab308", "#f97316", "#ec4899",
        "#64748b", "#dc2626", "#94a3b8",
        "#475569", "#d1d5db", "#222222",
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7),
        sharex=True, sharey=True,
    )

    handles = None

    for ax, subreddit in zip(
        axes.flat, common.SUBREDDITS
    ):
        subset = monthly[
            monthly["subreddit"].eq(subreddit)
        ]

        pivot = subset.pivot(
            index="month",
            columns="category",
            values="share_pct",
        ).reindex(columns=PLOT_CATEGORIES, fill_value=0)

        dates = pd.to_datetime(
            pivot.index + "-01"
        )

        if not pivot.empty:
            handles = ax.stackplot(
                dates,
                *[
                    pivot[category].to_numpy(dtype=float)
                    for category in PLOT_CATEGORIES
                ],
                colors=colors,
                alpha=0.9,
            )

        ax.set_title(
            f"r/{subreddit}",
            loc="left",
            fontsize=11,
        )

        if subreddit in common.HISTORICAL:
            ax.axvline(
                pd.Timestamp(common.INTERVENTION_MONTH + "-01"),
                color="black",
                linestyle="--",
                linewidth=1,
            )

        ax.set_ylim(0, 100)
        ax.grid(
            axis="y", alpha=0.2
        )
        ax.set_axisbelow(True)

        ax.xaxis.set_major_locator(
            mdates.YearLocator()
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )

    for ax in axes[:, 0]:
        ax.set_ylabel(
            "Share of link-bearing posts (%)"
        )

    if handles:
        fig.legend(
            handles,
            PLOT_CATEGORIES,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=8,
        )

    fig.suptitle(
        "Link categories in ML Reddit communities\n"
        "Dashed line: ChatGPT release",
        fontsize=13,
    )

    fig.tight_layout(
        rect=(0, 0.13, 1, 0.92)
    )

    output = (
        common.FIGURES
        / "fig_f_link_categories.png"
    )

    fig.savefig(
        output, dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved figure: {output}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    common.RESULTS.mkdir(
        parents=True, exist_ok=True
    )
    common.FIGURES.mkdir(
        parents=True, exist_ok=True
    )

    print("Loading modeling corpus...")

    corpus = common.load_corpus([
        "is_self", "url", "body", "domain"
    ])

    print(f"Loaded {len(corpus):,} posts")

    mapping = load_mapping()

    print("Extracting and classifying links...")

    features, domains, audit = extract_features(
        corpus, mapping
    )

    features.to_parquet(
        common.PROCESSED / "link_features.parquet",
        index=False,
    )

    domains.to_csv(
        common.RESULTS / "f_domains_top.csv",
        index=False,
    )

    coverage = coverage_table(features)
    coverage.round(5).to_csv(common.RESULTS / "f_coverage.csv", index=False)

    monthly = monthly_shares(features)

    monthly.round(5).to_csv(
        common.RESULTS / "f_category_monthly.csv",
        index=False,
    )

    chi2 = run_chi2(features)

    chi2.to_csv(
        common.RESULTS / "f_chi2.csv",
        index=False,
    )

    its = run_its(monthly)

    its.to_csv(
        common.RESULTS / "f_its.csv",
        index=False,
    )

    plot_categories(monthly)

    linked = features[
        features["has_link"].eq(1)
    ]

    overall = coverage.query("subreddit == 'ALL' and period == 'all'").set_index("metric")
    overall = overall[["mapped", "unmapped", "total", "coverage_pct"]]
    print(f"\nTotal posts: {len(features):,}; eligible link-bearing posts: {len(linked):,}")
    print(overall.round(2).to_string(na_rep="undefined"))
    print(f"Reddit-domain mismatch: {audit['mismatch_pct']:.2f}% "
          f"({audit['domain_mismatches']}/{audit['compared_primary_domains']})")
    full = overall.loc["fully_mapped_posts"]
    if not full["total"]:
        print("No eligible links; the 95% fully mapped post target cannot be evaluated.")
    elif full["coverage_pct"] < 95:
        remaining = domains[domains["mapped"].eq(0)].drop_duplicates("domain")
        top = remaining.sort_values(["total_posts", "domain"], ascending=[False, True])
        print(f"Below 95% fully mapped: {int(full['unmapped']):,} distinct posts have "
              f"at least one of {len(top):,} unclassified domains.")
        print("Most frequent unclassified domains (posts can overlap):")
        print(top[["domain", "total_posts"]].head(20).to_string(index=False))
        print("Full counts are in f_domains_top.csv. Use --review-domain HOST for local URL examples.")
    else:
        print("The 95% fully mapped link-bearing post target was met in this input.")

    if not chi2.empty:
        print("\nPRE/POST CHI-SQUARE")
        print(
            chi2[[
                "subreddit", "n_link_posts",
                "p_value", "cramers_v"
            ]].round(4).to_string(index=False)
        )

    if not its.empty:
        print("\nITS — EXPLORATORY")
        print(
            its[[
                "subreddit", "category",
                "beta_level", "q_level",
                "gap_pp", "beats_all_placebos"
            ]].round(4).to_string(index=False)
        )

    print("\nDone.")


def review_examples(domain):
    """Print up to ten distinct observed URLs locally. Aggregates never contain post text or URLs."""
    mapping = load_mapping()
    corpus = common.load_corpus(["is_self", "url", "body"])
    seen = set()
    for row in corpus.itertuples(index=False):
        _, urls = post_urls(row)
        for url in urls:
            item = classify_url(url, mapping)
            if item and item[0] == domain and url not in seen:
                seen.add(url)
                print(url)
                if len(seen) == 10:
                    return
    if not seen:
        print(f"No eligible links for {domain}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Angle F link sources and coverage")
    parser.add_argument("--review-domain", metavar="HOST",
                        help="print local examples for a canonical f_domains_top domain and exit")
    args = parser.parse_args()
    if args.review_domain:
        review_examples(args.review_domain.strip().lower())
    else:
        main()

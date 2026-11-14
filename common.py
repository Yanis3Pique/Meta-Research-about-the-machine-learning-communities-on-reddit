"""
Shared settings and helpers for the analysis scripts (see docs/research_design.md).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROCESSED = BASE / "processed"
RESULTS = BASE / "results"
CONFIG = BASE / "config"
FIGURES = RESULTS / "figures"

CORPUS = PROCESSED / "modeling_corpus.csv"
TITLES_LEMMA = PROCESSED / "titles_lemma.parquet"

RANDOM_STATE = 42

SUBREDDITS = [
    "MachineLearning",
    "learnmachinelearning",
    "deeplearning",
    "LocalLLaMA",
]
HISTORICAL = SUBREDDITS[:3]

# Main intervention: ChatGPT released 30 Nov 2022 -> post period starts Dec 2022.
INTERVENTION_MONTH = "2022-12"

EVENTS = {
    "2022-08": "Stable Diffusion",
    "2022-12": "ChatGPT",
    "2023-03": "GPT-4 / LLaMA",
    "2023-06": "API blackout",
    "2023-07": "Llama 2",
}

# Score capture regime change (docs/research_design.md §1.6).
UNRELIABLE_SCORE_MONTHS = ["2023-06", "2023-07", "2023-08", "2023-09", "2023-10"]


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_corpus(columns=None):
    """Modeling corpus with month/quarter/time variables added."""
    base_cols = ["post_id", "subreddit", "created_utc"]
    usecols = None if columns is None else list(dict.fromkeys(base_cols + columns))

    df = pd.read_csv(CORPUS, usecols=usecols, low_memory=False)

    date = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["month"] = date.dt.strftime("%Y-%m")
    df["quarter"] = date.dt.year.astype(str) + "Q" + date.dt.quarter.astype(str)
    df["month_idx"] = (date.dt.year - 2021) * 12 + date.dt.month - 1
    df["post"] = (df["month"] >= INTERVENTION_MONTH).astype(int)

    return df


# --------------------------------------------------
# Title preprocessing
# --------------------------------------------------

TAG_RE = re.compile(r"^\s*[\[(]\s*[a-z]+\s*[\])]\s*", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
# Keep model names together: gpt-4 -> gpt4, llama-2 -> llama2, gpt-3.5 -> gpt35
VERSION_RE = re.compile(r"\b([a-z]+)[\s-]?(\d+)(?:\.(\d+))?\b")
KEEP_VERSIONED = {"gpt", "llama", "yolo", "mistral", "phi", "qwen", "gemma", "claude", "sd"}


def clean_title(title):
    title = str(title or "").lower()
    title = URL_RE.sub(" ", title)
    title = TAG_RE.sub("", title)

    def join_version(match):
        name, major, minor = match.groups()
        if name in KEEP_VERSIONED:
            return f"{name}{major}{minor or ''}"
        return match.group(0)

    return VERSION_RE.sub(join_version, title)


# spaCy lemmas that hurt readability of topics.
LEMMA_FIXES = {"datum": "data", "medium": "media"}


def lemmatize_titles(titles, n_process=4, batch_size=2000):
    """spaCy lemmas, lowercase, alphanumeric tokens only."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    out = []

    for doc in nlp.pipe(titles, n_process=n_process, batch_size=batch_size):
        tokens = []
        for tok in doc:
            if not any(ch.isalpha() for ch in tok.text):
                continue
            # The lemmatizer mangles acronyms/model names; keep them as written.
            lemma = tok.text if tok.text.isupper() or any(ch.isdigit() for ch in tok.text) else tok.lemma_
            lemma = lemma.lower().strip("-.'")
            lemma = LEMMA_FIXES.get(lemma, lemma)
            if len(lemma) >= 2:
                tokens.append(lemma)
        out.append(" ".join(tokens))

    return out


def load_lemmatized_titles():
    """Cached lemmatized titles aligned with the corpus (post_id, lemma)."""
    if TITLES_LEMMA.exists():
        return pd.read_parquet(TITLES_LEMMA)

    df = pd.read_csv(CORPUS, usecols=["post_id", "title"], low_memory=False)
    cleaned = [clean_title(t) for t in df["title"].fillna("")]
    lemmas = lemmatize_titles(cleaned)

    result = pd.DataFrame({"post_id": df["post_id"], "lemma": lemmas})
    result.to_parquet(TITLES_LEMMA, index=False)
    return result


def stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    domain = (CONFIG / "stopwords_domain.txt").read_text().split()
    return sorted(set(ENGLISH_STOP_WORDS) | set(domain))


# --------------------------------------------------
# Topic weights
# --------------------------------------------------

def normalize_rows(W):
    """Row-normalise document-topic weights; all-zero rows stay zero."""
    sums = W.sum(axis=1, keepdims=True)
    return np.divide(W, sums, out=np.zeros_like(W), where=sums > 0)

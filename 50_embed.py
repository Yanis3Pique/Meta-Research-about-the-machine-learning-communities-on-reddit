"""
Sentence embeddings of all titles (GPU recommended).

Usage:
  python 50_embed.py export              # local: write processed/titles_for_embedding.parquet
  python 50_embed.py embed [MODEL]       # GPU machine: write processed/embeddings_<name>.npy

Embeddings are L2-normalised, float16, row-aligned with processed/modeling_corpus.csv.
Default model: Qwen/Qwen3-Embedding-0.6B (fallback: BAAI/bge-base-en-v1.5).
"""

import re
import sys
import time

import numpy as np
import pandas as pd

import common

EXPORT = common.PROCESSED / "titles_for_embedding.parquet"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

TAG_RE = re.compile(r"^\s*[\[(]\s*[a-z]+\s*[\])]\s*", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)


def model_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.split("/")[-1].lower()).strip("-")


def embeddings_path(name=DEFAULT_MODEL):
    return common.PROCESSED / f"embeddings_{model_slug(name)}.npy"


def export():
    df = common.load_corpus(["title", "year"])
    # Keep casing and wording; only strip [D]/[R] tags and URLs.
    text = (df["title"].fillna("").astype(str)
            .str.replace(URL_RE, " ", regex=True)
            .str.replace(TAG_RE, "", regex=True)
            .str.strip())
    out = pd.DataFrame({"post_id": df["post_id"], "subreddit": df["subreddit"],
                        "year": df["year"], "text": text})
    out.to_parquet(EXPORT, index=False)
    print(f"Wrote {len(out):,} titles to {EXPORT}")


def embed(model_name):
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(EXPORT)
    texts = df["text"].tolist()
    print(f"{len(texts):,} titles | model {model_name} | "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 64       # titles: >99% fit

    # Sort by length so batches have similar padding -> much faster.
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    emb = model.encode([texts[i] for i in order], batch_size=512, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    out = np.empty_like(emb)
    out[order] = emb
    np.save(embeddings_path(model_name), out.astype(np.float16))
    print(f"Saved {out.shape} to {embeddings_path(model_name)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "embed"}:
        sys.exit(__doc__)
    if sys.argv[1] == "export":
        export()
    else:
        embed(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL)

"""
A §2 validation — an LLM annotator as an independent check of the topic measure.

Steps (run in order):
  python 60_llm_annotate.py sample      # local: stratified random sample + hand-label sheet
  python 60_llm_annotate.py annotate    # GPU server: label the sample via llama-server
  python 60_llm_annotate.py evaluate    # local: agreement with NMF / BERTopic / hand labels

Sample: N_SAMPLE titles drawn at random from the 3 historical subreddits,
stratified equally by subreddit x year (12 cells). Because it is random within
cells, LLM-label shares estimate the true LLM share per cell directly.

Annotator: local open model served by llama.cpp (llama-server, OpenAI-compatible
API), temperature 0, thinking disabled, output constrained by JSON schema.

Files (post-level text stays in processed/, never committed):
  processed/llm_sample.parquet           sample (post_id, subreddit, year, text)
  processed/llm_labels.jsonl             one LLM label per line (raw)
  processed/hand_labels.csv              200-title sheet to fill in by hand
Aggregate outputs:
  results/a_llm_validation.csv           agreement metrics
  results/a_llm_shares.csv               LLM share per subreddit-year: annotator vs NMF vs BERTopic
"""

import json
import re
import sys
import time

import numpy as np
import pandas as pd

import common

N_SAMPLE = 2000
N_HAND = 200
SERVER = "http://127.0.0.1:8080"
MODEL_NAME = "Qwen3.6-27B-IQ4_XS (llama.cpp)"
PROMPT_VERSION = "v1"

SAMPLE_PATH = common.PROCESSED / "llm_sample.parquet"
LABELS_PATH = common.PROCESSED / "llm_labels.jsonl"
HAND_PATH = common.PROCESSED / "hand_labels.csv"

POST_TYPES = ["question", "project", "news", "discussion", "resource", "career", "research", "other"]

SYSTEM = """You annotate titles of Reddit posts from machine-learning communities (2021-2024).
Return JSON only.

Field "llm_genai" (true/false): true if the title is mainly about large language models or
generative AI. This includes: LLMs and chatbots (ChatGPT, GPT-3/4, LLaMA, Mistral, Claude,
Gemini, etc.), prompting, RAG, LLM agents, fine-tuning or running LLMs, text/image/audio/video
generation (diffusion models, Stable Diffusion, DALL-E, Midjourney, GANs used for generation),
and news or opinions about generative AI. It is false for other ML topics (classic deep learning,
computer vision, tabular data, time series, reinforcement learning, general ML math or careers)
even if they mention "AI" in passing. Generic "AI" with no sign of generative models -> false.

Field "post_type", exactly one of:
  question   - asks for help, explanation or a recommendation
  project    - shows something the author built or made
  news       - announces a release, event or news item
  discussion - opinion or open debate
  resource   - shares a tutorial, course, article, dataset or tool by others
  career     - about jobs, studying, careers, degrees
  research   - presents or summarizes a research paper
  other      - none of the above or unclear"""

SCHEMA = {
    "type": "object",
    "properties": {
        "llm_genai": {"type": "boolean"},
        "post_type": {"type": "string", "enum": POST_TYPES},
    },
    "required": ["llm_genai", "post_type"],
}


# --------------------------------------------------
# 1. Sample (local)
# --------------------------------------------------

def sample():
    titles = pd.read_parquet(common.PROCESSED / "titles_for_embedding.parquet")
    hist = titles[titles["subreddit"].isin(common.HISTORICAL)]
    per_cell = N_SAMPLE // (len(common.HISTORICAL) * 4)
    s = (hist[hist["text"].str.len() > 0]
         .groupby(["subreddit", "year"])
         .sample(per_cell, random_state=common.RANDOM_STATE)
         .sample(frac=1, random_state=common.RANDOM_STATE)      # shuffle order
         .reset_index())
    s = s.rename(columns={"index": "row"})
    s.to_parquet(SAMPLE_PATH, index=False)

    hand = s.groupby(["subreddit", "year"]).head(N_HAND // 12 + 1).head(N_HAND)
    hand = hand[["post_id", "subreddit", "year", "text"]].assign(
        llm_genai="", post_type="", annotator="", note="")
    if HAND_PATH.exists():
        print(f"{HAND_PATH} exists — not overwritten")
    else:
        hand.to_csv(HAND_PATH, index=False)
    print(f"Sample: {len(s):,} titles ({per_cell} per subreddit-year) -> {SAMPLE_PATH}")
    print(f"Hand-label sheet: {len(hand)} titles -> {HAND_PATH}")
    print("Fill llm_genai with yes/no and post_type with one of: " + ", ".join(POST_TYPES))


# --------------------------------------------------
# 2. Annotate (server, llama-server running)
# --------------------------------------------------

def annotate(n_workers=4):
    import concurrent.futures as cf
    import urllib.request

    s = pd.read_parquet(SAMPLE_PATH)
    done = set()
    if LABELS_PATH.exists():
        done = {json.loads(l)["post_id"] for l in LABELS_PATH.read_text().splitlines() if l.strip()}
    todo = s[~s["post_id"].isin(done)]
    print(f"{len(done):,} already labelled, {len(todo):,} to go")

    def call(row):
        body = {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Title: {row.text}"}],
            "temperature": 0,
            "max_tokens": 64,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "label", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(f"{SERVER}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out = json.loads(r.read())
                content = out["choices"][0]["message"]["content"]
                label = json.loads(content)
                return {"post_id": row.post_id, **label, "raw": content,
                        "model": MODEL_NAME, "prompt_version": PROMPT_VERSION}
            except Exception as e:      # retry transient errors
                err = repr(e)
                time.sleep(2 * (attempt + 1))
        return {"post_id": row.post_id, "error": err,
                "model": MODEL_NAME, "prompt_version": PROMPT_VERSION}

    t0 = time.time()
    with open(LABELS_PATH, "a") as f, cf.ThreadPoolExecutor(n_workers) as ex:
        for i, res in enumerate(ex.map(call, todo.itertuples()), 1):
            f.write(json.dumps(res) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"  {i:,}/{len(todo):,}  {i / (time.time() - t0):.1f} titles/s", flush=True)
    print(f"Done in {time.time() - t0:.0f}s")


# --------------------------------------------------
# 3. Evaluate (local)
# --------------------------------------------------

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def prf(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = (pred & gold).sum()
    p = tp / pred.sum() if pred.sum() else np.nan
    r = tp / gold.sum() if gold.sum() else np.nan
    return p, r, 2 * p * r / (p + r) if p + r else np.nan


def evaluate():
    s = pd.read_parquet(SAMPLE_PATH)
    lab = pd.read_json(LABELS_PATH, lines=True)
    lab = lab[lab.get("error").isna()] if "error" in lab else lab
    lab = lab.drop_duplicates("post_id", keep="last")
    df = s.merge(lab[["post_id", "llm_genai", "post_type"]], on="post_id")
    print(f"Labelled: {len(df):,}/{len(s):,}")

    # NMF: LLM-family weight; BERTopic: mapped family
    labels = pd.read_csv(common.CONFIG / "topic_labels.csv")
    llm_ids = labels.loc[labels["family"] == "llm_genai", "topic_id"]
    nmf = pd.read_parquet(common.PROCESSED / "doc_topics.parquet")
    nmf_w = nmf[[f"t{i:02d}" for i in llm_ids]].sum(axis=1).values
    topic_cols = [c for c in nmf.columns if re.fullmatch(r"t\d+", c)]
    nmf_dom = pd.Series(np.argmax(nmf[topic_cols].values, axis=1)).map(
        dict(zip(labels["topic_id"], labels["family"]))).values
    bt = pd.read_parquet(common.PROCESSED / "bertopic_doc_topics.parquet")
    bt_map = pd.read_csv(common.RESULTS / "a_bertopic_family_map.csv").set_index("topic_id")["family"]
    bt_fam = bt["topic"].map(bt_map).values

    df["nmf_weight"] = nmf_w[df["row"]]
    df["nmf_dominant"] = nmf_dom[df["row"]] == "llm_genai"
    df["bertopic"] = bt_fam[df["row"]] == "llm_genai"
    gold_llm = df["llm_genai"].astype(bool)

    rows = []

    def add(name, pred, gold, gold_name):
        p, r, f = prf(pred, gold)
        rows.append({"comparison": f"{name} vs {gold_name}", "n": len(gold),
                     "precision": p, "recall": r, "f1": f, "kappa": kappa(pred, gold),
                     "pred_share": np.mean(pred), "gold_share": np.mean(gold)})

    add("NMF dominant family", df["nmf_dominant"], gold_llm, "LLM annotator")
    for cut in (0.3, 0.5):
        add(f"NMF LLM weight >= {cut}", df["nmf_weight"] >= cut, gold_llm, "LLM annotator")
    add("BERTopic mapped family", df["bertopic"], gold_llm, "LLM annotator")
    # Correlation of continuous NMF weight with LLM label (point-biserial)
    r_pb = np.corrcoef(df["nmf_weight"], gold_llm)[0, 1]

    # Hand labels, if filled in
    if HAND_PATH.exists():
        hand = pd.read_csv(HAND_PATH, dtype=str).fillna("")
        hand = hand[hand["llm_genai"].str.lower().isin(["yes", "no"])]
        if len(hand):
            h = hand.merge(df, on="post_id", suffixes=("_hand", ""))
            gold = h["llm_genai_hand"].str.lower().eq("yes")
            add("LLM annotator", h["llm_genai"].astype(bool), gold, "hand labels")
            add("NMF dominant family", h["nmf_dominant"], gold, "hand labels")
            add("BERTopic mapped family", h["bertopic"], gold, "hand labels")
            pt = h[h["post_type_hand"].isin(POST_TYPES)]
            if len(pt):
                rows.append({"comparison": "post_type: LLM annotator vs hand labels",
                             "n": len(pt), "accuracy": (pt["post_type"] == pt["post_type_hand"]).mean()})
            print(f"Hand labels used: {len(h)}")

    val = pd.DataFrame(rows)
    val.round(4).to_csv(common.RESULTS / "a_llm_validation.csv", index=False)

    shares = (df.assign(llm=gold_llm)
              .groupby(["subreddit", "year"])
              .agg(n=("llm", "size"), llm_annotator=("llm", "mean"),
                   nmf_dominant=("nmf_dominant", "mean"), nmf_mean_weight=("nmf_weight", "mean"),
                   bertopic=("bertopic", "mean"))
              .mul(100).reset_index())
    shares["n"] = shares["n"] / 100
    shares.round(2).to_csv(common.RESULTS / "a_llm_shares.csv", index=False)

    ptype = (pd.crosstab([df["subreddit"], df["year"]], df["post_type"], normalize="index")
             .mul(100).round(1))
    ptype.to_csv(common.RESULTS / "a_llm_post_types.csv")

    pd.set_option("display.width", 200)
    print(val.round(3).to_string(index=False))
    print(f"\nPoint-biserial r (NMF LLM weight vs annotator label): {r_pb:.3f}")
    print("\nLLM share (%) per subreddit-year")
    print(shares.round(1).to_string(index=False))
    print("\nPost types (%)")
    print(ptype.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sample":
        sample()
    elif cmd == "annotate":
        annotate()
    elif cmd == "evaluate":
        evaluate()
    else:
        sys.exit(__doc__)

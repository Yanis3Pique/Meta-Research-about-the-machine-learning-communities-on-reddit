# Research Design — ML Communities on Reddit (A + B + D + F)

Working document. Status: draft, 14 Nov 2026.
Deadlines: preliminary results on the "Project" channel by **Tue 1 Dec 2026**; ACL-format paper (6–8 pages + unlimited appendix) at the exam session.

---

## 0. The paper in one paragraph

> Between 2021 and 2024, machine-learning communities on Reddit were reshaped by the arrival of consumer LLMs. We show **what** people discussed (A: topics), **what they pointed to** (F: link sources), **what the community rewarded** (B: engagement), and **whether communities became alike** (D: convergence), using a shared topic model, interrupted time series around ChatGPT's release, and controls for community growth.

Every analysis must serve this story. If a result does not fit, it goes to the appendix or "negative results", not the main text.

### Hypotheses (pre-registered for ourselves — write them down before seeing results)

| ID | Hypothesis | Angle | Test |
|---|---|---|---|
| H1 | LLM-related topic prevalence jumps and/or accelerates after Nov 2022 in all three historical subreddits | A | ITS level (β2) / slope (β3) > 0 |
| H2 | The shift is largest in r/MachineLearning and smallest in r/learnmachinelearning (beginners keep asking beginner questions) | A | Compare β2 across subreddits |
| H3 | Classic deep-learning topics (CNNs, object detection, RL) decline in relative share | A | ITS β2/β3 < 0 |
| H4 | Research content ([R] tag, arXiv links) loses share to applied content (code/model links, [P] tag) | A, F | ITS on shares |
| H5 | HuggingFace and GitHub link shares rise; arXiv share falls or stagnates | F | ITS + χ² |
| H6 | After ChatGPT, LLM posts receive an engagement premium relative to non-LLM posts in the same subreddit-month | B | topic × post interaction > 0 |
| H7 | Historical subreddits become more similar to each other (lower JSD, lower subreddit-classifier accuracy) | D | trend test on quarterly series |
| H8 | Historical subreddits move toward LocalLLaMA's 2023 profile | D | distance-to-reference decreases |

Rejecting a hypothesis is a valid result. H2 and H6 in particular could go either way — that is fine.

---

## 1. Step 0 — Data foundation (do this before any angle)

### 1.1 What the current pipeline does (audit of the code)

From `clean_data.py` and `build_corpus.py`:

- Kept fields: `post_id, subreddit, created_utc, year, month, title, body, text, has_body, is_self, eligible_50, score, num_comments, flair, url`.
- **Moderator-removed posts are kept and are indistinguishable from title-only posts.** `normalize_text()` turns a body of `[removed]` into `""`, so `has_body = 0` mixes (a) link/title-only posts and (b) posts removed by moderators whose title survived. Only posts whose *title* is a placeholder get dropped in `build_corpus.py`.
- **Not kept** but present in the raw JSON and needed:
  - `removed_by_category` (moderator / automod / deleted / reddit) → needed to define the "visible" sample (B) and to measure moderation as a confound (A, D).
  - `retrieved_on` / `retrieved_utc` (name depends on the dump) → needed to check when `score` was captured (B validity gate).
  - `domain` → Reddit's own domain field, a cross-check for F.
  - `upvote_ratio`, `stickied`, `over_18`, `author` (only to count unique authors per month — never published).
- Deduplication is within subreddit-year only (cross-posts across subreddits survive, which is fine for per-community analysis but must be remembered for D: identical cross-posted titles inflate similarity).
- r/MachineLearning titles carry tags `[D] [R] [P] [N]` (and spelled-out variants). The default TF-IDF tokenizer drops them (single letters), so they are **currently unused** — they are a free, author-provided post-type label.

### 1.2 Action: extend the extraction

Modify `clean_data.py` to also write: `removed_by_category`, `retrieved_on` (whichever key exists), `domain`, `upvote_ratio`, `stickied`, `over_18`, `author_hash` (salted SHA-256 of author, salt not committed), `title_tag` (parsed `[D]/[R]/[P]/[N]` → `discussion/research/project/news/none`).

Add `is_removed = removed_by_category not empty or body in REMOVED_VALUES` **before** the body is blanked.

### 1.3 Analysis samples (name them and use the names in the paper)

| Sample | Definition | Used for |
|---|---|---|
| `S_submitted` | current modeling corpus (224,300) | A, D, F — "what people tried to post" |
| `S_visible` | `S_submitted` minus `is_removed` minus `stickied` | B (removed posts have meaningless engagement); robustness for A, D |
| `S_balanced(q)` | equal-size random sample per subreddit per quarter | D (divergence and classifiers are size-sensitive) |
| `S_train` | current balanced sample for fitting topic models (≤3,000 per subreddit-year) | A |

Report A and D on both `S_submitted` and `S_visible`. If they disagree, the difference *is* a finding about moderation.

### 1.4 Time variables

- `month_idx` = months since Jan 2021 (0…47).
- `post` = 1 if created on/after **1 Dec 2022** (ChatGPT released 30 Nov 2022).
- `t_since` = max(0, month_idx − 23).
- `quarter` = 2021Q1…2024Q4 (16 points) for D.

Event calendar to mark on all time-series plots (vertical dashed lines, labelled in the caption):

| Date | Event |
|---|---|
| Aug 2022 | Stable Diffusion public release |
| 30 Nov 2022 | ChatGPT (**main intervention**) |
| Feb–Mar 2023 | LLaMA release/leak; GPT-4 (14 Mar); r/LocalLLaMA created |
| 12–14 Jun 2023 | Reddit API protest / subreddit blackouts |
| Jul 2023 | Llama 2 |

### 1.5 Data validity checks (with decision rules)

Produce `results/data_checks.csv` and one overview figure (Figure 1 of the paper: monthly post volume per subreddit, removed share stacked).

| Check | How | Decision rule |
|---|---|---|
| Score capture time | median `retrieved_on − created_utc` per month; share of posts with `score ≤ 1` per month | If the capture lag differs by >1 order of magnitude between periods, or `score ≤ 1` share jumps by >20 pp at some month → engagement analysis restricted to periods with comparable capture, or use `num_comments` only, and say so |
| Removal rate | share `is_removed` per subreddit-month | If it moves by >10 pp around an event → moderation-policy change; must be a control in A/D and discussed |
| Volume | posts per subreddit-month | Months with <100 posts flagged; excluded from monthly ITS for that subreddit |
| Blackout | volume in Jun 2023 | Expect a dip; add a dummy for Jun 2023 in ITS |
| Tag coverage | share of r/ML titles with a parsable tag | If <80%, tag-based analysis reported only as descriptive |
| Missing fields | share of empty `score`, `num_comments`, `url` | Report; drop rows only for the analysis that needs the field |

---

## 2. Angle A — Topics over time

### 2.1 Questions
1. Which themes exist across ML Reddit (one shared topic space for all four subreddits)?
2. How did their prevalence change month by month, and is there a break at ChatGPT?
3. Does the change differ by community? (H1–H4)

### 2.2 Preprocessing (changes vs. the current baseline)

- Input: titles (main), title+body (robustness).
- Lowercase, strip `[D]/[R]/[P]/[N]` tags (they are kept as a separate variable instead).
- Lemmatize with spaCy `en_core_web_sm` (merges *network/networks*, *model/models*). Keep acronyms (`llm`, `gpt`, `cnn`, `rag`) as-is — check the lemmatizer does not mangle them.
- Stop words: sklearn English + **domain stop list**: subreddit-name tokens (`machine learning`, `deep learning`, `ml`, `dl`, `ai ml`, `learn`, `learning`) and generic request words (`help`, `need`, `best`, `question`, `using`, `use`, `way`, `looking`, `new`). Keep this list in `config/stopwords_domain.txt` and report it in the appendix.
  - Careful: `ai` is a real signal (H1). Do **not** stop-list it; check instead whether a topic is *only* `ai`.
- Phrases: keep unigrams + bigrams (`min_df=20`, `max_df=0.5` — lower than the current 0.9 so near-ubiquitous terms do not dominate).

### 2.3 Model selection protocol (NMF)

1. Grid: k ∈ {10, 12, 15, 20, 25, 30}; for each k, 5 seeds.
2. For each (k, seed) compute:
   - **NPMI coherence** (gensim `CoherenceModel`, `coherence='c_npmi'`, top-10 words, reference corpus = all titles+bodies to reduce short-text sparsity).
   - **Topic diversity** = share of unique words among all topics' top-25 words (Dieng et al. 2020).
   - **Stability**: match topics between seed pairs with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on cosine similarity of topic-word vectors; report mean matched cosine.
3. Choose k by: maximise coherence × diversity among k with stability ≥ 0.80. Ties → smaller k (easier to label and to fit in the paper).
4. Final model: seed with median coherence at the chosen k. Topics that do not reappear (matched cosine < 0.7) in ≥3/5 seeds are marked *unstable* and excluded from hypothesis tests.

### 2.4 Labeling protocol

- For each topic: top-15 words + 20 titles sampled from documents where that topic's normalised weight ≥ 0.5 (stratified across subreddits and years).
- Both team members label **independently** (short label + one-line description + "coherent? yes/partly/no"), then reconcile. Report the agreement rate.
- Merge or drop rule: topics labelled "no" are grouped as *Miscellaneous*; near-duplicates are merged at analysis time (sum weights) and the merge listed in the appendix.
- Then define a **topic family** layer (≈5–7 families) used in the main text for readability, e.g. *LLMs & generative AI*, *classic DL architectures*, *computer vision*, *learning & career*, *tooling & code*, *research & papers*, *data & datasets*. Individual topics stay in the appendix.

### 2.5 BERTopic (second model, robustness)

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` on titles (≈224k × 384 floats — feasible on a laptop CPU in roughly an hour; cache to `processed/embeddings.npy`).
- UMAP (n_neighbors=15, n_components=5, `random_state` fixed) → HDBSCAN (min_cluster_size ≈ 100) → c-TF-IDF; reduce to ~same number of topics as NMF with `reduce_topics`.
- Outliers (topic −1) are typically 30–50% on short texts: report the share; use `reduce_outliers` with the embedding strategy for prevalence estimates and say so.
- **Agreement with NMF**: map each BERTopic topic to the NMF topic family with the largest document overlap; report the cross-tab and whether H1/H3 conclusions hold under both models. This is the robustness claim — not "BERTopic is better".

### 2.6 Measuring prevalence

- Per post: normalise the NMF weight vector to sum to 1 (posts with zero weight → *unassigned*, reported).
- Prevalence of topic *k* in subreddit *s*, month *m* = mean normalised weight over posts in (*s*, *m*). Also compute the dominant-topic share as a check; report it if both agree, discuss if not.
- 95% CIs: bootstrap posts within each cell (1,000 resamples).

### 2.7 Interrupted time series (ITS)

For each subreddit *s* ∈ {ML, learnML, DL} and each topic family *f*:

```
logit(prev_{s,f,m}) = β0 + β1·month_idx + β2·post + β3·t_since
                      + γ·blackout_Jun2023 + δ·academic_term_m + ε_m
```

- Logit transform because shares are bounded (add 1e-4 before transforming).
- Seasonality: 48 points cannot afford 11 month dummies. Use one `academic_term` dummy (Sep–Dec, Feb–May) or a single sine/cosine pair; pick one and keep it fixed across all models.
- Errors: OLS with Newey–West (HAC) SE, lag 3.
- Multiple testing: Benjamini–Hochberg over all (subreddit × family × {β2, β3}) tests; report q-values.
- Effect size for the paper: predicted prevalence in Dec 2024 vs. the **counterfactual** (pre-trend extrapolated) — "X pp above what the 2021–22 trend predicts". This is far more readable than raw β.

**Placebo tests** (essential — a reviewer will ask "would you find a break at any date?"):
- Rerun ITS with fake intervention dates: Jun 2021, Dec 2021, Jun 2022. Real effects should be clearly larger at Dec 2022 than at placebo dates.
- Data-driven check: `ruptures` PELT (cost `rbf`) on each family series without telling it where the break is. Report the detected breakpoints; do they cluster around Nov 2022–Mar 2023?

LocalLLaMA has no pre-period → no ITS. Report its topic profile in 2023 vs. 2024 descriptively, and use it as the reference in D.

### 2.8 Emerging vocabulary (cheap, high-value figure)

- Candidate terms: those with prevalence < 0.05% of titles in 2021 and > 0.5% in any later quarter.
- For each: first month above 0.1%, peak month, peak share. Plot as a timeline ("term birth chart"): `chatgpt`, `gpt-4`, `llama`, `lora`, `rag`, `gguf`, `ollama`, `agent`, `fine-tune`, `prompt`, `stable diffusion`, `transformer` (control: should be stable).
- Shows the **order** in which the vocabulary arrived, which no topic model shows.

### 2.9 Post-type tags (r/MachineLearning only)

- Monthly share of `[R]`, `[D]`, `[P]`, `[N]`. ITS as above. Directly tests H4 with author-provided labels — independent from the topic model, so it validates it.

### 2.10 Outputs
- `results/a_topic_labels.csv`, `results/a_prevalence_monthly.csv`, `results/a_its.csv`, `results/a_placebo.csv`, `results/a_emerging_terms.csv`
- Figures: prevalence of 5–7 families per subreddit with event lines (main); counterfactual plot for the LLM family (main); term birth chart (main or appendix); full topic table and model-selection curves (appendix).

### 2.11 Failure modes and fallbacks
| Problem | Fallback |
|---|---|
| No k gives stability ≥ 0.8 | Report families (aggregated) only; families are stable even if individual topics are not |
| BERTopic and NMF disagree on a hypothesis | Report both; the claim becomes "model-dependent" — honest and acceptable |
| Short titles → weak topics | Robustness on title+body; restrict to `eligible_50` posts |

---

## 3. Angle F — Link sources

### 3.1 Questions
What do communities point to, and did the "currency" of ML Reddit shift from papers to code, models, and video? (H4, H5)

### 3.2 Two link measures
1. **Primary link** — the post's `url` for link posts (`is_self = false`).
2. **Body links** — URLs found inside self-post bodies (`body` still contains URLs; only `text` has them replaced). Many technical posts are self posts with a GitHub/arXiv link in the body; ignoring them would undercount.

Report both; main figure uses "any link in the post" (primary ∪ body), so self vs. link post format changes do not drive the result.

### 3.3 Domain normalisation
- `tldextract` → registered domain (`arxiv.org`, `github.com`).
- Merge aliases: `youtu.be`→youtube, `x.com`/`twitter.com`→x, `hf.co`→huggingface, `export.arxiv.org`→arxiv, `i.redd.it`/`v.redd.it`/`preview.redd.it`/`reddit.com/gallery`→reddit-media, `reddit.com/r/...` → reddit-internal (cross-links).
- Cross-check against Reddit's own `domain` field (from Step 0) — disagreement rate reported.

### 3.4 Taxonomy

Maintain `config/domain_categories.csv` (domain → category), committed to Git.

| Category | Examples |
|---|---|
| academic | arxiv, openreview, paperswithcode, aclanthology, semanticscholar, proceedings.*, doi.org |
| code | github, gitlab, colab.research.google.com, kaggle |
| models/hubs | huggingface, civitai, replicate, ollama.com |
| video | youtube, vimeo, twitch |
| blog/tutorial | medium, towardsdatascience, substack, dev.to, personal blogs (judgement call, documented) |
| social | x/twitter, linkedin, threads, bsky |
| news/media | theverge, techcrunch, wired, nytimes, … |
| company/product | openai.com, anthropic.com, deepmind, nvidia, … |
| reddit-media / reddit-internal | as above |
| other | everything not mapped |

Coverage rule: manually map the top-N domains until **≥ 95% of link-bearing posts** are covered. Put the remainder in *other* and report its share. Both team members map the top 100 independently → report agreement on categories.

### 3.5 Analysis
- Monthly share of each category among link-bearing posts, per subreddit (stacked area — the F headline figure).
- Pre/post χ² test per subreddit (category × period contingency table), Cramér's V as effect size.
- ITS (same specification as A) on academic, code, models/hubs shares.
- Cross with A: academic-link share *within* the LLM family vs. other families (are LLM discussions less paper-driven?).
- Cross with r/ML tags: does `[R]` still link to arXiv, or increasingly to GitHub/HF?

### 3.6 Outputs
`results/f_domains_top.csv`, `results/f_category_monthly.csv`, `results/f_its.csv`; one main figure, mapping table in the appendix.

### 3.7 Pitfalls
- Spam/self-promotion domains (blog farms) can produce spikes → check any category spike for a single dominating domain; report the top domain per spike.
- Link shorteners (`bit.ly`, `t.co`) → category *other*, report share; do not resolve (network calls, slow, and links are dead).

---

## 4. Angle B — Engagement

### 4.1 Questions
What does each community reward? Did the reward for LLM content change after ChatGPT? (H6) Secondary: do link types (F) and post types (tags) predict engagement?

### 4.2 Validity gate (from §1.5) — B only proceeds if passed
If scores were captured within minutes of posting in one period and weeks later in another, score comparisons across time are invalid. Outcome of the gate decides:
- passed → use score and comments;
- failed for score only → `num_comments` as the main outcome, score in the appendix;
- failed for both → restrict to within-period comparisons (topic differences inside 2021–22 and inside 2023–24 separately), and drop H6's time comparison. State it in Limitations.

Sample: `S_visible` (removed and stickied posts excluded).

### 4.3 Outcomes
- `score_pct` — percentile rank of `score` within (subreddit, month). Controls for subreddit growth and monthly activity: a post in the top 10% of its month is "successful" regardless of audience size.
- `comments_pct` — same for `num_comments`.
- `no_reply` — 1 if `num_comments = 0` (a sharp signal in help-seeking communities: was the question ignored?).
- Raw `score`, `num_comments` for count models (robustness).

Ties in percentile ranks (many posts with score 1) → use the average-rank method; report the share of ties.

### 4.4 Predictors
- Topic: dominant topic family from A, only posts with dominant normalised weight ≥ 0.4 (others → *mixed*). Also a continuous version: the LLM-family weight.
- `post` (ChatGPT) indicator.
- Controls: title length (tokens), title is a question (`?` or starts with wh-word/how/can/is), has body, link category (F), r/ML tag, weekday and hour of posting (UTC), month fixed effects.

### 4.5 Models (per subreddit; LocalLLaMA 2023 vs. 2024 separately)

Main:
```
score_pct ~ C(family) * post + controls + C(month)        (OLS, SE clustered by month)
```
H6 test: coefficient of `LLM_family × post`.

Robustness:
- Negative binomial on raw `num_comments` with the same right-hand side (statsmodels `NegativeBinomial`), incidence-rate ratios reported.
- Logistic regression on `no_reply`.
- Top-decile indicator (`score_pct ≥ 0.9`) logistic regression — "what goes viral" view.
- Exclude the first 3 months after ChatGPT (novelty spike) to check the effect is not only hype.

Minimum cell size: 200 posts per (family × period × subreddit); smaller cells merged into *other* families.

### 4.6 Interpretation rules (write these into the paper)
- Effects are **associations**: no causal language ("receives more engagement", not "causes").
- `S_visible` is a survivor sample: moderation removes low-quality posts, so engagement estimates condition on survival. Mention in Limitations.
- Percentile normalisation answers "relative success within the community at that time" — not absolute popularity.

### 4.7 Outputs
`results/b_validity.csv`, `results/b_models.csv`; main figure: coefficient (forest) plot — engagement premium per family, before vs. after, one panel per subreddit; appendix: full regression tables.

---

## 5. Angle D — Cross-community convergence

### 5.1 Questions
Did the three historical subreddits become harder to tell apart? Toward what? Did one community move more than the others? (H7, H8)

### 5.2 Measure 1 — Topic-distribution divergence
- For each (subreddit, quarter): mean normalised topic vector from A (individual topics, not families).
- Pairwise **Jensen–Shannon divergence** (base 2, in [0, 1]) for ML–learnML, ML–DL, learnML–DL.
- Size bias: JSD between finite samples is biased upward for small samples. Use `S_balanced(q)` with n = min cell size (expect ≈1,000–3,000 per subreddit-quarter; check), 200 bootstrap repetitions → mean and 95% CI.
- **Null baseline**: shuffle subreddit labels within the quarter, recompute JSD (200 times). Report JSD minus null as "excess divergence".
- Trend: OLS of excess JSD on quarter index + post-ChatGPT level shift; with 16 points, also report a Mann–Kendall trend test.

### 5.3 Measure 2 — Subreddit-prediction classifier (distinctiveness)
- Per quarter: `S_balanced(q)` titles from the 3 historical subreddits (same n per subreddit), TF-IDF + logistic regression, 5-fold CV, **macro-F1**. Chance = 0.33.
- Leakage control: remove subreddit-name tokens, r/ML tags, and flair-like prefixes before vectorising; otherwise the classifier learns formatting, not content.
- Falling F1 over time = convergence. Also look at the per-class confusion: which pair became most confusable?
- **Masking test** (key to interpretation): rerun with the top-100 LLM-family terms removed. If convergence disappears → "everyone started talking about LLMs" (convergence is just one shared topic). If it persists → broader convergence in style/content. Either result is reportable; the distinction is the interesting part.

### 5.4 Measure 3 — Distinctive vocabulary (Fightin' Words)
- Log-odds ratio with informative Dirichlet prior (Monroe et al. 2008), each subreddit vs. the other two, separately for 2021–22 and 2023–24.
- Table: top-15 distinctive words per subreddit per period (z-scores). Shows *what* each community's identity was and how it changed — the qualitative counterpart to measures 1–2.

### 5.5 Measure 4 — Distance to LocalLLaMA (H8)
- Reference = LocalLLaMA 2023 mean topic vector (and its TF-IDF centroid).
- For each historical subreddit-quarter: JSD to the reference (balanced sampling as above).
- Also **drift**: JSD of each subreddit-quarter to its own 2021Q1–Q2 profile. Answers "which community changed the most from where it started?"

### 5.6 Optional (only if time)
Embedding centroids (from BERTopic's cached embeddings): cosine similarity between subreddit-quarter centroids. Cheap once embeddings exist; a model-free robustness check for measure 1.

### 5.7 Outputs
`results/d_jsd_quarterly.csv`, `results/d_classifier.csv`, `results/d_fightin_words.csv`, `results/d_reference_distance.csv`. Main figure: two panels — excess JSD per pair over time; classifier macro-F1 over time (with and without LLM masking). Fightin' Words table in main text (compact) or appendix.

### 5.8 Pitfalls
- Cross-posts (same title in several subreddits) mechanically increase similarity → detect exact-title matches across subreddits within ±7 days, report the rate, exclude them in a robustness run.
- Different growth rates → handled by balanced sampling; never compare raw-count-based measures.

---

## 6. How the four angles fit together

```
Step 0 data  ─┬─> A topics ──┬─> B engagement (needs families + F categories)
              │              └─> D convergence (needs topic vectors)
              └─> F links ───────> B (control) and A (cross-analysis)
```

Cross-angle results worth a paragraph each in the Discussion:
1. **Supply vs. demand**: A shows what is posted, B shows what is rewarded. If LLM share rises *and* the LLM premium rises → demand-led shift. If share rises but premium falls → saturation / fatigue.
2. **Research to applied**: F (arXiv ↓, HF/GitHub ↑) + r/ML tags ([R] ↓, [P] ↑) + A (research family ↓) — three independent signals for one claim.
3. **Convergence mechanism**: D's masking test tells us whether convergence is driven by the LLM topic (from A) or broader.

---

## 7. Paper plan (ACL, ≤ 8 pages main text)

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 0.75 | Motivation, 4 questions, contributions list |
| 2 Related work | 0.75 | Reddit community studies, language change in communities, topic models for social media, ITS in computational social science |
| 3 Data | 1.0 | Source, subreddits, cleaning, samples (`S_submitted`/`S_visible`), validity checks; **Fig 1** volume + removal |
| 4 Methods | 1.5 | Topic model + selection, ITS, link taxonomy, engagement models, convergence measures |
| 5 Results | 3.0 | 5.1 Topics (**Fig 2** family prevalence, **Fig 3** counterfactual) · 5.2 Links (**Fig 4**) · 5.3 Engagement (**Fig 5**) · 5.4 Convergence (**Fig 6**) |
| 6 Discussion | 0.75 | The three cross-angle paragraphs (§6) |
| 7 Conclusion | 0.25 | |
| Limitations | (outside page limit in ACL — confirm with course) | Titles only, score capture, survivor bias, no causality, Reddit ≠ whole ML community, dataset ends Dec 2024 |
| Ethics | short | Public data, no usernames, aggregate reporting, example titles paraphrased |
| Appendix | unlimited | Full topic table, model selection curves, BERTopic comparison, placebo tests, domain mapping, full regression tables, stop lists |

Rule: 6 figures max in the main text; each figure must answer a stated research question in its caption.

---

## 8. Code layout

Keep the existing scripts; add numbered analysis scripts so the order is obvious:

```
config/
  stopwords_domain.txt
  domain_categories.csv
  events.csv                       # event calendar (§1.4)
common.py                          # paths, event dates, loading, sample definitions, period vars
00_data_checks.py                  # §1.5  -> results/data_checks.csv, fig_01
10_topics_select.py                # §2.3  grid over k × seeds -> results/a_model_selection.csv
11_topics_fit.py                   # final NMF -> processed/doc_topics.parquet, results/a_topics.csv
12_topics_bertopic.py              # §2.5
13_topics_time.py                  # §2.6–2.9 prevalence, ITS, placebo, emerging terms
20_links.py                        # §3
30_engagement.py                   # §4
40_convergence.py                  # §5
90_figures.py                      # all paper figures from results/*.csv only
```

Conventions: every script reads from `processed/` or `results/` and writes to `results/` (aggregates only — never post-level text). Figures are built **only from `results/*.csv`**, so anyone can redraw them without the raw data. Fixed `RANDOM_STATE = 42` in `common.py`.

New dependencies: `matplotlib`, `statsmodels`, `scipy`, `tldextract`, `gensim` (coherence), `spacy` + `en_core_web_sm`, `ruptures`, `bertopic` + `sentence-transformers` (BERTopic step only), `pyarrow` (parquet).

---

## 9. Timeline and split

Person 1 = A + D. Person 2 = F + B. Step 0 together (it is short and both depend on it).

| Dates | Person 1 (A, D) | Person 2 (F, B) | Joint |
|---|---|---|---|
| Sat 14 – Mon 16 Nov | — | — | Extend `clean_data.py` (§1.2), `common.py`, `00_data_checks.py` |
| Tue 17 – Sun 22 Nov | NMF grid + selection, independent topic labels | Domain extraction, taxonomy mapping (top 100 both), B validity gate | Reconcile topic labels and domain categories |
| Mon 23 – Sun 29 Nov | Monthly prevalence + first ITS for LLM family; first JSD series | Category stacked areas, χ²; first engagement-by-family table | Choose 4 plots for the channel |
| Mon 30 Nov – Tue 1 Dec | | | **Post preliminary results** (4 plots + 1 paragraph per question + open problems) |
| 2 – 20 Dec | BERTopic, placebos, change points, emerging terms; classifier + masking, Fightin' Words, reference distance | ITS on links, cross-analyses, full engagement models + robustness | Mid-point review: which hypotheses hold? |
| 21 Dec – 10 Jan | Robustness (title+body, `S_visible`), freeze results | Robustness, freeze results | `90_figures.py` final |
| 11 Jan – exam | Write §2 related work, §5.1, §5.4 | Write §3 data, §5.2, §5.3 | Intro, Discussion, Limitations, appendix; two full read-throughs |

Preliminary-results post (1 Dec) — candidate plots:
1. Monthly LLM-family prevalence per subreddit with ChatGPT line.
2. Link-category stacked area per subreddit.
3. Term birth chart (`chatgpt`, `llama`, `lora`, `rag`, …).
4. Excess JSD between subreddit pairs per quarter (or engagement by family, whichever is more solid by then).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Score captured at very different delays across periods | medium | B weakened | Validity gate; fall back to comments / within-period analysis |
| Topics unstable or uninterpretable | medium | A, B, D weakened | Family layer; BERTopic; title+body |
| Moderation policy changes confound topic shifts | medium | A, D | `S_visible` vs. `S_submitted` comparison; removal-rate control |
| ITS significant at placebo dates too | low–medium | H1 claim weakened | Report honestly; frame as trend, not break |
| Convergence = only LLM vocabulary | high | D less interesting | Masking test turns this into a finding either way |
| BERTopic too slow | low | robustness missing | Run on a 50k balanced sample; cache embeddings |
| Running out of time | medium | paper quality | §5.6 and the `S_visible` robustness reruns are the first things to cut; the 4 main figures are not |

---

## 11. Key references

- Zhang, Hamilton, Chang, Danescu-Niculescu-Mizil, Jurafsky, Leskovec (2017). *Community Identity and User Engagement in a Multi-Community Landscape.* ICWSM.
- Danescu-Niculescu-Mizil, West, Jurafsky, Leskovec, Potts (2013). *No Country for Old Members: User Lifecycle and Linguistic Change in Online Communities.* WWW.
- Monroe, Colaresi, Quinn (2008). *Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict.* Political Analysis.
- Lee & Seung (1999). *Learning the parts of objects by non-negative matrix factorization.* Nature.
- Grootendorst (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv.
- Röder, Both, Hinneburg (2015). *Exploring the Space of Topic Coherence Measures.* WSDM.
- Dieng, Ruiz, Blei (2020). *Topic Modeling in Embedding Spaces.* TACL (topic diversity).
- Bernal, Cummins, Gasparrini (2017). *Interrupted time series regression for the evaluation of public health interventions: a tutorial.* Int. J. Epidemiology.
- Baumgartner et al. (2020). *The Pushshift Reddit Dataset.* ICWSM (data source — cite whichever dump you actually use).

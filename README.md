# Meta-Research about Machine Learning Communities on Reddit

**NLP 2 — AI Master's project | Work in progress**

This project examines how machine-learning discussions on Reddit changed between **2021 and 2024**, and whether different communities evolved similarly.

## Research questions

1. How did discussion topics and terminology change between **2021–2022** and **2023–2024**?
2. How do these changes differ across communities?

| Community | Period | Comparison |
|---|---|---|
| r/MachineLearning | 2021–2024 | 2021–2022 vs. 2023–2024 |
| r/learnmachinelearning | 2021–2024 | 2021–2022 vs. 2023–2024 |
| r/deeplearning | 2021–2024 | 2021–2022 vs. 2023–2024 |
| r/LocalLLaMA | 2023–2024 | 2023 vs. 2024 |

**Important:** LocalLLaMA has no 2021–2022 data in this dataset. Do not treat it as part of the same before/after comparison.

## Current data and methods

- **230,298** raw posts (2021–2024).
- **230,159** posts after initial cleaning.
- **224,300** posts after additional filtering.
- The current NLP experiments use **titles only**, not full post bodies, to reduce bias from different rates of missing body text.

Methods implemented:

1. Streaming `.zst` JSONL archive inspection and cleaning.
2. Auditing repeated content and removing documented placeholders, recurring megathreads, specific spam patterns, and conservatively defined duplicates.
3. TF-IDF lexical analysis (unigrams and bigrams), comparing term prevalence in **percentage points (pp)**.
4. A shared **15-topic NMF baseline** trained on a balanced sample of up to 3,000 posts per subreddit/year (**42,000** training posts in the current run), then applied to all posts.

Topic quality and statistical significance have **not yet been fully evaluated**. BERTopic is planned, not implemented.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── inspect_data.py
├── clean_data.py
├── audit_data.py
├── inspect_duplicates.py
├── build_corpus.py
├── inspect_lexical_noise.py
├── lexical_analysis.py
├── nmf_analysis.py
├── results/                # Aggregate CSV results (tracked by Git)
├── reddit/subreddits24/    # Raw .zst archives (not tracked)
└── processed/              # Intermediate CSV datasets (not tracked)
```

## Setup — Windows PowerShell

### 1. Clone and install

Install Python 3 and Git, then run:

```powershell
git clone https://github.com/Yanis3Pique/Meta-Research-about-the-machine-learning-communities-on-reddit.git
cd Meta-Research-about-the-machine-learning-communities-on-reddit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `zstandard`. Commands use the virtual environment directly; no PowerShell activation is required.

### 2. Add the raw data

The dataset is **not included** in this repository. Obtain the same Reddit archive snapshot from the project team/original source and place the files at these exact paths:

```text
reddit/
└── subreddits24/
    ├── MachineLearning_submissions.zst
    ├── learnmachinelearning_submissions.zst
    ├── deeplearning_submissions.zst
    └── LocalLLaMA_submissions.zst
```

These files contain compressed JSONL Reddit submissions. The same archive snapshot is required to reproduce the published counts.

### 3. Inspect and preprocess

Run from the repository root **in this order**:

```powershell
# Optional: inspect archive counts and years
.\.venv\Scripts\python.exe inspect_data.py

# Required: creates processed/cleaned_posts.csv
.\.venv\Scripts\python.exe clean_data.py

# Optional: inspect data quality and duplicates
.\.venv\Scripts\python.exe audit_data.py
.\.venv\Scripts\python.exe inspect_duplicates.py

# Required: creates processed/modeling_corpus.csv
.\.venv\Scripts\python.exe build_corpus.py

# Optional: inspect specific lexical noise
.\.venv\Scripts\python.exe inspect_lexical_noise.py

# Data-validity checks (removal, score capture, title tags)
.\.venv\Scripts\python.exe 00_data_checks.py
```

`clean_data.py` also extracts `title_tag`, `is_removed`, `removed_by_category`, `retrieved_on`, `updated_on`, `subreddit_subscribers`, `domain`, `upvote_ratio`, `stickied`, `over_18` and a salted `author_hash` (no usernames are stored). Set the environment variable `REDDIT_AUTHOR_SALT` to keep author hashes stable across runs; never commit it. See `docs/research_design.md` §1.6 for what the data checks found.

`build_corpus.py` requires the `url` column from the current `clean_data.py`. If missing, rerun `clean_data.py`.

### 4. Run the NLP experiments

After `processed/modeling_corpus.csv` exists:

```powershell
.\.venv\Scripts\python.exe lexical_analysis.py
.\.venv\Scripts\python.exe nmf_analysis.py
```

NMF can take several minutes. If you already have the processed corpus, you can skip raw-data preparation and run these two scripts directly.

## Outputs

| Script | Output file | Meaning |
|---|---|---|
| `clean_data.py` | `processed/cleaned_posts.csv` | Initially cleaned posts |
| `clean_data.py` | `processed/cleaning_report.csv` | Cleaning counts |
| `audit_data.py` | `processed/audit_report.csv` | Quality diagnostics |
| `build_corpus.py` | `processed/modeling_corpus.csv` | Final modeling corpus |
| `build_corpus.py` | `processed/corpus_filter_report.csv` | Exclusion reasons |
| `00_data_checks.py` | `results/data_checks_monthly.csv`, `results/data_checks_yearly.csv` | Removal, score-capture and tag diagnostics per subreddit-month/year |
| `lexical_analysis.py` | `results/tfidf_top_terms.csv` | Highest mean TF-IDF terms |
| `lexical_analysis.py` | `results/term_shifts.csv` | Changes in term prevalence (pp) |
| `nmf_analysis.py` | `results/nmf_topics.csv` | Top words per topic |
| `nmf_analysis.py` | `results/nmf_topic_shares.csv` | Topic shares by community/period |
| `nmf_analysis.py` | `results/nmf_topic_shifts.csv` | Changes in topic shares (pp) |
| `10_topics_select.py` | `results/a_model_selection*.csv` | NMF grid over k: NPMI, diversity, stability |
| `11_topics_fit.py K` | `results/a_topics.csv`, `processed/doc_topics.parquet` | Final NMF (k=20), labelling sheet in `processed/` |
| `13_topics_time.py` | `results/a_*.csv`, `results/figures/fig_a_*.png` | Monthly topic-family prevalence, ITS + placebos, r/ML tags, emerging terms |
| `40_convergence.py` | `results/d_*.csv`, `results/figures/fig_d_*.png` | Divergence between subreddits, classifier distinctiveness, distance to r/LocalLLaMA |

The five aggregate CSVs under `results/` are tracked by Git. The raw archives and individual post datasets are ignored. Re-running scripts overwrites their corresponding output CSVs.

## Preliminary findings

The proportion of titles containing `llm` increased from 2021–2022 to 2023–2024:

| Community | Change |
|---|---:|
| r/MachineLearning | +3.56 pp |
| r/learnmachinelearning | +2.12 pp |
| r/deeplearning | +2.34 pp |

The NMF baseline also finds an AI/generative-AI-associated component with increasing share across the three historical communities.

**These are exploratory associations, not proof of causation.** Some NMF components overlap or mix concepts and require manual validation.

## Limitations and next steps

- Title-only analysis; title + body can be tested for robustness.
- NMF currently uses 15 topics; alternative counts and interpretability require evaluation.
- Some off-topic content may remain after conservative filtering.
- Data end in December 2024.
- BERTopic and formal statistical tests are not yet implemented.

Next: inspect representative posts, evaluate NMF topic quality/stability, test other topic counts, and compare against BERTopic.

## Collaboration workflow

Before working:

```powershell
git pull
git switch -c feature/describe-your-change
```

After editing and testing:

```powershell
git status
git add .
git commit -m "Describe your change"
git push -u origin feature/describe-your-change
```

Open a pull request to merge into `main`. **Never commit** raw archives, processed post-level data, secrets, or unreviewed post-level examples.

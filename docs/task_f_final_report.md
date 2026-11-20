# Task F — Link Sources: Final Research Report

**Project:** Meta-Research about the Machine Learning Communities on Reddit
**Date:** 20 November 2026
**Status:** Computational analysis complete; classification agreement and adjudication complete. Residual data-quality limitations are disclosed, not claimed to have been exhaustively resolved.

## 1. Objective and scope

Task F examines which web-source types Reddit machine-learning communities share, whether source composition changed around ChatGPT's November 2022 release, and how these changes inform **H4** (research versus applied resources) and **H5** (arXiv versus GitHub/Hugging Face). The modeling corpus contains **224,300 posts** across r/MachineLearning, r/learnmachinelearning, r/deeplearning, and r/LocalLLaMA. Source types are assigned using the project's domain taxonomy; the unit of the headline chart is a link-bearing post, with source-category contributions normalized within each post.

## 2. Taxonomy integration and coverage

The final domain taxonomy contains **4,680 unique mappings**. The classification workflow processed **4,635 Work domains**; **87 suspected false-positive link extractions** were excluded from the new mapping import. Existing mappings were preserved, and three post-review taxonomy corrections were applied: `lmsys.org` → `models/hubs`, `qualtrics.com` → `company/product`, and `producthunt.com` → `social`.

| Coverage measure | Mapped / total | Coverage |
|---|---:|---:|
| Fully mapped link-bearing posts **(primary criterion)** | 69,545 / 73,111 | **95.1225%** |
| Individual links | 120,854 / 125,734 | 96.1188% |
| Primary links | 36,309 / 37,343 | 97.2311% |
| Post-domain occurrences | 92,458 / 96,591 | 95.7211% |

The predefined **pooled ≥95% fully mapped post** requirement is met. The checker reproduces all four measures from the current analysis outputs. This is **taxonomy-lookup coverage**, not an estimate of independently verified website classification accuracy.

| Community | Fully mapped post coverage |
|---|---:|
| r/LocalLLaMA | 98.46% |
| r/MachineLearning | 93.70% |
| r/deeplearning | 95.95% |
| r/learnmachinelearning | 94.03% |

Two individual communities fall below 95%, although the research-design threshold applies to the pooled corpus. The margin above the pooled threshold is narrow, so material changes to extraction or mapping require recomputation.

## 3. Independent annotation and adjudication

Two reviewers independently labeled **100 review entries**, achieving **95/100 exact agreement (95.0%)** and **Cohen's κ = 0.9405**. The five disagreements were adjudicated as follows:

| Entry | Adjudicated label |
|---|---|
| spotify.com | `other` |
| lmsys.org | `models/hubs` |
| qualtrics.com | `company/product` |
| pcpartpicker.com | `company/product` |
| producthunt.com | `social` |

Three adjudications changed the existing taxonomy; the pipeline was rerun afterward. **Qualification:** the reviewed top-100 sheet included the synthetic `reddit-media` grouping, so this was **100 entries including 99 actual domains**, not strictly 100 distinct website domains. The reported agreement measures reproducibility between reviewers, not classification accuracy against external ground truth.

## 4. H5 — Source-domain shifts

Shares below are the percentages of link-bearing posts containing each domain; domains are not mutually exclusive within a post. The pre/post split is around December 2022.

| Community | arXiv Δ (pp) | GitHub Δ (pp) | Hugging Face Δ (pp) |
|---|---:|---:|---:|
| r/MachineLearning | −1.75 | +2.16 | +2.16 |
| r/learnmachinelearning | −0.39 | +2.49 | +1.12 |
| r/deeplearning | −5.63 | +0.11 | +0.93 |

Hugging Face increased in all three historical communities; arXiv decreased in all three; GitHub increased noticeably in two and changed little in r/deeplearning. Seven of nine domain-specific chi-square comparisons survived FDR correction in the pre-adjudication closeout tables, generally with small effect sizes. **No domain-specific ITS level-change test exceeded all placebo breakpoints.** Accordingly, **H5 is descriptively supported**, but these results do not establish a causal or uniquely timed effect of ChatGPT's release.

r/LocalLLaMA only has post-intervention observations and cannot provide a 2021–2022 baseline for ITS.

## 5. H4 — Research versus applied sources

Within r/MachineLearning's `[R]`-tagged link-bearing posts, arXiv presence was **58.80% before** versus **57.97% after**; GitHub presence rose **20.91% → 26.44%**, and Hugging Face **1.00% → 4.79%**. Within `[P]` posts, GitHub presence changed **54.70% → 50.56%**, while Hugging Face rose **2.58% → 6.49%**. These are **within-tag link shares**, not tag-prevalence rates.

Previously generated tag summaries show research-tag share approximately **20.06% pre → 15.67% post**, while project-tag share was **17.55% → 17.94%**. In the post-period r/MachineLearning LLM/non-LLM comparison, LLM-classified posts were more likely to include academic, code, and model-hub links, so research and applied resources are not mutually exclusive.

**H4 is mixed/partially supported:** research-oriented discussions incorporated more code and model resources, but the evidence does not establish wholesale replacement of research by project posts.

## 6. Category-level inference and visualization

The regenerated category-level pre/post chi-square tests indicate differences for r/MachineLearning, r/learnmachinelearning, and r/deeplearning, with **Cramér's V = 0.1187, 0.1205, and 0.1621**, respectively (small associations). Among nine exploratory category ITS tests, the `code` level term in r/learnmachinelearning had **q = 0.0002**, but **none of the nine passed every placebo comparison**. Statistical significance alone does not demonstrate a ChatGPT-caused structural break.

The headline visualization is `results/figures/fig_f_link_categories.png`. Its `unclassified` area denotes **unmapped observed links**, not remaining unclassified Work-domain rows. Monthly category spikes can reflect concentration in dominant sources: for example, arXiv accounted for **67.1%** of r/MachineLearning's academic-category contribution in June 2021, while YouTube accounted for **100%** of r/deeplearning's video-category contribution in October 2023. These figures describe composition, not causal explanations of spikes.

## 7. Data-quality limitations and audit record

The closeout exception table contains **198 flagged domain entries**: **87 suspected extraction artifacts** (such as `model.fit`, `main.py`, `datetime.now`, and `requests.post`) and **111 other flagged identities, redirects, or shared-host cases**. The source table reports **287** and **1,085** summed affected-post incidences, respectively; these are **not deduplicated counts of posts**.

Multi-tenant domains such as `sites.google.com`, `vercel.app`, `streamlit.app`, `wordpress.com`, `notion.site`, `herokuapp.com`, and `googleapis.com` can host different kinds of content; domain-level labels can misrepresent a particular destination. These cases were flagged, **not all individually resolved**.

Missingness differs across periods. In r/MachineLearning, fully mapped post coverage was **90.98% before** versus **96.29% after** the intervention. This imbalance may influence apparent pre/post shifts. A dedicated sensitivity analysis demonstrating that missingness does not alter the headline conclusions has **not been established** in this report. Interpret results as descriptive associations conditional on observed mappings.

## 8. Conclusion and status

Task F's **classification integration, main analysis, pooled 95% coverage gate, independent agreement calculation, five-way adjudication, and rerun after taxonomy corrections are complete**. The findings are consistent with a descriptive move toward code/model platforms, especially Hugging Face, while research links remain important. The ITS/placebo evidence does **not** justify a causal claim about ChatGPT's release.

The project can mark **Task F's computational work as complete with documented quality limitations**. Do **not** claim that all 198 flagged cases were independently verified, that the review comprised 100 genuine domains, or that the causal hypotheses were proven. Any later extraction fixes or substantive taxonomy changes require rerunning the coverage and statistical pipeline.

## 9. Reproducibility files

- `config/domain_categories.csv` — accepted and adjudicated taxonomy
- `20_links.py` — extraction, category aggregation, statistical analyses, and figure
- `results/f_coverage.csv` and `processed/link_features.parquet` — current coverage and features
- `results/f_chi2.csv`, `results/f_its.csv`, `results/f_category_monthly.csv` — primary results
- `results/figures/fig_f_link_categories.png` — headline figure
- `results/f_closeout/top100_reviewer_A.csv`, `top100_reviewer_B.csv`, `top100_interannotator_agreement.csv`, `top100_disagreements_to_adjudicate.csv` — review and adjudication record
- `results/f_closeout/h5_domain_prepost.csv`, `h5_domain_chi2.csv`, `h5_domain_its.csv`, `h4_tags_visible_machinelearning.csv`, `h4_llm_vs_other_links.csv`, `coverage_by_community_period.csv`, `category_spikes_top_domains.csv`, `quality_exceptions_with_impact.csv` — supporting analyses

**Provenance:** numerical results are taken from the supplied Task F run logs and closeout tables. Domain-specific closeout statistics are from the supplied preparation run; the final post-adjudication log independently confirms the updated category-level statistics and unchanged coverage. Do not represent domain-specific closeout estimates as newly recomputed after adjudication without rerunning and checking those tables.

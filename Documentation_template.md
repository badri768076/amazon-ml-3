# Amazon ML Challenge 2026 — Business Entity Resolution: Documentation

**Team:** `<TEAM NAME>` · **Code:** `code/business_entity_resolution/` · **Reproduce:** `python src/main.py --stage all`

> **Before submitting:** every cell marked **⟨fill⟩** must be copied from the run's reports
> (`/kaggle/working/artifacts/reports/`, most conveniently `dataset_quality_report.md`, which
> collects them all). The file behind each number is named in the tables. Numbers in the
> *Smoke test* column come from a synthetic dataset shaped like the challenge (same schema,
> TSV layout and noise types; 40 k S1, 350 k S2 + S3). They show that the pipeline works end to
> end and are **not** challenge results.

---

## 1. Problem understanding

For each Source 1 (reference) business, the task is to find every Source 2 and Source 3 record
describing the same real-world business. There may be none, one or several. The data is
large, noisy (typos, abbreviations, legal suffixes, DBA names, transliteration, reordered or
partial addresses) and multi-country, and the test set adds France, which is absent from
training. Scoring is **macro F0.5 per Source 1 entity**, which has two consequences:

* a wrong merge costs more than a missed match;
* a correct empty answer for a no-match entity scores in full.

The design therefore separates two problems: **finding** the true match (blocking recall) and
**deciding** precisely (classifier + entity-level rule tuned on macro F0.5).

## 2. Dataset source

The project uses the supplied Kaggle dataset located at:

/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml

No Hugging Face dataset, external business database, external API,
or external business information source was used.

Nothing is downloaded at any stage. The `submission` stage verifies this automatically (§18).

## 3. Dataset loading

* **Discovery:** `DATA_ROOT.rglob("*")` over the Kaggle input folder. Files are mapped to the
  seven logical tables whatever the folder layout, and the reader is chosen by extension
  (`.tsv`, `.csv`, `.parquet`).
* **Reading:** one table at a time, only the required columns, all values as text (IDs and
  PIN/ZIP codes keep leading zeros), with empty strings preserved.
* **Caching:** each table is written once to `artifacts/raw/<table>.parquet` with a manifest
  (file size, format, original dtypes). All later work uses lazy Polars / DuckDB scans, never
  all tables in pandas at once.
* **Output location:** `/kaggle/input` is read-only, so artefacts and outputs go to
  `/kaggle/working`.

## 4. Dataset structure

| Table | Columns | Rows (`reports/eda.json`) | Smoke test |
|---|---|---|---|
| train_source1 | entity_id, business_name, business_address, country | ⟨fill⟩ | 40,000 |
| train_source2 | same | ⟨fill⟩ | 200,000 |
| train_source3 | same | ⟨fill⟩ | 150,000 |
| train_ground_truth | source1_entity_id, matched_entity_ids | ⟨fill⟩ | 40,000 |
| test_source1 | entity_id, business_name, business_address, country | ⟨fill⟩ | 20,000 |
| test_source2 | same | ⟨fill⟩ | 100,000 |
| test_source3 | same | ⟨fill⟩ | 80,000 |

IDs are prefixed `S1-`, `S2-` and `S3-`. An empty `matched_entity_ids` means the Source 1
entity has no match. The test set has no labels.

## 5. Data validation

The checks are listed in `reports/validation_report.json`.

* **Source tables:** missing IDs, wrong or unexpected prefixes (including IDs of another
  source), duplicate IDs and rows in duplicate-ID groups.
* **Ground truth:**
  * every `source1_entity_id` must exist in `train_source1`;
  * every S2 target must exist in `train_source2`, and every S3 target in `train_source3`;
  * no S1 ID may appear as a target;
  * no duplicate rows, no duplicate IDs within a row, no self-matches.
* **What stops the run:** an empty table, missing IDs, wrong prefixes and duplicate
  ground-truth rows. Invalid targets are reported and ignored.

| Result | Official | Smoke test |
|---|---|---|
| Errors / warnings | ⟨fill⟩ | 0 / 0 |
| Invalid GT targets (S2 / S3 / S1-as-target) | ⟨fill⟩ | 0 / 0 / 0 |

## 6. EDA

The full profile is in `reports/eda.md` and `eda.json`. It covers:

* files: format, size, dtypes, unique values;
* null, empty, whitespace-only and null-like counts per column;
* exact, ID and normalised duplicates;
* name and address length and word-count statistics, numeric tokens and postal availability;
* raw and normalised country frequencies;
* ground-truth match distribution, including S2 / S3 / both and multi-match statistics;
* cross-source consistency of true matches;
* a list of **findings turned into modelling decisions**.

| Measure | Official | Smoke test |
|---|---|---|
| Missing address (S2 / S3, train) | ⟨fill⟩ | 4.97% / 5.01% |
| Missing country (S2 / S3, train) | ⟨fill⟩ | 5.92% / 6.02% |
| Duplicate normalised names (train S2) | ⟨fill⟩ | 115,434 of 200,000 (chain-heavy by design) |
| Countries (train S1) / only in test | ⟨fill⟩ | in 50.1%, us 49.9% / fr |
| Zero / one / multi-match S1 entities | ⟨fill⟩ | 9,877 (24.7%) / 17,091 (42.7%) / 13,032 (32.6%) |
| S2 / S3 matches; entities with both | ⟨fill⟩ | 25,430 / 21,034; 9,875 |
| True matches with identical normalised name | ⟨fill⟩ | 54.7% |
| Postal agreement when both present | ⟨fill⟩ | 100% |

**Decisions driven by the data:**

* **Missing addresses** → address similarities are NaN when missing (never 0), there are
  explicit missing flags, and name-only blocking keys exist.
* **Missing and variant countries** → alias normalisation, and `?`/`*`-scoped blocking keys so
  missing-country records are never excluded.
* **Unseen France** → country-agnostic features, and TF-IDF fitted per split on unlabelled text.
* **Many no-match entities** → an explicit empty-list decision tuned on macro F0.5.
* **Multi-match entities** → no argmax; every candidate passing the rule is kept.
* **Frequent name collisions (chains)** → address, postal and building evidence in candidate
  re-ranking and in the features.
* **Low exact-name agreement** → character-n-gram, token and fuzzy blocking and features.
* **Scale** → blocking instead of Cartesian products.

## 7. Preprocessing

All rules are vectorised Polars expressions; `normalize_name`, `normalize_address` and
`normalize_country` run the same expressions on single values.

* **Name:** NFKC → lowercase → strip Latin diacritics only → collapse dotted acronyms → drop
  the `M/s` honorific → punctuation to spaces, `&` → `and` → safe legal-suffix and abbreviation
  mapping (`Pvt. Ltd.` / `Private Limited` / `Pvt Ltd` → `private limited`). A **core name**
  without legal forms is also kept.
* **Address:** the same cleaning, plus PIN repair (`560 001` → `560001`), ZIP+4 → ZIP,
  `#` → `number`, letter/digit split and street-type abbreviations. Numeric tokens, postal/PIN
  candidates and building numbers are extracted and preserved.
* **Country:** formatting normalisation. Spelling variants of the same country are unified
  (configurable), and unknown values pass through (open set).
* **Missing values:** no record is ever removed for missing values.

## 8. Blocking

| Strategy | Key | Handles |
|---|---|---|
| Country scoping | country prefix on scoped keys; `?`/`*` variants | cross-country false candidates without excluding missing countries |
| Exact name / exact core name | normalised name / name without legal forms (unscoped) | clean matches, legal-suffix variants, wrong country |
| Name signature | sorted core tokens | word order |
| Name tokens | 3 rarest core tokens | partial and DBA names |
| Character n-grams | 4 rarest 3-grams of the core name | typos, transliteration |
| Address tokens / signature | rarest address words / sorted address words | renamed businesses, reordered addresses |
| Postal + name prefix / building / rare token | combined keys | common names in one locality, same premises |
| Fuzzy retrieval | TF-IDF char-3-gram cosine + RapidFuzz re-rank | S1 records with < 3 candidates from a source |

Keys shared by too many candidates are dropped, and pairs are produced by hash joins only.

## 9. Candidate generation

The union of all strategies is de-duplicated, and each pair gets a score Σ wₛ·log(1 + N/df).
Per (S1, source), the top 200 are re-ranked by a cheap RapidFuzz score and the best 25 are
kept. The score weights name evidence 0.50, address evidence 0.45 (address token-set and ratio,
postal and building agreement) and blocking 0.05. A missing address or postal code is treated
as neutral rather than as a mismatch, which lifted candidate recall from 0.958 to 0.994 on the
small synthetic set and from 0.905 to 0.942 on the chain-heavy one. That set is the
model input and the content of `candidate_pairs.tsv`.

## 10. Candidate recall

| `reports/candidate_recall_train.json` | Official | Smoke test |
|---|---|---|
| Total possible pairs | ⟨fill⟩ | 14,000,000,000 |
| Generated candidate pairs | ⟨fill⟩ | 2,000,000 |
| Candidate recall (all / S2 / S3) | ⟨fill⟩ | 0.946 / 0.943 / 0.950 |
| Recall before capping | ⟨fill⟩ | 0.977 |
| Reduction ratio | ⟨fill⟩ | 0.99986 |
| Avg / median / max candidates per S1 | ⟨fill⟩ | 50 / 50 / 50 |

The synthetic set is deliberately harsh on chains (median ≈ 90 same-name candidates per S1),
so the cap costs more recall there than on realistic data.

## 11. Feature engineering (≈ 90 features, no IDs)

* **Name:** exact / exact-core match; Levenshtein; Indel ratio; Jaro; Jaro-Winkler; token
  sort / set / partial; prefix; char-(2–4)-gram and word TF-IDF cosine; token Jaccard /
  overlap / common count / ratio; length difference and ratio; word-count difference; acronym
  match.
* **Address:** exact match; Levenshtein; Jaro; Jaro-Winkler; token sort / set / partial;
  char-3-gram TF-IDF cosine; token Jaccard / overlap / common; numeric-token overlap and
  Jaccard; postal exact / prefix / both-present / conflict; building-number match; length
  difference and ratio. All are NaN when an address is missing.
* **Cross-field:** country match and mismatch; name × address; name × country;
  address × country; combined name + address token-set; source indicator.
* **Missing indicators:** name, address and country missing, both names missing, both
  addresses missing.
* **Blocking and group context:** blocking score, quick score and rank, the strategies that
  fired, candidate count, similarity rank and gap to the best candidate, candidates with the
  same name.

## 12. Training-pair construction

* **Positives:** ground-truth S1 ↔ S2/S3 pairs retrieved by blocking.
* **Hard negatives:** non-matching pairs from the blocking output. They share a country, name
  tokens, n-grams, address tokens or a postal code with the S1 record. The 20 most similar per
  S1 are kept, plus 5 random in-block negatives. There are no random cross-product negatives.

| `reports/training_pairs.json` | Official | Smoke test |
|---|---|---|
| Train pairs (pos / neg) | ⟨fill⟩ | 31,759 / 720,000 |

## 13. ML models

| Model | Setup |
|---|---|
| Logistic Regression (baseline) | StandardScaler, NaN → −1 |
| LightGBM | 63 leaves, lr 0.05, early stopping on the inner hold-out, deterministic |
| XGBoost | hist, depth 8, eta 0.05, early stopping on the inner hold-out |
| Random Forest | 200 trees, depth 16, 50 % bootstrap, ≤ 1.5 M rows |

| Validation, each at its own best threshold (`reports/model_validation.json`) | Official macro F0.5 | Smoke test macro F0.5 (micro P / R) |
|---|---|---|
| Logistic Regression | ⟨fill⟩ | 0.9289 (0.9845 / 0.8862) |
| LightGBM | ⟨fill⟩ | 0.9380 (0.9930 / 0.8957) |
| XGBoost | ⟨fill⟩ | 0.9385 (0.9951 / 0.8955) |
| Random Forest | ⟨fill⟩ | 0.9328 (0.9873 / 0.8927) |

## 14. Ensemble

`P_final = w1·P_LGBM + w2·P_XGB + w3·P_RF`. The reference weights 0.45 / 0.35 / 0.20 and equal
weights are evaluated first. Then all 66 weight vectors on a 0.1-step simplex are scored at
their own best threshold, and the best validation macro F0.5 wins. Platt or isotonic
calibration (fitted on the inner hold-out) is kept only for a gain above 0.001.

| `reports/ensemble_selection.json` | Official | Smoke test |
|---|---|---|
| Macro F0.5 with 0.45 / 0.35 / 0.20 | ⟨fill⟩ | 0.9385 |
| Macro F0.5 with equal weights | ⟨fill⟩ | 0.9378 |
| Selected weights (LGBM / XGB / RF) · calibration | ⟨fill⟩ | 0.3 / 0.5 / 0.2 · none (Platt +0.0006, isotonic +0.0004: below the 0.001 bar) |

## 15. Validation

The split is made at the **Source 1 entity level** with a fixed seed, before any pair is
built: 80 % train, 20 % validation, and 10 % of the train entities as an inner hold-out for
early stopping and calibration. All zero-, one- and multi-match types are present on both
sides. An assertion checks that no S1 entity appears in both train and validation, and no
validation label is used as a feature. Validation keeps the full candidate set, and counts
true matches missed by blocking as errors.

## 16. F0.5 optimisation

**Macro F0.5.** For each S1 entity, the score is 1 if the true set T and predicted set P are
both empty, 0 if exactly one of them is empty, and otherwise F0.5 of that entity's precision
and recall. The macro score is the mean over all entities. The vectorised evaluator was
checked against a set-based reference implementation and gave identical values.

**Decision rule.** Keep candidate *j* if `p_j ≥ t`, `p_j ≥ r·max(p)` and `rank_j ≤ K`, with
t ∈ {0.10 … 0.99}, r ∈ {0, 0.5, 0.7, 0.85} and K ∈ {∞, 3, 10}. If nothing passes, the entity
gets an empty list. Everything that passes is returned.

| `reports/final_validation.json`, `threshold_search.tsv` | Official | Smoke test |
|---|---|---|
| **Macro F0.5** | ⟨fill⟩ | **0.9386** |
| Micro precision / recall / F0.5 / F1 | ⟨fill⟩ | 0.9956 / 0.8934 / 0.9734 / 0.9418 |
| Selected t / r / K | ⟨fill⟩ | 0.60 / 0.70 / ∞ |
| False merges / missed matches | ⟨fill⟩ | 36 / 983 |
| Zero-match entities correct | ⟨fill⟩ | 1,956 / 1,967 |
| One-match / multi-match entities exactly right | ⟨fill⟩ | 3,093 / 3,470 · 1,990 / 2,563 |
| At t = 0.50 (reference only): macro F0.5 / false merges | ⟨fill⟩ | 0.9377 / 92 |

## 17. Inference

1. LightGBM, XGBoost and RF are refit on all labelled entities with the early-stopped
   iteration counts. When calibration was selected, the selection models are kept instead.
2. Test candidates are generated with the same blocking.
3. Features are computed for exactly those pairs.
4. P_LGBM, P_XGB and P_RF are combined into P_final.
5. The validated rule is applied, giving zero, one or several matches per entity.

On the synthetic test split (which has hidden labels), the result was macro F0.5
**0.952**, including France, which is absent from training: 0.951 (India 0.945, US 0.961).

## 18. Output generation

* **`output/matching_results.tsv`:** `source1_entity_id`, `matched_entity_ids`.
* **`output/candidate_pairs.tsv`:** `source1_entity_id`, `candidate_entity_ids`.

Both files are tab-separated, with one row per test S1 entity sorted by ID. They contain only
S2/S3 IDs, sorted and de-duplicated, and allow empty lists; the final matches are a subset of
the candidates.

The outputs are checked by an internal validator, then by the challenge validator from the
supplied `utils/`, then by the **compliance checklist** (`reports/compliance_report.md`). The
checklist confirms that:

* the data came only from the Kaggle input;
* there are no Hugging Face references, network imports or URLs in the project;
* IDs and outputs are valid;
* the threshold was optimised on validation.

## 19. Scalability

* Column-pruned, text-typed Parquet cache; lazy Polars / DuckDB.
* u64-hashed blocking keys, frequency caps, 5 000-entity S1 chunks and hash joins only.
* Sparse TF-IDF retrieval with bounded posting lists.
* Multi-threaded RapidFuzz `cpdist`.
* Feature chunks of 400 k pairs streamed to disk; training sets built part by part.
* Checkpoints at every stage.
* Smoke test at 10× scale (2 M train and 1 M test candidate pairs, 2-core VM): 13 min end to end, peak RSS 3.5 GB.

## 20. Error analysis

`reports/error_analysis.json`, `error_analysis_false_positives.tsv` and
`error_analysis_false_negatives.tsv` hold the details.

| | Official | Smoke test |
|---|---|---|
| Missed matches: not retrieved by blocking / below threshold | ⟨fill⟩ | 519 / 464 |
| False merges with name similarity ≥ 0.9 / sharing the postal code | ⟨fill⟩ | 25 of 36 / 16 of 36 |

In the smoke test, about half of the misses come from blocking: chains with identical names
exceed the per-source cap. The other half fall just below the precision-oriented threshold. The remaining false merges are almost all *same name, same postal code,
different branch*. That is the expected hard case, and the reason the selected threshold is
high.

## 21. Limitations

* Cross-script transliteration relies on character-n-gram and fuzzy similarity only; no
  external transliteration resource is allowed.
* Chains with noisy or missing addresses can lose the true branch at the per-source cap
  (`BER_MAX_CANDS_PER_SOURCE` trades recall for runtime).
* Abbreviation and country-alias dictionaries are intentionally small.
* TF-IDF is fitted per split on unlabelled text (transductive, but label-free).
* Every threshold and weight is selected on one validation split. On the real data, a second
  seed is a cheap stability check.

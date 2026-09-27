# Business Entity Resolution — Amazon ML Challenge 2026

A scalable, reproducible, checkpointed pipeline that decides which **Source 2** and
**Source 3** records describe the same real-world business as each **Source 1**
entity, and writes the two official submission files:

```
output/matching_results.tsv
output/candidate_pairs.tsv
```

The approach: **multi-strategy blocking → two-stage candidate capping → ~90 pairwise
similarity features → Logistic Regression baseline + LightGBM / XGBoost / Random Forest
ensemble → entity-level decision rule tuned for macro F0.5**.

---

## Dataset source

```
Dataset Source:   Supplied Kaggle dataset
Dataset Location: /kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml
```

* No Hugging Face dataset is used.
* No external business databases are used.
* No external business APIs are used.
* No external business information is used.
* All entity resolution is performed using only the supplied challenge data.

Nothing is downloaded at any stage: no geocoding, no web search, no business directories, no
government registries and no commercial entity-resolution services. The only libraries used
are standard open-source Python packages that process the supplied files locally (see
`requirements.txt`). The final `submission` stage checks this automatically (see
**Compliance check** below).

## 1. Project overview

| Stage | Module | What it does |
|---|---|---|
| Discover, load, validate | `data_loader.py` | Recursive discovery under the Kaggle input folder, TSV/CSV/Parquet readers, schema, ID and ground-truth checks |
| EDA | `evaluation.py` (`run_eda`) | Full data profile with DuckDB/Polars and findings turned into modelling decisions |
| Normalise | `preprocessing.py` | Conservative name/address/country normalisation as vectorised Polars expressions |
| Block | `blocking.py` | 10 key-based strategies, country-scoped, frequency-capped inverted indexes |
| Candidates | `candidate_generation.py` | Union, fuzzy fallback (sparse TF-IDF + RapidFuzz), two-stage capping, recall measurement |
| Pairs | `training_pairs.py` | Source-1 entity-level split, positives + hard negatives |
| Features | `feature_engineering.py` | Name, address, numeric/postal, country, cross-field, missing-value, blocking and group-context features |
| Models | `models.py` | LR baseline, LightGBM, XGBoost, Random Forest |
| Ensemble | `ensemble.py` | Weighted probabilities, weight search, optional Platt/isotonic calibration |
| Decision | `threshold_optimization.py` | Threshold / relative-to-best / top-K rule chosen on validation macro F0.5 |
| Inference | `inference.py` | Refit on all labelled data, score test candidates, apply the rule |
| Submission | `submission.py` | Deterministic TSVs, internal validator, official validator, compliance check |
| Orchestration | `main.py` | `--stage` CLI, checkpoints, logging |
| Settings | `config.py` | `DATA_ROOT`, output paths, seed, caps, model parameters and grids |

## 2. Problem statement

Source 1 is the de-duplicated reference. For every Source 1 entity the system must output
**all** matching S2/S3 records — zero, one or many. One-to-one matching is never assumed, and
`argmax` is never the only decision. The challenge metric is **macro F0.5** at the Source 1
entity level, where precision counts more than recall. False merges are expensive, and a
correct empty list for a no-match entity scores in full.

## 3. Running on Kaggle

1. Attach the supplied dataset to the notebook, so it appears under
   `/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml`.
2. Put this code in the notebook, either uploaded as a dataset or copied into
   `/kaggle/working/code/`.
3. Run:

```python
!pip install -q -r /kaggle/working/code/business_entity_resolution/requirements.txt   # only if a package is missing
!python /kaggle/working/code/business_entity_resolution/src/main.py --stage all
```

`/kaggle/input` is read-only, so on Kaggle every artefact goes to `/kaggle/working/artifacts/`
and the submission goes to `/kaggle/working/output/`. Both are detected automatically. The
Kaggle image normally has every package in `requirements.txt` pre-installed. If internet is
off and one is missing, attach its wheel as a dataset and `pip install` it from there.

Individual stages:

```bash
python src/main.py --stage eda          # discovery, validation, normalisation, EDA
python src/main.py --stage preprocess
python src/main.py --stage blocking     # candidates + candidate recall
python src/main.py --stage train        # features, pairs, LR / LGBM / XGB / RF
python src/main.py --stage evaluate     # ensemble, threshold, final validation, error analysis
python src/main.py --stage inference
python src/main.py --stage submission   # TSVs, validators, compliance check
python src/main.py --stage compliance   # re-check existing outputs only
python src/main.py --stage all
```

## 4. Dataset loading

`config.DATA_ROOT` is the only data location. It can be overridden with `BER_DATA_ROOT` or
`--data-root` for a local copy of the same files.

* **Discovery:** `sorted(p for p in DATA_ROOT.rglob("*") if p.is_file())`. Each file is mapped
  to one of the seven logical tables whatever the folder layout (`train_source1.tsv`,
  `dataset/train/train_source1.tsv`, `train/source1.parquet`, sharded Parquet, …). Other files
  (README, `utils/`) are ignored.
* **Reader by extension:** `.tsv` (tab), `.csv` (comma) and `.parquet`. If a table exists in
  several formats, Parquet is preferred.
* **Memory:** one table at a time, only the required columns. Every value is read as text, so
  IDs, PIN and ZIP codes keep their leading zeros, and empty strings are kept so EDA can tell
  null, empty and whitespace-only apart. Each table is converted once to
  `artifacts/raw/<table>.parquet`, together with a manifest of file sizes, formats and original
  dtypes. Later stages read these lazily with Polars / DuckDB, so there is never a
  `pd.concat` of all tables.
* **Errors:** a missing dataset folder, a missing table, missing columns and unreadable or
  corrupt files all stop the run with a clear message.

## 5. Dataset structure

| Table | Columns |
|---|---|
| `train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs; empty = no match) |

ID prefixes: `S1-`, `S2-`, `S3-`. `country` is an **open set**: the test data also contains
France, which never appears in training. The test set has no labels, so validation comes from
a Source 1 entity split of the training data.

## 6. Data validation (`artifacts/reports/validation_report.json`)

* **Every source table:** row count, unique IDs, missing IDs, invalid or unexpected prefixes
  (S1/S2/S3 per table, including IDs belonging to another source), duplicate IDs and the rows
  in duplicate-ID groups.
* **Ground truth:** duplicate S1 rows, S1 IDs missing from `train_source1`, S2 IDs missing
  from `train_source2`, S3 IDs missing from `train_source3`, S1 IDs used as targets,
  unexpected prefixes, duplicate IDs within a row, self-matches, and targets shared by
  several S1 entities.
* **What stops the run:** an empty table, missing IDs, wrong prefixes and duplicate
  ground-truth rows. Duplicate entity IDs are reported as warnings and handled, or become
  errors with `BER_STRICT_VALIDATION=1`. Invalid ground-truth targets are ignored for
  training, and the count is reported.

## 7. EDA (`artifacts/reports/eda.json`, `eda.md`, `dataset_quality_report.md`)

* **Files:** format, size, rows, columns, original dtypes, unique values per column.
* **Missing values per column:** null, empty string, whitespace-only, null-like tokens
  (`NA`, `null`, `-`, …), and combined count and %.
* **Duplicates:** exact duplicate records, duplicate entity IDs and rows in those groups, and
  after normalisation duplicate names, duplicate addresses and same name + address.
* **Names and addresses:** length min / max / mean / median / std, word-count
  min / max / mean / median, numeric-token frequency, postal/PIN availability, non-ASCII
  share, top tokens, Unicode character classes.
* **Countries:** raw and normalised frequencies with percentages per source, missing counts,
  and countries that appear only in test.
* **Ground truth:** zero / one / multi-match counts and %, match-count distribution, entities
  with S2 matches, with S3 matches and with both, multi-match mean / median / max, S2 vs S3
  share, and entities with several matches in one source.
* **Cross-source consistency of true matches** (overall, S2, S3): shared country, exact name,
  exact core name, shared name token, shared address token, shared numeric token, postal and
  building-number agreement, exact address.
* **Findings → modelling decisions:** each measured fact is turned into the design choice it
  drives (for example "34% of records share a name" → address evidence weighted in re-ranking).
* **`dataset_quality_report.md`** fills every metric of the required quality report (size,
  missing values, duplicates, India / USA / France / other, ground truth, text statistics,
  candidate generation and validation) from the run's own reports.

## 8. Preprocessing

Rules are written once as Polars expressions (Rust regex, no Python loops). The scalar
`normalize_name()`, `normalize_address()` and `normalize_country()` run the same expressions.

* **Names:** NFKC → lowercase → strip Latin accents only (other scripts untouched) → collapse
  dotted acronyms (`S.A.S.` → `sas`) → drop the `M/s` honorific → punctuation to spaces,
  `&` → `and` → token-level mapping of legal forms and safe abbreviations (`pvt` → `private`,
  `ltd` → `limited`, `corp` → `corporation`, `mfg` → `manufacturing`, …). For example,
  `ABC Technologies Pvt. Ltd.` becomes `abc technologies private limited`. The **core name**
  drops legal forms and fillers.
* **Addresses:** the same cleaning, plus PIN repair (`560 001` → `560001`), ZIP+4 → ZIP,
  `#` → `number`, letter/digit splitting (`plot12` → `plot 12`) and street-type abbreviations.
  Numeric tokens, postal/PIN candidates (5–6 digits), building numbers and the primary postal
  code are extracted and preserved.
* **Country:** formatting normalisation (Unicode, case, punctuation, whitespace). Spelling
  variants of the same country (`USA` / `United States` → `us`) are also unified, because
  otherwise true matches would look like cross-country pairs. This can be switched off with
  `BER_COUNTRY_ALIASES=0`. Unknown values pass through unchanged (open set), and missing stays
  null.
* **Missing values** never remove records. They become explicit indicator features.

## 9. Blocking and candidate generation

No S1×S2 or S1×S3 Cartesian product is ever built. Each strategy emits hashed keys, the
candidate side becomes a frequency-capped inverted index, and Source 1 chunks are hash-joined
against it.

| Strategy | Key |
|---|---|
| Country scoping | every key below except the two exact-name keys is prefixed with the country |
| `exact_name` / `exact_core` | exact normalised / core name (unscoped, so they survive a wrong or missing country) |
| `name_signature` | sorted unique core tokens (word order) |
| `name_token` | the 3 rarest informative core tokens |
| `char_ngram` | the 4 rarest character 3-grams of the core name (typos, transliteration) |
| `addr_token` / `addr_signature` | rarest address words / sorted address words |
| `postal_name`, `postal_building`, `token_postal` | postal/PIN + name prefix / building number / rare name token |
| `fuzzy` | TF-IDF char-3-gram sparse cosine (country-partitioned) + RapidFuzz re-rank, for S1 records with < 3 candidates from a source |

**Country scoping keeps missing-country records.** A candidate emits `c|k` (or `?|k` when it
has no country) plus `*|k`. A Source 1 record with country *c* queries `c|k` and `?|k`; one
without a country queries `*|k`. Records without a country are never excluded, and unseen
countries need no special handling.

**Union and capping.** Pairs from all strategies are unioned and de-duplicated. Each pair is
scored Σ wₛ·log(1 + N/df). Per (S1, source), the top 200 by that score are re-ranked with a
cheap RapidFuzz score (name 0.50, address / postal / building 0.45, blocking 0.05; a missing
address or postal code counts as neutral, not as a mismatch), and the top 25 are kept.
This capped set is **exactly** what the model scores and what `candidate_pairs.tsv` contains.

**Candidate recall** (`candidate_recall_train.json`) reports recall overall, per source and
before capping; hits per strategy; mean / median / max candidates per S1; S1 entities with no
candidates; total possible pairs; and the reduction ratio. Validation-only candidate recall is
in `final_validation.json`.

## 10. Feature engineering (≈ 90 features, never raw IDs)

* **Name:** exact and exact-core match; Levenshtein; normalised edit (Indel) ratio; Jaro;
  Jaro-Winkler; token sort / set / partial; prefix; character-n-gram and word TF-IDF cosine;
  token Jaccard / overlap / common count / common ratio; length difference and ratio;
  word-count difference; acronym match.
* **Address:** exact match; Levenshtein; Jaro; Jaro-Winkler; token sort / set / partial;
  character TF-IDF cosine; token Jaccard / overlap / common; numeric-token overlap and Jaccard;
  postal exact, prefix, both-present and conflict; building-number match; length difference
  and ratio. All are NaN when an address is missing.
* **Cross-field:** country match and mismatch; country missing; name × address agreement;
  name × country; address × country; combined name + address token-set; source indicator.
* **Missing indicators:** name / address missing on each side, both names missing, both
  addresses missing.
* **Blocking and group context:** blocking score, quick score, rank, strategies that fired,
  number of candidates, similarity rank and gap to the best candidate, candidates with the
  same name.

## 11. Training pairs, validation and models

* **Split** by **Source 1 entity** with a fixed seed: 80 % train, 20 % validation, and 10 % of
  the train entities as an inner hold-out for early stopping and calibration. Zero-, single-
  and multi-match entities are present in both sides. An assertion checks that no S1 entity
  appears in both train and validation.
* **Positives:** ground-truth S1 ↔ S2/S3 pairs retrieved by blocking.
* **Hard negatives:** the 20 most similar non-matches from each S1 entity's candidate list,
  plus 5 random in-block ones. There are no random cross-product negatives.
* **Models:** Logistic Regression baseline, LightGBM, XGBoost and Random Forest, all on the
  same features and split.

## 12. Ensemble and F0.5 optimisation

`P_final = w1·P_LGBM + w2·P_XGB + w3·P_RF`. The documented starting point 0.45 / 0.35 / 0.20
and equal weights are evaluated first as references. Then all 66 weight vectors on a 0.1-step
simplex are scored, each at its own best threshold, on **validation macro F0.5**. Platt and
isotonic calibration are kept only if they help.

**Metric.** For each Source 1 entity with true set T and predicted set P:

* if T and P are both empty, the score is 1;
* if exactly one of them is empty, the score is 0;
* otherwise, F0.5 = 1.25·P·R / (0.25·P + R).

The macro score is the mean over all entities. Entities whose true matches were never
retrieved count as misses. Micro precision / recall / F0.5 / F1 are reported too
(`BER_OBJECTIVE=micro_f0_5` switches the selection target).

**Decision rule.** Candidate *j* is kept when `p_j ≥ t`, `p_j ≥ r·max(p)` and `rank_j ≤ K`.
The grid is t ∈ {0.10 … 0.95, 0.97, 0.99}, r ∈ {0, 0.5, 0.7, 0.85} and K ∈ {∞, 3, 10}. When
nothing passes, the entity gets an empty list (no-match). Everything that passes is returned
(multi-match). `threshold = 0.5` is never assumed; its score is reported only for comparison.
`MATCH_THRESHOLD = 0.80` in `config.py` is a fallback used only if no validated rule exists.

**Reports:**

* `model_validation.json`: per-model scores, including the LR baseline.
* `ensemble_selection.json` and `ensemble_weight_search.tsv`.
* `threshold_search.tsv`: precision, recall, F0.5, false merges, missed matches and zero-match
  accuracy for every threshold.
* `decision_rule_search.tsv`.
* `final_validation.json`: macro and micro scores, false merges, missed matches, correctly
  identified zero-, one- and multi-match entities, and validation candidate recall.
* `error_analysis*.{json,tsv}`.
* `feature_importance.json`.

## 13. Output generation and submission validation

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00003
```

`candidate_pairs.tsv` has the same layout with `candidate_entity_ids`. Both files are
tab-separated, have exactly one row per test Source 1 entity sorted by ID, contain only S2/S3
test IDs (sorted, de-duplicated, never S1), allow empty lists, and are deterministic. The
internal validator fails on missing or duplicate rows, wrong headers, unknown or S1 IDs,
duplicate IDs, or any match missing from `candidate_pairs.tsv`. The challenge validator
(`utils/validate_submission.py`, found in the supplied dataset folder or at `BER_VALIDATOR`)
is then run with `--test-dir` pointing at the supplied test files.

## 14. Compliance check (`artifacts/reports/compliance_report.md`)

After the outputs are written, every item of the final checklist is verified automatically,
and the run fails if any item fails:

* **Data source:**
  * all loaded files come from `DATA_ROOT`;
  * the whole project text (code, README, requirements, documentation) contains no Hugging
    Face reference or authentication;
  * no Python module imports a network or data-download library (requests, urllib, sockets,
    Kaggle API, geocoders, scrapers, dataset hubs, cloud storage);
  * no URL appears in the code.
* **Data validity:** required columns exist, and source and ground-truth IDs are valid.
* **Output files:**
  * outputs contain only S2/S3 IDs, and matches ⊆ candidates;
  * every test S1 entity appears exactly once, with no duplicate IDs;
  * empty lists and multi-matches are supported;
  * files are TSV;
  * the threshold was optimised on validation;
  * the validators pass.

## 15. Configuration

Everything lives in `src/config.py` (`DATA_ROOT`, `OUTPUT_DIR`, `RANDOM_STATE`,
`MATCH_THRESHOLD`, the source prefixes, and the dataclasses for blocking, models and search
grids). Common overrides:

| Setting | Env / CLI | Default |
|---|---|---|
| dataset folder | `BER_DATA_ROOT`, `--data-root` | `/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml` |
| artifacts | `BER_ARTIFACTS_DIR`, `--artifacts-dir` | `/kaggle/working/artifacts` on Kaggle, else `artifacts/` |
| output | `BER_OUTPUT_DIR`, `--output-dir` | `/kaggle/working/output` on Kaggle, else `TEAM_submission/output` |
| seed | `BER_SEED`, `--seed` | 42 |
| selection objective | `BER_OBJECTIVE` | `macro_f0_5` |
| labelled S1 cap | `BER_MAX_TRAIN_S1`, `--max-train-s1` | all |
| candidates per source / pre-rank depth | `BER_MAX_CANDS_PER_SOURCE`, `BER_PRE_RANK_KEEP` | 25 / 200 |
| block-size caps | `BER_MAX_BLOCK_{EXACT,TOKEN,NGRAM,ADDR,COMBO}` | 5000 / 1000 / 600 / 600 / 1500 |
| chunking | `BER_S1_CHUNK`, `BER_FEATURE_CHUNK` | 5 000 S1 / 400 000 pairs |
| workers | `BER_WORKERS`, `--workers` | all cores |
| ground-truth policy | `BER_GT_ABSENT_POLICY` | `auto` |
| strict duplicate-ID validation | `BER_STRICT_VALIDATION` | off |
| country alias mapping | `BER_COUNTRY_ALIASES` | on |

## 16. Hardware, runtime and checkpointing

* **Hardware:** a Kaggle CPU notebook (4 cores, ~30 GB RAM, 20 GB `/kaggle/working`) is enough.
  The main memory levers are `BER_S1_CHUNK`, `BER_FEATURE_CHUNK` and `BER_MAX_TRAIN_S1`.
* **Runtime:** measured on a 2-core VM with a synthetic dataset shaped like the challenge
  (40 k S1, 350 k S2 + S3; 2 M train and 1 M test candidate pairs): about 13 minutes end to
  end, with a peak RSS of 3.5 GB. Runtime grows linearly with the number of candidate pairs.
* **Checkpointing:** validation report, raw and normalised tables, TF-IDF indexes, candidate
  pairs, feature parts, training pairs, models, validation predictions, ensemble weights,
  decision rule, test predictions and the submission all persist in `artifacts/`. A rerun
  skips finished work. `--force` recomputes the selected stage(s).

## 17. Reproducibility

One seed drives the split, sampling and all models (LightGBM runs with `deterministic=True`),
and every ranking uses explicit tie-breaks. Two identical runs produce byte-identical output
files, checked by MD5. The configuration used is saved to `artifacts/config_used.json`, and
the log to `artifacts/logs/pipeline.log`.

## 18. Known limitations

* Transliteration between scripts is only covered by character-n-gram and fuzzy similarity.
  No external transliteration resource is allowed.
* Chains (same name, many branches) depend on address evidence. A noisy or missing address can
  push the true branch below the per-source cap.
* Abbreviation dictionaries are deliberately small and conservative.
* TF-IDF vocabularies are fitted per split on unlabelled text. This is transductive, but uses
  no labels.

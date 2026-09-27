# Amazon ML Challenge 2026 — Business Entity Resolution

Comprehensive, production-grade business entity resolution pipeline designed for the Amazon ML Challenge 2026. Resolves multi-source business entities (`source1` queries against `source2` and `source3` candidates) with heavy noise, abbreviations, legal suffix variations, and missing address components.

Optimized directly for the precision-weighted **$F_{0.5}$ metric** ($\beta = 0.5$), which penalizes False Positives (incorrect merges) 4x more heavily than False Negatives.

---

## Key Results on Hold-out Validation (20,000 Entities, Zero-Leakage Split)

| Model | Macro $F_{0.5}$ | Micro Precision | Micro Recall | Micro $F_{0.5}$ | False Merges |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble (90% LGBM + 10% XGB)** | **0.8956** | **98.81%** | **79.17%** | **0.9414** | **660** |
| LightGBM (1,975 trees) | 0.8954 | **98.83%** | 79.15% | 0.9414 | **649** |
| XGBoost (1,622 trees) | 0.8953 | 98.64% | **79.38%** | 0.9408 | 755 |
| Random Forest (200 trees) | 0.8810 | 97.88% | 77.32% | 0.9294 | 1,155 |
| Logistic Regression (Baseline) | 0.8402 | 95.09% | 73.43% | 0.8979 | 2,618 |

- **Optimal Decision Rule:**
  $$P_j \ge 0.65 \quad \text{AND} \quad P_j \ge 0.70 \cdot \max(P) \quad \text{AND} \quad \text{rank}_j \le K \ (\text{unlimited})$$
- **Singleton Accuracy:** **95.9%** (1,111 / 1,158) of zero-match entities correctly identified.
- **Candidate Reduction Ratio:** **>99.98%** reduction in Cartesian pair space via multi-strategy hash blocking and fuzzy fallback.

---

## Pipeline Architecture

1. **`eda`**: DuckDB and Polars profiling across all 7 canonical tables with character distribution and missingness audits.
2. **`preprocess`**: Vectorized unicode NFKC normalization, legal form token canonization, address tokenization, and open-set country mapping.
3. **`blocking`**: 9 hashed blocking keys (exact name, core name, sorted signatures, rare tokens, address/postal combinations) + sparse TF-IDF char 3-gram fuzzy retrieval with RapidFuzz re-ranking keeping $\le 25$ candidate pairs per source.
4. **`train`**: Strict Source-1 entity splitting (80/20 with inner hold-out), hard negative mining, 88 similarity/group features, and training LightGBM, XGBoost, Random Forest, and Logistic Regression.
5. **`evaluate`**: Simplex-constrained validation $F_{0.5}$ weight search and threshold/margin parameter optimization.
6. **`inference`**: High-throughput vectorized scoring on test candidate pairs.
7. **`submission`**: TSV generation for `matching_results.tsv` and `candidate_pairs.tsv` + official compliance verification.

---

## Directory Structure

```text
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py                 # Paths, seeds, device autodetection
│       │   ├── data_loader.py            # Dataset discovery and validation
│       │   ├── preprocessing.py          # Vectorized text and address cleaners
│       │   ├── blocking.py               # Hashed blocking keys & indexing
│       │   ├── candidate_generation.py   # Candidate retrieval & recall
│       │   ├── training_pairs.py         # Entity splitting & hard negative mining
│       │   ├── feature_engineering.py    # 88 string, token, and group features
│       │   ├── models.py                 # LightGBM, XGBoost, RF, LR
│       │   ├── ensemble.py               # Calibration and weight optimization
│       │   ├── threshold_optimization.py # F0.5 grid search over (t, r, K)
│       │   ├── inference.py              # Test set scoring
│       │   ├── submission.py             # TSV formatting & compliance checks
│       │   ├── evaluation.py             # GroupEvaluator & metrics
│       │   └── main.py                   # CLI orchestrator
│       ├── artifacts/reports/            # Validation, EDA, and compliance reports
│       └── requirements.txt
├── Documentation_template.md             # Competition documentation template
└── README.md
```

---

## Execution Instructions

```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run full pipeline end-to-end
python code/business_entity_resolution/src/main.py --stage all --max-train-s1 100000

# Or run individual stages
python code/business_entity_resolution/src/main.py --stage eda
python code/business_entity_resolution/src/main.py --stage blocking --max-train-s1 100000
python code/business_entity_resolution/src/main.py --stage train --max-train-s1 100000
python code/business_entity_resolution/src/main.py --stage evaluate
python code/business_entity_resolution/src/main.py --stage inference
python code/business_entity_resolution/src/main.py --stage submission
```

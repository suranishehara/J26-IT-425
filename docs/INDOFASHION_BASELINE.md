# IndoFashion import and initial CLIP pilot — 16 October 2026

Raw location: runtime/datasets/raw/indofashion_dataset. Raw annotations and images were not modified.
Official annotations use JSON Lines in train_data.json, val_data.json and test_data.json.

The mapped subset had 48,821 records and no missing mapped image files. Content hashing found 138 exact-image groups spanning splits.
The derived manifest excludes 1,260 records with conflicting mapped labels for identical images and 9 additional exact duplicates.
Cleaning keeps test, then validation, then training examples; conflicting-label hashes are excluded entirely.
Only exact duplicates are detected; near-duplicates and related product photographs can remain.

## Usable mapped records
| Split | Saree | Blouse | Dress (source gowns) | Kurta | Total |
|---|---:|---:|---:|---:|---:|
| Training | 10,767 | 9,162 | 4,692 | 18,079 | 42,700 |
| Validation | 500 | 500 | 447 | 981 | 2,428 |
| Test | 499 | 499 | 452 | 974 | 2,424 |
| Total | | | | | 47,552 |

Prepared manifest: runtime/datasets/indofashion_clean_01/manifest.jsonl.
Audit reports are in indofashion_import_01 (strict audit) and indofashion_clean_01 (derived cleanup).
Excluded source categories are documented, not silently mapped to supported labels.

## Initial validation pilot
- Frozen CLIP revision and taxonomy 1.0; seed 42.
- 25 examples per mapped category, 100 total; all processed successfully.
- Accuracy: 58/100 = 58%.
- Macro-F1 over the four ground-truth classes: 0.5656.
- Correct category predictions: blouse 6/25, dress 10/25, kurta 17/25, saree 25/25.
- Saree precision is only 0.5319 because other garments were often predicted as sarees; 25/25 saree recall does not establish a perfect saree classifier.
- Full outputs: runtime/evaluations/clip_indofashion_validation_01/report.json and predictions.jsonl.
- No adaptation/training has been performed. Original test split has not been evaluated.

This pilot scores agreement with provider garment-category labels only. It does not establish pattern/texture/drape accuracy or recommendation quality. Inspect misclassified images and source labels before changing prompts or training. Use validation for development; reserve the cleaned test split for the final fixed comparison.

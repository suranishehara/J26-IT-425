# DeepFashion pattern-head development result

## Data and method

Dataset: runtime/datasets/deepfashion_pattern_development_01. From 12,000 seeded raw-image candidates, the upper-garment importer produced 1,697 crops. Missing parsing and unsupported/ambiguous regions were skipped, with no processing failures. The audit found no exact decoded duplicates or cross-split source-group overlap. It cannot rule out near-duplicate, person-identity or source leakage.

Pattern-known training examples: 1,323 (1,160 solid, 106 striped, 32 checked, 25 floral). Pattern-known validation examples: 155 (134 solid, 14 striped, 2 checked, 5 floral). The dataset also contains 151 test crops (148 with known patterns); no test features were extracted and no test performance was calculated.

One 512-dimensional frozen CLIP embedding feeds a trained four-class linear head. Only pattern_type was trained, using AdamW for 100 epochs and validation loss to select the epoch. CLIP and the previous IndoFashion checkpoint remain unchanged. Unknown targets were excluded from this head. Labels remain provider annotations mapped conservatively, not newly verified Sri Lankan labels.

The baseline uses the same four candidate pattern labels, not the broader seven-class General CLIP task. This is an important candidate-set distinction.

## Development validation results

| Metric | Matched four-label CLIP | Trained pattern head |
|---|---:|---:|
| Correct / 155 | 151 | 152 |
| Accuracy | 97.42% | 98.06% |
| Macro-F1 | 0.8809 | 0.8769 |

The head gains one correct prediction overall but slightly reduces macro-F1. It is not a consistent improvement across metrics. Validation was used for checkpoint selection, so these are development results, not independent final-test evidence or statistical significance.

Class imbalance is substantial and checked/floral validation counts are tiny. Do not present 98.06% as general pattern accuracy, Sri Lankan accuracy or a full multi-attribute research result. A small visual check found plausible masks but noisy source labels, including pleated or lace-textured garments labelled striped/floral. Crops remove skin/background but can have holes/disconnected regions; this preprocessing also differs from ordinary uncropped uploads.

## Artifacts and status

- Frozen run: runtime/adaptation/deepfashion_pattern_01.
- Checkpoint, report, per-head history and validation predictions are saved there.
- Train/validation contact sheet: runtime/evaluations/deepfashion_pattern_crop_review_01/contact_sheet.jpg.
- The real pattern head has been trained; it is **not activated** in the application.
- No new garment-type, drape or texture head was trained. Other attributes remain on existing baselines.
- Test evaluation and any further model/development decisions remain separate steps. Do not tune on final test results.

See DEEPFASHION_RESEARCH_USE.md for the user's educational non-commercial research assertion and remaining usage obligations. Raw and derived images must not be redistributed with the code/report.

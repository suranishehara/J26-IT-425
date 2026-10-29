# Full IndoFashion validation comparison — 16 October 2026

All 2,428 cleaned validation images processed successfully. No model training, label changes or threshold tuning took place in this evaluation. The checkpoint hash was identical before and after the run. No test images were read.

| Model | Candidate categories | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original zero-shot CLIP | 8 | 1,544 / 2,428 | 63.59% | 0.6358 |
| Matched zero-shot CLIP | 4 | 1,666 / 2,428 | 68.62% | 0.6511 |
| Adapted frozen-CLIP linear probe | 4 | 2,184 / 2,428 | 89.95% | 0.8989 |

Controlled improvement over the same four-class candidate set: 21.33 percentage points in accuracy. There are 597 baseline mistakes corrected and 79 new mistakes introduced by adaptation. No claim of universal improvement is made.

## Adapted per-class results
| Category | Support | Correct | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Blouse | 500 | 490 | 0.9722 | 0.9800 | 0.9761 |
| Dress (source gowns) | 447 | 366 | 0.7409 | 0.8188 | 0.7779 |
| Kurta | 981 | 846 | 0.9256 | 0.8624 | 0.8929 |
| Saree | 500 | 482 | 0.9341 | 0.9640 | 0.9488 |

On the 2,328 validation images outside the previously inspected 100-image pilot, matched baseline accuracy was 68.94% and adapted accuracy 89.99%.

## Error review
The adapted confusion matrix includes 119 kurtas predicted as dresses and 61 dresses predicted as kurtas. Some examples are visually ambiguous, while the provider labels appear incorrect for others.
A sheet of high-score disagreements (up to three per ground-truth category) shows, for example, saree images labelled as blouse or dress, and blouse product images labelled as saree. These observations are candidates for independent manual review, not automatic corrections. Confidence alone does not establish that the model or label is correct.
Original provider labels were preserved throughout the comparison.

## Artifacts
runtime/evaluations/indofashion_full_validation_01 contains report.json, predictions.json, validation_features.npz, review_examples.json, adapted_error_contact_sheet.jpg and comparison.html.
The report includes full confusion matrices, per-class metrics, cohort results, file hashes and coverage. The evaluation script rejects a changed manifest and verifies that the checkpoint did not change during evaluation.

## Frozen development choice
Keep the four-class adapter as an optional South Asian category classifier. General CLIP remains the default for wider garment coverage. The saved checkpoint and training settings remain unchanged from indofashion_probe_01; see config/garment_probe_release.json for fingerprints.
Use the unchanged checkpoint and fixed class definitions in the eventual cleaned-test comparison. No test-set results were obtained in this stage.

## Limits
These are development-validation scores against mapped IndoFashion category labels, not accuracy for Sri Lankan draping, visible texture, skin tone, fit, or recommendation usefulness. Exact duplicate removal does not eliminate all possible product/near-duplicate leakage. The final independent test evaluation and Sri Lankan-specific validation remain outstanding.

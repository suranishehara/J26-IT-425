# Final frozen IndoFashion test evaluation — 16 October 2026

The previously reserved cleaned test split was evaluated once using the checkpoint and settings frozen after validation. All 2,424 selected images processed successfully. No training, checkpoint selection, label correction or threshold tuning occurred during this run. Frozen checkpoint, taxonomy and model revision were verified before processing; checkpoint hash was unchanged after processing.

| Model | Candidate categories | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original zero-shot CLIP | 8 | 1,591 / 2,424 | 65.64% | 0.6547 |
| Matched zero-shot CLIP | 4 | 1,697 / 2,424 | 70.01% | 0.6670 |
| Frozen adapted linear probe | 4 | 2,184 / 2,424 | 90.10% | 0.8990 |

Controlled accuracy improvement over the matched four-class baseline: 20.09 percentage points. Adaptation corrected 566 baseline errors and introduced 79 new errors.

## Adapted per-class results
| Class | Support | Correct | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Blouse | 499 | 489 | 0.9760 | 0.9800 | 0.9780 |
| Dress / source gowns | 452 | 357 | 0.7711 | 0.7898 | 0.7803 |
| Kurta | 974 | 860 | 0.9130 | 0.8830 | 0.8977 |
| Saree | 499 | 478 | 0.9228 | 0.9579 | 0.9400 |

The largest adapted confusions are kurta to dress (94) and dress to kurta (72). Provider category labels remain unchanged, including potentially questionable examples. This report is about garment categories only; it does not establish performance on Sri Lankan draping, skin tone, measurements, fit or recommendation usefulness.

## Evaluation integrity
- Checkpoint SHA256: ad8b483cd2de519805772ff4c359b2ccf834a7ec78813fb91d6560f9bce8cdfc.
- Full class taxonomy fingerprint matches config/garment_probe_release.json.
- The cleaned manifest matches the training-run fingerprint.
- Exact content overlap between test and development splits was checked before inference. Related products/near-duplicates may still remain.
- No training was performed and all parameters remain fixed.
- The final test set has now been evaluated. It must not be described as untouched in subsequent experiments or used to choose improvements while still claiming a fresh test comparison.
- Original data and the historical validation reports were not modified.

## Saved artifacts
runtime/evaluations/indofashion_final_test_01/report.json contains counts, coverage, metrics, confusion matrices and hashes. The same directory contains predictions.json, test_features.npz, comparison.html, review_examples.json and an error contact sheet.
config/garment_evaluation_status.json records the current evaluation state. The original release file preserves the pre-test freeze state.

## Report-ready statement
Using a frozen CLIP ViT-B/32 image encoder, a four-class linear head was learned from 7,200 IndoFashion training images, with an additional 800 training-split images used for checkpoint selection. On 2,424 cleaned test images, the adapted head achieved 90.10% accuracy and 0.8990 macro-F1, compared with 70.01% and 0.6670 for a matched four-class zero-shot CLIP baseline. These results quantify category recognition on the mapped IndoFashion subset, not the entire fashion recommendation component.

Body and skin-tone image analysis, local draping validation, sizing integration and recommendation-quality evaluation remain separate work.

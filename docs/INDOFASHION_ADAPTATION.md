# First garment-category adaptation — 16 October 2026

## What was inspected
The original pilot has 42 disagreements with provider labels. An inspection sheet of 12 representative errors is saved at runtime/evaluations/clip_indofashion_validation_01/error_contact_sheet.jpg.
It shows confusion between South Asian blouses and sarees, between kurtas and shirts/dresses, and apparent annotation noise (e.g. accessory pictures labelled as gowns).
No inspected validation labels were changed. Scores below measure agreement with the original mapped provider labels, not independently adjudicated visual truth.

## Training
- Frozen CLIP ViT-B/32 image encoder, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.
- Balanced 8,000 examples from the cleaned TRAIN split only: 2,000 each of blouse, dress (source gowns), kurta and saree.
- 7,200 used for gradient updates; 800 for an internal stratified checkpoint-selection holdout drawn from that training subset.
- Normalized 512-dimensional CLIP features, trained four-output linear head.
- AdamW, learning rate 0.02, weight decay 0.01, 200 epochs; selected epoch 200 by lowest internal-holdout loss.
- External validation images were not used for gradient updates or checkpoint selection.
- No test images were read for training or evaluation.
- Training and validation image failures: zero.
- This is linear-probe adaptation. CLIP backbone weights, pattern, drape and texture predictors were not fine-tuned.

## Comparison on the same 100 validation examples
| Method | Candidate classes | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Original zero-shot CLIP | 8 | 58% | 0.5656 |
| Matched zero-shot CLIP | 4 | 61% | 0.5790 |
| Adapted linear probe | 4 | 89% | 0.8895 |

Primary comparable improvement: 28 percentage points over the matched four-class baseline.
The 58-to-89 comparison changes the number of candidate classes and is not a controlled like-for-like improvement.
Macro-F1 includes the four ground-truth classes; predictions outside those classes in the eight-class baseline still count as errors.

Adapted correct predictions: blouse 23/25, dress 19/25, kurta 22/25, saree 25/25.
The same validation pilot was previously inspected; these are development results, not a final held-out test estimate.

## Saved artifacts
runtime/adaptation/indofashion_probe_01 contains:
- probe.safetensors: trained head weights.
- report.json: recipe, model revision, data hashes, sample counts, metrics and confusion matrices.
- selected_samples.json: exact training, internal-holdout and validation sample IDs.
- training_history.json: fit/internal-holdout loss per epoch.
- comparison_predictions.json: each validation truth and all three predictions.
- training_features.npz and validation_features.npz: frozen feature caches.

## Using it
Restart backend/app.py and select the adapted IndoFashion classifier in the garment-image panel.
The general eight-class CLIP baseline remains the default.
The adapter always chooses among blouse/dress/kurta/saree; it has no reliable unknown-category rejection.
IndoFashion blouses are primarily South Asian examples. A smoke test on the synthetic Western floral blouse returned kurta, whereas general CLIP returned blouse. Therefore the adapter is optional, not a global replacement for Western garments.
All suggestions still need review. No automatic confidence threshold has been validated.

The image-analysis HTTP endpoint was tested with classifier_mode=indofashion_probe and returned HTTP 200 with the saved weights loaded on CUDA.

## Remaining research
Evaluate on a larger validation set and review label noise without retrospectively altering the locked pilot. Add an appropriate Western dataset before attempting a unified trained classifier. Validate Sri Lankan draping labels separately. Reserve the cleaned original test split for a final frozen comparison; do not use test results to tune the model.

Reproduction uses a NEW output directory:
```powershell
.\.venv\Scripts\python.exe scripts\train_garment_probe.py --manifest runtime\datasets\indofashion_clean_01\manifest.jsonl --baseline runtime\evaluations\clip_indofashion_validation_01\predictions.jsonl --output runtime\adaptation\indofashion_probe_reproduction --per-class 2000 --batch-size 16
```
The web adapter loader currently targets indofashion_probe_01; reproduction does not replace the deployed checkpoint automatically.

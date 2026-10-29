# Nine-category ConvNeXt V2 training experiment

This run is a **provisional development experiment**, not independent fabric-identification accuracy. The prepared collection has unresolved similarity candidates, filename/content-derived groups rather than verified source/specimen identities, and very small evaluation groups for leather/linen. Test membership is reserved; no classifier runs on test images in training or smoke scripts.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\train_convnextv2.py --nine-categories --acknowledge-provisional-split
```

Default dataset: `runtime/datasets/fabric_nine_categories_01`; default new output: `runtime/training/convnextv2_nine_categories_01`. Existing output directories are rejected. A repeat requires a new `--output`; the dataset's recorded manifest/scope hashes must match. The acknowledgement explicitly recognizes grouping/similarity limitations rather than approving the split as scientifically independent.

The seven-category default invocation is preserved for reproducibility. Nine-category runs use the new guarded loader; they do not rewrite active release configs, old model reports or original provider datasets.

## Protocol

- Nine outputs: cotton, denim, leather, linen, nylon, polyester, silk, viscose, wool. No cotton_mixed output and no batik/handloom/pattern/weave/texture heads.
- Same pinned non-commercial FCMAE/ImageNet-22K/1K ConvNeXt V2-Tiny pretrained checkpoint as the earlier experiment, not a relabelled old seven-output head. A fresh nine-output head is trained.
- 3,432 train / 740 validation images. The 740-image test partition is never passed to the development loader.
- FP32 feature extraction with the recorded standard timm 224-pixel transform. Frozen normalized 768-D features; weighted cross entropy; 100 full-feature head epochs, AdamW LR 0.02 / weight decay 0.01; choose validation macro-F1.
- Five final-stage fine-tuning epochs, AMP float16 training / FP32 validation, batch 16, gradient clip 1.0, backbone/head-normalization LR 0.0001 and head LR 0.001. Earlier stages remain fixed and stochastic depth stays in eval mode. Seed 42. No extra augmentation or oversampling.
- Loss weights must match prepared **training-only** inverse-frequency weights in the exact nine-label order. Validation/test distributions are not rebalanced.
- Keep the best validation macro-F1 checkpoint, including frozen epoch zero if fine-tuning fails to improve it. Verify protected tensors unchanged, save/reload full safetensors and repeat FP32 validation predictions.

Full results are valid only after `report.json` is written. Reports pin manifest/scope/source-weight hashes, class weights/counts, group counts and similarity caveats. Saved validation logits support a development error review without additional image inference. Original photos are not modified.

## Completed run, 18 October 2026

Selected **epoch 4/5**, with **90.14% image validation accuracy** (667/740 correct) and **macro-F1 0.8690**. Frozen-head baseline on the exact same split: **71.89% / macro-F1 0.6720**, selected head epoch 100/100. Epoch 5 reached 89.86% / 0.8683 and was not selected. These are development results, not independent or final-test accuracy.

Checkpoint SHA-256: `7a32c9ebb487d314ea148d51d8042577de0ac89bea2579055d01a55428243aa9`. Training manifest SHA-256: `a4341ee939350b8b0b161b3d3bbf276137a1aa66965b42178d30ddfa13e9ec68`. Extraction/training/evaluation elapsed 431.62 seconds after pretrained loading on RTX 4050 Laptop GPU; peak allocated GPU memory 631,841,792 bytes. Actual loss weights matched the prepared training-only values. Frozen stages and full-state FP32 reload predictions were verified unchanged/reproduced.

| Category | Validation images | F1 |
| --- | ---: | ---: |
| Cotton | 348 | 0.9353 |
| Denim | 96 | 0.9519 |
| Leather | 8 | 1.0000 |
| Linen | 12 | 0.8462 |
| Nylon | 36 | 0.9014 |
| Polyester | 132 | 0.8622 |
| Silk | 32 | 0.8125 |
| Viscose | 24 | 0.6818 |
| Wool | 52 | 0.8298 |

Leather's perfect F1 is based on only **two inferred families**, not evidence that leather recognition is solved. Linen has only three evaluation families. Viscose remains the weakest F1 in this split. Per-class confusion and histories are retained in `runtime/training/convnextv2_nine_categories_01/report.json`.

CPU smoke passed on one **training** photo: nine finite logits, unchanged checkpoint and identical repeats; inference-only times approximately 0.075 / 0.074 / 0.069 seconds on this machine (not a representative benchmark). Artifact: `runtime/evaluations/nine_category_checkpoint_smoke_01/report.json`.

Cached validation review found 73 image errors. A separate majority-vote aggregation over 185 inferred families gave 92.43% accuracy; this is an exploratory aggregation, **not** proof those families are independent physical specimens. All error paths, group metrics and source identities are saved in `runtime/evaluations/nine_validation_review_01/report.json`. No labels or splits were changed after viewing outcomes.

Latest suite: **101 tests passed**. No test-image inference or app activation occurred. The old seven-class 72.47% result is on a different data/task and must not be used as a direct improvement claim.

## Verification commands

```powershell
.\.venv\Scripts\python.exe scripts\smoke_nine_checkpoint.py
.\.venv\Scripts\python.exe scripts\review_nine_validation.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The CPU smoke checks one training photo, nine-logit shape, finite outputs, checkpoint identity and repeatability. It does not estimate accuracy or activate the app. Validation review reproduces reported image metrics, lists mistakes and reports a separate inferred-family majority vote; ties use label order. That aggregation is not proof that the inferred families are independent physical specimens.

There is no direct accuracy-improvement claim versus the old seven-category 72.47% result: data and category sets differ. The valid within-run comparison is frozen-head versus partial fine-tuning on this exact nine-class validation split.

## Release boundary

**Integration checkpoint, 18 October 2026:** the user approved activation after the training record below. The nine-output run is now the default V2 release; V1/EfficientNet retain seven outputs. Model construction, output decoding, cache identity and UI categories use each verified report's vocabulary. Old saved records remain accepted. Legacy seven-class evaluators reject nine-class evaluation rather than silently remapping labels. Training reports and frozen preparation snapshots retain historical activation fields; current release JSON and API health determine the active release. Three-model HTTP switching/repeatability passed without opening test images. Independent calibration and final evaluation are still pending.

Historical pre-integration boundary:

The active app remains on its existing seven-category models until a separate reviewed integration step supports a new nine-output release and per-model vocabulary. Old output mappings, saved records, API validation, evaluation tools and dropdown comparison models must not silently be relabelled. Independent calibration, external unknown-data evaluation and final-test inference are not completed by training this candidate.

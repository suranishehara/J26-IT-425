# Local multi-attribute training pipeline

## What is implemented

One shared frozen CLIP encoder feeds independently trained linear heads for garment_type, pattern_type, drape_style and fabric_texture. Unknown labels are masked separately, so an image with known pattern but unknown texture can train only its known targets. Each head supports the classes actually present in its training split; it is not an open-world classifier. Heads with no labels are omitted and the application falls back to general CLIP for those attributes.

No new local research model has been trained yet. IndoFashion does not provide these project's full pattern, Osariya drape and visible-texture labels. The seven generated starter records deliberately contain unknown labels, unassigned splits and unconfirmed permissions. They cannot pass the training audit. They are an example format, not a sufficient dataset or final evaluation set.

## Annotation format

`data/local_attributes/annotations_starter.jsonl` has one JSON record per image. Image paths are relative to the explicitly chosen image root. Add permitted real images and independently review labels using the vocabulary in `backend/taxonomy.py`.

- Set sample_id uniquely and source_group to the physical garment/source group. Keep all views/variants/near-duplicates of one garment in the same split.
- Assign train, validation or test before adaptation. Avoid retailer/photographer/source overlap where feasible; source-group naming is a reviewer responsibility.
- Use `unknown` for insufficient evidence, not `other` as an uncertainty substitute. `other` is an explicit trained class when genuinely supported by annotation evidence.
- Record source.image_kind as real or synthetic, permission_confirmed=true only after checking, and a permission_reference.
- Record annotator and review_status=reviewed after checking. Local draping labels need knowledgeable review. These assertions are not automatic legal or expert verification.
- Documented existing dataset annotations may instead use review_status=provider_annotated with original-annotator attribution and label_provenance containing kind=provider_annotation, reference and mapping_version. This is not an independent human review by the current researcher; permission checks still apply. DeepFashion import details are in DEEPFASHION_MULTIMODAL_IMPORT.md.
- Saree Nivi/Osariya labels require garment_type=saree. Sarees cannot have not_applicable drape.
- Textile production techniques and fibre composition are not inferred from visible texture labels.

At least two observed training classes per requested head, five training images and two validation images per class are required by the implementation. **These are software guardrails, not an adequate research sample-size recommendation.** Use representative real images and a separate test split for meaningful claims. Include varied lighting, pose, sources and garment construction.

## Audit, train, evaluate, activate

Run from the Component 2 folder. These commands assume you have created a genuinely reviewed `annotations_reviewed.jsonl`; renaming the starter does not make it reviewed.

```powershell
.\.venv\Scripts\python.exe scripts\train_local_attributes.py --manifest data\local_attributes\annotations_reviewed.jsonl --image-root . --audit-only

.\.venv\Scripts\python.exe scripts\train_local_attributes.py --manifest data\local_attributes\annotations_reviewed.jsonl --image-root . --output runtime\adaptation\local_attributes_01

.\.venv\Scripts\python.exe scripts\evaluate_local_attributes.py --run runtime\adaptation\local_attributes_01 --manifest data\local_attributes\annotations_reviewed.jsonl --image-root . --output runtime\evaluations\local_attributes_validation_01 --split validation
```

Use `--attributes pattern_type drape_style fabric_texture` to train only those heads. Validation selects the lowest-loss epoch independently per head, so validation metrics are development results, not unbiased final performance. Frozen CLIP features are extracted only from training/validation during training. AdamW trains the small heads; no CLIP or existing IndoFashion weights are modified.

The manifest audit reads every image to verify its integrity and decoded-pixel SHA, including test files, but does not extract test features. It rejects exact RGB duplicates anywhere, including within a split, and cross-split source-group overlap. It does not automatically detect near-duplicates. Evaluation requires unchanged manifest, images and groups and verifies checkpoint/model/taxonomy fingerprints.

Once development is complete, freeze the run and evaluate the test set explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_local_attributes.py --run runtime\adaptation\local_attributes_01 --manifest data\local_attributes\annotations_reviewed.jsonl --image-root . --output runtime\evaluations\local_attributes_test_01 --split test --final-test
```

Do not tune on those final results. Metrics include accuracy, per-class precision/recall/F1, macro-F1 and confusion matrices against matched-label zero-shot CLIP. Unsupported test truth labels count as errors and are reported. Reports include real/synthetic counts; synthetic-only performance cannot establish real-world Sri Lankan performance.

Activate a reviewed run explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\activate_local_attributes.py --run runtime\adaptation\local_attributes_01
```

Restart the backend and refresh the browser. The third **Local attributes** mode then becomes selectable; before activation it is disabled. This is deployment, not certification of research validity. Existing active releases are not overwritten by the activation command. Available local heads also support Grad-CAM for their predicted labels. Missing heads use CLIP, and the non-saree drape rule still applies.

## Software checks

`tests/test_local_attributes.py` checks unknown masking, group leakage, duplicate images, permission/review metadata, path boundaries, drape consistency, minimum counts, missing training classes, and deterministic head fitting. `scripts/smoke_local_pipeline.py` exercises real CLIP feature extraction and train/test mechanics using explicitly generated test pixels. It never activates its fixture model or claims local research performance.

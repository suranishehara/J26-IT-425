# Component 3 — saved stopping point

**Saved: 19 October 2026, Asia/Colombo. Implementation paused at the user's request before starting Component 4.**

This is the authoritative resume summary for Component 3. It records the current files/artifacts; it is not a full project backup or a newly trained model checkpoint. Existing trained weights, datasets and SQLite catalogue remain in their original locations. Components 1 and 2 were not changed by this checkpoint. Do not resume Component 3 implementation unless requested.

## Status at pause

Working end-to-end **research prototype**, not a scientifically completed/evaluated component. Rough planning estimate previously discussed: about 75% of the original scope, not a measured acceptance/completion percentage.

Current flow:

1. Upload a permitted close-up, validate image and classify into candidate fabric categories.
2. Apply/correct the category; enter known attributes or choose a saved specimen / explicit typical-profile variant.
3. Enter measurements, fit and style; reuse Component 2's shared validator.
4. Generate four provisional design ideas with transparent rule points, conflicts and illustrative positive-ease calculations.
5. Optionally compare property-aware ranking with a style/measurement-only baseline.

**Latest verification: 123 automated tests passed**, rerun when saving this checkpoint. JavaScript syntax and real frontend-handler tests passed. Live port-8002 typical-profile comparison returned cotton Oxford 194 GSM, structured-shirt score 7 versus baseline 2. This was a software mechanics check, not measurement of any uploaded sample. No final-test classifier inference was performed for these changes.

## Implemented features

### 01 — Fabric photograph

- Permission-gated local JPEG/PNG/WebP upload, size/resolution checks, EXIF orientation, preview and top-three category scores.
- Default **Experimental ConvNeXt V2-Tiny**, plus **Experimental ConvNeXt-Tiny** and **Experimental EfficientNet-B0** comparison choices. Exact per-model vocabulary and checksum verification; strict CPU model loading and cache switching.
- Concise UI: **Use top category**, no former top warning banner, no review-only dropdown item or review-required model-name suffix. Errors/quality checks and final review confirmation still exist.
- Raw softmax scores are relative scores, not calibrated correctness probabilities. Capture/review heuristics are not validated unknown-fabric detection.

### 02 — Attributes and properties

- Category transfer and manual material, technique, pattern and texture; unknown stays valid.
- **Check fabric record** validates labels, numerical ranges and property-source references; shows ranking inputs and save readiness. It does not certify sources or auto-confirm review.
- Drapability and weight band influence ranking; GSM/stretch are optional context only.
- Append-only SQLite specimen records with integrity checks, reviewer/sample name and per-property provenance. No photos/body profiles are persisted by the specimen catalogue.
- Saved-sample selector directly fills read-only fields, preserves mixed per-property sources, restores previous manual draft and guards against stale asynchronous responses.
- Optional appearance/GSM/stretch sections collapse to reduce typing; populated saved values expand them.
- Catalogue JSON export/import, batch validation before transaction, source-review confirmation and duplicate skipping. Existing records are preserved; imports append. No genuine sample catalogue was fabricated/populated by the assistant.
- **Typical fabric profile** demonstration mode: explicit variants for cotton Oxford shirting, polyester chiffon and rigid cotton denim. Sources and interpretation are in `docs/TYPICAL_PROFILES.md`. Compact label, source link, manual-draft restoration; server reconstructs allowlisted profile values and ignores client overrides. These profiles cannot be saved/imported as documented physical-sample measurements.
- There is no default for every category and no physical-property inference from the downloaded images. Typical profiles are not sample evidence or evaluation ground truth.

### 03 — Measurements and style

- Height, chest, waist, hips, shoulder width and inseam, centimetres/inches, fit/style/silhouette choice.
- Read-only reuse of Component 2's `backend/user_profile.py` and `backend/taxonomy.py` via `backend/shared_contract.py`; shared pattern/texture vocabulary and measurement validation/conversion.
- Not an automatic transfer of Component 2's saved profile/catalogue. No skin-tone/body-shape inference in Component 3.
- Garment drape style is distinct from a material's drapability.

### 04 — Provisional design ideas

- Four starter designs: structured shirt/blouse, A-line dress, relaxed tunic/kurta, flowing gathered skirt.
- Versioned rule configuration in `config/design_rules.json`: styles, drapability/weight preferences, weights, positive-ease ranges and relaxed-fit extra.
- Rule SHA-256/version/review status accompanies results. Baseline comparison uses the same rules snapshot.
- **Rule review is pending.** Current ease ranges are illustrative, not pattern-maker validated, cutting patterns or fit guarantees. Seam allowance and negative ease are not calculated. A category does not directly add ranking points.

## Models and immutable artifact identities

Active config: `config/fabric_classifier_release.json`.

### Default nine-category ConvNeXt V2-Tiny

- Run: `runtime/training/convnextv2_nine_categories_01`.
- Architecture: `timm_convnextv2_tiny_fcmae_in22k_in1k_l2_linear`.
- Ordered labels: cotton, denim, leather, linen, nylon, polyester, silk, viscose, wool.
- Pretrained FCMAE/ImageNet transfer learning, fresh weighted nine-output head, final-stage partial fine-tuning; selected epoch 4/5. See `docs/NINE_CATEGORY_TRAINING.md` for the reproducible protocol.
- **Provisional validation**: 667/740 correct, accuracy 90.14%, macro-F1 0.8690. Same-split frozen baseline 71.89%, macro-F1 0.6720. These are not independent/final-test accuracy or a directly comparable improvement over the different seven-class dataset.
- Checkpoint SHA-256: `7a32c9ebb487d314ea148d51d8042577de0ac89bea2579055d01a55428243aa9`.
- Training report SHA-256: `0a29d03729b1bcef38088f54e56c6156c80daf2732bbd2f5738798b9fa72a23d`.
- Dataset manifest SHA-256: `a4341ee939350b8b0b161b3d3bbf276137a1aa66965b42178d30ddfa13e9ec68`.
- Checkpoint terms: CC-BY-NC-4.0, non-commercial research; retain source attribution/restrictions. No confidence-calibration temperature is activated.

### Previous models remain seven-output

Ordered labels: cotton, cotton_mixed, denim, polyester, silk, viscose, wool. Never relabel these seven logits as nine.

| Choice | Run | Checkpoint SHA-256 |
|---|---|---|
| ConvNeXt-Tiny | `runtime/training/convnext_tiny_finetune_01` | `aa0cd052667b718e5854039a0098e71d0292cfbcc31efcc4986227f750dc766e` |
| EfficientNet-B0 | `runtime/training/efficientnet_b0_finetune_01` | `8550a8b49bc911e83d55179cb0e67a239dd2e01dc308b6a1d8f779c15bf6b73a` |

Their independent release JSONs remain in `config/`. The old seven-category V2 metadata is preserved in `config/fabric_classifier_release.convnext_v2_seven_rollback.json`; restoring it requires deliberately restoring the matching selection policy, not just exchanging release JSON files. Training reports remain immutable; their historical activation flags are not current app status.

## Datasets — preserve paths and reserved splits

- Original downloaded Fabric Classification dataset and earlier seven-category experiments remain available under `runtime/datasets/` and documented in the README/experiment guides.
- Nine-category raw collection: `data/research_collection/images`.
- Prepared collection: `runtime/datasets/fabric_nine_categories_01`.
- 4,980 input images; 4,912 retained, with 68 exact duplicate-copy exclusions in the manifest only. Original images were not deleted.
- Fixed inferred-group splits: 3,432 train / 740 validation / **740 reserved test**; class imbalance handled using training-only inverse-frequency loss weights.
- Leather/linen/nylon folders remain under `images/unsupported_fabrics` for manifest-path stability, despite those categories now being model-supported. **Do not move them without a controlled manifest/provenance migration.**
- Filename/content families and similarity checks are not verified physical-specimen/source independence. Cross-split similarity candidates remain unresolved and leather/linen have very few validation families. Validation metrics remain provisional.
- Both the new reserved test partition and original provider test set remain reserved. Do not tune on test results or casually rerun final evaluation.
- Legacy seven-category research evaluators reject nine-category releases; do not bypass the guard or interpret nine outputs using seven labels. Independent nine-category evaluation requires an approved compatible protocol/manifest.

## Files and interfaces to resume from

| Area | Key paths |
|---|---|
| Server / UI | `backend/app.py`, `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` |
| Classifier | `backend/classifier.py`, `backend/convnextv2_model.py`, `config/fabric_classifier_release.json` |
| Properties / design engine | `backend/engine.py`, `backend/design_rules.py`, `backend/comparison.py` |
| Shared Component 2 bridge | `backend/shared_contract.py` |
| Saved catalogue | `backend/specimens.py`, `runtime/fabric_properties.sqlite3` if present |
| Typical demonstration variants | `backend/typical_profiles.py`, `docs/TYPICAL_PROFILES.md` |
| Rule configuration | `config/design_rules.json` (review pending, version `prototype-0.1`) |
| Catalogue and review guide | `docs/CATALOGUE_AND_DESIGN_RULES.md` |
| Training evidence | `docs/NINE_CATEGORY_PREPARATION.md`, `docs/NINE_CATEGORY_TRAINING.md` |

API includes health/options; fabric analysis; record checking; specimen save/list/get; catalogue import/export; typical-profile list; read-only rule inspection; design recommendation and property-ablation comparison. Server binds loopback port **8002**. Components 1/2 retain their own ports and code.

## Remaining major work — recommended order

1. **Real documented fabric-property evidence:** collect sample-level supplier specifications, measurements or documented expert assessments, with identity/protocol/reference. Keep demo samples visibly identified; do not use presets as measurements. Import genuine records into the catalogue.
2. **Expert rule review:** have a fashion designer/pattern maker review silhouettes, ease ranges, fit assumptions, property compatibility and exclusions. Record actual reviewer/reference; expand rules only with justification, update version and rerun regression tests. Do not invent approval.
3. **Independent classifier validation/calibration:** collect genuinely independent real phone photos, supported and unknown/non-fabric inputs. Resolve grouping/leakage concerns, define nine-class evaluation schemas, use separate calibration data, freeze choices and then perform reserved final-test evaluation once. Report per-class metrics/confusions and unknown-input behaviour; do not call raw softmax a calibrated probability.
4. **Design-suitability evaluation:** prepare blinded scenarios and independent relevance/suitability ratings, compare property-aware versus baseline ranking; report Sri Lankan/global subsets separately where actual verified provenance supports them. Rule-score changes alone are not suitability improvement.
5. **Unimplemented original attributes:** verified labels and trained/evaluated heads for batik/handloom, print, weave and texture, or explicitly revise the project scope. Manual dropdowns do not satisfy this prediction scope. Real-image categories do not supply these labels automatically.
6. **Finalize reproducible reporting/integration:** update README, evidence tables and acceptance status. Coordinate shared interfaces with Component 4 after its scope is supplied; do not assume its specification or modify Components 1/2.

## Safe resume commands

From this component's PowerShell directory:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts\check_design_rules.py
.\.venv\Scripts\python.exe backend\app.py
```

Open `http://localhost:8002` and press Ctrl+F5. If port 8002 is already serving Component 3, reuse it; do not launch a duplicate server or terminate unrelated processes. Activation is optional when invoking the explicit venv Python. Dependencies are recorded in `requirements*.txt`; preserve the working venv/CUDA environment and do not reinstall/retrain merely to resume.

Model-switching and API smoke output directories already exist; use new output directories for repeat smoke runs. Do not overwrite immutable reports/checkpoints or regenerate dataset splits. No need to restart training at resume.

Suggested future prompt: **“Resume Component 3 from CHECKPOINT.md. Inspect the saved state first, then continue with [chosen remaining task]. Preserve trained artifacts, original images and reserved test partitions.”**

Component 4 can now be discussed separately. Saving this checkpoint does not implement or infer Component 4's requirements, and does not stop a running local server.

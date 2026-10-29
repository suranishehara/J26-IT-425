# RP_Planner_Test_Repo
# Component 2 — Fabric-Aware Garment Design Recommendation

**New laptop setup:** [Installation.md](Installation.md), including the required shared Component 2 source files.

An end-to-end university research prototype: upload a fabric photograph, review its likely category, choose documented or explicit demonstration properties, enter measurements/style, and receive explained garment-design ideas.

**Last consolidated: 19 October 2026, Asia/Colombo. Implementation is paused before Component 4.** Read [CHECKPOINT.md](CHECKPOINT.md) before resuming. This README describes the current state; the linked experiment documents preserve historical milestones.

## Contents

1. [Purpose and current completion](#1-purpose-and-current-completion)
2. [End-to-end architecture](#2-end-to-end-architecture)
3. [Datasets and preparation](#3-datasets-and-preparation)
4. [Models, training and results](#4-models-training-and-results)
5. [Four-panel application workflow](#5-four-panel-application-workflow)
6. [Design-ranking and ease rules](#6-design-ranking-and-ease-rules)
7. [Connection to Component 2](#7-connection-to-component-2)
8. [Environment and running the app](#8-environment-and-running-the-app)
9. [Complete testing guide](#9-complete-testing-guide)
10. [Reproducing experiments](#10-reproducing-experiments)
11. [API reference](#11-api-reference)
12. [Project structure and saved artifacts](#12-project-structure-and-saved-artifacts)
13. [Evaluation, privacy and limitations](#13-evaluation-privacy-and-limitations)
14. [Next major steps and resuming](#14-next-major-steps-and-resuming)
15. [Detailed documentation](#15-detailed-documentation)

## 1. Purpose and current completion

The question answered by this component is:

> Given a particular fabric, what garment could we design or make from it, considering its recorded properties, the user's measurements and their style preferences?

| Component | Responsibility |
|---|---|
| Component 1 | Existing-garment analysis and outfit recommendation; shared taxonomy/measurement validation  |
| Component 2 | Fabric-category suggestions and measurement/property-aware garment-design ideas |
| Component 3 | Virtual try-on and draping work, maintained separately |



Component 2 does not generate a try-on image, sewing pattern or fit-certified finished garment. It does not measure the body from the fabric photograph. Category/material, production technique, pattern, weave, surface texture and physical drapability are distinct concepts.

**Current result: a working end-to-end prototype, not fully completed research validation.** The previously discussed approximately 75% completion estimate was a planning judgement, not a measured acceptance metric.

| Area | Implemented | Still needed |
|---|---|---|
| Fabric photographs | Local upload validation, preview, category inference, model switching | Independent phone-photo/unknown-input evaluation |
| Category models | Nine-category ConvNeXt V2 default; older seven-category comparisons | Independent calibration and frozen final-test evaluation |
| Attributes | Manual labels, shared vocabulary, provenance and validation | Verified data and trained/evaluated batik, handloom, print/pattern, weave and texture heads |
| Physical properties | Documented records, saved-sample autofill and reuse | Genuine sample-level evidence and approved measurement protocols |
| Demonstration autofill | Three explicitly selected Typical profiles | More sourced variants only if deliberately added; no universal class defaults |
| Catalogue | Immutable records, JSON import/export, duplicate-import handling | Populate with genuinely documented samples |
| Measurements | Component 2 validation and unit conversion reused | Broader automatic profile-sharing would be separate work |
| Design ideas | Four silhouettes, configurable rules, explanations and positive ease | Pattern-maker review and independent suitability evaluation |
| Software verification | **123 automated tests passing** at the pause checkpoint | Continue regression and real-user checks when development resumes |
| Documentation | README, detailed protocols, source guides and checkpoint | Final research reporting after actual evaluation |

No retraining or new implementation is needed just to reopen the app. Components 1 and 2 were not modified by this Component 3 checkpoint.

## 2. End-to-end architecture

~~~text
Permitted fabric photograph
    → image validation / EXIF correction / preview
    → selected local category classifier
    → top-three relative scores
    → user applies or corrects the category

Fabric-record route (choose one)
    ├─ Manual known labels and documented properties
    ├─ Saved specimen → canonical server-retrieved record
    └─ Explicit Typical profile → illustrative variant defaults
             ↓
Validated record + review confirmation
             +
Body measurements / units / fit / style / preferred silhouette
             ↓
Component 2 measurement validator + validated design-rule configuration
             ↓
Ranked provisional designs + reasons / conflicts + ease calculations
             ↓
Optional comparison with a style/measurement-only baseline
~~~

Technology: Python loopback HTTP server, plain HTML/CSS/JavaScript, local CPU inference, SQLite specimen storage, PyTorch/Torchvision/timm for models and CUDA for recorded training experiments. The app does not send photographs to a cloud model or image-generation service.

## 3. Datasets and preparation

### 3.1 Downloaded TextileClass-7 v2

The first dataset was [TextileClass-7 v2 on Mendeley Data](https://data.mendeley.com/datasets/m2h53zr39s/2), downloaded as `Fabric Classification.v2.zip`, approximately 57.9 MB.

~~~text
runtime/datasets/raw/Fabric Classification.v2/
├── train/
├── valid/
├── test/
├── README.dataset.txt
└── README.roboflow.txt
~~~

The included README declares CC BY 4.0, provider preprocessing to stretched 224×224 images and augmented training exports. Preserve attribution/licence information. Public accessibility alone does not establish permission for unrelated collected photos.

Preserved seven-category output order:

~~~text
cotton, cotton_mixed, denim, polyester, silk, viscose, wool
~~~

| Split | Raw images | Accepted conservative pilot |
|---|---:|---:|
| Train | 5,499 | 5,043 |
| Validation | 392 | 356 |
| Reserved test | 393 | 393 |
| Total | 6,284 | 5,792 |

Preparation:

1. Audit readability, hashes, original-filename families, cross-label conflicts and perceptual similarity candidates.
2. Preserve source files and original test membership.
3. Exclude uncertain train/validation groups through the derived manifest, not by deleting/relabeling originals.
4. Record 492 exclusions: 456 mixed-label/filename-collision cases, 15 possible test relations and 21 possible validation relations. Not every conservative exclusion is a proven duplicate.
5. Freeze the accepted manifest at `runtime/datasets/textileclass_pilot_02/manifest.jsonl`.

Manifest SHA-256:

~~~text
3c888298dd97a636cb336c878e064700fa39f0e441fd02e3a5a8b8a4aa9ebbca
~~~

`textileclass_pilot_01` was an unused attempt rejected for a Windows newline/checksum mismatch. Do not use it for training. Physical specimen independence remains unverified even for pilot_02.

Test files were decoded/hashed for integrity/overlap screening, but no test model features, predictions or accuracy were computed. Details: [TEXTILECLASS_BASELINE.md](docs/TEXTILECLASS_BASELINE.md).

### 3.2 User-supplied nine-category real-photo collection

The researcher supplied category-organized real photos and confirmed labels/permission. These remain researcher assertions, not independently certified composition or ownership checks by the code.

~~~text
data/research_collection/images/
├── supported/
│   ├── cotton/
│   ├── cotton_mixed/          # Excluded from new nine-category scope
│   ├── denim/
│   ├── polyester/
│   ├── silk/
│   ├── viscose/
│   └── wool/
└── unsupported_fabrics/
    ├── Leather/              # Historical path; now model-supported
    ├── linen/
    ├── Nylon/                # Historical path; now model-supported
    └── other_documented_fabrics/
~~~

**Do not move leather/linen/nylon just because they are supported now.** Frozen manifests reference the current paths. Support is determined by trained labels, not a parent folder name. Future reorganization requires a controlled manifest/provenance migration.

Prepared version: `runtime/datasets/fabric_nine_categories_01`.

- 4,980 input images; 4,912 retained.
- 68 exact duplicate-copy exclusions in the manifest only; originals were not deleted.
- 1,228 inferred filename/content groups, not certified independent specimens.
- Seed-42 fixed group splits, approximately 70/15/15 by group count.
- Related filename families and exact-content links remain in one split.
- Training-only inverse-frequency loss weights; no validation/test rebalancing.
- No `cotton_mixed` output in this model; old checkpoints retain it.

| Category | Train | Validation | Reserved test |
|---|---:|---:|---:|
| Cotton | 1,624 | 348 | 348 |
| Denim | 452 | 96 | 96 |
| Leather | 44 | 8 | 8 |
| Linen | 52 | 12 | 12 |
| Nylon | 156 | 36 | 36 |
| Polyester | 628 | 132 | 132 |
| Silk | 136 | 32 | 32 |
| Viscose | 100 | 24 | 24 |
| Wool | 240 | 52 | 52 |
| **Total** | **3,432** | **740** | **740** |

~~~text
class_weight[c] = N_train / (9 × n_train[c])
~~~

Prepared artifacts: `manifest.jsonl`, `preparation.json`, `inventory.json`, `excluded.json` and `similarity_triage.json`.

Manifest SHA-256:

~~~text
a4341ee939350b8b0b161b3d3bbf276137a1aa66965b42178d30ddfa13e9ec68
~~~

Similarity triage found 18,021 perceptual candidates, including 8,850 retained cross-split candidate pairs. Repeated/plain textures can cause false positives: these are not proven duplicate counts. Source/specimen review remains unresolved. Leather validation has two inferred families and linen three; their results are especially unstable. The split/metrics are **provisional**, not an independently verified benchmark.

Details: [preparation](docs/NINE_CATEGORY_PREPARATION.md), [training](docs/NINE_CATEGORY_TRAINING.md), [added-class audit](docs/ADDED_CATEGORIES.md).

### 3.3 Other evidence types

| Data | Actual status |
|---|---|
| Batik/handloom, pattern, weave and texture labels | No completed verified collection or real-data trained heads; schema/training/evaluation tooling exists |
| Independent calibration/assessment/unknown-photo data | Still required; existing category photos are not automatically independent evaluation data |
| Physical-property catalogue | Implemented storage; real records require sample-level evidence. Demo entries remain demos |
| Typical profiles | Three sourced demonstration variants, not sample measurements or training/evaluation ground truth |
| Artificial fixtures | Software mechanics/regression only, not research performance evidence |

`data/manifest.example.json` and collection-schema examples are templates, not collected evidence. Downloaded category labels do not supply GSM, drape, stretch or production-method truth. Unknown materials for the active classifier must be outside the **current nine-category inventory**; linen is no longer an unknown category for V2.

## 4. Models, training and results

### 4.1 Model choices

| Model | Pretrained source | Feature size | Current app role |
|---|---|---:|---|
| EfficientNet-B0 | Torchvision ImageNet1K V1 | 1,280 | Selectable seven-category comparison |
| ConvNeXt-Tiny | Torchvision ImageNet1K V1 | 768 | Selectable seven-category comparison |
| ConvNeXt V2-Tiny | timm FCMAE, ImageNet-22K/1K fine-tuned source | 768 | **Default nine-category model** |

The networks use L2-normalized pooled features and learned linear heads. Final selected app checkpoints were partially fine-tuned; this is not zero-shot classification. CLIP, DINOv2 and SegFormer are not the implemented Component 3 category classifiers.

Preprocessing:

- EfficientNet-B0: official resize-256 / centre-224 and ImageNet normalization.
- ConvNeXt-Tiny: official resize-236 / centre-224 and ImageNet normalization.
- ConvNeXt V2: resolved timm 224-pixel bicubic transform, crop fraction 0.875, ImageNet normalization. Inference checks against recorded preprocessing; the alternate provider 288-pixel test setting is not used.

V2 source: [timm model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k), revision `b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817`. Provider weights SHA-256: `6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9`.

V2 weights are **CC-BY-NC-4.0**, used for the stated non-commercial university project. Retain attribution/restrictions. Model and dataset licences are separate obligations.

### 4.2 Training chronology

1. Train an EfficientNet frozen-feature linear-head baseline on the accepted seven-category manifest.
2. Review validation mistakes; partially fine-tune the final two feature modules/head, protecting earlier stages and normalization statistics.
3. Run ConvNeXt-Tiny frozen-head and partial-fine-tuning experiments.
4. Train seven-category ConvNeXt V2 with its broader pretraining: frozen head, then final-stage fine-tuning.
5. Prepare the separate nine-category collection; initialize a **fresh nine-output head** and train/fine-tune a new V2 run. Old checkpoints are not relabelled as nine classes.
6. Explicitly integrate that nine-category run as default while retaining old seven-output comparison mappings.

Common principles: seed 42, training-count-weighted cross entropy, AdamW, validation macro-F1 selection, earlier epoch on ties, immutable output directories, saved checkpoint/reload checks, no test inference during selection.

### 4.3 Recorded validation results

All values are **development validation**, not independent/final-test results.

| Experiment | Validation N | Accuracy | Macro-F1 | Run under `runtime/training/` |
|---|---:|---:|---:|---|
| EfficientNet frozen head, cached extraction | 356 | 61.52% | 0.6156 | `efficientnet_b0_pilot_01` |
| EfficientNet partial fine-tuning | 356 | 68.54% | 0.6881 | `efficientnet_b0_finetune_01` |
| ConvNeXt-Tiny frozen head | 356 | 59.27% | 0.5998 | `convnext_tiny_pilot_01` |
| ConvNeXt-Tiny partial fine-tuning | 356 | 69.38% | 0.6993 | `convnext_tiny_finetune_01` |
| V2 seven-category frozen head | 356 | 63.20% | 0.6384 | Frozen artifact inside `convnextv2_tiny_pilot_01` |
| V2 seven-category partial fine-tuning | 356 | 72.47% | 0.7313 | `convnextv2_tiny_pilot_01` |
| V2 nine-category frozen head | 740 | 71.89% | 0.6720 | Frozen artifact inside `convnextv2_nine_categories_01` |
| **V2 nine-category partial fine-tuning** | **740** | **90.14%** | **0.8690** | **`convnextv2_nine_categories_01`** |

The original EfficientNet cached extraction used mixed precision. Common full-network FP32 re-evaluation of the same checkpoint gives 61.24% / 0.6129, a one-image numerical difference. Controlled fine-tuning comparisons use this recomputed baseline.

**90.14% on nine categories must not be compared directly with 72.47% on seven categories as an improvement claim:** datasets/tasks differ. Different pretraining/budgets also prevent an isolated architecture-superiority conclusion. Repeated validation selection can overfit development data.

### 4.4 Default nine-category protocol

- 3,432 training / 740 validation; 740 test reserved.
- Frozen normalized 768-D FP32 features.
- Head: 100 full-feature-matrix epochs, weighted cross entropy, AdamW LR 0.02 / weight decay 0.01.
- Partial fine-tuning: five epochs, batch 16, AMP float16 training, FP32 evaluation; final stage/head-normalization LR 0.0001, head LR 0.001, gradient clip 1.0.
- Earlier stages fixed, stochastic depth in eval mode, no extra augmentation/oversampling.
- Frozen epoch zero remains eligible; protected-state equality and saved full-state FP32 reload predictions checked.
- Selected epoch **4/5**, 667/740 correct; frozen head selected epoch 100/100.
- Recorded RTX 4050 Laptop GPU: 431.62 seconds after pretrained loading; peak PyTorch allocated GPU memory 631,841,792 bytes, not total VRAM.

| Category | Validation support | F1 |
|---|---:|---:|
| Cotton | 348 | 0.9353 |
| Denim | 96 | 0.9519 |
| Leather | 8 | 1.0000 |
| Linen | 12 | 0.8462 |
| Nylon | 36 | 0.9014 |
| Polyester | 132 | 0.8622 |
| Silk | 32 | 0.8125 |
| Viscose | 24 | 0.6818 |
| Wool | 52 | 0.8298 |

Leather's eight-image/two-family score does not mean leather is solved. Cached validation review records 73 mistakes; exploratory family-majority metrics are separate and do not establish specimen independence. Full history/confusion matrices are in the saved report.

### 4.5 Active release and exact mappings

UI selections:

1. **Experimental ConvNeXt V2-Tiny** — default, `convnext_v2`.
2. **Experimental ConvNeXt-Tiny** — `convnext_v1`.
3. **Experimental EfficientNet-B0** — `efficientnet_b0`.

V2 output order:

~~~text
cotton, denim, leather, linen, nylon, polyester, silk, viscose, wool
~~~

Older V1/EfficientNet output order:

~~~text
cotton, cotton_mixed, denim, polyester, silk, viscose, wool
~~~

Category controls follow the selected model. Reviewed records accept the union for old-sample compatibility; this does not add outputs to old models.

| Choice | Release configuration |
|---|---|
| V2 nine-class | `config/fabric_classifier_release.json` |
| V1 seven-class | `config/fabric_classifier_release.convnext_v1.json` |
| EfficientNet seven-class | `config/fabric_classifier_release.efficientnet_rollback.json` |
| Historical V2 seven-class metadata | `config/fabric_classifier_release.convnext_v2_seven_rollback.json` |

Default V2 identity:

~~~text
Run:        runtime/training/convnextv2_nine_categories_01
Checkpoint: 7a32c9ebb487d314ea148d51d8042577de0ac89bea2579055d01a55428243aa9
Report:     0a29d03729b1bcef38088f54e56c6156c80daf2732bbd2f5738798b9fa72a23d
Manifest:   a4341ee939350b8b0b161b3d3bbf276137a1aa66965b42178d30ddfa13e9ec68
~~~

The loader validates architecture, label order, hashes and V2 preprocessing, constructs the correct head and decodes against that report's labels. Cache identity follows the selected checkpoint/vocabulary. Missing/corrupt models fail to manual review, not silent model substitution. CPU FP32 inference loads local safetensors without downloads.

Training reports remain immutable: historical `activated_in_app: false` fields describe training-time state, not current activation. The old V2 rollback metadata also requires restoring its matching selection policy; merely exchanging JSON files is insufficient.

## 5. Four-panel application workflow

### 5.1 Panel 01 — Fabric photograph

1. Choose a permitted close-up and confirm permission.
2. Select a classifier; default is nine-category V2.
3. Click **Analyse photograph**.
4. Review the preview and top-three suggestions.
5. Click **Use top category** when enabled, or manually choose/correct the category.

Validation: PNG/JPEG/WebP, decoded size at most 6,000,000 bytes, minimum 64 pixels per side, at most 20 megapixels, EXIF correction/RGB conversion. A separate preview is resized to fit 1000 pixels. Uploaded photos are not stored by the application.

Capture checks flag low resolution, extreme-brightness areas or low tonal variation. Quick apply requires top score ≥0.60, top-two margin ≥0.15 and no capture warnings. These are provisional UI heuristics, **not calibrated confidence or validated unknown detection**. Unfamiliar fabrics/non-fabric objects can still receive confident wrong labels.

Category application does not populate composition, technique or physical properties. Legacy API `review_only` mode remains, but is no longer in the UI dropdown.

### 5.2 Panel 02 — Attributes and properties

Choose one route:

| Route | Source | Behaviour |
|---|---|---|
| Manual | Known labels/documented properties | Unknown remains valid; reviewed record used for this request |
| Saved specimen | Explicit selected physical sample | Read-only autofill; server retrieves canonical saved record |
| Typical profile | Explicit construction variant | Separate illustrative defaults with compact Typical profile label |

Label fields: reviewed category; known material; production technique; visible pattern; visible texture. Material includes unknown/cotton/linen/nylon/leather/silk/polyester/viscose/wool/blend/other. Technique includes unknown/batik/handloom/other as **manual records**, not trained outputs.

| Property | Entry | Effect |
|---|---|---|
| Drapability | unknown / fluid / moderate / structured | Ranking input |
| Weight band | unknown / light / medium / heavy | Ranking input |
| GSM | Optional >0 to 2000 g/m² | Context only |
| Stretch | Optional 0–500% | Context only; no negative-ease calculation |

Known manual properties require a source (`user_measured`, `supplier_documented`, `expert_reviewed`) and a short nonempty reference. A shared manual source applies to that form's entered properties; saved/imported records retain individual sources. Range/source-presence validation is not certification of truth.

**Check fabric record** shows normalized details, sources, ranking inputs and save readiness. It does not save or automatically confirm review. Optional appearance/numerical sections collapse to reduce typing. Changes invalidate summaries, review and recommendations.

#### Saved samples and catalogue

- Save name/sample reference, reviewer ID and at least one documented property after review confirmation.
- Local SQLite: `runtime/fabric_properties.sqlite3`, append-only through the app. Corrections create new records.
- Canonical fabric JSON SHA-256 detects changed stored fabric values.
- Selecting a saved sample fills read-only fields and displays per-property provenance. Manual mode restores the previous draft; stale responses cannot overwrite a newer selection.
- Selection is the user's choice: no automatic verification that a specimen matches the photograph.
- Export contains name/reviewer/fabric/hash, not photos or body measurements.
- Import requires reviewed sources, schema version 1, 1–200 records and a body ≤500,000 bytes including the envelope.
- Whole batch validates before an atomic transaction. Matching name/reviewer/fabric hash is skipped; existing records are not replaced.

See [CATALOGUE_AND_DESIGN_RULES.md](docs/CATALOGUE_AND_DESIGN_RULES.md) for JSON/collection instructions. Fictional `DEMO ONLY` records are not research evidence.

#### Typical profiles

Choose a construction variant, then **Use typical fabric profile**. The variant replaces the form's category for an explicit what-if scenario; the image does not identify that variant automatically.

| Profile | Example GSM | Drape mapping | Demonstration weight band | Stretch default |
|---|---:|---|---|---:|
| Cotton Oxford shirting | 194 | structured | medium | 0% |
| Polyester chiffon | 70 | fluid | light | 0% |
| Rigid cotton denim | 432 | structured | heavy | 0% |

Supplier example sources: [Oxford item 447586](https://www.moodfabrics.com/products/cotton-oxford-shirting-denim-blue-deadstock-447586), [chiffon MP5000-196](https://www.moodfabrics.com/products/netta-black-polyester-high-multi-chiffon-mp5000-196), [denim item 331499](https://www.moodfabrics.com/products/structured-cotton-denim-dark-blue-deadstock-331499). Details/source interpretations: [TYPICAL_PROFILES.md](docs/TYPICAL_PROFILES.md).

These are specific examples, not class averages or measurements of uploaded images. Qualitative drape is normalized into the project vocabulary; preset bands use light <150, medium 150–300 and heavy >300 GSM as a project convention, not a textile standard. Supplier no-stretch becomes illustrative 0%, not a controlled-load measurement. Uploaded-sample composition/technique/appearance remain unknown.

Only three variants exist. Server-side profile IDs are allowlisted and values reconstructed, ignoring overrides. Presets cannot be saved/imported as documented specimen measurements or silently combined with saved specimens. They retain `typical_profile` source/output metadata. **Return to manual properties** restores the draft; changing/reanalysing a photo or model clears typical mode.

### 5.3 Panel 03 — Measurements and style

- Optional height, chest, waist, hips, shoulder width and inseam.
- Centimetres or inches, normalized through Component 2.
- Fit: no preference, regular, relaxed or fitted. Fitted selection does not implement fitted patterns/negative ease; that limitation is reported.
- Style: any, casual, office, formal, traditional.
- Preferred silhouette: any or a configured design.
- Review confirmation before recommendation/saved-specimen use.

Only supplied dimensions are used. Missing measurements are reported rather than estimated from the fabric image.

### 5.4 Panel 04 — Provisional design ideas

Cards show rank, rule points, matches, conflicts, missing measurements, illustrative circumferences and rule version/hash/review status. Optional comparison removes properties while keeping measurements, fit, style, silhouette and rule snapshot identical.

Typical-profile results retain a compact label. Rule points are not neural probabilities, evaluated suitability accuracy or evidence of garment fit.

## 6. Design-ranking and ease rules

Configuration: [config/design_rules.json](config/design_rules.json), validated by `backend/design_rules.py`. Version `prototype-0.1`; **review status pending**. No designer/pattern-maker approval is claimed.

| Design | Matching styles | Preferred drapability | Preferred weight |
|---|---|---|---|
| Relaxed tunic / kurta | casual, traditional | moderate, fluid | light, medium |
| Structured shirt / blouse | office, casual | structured, moderate | light, medium |
| A-line dress | casual, formal, traditional | moderate, structured | light, medium |
| Flowing gathered skirt | casual, formal, traditional | fluid | light |

~~~text
Matching selected style:                 +2
Matching preferred silhouette:           +3
Known drapability:                       +3 match / −3 conflict
Known weight band:                       +2 match / −2 conflict
Unknown property:                         0
~~~

Ties use stable design IDs. Negative points are rule conflicts, not errors or negative probabilities. Category/material/technique/pattern/texture/GSM/stretch do not add ranking points. Measurements affect illustrative dimensions, not ranking.

Positive circumference-ease ranges (cm):

| Design | Chest | Waist | Hips |
|---|---|---|---|
| Relaxed tunic / kurta | 10–14 | 10–16 | 10–14 |
| Structured shirt / blouse | 6–10 | 6–12 | Not calculated |
| A-line dress | 6–10 | 8–14 | 10–16 |
| Flowing gathered skirt | Not calculated | 2–4 | 12–20 |

~~~text
Draft finished circumference = supplied body circumference + illustrative ease
Relaxed fit adds 2 cm to each ease bound.
~~~

These are not cutting-pattern dimensions. Seam allowance, shrinkage, construction details, stretch compensation and negative ease are not included. Unknown dimensions stay missing.

Validator checks include unique design IDs, valid vocabularies, finite/nonnegative numbers and ordered ease ranges. Reviewed status needs a reviewer/reference, but field validation is not proof of actual review. Configuration editing is local-file work; web inspection is read-only. Baseline and property-aware runs use one shared snapshot.

## 7. Connection to Component 1

`backend/shared_contract.py` loads these sibling modules read-only:

~~~text
../component-1-fashion-recommendation/backend/user_profile.py
../component-1-fashion-recommendation/backend/taxonomy.py
~~~

Reused: measurement fields/validation, unit/fit handling, pattern/texture vocabulary and taxonomy version. Health reports shared-module identity. Keep the sibling folder available; its web server need not run.

Not automatically reused: Component 2's classifier, skin-tone/body-shape image analysis, recommender, saved profile or catalogue. No Component 1 implementation was altered. Garment `drape_style` describes arrangement; material `drapability` describes bending/hanging, so one does not substitute for the other.

## 8. Environment and running the app

### Existing environment

PowerShell accepts forward-slash Windows paths as used below:

~~~powershell
Set-Location 'C:/Users/Ravindu/Videos/Component 3 – Virtual Try-On and Draping/component-3-fabric-design'
./.venv/Scripts/Activate.ps1
./.venv/Scripts/python.exe backend/app.py
~~~

Open [http://localhost:8002](http://localhost:8002). Keep the terminal running; Ctrl+C stops it. After frontend changes press Ctrl+F5. Activation is optional when using the explicit interpreter:

~~~powershell
./.venv/Scripts/python.exe backend/app.py
~~~

If activation is blocked, use explicit venv Python; a quoted activation path alone prints a string. Do not change machine-wide policy merely to run the app.

### Dependencies

Recorded environment: Windows, Python 3.12, Torch `2.14.1+cu126`, Torchvision `0.29.1+cu126`, timm `1.0.30`, safetensors, NumPy, Pillow. Training used an RTX 4050 Laptop GPU; application inference is CPU.

| File | Purpose |
|---|---|
| `requirements.txt` | Minimal Pillow-based manual/image-review operation |
| `requirements-lock.txt` | Initial minimal environment snapshot |
| `requirements-ml.txt` | ML declarations |
| `requirements-ml-lock.txt` | Recorded full ML environment |

Preserve the working `.venv`; do not reinstall just to resume. For a genuinely new environment only:

~~~powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
# Only when recreating the recorded CUDA ML environment:
./.venv/Scripts/python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
./.venv/Scripts/python.exe -m pip install -r requirements-ml-lock.txt
./.venv/Scripts/python.exe -m pip check
~~~

Check wheel/driver compatibility on a different machine; historical locks do not guarantee portability. Pillow alone does not enable classifiers. Local reports/weights and sibling Component 2 modules are required. Do not install into paused components' environments.

### Troubleshooting

| Symptom | Check |
|---|---|
| WinError 10048 | Reuse an existing Component 3 server. Inspect `Get-NetTCPConnection -LocalPort 8002 -State Listen` and identify the process before stopping anything |
| Failed to fetch | Server running? Open served localhost URL, not the HTML file directly |
| Classifier unavailable | Inspect `/api/health`, dependencies, release/report/weight hashes. Do not bypass verification |
| Shared module missing | Restore the sibling Component 2 folder |
| Property validation error | Known values require source/reference; unknown is valid |
| Old UI | Ctrl+F5 and confirm the correct port |
| Existing output error | Choose a new research output; preserve previous evidence |

## 9. Complete testing guide

### Automated checks

~~~powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -q
./.venv/Scripts/python.exe scripts/check_design_rules.py
node --check frontend/app.js
~~~

**123 tests passed at the saved pause checkpoint.** Node executes real frontend-handler regression checks in a minimal DOM harness. Without Node that specific check skips; disclose skips honestly. Coverage includes releases, datasets/split guards, shared measurements, arithmetic, provenance, atomic catalogue validation, preset isolation, HTTP origin checks, autofill, draft restoration and stale responses. Fixtures are not real classification/suitability evidence.

Key smoke/review reports:

~~~text
runtime/evaluations/nine_category_checkpoint_smoke_01/report.json
runtime/evaluations/nine_category_model_selection_smoke_01/report.json
runtime/evaluations/nine_category_fabric_api_smoke_01/report.json
runtime/evaluations/nine_validation_review_01/report.json
~~~

### Manual acceptance walkthrough

Use permitted photos. The body measurements here are fictional software-test inputs, not inferred values.

1. Analyse with default V2; apply/correct the category. Expect three suggestions and no automatic physical properties.
2. Leave properties unknown; Check fabric record. Enter cm, regular fit, height 165, chest 90, waist 75, hips 100, shoulder width 38, inseam 76, office style, any silhouette.
3. Confirm review and generate. Expect four ideas, structured shirt score 2, chest draft 96–100 cm and waist 81–87 cm. Without recorded properties these are preference-based.
4. Choose Cotton Oxford shirting and Use typical fabric profile. Expect structured/medium, 194 GSM, illustrative 0% stretch, Typical profile label and disabled specimen save. This does not prove a match to the photo.
5. Confirm review, enable baseline comparison and generate. Expect shirt 7 versus baseline 2; A-line 5 versus 0; tunic −1 versus 0; skirt −5 versus 0. Top design stays shirt while overall ranking changes.
6. Return to manual; draft restores. To test a real specimen, enter genuine documented properties/source/reference, sample name/reviewer, review and save. Fictional save tests must be named DEMO ONLY and excluded from research evidence.
7. Select saved sample: read-only values and per-property sources appear. Switch to manual/back; check restoration. Refresh: saved records persist, unsaved drafts/measurements do not.
8. Export/inspect/reimport catalogue: matching records skip; malformed batches must not partially save. Typical profiles must not import as documented specimens.
9. Verify rejection of missing photo permission, source/reference omissions, negative body measurements, missing review, and unknown-only specimen saves.

Passing these checks establishes functional behaviour, not real-world accuracy or validated tailoring suitability.

## 10. Reproducing experiments

Training is already complete for saved candidates. Do not retrain/regenerate splits merely to use the app. Existing outputs are protected; the illustrative REPRO paths below must be unused too.

### Seven-category lineage

~~~powershell
./.venv/Scripts/python.exe scripts/train_fabric_baseline.py --dataset runtime/datasets/textileclass_pilot_02 --output runtime/training/efficientnet_b0_pilot_REPRO
./.venv/Scripts/python.exe scripts/finetune_fabric.py --output runtime/training/efficientnet_b0_finetune_REPRO
./.venv/Scripts/python.exe scripts/train_convnext.py --output runtime/training/convnext_tiny_pilot_REPRO
./.venv/Scripts/python.exe scripts/finetune_convnext.py --output runtime/training/convnext_tiny_finetune_REPRO
./.venv/Scripts/python.exe scripts/train_convnextv2.py --output runtime/training/convnextv2_tiny_pilot_REPRO
~~~

These are separate examples. Fine-tuning defaults reference the existing accepted baselines; to construct a wholly new lineage, inspect `--help` and deliberately pass the matching new baseline path. Commands do not automatically repoint to the previous newly created output. Exact protocols are linked below.

### Nine-category training and non-test verification

~~~powershell
./.venv/Scripts/python.exe scripts/train_convnextv2.py --nine-categories --acknowledge-provisional-split --dataset runtime/datasets/fabric_nine_categories_01 --output runtime/training/convnextv2_nine_categories_REPRO
./.venv/Scripts/python.exe scripts/smoke_nine_checkpoint.py --output runtime/evaluations/nine_checkpoint_smoke_REPRO
./.venv/Scripts/python.exe scripts/review_nine_validation.py --output runtime/evaluations/nine_validation_review_REPRO
./.venv/Scripts/python.exe scripts/smoke_model_selection.py --output runtime/evaluations/model_selection_smoke_REPRO
./.venv/Scripts/python.exe scripts/smoke_fabric_api.py --output runtime/evaluations/fabric_api_smoke_REPRO
~~~

The provisional-split acknowledgement does not certify independence. Training does not activate a release. Deliberate integration must pin architecture/hashes/label order.

Smokes above default to the existing accepted model/app release; pass `--run` where supported to inspect a different run. Training-image smokes check loading/repeats; cached validation review reproduces development findings. Neither estimates independent accuracy. Development loaders refuse reserved test rows.

### Script inventory

| Purpose | Scripts in `scripts/` |
|---|---|
| Provider audit/preparation | `audit_textileclass.py`, `prepare_textileclass.py`, `review_audit.py` |
| Collection/new categories | `audit_added_categories.py`, `prepare_nine_categories.py`, `prepare_collection_manifest.py`, `audit_research_collection.py` |
| Category training | `train_fabric_baseline.py`, `finetune_fabric.py`, `train_convnext.py`, `finetune_convnext.py`, `train_convnextv2.py` |
| Development review | `review_validation_errors.py`, `compare_fabric_runs.py`, `review_nine_validation.py` |
| Checkpoint/app mechanics | `smoke_fabric_checkpoint.py`, `smoke_candidate.py`, `smoke_nine_checkpoint.py`, `smoke_fabric_api.py`, `smoke_model_selection.py` |
| Calibration/unknown tooling | `calibrate_fabric.py`, `evaluate_unknown_inputs.py`, `evaluate_research_release.py`, `smoke_independent_mechanics.py` |
| Future attribute tooling | `train_attribute_heads.py`, `evaluate_attribute_heads.py`, `smoke_attribute_mechanics.py` |
| Rule validation | `check_design_rules.py` |

**Compatibility boundary:** independent schemas/evaluators began with seven categories. `evaluate_research_release.py` and `evaluate_unknown_inputs.py` reject the nine-class release instead of misdecoding it. Approve/adapt a versioned nine-class protocol before using them with the current model. Review other schema/attribute-tool assumptions too. Do not bypass guards or reuse another checkpoint's calibration temperature. Tooling availability is not evidence of a completed real-data study.

## 11. API reference

Base: `http://localhost:8002`. JSON POSTs enforce origin checks when an Origin header is supplied, finite JSON and body-size limits. This is a local prototype, not authenticated multi-user hosting.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Default/available models, release identities and shared metadata |
| GET | `/api/options` | Vocabularies, measurement fields, configured designs |
| GET | `/api/typical-profiles` | Three demonstration variants with sources |
| GET | `/api/design-rules` | Validated read-only rule configuration/hash |
| POST | `/api/analyse-fabric` | Permission-gated data-URL image analysis |
| POST | `/api/fabric-record/check` | Draft or confirmed typical-record validation |
| GET | `/api/specimens` | Saved sample summaries |
| GET | `/api/specimens/{id}` | Canonical sample and fabric hash |
| POST | `/api/specimens` | Append reviewed documented sample |
| GET | `/api/catalogue/export` | Portable sample JSON |
| POST | `/api/catalogue/import` | Reviewed atomic batch import |
| POST | `/api/designs` | Design ranking/ease output |
| POST | `/api/designs/compare` | Controlled property-ablation comparison |

Image payload shape:

~~~json
{
  "permission": true,
  "mode": "experimental",
  "model_id": "convnext_v2",
  "image": "data:image/png;base64,<actual encoded image>"
}
~~~

Typical-profile mechanics example with fictional measurements:

~~~json
{
  "typical_profile_id": "cotton_oxford",
  "reviewed": true,
  "profile": {
    "unit": "cm",
    "measurements": {"chest": 90, "waist": 75, "hips": 100},
    "fit_preference": "regular"
  },
  "style": "office",
  "silhouette": "any"
}
~~~

Manual requests provide `fabric.reviewed: true`, labels and per-property `{value, source, reference}`. Saved requests provide `specimen_id` plus top-level `reviewed: true`; server retrieval defeats client property overrides. Typical requests provide `typical_profile_id` and confirmation; do not combine with a specimen.

Catalogue envelope:

~~~json
{
  "reviewed": true,
  "catalogue": {"schema_version": 1, "records": []}
}
~~~

The empty records array above illustrates shape only; actual imports need 1–200 documented records. See the catalogue guide for valid entries.

POST body limits: image analysis 8,100,000 bytes; catalogue import 500,000; other routes 50,000. Decoded-image/record-count limits apply separately. No web endpoint retrains models, changes releases or edits/approves rules.

## 12. Project structure and saved artifacts

~~~text
component-3-fabric-design/
├── README.md / CHECKPOINT.md
├── requirements*.txt
├── backend/
│   ├── app.py                         # Loopback server/routes
│   ├── images.py / photo_review.py     # Validation/review heuristics
│   ├── classifier.py                  # Release verification/inference
│   ├── fabric_model.py                # EfficientNet
│   ├── convnext_model.py              # ConvNeXt V1
│   ├── convnextv2_model.py             # ConvNeXt V2
│   ├── shared_contract.py             # Component 2 reuse
│   ├── engine.py / design_rules.py    # Records, rules, ranking/ease
│   ├── comparison.py                  # Property ablation
│   ├── specimens.py                   # SQLite/import/export
│   └── typical_profiles.py            # Separate illustration presets
├── frontend/index.html / app.js / styles.css
├── config/
│   ├── design_rules.json
│   ├── fabric_category_expansion.json # Frozen preparation input
│   └── fabric_classifier_release*.json
├── data/research_collection/images/   # Preserve original paths
├── data/manifest.example.json         # Template only
├── runtime/
│   ├── datasets/                      # Raw/derived immutable manifests
│   ├── models/                        # Cached pretrained sources
│   ├── training/                      # Saved runs/reports/weights
│   ├── evaluations/                   # Audits/smokes/development evidence
│   └── fabric_properties.sqlite3      # Created on catalogue use
├── scripts/
├── tests/                             # Python + JS handler harness
└── docs/
~~~

Runs preserve final `report.json`, full `model.safetensors`, histories/progress, feature caches and validation outputs; V2 preserves frozen artifacts too. A partial run folder/progress file alone is not completion evidence.

Reports, manifests and release choices are separate. Do not edit training reports to change activation, overwrite results, or regenerate splits to improve metrics. `fabric_category_expansion.json` is a hashed historical input, not a casual settings file. Back up raw photos, saved weights, SQLite records and source evidence separately; a checkpoint document is not a full backup.

## 13. Evaluation, privacy and limitations

### Classification/calibration

Accuracy is correct predictions divided by labelled evaluation examples; macro-F1 averages class F1. A photo's softmax value distributes relative score across known outputs, not verified correctness probability.

Older seven-category post-selection temperature diagnostics:

| Model | Exploratory fitted temperature | Assessment ECE before → after |
|---|---:|---|
| EfficientNet-B0 | 1.2306688 | 0.1504 → 0.1003 |
| ConvNeXt V1 | 2.2322385 | 0.2226 → 0.0966 |
| ConvNeXt V2 seven-class | 2.0372503 | 0.1861 → 0.0700 |

They fit 249 validation images and assess 107 after the entire validation set had already influenced model selection. They are not independent calibration evidence. No temperature is deployed or applicable to the new nine-class weights. Scaling does not change top-1 predictions or solve unknown detection. Full NLL/Brier/ECE data: [CALIBRATION_AND_UNKNOWN.md](docs/CALIBRATION_AND_UNKNOWN.md); its seven-class examples are historical.

Independent real photos, unknown/non-fabric assessment, a nine-class calibration protocol and final-test evaluation remain pending. The provider's 393 and new collection's 740 test images remain reserved for model evaluation. Integrity hashing/decoding is distinct from classifier inference.

### Design and Sri Lankan scope

The recommender is transparent rules, not a separately trained neural suitability model or a knowledge graph. Rule-point changes are not measured suitability improvements. No completed independent design-suitability/fairness or Sri Lankan/global subgroup results exist. Region/provenance requires documentation, not an appearance guess.

Batik/handloom can coexist with supported categories; they are not inherently unknown materials. Manual fields and artificial attribute-head smokes do not complete the multi-attribute plan. Neither predicted category nor Typical profile certifies physical properties.

### Privacy and preservation

- Web uploads, measurements and recommendation responses are not automatically persisted.
- Explicit specimen labels/properties/reviewer references persist locally. Export may contain private supplier references; inspect before sharing.
- Research data/reports intentionally persist and can include paths, source references and validation contact sheets, unlike transient web uploads.
- Local inference uses saved weights; setup/pretraining downloads are separate from photo processing.
- Origin/size checks are not production authentication or authorization.
- Original photos were not deleted for imbalance/duplicate handling. Preserve paths, weights, reports and catalogue when resuming.

## 14. Next major steps and resuming

Development remains paused. When explicitly resumed:

1. Collect genuine sample-level property/source evidence and approved protocols; populate the real catalogue without promoting demos/presets into measurements.
2. Obtain designer/pattern-maker review of silhouettes, ease, compatibility and exclusions. Record actual reviewer/reference, revise rule version/tests appropriately.
3. Resolve grouping/similarity/provenance concerns and prepare independent supported/unknown nine-class photos; separate calibration from assessment.
4. Freeze model, calibration and protocol, correctly adapt guarded evaluators, then conduct reserved final evaluation without tuning on outcomes.
5. Collect blinded independent design-suitability ratings; compare against the baseline and report documented Sri Lankan/global subsets separately.
6. Collect verified technique/pattern/weave/texture labels and train/evaluate separate heads, or explicitly narrow the approved scope.
7. Finalize reproducibility, reporting and acceptance criteria; coordinate Component 4 only after its scope is supplied.


## 15. Detailed documentation

| Document | Contents |
|---|---|
| [CHECKPOINT.md](CHECKPOINT.md) | Pause/resume summary and identities |
| [DATASET_PLAN.md](docs/DATASET_PLAN.md) | Required evidence/capture dimensions |
| [TEXTILECLASS_BASELINE.md](docs/TEXTILECLASS_BASELINE.md) | Provider audit and first baseline |
| [EFFICIENTNET_FINETUNING.md](docs/EFFICIENTNET_FINETUNING.md) | Controlled EfficientNet fine-tuning |
| [CONVNEXT_EXPERIMENT.md](docs/CONVNEXT_EXPERIMENT.md) | V1 experiments |
| [CONVNEXTV2_EXPERIMENT.md](docs/CONVNEXTV2_EXPERIMENT.md) | Seven-class V2 protocol/licence |
| [ADDED_CATEGORIES.md](docs/ADDED_CATEGORIES.md) | Historical new-class audit |
| [NINE_CATEGORY_PREPARATION.md](docs/NINE_CATEGORY_PREPARATION.md) | Current manifest/splits/caveats |
| [NINE_CATEGORY_TRAINING.md](docs/NINE_CATEGORY_TRAINING.md) | Default model protocol/results/integration |
| [EXPERIMENTAL_INFERENCE.md](docs/EXPERIMENTAL_INFERENCE.md) | Release/inference interpretation |
| [CALIBRATION_AND_UNKNOWN.md](docs/CALIBRATION_AND_UNKNOWN.md) | Historical calibration and missing evidence |
| [INDEPENDENT_RESEARCH_PIPELINE.md](docs/INDEPENDENT_RESEARCH_PIPELINE.md) | Original guarded seven-class research tools; nine-class adaptation needed |
| [CATALOGUE_AND_DESIGN_RULES.md](docs/CATALOGUE_AND_DESIGN_RULES.md) | Catalogue JSON and expert-review procedure |
| [TYPICAL_PROFILES.md](docs/TYPICAL_PROFILES.md) | Demonstration variants/sources/assumptions |
| [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | Original research scope |

Earlier activation statements/test counts in experiment guides are milestone history. Current configs, final run reports, this consolidated README and the pause checkpoint describe the present state.

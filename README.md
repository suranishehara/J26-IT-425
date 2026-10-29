# Component 1 — Culture Context Aware Recommendation Engine

**New laptop setup:** [Installation.md](Installation.md).

Component 2 helps a user choose garments and understand **why** they were recommended. It combines garment-image analysis, an optional personal profile, a reviewed catalogue, a lightweight knowledge graph and explainable ranking.

## Contents

- [1. Current capabilities](#1-current-capabilities)
- [2. Setup and running](#2-setup-and-running)
- [3. End-to-end process](#3-end-to-end-process)
- [4. Models and application modes](#4-models-and-application-modes)
- [5. Datasets and preparation](#5-datasets-and-preparation)
- [6. Training and recorded results](#6-training-and-recorded-results)
- [7. Body, measurements and skin appearance](#7-body-measurements-and-skin-appearance)
- [8. Catalogue, knowledge graph and recommendations](#8-catalogue-knowledge-graph-and-recommendations)
- [9. Grad-CAM explanations](#9-grad-cam-explanations)
- [10. Technology stack and folder structure](#10-technology-stack-and-folder-structure)
- [11. API](#11-api)
- [12. Tests and experiment commands](#12-tests-and-experiment-commands)
- [13. Privacy, permissions and reproducibility](#13-privacy-permissions-and-reproducibility)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Remaining research work](#15-remaining-research-work)
- [16. Detailed guides and primary sources](#16-detailed-guides-and-primary-sources)

## 1. Current capabilities

| Area | Implemented | Important limitation |
|---|---|---|
| Garment attributes | Garment type, pattern, drape and visible-texture suggestions | All suggestions require review; modes use different models |
| Garment-type adaptation | Trained IndoFashion four-class CLIP head | Blouse, dress, kurta and saree only |
| Pattern adaptation | Trained CLIP/DINOv2 heads; optional DINOv2 application mode | Four patterns; development validation only |
| Sri Lankan drape adaptation | Trained and optionally activated Nivi/Osariya head | Small researcher-labelled, two-class pilot |
| Visible texture | CLIP suggestions and a training pipeline | No complete locally trained texture head yet |
| Personal profile | Preferences, measurements, optional consented photo analysis/correction | No calibrated body-shape classifier or physical measurements from a photo |
| Skin appearance | Tentative nearest Monk colour reference | Uncalibrated, not fairness validated and not used in ranking |
| Size compatibility | Supplied body-size charts compared with entered measurements | Missing charts remain unknown; not a fit guarantee |
| Catalogue | Reviewed records/images saved locally | Demonstration prices/tags are illustrative |
| Recommendations | Combined global/traditional ranking and reasons | Explicit rules, not a trained neural/collaborative recommender |
| Cultural guidance | Optional versioned rule bonuses | Proposed hypotheses requiring independent cultural validation |
| Evaluation | Classification reports, software tests and one-reviewer recommendation pilot | Pattern/drape tests remain reserved; broader human evaluation remains |

**Working software is not the same as validated research.** Garment accuracy does not measure skin-tone accuracy, fit, drape generalization or recommendation usefulness.

## 2. Setup and running

### Existing environment — PowerShell

From this Component 2 folder:

```powershell
.\.venv\Scripts\Activate.ps1
python backend\app.py
```

Open **http://localhost:8001**. Keep the terminal open. Press **Ctrl+C** to stop the server.

If activation is blocked, run directly without activating:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Run only one server at a time. `(.venv)` in the prompt means activation is already complete. A quoted activation-file path by itself only prints text in PowerShell; it does not activate the environment.

### Fresh installation

The independent environment uses **Python 3.12**. Do not recreate an existing working environment just to start the app.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pip check
```

`setup.ps1` automates GPU dependency installation and verification, using the selected `python` executable. Ensure that executable is the intended Python version first.

Development hardware is an **NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM**. GPU requirements pin the project's CUDA 12.6 PyTorch build; `requirements-lock.txt` records the environment snapshot. CPU inference paths exist but are slower and are not a validated deployment on every laptop. Use compatible CPU PyTorch packages rather than the GPU installation instructions on a non-NVIDIA machine.

Installing packages does not restore model weights/datasets:

- CLIP and DINOv2 pretrained weights download into `runtime/models` when used.
- Run `scripts/setup_person_models.py` for MediaPipe/SegFormer assets on a fresh installation.
- Restore trained head checkpoints/reports privately or reproduce them. Release configurations without matching artifacts do not make models usable.
- `.venv`, `runtime` and Python caches are ignored by Git. Restricted dataset images/crops must not be included in a public repository.

## 3. End-to-end process

Two independent input branches meet at recommendation time:

```text
GARMENT BRANCH
Image -> selected attribute models -> review/correction -> catalogue details
      -> permission confirmation -> SQLite garment catalogue

USER BRANCH
Preferences + optional measurements + optional consented photo
      -> review/correct optional estimates -> apply profile in browser memory

RECOMMENDATION BRANCH
Catalogue + profile/preferences + occasion/weather/budget
      -> graph -> budget/optional size filtering -> rule scores
      -> optional cultural bonus -> ranked garments + reasons + images
```

### A. Analyse and save a garment

1. Open **Garment catalogue** and upload a PNG, JPEG or WebP reference.
2. Select the classifier mode and click **Analyse garment**.
3. Review garment type, pattern, drape and visible texture. Correct suggestions or select `unknown` when evidence is insufficient.
4. Optionally generate Grad-CAM. It explains the model's predicted label, not a manually corrected label.
5. Enter garment name, fashion group, LKR price/status, colours, styles, occasion/weather tags and reviewer ID. The name placeholder is not an AI-generated name; enter the actual name.
6. Optionally supply a sourced body-size chart.
7. Confirm review/image permission and click **Save garment to catalogue**.

**Analysing an image alone does not add it to recommendations.** Saving creates that connection. Saving/correcting records does not retrain the models or automatically create independently verified research ground truth.

### B. Apply an optional personal profile

1. Open **Your profile** and enter colour/style preferences and optional cm/inch measurements.
2. If using a photo, confirm permission before **Analyse person photo**.
3. Inspect/correct/clear the optional landmark and skin-appearance estimates. Confirm review if retaining them.
4. Click **Apply profile**. Measurements are normalized to cm and preference defaults carry into recommendations.

Photo and measurements are optional. Recommendations can use preferences alone. Explicit recommendation choices override profile defaults. Applied profile data stays in browser memory until refresh/clear; this is not a stored user account.

### C. Find recommendations

1. Open **Recommendations**.
2. Choose occasion, maximum LKR budget, weather and any colour/style/type/pattern preferences.
3. Choose **Baseline** or **Sri Lankan guidance — proposed rules**.
4. Optionally enable **Only show size-compatible garments** if measurements/charts exist.
5. Click **Find recommendations** and inspect garment details and reasons.

Example: a reviewed floral saree saved with `pattern_type=floral` receives a pattern-match bonus when the user prefers floral. A saved `drape_style=kandyan_osariya` can match the optional wedding rule. These are explicit graph/ranking connections, not automatic cultural validation.

## 4. Models and application modes

### Models and learned components

| Model/component | Purpose | Project training |
|---|---|---|
| `openai/clip-vit-base-patch32` | Image/text matching; 512-dimensional image features | Backbone frozen |
| IndoFashion linear head | Blouse/dress/kurta/saree classification | Four-class head trained on CLIP features |
| DeepFashion CLIP pattern head | Four-pattern comparison experiment | Small head trained; saved but not selected by the active local release |
| `facebook/dinov2-small` + pattern head | 384-dimensional CLS image features; pattern classification | DINO backbone frozen; four-class head trained |
| Sri Lankan drape linear head | Nivi versus Kandyan/Osariya | Two-class head trained on CLIP features |
| MediaPipe Pose Landmarker Lite float16 v1 | Shoulder/hip landmarks | No local training |
| `mattmdjaga/segformer_b2_clothes` | Face-region parsing for cheek sampling | No local training |
| Monk-reference comparison | Tentative skin-appearance colour reference | Deterministic colour method, not a trained skin-tone model |

Pinned revisions: CLIP `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, DINOv2 `ed25f3a31f01632728cabb09d1542f84ab7b0056`, SegFormer `584abc1e1d260e23c0fc627c5217a09b2b461046`. Trained heads are checked against checkpoint SHA and taxonomy/model metadata.

### Exact selectable behaviour

| Dropdown mode | API value | Garment type | Pattern | Drape | Texture |
|---|---|---|---|---|---|
| General CLIP — default | `zero_shot` | CLIP zero-shot | CLIP zero-shot | CLIP zero-shot + rule | CLIP zero-shot |
| Adapted IndoFashion | `indofashion_probe` | Trained IndoFashion head | CLIP zero-shot | CLIP zero-shot + rule | CLIP zero-shot |
| DINOv2 patterns + CLIP — experimental | `dinov2_pattern` | CLIP zero-shot | Trained DINO head | CLIP zero-shot + rule | CLIP zero-shot |
| Sri Lankan drape + CLIP — experimental | `local_attributes` | CLIP zero-shot | CLIP zero-shot | Trained local head + rule | CLIP zero-shot |

**The modes are separate, not a single combined model.** DINO mode does not also select IndoFashion or the local drape head. Sri Lankan drape mode does not use DINO for patterns. The active generic local-head release currently contains only the drape head.

The rule sets drape to `not_applicable` when the selected garment classifier predicts a non-saree. A mistaken garment prediction can suppress drape analysis. Changing a dropdown manually does not rerun inference. The trained two-class drape head cannot reliably reject unseen/unclear drapes; reviewers can select `other`/`unknown` manually.

### Label vocabulary

- **Garment:** shirt, blouse, kurta, dress, saree, trousers, skirt, other. IndoFashion predicts only blouse/dress/kurta/saree.
- **Pattern:** solid, striped, checked, floral, geometric, abstract, other. Trained pattern heads predict only solid/striped/checked/floral.
- **Drape:** not_applicable, nivi, kandyan_osariya, other. The trained head predicts only nivi/kandyan_osariya.
- **Visible texture:** smooth_matte, shiny, coarse_woven, knitted, lace, other.
- **Unknown:** review/annotation option for insufficient evidence; excluded from supervised targets.

Kandyan saree and Osariya are one drape category. Batik and handloom are separately sourced textile-production metadata, not proof of garment construction or fibre content from a photograph.

Relative/softmax scores depend on the candidate labels and are **not calibrated confidence or measured accuracy**. The score-gap ambiguity flag is an unvalidated review heuristic.

## 5. Datasets and preparation

### 5.1 IndoFashion — garment type

Raw location: `runtime/datasets/raw/indofashion_dataset`, with provider train/validation/test JSON files and `images/train`, `images/val`, `images/test`.

The importer handles JSON-lines annotations and maps only compatible source labels: saree, blouse, men's/women's kurta and gowns as dress. Other categories are excluded rather than guessed. It does not invent pattern, texture or Osariya labels.

Original files were preserved. Derived cleanup removed conflicting exact-image labels/duplicates and checked overlap. The cleaned manifest contains **47,552 images**: 42,700 train, 2,428 validation, 2,424 test. Exact hashes do not rule out every near-duplicate or related product.

The experiment selected **8,000 balanced training-split examples**: 7,200 fitted the head and 800 formed an internal holdout for checkpoint selection. Full validation and a frozen final test were subsequently evaluated. **The IndoFashion test split has already been evaluated and must not be called untouched.**

### 5.2 DeepFashion-MultiModal — pattern

```text
runtime/datasets/raw/deepfashion_multimodal/
├── images/                         # 44,096 JPEGs
├── segm/                           # 12,701 indexed parsing masks
└── labels/
    ├── shape/shape_anno_all.txt
    └── texture/
        ├── pattern_ann.txt
        └── fabric_ann.txt
```

Masks use `IMAGE_STEM_segm.png`; import with `--mask-suffix _segm`. Labels have separate upper/lower/outer columns. Matching parsing regions create garment crops, avoiding attaching an upper-garment label to a whole multi-garment person image.

Direct pattern mappings: floral -> floral, striped -> striped, pure colour -> solid, lattice -> checked. Graphic, colour block, other and NA remain unknown. Only knitted maps directly to our visible-texture vocabulary. Cotton/chiffon/denim are not automatically called smooth, shiny or coarse woven.

The development import inspected 12,000 seeded raw candidates and retained **1,697 upper-clothing crops**. Missing masks, unsupported labels and small/dress/romper regions were skipped. Exact-duplicate and group-overlap audits passed; identity/source independence is not established.

| Split | Retained crops | Known pattern targets |
|---|---:|---:|
| Train | 1,383 | 1,323 |
| Validation | 163 | 155 |
| Reserved test | 151 | 148 |

Training targets: 1,160 solid, 106 striped, 32 checked, 25 floral. Validation: 134 solid, 14 striped, 2 checked, 5 floral. These are **project-created garment-group splits**, not an official benchmark split. Class imbalance and tiny minority counts are major limitations.

Records identify original provider annotations (`provider_annotated`), not new independent human reviews. A visual sample contained noisy labels. Training used segmented garment crops, while uploads are not automatically segmented; that distribution difference matters.

The provider restricts the dataset to non-commercial research and restricts redistribution. Keep raw files, crops and image-containing review packets local. The user's educational/non-commercial purpose is recorded as an assertion, not independent legal verification.

### 5.3 Researcher-collected real saree drapes

```text
data/local_attributes/drape_candidates/real/
├── nivi/                 # 52 supplied files
└── kandyan_osariya/       # 63 supplied files
```

The inventory supports JPEG/JFIF/PNG/WebP. No needs_review folder is required. The researcher confirmed category labels, authorization and absence of extra same-garment views beyond the identified copies. This is not independent expert/legal verification. Missing individual source/licence references were not fabricated.

One exact pair and three visually corroborated resized-photo pairs were deduplicated in the manifest, without deleting originals. **111 candidates** remain:

| Split | Nivi | Kandyan/Osariya | Total |
|---|---:|---:|---:|
| Train | 36 | 41 | 77 |
| Validation | 8 | 9 | 17 |
| Reserved test | 8 | 9 | 17 |

Accepted manifest: `runtime/datasets/drape_reviewed_01/manifest.jsonl`. Group decisions: `config/drape_candidate_groups.json`. Exact duplicate/group checks passed, but unrecognized related views, source/model/background overlap and selection bias remain possible.

### 5.4 Synthetic assets and human relevance ratings

Seven demonstration catalogue records in `data/garments.json` use copied synthetic references in `frontend/assets/demo`. Prices and suitability tags are illustrative, not verified retail listings. Synthetic smoke-test success is not real-world accuracy evidence.

Recommendation study `runtime/recommendation_studies/pilot_02` freezes four constructed scenarios and **24 eligible candidate/scenario judgments**. One user-supplied reviewer export has been evaluated. No consented local person-reference dataset for body/skin calibration or fairness has been prepared yet.

## 6. Training and recorded results

### Method: frozen encoder + trained linear head

The pretrained encoder extracts an image vector. A small linear layer learns the target labels. This is **transfer learning through linear probing**, not full-backbone fine-tuning.

Local CLIP/DINO pattern and drape runs use AdamW, learning rate .02, weight decay .01, seed 42 and 100 epochs, selecting the lowest validation-loss epoch. IndoFashion used 200 epochs and its separate internal training-split holdout. Unknown labels are masked per attribute, so partial annotations can train only their known targets.

Runs save checkpoint/model/taxonomy fingerprints, sample IDs, image/group/split hashes, histories and predictions. Training extracts no test features. The audit reads all split image bytes for integrity hashing only; that is not test inference or training.

### 6.1 IndoFashion — final frozen test

| Model | Candidate classes | Accuracy on 2,424 test images | Macro-F1 |
|---|---:|---:|---:|
| General zero-shot CLIP | 8 | 65.64% | 0.6547 |
| Matched zero-shot CLIP | 4 | 70.01% | 0.6670 |
| Trained IndoFashion head | 4 | **90.10%** | **0.8990** |

The controlled comparison is against the four-class baseline, not the different eight-class task. Full validation accuracy on 2,428 images was 68.62% matched baseline versus 89.95% adapted. These scores measure garment type on the mapped IndoFashion subset only.

### 6.2 DeepFashion pattern — development validation

| Model | Correct / 155 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Matched four-pattern CLIP zero-shot | 151 | 97.42% | 0.8809 |
| Trained CLIP pattern head | 152 | 98.06% | 0.8769 |
| Trained DINOv2-Small pattern head | 153 | **98.71%** | **0.9056** |

DINO gains one correct prediction over the CLIP head. Validation selected checkpoints/models, so this is not independent final evidence, proof of general superiority or Sri Lankan pattern accuracy. **No final pattern-test evaluation has been performed.**

DINO extracted 1,478 train/validation features in 24.03 seconds at batch size 16 on the recorded CUDA machine. Original CLIP timing was not captured, so no controlled speed superiority is claimed.

### 6.3 Local drape — development validation

| Model | Correct / 17 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Matched two-drape CLIP zero-shot | 10 | 58.82% | 0.5641 |
| Trained CLIP drape head | 16 | **94.12%** | **0.9412** |

The head classified 8/8 Nivi and 8/9 Osariya correctly. Its checkpoint was selected using this very small, researcher-labelled validation set. The two-class baseline is not the full application's four-label drape task/applicability gate. **No final drape-test evaluation has been performed.**

### 6.4 Recommendation pilot — one reviewer, four scenarios

| Mean metric | Baseline | Optional cultural guidance |
|---|---:|---:|
| Precision@5 | **0.9000** | 0.8500 |
| Recall@5 | **0.9167** | 0.8667 |
| NDCG@5 | **0.9170** | 0.8717 |

Cultural bonuses reduced quality under this reviewer's judgments. The result is preserved, not adjusted to favour the guided method. Scenario-level analysis and independent reviews were deferred at the user's request, not completed.

Grades 2–3 count as relevant. NDCG uses graded gain `2^grade - 1`. Precision divides by K even when fewer candidates exist. Recall uses all relevant items in the complete eligible pool. No-relevant cases have undefined recall/NDCG. This small study is not broad usefulness or demographic fairness evidence.

## 7. Body, measurements and skin appearance

### Body and measurements

MediaPipe finds shoulder/hip landmarks. Where visibility permits, the module calculates a **projected shoulder-span / hip-span ratio**. It is not a physical circumference, exact body dimension, clothing size or validated body-shape category. Pose, perspective and clothing affect it.

Size matching instead uses self-reported height, chest/bust, waist, hips, shoulder width and inseam as relevant. Inches convert to cm. Photo estimates never overwrite entered measurements; unknown measurements are not guessed.

### Skin appearance

SegFormer locates a high-score face region. Interior cheek patches are sampled; median RGB is converted to CIELAB and compared by colour distance with ten Monk reference swatches. Users can correct, clear or leave estimates unknown. Insufficient evidence can produce unavailable status.

This is a **lighting-dependent appearance estimate**, not verified intrinsic skin tone. White balance, illumination, makeup and parsing errors matter. It does not infer ethnicity, Fitzpatrick type or medical characteristics. Local calibration, reference-label evaluation and fairness checks remain unfinished.

Photo-derived ratios and colour estimates **do not affect ranking**. Preferences and entered measurements can. Person landmark/cheek overlays show calculation regions; they are not Grad-CAM.

## 8. Catalogue, knowledge graph and recommendations

### Storage and graph

`runtime/catalogue.sqlite3` stores reviewed garment metadata, normalized PNG images, reviewer/time and permission assertion. Back up this database to preserve additions. Demonstration records are loaded separately from `data/garments.json`.

`backend/engine.py` rebuilds an in-memory adjacency-based knowledge graph for each recommendation request. It is **not Neo4j, a graph neural network or a learned graph-embedding recommender**.

Relations: suitable_for, comfortable_in, has_colour, has_style, made_from, has_type, has_pattern, has_drape, has_visible_texture and uses_textile_technique. Stored graph attributes do not automatically all contribute points.

### Exact scoring

Over-budget garments are removed. Remaining garments receive:

| Match/condition | Points |
|---|---:|
| Within budget | +1 |
| Occasion tag | +2 |
| Weather tag | +2 |
| At least one preferred colour | +3 |
| At least one preferred style | +3 |
| Preferred garment type | +3 |
| Preferred pattern | +2 |
| Complete supplied body-size chart compatible | +2 |
| Compatible variant matches preferred cut | +1 |
| Optional cultural rules | Up to +2 total |

Each condition awards once, not once per matching colour/style. Sort order is descending score, ascending price, then ID. Scores are rules, not probabilities. Occasion/weather usually add points rather than hard-filtering; budget and the optional strict size filter are hard constraints.

### Cultural guidance

`data/cultural_guidance.json` holds proposed project hypotheses: Osariya/wedding +2, batik/casual +1, batik/office +1 and handloom/festival +1, capped at +2 per garment. Baseline mode adds none of these bonuses, but already uses catalogue occasion/style tags: it is not a culture-free versus culture-aware comparison.

Rules are optional, shown in reasons and do not prohibit Western clothing. They require local/contextual validation. Batik/handloom bonuses require sourced metadata, not an image-proven technique; the form does not automatically establish production methods. Visible texture has no direct scoring bonus; drape currently affects scores through the optional matching cultural rule.

### Size compatibility

The UI accepts a size variant with supplier **body-measurement ranges** in cm and a reference, not flat garment widths. The backend supports multiple variants.

- Dress requires chest, waist and hips.
- Skirt/trousers require waist and hips.
- Other supported upper-garment/saree-blouse cases require chest and waist.
- Any additional supplied range also needs its corresponding user measurement.

Missing/incomplete records are unknown, not presumed compatible. Strict filtering excludes unknown, incompatible and not-assessed cases. Saree matching covers the **blouse only**, not saree length/drape. Chart compatibility is not a physical fit guarantee or simulation. No invented size charts were added to demo garments.

## 9. Grad-CAM explanations

The optional panel explains the selected model's predicted attribute:

- CLIP: `vision_model.encoder.layers[-1].layer_norm1`, before final attention; 7×7 patches.
- DINO pattern: `encoder.layer[-1].norm1`; 16×16 patches.
- Learned CLIP heads: differentiate the actual learned class logit, not a zero-shot text score.
- Non-saree not_applicable rule: no heatmap is invented.

The class token is excluded from the spatial map. The UI displays the actual 224×224 model crop and orange/red positive-attribution overlay, not a crop heatmap stretched over the full original photo. Uniform/no-positive maps are flagged as uninformative.

Heatmaps are not segmentation, calibrated confidence, causal proof or accuracy/fairness evidence. Background highlighting can expose questionable behaviour. Gradient calculation does not train the model, and hooks are removed afterwards. Manual corrections are not what the original heatmap explains.

## 10. Technology stack and folder structure

| Layer | Technology |
|---|---|
| Backend | Python 3.12, standard-library ThreadingHTTPServer, 127.0.0.1:8001 |
| Frontend | HTML/CSS/vanilla JavaScript; three-column workspaces |
| Models/training | PyTorch, torchvision, Transformers 4.46.3, safetensors |
| Downloads | huggingface-hub 0.36.0; pinned revisions |
| Images/maths | Pillow, NumPy, SciPy |
| Landmarks | MediaPipe 1.1.1 |
| Persistence | Standard-library SQLite |
| Ranking | Project Python graph/rules; NetworkX is installed but is not the ranking engine |
| Verification | unittest, HTTP smoke scripts and saved reports |

No paid inference API, external LLM, Flask/FastAPI, React build server or graph database is needed. Desktop layout avoids a long body scroll; panels can scroll when content overflows. Small screens stack the panels.

```text
component-2-fashion-recommendation/
├── backend/                 # API, models, attributes, person, graph, sizing, storage
├── frontend/                # Three workspaces and explanation/model controls
│   └── assets/demo/         # Synthetic demo thumbnails
├── config/                  # Selected/frozen releases and evaluation status
├── data/                    # Demo catalogue, cultural rules, candidate labels
├── docs/                    # Detailed methods/results/limits
├── scripts/                 # Setup, import, audit, train, evaluate, smoke checks
├── tests/                   # Automated correctness checks
├── runtime/                 # Private/generated artifacts, ignored by Git
│   ├── catalogue.sqlite3
│   ├── models/              # Pretrained cache and person assets
│   ├── datasets/            # Raw downloads and derived data/manifests
│   ├── adaptation/          # Trained heads, reports, histories
│   ├── evaluations/         # Metrics/predictions/verification images
│   └── recommendation_studies/
├── requirements.txt
├── requirements-gpu.txt
├── requirements-lock.txt
└── setup.ps1
```

Important artifacts:

| Artifact | Location |
|---|---|
| IndoFashion head | `runtime/adaptation/indofashion_probe_01` |
| IndoFashion final test | `runtime/evaluations/indofashion_final_test_01` |
| CLIP pattern head | `runtime/adaptation/deepfashion_pattern_01` |
| DINO pattern head | `runtime/adaptation/dinov2_pattern_01` |
| Sri Lankan drape head | `runtime/adaptation/sri_lankan_drape_clip_01` |
| Recommendation packet/results | `runtime/recommendation_studies/pilot_02` |
| Accepted drape dataset | `runtime/datasets/drape_reviewed_01` |

`config/local_attribute_release.json` currently selects the drape head. `config/dinov2_pattern_release.json` selects DINO. Never replace them with synthetic smoke-test models. Older prototype reports remain historical; use the accepted runs above.

## 11. API

The frontend uses local same-origin JSON requests:

| Endpoint | Purpose |
|---|---|
| GET `/api/health` | Component/API/features and optional-model availability |
| POST `/api/analyse-garment` | Image data URL, classifier_mode and optional include_explanation/explanation_attribute |
| POST `/api/analyse-person` | Person data URL with explicit consent=true |
| POST `/api/profile` | Validate preferences, units, measurements and reviewed estimates |
| POST `/api/catalogue` | Save reviewed image/record with reviewer and permission confirmation |
| GET `/api/garment-images/<id>` | Saved PNG image |
| POST `/api/recommendations` | Budget/context/preferences and optional normalized user_profile -> ranked catalogue |

Image formats: PNG/JPEG/WebP. Decode limit: 20 megapixels, resized to at most 1600×1600. Browser upload cap: 8 MB. Image endpoint JSON-body cap: 12,000,000 bytes; other bodies: 100,000 bytes. These differ because base64 enlarges images.

Validation errors generally return 400; unexpected failures return 503 with a generic user message and terminal diagnostic. Optional missing/corrupt models are not advertised as available. The server has no authentication and is intended for local research, not direct public deployment.

## 12. Tests and experiment commands

### Software checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
```

The current verified suite has **71 passing tests**. This snapshot count will change as tests are added; it is not research accuracy.

Real-model checks create new output directories. If a name exists, choose another; results are deliberately not overwritten:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_dinov2_api.py --output runtime\evaluations\dinov2_hybrid_smoke_02
.\.venv\Scripts\python.exe scripts\smoke_drape_api.py --output runtime\evaluations\drape_api_smoke_02
```

The drape check uses a confirmed training image, not the reserved test set. These verify functionality, not new accuracy. Person smoke instructions are in [PERSON_ANALYSIS_BASELINE.md](docs/PERSON_ANALYSIS_BASELINE.md).

### Reproduce development training

Use accepted manifests and required original comparison artifacts. These train separate new outputs without automatically replacing the app models:

```powershell
# CLIP pattern head
.\.venv\Scripts\python.exe scripts\train_local_attributes.py --manifest runtime\datasets\deepfashion_pattern_development_01\manifest.jsonl --image-root runtime\datasets\deepfashion_pattern_development_01 --attributes pattern_type --output runtime\adaptation\deepfashion_pattern_02 --epochs 100 --batch-size 16

# DINO pattern head, aligned with the original CLIP run
.\.venv\Scripts\python.exe scripts\train_dinov2_pattern.py --manifest runtime\datasets\deepfashion_pattern_development_01\manifest.jsonl --image-root runtime\datasets\deepfashion_pattern_development_01 --clip-run runtime\adaptation\deepfashion_pattern_01 --output runtime\adaptation\dinov2_pattern_02 --batch-size 16

# Local drape head
.\.venv\Scripts\python.exe scripts\train_local_attributes.py --manifest runtime\datasets\drape_reviewed_01\manifest.jsonl --image-root data\local_attributes\drape_candidates\real --attributes drape_style --output runtime\adaptation\sri_lankan_drape_clip_02 --epochs 100 --batch-size 8
```

For the CLIP-head script, use `--audit-only` without `--output` to check inputs before training. Minimum label-count checks are software guardrails, not research sample-size recommendations. Full IndoFashion preparation/training/comparison details are in its linked guides. Freeze development choices before final evaluation; do not repeatedly use a test split for selection while calling it independent.

### Collect and score recommendation ratings

Create a new frozen study packet after deliberate catalogue/scenario changes:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_recommendation_study.py --output runtime\recommendation_studies\pilot_03
```

Open its `rate_garments.html`, enter a reviewer ID, score every candidate 0–3 and export **before closing/refreshing**. Ratings stay in memory until export. Model order/identity is hidden in the interface but inspectable in the embedded packet; this is interface blinding, not secure concealment.

Use actual exported paths and a new output name:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_recommendations.py --packet runtime\recommendation_studies\pilot_03\packet.json --ratings "C:\Users\Ravindu\Downloads\relevance_ratings.json" --output runtime\recommendation_studies\pilot_03\results_reviewer_01.json --k 5
```

Old pilot_02 ratings do not match a new packet. The evaluator checks packet identity/integrity and complete eligible-pool judgments. AI/example scores must not be reported as genuine human/expert ratings.

## 13. Privacy, permissions and reproducibility

- Inference uploads/results are processed in memory and not intentionally saved just for analysis.
- Explicit garment saving persists metadata/images in SQLite. Study/contact-sheet scripts can also create private image-containing artifacts.
- Profiles, measurements and optional photo estimates are not saved as user accounts. Refresh/clear removes applied browser state.
- No photo is intentionally sent to an external inference API. Model downloads/metadata checks still produce normal model-host network traffic.
- Keep per-image source/licence references where available. User confirmation is an assertion, not legal verification. Public visibility is not automatically permission; model and dataset licences are separate.
- Do not publish restricted datasets, crops or image-containing review packets. Cite providers.
- Keep garment views in one split and separate identities/sources where feasible. Hashes and user grouping do not guarantee complete absence of leakage.
- Preserve manifests, group decisions, revisions, checkpoint hashes and reports. Privately back up the catalogue database and permitted model artifacts when moving machines.
- IndoFashion's test is evaluated; pattern and drape tests remain reserved. Final-test-based tuning requires a new independent evaluation before fresh-test claims.

## 14. Troubleshooting

| Symptom | Check |
|---|---|
| Failed to fetch | Check port 8001, server terminal and same localhost origin. An old backend can lack new routes/features. |
| New mode missing/disabled | Restart backend, refresh and verify configuration/checkpoint/report integrity. Do not activate fixture weights. |
| Address already in use | Stop the known old server; do not kill unrelated Python processes. |
| First analysis slow | Library/model loading and model-host checks/downloads take time. Later requests reuse the model. Cached-only browser verification was used; no permanent offline startup fix is claimed. |
| Saree drape says not applicable | Inspect whether the garment classifier predicted a non-saree; manual dropdown changes do not rerun inference. |
| High relative score but wrong label | Closed-set heads and uncalibrated scores require review; use unknown/other as needed. |
| Strict size filter returns nothing | Supply required measurements and sourced complete charts, or disable the filter. |
| Existing output directory | Choose a new name and preserve old results. |
| Checkpoint mismatch | Check revision/taxonomy/SHA; do not bypass checks by editing hashes or reported results. |
| Windows process stays alive after completion | This happened after two successful training/smoke runs. Processes were closed only after artifacts/success were verified; shutdown behaviour was not diagnosed as fixed. |

Dependency installation, model training and starting the server are separate operations. Do not recreate the environment or change drivers just because an output folder already exists.

## 15. Remaining research work

The original plan includes global/local attribute models, body/skin calibration and fairness, locally aware graph recommendations and measured evaluation. The implementation is substantial but not a fully validated research component.

Remaining major tasks:

1. A **multi-class locally reviewed texture head** and evaluation; knitted-only annotations are insufficient.
2. Larger, better-sourced, independently reviewed drape data and evaluation beyond forced Nivi/Osariya classification.
3. Local skin reference labels, **calibration and fairness/lighting evaluation**. FairFace/Casual Conversations were mentioned in the plan but have not been imported or used here. No Fitzpatrick classifier is implemented.
4. Validated **body-shape analysis** beyond a projected landmark ratio and self-entered measurements.
5. Frozen final evaluation of selected pattern/drape models, with uncertainty appropriate to sample sizes.
6. Independent cultural-rule/catalogue validation and recommendation reviews, more representative scenarios and statistical analysis.
7. Optional unified use of the separately selected trained heads. Current modes do not combine them all.
8. A reviewed real catalogue, user studies and optional Component 1 handoff.

We do not claim full completion, demographic fairness, universal cultural rules or 94.12% general drape accuracy from 17 development examples.

## 16. Detailed guides and primary sources

### Project guides

- [Attribute vocabulary and annotation](docs/ATTRIBUTE_ANNOTATION_GUIDE.md)
- [IndoFashion adaptation](docs/INDOFASHION_ADAPTATION.md), [full validation](docs/FULL_VALIDATION_RESULTS.md), [final test](docs/FINAL_TEST_RESULTS.md)
- [DeepFashion import](docs/DEEPFASHION_MULTIMODAL_IMPORT.md), [CLIP pattern results](docs/DEEPFASHION_PATTERN_DEVELOPMENT_RESULTS.md)
- [DINO pattern experiment/integration](docs/DINOV2_PATTERN_EXPERIMENT.md)
- [Drape candidate audit](docs/DRAPE_CANDIDATE_CHECK.md), [preparation](docs/DRAPE_DATASET_PREPARATION.md), [trained drape pilot/integration](docs/DRAPE_CLASSIFIER_PILOT.md)
- [Local multi-attribute training](docs/LOCAL_ATTRIBUTE_TRAINING.md)
- [Person-analysis baseline](docs/PERSON_ANALYSIS_BASELINE.md)
- [Sizing rules](docs/SIZING_RULES.md)
- [Visual explanations](docs/VISUAL_EXPLANATIONS.md)
- [Recommendation study and metrics](docs/CULTURAL_RECOMMENDATION_STUDY.md)

### Providers

- [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [Meta DINOv2 repository](https://github.com/facebookresearch/dinov2), [DINOv2-Small model](https://huggingface.co/facebook/dinov2-small)
- [IndoFashion](https://indofashion.github.io/)
- [DeepFashion-MultiModal: downloads, schema and research agreement](https://github.com/yumingj/DeepFashion-MultiModal)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [SegFormer clothes checkpoint](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
- [Monk Skin Tone scale](https://skintone.google/)
- [ViT Grad-CAM guidance](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md)

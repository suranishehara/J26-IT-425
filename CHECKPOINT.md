# Component 2 — paused implementation checkpoint

Saved: 18 October 2026. Implementation is paused at the user's request. No further training, release changes or implementation should be inferred from this checkpoint.

## Resume reference

Read [README.md](README.md) first. It records the complete implementation, dataset locations, model revisions, experiment results, commands and limitations as of this checkpoint. Consult the linked project guides and saved machine-readable reports before changing models.

## Completed implementation

- Three-tab application: personal profile, garment catalogue and recommendations; desktop three-column layout.
- General CLIP garment attributes and optional adapted IndoFashion four-class garment-type classifier.
- DeepFashion-MultiModal import, segmented pattern crops and frozen-encoder pattern experiments; optional DINOv2-Small pattern classifier.
- Local Nivi/Kandyan Osariya dataset audit, duplicate grouping, reviewed manifest, group-separated splits and trained CLIP drape head.
- Optional experimental drape mode activated; General CLIP remains the default. Separate modes do not combine all trained heads.
- Model-specific garment Grad-CAM, including rule-based no-heatmap handling for non-saree drape labels.
- Optional person landmarks, projected shoulder/hip ratio, reviewed Monk appearance estimate and self-reported measurements. These are not validated body-shape or skin-tone classifiers.
- Reviewed SQLite garment catalogue, adjacency knowledge graph, explainable ranking, optional cultural guidance and size-chart compatibility.
- Recommendation-rating study/export and evaluation tooling.
- Complete README updated and its 16 local documentation links checked.

## Accepted artifacts and evaluation boundary

| Item | Current artifact / status |
| --- | --- |
| IndoFashion garment head | `runtime/adaptation/indofashion_probe_01`; final test already evaluated: 90.10% accuracy, macro-F1 0.8990 |
| CLIP development pattern head | `runtime/adaptation/deepfashion_pattern_01` |
| Optional DINO pattern head | `runtime/adaptation/dinov2_pattern_01`; validation 153/155 correct; pattern test remains reserved |
| Reviewed local drape manifest | `runtime/datasets/drape_reviewed_01/manifest.jsonl`; 111 images: 77 train, 17 validation, 17 test |
| Optional CLIP drape head | `runtime/adaptation/sri_lankan_drape_clip_01`; validation 16/17 correct; drape test remains reserved |
| Local drape release selection | `config/local_attribute_release.json` |
| DINO release selection | `config/dinov2_pattern_release.json` |
| Recommendation pilot | `runtime/recommendation_studies/pilot_02`; baseline outperformed optional cultural guidance in the small pilot |
| Automated tests | Last recorded verified suite: 71 passing tests; not rerun for this documentation checkpoint |

Synthetic smoke runs verify mechanics only; do not activate their heads or claim real-data accuracy from them. Drape validation is a small development result, not general Sri Lankan drape accuracy. Source/permission information remains user-confirmed rather than independently verified per image.

## Remaining work when resumed

1. Agree on the next implementation task before making changes. Potential work includes unified optional head selection and completing attribute coverage with appropriately annotated data.
2. Improve local dataset diversity and independently review labels, grouping, provenance and permissions.
3. Freeze models before final pattern/drape test evaluation; do not tune on held-out test outcomes.
4. Complete local person-analysis calibration and fairness evaluation if retaining those research claims. Photo appearance estimates must not be represented as ethnicity or clinical skin type.
5. Validate cultural rules and catalogue metadata with independent reviewers. The user previously deferred scenario-by-scenario recommendation error analysis; do not restart it automatically.
6. Expand recommendation evaluation and document uncertainty. Batik/handloom production methods cannot be verified from appearance alone.
7. Component 1 integration is not implemented; Component 1 has not been modified during this checkpoint.

## Restart application

From this Component 2 folder in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe backend\app.py
```

Open `http://localhost:8001`. Keep the server terminal running. For checks in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Do not assume an application server is running after resuming. Preserve original datasets, accepted checkpoints, release configurations and the user's catalogue. This pause does not authorize deletion or retraining.

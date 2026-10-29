# Calibration and unknown-input evaluation

## What is implemented

1. `scripts/finetune_convnext.py`: controlled five-epoch update of the final two feature modules, final normalization and classifier head. Earlier stages are checked unchanged, epoch-zero frozen model remains eligible, selection uses validation macro-F1 and saved-checkpoint FP32 predictions are rechecked. New output only; no final-test inference or automatic activation.
2. `scripts/calibrate_fabric.py`: temperature fitting on 70% of validation groups, assessment on the remaining groups, using deterministic seed-tagged group hashes. Scalar temperature minimizes fit-partition negative log-likelihood over [0.05, 20], with temperature 1 retained as a candidate. Metrics include negative log-likelihood, multiclass Brier **sum** score and 10-bin equal-width top-label ECE. This follows the basic temperature-scaling method in [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html).
3. `scripts/evaluate_unknown_inputs.py`: evaluate the existing raw-score/capture review policy on a separately labelled external photo manifest. No threshold fitting or automatic model changes. Reports supported coverage and accuracy, unknown false-quick-apply/rejection rates and unknown-positive AUROC using one minus the maximum raw supported-class score. It distinguishes unsupported textiles from non-fabric objects and reports provenance subsets where both known/unknown examples are present.

## Important boundaries

The classifier has already been selected using the entire validation split. A later split of those images is **not independent calibration evidence**, even with group separation. Results are exploratory development measurements; a separately collected calibration/assessment set is needed before claiming reliable deployed probabilities. Scalar temperature changes scores, not top-1 predictions. Calibration is not an unknown-input detector. No fitted temperature is activated in the application in this milestone.

Unknown-input evaluation needs actual permitted, independently labelled photos. Unit fixtures are not real OOD evaluation. The existing 60% score / 15-point margin / capture-warning checks remain explicitly unvalidated review heuristics. Batik and handloom are techniques that may occur within the seven supported fabric categories; they are not automatically unknown or unsupported composition categories.

## Calibration pilot measured for the active EfficientNet checkpoint

Output: `runtime/evaluations/efficientnet_calibration_01/report.json` with logits beside it. Fit: 249 images; assessment: 107. Fitted temperature: **1.2306688**.

| Assessment metric | Before | After temperature scaling |
| --- | --- | --- |
| Top-1 accuracy | 65.42% | 65.42% |
| Negative log-likelihood | 1.2513 | 1.1583 |
| Multiclass Brier sum | 0.5298 | 0.5130 |
| 10-bin equal-width ECE | 0.1504 | 0.1003 |

This subset accuracy is not a replacement for the 356-image full validation score. ECE depends on bins and sample size; improvement here is not proof of deployment calibration. The report pins the checkpoint/manifest and records fit/assessment group membership. App release and raw-score policy remain unchanged unless a separate explicitly recorded release decision is made.

### ConvNeXt fine-tuned candidate calibration pilot

Output: `runtime/evaluations/convnext_calibration_01/report.json`. Same group-separated 249 fit / 107 assessment images, temperature **2.2322385**, fitted independently for the ConvNeXt checkpoint. On the assessment subset, NLL changed **1.7079 → 1.0909**, Brier sum **0.5806 → 0.5065**, and ECE **0.2226 → 0.0966**. Top-1 subset accuracy stayed **63.55%**. This subset is not a new model-selection test; the whole-validation model accuracy is 69.38%. The same post-selection bias warning applies. Neither temperature is activated in the app. All 74 automated tests passed, including metric fixtures, group separation and unknown-evaluation manifest validation.

### ConvNeXt V2 candidate diagnostic

The newer V2 candidate has a separate report at `runtime/evaluations/convnextv2_calibration_01/report.json`: temperature **2.0372503**, assessment NLL **1.4341 → 0.9831**, Brier sum **0.4837 → 0.4421**, ECE **0.1861 → 0.0700**, and unchanged subset accuracy **71.03%**. Same 249/107 group partition and post-selection limitations; no activation or actual external OOD evaluation. The V2 whole-validation classification result is 72.47%, not the subset value. Latest suite across the component: 77 tests passed. Do not reuse any model's fitted temperature with another checkpoint.

```powershell
.\.venv\Scripts\python.exe scripts\finetune_convnext.py
.\.venv\Scripts\python.exe scripts\calibrate_fabric.py
```

Both default output directories already exist after successful runs: choose a new `--output` to reproduce without overwriting. Calibration accepts `--run runtime\training\convnext_tiny_finetune_01` to assess that candidate separately; never reuse a temperature fitted for different weights.

## Prepare the missing external evaluation set

Use permitted close-ups from **new physical samples**, not provider train/validation/test images or new crops of those same samples. Include:

- Supported examples with trustworthy category labels for the seven-class inventory.
- Unsupported fabrics, for example documented linen, distinct from supported categories/blends.
- Non-fabric objects with varied backgrounds/capture conditions.

Capture multiple conditions per sample while keeping a shared specimen ID. Record actual source/permission references; set provenance to unknown unless documented. Keep near-duplicates and sources separated from any future tuning set. A small pilot can test mechanics, but no fixed image count guarantees reliable rejection metrics or coverage across all populations.

Place photos under `data/unknown_inputs/images/` (or pass another root). Create `data/unknown_inputs/manifest.jsonl`, one JSON object per image:

```json
{"id":"sample_001_view_1","specimen_id":"sample_001","path":"sample_001_view_1.jpg","split":"development","kind":"unsupported_fabric","true_category":"linen","provenance":"unknown","permission":true,"reference":"Actual supplier composition record and image permission reference","file_sha256":"actual lowercase SHA-256 of this image"}
```

This is a schema example, not collected evidence. Allowed kinds: `supported`, `unsupported_fabric`, `non_fabric`. Supported rows need a true_category from `cotton,cotton_mixed,denim,polyester,silk,viscose,wool`; unknown rows need a label outside that inventory. Provenance: `sri_lankan`, `global`, `unknown`. Files must resolve inside the image root. Each row needs an explicit permission assertion, reference, specimen ID, unique image ID and checksum. Do not place personal body photos in this set.

Compute a checksum in PowerShell:

```powershell
(Get-FileHash -LiteralPath 'data\unknown_inputs\images\sample_001_view_1.jpg' -Algorithm SHA256).Hash.ToLower()
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_unknown_inputs.py --manifest data\unknown_inputs\manifest.jsonl --image-root data\unknown_inputs\images --output runtime\evaluations\unknown_inputs_01
```

The tool requires both supported and unknown samples and rejects exact hash overlaps with provider/raw/development data. It reads dataset hashes for overlap checks but does not perform inference on final-test images. Hash checks cannot detect all near-duplicates or prove specimen/source independence: human provenance review remains necessary. Reports contain scores/labels/IDs, not copied photos. Repeated views are reported as image-weighted, not falsely counted as independent specimens. Collect a separate assessment set if thresholds are subsequently tuned; do not tune to this report and present it as final evaluation.

**Current unknown-input evidence status:** tooling and artificial-fixture tests only; no external real-image evaluation or validated OOD detector until the researcher supplies the required set.

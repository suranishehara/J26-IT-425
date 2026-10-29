# Independent evaluation and fabric-attribute research pipeline

## Current status — implementation, not completed research

The researcher confirmed there is **no new labelled collection yet**. The existing provider dataset has seven category labels, not verified batik/handloom, print, weave or texture annotations. Therefore no real attribute heads have been trained, no independent calibration/assessment results exist, and the provider final test remains reserved. The web app and its three category models are unchanged by this milestone.

Implemented now: collection inventory/schema auditing, masked independent technique/categorical head training, external calibration/assessment, attribute assessment, and explicitly guarded final evaluators. Unit tests and a temporary artificial-fixture smoke verify mechanics only. No placeholder metrics count as research results. No new head is activated automatically.

## 1. Collect and inventory permitted photos

Put new fabric close-ups in `data/research_collection/images/`. Include actual supported-category samples, unsupported fabrics and non-fabric objects for unknown-input evaluation. A documented material label is needed: do not guess composition from the photo. Batik/handloom need production/source records or expert review; a printed motif is not proof of batik and visible coarse texture is not proof of handloom.

Keep all views/crops of one physical specimen in one split. Record an acquisition/source-batch group, and keep that group in one split too. This is not just the website's name: use the actual source/acquisition grouping and document it. Review near-duplicates/source dependence manually even if hashes differ. Use documented `sri_lankan` or `global` provenance; otherwise `unknown`. Separate genuine new samples from provider/development samples and synthetic demonstrations.

Generate an **unreviewed** inventory:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_collection_manifest.py --image-root data\research_collection\images --output data\research_collection\manifest.draft.jsonl
```

The script writes paths/hashes only, refuses overwrite, and leaves scientific labels, permissions, splits and reviewers unset. No image means no draft is written. Make a reviewed `manifest.jsonl` only after supplying the actual evidence. We can help prepare it when photos are available.

## 2. Review labels and assign splits before training

Splits: `train`, `validation`, `calibration`, `assessment`, `final_test`. Do not move samples between them in response to results. Use train/validation for attribute development; calibration solely fits category temperature; assessment evaluates the frozen category model/temperature and attribute heads; final_test is reserved for frozen final reporting. A single collection may contain samples useful for only some tasks; unsupported/non-fabric examples are not temperature-fit examples for the seven-category classifier.

Each row needs unique `id`, project-relative-to-image-root `path`, `file_sha256`, physical `specimen_id`, `source_group_id`, `image_kind: "real"`, `source_reference`, `permission_reference`, `permission_confirmed: true`, `independence_reviewed: true`, `reviewer`, `region`, `split`, `kind`, `category`, and `labels`. These are provenance assertions, not automatic certification.

Kinds: `supported`, `unsupported_fabric`, `non_fabric`. Supported category must be one of `cotton`, `cotton_mixed`, `denim`, `polyester`, `silk`, `viscose`, `wool`; unknown kinds require a descriptive category outside that inventory. Non-fabric objects must not have fabric-attribute annotations.

Label schema example **only**, not a collected record:

```json
"labels": {
  "batik": {"value": true, "reference": "Actual production record or expert review", "reviewer": "Reviewer code"},
  "handloom": null,
  "pattern_type": {"value": "floral", "reference": "Actual reviewed visible-pattern annotation", "reviewer": "Reviewer code"},
  "weave": null,
  "fabric_texture": null
}
```

- Batik and handloom are **separate** binary outputs: true, explicitly verified false, or unknown null. Both can be true. Missing is never converted to false.
- Print/pattern uses Component 2's vocabulary: solid, striped, checked, floral, geometric, abstract, other. Unknown is null, not other. This head describes visible pattern, not proof of printing/dyeing method.
- Weave: plain, twill, satin, other; unknown null. These are provisional supported labels requiring expert review/adequate image scale.
- Texture uses shared vocabulary: smooth_matte, shiny, coarse_woven, knitted, lace, other; unknown null. This visual-surface label does not certify material composition or physical drape.
- Every known label needs its own reference and reviewer. Do not fill unsupported labels from filenames, category guesses or model predictions.

Agree definitions/capture protocols with an appropriate reviewer. Balance physical specimens, not near-identical views. The trainer's two-distinct-training-values requirement is a mechanics minimum, **not** an adequate research sample-size recommendation or accuracy guarantee. Calibration requires all seven supported categories; report small/absent region/class groups honestly.

Audit before use:

```powershell
.\.venv\Scripts\python.exe scripts\audit_research_collection.py --manifest data\research_collection\manifest.jsonl --image-root data\research_collection\images --output runtime\evaluations\research_collection_audit_01
```

The validator rejects unreviewed permission/independence flags, bad/missing label provenance, split-crossing specimen/source groups, conflicting known specimen labels, invalid paths/hashes, duplicate encoded/decoded photos and exact provider overlap. Raw provider bytes may be hashed for overlap checks, but are not fed through a model by the audit. It does not prove independence or detect all recrops/near-duplicates.

## 3. Train attribute heads only when real labels exist

```powershell
.\.venv\Scripts\python.exe scripts\train_attribute_heads.py --manifest data\research_collection\manifest.jsonl --image-root data\research_collection\images --output runtime\training\fabric_attribute_heads_01
```

The current pinned V2 category checkpoint supplies a **frozen** 768-D encoder. Separate linear heads learn batik, handloom, visible pattern, weave and texture using only known train labels. Binary heads use masked weighted BCE; categorical heads use masked weighted cross entropy. Each head is selected independently by validation macro-F1; default 100 AdamW epochs, LR 0.02, weight decay 0.01, seed 42. Heads lacking two distinct known training values or known validation examples are skipped; no eligible head means an error. Only supported trained categorical outputs are created; known validation-only labels count as errors, not quietly excluded.

Only train/validation features are extracted. The schema audit can decode other splits for duplicate checks, not model inference. Encoder tensors must remain equal to the pinned source checkpoint. Safetensor head reload/logit/prediction checks are required. Outputs: `attribute_heads.safetensors` and `report.json` with encoder identity, supported-label coverage, skipped heads, per-head metrics, histories and manifest hash. No new head is automatically loaded into the web app; independent assessment is required before an explicit release decision.

## 4. Fit independent category calibration and assess it

Freeze the category model ID before fitting. This tool does not pick the best model from assessment outcomes:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_research_release.py --stage calibration --manifest data\research_collection\manifest.jsonl --image-root data\research_collection\images --model-id convnext_v2 --output runtime\evaluations\independent_calibration_01
.\.venv\Scripts\python.exe scripts\evaluate_research_release.py --stage assessment --manifest data\research_collection\manifest.jsonl --image-root data\research_collection\images --model-id convnext_v2 --calibration runtime\evaluations\independent_calibration_01\calibration.json --output runtime\evaluations\independent_assessment_01
```

Calibration uses only supported calibration images, fits a positive scalar temperature by NLL, and records model/report/manifest identities and fit membership. Assessment uses disjoint specimens/source groups/images, reports raw/calibrated NLL, Brier sum and 10-bin ECE, category metrics and raw-score unknown-input metrics where supported and unknown examples coexist. Region subsets are reported separately where available. Unknown metrics here apply the pre-existing raw 0.60 / 0.15 score/margin policy only, **not** capture heuristics or tuned OOD thresholds. This is not equivalent to validating a learned OOD detector. Temperature never changes top-1 labels and is not automatically deployed. Independence still depends on trustworthy collection/provenance review.

Existing `calibrate_fabric.py` results are post-selection development diagnostics; they do **not** replace this new external stage. Independent category calibration also does not calibrate each new binary/categorical attribute head.

Assess trained attributes:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_attribute_heads.py --run runtime\training\fabric_attribute_heads_01 --manifest data\research_collection\manifest.jsonl --image-root data\research_collection\images --split assessment --output runtime\evaluations\attribute_assessment_01
```

Known labels only, unsupported known values counted as errors; results and region groups with no truth are explicitly unavailable. These are image-weighted metrics, not confidence intervals or specimen-weighted estimates; repeated views are not independent. Attribute confidence calibration and formal uncertainty/statistical analysis remain additional work.

## 5. Final evaluation — only after freezing, not now

Do not run final evaluation while models, thresholds, label definitions or calibration choices are still changing. Final stages require `--acknowledge-final-test`. The tool writes an exclusive attempt receipt **before final inference** under `runtime/evaluations/final_test_ledger/`, keyed by checkpoint and a sorted image-hash fingerprint. Repeated attempts with the same weights/data fail even if row order or manifest formatting changes. Interrupted attempts remain recorded; investigate failures and disclose any manually authorized recovery rather than deleting receipts to tune/retry silently.

- External category final: same category command, `--stage final-test`, frozen `--calibration`, new output and the acknowledgement flag.
- Reserved provider category final: `--stage provider-final-test` with the frozen external calibration/collection arguments plus acknowledgement. Only then does a dedicated evaluator perform inference on the provider's 393 reserved test images. It is **not** an independent Sri Lankan dataset; specimen/region uncertainty remains disclosed.
- External attribute final: same attribute command with `--split final_test` and acknowledgement. The provider dataset has no attribute truths, so cannot evaluate those heads.

All stages refuse output overwrite. Do not tune on final results or relabel/move samples after seeing outcomes. The ledger is a local workflow guard, not tamper-proof scientific governance. App deployment of new heads or temperatures is a separate reviewed step.

## Mechanics verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts\smoke_attribute_mechanics.py
.\.venv\Scripts\python.exe scripts\smoke_independent_mechanics.py
```

Artificial temporary fixtures test masked losses, unknown-head skipping, frozen encoder, reload and assessment plumbing only. The smoke uses a schema-real flag solely to exercise validation in a temporary fixture; its images, invented labels and metrics are explicitly not real collected data and are deleted afterwards. No final provider evaluation is run by tests or smoke.

Latest unit suite: **91 tests passed**. The independent smoke similarly exercises temperature freeze and supported/unknown assessment in a temporary directory, never real calibration or final evaluation. These scripts require the existing local V2 checkpoint and ML dependencies.

# Experimental fabric-category inference

Integrated 18 October 2026; now defaulting the UI dropdown to **Experimental ConvNeXt V2-Tiny** after the user's approval. The other selectable labels are **Experimental ConvNeXt-Tiny** and **Experimental EfficientNet-B0**. Review-only is no longer a UI option, but remains an API compatibility mode. Processing still requires permission and an explicit Analyse click. No fitted calibration temperature is activated.

Model IDs: `convnext_v2`, `convnext_v1`, `efficientnet_b0`. The upload API uses `mode: "experimental"` plus `model_id`; the backend accepts only these fixed IDs, validates the corresponding pinned architecture/report/weights and reloads the single-model cache on model changes. Missing/invalid selections do not fall back to a different classifier. The main release JSON supplies V2, `fabric_classifier_release.convnext_v1.json` supplies V1, and the preserved EfficientNet config supplies B0. Health exposes all three choices. Three-model HTTP switching/switch-back checks passed; see `runtime/evaluations/model_selection_smoke_01/report.json`.

## Selected release

`config/fabric_classifier_release.json` selects `runtime/training/convnextv2_nine_categories_01` and pins:

- Checkpoint SHA-256: `7a32c9ebb487d314ea148d51d8042577de0ac89bea2579055d01a55428243aa9`.
- Training report SHA-256: `0a29d03729b1bcef38088f54e56c6156c80daf2732bbd2f5738798b9fa72a23d`.
- Manifest SHA-256: `a4341ee939350b8b0b161b3d3bbf276137a1aa66965b42178d30ddfa13e9ec68`.

Checkpoint licence: **CC-BY-NC-4.0**, for the user's non-commercial university research; preserve source attribution/restrictions. Previous EfficientNet release metadata is retained in `config/fabric_classifier_release.efficientnet_rollback.json`. Only the main release JSON is active. Training reports remain immutable, including their historical `activated_in_app: false` fields.

The run directory must remain inside this project's `runtime/training`. Resolved checkpoint/report paths cannot escape the selected run, and architecture/category order must match the supported implementation. Missing files, changed hashes, invalid tensor files or model-load errors fail closed to manual review. Never edit frozen training reports to update application status; their `activated_in_app: false` fields record training/smoke-time status. The release configuration and current API health are the authority for current app selection.

## User flow

1. Start/restart the server, open `http://localhost:8002` and refresh the page.
2. Upload a permitted fabric close-up and confirm permission.
3. Select the **Experimental ConvNeXt V2-Tiny** option in Photo analysis mode. Health supplies the current model label.
4. Select **Analyse photograph**. The first call imports/loads the local model; subsequent calls reuse it.
5. Inspect the three category suggestions and relative scores. Optionally select **Use top category · still needs review**, or manually select another supported category/unknown.
6. Enter only known composition, technique and physical-property records. Applying a category does not populate these fields.
7. Enter measurements/style, review the complete form and request design ideas. Without recorded physical properties, these remain preference-based ideas rather than fabric-validated recommendations.

Mode/photo/permission changes clear old suggestions and the category selection and invalidate the review confirmation. Permission revocation clears the preview. Responses from older photo/mode versions are ignored. Design responses are similarly guarded against changed inputs. An already running server-side request is not cancelled by these UI guards, but its stale result is not applied.

## Interpretation

Default V2 categories: cotton, denim, leather, linen, nylon, polyester, silk, viscose, wool. Older V1/EfficientNet categories: cotton, cotton_mixed, denim, polyester, silk, viscose, wool. Health/inference responses expose each model's exact vocabulary and the UI follows it. Reviewed records accept the union for backwards compatibility. Categories are not verified fibre chemistry. Denim is not synonymous with cotton; cotton_mixed is not a specific blend composition. No material field is automatically filled.

Scores are **uncalibrated softmax values over the selected model's category set** (nine for V2, seven for V1/EfficientNet). A relative score is not measured accuracy or a guarantee that the fabric is supported. No validated OOD/unknown detector is available. The provisional 60% top-score / 15-point margin / capture-warning review heuristics disable quick-apply for uncertain results; they are not calibrated confidence thresholds. Unsupported fabric, non-fabric objects, unusual lighting and out-of-distribution photos can still receive confident incorrect labels; user review and unknown remain essential.

No production-method, print, weave, weight, drapability, stiffness or stretch prediction is made. The category alone adds no design-ranking points. Current V2 validation accuracy is **90.14% / macro-F1 0.8690** on a provisional 740-image development set, not final-test/real-world accuracy. See [nine-category results](NINE_CATEGORY_TRAINING.md). The earlier seven-class 72.47% score concerns different data/classes and is not directly comparable. Calibration diagnostics remain exploratory and are not deployed.

## Execution and privacy

- Inference is CPU float32, uses the exact recorded timm 224-pixel transform on the original EXIF-oriented RGB image and runs under `torch.inference_mode()` with frozen parameters/evaluation mode. Resolved preprocessing is checked against the report, including on warm-cache requests.
- A separate <=1000-pixel thumbnail is generated for display; it does not replace the original inference input.
- A single lock serializes model loading/inference. The model is cached, not the user photos. Release files are checked before every prediction, even when the model is warm; a changed or disabled release cannot silently use stale cached weights.
- Full model state is loaded from local safetensors; no pickle loading, external image API or model download is used during inference. Optional ML dependencies must already be installed.
- Health verifies configured files/dependency availability without importing/loading the model; successful inference is confirmed by actual requests, not health metadata alone.
- Uploads remain in process memory and browser preview state; no user photo/profile is deliberately written by the app. Developer smoke artifacts record their train-sample path and predictions only. There is no production authentication, resource isolation or retention/compliance guarantee.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts\smoke_fabric_api.py --output runtime\evaluations\your_new_smoke_directory
```

Activation verification: **77 tests passed**. `runtime/evaluations/convnextv2_active_api_smoke_01/report.json` confirms real V2 CPU HTTP inference, opt-in/default separation, top-three outputs, unchanged physical-property fields, repeated predictions, missing-release fallback with a warm cache, restoration and unchanged checkpoint checksum. The running Component 3 server was restarted and live health/upload requests confirmed V2. Only one training image was used; no final-test inference. The new model label is populated from API health; refreshed browser upload testing with the user's own examples remains to be done.

Browser verification used that same training sample: category suggestion applied only after explicit selection, composition/technique/properties stayed unknown, and a design request displayed the expected missing-property warning. Switching analysis mode cleared the suggestions, selected category and review checkbox. These verify mechanics, not accuracy or generalisation.

To disable optional inference, set `enabled` to false in the release config; health and analysis then return unavailable/manual mode. Keep hashes and immutable reports intact. Selecting a different trained run requires deliberately reviewed new checksums and release validation, not copying a random checkpoint over this one.

# First Sri Lankan drape-classifier pilot

Dataset: runtime/datasets/drape_reviewed_01. Researcher-confirmed folder labels and grouping; not independently expert-verified. The original image files remain untouched. The four identified duplicate/resized copies are excluded.

| Split | Nivi | Kandyan/Osariya |
|---|---:|---:|
| Train | 36 | 41 |
| Validation | 8 | 9 |
| Reserved test | 8 | 9 |

Run directory: runtime/adaptation/sri_lankan_drape_clip_01.

The method is a two-class linear drape head on frozen normalized CLIP ViT-B/32 image embeddings. It is not a fine-tuned image backbone. Training uses 100 epochs, seed 42, AdamW learning rate .02 and weight decay .01, selecting the epoch with lowest validation cross-entropy. Only drape_style is trained. garment_type=saree is metadata, while pattern/texture remain unknown and are not targets.

The comparison uses the same two candidate labels (Nivi and Kandyan/Osariya) for zero-shot CLIP and the trained head. It is not an end-to-end comparison with the application's four-label drape suggestions/non-saree applicability gate. A two-class head will force unfamiliar/unclear drapes into one of its known classes; open-set rejection has not been validated.

No test features or test predictions should be produced during development. The existing audit reads image bytes from all splits for integrity hashes/duplicate checks only. No final test-set performance is claimed. Small validation counts make results unstable, and clothing colours, image source, identity and background can remain confounded with the classes. User-confirmed grouping and exact-duplicate checks do not establish complete independence.

Training artifacts include the checkpoint, report, validation predictions and learning history. No automatic application activation is performed. Do not run the activation command until the trained result and scope have been reviewed. The existing application models, DINO pattern model and Component 1 remain unchanged.

## Completed development validation

| Metric | Matched two-label CLIP | Trained CLIP drape head |
|---|---:|---:|
| Correct / 17 | 10 | 16 |
| Accuracy | 58.82% | 94.12% |
| Macro-F1 | 0.5641 | 0.9412 |

The head selected epoch 100. It correctly predicted 8/8 Nivi and 8/9 Kandyan/Osariya examples, with one Osariya predicted as Nivi. These are validation results on researcher-supplied labels, not independent expert labels. The same validation split selected the checkpoint, and 17 examples are too few for a strong generalization claim. No significance claim or final-test result is made.

See report.json for exact checkpoint fingerprint, model revision, manifest SHA, per-class metrics and confusion matrix. A functional reload/Grad-CAM smoke test uses nivi/10.jpg, confirmed to belong to the training split, not the reserved test split. This check is not another accuracy evaluation.

Checkpoint reload, local drape prediction, CLIP fallback, real Grad-CAM and unchanged repeated predictions passed in the smoke script. All 70 automated tests pass. The checkpoint SHA is `140e32b965cfe668c43070906a831852fed2b7a4b388a951e4ae6937d812e44a`.

Operational note: the completed training process and completed smoke process remained alive after printing their saved/success results on Windows. They were closed by exact verified process ID after the checkpoint/report hash and success output were confirmed. No originals, model weights or results were deleted; this runtime shutdown behaviour was not diagnosed or claimed fixed. The model was unactivated at that initial handoff; later optional activation is described below.

## Optional application integration

Following the user's request, the saved drape head is now activated through config/local_attribute_release.json. The dropdown calls this mode **Sri Lankan drape + CLIP · experimental** (API value local_attributes). General CLIP remains the default. The optional configuration points to the existing frozen head; no retraining or checkpoint modification was performed.

In this mode only drape_style uses the trained head. Garment type, pattern and visible texture use General CLIP; the DINO pattern model is a separate selectable mode and is not combined here. Predictions are Nivi or Kandyan/Osariya only, and only when General CLIP predicts saree. This applicability gate can itself be wrong; manual attribute corrections do not automatically rerun the head. Users may select other/unknown manually for unsupported drapes. There is no validated automatic open-set rejection.

Choosing Drape style under Optional visual explanation differentiates the actual learned drape logit through CLIP, not the zero-shot text score, and displays that distinction in its notice. Non-saree applicability rules yield no heatmap. Model crop and overlay are 224×224, based on the coarse 7×7 CLIP patch grid; clothing/background attention is not evidence of causal or culturally correct reasoning.

Health checks verify the selected local model's metadata and checksum before enabling the option; an invalid/missing release is not advertised as usable. Existing modes remain available. Restart the backend and refresh the page after code/config changes.

The real-model HTTP integration check in scripts/smoke_drape_api.py passed on the confirmed training example nivi/10.jpg. It verifies the trained-head 7×7 Grad-CAM target, unchanged CLIP fallback predictions, stable repeated predictions and not_applicable/no-heatmap behaviour for a non-saree demo dress. Artifacts are under runtime/evaluations/drape_api_smoke_01. All 71 automated tests pass. No final-test examples were inferred or scored during this integration.

Browser verification also passed: selecting the optional mode produced a Nivi suggestion for the same training example, and Drape style Grad-CAM displayed the trained-head notice and crop/overlay. A screenshot is saved as drape_ui.png in the integration artifact folder. The initial online model load was slow; cached-only loading was used for the completed browser check. This was a temporary process environment setting, not a permanent offline configuration or a claimed startup fix. The verification server was stopped after testing, so the researcher can start the normal application command without a port conflict.

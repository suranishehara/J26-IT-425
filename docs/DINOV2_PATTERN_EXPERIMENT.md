# DINOv2-Small pattern comparison

This is a separate experiment, not a replacement for General CLIP, the IndoFashion classifier or an activated application model.

## Completed development run

Run: runtime/adaptation/dinov2_pattern_01. All 1,323 labelled training and 155 labelled validation crops match the previous CLIP experiment exactly. No test features were extracted. The head selected epoch 100 of 100.

| Method | Correct / 155 | Validation accuracy | Macro-F1 |
|---|---:|---:|---:|
| CLIP matched four-label zero-shot | 151 | 97.42% | 0.8809 |
| CLIP trained linear head | 152 | 98.06% | 0.8769 |
| DINOv2-Small trained linear head | 153 | 98.71% | 0.9056 |

DINO improves both metrics on this development split, but only by one correct example over the CLIP head. Its checked recall is 1/2 and striped recall is 13/14; floral is 5/5 and solid is 134/134. These minority supports are too small for strong generalization or statistical-significance claims. Validation was used for selection and is not final-test evidence. No additional hyperparameter search was performed in this comparison.

On this RTX 4050 CUDA run, feature extraction for 1,478 train/validation crops took 24.03 seconds (61.50 images/second) at batch size 16. Peak PyTorch-allocated CUDA memory was 194,828,800 bytes; this is not total device/driver/process memory. CLIP timing was not captured, so this is not a cross-model speed comparison.

The saved model reload and repeated inference passed on a synthetic demonstration garment (functional check only). The checkpoint SHA is `a0e13141f2025b4aa1e62da6ac211ba8d5a6735d2326a8b0148beda518c9207a`. The CLIP-head checkpoint SHA remains unchanged. DINO is now selected in a separate optional application-mode configuration; it does not replace the default model. The CLIP local-head experiment remains unactivated.

## Model and method

Encoder: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), pinned revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. The existing Transformers version supports this architecture; no package upgrades are required. Safetensors weights are loaded without executing remote repository code. The encoder is frozen.

The last normalized CLS token provides 384 features. L2-normalized features feed a four-class linear head for solid, striped, checked and floral. This is linear probing, not full-backbone fine-tuning. It is not a zero-shot language/image model and does not provide CLIP's text-label matching capability. Other garment attributes are outside this experiment.

The native pretrained image processor uses shortest-edge resize 256, centre crop 224×224 and ImageNet normalization, giving 16×16 ViT patches (patch size 14). It differs from CLIP's native preprocessing; this is an encoder-plus-native-preprocessing comparison, not an isolated architecture ablation. Exact preprocessing is stored in the report.

## Controlled development comparison

- Dataset: runtime/datasets/deepfashion_pattern_development_01.
- 1,323 known-pattern training crops; 155 known-pattern validation crops.
- Frozen source image/group/split hashes and exact feature sample IDs/order are checked against runtime/adaptation/deepfashion_pattern_01.
- Four candidate labels are identical. Baseline prediction IDs and target labels must match the full validation set exactly.
- Same seed, 100-epoch budget, linear head, AdamW learning rate .02 and weight decay .01. Each head selects its lowest-validation-loss epoch.
- No test features, test predictions or final-test metrics are produced. Image hashes are checked for integrity by the existing audit.
- Validation has already been used in development and is used for model/checkpoint selection. It is not an independent final benchmark. The tiny checked/floral counts and source-label noise remain limitations.

## Reproduce (new output directory)

```powershell
.\.venv\Scripts\python.exe scripts\train_dinov2_pattern.py --manifest runtime\datasets\deepfashion_pattern_development_01\manifest.jsonl --image-root runtime\datasets\deepfashion_pattern_development_01 --clip-run runtime\adaptation\deepfashion_pattern_01 --output runtime\adaptation\dinov2_pattern_02 --batch-size 16
```

The report includes per-class precision/recall/F1, macro-F1, accuracy and confusion matrices for matched-label zero-shot CLIP, the previous CLIP linear head and DINO's linear head. It stores checkpoint, taxonomy and CLIP-report fingerprints, training history, validation predictions and frozen source records. New runs never overwrite old outputs.

DINO extraction throughput and peak allocated CUDA memory are measured on this machine. CLIP timing was not captured in the original run, so these measurements cannot establish that DINO is faster or slower. Model loading/download time is outside the extraction timing. CPU/GPU runtime and batch size are recorded.

## Inference smoke test

```powershell
.\.venv\Scripts\python.exe scripts\predict_dinov2_pattern.py --run runtime\adaptation\dinov2_pattern_01 --image frontend\assets\demo\dress.png
```

This verifies checkpoint/model/taxonomy fingerprints, reloads the model and tests repeated pattern prediction. A synthetic demo image is only a functional check, not an accuracy evaluation. Relative softmax scores are not calibrated confidence.

## Optional application mode

The web application now offers **DINOv2 patterns + CLIP · experimental** as an optional hybrid. General CLIP remains the default. The frozen DINO checkpoint is selected separately in config/dinov2_pattern_release.json and checked against its model revision, taxonomy, labels and SHA before loading. A missing/mismatched run disables the UI option; API failures do not silently substitute a different pattern model.

In this mode, pattern_type uses DINOv2; garment_type, drape_style and fabric_texture use General CLIP, not the IndoFashion garment head. Pattern labels are limited to solid, striped, checked and floral, although reviewers may manually correct to the wider taxonomy. A mixed/multiple-garment image is not segmented by this mode: it uses the uploaded image crop, whereas training used segmented single-region crops. This distribution difference remains a limitation.

Pattern Grad-CAM differentiates the DINO head logit through the real DINO encoder at encoder.layer[-1].norm1, giving 16×16 patches on the actual 224×224 crop. Non-pattern targets still run CLIP Grad-CAM and identify CLIP in the displayed result. The generic non-saree drape rule still has no heatmap. No explanation is presented as measured confidence or proof of correct reasoning.

Restart the Component 2 backend and refresh the browser after these changes. In Garment catalogue choose the hybrid, analyse an image, then select Pattern under Optional visual explanation. The saved checkpoints have not been retrained for integration, and no test evaluation was performed. The CLIP activation script deliberately does not accept DINO artifacts.

Integration verification: 65 unit/API tests pass. scripts/smoke_dinov2_api.py uses a temporary HTTP server to check DINO pattern inference, a real 16×16 DINO Grad-CAM, unchanged CLIP fallback predictions, repeated-prediction stability and gradient/hook cleanup. The browser path was also verified on a separate temporary port, without stopping the user's existing server or saving a catalogue record. Outputs are under runtime/evaluations/dinov2_hybrid_smoke_01. These functional checks use a synthetic demo image, not test-split images or an additional accuracy evaluation.

Unit tests cover exact validation alignment, truth/candidate consistency, duplicate predictions and exclusion of test/unknown records. Actual feature extraction/training and checkpoint reload are verified separately by the development run and inference smoke test.

# ConvNeXt V2-Tiny research candidate

## Source and boundaries

Model: `timm/convnextv2_tiny.fcmae_ft_in22k_in1k`, pinned repository revision `b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817`. Provider safetensors SHA-256: `6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9` (114,561,694 bytes). The downloaded file was independently checked against that value.

The [provider model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k) describes FCMAE pretraining followed by ImageNet-22K/1K fine-tuning and marks the checkpoint **CC-BY-NC-4.0**. Preserve attribution and the non-commercial restriction; this experiment is for the user's stated non-commercial university research. Broader pretraining differs from the older candidates, so results do not isolate architectural improvements.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
.\.venv\Scripts\python.exe scripts\train_convnextv2.py
```

CUDA Torch/Torchvision must come from the previously documented CUDA wheel source. `timm==1.0.30` and its added dependencies are pinned in the ML lock file. The original Torch/Torchvision versions were preserved; `pip check` passed. Changes are isolated to Component 3. Review-only operation does not need timm.

Default output: `runtime/training/convnextv2_tiny_pilot_01`; existing output directories are rejected. Use a new `--output` when repeating. Download/cache location: `runtime/models/huggingface`, pinned revision and SHA; no executable model repository code is trusted or loaded. Inference constructs the known timm architecture without downloads and loads only the saved local safetensors state.

## Training protocol

- Existing audited manifest: 5,043 train / 356 validation images, identical seven-category order and manifest checksum to previous experiments. The development loader refuses test images and verifies source-file hashes and group exclusions.
- Pretrained normalized pooled 768-D features plus a newly initialized seven-class linear head. Use the installed model's standard `resolve_model_data_config` transform for every train/validation image: **224 pixels**, bicubic resize/crop with crop fraction 0.875, ImageNet normalization. The provider also documents a 288-pixel test setting; this pilot does **not** use that alternative setting or tune crop settings on validation results.
- Frozen feature extraction and head training are FP32. Head: 100 full-matrix weighted-cross-entropy AdamW epochs, learning rate 0.02, weight decay 0.01; seed 42, select validation macro-F1 and retain first epoch in ties.
- Then five AMP-float16 training epochs: last ConvNeXt stage and final head normalization LR 0.0001; linear head LR 0.001; batch size 16, AdamW weight decay 0.01, gradient clip 1.0, eval-mode stochastic depth. Earlier stages/stem remain frozen. No extra augmentation is introduced.
- The selected frozen-head checkpoint remains eligible as epoch zero. Check protected tensors unchanged, save the selected full model, reload it and repeat FP32 full-network validation. Prediction equality is required. No test inference or automatic app release changes.

## Artifacts and status

Completed 18 October 2026. Frozen-head selection: epoch 98/100, **63.20% validation accuracy / macro-F1 0.6384**. The subsequent final-stage run selected epoch **5/5**, **72.47% accuracy / macro-F1 0.7313**, with 258/356 correct predictions.

| Candidate | Validation accuracy | Macro-F1 |
| --- | --- | --- |
| EfficientNet-B0, partially fine-tuned | 68.54% | 0.6881 |
| ConvNeXt V1-Tiny, partially fine-tuned | 69.38% | 0.6993 |
| ConvNeXt V2-Tiny, partially fine-tuned | 72.47% | 0.7313 |

V2 has 11 more correct validation images than V1 and 14 more than EfficientNet; no statistical superiority or independent-test claim is made. Validation class F1 remains uneven (polyester 0.6078, cotton mixed 0.6364 versus wool 0.8736), so the result does not justify calling all seven classes reliably solved.

Selected checkpoint SHA-256: `3eaf7f2f044e18561fba860a71a4814555ab10f677a5dba463a5607c1e91bf86`. Protected-state equality and full-network saved-checkpoint FP32 reload predictions passed. Training/extraction/evaluation took 278.90 seconds after model loading on the RTX 4050 Laptop GPU; peak allocated GPU memory 635,779,584 bytes. The app release configuration remains unchanged.

CPU smoke: `runtime/evaluations/convnextv2_candidate_smoke_01/report.json`. On one training photo, cold checks/load/inference took 0.758 seconds and warm requests 0.183 / 0.184 seconds. Repeat predictions and unchanged model/config checks passed. These single-image times are not a representative latency benchmark. Latest suite: **77 tests passed**, package consistency and JavaScript syntax checks passed.

Development calibration diagnostic: `runtime/evaluations/convnextv2_calibration_01/report.json`. With the same deterministic 249-image fit / 107-image assessment group partition as older candidates, temperature 2.0372503 changed assessment NLL **1.4341 → 0.9831**, Brier sum **0.4837 → 0.4421**, and 10-bin ECE **0.1861 → 0.0700**; assessment accuracy stayed 71.03%. The model had already been selected using all validation images, so this is post-selection exploratory evidence, **not independent calibrated accuracy**. No temperature is activated; raw-score review heuristics remain unchanged. Actual external unknown-input evaluation is still pending real permitted labelled photos.

An experiment is complete only when `report.json` exists. That report contains full per-class metrics, confusion matrix, comparison checkpoint identities, training history, preprocessing, package/hardware versions and hashes. Additional artifacts: `frozen_model.safetensors`, `features.safetensors`, `progress.json`, selected `model.safetensors` and `validation_logits.safetensors`.

The opt-in inference loader supports the candidate architecture, checks report/model hashes and verifies its resolved preprocessing against the training report. This capability alone does not activate it; `config/fabric_classifier_release.json` remains the separate release choice. Tests use random fixtures and must not be reported as fabric accuracy evidence.

Validation-based selection and repeated experiments can overfit the development set. Source labels and specimen independence remain provisional. The seven categories do not add batik/handloom, weave or print prediction; physical properties/composition are not certified from a photograph. Independent calibration, external unknown-input evaluation and the reserved final test remain separate tasks. See [the collection/evaluation guide](CALIBRATION_AND_UNKNOWN.md).

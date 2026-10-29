# TextileClass-7 audit and EfficientNet-B0 baseline

## Dataset and original split

User downloaded the [TextileClass-7 v2 dataset](https://data.mendeley.com/datasets/m2h53zr39s/2), approximately 57.9 MB, and extracted it to:

`runtime/datasets/raw/Fabric Classification.v2`

Its included README declares CC BY 4.0, 224×224 stretched preprocessing and three augmented versions per training source. The seven provider categories are retained: cotton, cotton_mixed, denim, polyester, silk, viscose and wool. Denim is a fabric category, not fibre composition; cotton_mixed does not identify an exact blend. No batik, handloom, weave, print, GSM or drape labels are created from these category names.

| Split | Raw images | Conservative pilot images |
| --- | ---: | ---: |
| Train | 5,499 | 5,043 |
| Validation | 392 | 356 |
| Test | 393 | 393 — reserved, no inference |
| Total | 6,284 | 5,792 |

## Audit findings

`runtime/evaluations/textileclass_audit_01/report.json` records:

- All 6,284 files decode; no integrity errors.
- No exact decoded-pixel duplicates were found. This does not rule out recompressed or transformed duplicates.
- 39 original-filename groups span splits; 78 original-filename groups span labels. Filename reuse is ambiguous, not proof that all members are the same original image or specimen.
- 81 cross-split dHash candidates at distance <=4: 33 train/test, 43 train/validation and 5 validation/test. A dHash match is similarity triage, not identity proof, especially for plain fabrics and repeating patterns.
- A train/validation-only contact sheet showed both false-positive filename/similarity matches and visually similar image pairs, including a pair carrying different labels. No test image content was visually reviewed for model development.

The raw dataset remains unchanged. Grouping links same original filename prefixes, exact decoded hashes and the dHash candidates. To err on the side of avoiding leakage, uncertain train/validation records are excluded rather than relabelled. In grouped overlaps, test has priority over validation and train; validation has priority over train. Test membership is never removed or relabelled by preparation.

492 train/validation images are excluded: 456 for mixed-label/filename-collision groups, 15 for possible test relations and 21 for possible validation relations. `excluded.json` records the reason per image. This deliberately over-excludes some unrelated examples; it does not claim that all exclusions were proven duplicates.

Accepted derived manifest: `runtime/datasets/textileclass_pilot_02/manifest.jsonl`.

Manifest SHA-256: `3c888298dd97a636cb336c878e064700fa39f0e441fd02e3a5a8b8a4aa9ebbca`.

`textileclass_pilot_01` is an unused preparation attempt with a Windows newline/checksum mismatch. Training rejected it before opening training images. The generator was fixed to write canonical LF newlines and a new version was prepared; do not use pilot_01 for training.

## What this preparation cannot establish

Physical specimen/source-session identifiers are not provided. Related views may still exist under different filenames; many augmented training images are not independent samples. Labels are provider annotations, not independently verified fibre tests. Therefore the pilot is **not a specimen-independent scientific benchmark**, even after conservative checks. Validation performance can be optimistic and cannot establish real phone-photo or Sri Lankan fabric performance.

Test files were decoded/hashed for integrity screening, but no test features, classifications or accuracy are computed. Reserved does not mean completely untouched at the file level. Known test grouping/label caveats must be disclosed in any final evaluation; do not repeatedly tune on test outcomes.

## Model and training method

The selected first model is **Torchvision EfficientNet-B0, ImageNet1K V1**—not CLIP, DINOv2, SegFormer or the earlier unimplemented ResNet plan. Its [official weights and preprocessing](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html) define resize256, centre224 and ImageNet normalisation. The source images were already stretched by the provider; this step cannot recover that lost aspect ratio.

1. Freeze the convolutional backbone and its batch-normalisation statistics in evaluation mode.
2. Extract 1,280-dimensional pooled features from **train and validation only**; L2-normalise them. CUDA autocast is used during extraction when available.
3. Train a seven-output linear head with weighted cross-entropy and AdamW: learning rate 0.02, weight decay 0.01, seed 42, full training-feature batch, 100 epochs.
4. Select the epoch by validation macro-F1, keeping the first best epoch on ties.
5. Save the full backbone + selected head in `model.safetensors`, feature cache, validation predictions, history and provenance/checksums in a new run directory.

This is a **frozen-feature transfer-learning baseline**, not full backbone fine-tuning. Fine-tuning and the proposed ConvNeXt-Tiny comparison are later experiments, selected on validation only. The current script intentionally has no final-test evaluation mode and does not activate a model in the web application.

## Commands

From Component 3 in PowerShell:

```powershell
# GPU packages are optional for the web-only prototype; already installed locally.
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install safetensors

# Existing audit/preparation outputs are immutable: choose new output paths to rerun.
.\.venv\Scripts\python.exe scripts\audit_textileclass.py --output runtime\evaluations\textileclass_audit_02
.\.venv\Scripts\python.exe scripts\prepare_textileclass.py --audit runtime\evaluations\textileclass_audit_02 --output runtime\datasets\textileclass_pilot_03
.\.venv\Scripts\python.exe scripts\train_fabric_baseline.py --dataset runtime\datasets\textileclass_pilot_02 --output runtime\training\efficientnet_b0_pilot_02
.\.venv\Scripts\python.exe scripts\smoke_fabric_checkpoint.py --dataset runtime\datasets\textileclass_pilot_02 --run runtime\training\efficientnet_b0_pilot_01
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The training command above uses accepted pilot_02 for comparability; substitute a newly audited/prepared manifest only deliberately. Scripts refuse to overwrite existing run/audit outputs. Pretrained downloads are cached under `runtime/models/torch` unless `TORCH_HOME` is already set. Packages are installed only in Component 3's environment.

## Recorded pilot results — 18 October 2026

Accepted run: `runtime/training/efficientnet_b0_pilot_01`.

- Validation: 219/356 correct, accuracy **61.52%**, macro-F1 **0.6156**, balanced accuracy **62.46%**.
- Selected epoch: 59/100 by validation macro-F1.
- Device: NVIDIA GeForce RTX 4050 Laptop GPU, Torch 2.14.1+cu126, Torchvision 0.29.1+cu126.
- Recorded extraction/head-training duration: 33.83 seconds **after model loading**, not total startup/download time.
- Peak PyTorch allocated GPU memory: 203,780,096 bytes, not total VRAM use.
- Full checkpoint SHA-256: `7fd6c21b3d8e9d035de0818af17c8391e0b6c5817b633bac21c7de8fccc75112`.
- Test evaluated: false; test images opened by training/smoke: zero.
- All 27 automated tests passed. Checkpoint reload, finite output, repeatability and unchanged-weight checks passed on one train image; its prediction was incorrect, underscoring that a successful smoke check is not an accuracy claim.
- Application activation: **not performed**. The browser still requires manual label review and exposes classifier-unavailable status.

| Provider class | Validation support | Recall | F1 |
| --- | ---: | ---: | ---: |
| cotton | 59 | 0.6949 | 0.6508 |
| cotton_mixed | 58 | 0.3103 | 0.3711 |
| denim | 50 | 0.8200 | 0.8200 |
| polyester | 53 | 0.4906 | 0.4906 |
| silk | 43 | 0.6744 | 0.6517 |
| viscose | 48 | 0.6042 | 0.5800 |
| wool | 45 | 0.7778 | 0.7447 |

See the saved run's `report.json` for predictions/history, provenance and actual validation results. Do not infer success merely from the presence of a partially created run folder. No full fine-tuning or ConvNeXt comparison has been performed yet.

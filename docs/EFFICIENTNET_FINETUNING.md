# Controlled EfficientNet-B0 development comparison

This experiment extends the frozen-feature baseline; it is not a new final-test evaluation. Components 1 and 2 are unchanged.

## Validation error review

The frozen-head baseline made 137 errors among 356 validation images. The most frequent directed confusions were cotton_mixed → cotton (11), viscose → polyester (9), polyester → viscose (8) and cotton_mixed → silk (8).

`runtime/evaluations/efficientnet_validation_review_01` contains the per-confusion report and the 24 highest relative-score errors as a contact sheet. Relative scores are not calibrated confidence. The sheet includes garment/product labels, print differences, shiny surfaces and near-plain fabric views. Those visible factors may become shortcuts; an image alone does not justify correcting fibre composition labels. No validation labels or splits were altered after inspecting predictions.

Physical specimen independence remains unverified. The conservative image-group pilot retains the same 5,043 train / 356 validation / 393 reserved test images documented in [the baseline record](TEXTILECLASS_BASELINE.md). No additional exclusions or relabelling were made to boost validation performance.

## Fixed experiment setup

- Initialise from `runtime/training/efficientnet_b0_pilot_01/model.safetensors`, validating its SHA-256 and manifest identity.
- Same seven provider classes, deterministic ImageNet V1 preprocessing and seed 42.
- Update final two feature stages (indices 7 and 8) and the seven-class head; freeze earlier stages.
- Keep model in evaluation mode **while enabling gradients on selected weights**. This intentionally fixes batch-normalisation running statistics and disables stochastic depth; evaluation mode does not disable gradients.
- Weighted cross-entropy based only on training counts.
- AdamW: feature-stage learning rate 0.0001, head learning rate 0.001, weight decay 0.01; gradient norm clipping at 1.
- Five epochs, batch size 32, shuffled training images, CUDA mixed precision and gradient scaling. No additional random image augmentation beyond provider-exported train variants.
- Evaluate baseline and every candidate on the same validation images using full-network **float32** inference.
- Retain the highest validation macro-F1 checkpoint, including the untouched baseline as epoch 0 if no epoch improves it. Ties keep the earlier checkpoint.
- Verify earlier feature weights and all batch-normalisation running buffers remain bitwise unchanged.
- Never instantiate a final-test image loader; the shared development loader refuses test rows.

The architecture identifier inherited from the initial run contains `frozen_l2_linear`; it identifies the saved network layout. The report's `method` and training fields describe the actual partially fine-tuned weights. It does not mean the fine-tuned run kept its final stages frozen.

## Fair comparison caveat

The original frozen baseline report used mixed-precision feature extraction and cached features. Its recorded validation accuracy is 61.52% (219/356) and macro-F1 0.6156. Recomputing the same checkpoint with the common full-network float32 evaluator gives 61.24% (218/356), macro-F1 0.6129. This one-image numerical difference is disclosed rather than treated as a model change.

Use the fine-tuning report's `baseline_validation_recomputed` for the controlled comparison. Neither outcome is independent test accuracy or proof of real-world improvement.

## Commands

```powershell
# New outputs only; existing reports/checkpoints are preserved.
.\.venv\Scripts\python.exe scripts\review_validation_errors.py --output runtime\evaluations\efficientnet_validation_review_02
.\.venv\Scripts\python.exe scripts\finetune_fabric.py --output runtime\training\efficientnet_b0_finetune_02
.\.venv\Scripts\python.exe scripts\smoke_fabric_checkpoint.py --run runtime\training\efficientnet_b0_finetune_01
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Review `runtime/training/efficientnet_b0_finetune_01/report.json` only after successful completion. During training, `progress.json` records completed epochs; its existence does not prove the final checkpoint was saved.

## Completed results — 18 October 2026

Accepted candidate: `runtime/training/efficientnet_b0_finetune_01`.

| Condition, common full-network float32 evaluator | Correct / validation N | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Frozen-head baseline | 218/356 | 61.24% | 0.6129 | 62.22% |
| Selected fine-tuned model, epoch 4 | 244/356 | 68.54% | 0.6881 | 68.98% |

Observed validation accuracy increased by 7.30 percentage points and macro-F1 by 0.0752. This is selection on one development split, not a statistical significance claim or independent test result. Epoch 5 reached 67.98% accuracy / 0.6866 macro-F1, so epoch 4 was restored.

Per-class candidate F1: cotton 0.6977; cotton_mixed 0.5607; denim 0.8387; polyester 0.6154; silk 0.6517; viscose 0.6526; wool 0.8000. Cotton-mixed remains difficult despite improvement. Denim recall declined from 0.82 to 0.78 while its precision and F1 improved; not every individual metric increased.

- Run duration: 126.75 seconds after setup/model load, including repeated development evaluations.
- Peak PyTorch allocated GPU memory: 403,831,296 bytes, not total process VRAM.
- Checkpoint SHA-256: `8550a8b49bc911e83d55179cb0e67a239dd2e01dc308b6a1d8f779c15bf6b73a`.
- Frozen weights and running-statistic checks: passed.
- Saved-model checksum/reload/finite-output/repeatability smoke: passed on one train image, without changing model weights.
- Automated suite: **30 tests passed**, including gradient-update boundaries, fixed statistics, test-loader refusal, manifest checksums and cross-split overlap rejection.
- No test predictions or test accuracy computed. All 393 test images remain reserved.
- No automatic application activation. The web app still reports classifier unavailable.

Comparison summary: `runtime/evaluations/efficientnet_finetune_comparison_01/report.json`, generated with `scripts/compare_fabric_runs.py`. The original baseline and all checkpoints are preserved. Seeds and fixed settings support reproducibility, but bitwise equality across hardware/software versions is not promised.

The checkpoint smoke uses one **training** image for reload/checksum/repeatability verification, not final-test accuracy. Deployment requires separately tested, explicit experimental configuration, reviewable category suggestions (not confirmed composition) and honest uncertainty/out-of-scope handling. Neither this dataset nor this experiment can train batik/handloom, print, weave or physical-property estimators without the appropriate labels.

# ConvNeXt-Tiny development comparison

This experiment is separate from the app's active EfficientNet-B0 classifier. It never edits the release configuration or opens final-test images.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\train_convnext.py
```

Default output: `runtime/training/convnext_tiny_pilot_01`. Existing outputs are rejected rather than overwritten. Pretrained weights are downloaded into Component 3's `runtime/models/torch`, not into another component. CUDA is required for this pilot.

## Protocol

- Reuse the checksum-verified `textileclass_pilot_02` manifest and its 5,043 train / 356 validation images. The development loader refuses test images and checks file hashes and group boundaries.
- Torchvision ConvNeXt-Tiny ImageNet1K V1 weights; official resize236 / centre224 bilinear transform and ImageNet normalization. See [official model documentation](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html).
- Frozen backbone and pretrained final normalization; L2-normalized 768-dimensional features plus a seven-class linear head. Evaluation mode throughout feature extraction.
- Seed 42; FP32 features, training and validation. Weighted cross entropy, AdamW learning rate 0.02 / weight decay 0.01, 100 full-feature-matrix head epochs. Select the highest validation macro-F1; first epoch wins ties.
- Save full weights in safetensors, reload and repeat full-network validation. Require identical class predictions to cached-feature evaluation. Save validation logits for later calibration development, without claiming they are independent calibration data.
- Report per-class metrics/confusion matrix, weight/manifest hashes, software/hardware, elapsed time and GPU allocation. Compare against EfficientNet's recomputed FP32 frozen baseline and partially fine-tuned candidate.

Architecture-specific preprocessing and training budgets differ. This is a practical candidate comparison, not a matched-compute architecture superiority study. Repeated validation selection may overfit the development split. Dataset specimen independence remains unverified. No test accuracy, calibration, OOD rejection or real-world composition-certification claim is justified by this experiment.

## Status

Completed 18 October 2026. The selected head epoch was 86/100, with **59.27% validation accuracy and macro-F1 0.5998** on 356 images. Saved full-network FP32 reload predictions matched cached-feature predictions. All 64 automated tests passed. The final test split was not opened for inference.

| Candidate | Validation accuracy | Macro-F1 |
| --- | --- | --- |
| EfficientNet-B0 frozen head, FP32 re-evaluation | 61.24% | 0.6129 |
| EfficientNet-B0 partially fine-tuned | 68.54% | 0.6881 |
| ConvNeXt-Tiny frozen head, FP32 | 59.27% | 0.5998 |

The ConvNeXt frozen candidate did not justify replacement of the active EfficientNet candidate. No release configuration was modified during that experiment.

## Subsequent controlled fine-tuning

`scripts/finetune_convnext.py` completed five epochs on the same manifest. Final two feature modules, final normalization and head were updated; earlier stages and stochastic-depth behavior remained fixed. Learning rates: backbone/normalization 0.0001, head 0.001, AdamW weight decay 0.01; batch size 16, seed 42, AMP training and FP32 evaluation. Epoch zero remained eligible. Selected epoch **4** achieved **69.38% accuracy / macro-F1 0.6993**. Epoch 5 fell to 66.57%, and was not selected.

Frozen-stage equality and checkpoint-reload predictions passed. Runtime was 274.34 seconds on the RTX 4050 Laptop GPU; peak allocated GPU memory 518,556,672 bytes. Detailed results: `runtime/training/convnext_tiny_finetune_01/report.json`. Checkpoint SHA-256: `aa0cd052667b718e5854039a0098e71d0292cfbcc31efcc4986227f750dc766e`.

The gain over EfficientNet (68.54%, macro-F1 0.6881) is about **0.84 percentage points / three additional correct validation images**. It is not statistical superiority or independent generalization evidence. The app remains on EfficientNet pending independent evidence/release choice; the inference loader now supports checksum-pinned candidates of either architecture without downloading weights.

One-train-image CPU candidate smoke passed: cold load/check/inference 0.656 seconds, warm requests 0.222 and 0.211 seconds on this machine. This is mechanics/indicative latency, not a representative benchmark. See `runtime/evaluations/convnext_candidate_smoke_01/report.json`. The temporary smoke configuration never changed the app release.

Development-only calibration pilots were completed for both models; independent calibration and actual external unknown-input evaluation remain outstanding. See [calibration and unknown-input evidence](CALIBRATION_AND_UNKNOWN.md). Final test inference has not been performed. Latest automated suite: **74 tests passed**.

Detailed artifact: `runtime/training/convnext_tiny_pilot_01/report.json`. Associated full weights, cached features and validation logits are saved beside the report. Dataset, weights and report metadata are reproducible from the command above with a new output directory. Consult the report for exact hashes/hardware/timing rather than treating rounded figures here as a separate measurement.

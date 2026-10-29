# Garment prediction explanations

In Garment catalogue, upload an image, expand **Optional visual explanation**, choose an attribute, and click **Generate Grad-CAM**. A separate forward/gradient pass explains the selected model's current predicted label. It does not explain a manually corrected catalogue label and does not train or change weights.

Supported targets:

- General CLIP: garment type, pattern, drape style and visible texture similarity logits.
- IndoFashion mode: trained four-class garment-type logit; other attributes still use CLIP similarity logits.
- A non-saree `not_applicable` drape label is produced by a rule, so no model heatmap is invented for it.

## Method

ViT Grad-CAM uses activations at `vision_model.encoder.layers[-1].layer_norm1`, immediately before the last attention block. The class token is excluded from the spatial map. Patch gradients are averaged across spatial tokens to form channel weights; their weighted activation sum is rectified, normalized and upsampled. CLIP ViT-B/32 gives a coarse 7×7 patch grid at 224×224 input. This choice follows the [pytorch-grad-cam ViT guidance](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md).

The UI shows the actual normalized-then-reconstructed model crop beside the overlay. It does not stretch a centre-crop heatmap over the original full photograph. Orange/red indicates stronger positive attribution within this one map; colours are not comparable confidence levels across images. Uniform/no-positive or non-finite maps are handled explicitly.

This is not garment segmentation, a causal explanation, proof of correctness or expert validation. A heatmap can expose background reliance but cannot establish that the model learned culturally meaningful concepts. Person pose/cheek overlays are geometric/sampling visualizations, **not Grad-CAM**; person skin-tone estimation is a colour heuristic and has no neural skin-tone classifier to explain.

## API

POST `/api/analyse-garment` with the usual image and classifier_mode plus:

```json
{"include_explanation":true,"explanation_attribute":"garment_type"}
```

The existing analysis response gains an `explanation` object with target metadata, PNG data URLs and limitations. Default requests do no gradient pass. Images are processed in memory and not persisted by this endpoint. Requests share the existing classifier lock; gradients are obtained with `autograd.grad` without populating parameter `.grad` buffers. Hooks are removed even on failure.

## Verification

`tests/test_explanations.py` covers class-token exclusion, shape checks, normalization, non-finite handling and empty gradients. `scripts/smoke_explanations.py` exercises both real model paths on a synthetic demo image, verifies nonempty maps and unchanged repeated predictions. This is a software smoke test, not an attribution-faithfulness or accuracy evaluation.

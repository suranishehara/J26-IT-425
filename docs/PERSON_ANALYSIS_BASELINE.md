# Opt-in person-analysis baseline — 16 October 2026

## Implemented
The Your profile tab accepts a person photo and explicit permission to process it locally. POST /api/analyse-person returns an annotated overlay, pose-based proportions, sampled cheek colour and an uncalibrated nearest Monk reference. No uploaded photo or inference result is persisted to disk. Outputs remain in browser memory until the page is refreshed/cleared.

Models:
- MediaPipe Pose Landmarker Lite, official float16 version 1 bundle, CPU. https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- SegFormer B2 clothes/human parsing checkpoint, mattmdjaga/segformer_b2_clothes, revision 584abc1e1d260e23c0fc627c5217a09b2b461046, CPU. https://huggingface.co/mattmdjaga/segformer_b2_clothes
- Monk reference HEX values verified against Google's published component: https://skintone.google/assets/tono-color-block-arranger.73c427e1.js

Downloaded/copied assets and their SHA256 hashes are recorded in runtime/models/person/assets.json. Component 1 was not modified; its existing segmentation checkpoint was copied into Component 2 to avoid a redundant download. scripts/setup_person_models.py supports a fresh installation through official download locations.

## Methods and limits
Body output is a ratio of projected shoulder-joint span to hip-joint span. Visibility and framing gates can return unavailable. It is not a body contour measurement, physical circumference, clothing size, or a validated body-shape category. Single-image perspective, pose and clothing affect the estimate. User-entered centimetre/inch measurements remain separate and are never overwritten.

Face parsing locates the largest high-score face region. Two interior cheek patches are sampled to reduce interference from hair, eyes and mouth. Median RGB is compared with the ten reference colours in CIELAB using Euclidean distance. The nearest reference and alternatives are not calibrated confidence or a verified person's skin tone. Illumination, camera white balance, makeup and parsing errors affect results. Simple exposure/variation warnings are unvalidated heuristics. No ethnicity or Fitzpatrick type is inferred.

The overlay shows sampled regions and landmarks, not Grad-CAM. It explains the input regions used by these calculations but does not establish accuracy or fairness. Grad-CAM remains a separate planned model-explanation task.

Users can select another Monk reference, edit/clear the projected ratio, or leave both unknown. Apply profile requires explicit review when either is supplied. Reviewed fields are carried with the profile but do not yet contribute recommendation scores. Profile/photo metadata is not automatically converted into research ground truth.

## Verification
- Actual local HTTP inference returned 200 on a synthetic Component 1 person photo.
- Pose ratio and a tentative Monk suggestion were returned; no real-person reference labels were available.
- This is functional verification only, not body/skin-tone accuracy or fairness evaluation.
- Unit checks cover colour conversion, missing-face abstention, visibility gating, reference matching and review requirements.

## Run and test
```powershell
.\.venv\Scripts\python.exe scripts\setup_person_models.py
.\.venv\Scripts\python.exe scripts\smoke_person_analysis.py
.\.venv\Scripts\python.exe backend\app.py
```
Open Your profile, upload a photo, confirm permission and click Analyse person photo. Review green cheek patches/yellow landmarks, inspect and correct the estimates, confirm review, then Apply profile.

## Research still required
Obtain consented, representative local images and independently reviewed skin-tone reference labels. Separate identities between development/test sets. Test varied lighting and report error, abstention/coverage, segmentation quality and performance across adequately represented skin-tone groups. Validate body proportions against an appropriate reference before using them for fit or body-shape claims. Local recalibration and recommendation usefulness must be evaluated separately. Current garment-classification test scores do not apply to these modules.

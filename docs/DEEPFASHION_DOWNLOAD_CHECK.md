# Actual download check and pilot

The raw dataset is correctly installed at runtime/datasets/raw/deepfashion_multimodal.

- images: 44,096 JPEG files.
- segm: 12,701 indexed parsing PNGs, with `_segm.png` filenames.
- labels/texture/pattern_ann.txt: pattern annotation file.
- labels/texture/fabric_ann.txt: fabric annotation file.
- labels/shape/shape_anno_all.txt: shape annotations, unused in this import.

Palette-index masks and JPEG dimensions match for the inspected example (750×1101). The importer was adjusted to support the real mask suffix explicitly. Original images, masks and annotation files remain unchanged.

## First import

Output: runtime/datasets/deepfashion_pattern_pilot_01.

From 500 reproducibly selected raw images, 81 upper-clothing crops were imported. Skips: 361 missing masks (the parsing subset is smaller than the image dataset), 31 without compatible labels and 27 missing/small or dress/romper regions. Processing errors: zero.

Splits: 63 train, 9 validation, 9 test. Pattern labels across all splits: 67 solid, 6 striped, 2 checked, 1 floral, 5 unknown. Texture: 7 knitted, 74 unknown. These totals do not establish sufficient class counts within each split. This pilot must not be used for four-class training as-is.

The manifest records original provider annotations, not a new human review. Permission remains unconfirmed because the researcher has not yet explicitly recorded eligibility under the research agreement. No checkpoint was trained, activated or evaluated. The crop/label quality and source grouping still require inspection; exact-duplicate and split audits are separate checks.

## Next

1. Researcher checks non-commercial research eligibility and records the provider agreement/reference. Do not mark permission confirmed without doing so.
2. Prepare a larger region-aware sample with sufficient floral, striped, checked and solid counts in train/validation/test. Keep garment groups intact; do not move final test examples to fix training counts.
3. Inspect representative crop-label pairs and run the shared manifest audit, including duplicate/group checks.
4. Train a pattern head and compare with matched-label CLIP. This does not train Sri Lankan draping or a complete visible-texture classifier.

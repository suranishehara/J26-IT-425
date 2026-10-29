# DeepFashion-MultiModal preparation

The importer is implemented against the [official provider schema](https://github.com/yumingj/DeepFashion-MultiModal). Actual archive layout and annotation filenames must be checked after extraction. Original data is never modified. The importer does not download, extract archives, accept agreements, train or activate a model.

The installed download has now been inspected: images/ contains 44,096 JPEGs, segm/ contains 12,701 palette-index masks named IMAGE_STEM_segm.png, and labels/texture/ contains pattern_ann.txt and fabric_ann.txt. Use --mask-suffix _segm for these files. The first real 500-image pilot produced 81 crops with no processing failures; see DEEPFASHION_DOWNLOAD_CHECK.md. It is too small/imbalanced for four-class training. No model has been trained from this import.

## Files needed

- Extracted JPEG image folder.
- Extracted parsing PNG folder with integer label IDs. Not coloured previews.
- Pattern annotation TXT: filename, upper, lower, outer codes. The provider calls these **color annotations**, though the codes describe patterns.
- Optional fabric annotation TXT: same three clothing-region columns.

Use the exact image/mask roots and annotation TXT paths after inspecting your files. The importer does not guess annotation filenames, archive layout, count/header lines or label types. A schema mismatch raises a useful error for inspection.

## Mapping

| Source pattern code | Our label |
|---|---|
| 0 floral | floral |
| 2 striped | striped |
| 3 pure color | solid |
| 4 lattice | checked |
| 1 graphic, 5 other, 6 color block, 7 NA | unknown |

Only fabric code 4 knitted maps directly to visible texture knitted. Denim, cotton, leather, furry, chiffon, other and NA remain unknown. They are not relabelled as shine/matte/weave classes. A single knitted class is insufficient to train a multi-class texture head. All imported garment types and drape styles remain unknown: this dataset does not provide our Sri Lankan drape labels.

Masks define upper=top ID1, lower=skirt/pants/leggings IDs3/5/6, outer=ID2. Crops use the matching regional annotation only, replace unrelated pixels with neutral background and retain the garment bounding box. Dress/romper images are skipped because their upper/lower annotation scope is ambiguous. Small/missing regions, RGB masks, orientation problems and size mismatches are skipped/reported rather than forced. This uses only the subset with available parsing; it does not cover all 44,096 images.

## Pilot command (replace actual TXT paths)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_deepfashion_multimodal.py --images "runtime\datasets\raw\deepfashion_multimodal\image" --parsing "runtime\datasets\raw\deepfashion_multimodal\parsing" --pattern-labels "PATH_TO_ACTUAL_COLOR_ANNOTATION.txt" --output runtime\datasets\deepfashion_pattern_pilot_01 --max-images 500 --regions upper --assign-group-splits
```

This inspects at most 500 raw images in a reproducible seeded order; fewer crops may be eligible. It is not a class-balanced selection. The limit bounds pilot runtime/disk use. Parsing is required; the importer will not attach upper labels to an entire multi-garment person image if parsing is missing.

The optional `--fabric-labels "ACTUAL_FABRIC_ANNOTATION.txt"` retains compatible knitted labels. Names must match pattern-file image keys exactly.

The default command leaves permission unconfirmed. Only after **you** check your eligibility under the provider's research agreement, add `--confirm-permission --permission-reference "Your recorded dataset agreement/source reference"` using a new output directory. These options record the user's assertion; they do not accept terms on your behalf or verify usage rights. Do not invent a permission reference.

## Splits and provenance

Without --assign-group-splits all records remain unassigned. With it, seeded SHA-based 80/10/10 splits are assigned by garment group, keeping regional crops and views of one garment together. These are **project-created splits**, not an official benchmark split. Group IDs preserve the category/gender prefix through id_NUMBER. Unrecognized filenames are reported instead of assigned an unsafe per-image group. Identity/source overlap and near-duplicates still require review.

Exact duplicate derived crops are skipped. The shared training audit additionally checks source groups and decoded-pixel duplicates. Inspect import_report.json (skips, failures, counts, annotation hashes) and representative crops before training. Check class counts by split; a small pilot might not meet minimum counts.

Records use review_status=provider_annotated, an original-provider annotator attribution and explicit mapping provenance. They are **not described as independently reviewed by you**. The audit accepts documented provider annotations separately from manually reviewed annotations and still requires permission confirmation. Reports preserve annotation-status counts.

After inspection, audit/train **pattern only** with --attributes pattern_type, --manifest pointing to the derived manifest.jsonl and --image-root pointing to its containing folder. See LOCAL_ATTRIBUTE_TRAINING.md. Do not include the texture head until multiple properly labelled visible-texture classes exist. Do not call a DeepFashion pattern model a locally validated Sri Lankan multi-attribute model.

## Usage limits

The provider states non-commercial research only and prohibits further distribution except internal copies within one organization/site. Keep raw files, crops and dataset-containing review packets local; do not commit them or publish them with a report. Cite Text2Human and DeepFashion. See the provider's agreement for complete conditions.

## Current verification

Unit tests cover mappings, grouping, split determinism, mask alignment, garment-region isolation, malformed labels and a generated end-to-end importer fixture. The actual downloaded dataset has not yet been inspected or imported. No model training or research evaluation has been performed on it yet.

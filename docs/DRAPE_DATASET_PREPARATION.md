# Drape dataset preparation — 2026-10-18

Prepared output: runtime/datasets/drape_proposed_01/manifest.jsonl.

There are 111 retained candidate images after excluding four exact/resized photograph copies from the original 115. No original image was deleted, renamed or moved. The proposed split is:

| Split | Nivi | Kandyan/Osariya | Total |
|---|---:|---:|---:|
| Train | 36 | 41 | 77 |
| Validation | 8 | 9 | 17 |
| Test | 8 | 9 | 17 |

This is a reproducible stratified group-split proposal, not a claim that 111 distinct physical garments or independent sources have been established. Exact/same-photo groups are in config/drape_candidate_groups.json; related_garment_groups can hold additional views of one physical garment or a conservatively grouped source collection. These groups must be resolved before accepting splits. Source/model/background overlap remains a possible confound even after garment grouping.

Folder labels remain user-proposed. The manifest explicitly records review_status=unreviewed and grouping_status=provisional. The training audit rejects provisional groups; no training or test evaluation was performed. The user's authorization assertion is recorded, while missing per-image source/license references remain unspecified rather than invented. A separate needs_review directory is not needed.

## Required human confirmation

The researcher should check that retained images are correctly labelled Nivi or Kandyan/Osariya and identify any remaining related garment views. Existing contact sheets are under runtime/datasets/drape_inventory_02. This check occurs before final split acceptance; do not later use final-test images to tune a model. It is not independent expert validation unless such a reviewer actually performs it.

If there are additional related images, add their paths to related_garment_groups and regenerate a new proposed output; no original movement is necessary. Once the researcher explicitly confirms category labels and source grouping, a new manifest may be created with --confirm-folder-labels and --confirm-source-groups. These flags must not be supplied merely to bypass the checks.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_drape_dataset.py --input data\local_attributes\drape_candidates\real --groups config\drape_candidate_groups.json --output runtime\datasets\drape_reviewed_01 --confirm-folder-labels --confirm-source-groups
```

Then run the shared training audit with --attributes drape_style, the resulting manifest and --image-root data/local_attributes/drape_candidates/real. garment_type=saree is metadata supporting the drape task; a one-class garment-type head must not be trained. Pattern and texture remain unknown. The final trained drape head will be a two-class closed-set model, not a general recognizer for all drapes or non-sarees.

Seventy automated tests pass, including reproducible splits, related-view grouping, duplicate exclusions and cross-label conflict rejection. These are software checks, not drape classification performance.

## User confirmation and accepted dataset

On 2026-10-18 the user confirmed that each folder contains its named category only, then answered "No" when asked whether any additional images show the same physical saree in different poses/views beyond the identified duplicate copies. Those statements were recorded as researcher confirmation, not independent expert/physical-garment verification.

Accepted manifest: runtime/datasets/drape_reviewed_01/manifest.jsonl. Counts remain 77 train, 17 validation, 17 reserved test. The shared audit passed with no exact duplicate or cross-split source-group overlap detected. This does not rule out identity/source/background confounding or missed near duplicates. The 17 test crops have not been used for feature extraction, training or final performance reporting.

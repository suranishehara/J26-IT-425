# Candidate nine-category replacement — data audit, not model support yet

The user subsequently requested removal of `cotton_mixed` from the next classifier. The planned inventory is **cotton, denim, leather, linen, nylon, polyester, silk, viscose, wool**. It is recorded separately in `config/fabric_category_expansion.json`. Existing seven-output checkpoints still retain their original seven-label mapping until a newly trained nine-output candidate is evaluated and activated. Do not suppress, shift or rename old logits as a substitute for retraining. The original dataset and reserved test artifacts remain unchanged.

The researcher confirmed ownership/use permission and labels for the linen, nylon and genuine-leather images on 18 October 2026. These are recorded as **user assertions**, not independently verified licences/composition. All source images remain in their original folders, and the app's seven-category checkpoints/release/taxonomy remain unchanged.

Audit artifacts: `runtime/evaluations/added_categories_audit_01/report.json` and `inventory.json`. Reproduce with `scripts/audit_added_categories.py`, using a new `--output` directory.

| Folder label | Readable images | Filename families |
| --- | --- | --- |
| linen | 76 | 19 |
| nylon | 228 | 57 |
| leather | 64 | 16 |

All 368 images decoded successfully. Filename suffixes `_im_1`, `_im_2`, etc. were conservatively grouped by prefix; these 92 families are not certified physical-specimen counts. Related views must stay together, with any known shared acquisition/source groups also kept in one split.

Four exact byte/decoded duplicate pairs were detected between `Leather/leather_1959_im_1.png` and `Leather/leather_1960_im_1.png`, likewise suffixes 2, 3 and 4. No files were deleted. A training manifest can retain one copy per pair and link both original families so no duplicate crosses splits. This would leave 60 distinct leather photos before other triage—not 64 independent observations.

No exact/decoded hash overlaps with the original recorded provider inventory were detected. There are 1,336 dHash-distance<=4 similarity candidates: 712 between added images (224 cross-label), and 624 against provider images. These are **visual review candidates, not duplicate proof**; uniform/repeated textures can produce false positives. Do not delete or relabel them automatically, or merge every texture collision into a supposedly verified specimen.

Before nine-category training: review similarity candidates, preserve original reserved test membership, deduplicate via manifest exclusions only, group related sample/source views, balance training for unequal class counts and create a new immutable nine-category manifest/report/model excluding cotton_mixed. Existing seven-output weights are not nine-category classifiers; never edit their output mapping. Plan an additional unknown-input set because linen/nylon/leather will cease to be unknown once supported. Their images cannot serve simultaneously as independent unknown-rejection examples. A cotton blend is not automatically an unknown sample unless its composition is documented and it is outside the agreed nine-category definition.

At this audit milestone no split/training had occurred. Subsequent group-aware preparation is now recorded in `NINE_CATEGORY_PREPARATION.md`: 4,912 retained photos, fixed 3,432/740/740 train/validation/test membership and train-only class weights. A separate nine-output model still needs training/evaluation; no existing release or original reserved test artifacts were changed.

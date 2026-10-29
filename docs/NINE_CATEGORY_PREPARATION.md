# Nine-category data preparation

Prepared from the user-supplied collection on 18 October 2026. The new scope excludes cotton_mixed; batik/handloom, pattern, weave and texture prediction are **not supported yet**, not extra category labels or negative labels. Existing seven-output checkpoints and their mappings stay unchanged until a new model is trained/evaluated.

## Immutable artifacts

`runtime/datasets/fabric_nine_categories_01/`:

- `manifest.jsonl`: retained rows with fixed labels, group IDs, split membership and file/decoded hashes.
- `preparation.json`: manifest/scope hashes, counts, grouping rules, source assertions and train-only loss weights.
- `inventory.json`: all 4,980 source-image records, including excluded copies.
- `excluded.json`: 68 duplicate-copy exclusions with retained-image references; no images deleted.
- `similarity_triage.json`: all perceptual similarity candidates, not automatic duplicate verdicts.

Source root remains `data/research_collection/images/`. Do not move images after freezing this manifest. The labels are cotton, denim, leather, linen, nylon, polyester, silk, viscose and wool. New linen/nylon/leather paths still contain `unsupported_fabrics` for historical collection organization; the prepared manifest supplies their candidate labels. A directory name alone does not alter the active classifier or define production technique.

## Grouping and duplicates

Link `_im_1`, `_im_2`, etc. by filename family, then union any families connected by identical file bytes or decoded pixels. Exclude entire linked groups on conflicting labels or exact relation to the recorded provider inventory; none were needed for those reasons in this preparation. Keep one lexicographically selected copy per identical decoded image. All remaining views/linked families share a split. Original provider train/validation/reserved-test artifacts are not modified or used as inputs to the new manifest.

This gives 4,912 retained images in 1,228 inferred groups. These are **filename/content-derived groups**, not certified physical specimens or acquisition batches. Missing true source/capture identities can still cause leakage. User supplied label/ownership assurances; composition, genuine leather and permission assertions are not independently verified by the code.

## Fixed split and class counts

Seed 42. Hash-sort group IDs within each category; reserve approximately 15% of groups for test and 15% for validation, leaving approximately 70% for training. Selection is fixed before model training, not tuned to scores. All categories appear in all three splits.

| Category | Train | Validation | Reserved test | Train loss weight |
| --- | ---: | ---: | ---: | ---: |
| Cotton | 1,624 | 348 | 348 | 0.2348 |
| Denim | 452 | 96 | 96 | 0.8437 |
| Leather | 44 | 8 | 8 | 8.6667 |
| Linen | 52 | 12 | 12 | 7.3333 |
| Nylon | 156 | 36 | 36 | 2.4444 |
| Polyester | 628 | 132 | 132 | 0.6072 |
| Silk | 136 | 32 | 32 | 2.8039 |
| Viscose | 100 | 24 | 24 | 3.8133 |
| Wool | 240 | 52 | 52 | 1.5889 |
| **Total** | **3,432** | **740** | **740** | |

Leather has only 2 inferred groups / 8 images in each evaluation split; linen has 3 / 12. Results for these classes will have substantial sampling limitations. Multiple photos of one sample are not independent evidence.

## Imbalance strategy

Weighted cross entropy: `weight[class] = N_train / (9 × n_train[class])`, derived only from the training partition. No image copying, no simultaneous weighted sampler plus weighted loss, and no rebalancing validation/test. The upcoming trainer must consume the recorded weights in the exact nine-label output order. `backend.category_dataset.read_prepared` verifies the manifest hash, group exclusivity, unique decoded content, class coverage and train-only weight arithmetic. Its development loader refuses reserved test rows and verifies each source image before decoding.

## Similarity triage and interpretation

64-bit dHash distance <=4 yielded **18,021** candidate pairs outside the immediate same-filename/exact-duplicate exclusions; **8,850** retained pairs cross the fixed splits. Plain/repeated fabric texture can create false positives; these numbers are **not counts of duplicate photos**. They are saved for visual/source review, without deleting/relabeling/group-merging based only on the score. The split status remains provisional. Review likely related specimens/source batches and create a **new version** if verified relations require new grouping; never silently modify a frozen split after viewing model scores.

No model training or final-test inference was performed by preparation. A subsequent provisional training experiment is recorded in `NINE_CATEGORY_TRAINING.md`, separate from this immutable preparation. No near-duplicate independence or production accuracy claim is made. New unknown-input evaluation examples must be outside the agreed nine categories; production techniques/visual attributes are not unknown-material classes.

## Reproduce and verify

```powershell
.\.venv\Scripts\python.exe scripts\prepare_nine_categories.py --output runtime\datasets\fabric_nine_categories_NEW_VERSION
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Outputs refuse overwrite. The default completed version is `fabric_nine_categories_01`; do not delete or overwrite it to reshuffle outcomes. Latest suite: **99 tests passed**, covering deterministic group splits, linked-family duplicate exclusion, conflict/provider exclusions, training-only weights and refusal of test data by the development loader.

# After the dataset finishes downloading

The project's current download is already imported at runtime/datasets/indofashion_clean_01/manifest.jsonl. See INDOFASHION_BASELINE.md for the audit and first validation results. The original strict audit detected cross-split duplicates; a derived manifest was generated using --deduplicate. Use that cleaned manifest for subsequent work.

Do not move or unpack files while the download is active. Keep the provider's terms and instructions.
Raw images and provider annotations belong under a local ignored directory such as runtime/datasets/raw/indofashion.
Extract archives with their original folders intact.

## Inspect and import
From Component 2, replace the placeholder with the folder containing provider annotation JSON:
```powershell
.\.venv\Scripts\python.exe scripts\prepare_indofashion.py --dataset "C:\path\to\indofashion" --output runtime\datasets\indofashion_import_01
```
If image_path values are relative to another directory, add --images-root "C:\path\to\image-root".
The importer expects source image_path and class_label fields. It handles official train_data.json/val_data.json/test_data.json files with one JSON record per line, a list of records, a dictionary of records, or records nested under data/images/samples.
Other formats require inspecting the extracted files before changing the importer.

The importer maps source saree/blouse/men-kurta/women-kurta/gown names into saree/blouse/kurta/dress.
Other source categories are excluded and counted, not silently treated as supported classes.
Numeric label IDs require a provider class mapping; the importer does not guess.
Original train/validation/test assignments are retained. Exact duplicates across splits and missing mapped images block the manifest.
Content hashes do not detect different photographs of the same physical garment.
Review import_report.json before evaluating. Imported metadata and images are not added to the shopper catalogue automatically.

## Run a validation pilot first
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_garments.py --manifest runtime\datasets\indofashion_import_01\manifest.jsonl --split validation --per-class 25 --output runtime\evaluations\clip_validation_01
```
This evaluates up to 25 unique images per mapped class, with seed 42. It produces predictions.jsonl and report.json.
Accuracy, per-class precision/recall/F1, macro-F1 and confusion counts are calculated from provider category labels.
Inaccessible/corrupt images are reported and also counted wrong in the all-selected accuracy.
Other attributes are predicted but not scored because IndoFashion category labels do not establish texture or local draping ground truth.

Tune using training/validation data. Reserve the original test split for the final fixed comparison.
Model revision and taxonomy are recorded. Existing manifests/reports are preserved by requiring new output directories.
Raw data, downloaded weights and evaluation records stay out of Git.

## What to show next
After extraction, share the dataset folder path and the names of its top-level files/folders (not credentials).
Then verify the actual schema and run the importer before launching a long evaluation.

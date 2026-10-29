# Dataset preparation and future collection

Update, 18 October 2026: the user supplied TextileClass-7 v2. It has been audited and a conservative development manifest prepared; see [the recorded findings](TEXTILECLASS_BASELINE.md). No new Sri Lankan specimen dataset has been collected. The schema below remains a future multi-attribute collection plan, not a claim that all fields exist in the downloaded data.

Start with real specimens and a documented class inventory. Do not download random images and assign physical properties from filenames. Public accessibility alone does not establish permission.

## Required record dimensions

- Unique image ID and relative path; specimen ID shared across all views/crops of one physical fabric.
- Source URL/provider, permission/licence reference, permitted usage and reviewer.
- Real/synthetic flag, Sri Lankan/global/unknown provenance (not guessed from appearance).
- Material/composition label and supporting record; blends need explicit treatment.
- Production techniques as independent annotations; potentially multiple such as handloom and batik.
- Shared pattern_type and fabric_texture; weave separate (plain/twill/satin/other/unknown only after label validation).
- Unknown label/missingness explicitly recorded for each head rather than guessed.
- Per-specimen GSM, drapability and stretch records with method, units, repeats, reviewer and source. The current qualitative property names are provisional; approve definitions before collection.
- Versioned train/validation/test split with specimen/source-group boundaries. Never split crops of one sample across splits. Reserve final test for the frozen final model.

Example JSON is in `data/manifest.example.json`; it is a documentation template, not a record of a real sample or accepted training schema. A future importer must validate paths, permissions, labels, duplicates and leakage before training. Do not rename it to a real manifest and claim that one placeholder image constitutes a dataset.

## Capture guidance

Use consistent scale references, even lighting, multiple phone/camera/lighting conditions, clear close-ups and a draped observation view where suitable. Photograph the same labelled specimen across conditions, retaining its group ID. Keep personal faces/unnecessary metadata out of fabric close-ups. Record missing or uncertain facts as unknown. Separate real training data from synthetic test fixtures and generated demonstration images.

## What is needed next from the researcher

1. Which classes have real available samples and reliable labels?
2. Where did each image/specimen come from, and what use is permitted?
3. Can a textile expert review material/method/weave labels and property testing?
4. Can a pattern maker validate silhouette rules and ease ranges?

Choose sample targets after the inventory and pilot error analysis; a fixed minimum image count alone cannot guarantee useful accuracy. Balance specimens, not just near-identical photos.

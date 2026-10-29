# Garment attribute baseline

This milestone uses openai/clip-vit-base-patch32 as a zero-shot baseline.
It compares a photograph with English candidate descriptions. It has not been trained on our Sri Lankan garments.
Model card: https://huggingface.co/openai/clip-vit-base-patch32

## Fixed labels
See backend/taxonomy.py for the versioned labels and exact prompt templates.
- garment_type: shirt, blouse, kurta, dress, saree, trousers, skirt, other.
- pattern_type: solid, striped, checked, floral, geometric, abstract, other.
- drape_style: not_applicable, nivi, kandyan_osariya, other.
- fabric_texture: smooth_matte, shiny, coarse_woven, knitted, lace, other.

An Osariya is recorded as garment_type=saree and drape_style=kandyan_osariya.
Batik and handloom belong to independently sourced textile_techniques metadata; their production process is not proven from an image.
Texture labels describe visible surfaces, not verified fibre composition.

## Annotation records
Use data/annotation_example.json as the record structure. Enter unknown for insufficient evidence.
Record image source, permission/license and physical garment/source group. Use the same group for near-duplicate images of one garment.
Assign train/validation/test splits by garment/source group before adaptation; keep final test labels independent.
Record annotator and date; have knowledgeable reviewers resolve cultural draping labels.
Browser corrections can be saved with the reviewed catalogue form. Saved records include reviewer ID and timestamp; they are user-reviewed metadata, not independent research ground truth or automatic training labels.

## Baseline interpretation and evaluation
Relative softmax scores depend on the candidate list. They are not measured confidence or accuracy.
All suggestions require review. Ambiguity flags use an unvalidated 0.15 score-gap heuristic.
For non-saree predictions the UI uses not_applicable for drape style through a documented rule.
Out-of-domain rejection is not validated; the model can label unrelated images incorrectly.
Treat generated Component 1 assets as synthetic smoke tests, never as proof of generalization.
Evaluate each attribute independently against reviewed held-out labels using per-class precision, recall and macro-F1, with confusion matrices.
Compare the same taxonomy/splits against subsequent fine-tuning or linear-probe adaptation.

## Privacy and operation
The local /api/analyse-garment endpoint processes the uploaded garment image in memory and does not persist it. Explicitly saving through /api/catalogue persists a normalized PNG and reviewed attributes locally in SQLite.
First use downloads model weights from Hugging Face. Later requests reuse one model instance.
Do not use this garment classifier to infer ethnicity, skin tone, body measurements or other person attributes.

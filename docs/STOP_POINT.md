# Component 1 checkpoint — 16 October 2026

Implementation stopped at the user's request. Preserve this component when developing Component 2.

## Working implementation
- Local CatVTON try-on on RTX 4050; Standard 384x512/30 steps and Detailed 768x1024/50 steps.
- Three-column desktop interface: inputs/settings, editable mask, result.
- SegFormer automatic clothing masks. Upper-body category selects tops; one-piece dress bodice boundary is approximate.
- Dress, saree and Kandyan/Osariya select upper and lower clothes and fill the gap between trouser legs to allow continuous skirts.
- User confirmed masking and generation work after this correction.
- VITON-HD baseline previously ran on six official pilot pairs.
- Generated garment and person references are synthetic demonstration assets.
- Evaluation exporter scripts/export_evaluation_records.py creates metadata rows; human scores remain blank.
- Last recorded test run: 10 tests passed after the trouser-gap mask correction.

## Resume here
1. Preserve test outputs, review representative examples and document failures.
2. Improve evaluation workflow; exporter currently rewrites its output, so save scored copies separately.
3. Obtain permitted paired local data and valid reference labels.
4. Implement and evaluate actual traditional-garment adaptation.
5. Compare baseline and adaptation on held-out data and obtain independent draping/print/identity ratings.

## Limits
Functional tests do not establish scientific accuracy. No locally fine-tuned model or formal baseline-versus-adapted study is complete. Category selection is manual, not a trained classifier. CatVTON inference category/mask settings are not model training.

# Body-size compatibility — prototype rules

Catalogue entries may include optional size_chart records. These describe supplier-recommended BODY measurements, in cm; finished garment widths/circumferences are not accepted because ease and fabric stretch have not been modelled.

Each record needs a size label, cut, source/reference, basis=body_measurements and ranges for chest/waist/hips as appropriate. The browser supports one size variant per saved garment entry; the API supports up to 30 variants.

- Dress: chest, waist and hips are required.
- Shirt/blouse/kurta: chest and waist are required.
- Skirt/trousers: waist and hips are required.
- Saree: chest/waist check applies to the blouse only; drape, saree length and pleats are not assessed.
- Every supplied range must have the corresponding user-entered measurement.
- At least one complete matching range produces compatible; outside all fully assessable variants produces incompatible. Missing chart/measurements produces unknown.
- Photo-derived landmark ratios and skin-tone estimates are never used as physical measurements.
- Compatible adds 2 rule points; a compatible variant matching the preferred cut adds 1. These heuristic weights require recommendation evaluation.
- The optional Only size-compatible garments filter excludes unknown/unassessed records as well as incompatible records. Without it, other garments remain visible with status explanations.

No sizing values were invented for demonstration images. To test the workflow, add a permitted garment with an actual chart, or use clearly labelled fictional test values without treating them as validated product measurements.

Example test chart: M, regular/relaxed cut, chest 88–96 cm, waist 72–80 cm, hips 96–104 cm. Example profile 92/76/100 cm falls within these ranges. These are demonstration values only.

Body-chart compatibility is not a tailoring or fit guarantee. Stretch, ease, garment length, shoulder fit and individual comfort are not evaluated. Source references are user supplied; the application does not independently verify them.

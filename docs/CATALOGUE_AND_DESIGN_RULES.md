# Documented catalogue and reviewable design rules

Implemented 19 October 2026. These are implementation facilities, not a completed textile evidence collection or expert-validation study. No invented default fabric properties have been added to the real catalogue, and no rules have been marked expert-reviewed.

## Collect and reuse real samples

For each distinct physical sample, obtain its identity, known category/composition/technique, and documented physical properties. Keep unknown fields unknown. Obtain drapability/weight-band assessments using a consistent recorded protocol. Record actual GSM/stretch only when available, not inferred from a class or photo. Retain the source document or measurement note named by each property reference. The software checks fields and provenance presence, not the truth or accessibility of a reference.

Use section 02 to enter and check a record, then save it with a name and reviewer ID after review confirmation. Existing records remain immutable; corrections create another record. Selecting a saved sample is a user assertion that it is the intended sample, not an automatic photo match.

## Portable JSON catalogue

Expand **Import / export documented catalogue** in section 02. Export downloads saved labels, properties, reviewer/name and fabric hashes. It contains no photos or body profiles. Keep fictional test samples named `DEMO ONLY`; do not treat them as research evidence. A catalogue may contain private supplier references, so inspect it before sharing.

Imports need explicit source-review confirmation, at most 200 records, and a request below 500,000 bytes (including the JSON envelope). Every record must have at least one documented physical property. The whole batch validates before an atomic transaction. Existing samples are never replaced. Records with the same name, reviewer and normalized fabric hash are skipped; changed records append under new local IDs. Import validates any supplied fabric hash. Imported IDs/creation timestamps are newly assigned locally.

Shape (replace the placeholders with real documented information; this is not an import-ready textile record):

```json
{
  "schema_version": 1,
  "records": [{
    "name": "YOUR ACTUAL SAMPLE ID",
    "reviewer": "YOUR REVIEWER ID",
    "fabric": {
      "fabric_category": "unknown",
      "material": "unknown",
      "technique": "unknown",
      "pattern_type": "unknown",
      "fabric_texture": "unknown",
      "properties": {}
    }
  }]
}
```

Add at least one property inside `properties`, with keys `value`, `source`, `reference`. Supported property names: `drapability`, `weight_band`, `gsm`, `stretch_percent`. Sources: `user_measured`, `supplier_documented`, `expert_reviewed`. Each property may have its own source/reference; imported mixed sources remain separate. Valid drapability: `fluid`, `moderate`, `structured`; weight band: `light`, `medium`, `heavy`. GSM range: >0 to 2000; stretch: 0 to 500. Numerical range checks are not evidence of scientific plausibility or measurement validity.

API: `GET /api/catalogue/export`; `POST /api/catalogue/import` with `{ "reviewed": true, "catalogue": <the object above> }`. Imports are subject to same-origin checks and schema/provenance validation.

## Designer / pattern-maker review

Review `config/design_rules.json` with a suitably qualified collaborator. The four existing starter designs and their scoring/ease arithmetic were moved unchanged from code. The configuration exposes:

- Design IDs, names, style preferences, accepted drapability/weight bands.
- Positive circumference ease ranges for chest/waist/hips, and relaxed-fit extra ease.
- Style/silhouette/drapability/weight scoring weights.
- Rule version, review status, reviewer identity and document reference.

Ask the reviewer to verify intended wearer/garment context, ease and fit assumptions, compatibility with measured fabrics, exclusions, and whether more silhouettes are justified. Store their actual documented assessment and revise the version when changing rules. Do not mark `review.status` as `reviewed` until a genuine review occurred; this status records an assertion and is not certification or evidence of independent suitability evaluation. Current status is `pending`.

Run this read-only validator after edits:

```powershell
.\.venv\Scripts\python.exe scripts\check_design_rules.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Malformed rules, duplicate IDs, invalid enums, non-finite/negative weights/ease, reversed ease ranges and missing reviewed-status references are rejected. Defaults remain illustrative. Restart after configuration edits, refresh the UI, then rerun controlled design examples; changed IDs require refreshing silhouette choices. Each recommendation exposes rule version, SHA-256 and review metadata. A comparison uses one shared rules snapshot for both baseline and fabric-aware results. `GET /api/design-rules` exposes the validated configuration read-only. There is no browser endpoint for editing/review-approving rules.

Only documented drapability and weight band affect physical-property ranking. GSM/stretch are context, not automatic ease modifiers; category, composition, technique/pattern/texture are not proxy physical properties. No class-dependent physical defaults, negative-ease calculations or seam allowance are introduced. Independent suitability ratings and reserved classifier evaluation remain outstanding.

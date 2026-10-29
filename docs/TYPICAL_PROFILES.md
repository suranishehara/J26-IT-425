# Typical profiles for demonstrations

Added 19 October 2026. Explicitly selected variants autofill provisional ranking properties with a compact **Typical profile** label. They are not measurements of downloaded images, not a classifier output, not a population average, and not research ground truth. A product specification illustrates one construction only. It does not describe every photo in that category.

## Starter variants and source references

Supplier pages checked 19 October 2026:

| Profile | GSM | Drape mapping | Project weight band | Stretch default | Source |
|---|---:|---|---|---:|---|
| Cotton Oxford shirting | 194 | structured | medium | 0% | [Mood item 447586](https://www.moodfabrics.com/products/cotton-oxford-shirting-denim-blue-deadstock-447586) |
| Polyester chiffon | 70 | fluid | light | 0% | [Mood item MP5000-196](https://www.moodfabrics.com/products/netta-black-polyester-high-multi-chiffon-mp5000-196) |
| Rigid cotton denim | 432 | structured | heavy | 0% | [Mood item 331499](https://www.moodfabrics.com/products/structured-cotton-denim-dark-blue-deadstock-331499) |

Numbers are example supplier specifications, not typical-class means or measurements of a user's material. Supplier qualitative drape descriptions are mapped into the project's fluid/structured vocabulary; these are interpretation, not standardized drape-coefficient tests. For these presets only, the project uses light <150 GSM, medium 150–300, heavy >300 as a display convention, not a textile standard. Supplier no-stretch descriptions map to a demonstration default of 0%; no controlled-load stretch test or direction is supplied. Composition/production technique/pattern/texture remain unknown for the uploaded sample.

Three starter variants are available, not a preset for every classifier output. Other categories require a sourced variant or a documented sample. No catch-all class default is silently used.

## Flow

1. Analyse/apply a photo category if desired.
2. Choose a variant and click **Use typical fabric profile**. This is an explicit what-if choice, not automatic identification of the variant in the image. The variant's category replaces the editable category in the preview.
3. Form values become read-only and the source link is available beside the compact label. Review the profile, choose measurements/style, confirm and generate provisional ideas.
4. Choose **Return to manual properties** to restore the earlier manual draft. Selecting a saved specimen exits typical mode; changing/reanalysing a photo or model clears it.

The server allowlists profile IDs and reconstructs the properties, ignoring client property overrides. Requests carry `typical_profile_id` and explicit `reviewed: true`. Results include `property_basis: typical_profile`, and property source fields retain `typical_profile`. The baseline comparison removes the preset's properties for the baseline run, not silently reapplying them. Presets cannot be saved or imported into the documented specimen catalogue. Original image folders, trained models, reserved tests and the real catalogue are untouched.

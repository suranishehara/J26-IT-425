# Component 1 Research Protocol

## Research question

Does local adaptation of a pretrained 2D virtual try-on baseline improve visual quality and traditional-garment fidelity for Sri Lankan garments, compared with the unchanged baseline?

## Experimental conditions

1. **Baseline:** official pretrained VITON-HD, with no local adaptation.
2. **Adapted model:** the same architecture fine-tuned only on consented, labelled Sri Lankan training pairs.
3. **Holdout evaluation:** global and traditional test sets remain separate. Traditional test subjects and physical garments do not appear in training.

## Measures

- Visual quality, identity retention, and garment colour/print preservation: 1–5 blinded human ratings.
- Traditional only: pleat/front-fall and drape fidelity, plus cultural appropriateness: 1–5 blinded ratings.
- Report mean, standard deviation, sample count, and baseline-versus-adapted difference by garment family. Do not claim a statistically reliable improvement without an appropriate paired analysis and supervisor review.

## Reproducibility record

Record repository commit, model-weight checksum, input resolution, GPU, random seed, data split manifest, and every model parameter. Store outputs with pseudonymous sample IDs only.

## Constraints

VITON-HD is an upper-body, frontal-view baseline and cannot itself validate Kandyan saree or Osariya draping. Traditional adaptation therefore needs additional paired data and may require an architecture change; this limitation belongs in the final report.

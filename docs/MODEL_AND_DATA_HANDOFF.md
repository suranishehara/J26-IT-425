# Model and Data Handoff

## Replace the pilot renderer

`backend/services/tryon_service.py` is the model boundary. Keep its response contract, but submit the person image and catalogue garment image to a selected baseline (for example, a VITON-derived implementation) and return the stored output URL plus model/version metadata.

## Required data records

- Consented portrait image and a stable anonymised subject identifier.
- Garment product image, garment family, category, colour, print, fabric, and licence/source.
- For supervised traditional adaptation: paired person, garment, and reference try-on image; drape-style labels including Kandyan pleating or Osariya front-fall.

## Evaluation protocol

Compare the baseline and local adaptation using separate global and traditional test sets. Record human ratings for visual quality, garment colour/print preservation, identity retention, and traditional draping fidelity. Do not split near-identical fabric images or the same person across train and test sets.

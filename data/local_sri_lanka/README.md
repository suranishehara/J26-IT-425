# Sri Lankan Traditional Garment Pilot Dataset

This folder is a template only. Store identifiable images and signed consent documents in a secure location outside Git. Add only pseudonymous records and derived metadata to the research workspace.

## Collection unit

One row in `annotations.csv` represents one matched try-on example:

1. A clear, front-facing person photograph.
2. A clean image of the selected garment.
3. A reference photograph of the same person wearing that garment.

Use the same `sample_id` as each file name. Example: `SL001.jpg` in the three matching folders.

## Capture guidance

- Obtain written, informed consent before every image capture.
- Use an adult participant or follow your institution's formal child-consent process.
- Use front-facing, full/upper-body photos with a plain background and even light.
- Keep camera angle, distance, and image size consistent.
- Photograph the garment flat or on a neutral mannequin/background, without a person.
- Record unknown fields as `unknown`, never guess them.

## Split rule

Assign `train`, `validation`, or `test` before model training. A participant and a physical garment must occur in one split only. For a pilot, reserve roughly 20% of participants/garments for testing.

## Minimum viable pilot

Aim for at least 20 matched samples per garment type initially. Start with one type—such as batik dress or handloom kurta—before attempting Kandyan saree or Osariya, which require more specialised drape labels and may exceed the baseline model's upper-body design.

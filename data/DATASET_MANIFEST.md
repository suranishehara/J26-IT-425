# Dataset Manifest

Do not add faces, clothing photographs, VITON-HD assets, model weights, or other restricted material to Git.

## Baseline data

| Asset | Purpose | Status | Licence / access |
|---|---|---|---|
| VITON-HD | Baseline train/test structure and pretrained inference | Download manually | Research-only CC BY-NC 4.0 |
| VITON-HD checkpoints | `seg_final.pth`, `gmm_final.pth`, `alias_final.pth` | Download manually | Follow upstream repository terms |
| Dress Code | Optional broader global-garment comparison | Request separately | Signed institutional agreement required |

## Local Sri Lankan pilot data

Collect only with written consent. Give each item a pseudonymous ID. Store original consent forms separately from images.

```text
sample_id, subject_id, person_image_path, garment_image_path, reference_tryon_path,
garment_family, garment_type, drape_style, pleat_style, front_fall_style,
print_type, colour, fabric, source_licence, consent_version, split
```

`reference_tryon_path` is required for supervised fine-tuning and paired evaluation. Split data by subject and physical garment, never by near-duplicate photo.

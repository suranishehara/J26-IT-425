"""Summarise and visualise validation errors only; labels are not corrected."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/efficientnet_b0_pilot_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/efficientnet_validation_review_01'))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise ValueError('Use a new output directory.')
    rows = json.loads((Path(args.run) / 'validation_predictions.json').read_text())
    if any(not row['path'].startswith('valid/') for row in rows):
        raise ValueError('Only provider validation images may be reviewed.')
    prep = json.loads((Path(args.dataset) / 'preparation.json').read_text())
    raw = Path(prep['raw_root']).resolve()
    errors = sorted((r for r in rows if r['truth'] != r['prediction']), key=lambda r: (-r['relative_score'], r['path']))
    selected = errors[:24]
    sheet = Image.new('RGB', (1000, 235 * ((len(selected) + 3) // 4)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        path = (raw / row['path']).resolve()
        if not path.is_relative_to(raw): raise ValueError('Image outside raw root.')
        x, y = (i % 4) * 250, (i // 4) * 235
        with Image.open(path) as image:
            tile = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'), (220, 175))
            sheet.paste(tile, (x + 10, y + 5))
        draw.text((x + 8, y + 183), 'Label: ' + row['truth'], fill='black')
        draw.text((x + 8, y + 199), 'Pred: ' + row['prediction'], fill='black')
        draw.text((x + 8, y + 215), f"Relative score {row['relative_score']:.3f}", fill='black')
    output.mkdir(parents=True)
    sheet.save(output / 'high_score_errors.jpg')
    pairs = Counter((r['truth'], r['prediction']) for r in errors)
    report = {'validation_n': len(rows), 'errors': len(errors),
              'confusions': [{'truth': a, 'prediction': b, 'count': count} for (a, b), count in pairs.most_common()],
              'review_rows': selected, 'test_images_opened': 0,
              'notice': 'Highest relative-score errors, not calibrated confidence. Visual inspection cannot verify fibre composition; no relabelling performed.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('validation_n', 'errors', 'confusions')}, indent=2))


if __name__ == '__main__': main()

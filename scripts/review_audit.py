"""Train/validation-only contact sheet. Test image content is never opened."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'runtime/evaluations/textileclass_audit_01'
report = json.loads((folder / 'report.json').read_text())
raw = Path(report['raw_root'])
pairs = []
for group in report['groups']['source_key']['label_conflicts']:
    if any(path.startswith('test/') for path in group):
        continue
    a = group[0]
    b = next((path for path in group if path.split('/')[1] != a.split('/')[1]), None)
    if b:
        pairs.append((a, b, 'Same filename, different labels'))
    if len(pairs) == 3:
        break
for entry in report['near_duplicate_triage']['examples_first_300']:
    if not entry['a'].startswith('test/') and not entry['b'].startswith('test/'):
        pairs.append((entry['a'], entry['b'], 'dHash candidate distance ' + str(entry['distance'])))
    if len(pairs) >= 8:
        break
canvas = Image.new('RGB', (680, 220 * len(pairs)), 'white')
draw = ImageDraw.Draw(canvas)
for i, (a, b, title) in enumerate(pairs):
    for x, path in ((0, a), (340, b)):
        with Image.open(raw / path) as image:
            thumb = ImageOps.contain(image.convert('RGB'), (200, 165))
            canvas.paste(thumb, (x + 70, i * 220 + 25))
        draw.text((x + 5, i * 220 + 192), '/'.join(path.split('/')[:2]), fill='black')
    draw.text((5, i * 220 + 3), title, fill='black')
target = folder / 'train_validation_review.jpg'
canvas.save(target)
print(target)

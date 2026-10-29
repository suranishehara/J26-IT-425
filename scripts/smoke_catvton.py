"""One public benchmark pair for testing local model loading and GPU memory."""
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'runtime' / 'outputs' / 'catvton_smoke'
folder.mkdir(parents=True, exist_ok=True)
dataset = ROOT / 'external' / 'VITON-HD' / 'datasets' / 'test'
person_name, garment_name = (dataset.parent / 'test_pairs_pilot.txt').read_text().splitlines()[0].split()
Image.open(dataset / 'image' / person_name).convert('RGB').resize((384,512)).save(folder / 'person.png')
Image.open(dataset / 'cloth' / garment_name).convert('RGB').resize((384,512)).save(folder / 'garment.png')
parse = Image.open(dataset / 'image-parse' / person_name.replace('.jpg','.png')).resize((384,512), Image.Resampling.NEAREST)
labels = np.asarray(parse)
if labels.ndim != 2:
    raise ValueError('Expected indexed semantic labels, not an RGB parsing visualization.')
Image.fromarray((np.isin(labels, [5,6,7]) * 255).astype('uint8')).save(folder / 'mask.png')
(folder / 'request.json').write_text(json.dumps({'seed':42,'category':'upper-body'}))
subprocess.run([sys.executable, str(ROOT / 'scripts' / 'catvton_worker.py'), str(folder)], check=True)

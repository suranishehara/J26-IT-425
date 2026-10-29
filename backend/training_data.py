"""Verified development inputs; final-test images cannot be opened by this loader."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .fabric_model import preprocessing


def read_development(dataset):
    dataset = Path(dataset)
    prep = json.loads((dataset / 'preparation.json').read_text())
    content = (dataset / 'manifest.jsonl').read_bytes()
    if hashlib.sha256(content).hexdigest() != prep['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch.')
    rows = [json.loads(line) for line in content.decode().splitlines() if line]
    split_rows = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    seen = {}
    for split, values in split_rows.items():
        for row in values:
            previous = seen.setdefault(row['group_id'], split)
            if previous != split:
                raise ValueError('Cross-split group overlap.')
    for split in ('train', 'validation'):
        if set(r['label'] for r in split_rows[split]) != set(prep['labels']):
            raise ValueError('Missing development class.')
    return prep, split_rows


class DevelopmentImages(Dataset):
    def __init__(self, raw_root, rows, labels):
        if not rows or any(r['split'] not in ('train', 'validation') for r in rows):
            raise ValueError('Only non-empty train/validation input is allowed.')
        self.root = Path(raw_root).resolve()
        self.rows, self.labels = rows, labels
        self.transform = preprocessing()

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Final-test inference forbidden.')
        path = (self.root / row['path']).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Image path escapes raw root.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']:
            raise ValueError('Image checksum mismatch: ' + row['path'])
        with Image.open(path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return tensor, self.labels.index(row['label'])

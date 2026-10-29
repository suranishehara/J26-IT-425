"""Versioned category preparation, separate from legacy seven-output classifiers."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from PIL import Image, ImageOps
from .dataset_audit import dhash

FOLDERS = {'cotton': 'supported/cotton', 'denim': 'supported/denim', 'polyester': 'supported/polyester',
           'silk': 'supported/silk', 'viscose': 'supported/viscose', 'wool': 'supported/wool',
           'leather': 'unsupported_fabrics/Leather', 'linen': 'unsupported_fabrics/linen', 'nylon': 'unsupported_fabrics/Nylon'}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root, labels):
    root = Path(root).resolve(); rows = []; errors = []
    for label in labels:
        folder = root / FOLDERS[label]
        if not folder.is_dir(): errors.append({'path': str(folder), 'error': 'Missing folder'}); continue
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'): continue
            try:
                if path.stat().st_size > 6_000_000: raise ValueError('Image exceeds 6 MB.')
                with Image.open(path) as image: image.verify()
                with Image.open(path) as image:
                    if min(image.size) < 64 or image.width * image.height > 20_000_000: raise ValueError('Unsupported dimensions.')
                    image = ImageOps.exif_transpose(image).convert('RGB'); image.load()
                rows.append({'path': path.relative_to(root).as_posix(), 'label': label,
                    'source_key': re.sub(r'_im_\d+$', '', path.stem.lower()),
                    'file_sha256': digest(path), 'decoded_sha256': hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                    'dhash': dhash(image), 'width': image.width, 'height': image.height,
                    'label_source': 'user_confirmed_folder_category', 'permission_source': 'user_asserted_collection_permission'})
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append({'path': path.relative_to(root).as_posix(), 'error': str(error)})
    return rows, errors


def grouped_split(records, labels, seed=42, blocked_paths=()):
    if len({row['path'] for row in records}) != len(records): raise ValueError('Repeated inventory paths.')
    parent = {row['path']: row['path'] for row in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    def join(a, b): parent[find(b)] = find(a)
    for field in ('source_key', 'file_sha256', 'decoded_sha256'):
        seen = {}
        for row in records:
            if row[field] in seen: join(seen[row[field]], row['path'])
            else: seen[row[field]] = row['path']
    groups = defaultdict(list)
    for row in records: groups[find(row['path'])].append(row)
    by_label = defaultdict(list); excluded = []
    for members in groups.values():
        paths = sorted(row['path'] for row in members)
        group_id = hashlib.sha256('\n'.join(paths).encode()).hexdigest()
        if any(path in blocked_paths for path in paths):
            excluded.extend({**row, 'group_id': group_id, 'reason': 'family_has_exact_overlap_with_provider_inventory'} for row in members)
            continue
        if len({row['label'] for row in members}) != 1:
            excluded.extend({**row, 'group_id': group_id, 'reason': 'conflicting_labels_in_linked_family_or_exact_duplicate'} for row in members)
            continue
        seen = {}; retained = []
        for row in sorted(members, key=lambda item: item['path']):
            key = row['decoded_sha256']
            if key in seen:
                excluded.append({**row, 'group_id': group_id, 'reason': 'exact_duplicate_copy', 'retained_path': seen[key]})
            else:
                seen[key] = row['path']; retained.append({**row, 'group_id': group_id})
        by_label[members[0]['label']].append((group_id, retained))
    accepted = []
    for label in labels:
        members = sorted(by_label[label], key=lambda item: hashlib.sha256(f'category-split-{seed}:{label}:{item[0]}'.encode()).hexdigest())
        if len(members) < 3: raise ValueError('Need at least three distinct retained groups for ' + label)
        n_test = max(1, round(.15 * len(members))); n_val = max(1, round(.15 * len(members)))
        if n_test + n_val >= len(members): raise ValueError('Too few training groups for ' + label)
        for index, (_, rows) in enumerate(members):
            split = 'test' if index < n_test else 'validation' if index < n_test + n_val else 'train'
            accepted.extend(dict(row, split=split) for row in rows)
    return sorted(accepted, key=lambda row: row['path']), sorted(excluded, key=lambda row: row['path'])


def training_class_weights(rows, labels):
    counts = Counter(row['label'] for row in rows if row['split'] == 'train')
    if set(counts) != set(labels): raise ValueError('Training needs every category.')
    total = sum(counts.values())
    return {label: total / (len(labels) * counts[label]) for label in labels}


def read_prepared(folder):
    folder = Path(folder); info = json.loads((folder / 'preparation.json').read_text())
    raw = (folder / 'manifest.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != info['manifest_sha256']: raise ValueError('Prepared manifest changed.')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line]
    if len(info['labels']) != 9 or 'cotton_mixed' in info['labels']: raise ValueError('Not the approved nine-category dataset.')
    groups = {}; hashes = set()
    for row in rows:
        if row['split'] not in ('train', 'validation', 'test') or row['label'] not in info['labels']: raise ValueError('Invalid prepared membership.')
        if groups.setdefault(row['group_id'], row['split']) != row['split']: raise ValueError('Group crosses splits.')
        if row['decoded_sha256'] in hashes: raise ValueError('Decoded duplicate remains in manifest.')
        hashes.add(row['decoded_sha256'])
    if training_class_weights(rows, info['labels']) != info['training_class_weights']:
        raise ValueError('Training weights do not match train-only counts.')
    counts = {split: dict(Counter(r['label'] for r in rows if r['split'] == split)) for split in ('train', 'validation', 'test')}
    if counts != info['counts']: raise ValueError('Recorded split counts differ from manifest.')
    family_splits = {}
    for row in rows:
        if family_splits.setdefault(row['source_key'], row['split']) != row['split']:
            raise ValueError('A filename family crosses prepared splits.')
    for split in ('train', 'validation', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(info['labels']): raise ValueError('A split is missing a category.')
    return info, rows


def development_images(info, rows, transform):
    """Only train/validation rows; immutable final-test membership is not a loader option."""
    from torch.utils.data import Dataset
    if not rows or any(row['split'] not in ('train', 'validation') for row in rows):
        raise ValueError('Only nonempty train/validation records are allowed.')
    class Images(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            if row['split'] not in ('train', 'validation'): raise ValueError('Final-test inference forbidden here.')
            root = Path(info['raw_root']).resolve(); path = (root / row['path']).resolve()
            if not path.is_relative_to(root) or digest(path) != row['file_sha256']: raise ValueError('Source image changed or escaped root.')
            with Image.open(path) as image: result = transform(ImageOps.exif_transpose(image).convert('RGB'))
            return result, info['labels'].index(row['label'])
    return Images()

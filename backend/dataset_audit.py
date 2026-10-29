"""Non-destructive TextileClass-7 integrity and leakage inventory."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps

LABELS = {'Cotton': 'cotton', 'Cotton MIxed': 'cotton_mixed', 'Denim': 'denim',
          'Polyester': 'polyester', 'Silk': 'silk', 'Viscose': 'viscose', 'Wool': 'wool'}
SPLITS = {'train': 'train', 'valid': 'validation', 'test': 'test'}


def original_key(filename):
    """Roboflow export variants share the portion before .rf.; not specimen ID."""
    return filename.lower().split('.rf.', 1)[0]


def dhash(image):
    small = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return int(value)


def audit(root):
    root = Path(root).resolve()
    records, errors = [], []
    for folder, split in SPLITS.items():
        split_path = root / folder
        if not split_path.is_dir():
            errors.append({'path': folder, 'error': 'Missing split directory'})
            continue
        present = {p.name for p in split_path.iterdir() if p.is_dir()}
        if present != set(LABELS):
            errors.append({'path': folder, 'error': 'Class folders differ from expected labels', 'folders': sorted(present)})
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in LABELS:
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                try:
                    with Image.open(path) as original:
                        original.verify()
                    with Image.open(path) as original:
                        image = ImageOps.exif_transpose(original).convert('RGB')
                        image.load()
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    records.append({'path': path.relative_to(root).as_posix(), 'split': split,
                                    'label': LABELS[class_dir.name], 'source_key': original_key(path.name),
                                    'decoded_sha256': digest, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'width': image.width, 'height': image.height, 'dhash': dhash(image)})
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    errors.append({'path': path.relative_to(root).as_posix(), 'error': str(error)})
    groups = {}
    for field in ('source_key', 'decoded_sha256'):
        by = defaultdict(list)
        for record in records:
            by[record[field]].append(record)
        repeated = [items for items in by.values() if len(items) > 1]
        groups[field] = {
            'repeated_groups': len(repeated),
            'extra_images': sum(len(items) - 1 for items in repeated),
            'cross_split': [[r['path'] for r in items] for items in repeated if len({r['split'] for r in items}) > 1],
            'label_conflicts': [[r['path'] for r in items] for items in repeated if len({r['label'] for r in items}) > 1],
        }
    counts = {split: dict(Counter(r['label'] for r in records if r['split'] == split)) for split in SPLITS.values()}
    # Similarity triage, not identity proof: solid colours and repeated prints
    # often collide. No automatic deletion or split changes based on dHash.
    near_counts = Counter()
    near_examples = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a['split'] == b['split'] or a['source_key'] == b['source_key'] or a['decoded_sha256'] == b['decoded_sha256']:
                continue
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            if distance <= 4:
                key = '|'.join(sorted((a['split'], b['split'])))
                near_counts[key] += 1
                if len(near_examples) < 300:
                    near_examples.append({'a': a['path'], 'b': b['path'], 'distance': distance,
                                          'same_label': a['label'] == b['label']})
    report = {'dataset': 'TextileClass-7 v2', 'raw_root': str(root), 'total_images': len(records),
              'counts': counts, 'errors': errors, 'groups': groups,
              'near_duplicate_triage': {'method': '64-bit dHash Hamming <=4', 'pair_counts': dict(near_counts),
                                        'examples_first_300': near_examples, 'notice': 'Candidate similarity only; not proof of duplication or specimen identity.'},
              'structural_checks_pass': not errors and not any(g['cross_split'] or g['label_conflicts'] for g in groups.values()),
              'notice': 'Training variants are not independent specimens. No physical specimen IDs are supplied; no guarantee of specimen-independent splits. No test predictions or accuracy computed.'}
    return report, records


def write_audit(root, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new audit output directory; existing records will not be overwritten.')
    report, records = audit(root)
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return report

"""Non-destructive audit of user-confirmed linen/nylon/genuine-leather candidates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dataset_audit import dhash


def sample_key(path):
    # Conservative filename-family hint, not verified physical specimen identity.
    return re.sub(r'_im_\d+$', '', Path(path).stem.lower())


def audit(root, previous):
    root = Path(root).resolve(); records, errors = [], []
    for folder, label in [('linen', 'linen'), ('Nylon', 'nylon'), ('Leather', 'leather')]:
        directory = root / folder
        if not directory.is_dir(): errors.append({'path': folder, 'error': 'Missing category folder'}); continue
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'): continue
            try:
                if path.stat().st_size > 6_000_000: raise ValueError('Exceeds current 6 MB upload limit.')
                with Image.open(path) as original:
                    original.verify()
                with Image.open(path) as original:
                    if min(original.size) < 64 or original.width * original.height > 20_000_000: raise ValueError('Unsupported dimensions.')
                    image = ImageOps.exif_transpose(original).convert('RGB'); image.load()
                records.append({'path': path.relative_to(root).as_posix(), 'label': label,
                    'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'decoded_sha256': hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                    'source_key': sample_key(path), 'width': image.width, 'height': image.height, 'dhash': dhash(image),
                    'label_source': 'user_confirmed_folder_category', 'permission_source': 'user_asserted_own_collection',
                    'split': 'unassigned'})
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                errors.append({'path': path.relative_to(root).as_posix(), 'error': str(error)})
    duplicate_groups = {}
    for field in ('file_sha256', 'decoded_sha256'):
        groups = defaultdict(list)
        for record in records: groups[record[field]].append(record['path'])
        duplicate_groups[field] = [paths for paths in groups.values() if len(paths) > 1]
    old = json.loads(Path(previous).read_text()) if previous else []
    overlap = []
    old_hashes = {record[field]: record['path'] for record in old for field in ('file_sha256', 'decoded_sha256')}
    for record in records:
        for field in ('file_sha256', 'decoded_sha256'):
            if record[field] in old_hashes: overlap.append({'new': record['path'], 'old': old_hashes[record[field]], 'match': field})
    near = []
    for index, record in enumerate(records):
        for other in records[index + 1:]:
            if record['source_key'] == other['source_key'] or record['decoded_sha256'] == other['decoded_sha256']: continue
            distance = (record['dhash'] ^ other['dhash']).bit_count()
            if distance <= 4: near.append({'new': record['path'], 'other_new': other['path'], 'distance': distance, 'same_label': record['label'] == other['label']})
        for other in old:
            distance = (record['dhash'] ^ other['dhash']).bit_count()
            if distance <= 4: near.append({'new': record['path'], 'old': other['path'], 'old_split': other['split'], 'distance': distance})
    families = defaultdict(list)
    for record in records: families[record['source_key']].append(record['path'])
    return {'status': 'candidate_category_audit_not_trained', 'root': str(root),
            'counts': dict(Counter(r['label'] for r in records)),
            'filename_family_counts': dict(Counter(next(r['label'] for r in records if r['path'] == paths[0]) for paths in families.values())),
            'filename_families': dict(families), 'duplicates': duplicate_groups, 'provider_overlap': overlap,
            'near_pair_candidates': near, 'errors': errors,
            'notice': 'User confirmed labels/ownership; not independent scientific verification. Filename families conservatively suggest related views, not proven specimen IDs. dHash candidates require visual review, not automatic deletion. No images moved, labels changed, splits assigned, training or final-test inference performed.'}, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT / 'data/research_collection/images/unsupported_fabrics'))
    parser.add_argument('--previous', default=str(ROOT / 'runtime/evaluations/textileclass_audit_01/inventory.json'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/added_categories_audit_01'))
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise ValueError('Use a new output directory.')
    report, records = audit(args.root, args.previous)
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'filename_family_counts': report['filename_family_counts'],
        'duplicate_groups': {k: len(v) for k, v in report['duplicates'].items()}, 'provider_overlap_n': len(report['provider_overlap']),
        'near_pair_n': len(report['near_pair_candidates']), 'errors': report['errors']}, indent=2))


if __name__ == '__main__': main()

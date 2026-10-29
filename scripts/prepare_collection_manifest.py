"""Inventory photos and hashes only; every scientific/permission label stays unset."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.image_root).resolve(); output = Path(args.output)
    if not root.is_dir(): raise ValueError('Image root must exist.')
    if output.exists(): raise ValueError('Never overwrite an existing annotation manifest.')
    rows = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'): continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'id': 'photo_' + digest[:16], 'path': path.relative_to(root).as_posix(), 'file_sha256': digest,
            'specimen_id': '', 'source_group_id': '', 'source_reference': '', 'permission_reference': '',
            'permission_confirmed': False, 'independence_reviewed': False, 'reviewer': '',
            'image_kind': 'unreviewed', 'split': 'unassigned', 'kind': 'unreviewed', 'category': '', 'region': 'unknown',
            'labels': {'batik': None, 'handloom': None, 'pattern_type': None, 'weave': None, 'fabric_texture': None}})
    if not rows: raise ValueError('No photos found; no draft written.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        for row in rows: handle.write(json.dumps(row) + '\n')
    print(f'Inventoried {len(rows)} photos. All labels/permissions/splits require review; this draft is deliberately not training-ready.')


if __name__ == '__main__': main()

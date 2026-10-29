"""External real-photo schema, provenance assertions and split/label validation."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from .classifier import CATEGORIES
from .shared_contract import contract

SPLITS = ('train', 'validation', 'calibration', 'assessment', 'final_test')
BINARY_HEADS = ('batik', 'handloom')


def vocabulary():
    shared = contract()
    return {'batik': [False, True], 'handloom': [False, True],
            'pattern_type': shared['pattern_type'], 'weave': ['plain', 'twill', 'satin', 'other'],
            'fabric_texture': shared['fabric_texture']}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provider_hashes():
    """Hash raw images for overlap only, never model inference on reserved data."""
    root = Path(__file__).resolve().parents[1] / 'runtime/datasets/raw/Fabric Classification.v2'
    return {sha(path) for path in root.rglob('*') if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')}


def load_manifest(path, image_root, forbidden_hashes=()):
    path, root = Path(path), Path(image_root).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows: raise ValueError('No collected images. Schema examples are not research data.')
    ids, hashes, decoded, groups, truths = set(), set(), {}, {}, {}
    vocab = vocabulary()
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Each manifest row must be an object.')
        for field in ('id', 'path', 'specimen_id', 'source_group_id', 'source_reference', 'permission_reference', 'reviewer', 'file_sha256'):
            if not isinstance(row.get(field), str) or not row[field].strip(): raise ValueError('Missing ' + field)
        if row.get('image_kind') != 'real' or row.get('permission_confirmed') is not True or row.get('independence_reviewed') is not True:
            raise ValueError('Need real images, permission and explicit independence review assertions.')
        if row.get('split') not in SPLITS: raise ValueError('Assign an approved research split.')
        if row.get('region') not in ('sri_lankan', 'global', 'unknown'): raise ValueError('Invalid documented region.')
        if row.get('kind') not in ('supported', 'unsupported_fabric', 'non_fabric'): raise ValueError('Invalid image kind category.')
        category = row.get('category')
        if not isinstance(category, str) or not category.strip() or (row['kind'] == 'supported') != (category in CATEGORIES):
            raise ValueError('Category must agree with supported/unknown kind.')
        if row['id'] in ids: raise ValueError('Duplicate image ID.')
        ids.add(row['id'])
        for field in ('specimen_id', 'source_group_id'):
            key = (field, row[field])
            if groups.setdefault(key, row['split']) != row['split']: raise ValueError('Specimen/source group crosses splits.')
        signature = (row['kind'], category, row['region'])
        key = ('specimen', row['specimen_id'])
        if truths.setdefault(key, signature) != signature: raise ValueError('Conflicting specimen category/provenance.')
        annotations = row.get('labels', {})
        if not isinstance(annotations, dict) or set(annotations) - set(vocab): raise ValueError('Unknown annotation head.')
        for head, record in annotations.items():
            if record is None: continue
            if not isinstance(record, dict): raise ValueError('Annotations need value/reference/reviewer or null.')
            value = record.get('value')
            if value is None: continue
            if head in BINARY_HEADS:
                if type(value) is not bool: raise ValueError('Technique labels must be true, false or null, never inferred negatives.')
            elif not isinstance(value, str) or value not in vocab[head]:
                raise ValueError('Unsupported ' + head + ' label; use null for unknown.')
            if row['kind'] == 'non_fabric': raise ValueError('Non-fabric objects must not carry fabric attribute labels.')
            for field in ('reference', 'reviewer'):
                if not isinstance(record.get(field), str) or not record[field].strip(): raise ValueError('Known labels need their own reference and reviewer.')
            key = (row['specimen_id'], head)
            if truths.setdefault(key, value) != value: raise ValueError('Conflicting known labels for one specimen.')
        image_path = (root / row['path']).resolve()
        if not image_path.is_relative_to(root): raise ValueError('Image escapes collection root.')
        if image_path.stat().st_size > 6_000_000: raise ValueError('Use photos below 6 MB.')
        digest = sha(image_path)
        if digest != row['file_sha256']: raise ValueError('Image checksum mismatch.')
        if digest in forbidden_hashes or digest in hashes: raise ValueError('Exact image overlap with provider or collection.')
        hashes.add(digest)
        with Image.open(image_path) as image:
            if image.format not in ('PNG', 'JPEG', 'WEBP') or min(image.size) < 64 or image.width * image.height > 20_000_000:
                raise ValueError('Unsupported image format or dimensions.')
            image = ImageOps.exif_transpose(image).convert('RGB')
            content = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if content in decoded: raise ValueError('Duplicate decoded photo, possibly re-encoded.')
        decoded[content] = row['split']
    return rows


def value(row, head):
    record = row.get('labels', {}).get(head)
    return record.get('value') if isinstance(record, dict) else None

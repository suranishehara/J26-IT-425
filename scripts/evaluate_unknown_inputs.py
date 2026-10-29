"""Evaluate existing review policy on real externally labelled inputs, without tuning."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_manifest(path, image_root, forbidden_hashes):
    from backend.classifier import CATEGORIES
    root = Path(image_root).resolve()
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    seen, ids, groups = set(), set(), {}
    if not rows: raise ValueError('No evaluation examples.')
    for row in rows:
        if not isinstance(row, dict) or row.get('permission') is not True or row.get('split') != 'development':
            raise ValueError('Each row needs explicit permission and development split.')
        for field in ('id', 'specimen_id', 'path', 'reference', 'true_category', 'file_sha256'):
            if not isinstance(row.get(field), str) or not row[field].strip(): raise ValueError('Missing ' + field)
        if row['id'] in ids: raise ValueError('Duplicate image ID.')
        ids.add(row['id'])
        if row.get('kind') not in ('supported', 'unsupported_fabric', 'non_fabric'):
            raise ValueError('Unknown example kind.')
        if row.get('provenance') not in ('sri_lankan', 'global', 'unknown'):
            raise ValueError('Provenance must be documented or unknown.')
        signature = (row['kind'], row['true_category'], row['provenance'])
        if groups.setdefault(row['specimen_id'], signature) != signature: raise ValueError('Conflicting specimen labels.')
        if (row['kind'] == 'supported') != (row['true_category'] in CATEGORIES):
            raise ValueError('Supported/unknown truth conflicts with the seven-category inventory.')
        image = (root / row['path']).resolve()
        if not image.is_relative_to(root): raise ValueError('Image path escapes evaluation root.')
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row['file_sha256']: raise ValueError('Image checksum mismatch.')
        if digest in forbidden_hashes or digest in seen: raise ValueError('Duplicate image or overlap with existing dataset.')
        seen.add(digest)
    if not any(r['kind'] == 'supported' for r in rows) or not any(r['kind'] != 'supported' for r in rows):
        raise ValueError('Collect both supported and unknown inputs before evaluation.')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    args = parser.parse_args()
    from backend.classifier import availability, CATEGORIES
    from backend.images import inspect
    from backend.training_data import read_development
    from backend.unknown_evaluation import metrics
    output = Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory.')
    prep, development = read_development(args.dataset)
    forbidden = {r['file_sha256'] for rows in development.values() for r in rows}
    # Include excluded/raw provider files too: they are not independent external inputs.
    for image in Path(prep['raw_root']).rglob('*'):
        if image.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            forbidden.add(hashlib.sha256(image.read_bytes()).hexdigest())
    rows = load_manifest(args.manifest, args.image_root, forbidden)
    release = availability()
    if not release['available']: raise ValueError('Active classifier unavailable.')
    if release['categories'] != CATEGORIES:
        raise ValueError('This legacy seven-category evaluator cannot evaluate a nine-category release. Prepare an approved nine-category independent evaluation first.')
    results = []
    for row in rows:
        path = (Path(args.image_root) / row['path']).resolve()
        suffix = path.suffix.lower()
        mime = {'.png': 'png', '.jpg': 'jpeg', '.jpeg': 'jpeg', '.webp': 'webp'}.get(suffix)
        if not mime: raise ValueError('Unsupported file suffix.')
        payload = {'permission': True, 'mode': 'experimental', 'image': f'data:image/{mime};base64,' + base64.b64encode(path.read_bytes()).decode()}
        result = inspect(payload)
        if result['classifier'].get('checkpoint_sha256') != release['checkpoint_sha256']:
            raise ValueError('Classifier unavailable or changed during evaluation.')
        candidates = result['predictions']['fabric_category']
        results.append({'id': row['id'], 'specimen_id': row['specimen_id'], 'kind': row['kind'], 'provenance': row['provenance'],
                        'truth': row['true_category'], 'prediction': candidates[0]['category'], 'max_score': candidates[0]['relative_score'],
                        'accepted': result['review_decision']['allow_quick_apply'], 'capture_quality': result['capture_quality']['status']})
        print('evaluated', len(results), '/', len(rows), flush=True)
    report = {'status': 'external_development_evaluation_not_production_validation', 'classifier': release,
        'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'metrics': metrics(results), 'rows': results,
        'specimen_count': len({r['specimen_id'] for r in results}),
        'by_unknown_kind': {kind: metrics([r for r in results if r['kind'] in ('supported', kind)]) for kind in ('unsupported_fabric', 'non_fabric') if any(r['kind'] == kind for r in results)},
        'by_provenance': {}, 'test_inference_performed': False, 'activated_in_app': False,
        'notice': 'No thresholds fitted. Exact hash overlap rejected; near-duplicate/source/specimen leakage still requires human review. '
                  'Unknown labels and permissions are researcher assertions. No photos or body profiles copied to output.'}
    for provenance in ('sri_lankan', 'global', 'unknown'):
        subset = [r for r in results if r['provenance'] == provenance]
        report['by_provenance'][provenance] = metrics(subset) if any(r['kind'] == 'supported' for r in subset) and any(r['kind'] != 'supported' for r in subset) else {'n': len(subset), 'status': 'insufficient_supported_and_unknown_examples'}
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__': main()

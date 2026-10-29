"""Non-destructive fixed grouped splits and train-only class balancing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(image_root, output, scope_path, previous_inventory, seed=42):
    from backend.category_dataset import inventory, grouped_split, training_class_weights, read_prepared, digest
    output = Path(output)
    if output.exists(): raise ValueError('Choose a new version directory; prepared splits are never overwritten.')
    scope = json.loads(Path(scope_path).read_text()); labels = scope['labels']
    rows, errors = inventory(image_root, labels)
    if errors: raise ValueError('Resolve image integrity errors before splitting: ' + json.dumps(errors[:10]))
    previous = json.loads(Path(previous_inventory).read_text())
    old_hashes = {r[field] for r in previous for field in ('file_sha256', 'decoded_sha256')}
    blocked = {r['path'] for r in rows if r['file_sha256'] in old_hashes or r['decoded_sha256'] in old_hashes}
    accepted, excluded = grouped_split(rows, labels, seed, blocked)
    by_path = {r['path']: r for r in accepted}
    similar = []
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            if first['source_key'] == second['source_key'] or first['decoded_sha256'] == second['decoded_sha256']: continue
            distance = (first['dhash'] ^ second['dhash']).bit_count()
            if distance <= 4:
                a, b = by_path.get(first['path']), by_path.get(second['path'])
                similar.append({'a': first['path'], 'b': second['path'], 'distance': distance,
                                'same_label': first['label'] == second['label'],
                                'retained_cross_split': bool(a and b and a['split'] != b['split'])})
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in accepted)
    counts = {split: dict(Counter(r['label'] for r in accepted if r['split'] == split)) for split in ('train', 'validation', 'test')}
    group_counts = {split: {label: len({r['group_id'] for r in accepted if r['split'] == split and r['label'] == label}) for label in labels} for split in counts}
    weights = training_class_weights(accepted, labels)
    info = {'dataset': 'user_collection_nine_categories', 'version': '1', 'status': 'provisional_filename_and_exact_content_grouped_not_specimen_independent_verified',
        'raw_root': str(Path(image_root).resolve()), 'labels': labels, 'excluded_labels': scope['excluded_labels'],
        'scope_sha256': digest(scope_path), 'manifest_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'seed': seed, 'split_policy': 'Per-label hash-sorted linked families; approximately 70/15/15 train/validation/test by group count. Never move members of a known family or exact-duplicate cluster across splits.',
        'input_images': len(rows), 'retained_images': len(accepted), 'excluded_images': len(excluded),
        'exclusions_by_reason': dict(Counter(r['reason'] for r in excluded)), 'counts': counts, 'group_counts': group_counts,
        'training_class_weights': weights,
        'balancing_policy': 'Weighted cross entropy only: N_train/(9*n_class_train). No oversampling plus weighting, no validation/test rebalance, no copied images.',
        'unimplemented_attribute_heads': ['batik', 'handloom', 'pattern_type', 'weave', 'fabric_texture'],
        'attribute_notice': 'Not supported as predictions yet; these are separate attributes, never material category names or inferred negatives.',
        'similarity_triage': {'method': '64-bit dHash distance <=4, candidate similarity only', 'pair_count': len(similar),
                              'retained_cross_split_candidates': sum(r['retained_cross_split'] for r in similar),
                              'notice': 'Uniform/repeated textures create false positives. No deletion/relabel/group merge solely from this score. Review candidates before claiming specimen-independent evaluation.'},
        'provider_inventory_reference': str(Path(previous_inventory).resolve()), 'provider_inventory_sha256': digest(previous_inventory),
        'provider_overlap_policy': 'Exclude entire new linked family if any exact raw/decoded hash overlaps the prior provider inventory; original provider reserved test stays unchanged.',
        'source_provenance': 'User-supplied category/permission assertions; acquisition/source-batch and physical specimen IDs are not independently verified.',
        'test_inference_performed': False, 'active_models_changed': False,
        'notice': 'No training or final-test inference. Tiny class-wise validation/test group counts and visual-similarity candidates limit evaluation. Filename grouping cannot detect every shared specimen/source. Original photos remain untouched.'}
    output.mkdir(parents=True)
    (output / 'manifest.jsonl').write_text(text, encoding='utf-8', newline='\n')
    (output / 'inventory.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output / 'excluded.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    (output / 'similarity_triage.json').write_text(json.dumps(similar, indent=2), encoding='utf-8')
    (output / 'preparation.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    read_prepared(output)
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=str(ROOT / 'data/research_collection/images'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/datasets/fabric_nine_categories_01'))
    parser.add_argument('--scope', default=str(ROOT / 'config/fabric_category_expansion.json'))
    parser.add_argument('--previous-inventory', default=str(ROOT / 'runtime/evaluations/textileclass_audit_01/inventory.json'))
    args = parser.parse_args()
    result = prepare(args.images, args.output, args.scope, args.previous_inventory)
    print(json.dumps({key: result[key] for key in ('input_images', 'retained_images', 'excluded_images', 'counts', 'group_counts', 'training_class_weights', 'similarity_triage')}, indent=2))

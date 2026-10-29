"""Build a conservative pilot manifest without modifying raw data or test membership."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dataset_audit import LABELS


def prepare(audit_dir, output):
    audit_dir, output = Path(audit_dir), Path(output)
    if output.exists():
        raise ValueError('Output already exists; use a new directory.')
    report = json.loads((audit_dir / 'report.json').read_text())
    records = json.loads((audit_dir / 'inventory.json').read_text())
    if report['errors']:
        raise ValueError('Resolve integrity errors first.')
    near = report['near_duplicate_triage']['examples_first_300']
    if sum(report['near_duplicate_triage']['pair_counts'].values()) > len(near):
        raise ValueError('Triage is truncated: export all candidate pairs before preparation.')
    parent = {r['path']: r['path'] for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a, b):
        parent[find(b)] = find(a)

    for field in ('source_key', 'decoded_sha256'):
        seen = {}
        for r in records:
            if r[field] in seen:
                join(seen[r[field]], r['path'])
            else:
                seen[r[field]] = r['path']
    for pair in near:
        join(pair['a'], pair['b'])
    groups = defaultdict(list)
    for r in records:
        groups[find(r['path'])].append(r)
    accepted, excluded = [], []
    for members in groups.values():
        paths = sorted(r['path'] for r in members)
        group_id = hashlib.sha256('\n'.join(paths).encode()).hexdigest()
        splits, labels = {r['split'] for r in members}, {r['label'] for r in members}
        for r in members:
            reason = None
            # Test is never relabelled, removed or predicted here. Group relations
            # are integrity screening only; label conflicts remain test caveats.
            if r['split'] != 'test':
                if len(labels) > 1:
                    reason = 'conservative_ambiguous_group_labels_or_filename_collision'
                elif 'test' in splits:
                    reason = 'possible_relation_to_reserved_test'
                elif r['split'] == 'train' and 'validation' in splits:
                    reason = 'possible_relation_to_validation'
            row = {**r, 'group_id': group_id}
            if reason:
                excluded.append({**row, 'reason': reason})
            else:
                accepted.append(row)
    accepted.sort(key=lambda r: r['path'])
    output.mkdir(parents=True)
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in accepted)
    (output / 'manifest.jsonl').write_text(text, encoding='utf-8', newline='\n')
    (output / 'excluded.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    counts = {split: dict(Counter(r['label'] for r in accepted if r['split'] == split)) for split in ('train', 'validation', 'test')}
    for split in ('train', 'validation'):
        if set(counts[split]) != set(LABELS.values()):
            raise ValueError('A class became empty; do not train this preparation.')
    info = {'raw_root': report['raw_root'], 'dataset': report['dataset'], 'labels': sorted(LABELS.values()),
            'manifest_sha256': hashlib.sha256(text.encode()).hexdigest(), 'counts': counts,
            'excluded_count': len(excluded), 'exclusions_by_reason': dict(Counter(r['reason'] for r in excluded)),
            'status': 'provisional_conservative_image_group_split_not_specimen_verified',
            'policy': 'Uncertain filename and dHash groups are conservatively linked; conflicts excluded from train/validation, never relabelled. Test membership unchanged. Possible false-positive exclusions, residual near-duplicates and shared specimens remain.',
            'test_evaluated': False}
    (output / 'preparation.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', default=str(ROOT / 'runtime/evaluations/textileclass_audit_01'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.output), indent=2))

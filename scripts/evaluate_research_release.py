"""Separate real-photo calibration/assessment and guarded one-shot final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def final_receipt(root, identity, rows, stage):
    """Formatting-independent final-data identity; failed attempts stay consumed."""
    membership = sorted(row['file_sha256'] for row in rows)
    fingerprint = hashlib.sha256(json.dumps(membership).encode()).hexdigest()
    folder = Path(root); folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity['checkpoint_sha256'] + '_' + fingerprint + '.json')
    with path.open('x', encoding='utf-8') as handle:
        json.dump({'stage': stage, 'started_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': identity,
                   'data_fingerprint': fingerprint, 'notice': 'Final inference attempt recorded before model inference. Do not tune on these outcomes.'}, handle, indent=2)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['calibration', 'assessment', 'final-test', 'provider-final-test'])
    parser.add_argument('--manifest', required=True); parser.add_argument('--image-root', required=True)
    parser.add_argument('--model-id', default='convnext_v2', choices=['convnext_v2', 'convnext_v1', 'efficientnet_b0'])
    parser.add_argument('--calibration'); parser.add_argument('--output', required=True)
    parser.add_argument('--acknowledge-final-test', action='store_true')
    args = parser.parse_args()
    final_stage = args.stage in ('final-test', 'provider-final-test')
    if final_stage and not args.acknowledge_final_test:
        raise ValueError('Final evaluation requires --acknowledge-final-test after freezing model and calibration; do not tune afterwards.')
    output = Path(args.output)
    if output.exists(): raise ValueError('Use a new output directory.')
    import torch
    from backend.research_data import load_manifest, provider_hashes, sha
    from backend.research_inference import load, collect_logits
    from backend.classifier import CATEGORIES
    from backend.calibration import fit_temperature, probability_metrics
    from backend.fabric_model import classification_metrics
    from backend.unknown_evaluation import metrics as unknown_metrics
    torch.set_num_threads(4)
    all_rows = load_manifest(args.manifest, args.image_root, provider_hashes())
    model, transform, identity = load(args.model_id)
    if identity['labels'] != CATEGORIES:
        raise ValueError('This legacy seven-category evaluator cannot evaluate a nine-category release. Use a separately approved nine-category evaluation manifest.')
    if args.stage == 'calibration':
        rows = [row for row in all_rows if row['split'] == 'calibration' and row['kind'] == 'supported']
        if {row['category'] for row in rows} != set(CATEGORIES): raise ValueError('Independent calibration needs all seven supported categories.')
        logits = collect_logits(model, transform, rows, args.image_root)
        labels = torch.tensor([CATEGORIES.index(row['category']) for row in rows])
        temperature = fit_temperature(logits, labels)
        result = {'status': 'external_calibration_fit_not_assessed', 'identity': identity, 'temperature': temperature,
            'manifest_sha256': sha(args.manifest), 'membership': [{key: row[key] for key in ('id', 'file_sha256', 'specimen_id', 'source_group_id')} for row in rows],
            'fit_before': probability_metrics(logits, labels), 'fit_after': probability_metrics(logits, labels, temperature),
            'notice': 'Fit metrics are not validation. Assess only on disjoint assessment/final groups. Independence/labels are researcher assertions. Not activated in the app.'}
        output.mkdir(parents=True); (output / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'n': len(rows), 'temperature': temperature})); return
    if not args.calibration: raise ValueError('Provide the frozen external calibration.json artifact.')
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get('status') != 'external_calibration_fit_not_assessed' or calibration.get('identity') != identity:
        raise ValueError('Calibration does not match this frozen checkpoint/report.')
    if calibration['manifest_sha256'] != sha(args.manifest): raise ValueError('Collection changed since calibration freeze.')
    temperature = calibration['temperature']
    if type(temperature) not in (int, float) or not .05 <= temperature <= 20: raise ValueError('Invalid fitted temperature.')
    image_root = args.image_root
    if args.stage == 'provider-final-test':
        from backend.training_data import read_development
        prep, splits = read_development(ROOT / 'runtime/datasets/textileclass_pilot_02')
        if prep['manifest_sha256'] != identity['training_manifest_sha256']:
            raise ValueError('Provider final split is not the frozen training manifest.')
        rows = [dict(row, id=row['path'], specimen_id=row['group_id'], source_group_id=row['group_id'],
                     kind='supported', category=row['label'], region='unknown') for row in splits['test']]
        image_root = prep['raw_root']
    else:
        split = 'final_test' if final_stage else 'assessment'
        rows = [row for row in all_rows if row['split'] == split]
    if not rows: raise ValueError('No collected rows for requested stage.')
    fit_membership = calibration['membership']
    for field in ('file_sha256', 'specimen_id', 'source_group_id'):
        if {r[field] for r in rows} & {r[field] for r in fit_membership}: raise ValueError('Calibration/evaluation overlap.')
    receipt = final_receipt(ROOT / 'runtime/evaluations/final_test_ledger', identity, rows, args.stage) if final_stage else None
    logits = collect_logits(model, transform, rows, image_root)
    supported = [i for i, row in enumerate(rows) if row['kind'] == 'supported']
    result = {'stage': args.stage, 'identity': identity, 'calibration_sha256': sha(args.calibration),
              'manifest_sha256': sha(args.manifest), 'temperature': temperature, 'n': len(rows),
              'specimens': len({r['specimen_id'] for r in rows}), 'final_receipt': receipt, 'by_region': {}}
    def known_metrics(indices):
        labels = torch.tensor([CATEGORIES.index(rows[i]['category']) for i in indices])
        return {'classification': classification_metrics(labels.tolist(), logits[indices].argmax(1).tolist(), CATEGORIES),
                'raw_probability_metrics': probability_metrics(logits[indices], labels),
                'temperature_probability_metrics': probability_metrics(logits[indices], labels, temperature)}
    result['supported'] = known_metrics(supported) if supported else {'n': 0}
    for region in ('sri_lankan', 'global', 'unknown'):
        indices = [i for i in supported if rows[i]['region'] == region]
        result['by_region'][region] = known_metrics(indices) if indices else {'n': 0}
    probabilities = logits.softmax(1); records = []
    for i, row in enumerate(rows):
        top = probabilities[i].sort(descending=True).values
        records.append({'id': row['id'], 'kind': row['kind'], 'region': row['region'], 'truth': row['category'], 'prediction': CATEGORIES[int(logits[i].argmax())],
                        'max_score': float(top[0]), 'accepted': bool(top[0] >= .6 and top[0] - top[1] >= .15)})
    result['unknown'] = unknown_metrics(records) if supported and any(row['kind'] != 'supported' for row in rows) else {'status': 'need_supported_and_unknown_examples'}
    result['unknown_by_region'] = {}
    for region in ('sri_lankan', 'global', 'unknown'):
        subset = [row for row in records if row['region'] == region]
        result['unknown_by_region'][region] = unknown_metrics(subset) if any(r['kind'] == 'supported' for r in subset) and any(r['kind'] != 'supported' for r in subset) else {'status': 'need_supported_and_unknown_examples', 'n': len(subset)}
    result['predictions'] = records
    result['notice'] = ('Frozen model/temperature. No threshold tuning or activation. Unknown quick-apply metrics here use raw score/margin only, not capture heuristics. '
                        'Metrics are image-weighted; repeated views are not independent. Provider final split has unverified specimen independence and region, not a Sri Lankan external evaluation. '
                        'Never tune on final results; disclosure/provenance assertions and manual near-duplicate review remain necessary.')
    output.mkdir(parents=True); (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'stage': args.stage, 'n': len(rows), 'report': str(output / 'report.json')}))


if __name__ == '__main__': main()

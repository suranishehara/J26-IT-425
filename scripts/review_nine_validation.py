"""Review cached validation logits only; no test access or new image inference."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/convnextv2_nine_categories_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/fabric_nine_categories_01'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/nine_validation_review_01'))
    args = parser.parse_args(); run, output = Path(args.run), Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory.')
    import torch
    from safetensors.torch import load_file
    from backend.category_dataset import read_prepared, digest
    from backend.fabric_model import classification_metrics
    info, records = read_prepared(args.dataset)
    report = json.loads((run / 'report.json').read_text())
    if report['labels'] != info['labels'] or report['manifest_sha256'] != info['manifest_sha256'] or report['checkpoint_sha256'] != digest(run / 'model.safetensors'):
        raise ValueError('Model/dataset identity mismatch.')
    rows = [r for r in records if r['split'] == 'validation']; labels = info['labels']
    cache = load_file(str(run / 'validation_logits.safetensors')); logits, truth = cache['logits'], cache['labels']
    if logits.shape != (len(rows), 9) or not bool(torch.isfinite(logits).all()) or truth.tolist() != [labels.index(r['label']) for r in rows]:
        raise ValueError('Cached validation alignment differs from frozen manifest.')
    predictions = logits.argmax(1).tolist()
    metric = classification_metrics(truth.tolist(), predictions, labels)
    if metric != report['validation']: raise ValueError('Cached logits do not reproduce reported validation metrics.')
    probabilities = logits.softmax(1); errors = []; families = defaultdict(list)
    for i, row in enumerate(rows):
        predicted = labels[predictions[i]]
        families[row['group_id']].append((int(truth[i]), predictions[i], row['path']))
        if predicted != row['label']:
            errors.append({'path': row['path'], 'group_id': row['group_id'], 'truth': row['label'], 'prediction': predicted,
                           'uncalibrated_top_score': float(probabilities[i].max())})
    group_truth, group_prediction = [], []
    for family in families.values():
        if len({item[0] for item in family}) != 1: raise ValueError('Mixed-label validation group.')
        votes = Counter(item[1] for item in family)
        group_prediction.append(min(votes, key=lambda index: (-votes[index], index)))
        group_truth.append(family[0][0])
    result = {'status': 'development_validation_review', 'checkpoint_sha256': report['checkpoint_sha256'],
        'manifest_sha256': info['manifest_sha256'], 'validation_cache_sha256': digest(run / 'validation_logits.safetensors'),
        'image_metrics': metric, 'inferred_family_majority_vote_metrics': classification_metrics(group_truth, group_prediction, labels),
        'majority_tie_break': 'Lowest label index', 'errors': sorted(errors, key=lambda r: -r['uncalibrated_top_score']),
        'frozen_baseline_same_split': report['frozen_validation'], 'test_images_opened': 0,
        'notice': 'Validation development review, not final-test evaluation. Filename/content groups are not certified independent specimens; small category group counts and perceptual-similarity candidates remain. No labels, splits or app releases changed.'}
    output.mkdir(parents=True); (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'image_accuracy': metric['accuracy'], 'image_macro_f1': metric['macro_f1'], 'errors': len(errors),
                      'inferred_family_accuracy': result['inferred_family_majority_vote_metrics']['accuracy'], 'inferred_family_n': len(families)}))


if __name__ == '__main__': main()

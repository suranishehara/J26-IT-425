"""Read saved development reports only; no test inference or model activation."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/efficientnet_b0_finetune_01'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/efficientnet_finetune_comparison_01'))
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if output.exists(): raise ValueError('Choose a new comparison output directory.')
    report = json.loads((run / 'report.json').read_text())
    base, candidate = report['baseline_validation_recomputed'], report['validation']
    if base['label_order'] != candidate['label_order'] or base['n'] != candidate['n']:
        raise ValueError('Comparison targets differ.')
    result = {'baseline_checkpoint_sha256': report['baseline_checkpoint_sha256'],
              'candidate_checkpoint_sha256': report['checkpoint_sha256'],
              'manifest_sha256': report['manifest_sha256'], 'evaluation_precision': 'float32',
              'validation_n': base['n'], 'best_epoch': report['best_epoch'],
              'baseline': {k: base[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')},
              'candidate': {k: candidate[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')},
              'per_class_f1_delta': {label: candidate['per_class'][label]['f1'] - base['per_class'][label]['f1'] for label in base['label_order']},
              'test_evaluated': False, 'application_activated': False,
              'notice': 'Single development split comparison, not statistical or specimen-independent proof. No test predictions.'}
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

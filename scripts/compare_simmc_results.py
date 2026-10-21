"""Post-selection paired conversation-bootstrap uncertainty; never tunes a model."""
import argparse
import json
from pathlib import Path


def scores(matrix):
    import numpy as np
    diagonal = np.diag(matrix).astype(float)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator != 0)
    return float(diagonal.sum() / matrix.sum()), float(f1.mean())


def compare(run, repetitions=1000, dataset=None):
    import numpy as np
    from scripts.train_simmc_acts import digest
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or not report['checkpoint_reload_verified']:
        raise ValueError('Completed, reloaded benchmark required.')
    rows = [json.loads(line) for line in (run / 'assessment_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = sorted({row['dialogue_id'] for row in rows})
    labels = report['labels']; index = {value: i for i, value in enumerate(labels)}
    group_index = {value: i for i, value in enumerate(groups)}
    names = ('majority', 'tfidf_logistic', 'deberta')
    matrices = {name: np.zeros((len(groups), len(labels), len(labels)), dtype=np.int64) for name in names}
    for row in rows:
        for name in names:
            matrices[name][group_index[row['dialogue_id']], index[row['truth']], index[row[name]]] += 1
    rng = np.random.default_rng(42)
    draws = {name: [] for name in names}; deltas = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), len(groups))
        values = {name: scores(matrix[selected].sum(0)) for name, matrix in matrices.items()}
        for name in names:
            draws[name].append(values[name])
        deltas.append([values['deberta'][i] - values['tfidf_logistic'][i] for i in (0, 1)])
    def interval(values):
        bounds = np.quantile(np.asarray(values), [.025, .975], axis=0)
        return {name: {'lower': float(bounds[0, i]), 'upper': float(bounds[1, i])}
                for i, name in enumerate(('accuracy', 'macro_f1'))}
    result = {'method': 'Paired percentile bootstrap of whole assessment conversations, 95% intervals',
              'seed': 42, 'repetitions': repetitions, 'assessment_dialogues': len(groups),
              'prediction_sha256': digest(run / 'assessment_predictions.jsonl'),
              'models': {name: {'point_accuracy': scores(matrix.sum(0))[0],
                                'point_macro_f1': scores(matrix.sum(0))[1], 'intervals': interval(draws[name])}
                         for name, matrix in matrices.items()},
              'deberta_minus_tfidf_intervals': interval(deltas),
              'notice': 'Sampling uncertainty for these fixed checkpoints and this public split only; not seed variance, speaker/template independence, cultural accuracy or an official multimodal benchmark.'}
    if dataset:
        from scripts.train_simmc_acts import read_split, metrics
        from scripts.prepare_simmc import normalize
        audit_path = dataset / 'audit.json'
        if digest(audit_path) != report['source_audit_sha256']:
            raise ValueError('Diagnostic dataset does not match the trained experiment.')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        training_inputs = {normalize(row['text']) for row in read_split(dataset, 'train', audit)}
        assessment_inputs = {(row['dialogue_id'], row['turn_idx']): normalize(row['text'])
                             for row in read_split(dataset, 'devtest', audit)}
        indices = {name: [] for name in ('seen_formatted_input', 'novel_formatted_input')}
        for i, row in enumerate(rows):
            text = assessment_inputs[(row['dialogue_id'], row['turn_idx'])]
            indices['seen_formatted_input' if text in training_inputs else 'novel_formatted_input'].append(i)
        result['input_overlap_diagnostic'] = {
            name: {'n': len(selected), 'models': {
                model: metrics([rows[i]['truth'] for i in selected], [rows[i][model] for i in selected], labels)
                for model in names} if selected else {}} for name, selected in indices.items()}
        result['input_overlap_notice'] = 'Post-selection descriptive slice, not an independent split or tuning target; normalization is NFKC/case/whitespace, not semantic/template deduplication.'
    target = run / 'comparison_uncertainty.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.run, dataset=args.dataset), indent=2))

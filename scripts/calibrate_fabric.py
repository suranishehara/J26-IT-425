"""Group-separated validation calibration pilot. No app activation or final test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def partition(rows):
    groups = sorted({r['group_id'] for r in rows}, key=lambda value: hashlib.sha256(('calibration-42:' + value).encode()).hexdigest())
    fit_groups = set(groups[:int(.7 * len(groups))])
    fit = [i for i, r in enumerate(rows) if r['group_id'] in fit_groups]
    assess = [i for i, r in enumerate(rows) if r['group_id'] not in fit_groups]
    if not fit or not assess: raise ValueError('Need multiple independent development groups.')
    return fit, assess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/efficientnet_b0_finetune_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/efficientnet_calibration_01'))
    args = parser.parse_args()
    output, run = Path(args.output), Path(args.run)
    if output.exists(): raise ValueError('Choose a new output directory.')
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    from backend.training_data import read_development, DevelopmentImages
    from backend.fabric_model import FabricNetwork, ARCHITECTURE, preprocessing
    from backend.calibration import fit_temperature, probability_metrics
    prep, rows = read_development(args.dataset)
    report = json.loads((run / 'report.json').read_text())
    checksum = hashlib.sha256((run / 'model.safetensors').read_bytes()).hexdigest()
    if report['checkpoint_sha256'] != checksum or report['manifest_sha256'] != prep['manifest_sha256'] or report['labels'] != prep['labels']:
        raise ValueError('Checkpoint/dataset identity mismatch.')
    if report['architecture'] == ARCHITECTURE:
        model = FabricNetwork(len(prep['labels'])); transform = preprocessing()
    else:
        from backend.convnext_model import ConvNextFabric, WEIGHTS, ARCHITECTURE as CN_ARCH
        if report['architecture'] == CN_ARCH:
            model = ConvNextFabric(len(prep['labels'])); transform = WEIGHTS.transforms()
        else:
            from backend.convnextv2_model import ConvNextV2Fabric, ARCHITECTURE as V2_ARCH
            if report['architecture'] != V2_ARCH: raise ValueError('Unsupported architecture.')
            model = ConvNextV2Fabric(len(prep['labels'])); transform = model.transform
            if json.dumps(model.data_config, sort_keys=True) != json.dumps(report['training']['data_config'], sort_keys=True):
                raise ValueError('Model preprocessing differs from training.')
    torch.set_num_threads(4); model.eval()
    model.load_state_dict(load_file(str(run / 'model.safetensors'), device='cpu'))
    dataset = DevelopmentImages(prep['raw_root'], rows['validation'], prep['labels']); dataset.transform = transform
    outputs, truth = [], []
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=16, shuffle=False):
            outputs.append(model(images)); truth.extend(labels.tolist())
    logits, labels = torch.cat(outputs), torch.tensor(truth)
    fit, assess = partition(rows['validation'])
    if set(labels[fit].tolist()) != set(range(len(prep['labels']))): raise ValueError('Calibration fit partition missing a class.')
    temperature = fit_temperature(logits[fit], labels[fit])
    result = {'status': 'development_calibration_not_activated', 'architecture': report['architecture'],
        'checkpoint_sha256': checksum, 'manifest_sha256': prep['manifest_sha256'], 'labels': prep['labels'],
        'temperature': temperature, 'temperature_bounds': [.05, 20.], 'objective': 'fit-partition negative log likelihood',
        'split_policy': 'sha256 calibration-42:group_id order; first 70% of groups fit, remainder assessment',
        'fit_n': len(fit), 'assessment_n': len(assess),
        'fit': {'before': probability_metrics(logits[fit], labels[fit]), 'after': probability_metrics(logits[fit], labels[fit], temperature)},
        'assessment': {'before': probability_metrics(logits[assess], labels[assess]), 'after': probability_metrics(logits[assess], labels[assess], temperature)},
        'membership': [{'path': row['path'], 'group_id': row['group_id'], 'partition': 'fit' if i in set(fit) else 'assessment'} for i, row in enumerate(rows['validation'])],
        'test_images_opened': 0, 'activated_in_app': False,
        'notice': 'The model was already selected using the full validation split. This post-selection pilot is not independent calibration evidence. '
                  'Temperature does not improve top-1 classification or establish unknown rejection. Deployment needs a separate calibration/evaluation dataset.',
        'method_source': 'https://proceedings.mlr.press/v70/guo17a.html'}
    output.mkdir(parents=True)
    save_file({'logits': logits, 'labels': labels}, str(output / 'validation_logits.safetensors'))
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'temperature', 'fit_n', 'assessment_n', 'assessment')}), flush=True)


if __name__ == '__main__': main()

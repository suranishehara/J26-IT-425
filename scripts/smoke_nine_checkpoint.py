"""Local nine-output reload/CPU mechanics on one TRAIN photo, never test data."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/convnextv2_nine_categories_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/fabric_nine_categories_01'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/nine_category_checkpoint_smoke_01'))
    args = parser.parse_args(); run, output = Path(args.run), Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory.')
    import torch
    from safetensors.torch import load_file
    from backend.convnextv2_model import ConvNextV2Fabric, ARCHITECTURE
    from backend.category_dataset import read_prepared, development_images, digest
    torch.set_num_threads(4)
    prep, rows = read_prepared(args.dataset)
    report = json.loads((run / 'report.json').read_text())
    if (report['architecture'], report['labels'], report['manifest_sha256']) != (ARCHITECTURE, prep['labels'], prep['manifest_sha256']):
        raise ValueError('Checkpoint architecture/labels/dataset identity mismatch.')
    checksum = digest(run / 'model.safetensors')
    if checksum != report['checkpoint_sha256']: raise ValueError('Checkpoint checksum mismatch.')
    model = ConvNextV2Fabric(len(prep['labels'])).eval()
    if json.dumps(model.data_config, sort_keys=True) != json.dumps(report['training']['data_config'], sort_keys=True): raise ValueError('Transform differs from training.')
    model.load_state_dict(load_file(str(run / 'model.safetensors')), strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    row = next(row for row in rows if row['split'] == 'train')
    image, _ = development_images(prep, [row], model.transform)[0]
    results, times = [], []
    with torch.inference_mode():
        for _ in range(3):
            started = time.perf_counter(); logits = model(image.unsqueeze(0))
            times.append(time.perf_counter() - started); results.append(logits.clone())
    if results[0].shape != (1, 9) or not bool(torch.isfinite(results[0]).all()): raise ValueError('Invalid nine-output model.')
    if not all(torch.equal(results[0], result) for result in results): raise ValueError('Repeat prediction changed.')
    if digest(run / 'model.safetensors') != checksum: raise ValueError('Weights changed.')
    probabilities = results[0].softmax(1)[0]; indices = probabilities.argsort(descending=True)[:3].tolist()
    result = {'status': 'passed', 'checkpoint_sha256': checksum, 'labels': report['labels'], 'image_split': 'train',
        'path': row['path'], 'cpu_inference_seconds': times,
        'predictions': [{'category': report['labels'][index], 'relative_score': float(probabilities[index])} for index in indices],
        'test_images_opened': 0, 'activated_in_app': False,
        'notice': 'One-train-image CPU shape, checksum, reload and repeatability mechanics; not independent accuracy or a latency benchmark.'}
    output.mkdir(parents=True); (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

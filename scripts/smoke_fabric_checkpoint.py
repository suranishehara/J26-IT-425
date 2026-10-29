"""Verify saved full model on one TRAIN image; does not activate/evaluate test."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/efficientnet_b0_pilot_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from backend.fabric_model import FabricNetwork, ARCHITECTURE, preprocessing
    torch.set_num_threads(4)
    run, dataset = Path(args.run), Path(args.dataset)
    report = json.loads((run / 'report.json').read_text())
    if report['architecture'] != ARCHITECTURE or hashlib.sha256((run / 'model.safetensors').read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch.')
    manifest = (dataset / 'manifest.jsonl').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != report['manifest_sha256']:
        raise ValueError('Manifest identity mismatch.')
    row = next(json.loads(line) for line in manifest.decode().splitlines() if json.loads(line)['split'] == 'train')
    prep = json.loads((dataset / 'preparation.json').read_text())
    path = (Path(prep['raw_root']) / row['path']).resolve()
    if not path.is_relative_to(Path(prep['raw_root']).resolve()): raise ValueError('Escaping path.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']: raise ValueError('Image changed.')
    model = FabricNetwork(len(report['labels']), pretrained=False).eval()
    model.load_state_dict(load_file(str(run / 'model.safetensors')))
    with Image.open(path) as image:
        tensor = preprocessing()(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        a, b = model(tensor), model(tensor)
    assert torch.equal(a, b), 'Repeat prediction changed.'
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items()), 'Weights changed.'
    predicted = report['labels'][int(a.argmax())]
    output = {'status': 'passed', 'image_split': 'train', 'path': row['path'], 'prediction': predicted,
              'test_images_opened': 0, 'activated_in_app': False,
              'notice': 'Checksum, reload, finite output and repeatability mechanics; not accuracy.'}
    assert bool(torch.isfinite(a).all())
    (run / 'smoke.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__': main()

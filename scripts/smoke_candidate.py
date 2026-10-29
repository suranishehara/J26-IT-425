"""Temporary candidate config, one train photo, CPU repeat/latency; no app switch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(ROOT / 'runtime/training/convnext_tiny_finetune_01'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/convnext_candidate_smoke_01'))
    args = parser.parse_args()
    from PIL import Image, ImageOps
    from backend import classifier
    from backend.training_data import read_development
    run, output = Path(args.run).resolve(), Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory.')
    report = json.loads((run / 'report.json').read_text())
    config = {'enabled': True, 'status': 'experimental_validation_only', 'run_directory': run.relative_to(ROOT).as_posix(),
              'checkpoint_sha256': report['checkpoint_sha256'], 'manifest_sha256': report['manifest_sha256'],
              'report_sha256': classifier.sha(run / 'report.json')}
    prep, rows = read_development(ROOT / 'runtime/datasets/textileclass_pilot_02')
    row = rows['train'][0]; path = Path(prep['raw_root']) / row['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']: raise ValueError('Train sample changed.')
    with Image.open(path) as source: image = ImageOps.exif_transpose(source).convert('RGB')
    original_config_hash = classifier.sha(classifier.CONFIG)
    with tempfile.TemporaryDirectory() as temp:
        config_path = Path(temp) / 'candidate.json'; config_path.write_text(json.dumps(config))
        with patch.object(classifier, 'CONFIG', config_path), patch.object(classifier, '_CACHE', None):
            samples = []; results = []
            for _ in range(3):
                start = time.perf_counter(); result = classifier.predict(image)
                samples.append(time.perf_counter() - start)
                if not result['classifier']['available']: raise ValueError(result['notice'])
                results.append(result['predictions'])
            if not all(r == results[0] for r in results): raise ValueError('Repeat predictions changed.')
            if classifier.sha(run / 'model.safetensors') != report['checkpoint_sha256']: raise ValueError('Weights changed.')
    if classifier.sha(classifier.CONFIG) != original_config_hash: raise ValueError('Active config changed.')
    result = {'status': 'passed', 'run': run.name, 'checkpoint_sha256': report['checkpoint_sha256'],
              'model': classifier.model_name(report['architecture']), 'image_split': 'train', 'path': row['path'],
              'cold_seconds_including_checks_and_load': samples[0], 'warm_seconds_including_checks': samples[1:],
              'predictions': results[0], 'test_images_opened': 0, 'active_config_unchanged': True,
              'notice': 'One-image local CPU mechanics/latency check, not a robust latency benchmark or accuracy assessment.'}
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

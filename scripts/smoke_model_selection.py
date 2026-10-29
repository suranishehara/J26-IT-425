"""Real three-model HTTP switching on one TRAIN image; test split not opened."""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/model_selection_smoke_01'))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise ValueError('Smoke output already exists.')
    from backend.app import Handler
    from backend.training_data import read_development
    from backend.classifier import selected_release, NINE_CATEGORIES, CATEGORIES
    prep, rows = read_development(ROOT / 'runtime/datasets/textileclass_pilot_02')
    sample = rows['train'][0]
    raw = (Path(prep['raw_root']) / sample['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample['file_sha256']: raise ValueError('Sample hash mismatch.')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    results = []
    try:
        with urlopen(base + '/api/health') as response: health = json.load(response)
        assert health['default_classifier'] == 'convnext_v2'
        assert len(health['classifiers']) == 3 and all(item['available'] for item in health['classifiers'])
        with urlopen(base + '/api/options') as response: options = json.load(response)
        assert options['fabric_categories'] == ['unknown'] + NINE_CATEGORIES
        for item in health['classifiers']:
            assert item['categories'] == (NINE_CATEGORIES if item['id'] == 'convnext_v2' else CATEGORIES)
        for model_id in ('convnext_v2', 'convnext_v1', 'efficientnet_b0', 'convnext_v2'):
            payload = {'permission': True, 'mode': 'experimental', 'model_id': model_id,
                       'image': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}
            req = Request(base + '/api/analyse-fabric', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(req, timeout=120) as response: result = json.load(response)
            assert result['classifier']['available'], result['notice']
            assert result['classifier']['model_id'] == model_id
            assert result['classifier']['checkpoint_sha256'] == selected_release(model_id)[2]
            assert len(result['predictions']['fabric_category']) == 3
            expected = NINE_CATEGORIES if model_id == 'convnext_v2' else CATEGORIES
            assert result['classifier']['categories'] == expected
            assert all(item['category'] in expected for item in result['predictions']['fabric_category'])
            results.append({'id': model_id, 'model': result['classifier']['model'], 'checkpoint_sha256': result['classifier']['checkpoint_sha256'], 'predictions': result['predictions']})
        assert results[0]['predictions'] == results[-1]['predictions']
        try:
            payload['model_id'] = '../arbitrary'
            with urlopen(Request(base + '/api/analyse-fabric', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})):
                raise AssertionError('Arbitrary model accepted.')
        except HTTPError as error: assert error.code == 400
        output.mkdir(parents=True)
        report = {'status': 'passed', 'image_split': 'train', 'test_images_opened': 0, 'runs': results,
                  'checks': ['three available models', 'default V2', 'selected checkpoint used', 'switch back repeats', 'invalid ID rejected'],
                  'notice': 'Mechanics check on one training image only, not accuracy evaluation.'}
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'passed', 'models': [item['model'] for item in results], 'test_images_opened': 0}))
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': main()

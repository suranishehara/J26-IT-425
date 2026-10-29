"""Real CPU checkpoint + HTTP smoke on one train image; no final-test inference."""
import argparse
import base64
import hashlib
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/fabric_api_smoke_01'))
    args = parser.parse_args()
    from backend.app import Handler
    from backend import classifier
    output, dataset = Path(args.output), Path(args.dataset)
    if output.exists(): raise ValueError('Choose a new smoke output directory.')
    manifest = (dataset / 'manifest.jsonl').read_bytes()
    prep = json.loads((dataset / 'preparation.json').read_text())
    if hashlib.sha256(manifest).hexdigest() != prep['manifest_sha256']: raise ValueError('Manifest changed.')
    row = next(json.loads(line) for line in manifest.decode().splitlines() if json.loads(line)['split'] == 'train')
    path = (Path(prep['raw_root']) / row['path']).resolve()
    if not path.is_relative_to(Path(prep['raw_root']).resolve()): raise ValueError('Escaping image path.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['file_sha256']: raise ValueError('Train image changed.')
    run, report, checksum = classifier.verify_release()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = 'http://127.0.0.1:' + str(server.server_port)

    def post(payload):
        with urlopen(Request(base + '/api/analyse-fabric', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=180) as response:
            assert response.headers['Cache-Control'] == 'no-store'
            return json.load(response)

    payload = {'image': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode(), 'permission': True}
    try:
        manual = post(payload)
        assert manual['predictions'] == {} and manual['classifier']['status'] == 'not_requested'
        first = post({**payload, 'mode': 'experimental'})
        assert first['classifier']['available'], first
        assert first['status'] == 'review_required'
        assert len(first['predictions']['fabric_category']) == 3
        assert not any(k in first for k in ('material', 'technique', 'properties'))
        second = post({**payload, 'mode': 'experimental'})
        assert first['predictions'] == second['predictions']
        for model_param in classifier._CACHE['network'].parameters(): assert not model_param.requires_grad
        with patch.object(classifier, 'CONFIG', output / 'deliberately_missing_release.json'):
            unavailable = post({**payload, 'mode': 'experimental'})
            assert unavailable['predictions'] == {} and unavailable['classifier']['available'] is False
        restored = post({**payload, 'mode': 'experimental'})
        assert restored['predictions'] == first['predictions']
        assert classifier.sha(run / 'model.safetensors') == checksum
        result = {'status': 'passed', 'run': run.name, 'checkpoint_sha256': checksum,
                  'image_split': 'train', 'path': row['path'], 'test_images_opened': 0,
                  'checks': ['review-only does not predict', 'opt-in CPU inference', 'top-three relative scores',
                             'no automatic composition/technique/properties', 'repeat predictions',
                             'missing release fails closed even with warm cache', 'restored configuration works', 'checkpoint unchanged'],
                  'predictions': first['predictions'], 'notice': 'Mechanics verification, not accuracy or OOD validation.'}
        output.mkdir(parents=True)
        (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': main()

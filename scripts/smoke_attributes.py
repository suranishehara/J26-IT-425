"""Exercise an actual local image-analysis HTTP request; no accuracy claim."""
import base64
import argparse
import json
from pathlib import Path
import sys
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app import Handler, ThreadingHTTPServer

if __name__ == '__main__':
    source = ROOT.parent / 'component-1-virtual-tryon' / 'data' / 'garment_references' / 'dresses' / 'dress-02-batik-wrap.png'
    parser=argparse.ArgumentParser()
    parser.add_argument('image',nargs='?',type=Path,default=source)
    parser.add_argument('--mode',choices=['zero_shot','indofashion_probe'],default='zero_shot')
    args=parser.parse_args()
    source=args.image
    payload = {'image': 'data:image/png;base64,' + base64.b64encode(source.read_bytes()).decode(), 'classifier_mode':args.mode}
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request('http://127.0.0.1:' + str(server.server_port) + '/api/analyse-garment',
                                         data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        assert set(result['attributes']) == {'garment_type','pattern_type','drape_style','fabric_texture'}
        assert result['classifier_mode']==args.mode
        for value in result['attributes'].values():
            assert value['suggested_label'] in value['allowed_labels']
            assert value['needs_review'] is True
        print(json.dumps(result, indent=2))
        print('Real image-analysis HTTP smoke test passed. This is not an accuracy evaluation.')
    finally:
        server.shutdown()
        server.server_close()

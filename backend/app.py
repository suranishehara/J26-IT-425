"""Loopback-only Component 3 server. No model training or uploads persisted."""
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import recommend, inspect_record, MATERIALS, TECHNIQUES, STYLES, DESIGNS
from backend.images import inspect
from backend.shared_contract import contract
from backend.classifier import availability, catalogue, NINE_CATEGORIES
from backend import specimens
from backend.comparison import compare
from backend.design_rules import load as load_rules
from backend.typical_profiles import catalogue as typical_catalogue


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'frontend'), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def send_json(self, status, result):
        raw = json.dumps(result, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/typical-profiles':
            return self.send_json(200, {'profiles': typical_catalogue()})
        if path in ('/api/design-rules', '/api/catalogue/export'):
            try:
                return self.send_json(200, load_rules() if path == '/api/design-rules' else specimens.export_catalogue())
            except (OSError, ValueError, TypeError) as error:
                return self.send_json(400, {'error': str(error)})
        if path == '/api/specimens':
            return self.send_json(200, {'specimens': specimens.listing()})
        if path.startswith('/api/specimens/'):
            try:
                return self.send_json(200, specimens.get(path.rsplit('/', 1)[1]))
            except ValueError as error:
                return self.send_json(400, {'error': str(error)})
        if path == '/api/health':
            classifier = availability()
            return self.send_json(200, {'component': 3, 'status': 'prototype', 'classifier_available': classifier['available'], 'classifier': classifier,
                                        'classifiers': catalogue(), 'default_classifier': 'convnext_v2', 'shared': contract()})
        if path == '/api/options':
            try:
                rules = load_rules()
                return self.send_json(200, {**contract(), 'materials': MATERIALS, 'techniques': TECHNIQUES,
                                            'styles': STYLES, 'fabric_categories': ['unknown'] + NINE_CATEGORIES,
                                            'designs': [{'id': d['id'], 'name': d['name']} for d in rules['designs']]})
            except (OSError, ValueError, TypeError) as error:
                return self.send_json(400, {'error': str(error)})
        return super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ('/api/analyse-fabric', '/api/designs', '/api/designs/compare', '/api/specimens', '/api/fabric-record/check', '/api/catalogue/import'):
            return self.send_json(404, {'error': 'Unknown endpoint.'})
        try:
            # Same-origin browser requests only; reject cross-origin JSON posts.
            origin = self.headers.get('Origin')
            port = self.server.server_address[1]
            if origin and origin not in (f'http://localhost:{port}', f'http://127.0.0.1:{port}'):
                return self.send_json(403, {'error': 'Cross-origin request rejected.'})
            length = int(self.headers.get('Content-Length', '0'))
            limit = 8_100_000 if route == '/api/analyse-fabric' else (500_000 if route == '/api/catalogue/import' else 50_000)
            if not 0 < length <= limit:
                raise ValueError('Request is empty or too large.')
            payload = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON number.')))
            if route == '/api/analyse-fabric':
                result = inspect(payload)
            elif route == '/api/fabric-record/check':
                result = inspect_record(payload)
            elif route == '/api/catalogue/import':
                result = specimens.import_catalogue(payload)
            elif route == '/api/specimens':
                result = specimens.save(payload)
            else:
                canonical, provenance = specimens.design_request(payload)
                result = compare(canonical) if route == '/api/designs/compare' else recommend(canonical)
                result['specimen'] = provenance
            self.send_json(200, result)
        except (OSError, ValueError, TypeError, UnicodeDecodeError) as error:
            self.send_json(400, {'error': str(error)})


if __name__ == '__main__':
    import socket

    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    with LocalServer(('127.0.0.1', 8002), Handler) as server:
        print('Component 3 running at http://localhost:8002', flush=True)
        server.serve_forever()

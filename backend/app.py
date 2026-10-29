"""Dependency-free development server for Component 1."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from services.catalogue import load_catalogue
from services.tryon_service import TryOnService, ValidationError
from services import generation, mask_detection

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
service = TryOnService(load_catalogue(ROOT / "data" / "garments.json"))


class ComponentOneHandler(SimpleHTTPRequestHandler):
    """Serve the client and a small JSON API without external dependencies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith('/api/jobs/'):
            return self.send_json(HTTPStatus.OK, generation.status(path.rsplit('/', 1)[-1]))
        if path.startswith('/results/'):
            parts = path.split('/')
            if len(parts) != 4 or len(parts[2]) != 32 or any(c not in '0123456789abcdef' for c in parts[2]) or parts[3] != 'result.png':
                return self.send_error(404)
            target = generation.JOBS / parts[2] / 'result.png'
            if not target.is_file():
                return self.send_error(404)
            content = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if urlparse(self.path).path == "/api/garments":
            return self.send_json(HTTPStatus.OK, {"garments": service.catalogue})
        if self.path == "/":
            self.path = "/local.html"
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path not in ("/api/try-on", "/api/detect-mask"):
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8_000_000:
                return self.send_json(HTTPStatus.BAD_REQUEST, {'error': 'Upload must be below 8 MB.'})
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = mask_detection.detect(payload) if urlparse(self.path).path == '/api/detect-mask' else generation.create(payload)
            return self.send_json(HTTPStatus.CREATED, result)
        except (ValueError, ValidationError, OSError, TypeError, TimeoutError) as error:
            return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def send_json(self, status: HTTPStatus, body: dict):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    # Windows reuse-address semantics can allow two versions to serve one port.
    # Refuse an occupied port instead of silently serving stale application code.
    import socket
    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()
    print("Component 1 running at http://localhost:8000")
    LocalServer(("127.0.0.1", 8000), ComponentOneHandler).serve_forever()

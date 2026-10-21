"""Read-only loopback preview of the annotation and collection pages; no save API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=ROOT / 'runtime/evaluations/volunteer_review_01/review.html')
    parser.add_argument('--port', type=int, default=8004)
    args = parser.parse_args()
    review = args.review.resolve()
    if not review.is_file(): raise ValueError('Build the review page first.')
    pages = {'/': review, '/review.html': review, '/collect.html': ROOT / 'review_tool/collect.html'}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = pages.get(urlsplit(self.path).path)
            if path is None:
                self.send_error(404); return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(data)

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            import socket
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    with Server(('127.0.0.1', args.port), Handler) as server:
        print(f'Local review: http://localhost:{args.port}/review.html', flush=True)
        print(f'Collection form: http://localhost:{args.port}/collect.html', flush=True)
        server.serve_forever()


if __name__ == '__main__': main()

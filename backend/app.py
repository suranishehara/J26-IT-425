"""Local recommendation demo; separate port from Component 1."""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from engine import recommend
from catalogue import validate_catalogue
from storage import load_saved, save_reviewed, image_bytes
from user_profile import validate_profile

ROOT = Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "frontend"), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def do_GET(self):
        if self.path=='/api/health':
            from local_attributes import available_heads
            from attributes import MODEL_REVISION
            local_heads=available_heads(MODEL_REVISION)
            from dinov2_pattern import available as dino_available
            content=json.dumps({'component':2,'api_version':2,'features':['garment-analysis','person-analysis','catalogue','profile','recommendations','local-attribute-heads'],
                                'local_attributes_configured':bool(local_heads),'local_attribute_labels':local_heads,
                                'dinov2_pattern_available':dino_available()}).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path.startswith('/api/garment-images/'):
            identifier = self.path.rsplit('/', 1)[-1]
            if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
                return self.send_error(404)
            image = image_bytes(identifier)
            if image is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in ("/api/recommendations", "/api/analyse-garment", "/api/analyse-person", "/api/catalogue", "/api/profile"):
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > (12_000_000 if self.path in ("/api/analyse-garment", "/api/analyse-person", "/api/catalogue") else 100000):
                raise ValueError("Invalid request size.")
            profile = json.loads(self.rfile.read(length))
            if self.path == "/api/analyse-garment":
                from attributes import analyse
                result = analyse(profile)
            elif self.path == '/api/analyse-person':
                from person_analysis import analyse_person
                result=analyse_person(profile)
            elif self.path == '/api/catalogue':
                result = {'garment': save_reviewed(profile)}
            elif self.path == '/api/profile':
                result = {'profile':validate_profile(profile)}
            else:
                garments = validate_catalogue(json.loads((ROOT / "data" / "garments.json").read_text(encoding="utf-8")))
                result = recommend(garments + validate_catalogue(load_saved()), profile)
            status = 200
        except (ValueError, TypeError) as error:
            result, status = {"error": str(error)}, 400
        except Exception as error:
            print('Request failed:', repr(error))
            result, status = {"error": "The request could not complete. Check the server terminal and retry."}, 503
        content = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8001), Handler)
    print("Component 2 running at http://localhost:8001")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

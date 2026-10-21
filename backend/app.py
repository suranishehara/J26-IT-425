"""Separate loopback service. No account, model downloads or upstream writes."""
import json
import sqlite3
import sys
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dialogue import new_session, chat, prepare, stream_prepared
from backend.retrieval import load
from backend.llm import models
from backend.semantic import status as retrieval_status
from backend import dialogue
from backend.history import HistoryStore


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(ROOT / 'frontend'), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def send_json(self, code, value):
        raw = json.dumps(value, allow_nan=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/api/chats':
            try: return self.send_json(200, dialogue.history_list())
            except sqlite3.Error: return self.send_json(503, {'error': 'Local history is temporarily unavailable.'})
        if route.startswith('/api/chats/'):
            try: return self.send_json(200, dialogue.saved_chat(route[len('/api/chats/'):]))
            except (ValueError, sqlite3.Error) as error: return self.send_json(400, {'error': str(error)})
        if route == '/api/models': return self.send_json(200, models())
        if route == '/api/retrieval': return self.send_json(200, retrieval_status())
        if route in ('/api/health', '/api/knowledge'):
            try:
                entries, checksum = load()
                return self.send_json(200, {'component': 4, 'status': 'baseline_prototype', 'entries': entries if route.endswith('knowledge') else len(entries),
                                           'knowledge_sha256': checksum, 'nlu': 'english_rules_v0.1', 'retrieval': 'bm25', 'llm_enabled': False,
                                           'semantic_retrieval_enabled': False, 'upstream_mode': 'explicit_json_read_only',
                                           'semantic_retrieval_supported': True, 'retrieval_selection_per_request': True,
                                           'history_enabled': dialogue.STORE is not None,
                                           'llm_supported': True, 'llm_selection_per_request': True, 'default_generation': 'baseline',
                                           'llm_provider': 'optional_local_ollama', 'streaming_supported': True})
            except (ValueError, OSError, KeyError) as error: return self.send_json(400, {'error': str(error)})
        return super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ('/api/session', '/api/chat', '/api/chat/stream', '/api/chats/rename', '/api/chats/delete', '/api/chats/restore'): return self.send_json(404, {'error': 'Unknown endpoint.'})
        try:
            origin = self.headers.get('Origin'); port = self.server.server_port
            if origin and origin not in (f'http://localhost:{port}', f'http://127.0.0.1:{port}'):
                return self.send_json(403, {'error': 'Cross-origin request rejected.'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 50000: raise ValueError('Request must be 1–50000 bytes.')
            payload = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON.')))
            if not isinstance(payload, dict): raise ValueError('Expected a JSON object.')
            if route.startswith('/api/chats/'):
                return self.send_json(200, dialogue.history_action(route.rsplit('/', 1)[-1], payload))
            if route == '/api/chat/stream':
                prepared = prepare(payload)
                events = stream_prepared(prepared)
                try:
                    first = next(events)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                    self.send_header('Connection', 'close'); self.end_headers(); self.close_connection = True
                    def emit(event):
                        self.wfile.write((json.dumps(event, allow_nan=False) + '\n').encode()); self.wfile.flush()
                    emit(first)
                    try:
                        for event in events: emit(event)
                    except (ValueError, OSError, TypeError, KeyError, sqlite3.Error):
                        emit({'type': 'error', 'error': 'Generation did not complete. Check the local model and retry. No baseline was substituted; incomplete text is not a committed answer.'})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    events.close()
                return
            result = new_session() if route == '/api/session' else chat(payload)
            return self.send_json(200, result)
        except (ValueError, TypeError, OSError, KeyError, sqlite3.Error) as error: return self.send_json(400, {'error': str(error)})


if __name__ == '__main__':
    dialogue.STORE = HistoryStore(ROOT / 'runtime/chat_history.sqlite3')
    import socket
    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'): self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()
    with Server(('127.0.0.1', 8003), Handler) as server:
        print('Component 4 running at http://localhost:8003', flush=True)
        server.serve_forever()

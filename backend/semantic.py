"""Local sentence embeddings; bounded in-memory corpus cache, no automatic downloads."""
import json
import math
import threading
from .llm import open_local

MODEL = 'all-minilm:22m'
LOCK = threading.Lock()
CACHE = {}


def identity():
    with open_local('/api/tags') as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError('Oversized model inventory.')
    for row in json.loads(raw)['models']:
        if row.get('name') == MODEL and isinstance(row.get('digest'), str) and row['digest']:
            return {'name': MODEL, 'digest': row['digest']}
    raise ValueError('Local embedding model all-minilm:22m is not installed. Select BM25 or run the explicit setup script.')


def status():
    try:
        return {'available': True, 'model': identity(), 'methods': ['bm25', 'dense', 'hybrid']}
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'model': None, 'methods': ['bm25'],
                'notice': 'Semantic retrieval needs local Ollama and all-minilm:22m. No download or silent fallback occurs.'}


def embed(texts):
    with open_local('/api/embed', {'model': MODEL, 'input': texts, 'truncate': False,
                                 'keep_alive': '5m'}, timeout=60) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError('Oversized embedding response.')
    vectors = json.loads(raw).get('embeddings')
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError('Embedding count does not match input.')
    normalized, dimension = [], None
    for vector in vectors:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 4096:
            raise ValueError('Invalid embedding dimension.')
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise ValueError('Non-finite or invalid embedding.')
        dimension = len(vector) if dimension is None else dimension
        if len(vector) != dimension:
            raise ValueError('Inconsistent embedding dimensions.')
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('Invalid embedding norm.')
        normalized.append([value / norm for value in vector])
    return normalized


def rank(query, entries, checksum):
    selected = identity()
    key = (checksum, selected['digest'])
    # Serialize embedding/cache builds, not all conversations or generation.
    with LOCK:
        if key not in CACHE:
            texts = [row['title'] + '. ' + row['text'] + ' Scope: ' + row['scope'] for row in entries]
            vectors = embed(texts) if texts else []
            CACHE.clear()  # Only one public corpus version; never cache user queries.
            CACHE[key] = vectors
        vectors = CACHE[key]
        query_vector = embed([query])[0]
    if vectors and len(query_vector) != len(vectors[0]):
        raise ValueError('Query/corpus embedding dimensions differ.')
    scored = [{**row, 'retrieval_score': sum(a * b for a, b in zip(query_vector, vector)),
               'embedding_model': selected} for row, vector in zip(entries, vectors)]
    scored.sort(key=lambda row: (-row['retrieval_score'], row['id']))
    return scored

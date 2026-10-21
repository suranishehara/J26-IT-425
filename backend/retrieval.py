"""Selectable BM25, local dense cosine search and reciprocal-rank fusion."""
import hashlib
import json
import math
from pathlib import Path
from .nlp import terms

KNOWLEDGE = Path(__file__).resolve().parents[1] / 'data/knowledge.json'
STOP = {'a','an','the','i','me','my','is','are','to','for','of','in','it','and','or','can','you','please','what','why','how'}


def load(path=None):
    raw = Path(path or KNOWLEDGE).read_bytes(); data = json.loads(raw)
    ids = set()
    for row in data['entries']:
        if row['id'] in ids: raise ValueError('Duplicate knowledge ID.')
        ids.add(row['id'])
        if row['status'] not in ('source_checked_not_expert_reviewed', 'project_verified', 'draft'): raise ValueError('Invalid knowledge review state.')
        if not row.get('text') or not row.get('scope') or not row.get('source', {}).get('title'): raise ValueError('Knowledge needs text, scope and source.')
        if not row['source'].get('url') and not row['source'].get('path'): raise ValueError('Missing evidence reference.')
    return data['entries'], hashlib.sha256(raw).hexdigest()


def retrieve(query, k=3, method='bm25'):
    if method not in ('bm25', 'dense', 'hybrid'):
        raise ValueError('Choose bm25, dense or hybrid retrieval.')
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError('Retrieval K must be 1–20.')
    entries, checksum = load()
    entries = [row for row in entries if row['status'] != 'draft']
    tokens = [terms(row['title'] + ' ' + row['text'] + ' ' + ' '.join(row['tags'])) for row in entries]
    query_terms = set(terms(query)) - STOP
    average = sum(map(len, tokens)) / max(1, len(tokens))
    scored = []
    for row, words in zip(entries, tokens):
        score = 0
        for term in query_terms:
            tf = words.count(term)
            if not tf: continue
            df = sum(term in document for document in tokens)
            idf = math.log(1 + (len(tokens) - df + .5) / (df + .5))
            score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * len(words) / average))
        if score > 0: scored.append({**row, 'retrieval_score': round(score, 6)})
    scored.sort(key=lambda row: (-row['retrieval_score'], row['id']))
    if method == 'bm25':
        return scored[:k], checksum
    from .semantic import rank
    dense = rank(query, entries, checksum)
    if method == 'dense':
        return dense[:k], checksum
    return fuse(scored, dense)[:k], checksum


def fuse(lexical, dense):
    """Equal-weight RRF, constant 60; scores are ranking values, not confidence."""
    results = {}
    for name, rows in (('bm25', lexical), ('dense', dense)):
        for position, row in enumerate(rows, 1):
            entry = results.setdefault(row['id'], {**row, 'retrieval_score': 0, 'component_ranks': {}})
            if 'embedding_model' in row:
                entry['embedding_model'] = row['embedding_model']
            entry['retrieval_score'] += 1 / (60 + position)
            entry['component_ranks'][name] = position
    return sorted(results.values(), key=lambda row: (-row['retrieval_score'], row['id']))

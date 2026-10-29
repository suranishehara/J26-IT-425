"""Append-only local fabric records; never stores photographs or body profiles."""
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from .engine import fabric_record

DATABASE = Path(__file__).resolve().parents[1] / 'runtime' / 'fabric_properties.sqlite3'


@contextmanager
def connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS specimens (id TEXT PRIMARY KEY, name TEXT NOT NULL, reviewer TEXT NOT NULL, created TEXT NOT NULL, fabric TEXT NOT NULL, sha256 TEXT NOT NULL)')
    try:
        with db:
            yield db
    finally:
        db.close()


def save(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a specimen object.')
    metadata = {}
    for field in ('name', 'reviewer'):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 120:
            raise ValueError('Specimen ' + field + ' is required (maximum 120 characters).')
        metadata[field] = value.strip()
    fabric = fabric_record(payload.get('fabric'))
    if fabric.get('record_kind') == 'typical_profile' or payload.get('typical_profile_id') is not None:
        raise ValueError('Typical profiles cannot be saved as documented samples.')
    if not fabric['properties']:
        raise ValueError('Save at least one documented physical property; unknown-only records are not specimens.')
    raw = json.dumps(fabric, sort_keys=True, separators=(',', ':'), allow_nan=False)
    record = dict(id=uuid.uuid4().hex, **metadata, created=datetime.now(timezone.utc).isoformat(),
                  fabric=raw, sha256=hashlib.sha256(raw.encode()).hexdigest())
    with connect() as db:
        db.execute('INSERT INTO specimens VALUES (:id,:name,:reviewer,:created,:fabric,:sha256)', record)
    return get(record['id'])


def get(identifier):
    if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Invalid specimen ID.')
    with connect() as db:
        row = db.execute('SELECT * FROM specimens WHERE id=?', (identifier,)).fetchone()
    if row is None:
        raise ValueError('Specimen not found.')
    record = dict(row)
    if hashlib.sha256(record['fabric'].encode()).hexdigest() != record['sha256']:
        raise ValueError('Stored specimen checksum mismatch.')
    record['fabric'] = json.loads(record['fabric'])
    record['notice'] = 'User-reviewed source assertions, not independent verification. Immutable record; save a new record for corrections.'
    return record


def listing():
    with connect() as db:
        return [dict(row) for row in db.execute('SELECT id,name,reviewer,created,sha256 FROM specimens ORDER BY created DESC')]


def export_catalogue():
    records = [get(row['id']) for row in listing()]
    return {'schema_version': 1, 'records': [
        {key: record[key] for key in ('name', 'reviewer', 'fabric', 'sha256')} for record in records],
        'notice': 'Documented user assertions, not independent verification. No images or body profiles included.'}


def import_catalogue(payload):
    """Validate the complete batch before an atomic, append-only import."""
    if not isinstance(payload, dict) or payload.get('reviewed') is not True:
        raise ValueError('Review the catalogue sources before importing.')
    catalogue = payload.get('catalogue')
    if not isinstance(catalogue, dict) or type(catalogue.get('schema_version')) is not int or catalogue['schema_version'] != 1:
        raise ValueError('Unsupported catalogue schema.')
    rows = catalogue.get('records')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 200:
        raise ValueError('Import between 1 and 200 documented samples at a time.')
    prepared = []
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Each sample must be an object.')
        for key in ('name', 'reviewer'):
            if not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 120:
                raise ValueError('Each sample needs a name and reviewer (maximum 120 characters).')
        if not isinstance(row.get('fabric'), dict): raise ValueError('Each sample needs a fabric record.')
        fabric = fabric_record({**row['fabric'], 'reviewed': True})
        if fabric.get('record_kind') == 'typical_profile' or row['fabric'].get('record_kind') == 'typical_profile':
            raise ValueError('Typical profiles cannot be imported as documented samples.')
        if not fabric['properties']: raise ValueError('Every imported sample needs a documented physical property.')
        raw = json.dumps(fabric, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if 'sha256' in row and row['sha256'] != digest: raise ValueError('Imported sample checksum mismatch.')
        prepared.append({'id': uuid.uuid4().hex, 'name': row['name'].strip(), 'reviewer': row['reviewer'].strip(),
                         'created': datetime.now(timezone.utc).isoformat(), 'fabric': raw, 'sha256': digest})
    imported, skipped = [], 0
    with connect() as db:
        existing = {(row['name'], row['reviewer'], row['sha256']) for row in db.execute('SELECT name,reviewer,sha256 FROM specimens')}
        for record in prepared:
            key = (record['name'], record['reviewer'], record['sha256'])
            if key in existing:
                skipped += 1; continue
            db.execute('INSERT INTO specimens VALUES (:id,:name,:reviewer,:created,:fabric,:sha256)', record)
            existing.add(key); imported.append(record['id'])
    return {'imported_ids': imported, 'imported_count': len(imported), 'skipped_count': skipped,
            'notice': 'Existing records preserved. Imported source assertions still require independent verification.'}


def design_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a design request object.')
    ref = payload.get('specimen_id')
    if ref is None:
        return payload, None
    if payload.get('reviewed') is not True:
        raise ValueError('Review the saved specimen before continuing.')
    record = get(ref)
    fabric = dict(record['fabric'], reviewed=True)
    return dict(payload, fabric=fabric), {key: record[key] for key in ('id', 'name', 'reviewer', 'created', 'sha256')}

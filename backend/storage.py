"""Transactional local catalogue storage; uploaded images live in SQLite."""
import io
import json
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

from attributes import decode_image
from catalogue import validate_catalogue
from taxonomy import ATTRIBUTES

DATABASE = Path(__file__).resolve().parents[1] / 'runtime' / 'catalogue.sqlite3'

@contextmanager
def connect(database=DATABASE):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30)
    try:
        with connection:
            connection.execute('CREATE TABLE IF NOT EXISTS garments (id TEXT PRIMARY KEY, record TEXT NOT NULL, image BLOB NOT NULL)')
            yield connection
    finally:
        connection.close()

def load_saved(database=DATABASE):
    with connect(database) as connection:
        return [json.loads(row[0]) for row in connection.execute('SELECT record FROM garments ORDER BY rowid')]

def image_bytes(identifier, database=DATABASE):
    with connect(database) as connection:
        row = connection.execute('SELECT image FROM garments WHERE id=?', (identifier,)).fetchone()
    return row[0] if row else None

def save_reviewed(payload, database=DATABASE):
    if not isinstance(payload, dict) or payload.get('confirmed') is not True:
        raise ValueError('Confirm that you reviewed the attributes and can use this image.')
    record = payload.get('record')
    if not isinstance(record, dict):
        raise ValueError('Garment details are required.')
    reviewer = payload.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
        raise ValueError('Enter a reviewer name or ID (up to 100 characters).')
    if not isinstance(record.get('name'), str) or not 1 <= len(record['name'].strip()) <= 150:
        raise ValueError('Enter a garment name (up to 150 characters).')
    identifier = uuid.uuid4().hex
    saved = {key: record.get(key) for key in (*ATTRIBUTES, 'name', 'family', 'price_lkr',
             'price_source', 'occasions', 'weather', 'colours', 'styles', 'textile_techniques')}
    saved.update(id=identifier, fabric='unknown', image_path=None,
                 image_url='/api/garment-images/' + identifier,
                 data_status='user_reviewed', attribute_status='user_reviewed',
                 source={'kind':'user_provided', 'permission_confirmed':True},
                 reviewer_id=reviewer.strip(), reviewed_at=datetime.now(timezone.utc).isoformat())
    saved['size_chart']=record.get('size_chart',[])
    validate_catalogue([saved])
    if saved['garment_type'] != 'saree' and saved['drape_style'] not in ('not_applicable', 'unknown'):
        raise ValueError('Select not applicable for the drape of a non-saree garment.')
    image = decode_image(payload)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    with connect(database) as connection:
        connection.execute('INSERT INTO garments VALUES (?, ?, ?)',
                           (identifier, json.dumps(saved), buffer.getvalue()))
    return saved

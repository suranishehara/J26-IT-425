"""Fictional fixtures in temporary databases only; not textile evidence."""
import tempfile
import unittest
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from pathlib import Path
from unittest.mock import patch
from backend import specimens
from backend.engine import recommend


class SpecimenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(specimens, 'DATABASE', Path(self.temp.name) / 'test.sqlite3')
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def payload(self):
        return {'name': 'FICTIONAL TEST ONLY', 'reviewer': 'test-fixture', 'fabric': {
            'reviewed': True, 'properties': {'drapability': {'value': 'fluid', 'source': 'user_measured', 'reference': 'Fictional fixture; not a real measurement'}}},
            'profile': {'secret': 'must not persist'}, 'image': 'must not persist'}

    def test_roundtrip_and_append_only(self):
        first = specimens.save(self.payload())
        second = specimens.save(self.payload())
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(first, specimens.get(first['id']))
        self.assertEqual(len(specimens.listing()), 2)
        self.assertNotIn('profile', first)
        self.assertNotIn('image', first)

    def test_requires_review_properties_and_reference(self):
        for fabric in ({'reviewed': False}, {'reviewed': True},
                       {'reviewed': True, 'properties': {'gsm': {'value': 100, 'source': 'supplier_documented'}}}):
            payload = self.payload(); payload['fabric'] = fabric
            with self.assertRaises(ValueError): specimens.save(payload)
        self.assertEqual(specimens.listing(), [])

    def test_server_uses_saved_properties_not_client_override(self):
        record = specimens.save(self.payload())
        payload, provenance = specimens.design_request({'specimen_id': record['id'], 'reviewed': True,
            'fabric': {'reviewed': True, 'properties': {}}})
        result = recommend(payload)
        self.assertEqual(result['fabric']['properties']['drapability']['value'], 'fluid')
        self.assertEqual(provenance['sha256'], record['sha256'])

    def test_missing_review_or_unknown_id_rejected(self):
        with self.assertRaises(ValueError): specimens.design_request({'specimen_id': '0' * 32})
        with self.assertRaises(ValueError): specimens.get('0' * 32)
        with self.assertRaises(ValueError): specimens.get('../escape')

    def test_tamper_detected(self):
        record = specimens.save(self.payload())
        with specimens.connect() as db:
            db.execute('UPDATE specimens SET fabric=? WHERE id=?', ('{}', record['id']))
        with self.assertRaisesRegex(ValueError, 'checksum'): specimens.get(record['id'])

    def test_http_save_list_read_and_recommend(self):
        from backend.app import Handler
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def call(path, payload=None):
                raw = None if payload is None else json.dumps(payload).encode()
                req = Request(f'http://127.0.0.1:{server.server_port}' + path, data=raw,
                              headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=10) as response:
                    return json.load(response)
            record = call('/api/specimens', self.payload())
            self.assertEqual(call('/api/specimens')['specimens'][0]['id'], record['id'])
            self.assertEqual(call('/api/specimens/' + record['id'])['sha256'], record['sha256'])
            result = call('/api/designs', {'specimen_id': record['id'], 'reviewed': True})
            self.assertEqual(result['specimen']['id'], record['id'])
            self.assertEqual(len(result['designs']), 4)
            comparison = call('/api/designs/compare', {'specimen_id': record['id'], 'reviewed': True,
                'fabric': {'reviewed': True, 'properties': {}}})
            self.assertEqual(comparison['specimen']['sha256'], record['sha256'])
            self.assertEqual(comparison['comparison']['properties_used_for_ranking'], ['drapability'])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

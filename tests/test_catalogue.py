import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from backend import specimens
from backend.app import Handler


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        patcher = patch.object(specimens, 'DATABASE', Path(self.temp.name) / 'samples.sqlite3')
        patcher.start(); self.addCleanup(patcher.stop)

    def payload(self):
        return {'reviewed': True, 'catalogue': {'schema_version': 1, 'records': [{
            'name': 'FICTIONAL TEST ONLY', 'reviewer': 'test', 'fabric': {'fabric_category': 'linen',
            'properties': {'gsm': {'value': 180, 'source': 'supplier_documented', 'reference': 'FICTIONAL TEST sheet'}}}}]}}

    def test_roundtrip_and_no_duplicate_writes(self):
        first = specimens.import_catalogue(self.payload())
        self.assertEqual(first['imported_count'], 1)
        exported = specimens.export_catalogue()
        again = specimens.import_catalogue({'reviewed': True, 'catalogue': exported})
        self.assertEqual(again['skipped_count'], 1)
        self.assertEqual(len(specimens.listing()), 1)
        self.assertNotIn('reviewed', exported['records'][0]['fabric'])

    def test_whole_batch_validated_before_writes(self):
        payload = self.payload(); bad = copy.deepcopy(payload['catalogue']['records'][0])
        bad['fabric']['properties']['gsm'].pop('reference')
        payload['catalogue']['records'].append(bad)
        with self.assertRaises(ValueError): specimens.import_catalogue(payload)
        self.assertEqual(specimens.listing(), [])

    def test_reject_missing_review_tamper_and_undocumented_samples(self):
        for mutation in ('review', 'hash', 'properties'):
            payload = self.payload()
            if mutation == 'review': payload['reviewed'] = False
            if mutation == 'hash': payload['catalogue']['records'][0]['sha256'] = 'wrong'
            if mutation == 'properties': payload['catalogue']['records'][0]['fabric']['properties'] = {}
            with self.assertRaises(ValueError): specimens.import_catalogue(payload)
        self.assertEqual(specimens.listing(), [])

    def test_import_never_persists_photos_or_profiles(self):
        payload = self.payload(); row = payload['catalogue']['records'][0]
        row.update(image='secret', profile={'waist': 75}); row['fabric']['image'] = 'secret'
        specimens.import_catalogue(payload)
        exported = json.dumps(specimens.export_catalogue())
        self.assertNotIn('secret', exported); self.assertNotIn('waist', exported)

    def test_http_catalogue_and_rules(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            req = Request(base + '/api/catalogue/import', data=json.dumps(self.payload()).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(req) as response: self.assertEqual(json.load(response)['imported_count'], 1)
            with urlopen(base + '/api/catalogue/export') as response: self.assertEqual(len(json.load(response)['records']), 1)
            with urlopen(base + '/api/design-rules') as response: self.assertEqual(json.load(response)['review']['status'], 'pending')
            req.add_header('Origin', 'https://example.com')
            with self.assertRaises(HTTPError) as error: urlopen(req)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown(); server.server_close(); thread.join()

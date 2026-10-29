import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from backend.engine import inspect_record
from backend.app import Handler


class FabricRecordCheckTests(unittest.TestCase):
    def test_unknown_is_valid_and_does_not_invent_properties(self):
        result = inspect_record({'fabric': {'fabric_category': 'leather'}})
        self.assertEqual(result['fabric']['material'], 'unknown')
        self.assertEqual(result['fabric']['properties'], {})
        self.assertEqual(result['ranking_properties'], [])
        self.assertFalse(result['can_save_specimen'])

    def test_documented_properties_and_ranking_scope(self):
        properties = {name: {'value': value, 'source': 'supplier_documented', 'reference': 'sample A specification'}
                      for name, value in [('drapability', 'structured'), ('weight_band', 'medium'), ('gsm', 180), ('stretch_percent', 0)]}
        result = inspect_record({'fabric': {'properties': properties}})
        self.assertEqual(result['ranking_properties'], ['drapability', 'weight_band'])
        self.assertEqual(result['context_only_properties'], ['gsm', 'stretch_percent'])
        self.assertTrue(result['can_save_specimen'])
        self.assertNotIn('reviewed', result['fabric'])

    def test_invalid_drafts_fail(self):
        for payload in (None, {}, {'fabric': []}, {'fabric': {'fabric_category': 'made_up'}},
                        {'fabric': {'properties': {'gsm': {'value': 100}}}}):
            with self.assertRaises(ValueError): inspect_record(payload)

    def test_http_check_and_cross_origin(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/fabric-record/check'
            data = json.dumps({'fabric': {'fabric_category': 'linen'}}).encode()
            with urlopen(Request(url, data=data, headers={'Content-Type': 'application/json'})) as response:
                self.assertEqual(json.load(response)['fabric']['fabric_category'], 'linen')
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url, data=data, headers={'Content-Type': 'application/json', 'Origin': 'https://example.com'}))
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown(); server.server_close(); thread.join()

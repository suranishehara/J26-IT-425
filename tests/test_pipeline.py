import base64
import copy
import io
import json
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from backend.engine import recommend
from backend.images import inspect
from backend.shared_contract import contract, measurements
from backend.app import Handler


def payload():
    return {'fabric': {'reviewed': True, 'material': 'cotton', 'technique': 'handloom', 'properties': {}},
            'profile': {'unit': 'cm', 'measurements': {'chest': 90, 'waist': 75, 'hips': 100}}, 'style': 'office'}


class PipelineTests(unittest.TestCase):
    def test_shared_inches(self):
        self.assertEqual(measurements({'unit': 'in', 'measurements': {'waist': 30}})['measurements_cm']['waist'], 76.2)

    def test_shared_invalid(self):
        for value in (True, -3, 'nan', 401):
            with self.assertRaises(ValueError):
                measurements({'measurements': {'waist': value}})

    def test_shared_taxonomy(self):
        self.assertIn('coarse_woven', contract()['fabric_texture'])
        self.assertIn('kandyan_osariya', contract()['drape_style'])

    def test_labels_never_imply_properties(self):
        result = recommend(payload())
        self.assertEqual(result['fabric']['properties'], {})
        self.assertTrue(any('No recorded' in w for w in result['warnings']))

    def test_technique_does_not_change_score(self):
        a, b = payload(), payload()
        b['fabric']['technique'] = 'batik'
        self.assertEqual(recommend(a)['designs'], recommend(b)['designs'])

    def test_known_property_needs_provenance(self):
        p = payload(); p['fabric']['properties']['drapability'] = {'value': 'fluid'}
        with self.assertRaises(ValueError): recommend(p)

    def test_property_changes_rule_score(self):
        p = payload()
        p['fabric']['properties']['drapability'] = {'value': 'structured', 'source': 'expert_reviewed', 'reference': 'sample 1 review'}
        result = recommend(p)
        self.assertEqual(result['designs'][0]['id'], 'structured_shirt')
        self.assertEqual(result['designs'][0]['rule_score'], 5)

    def test_ease_arithmetic(self):
        result = recommend(payload())
        tunic = next(d for d in result['designs'] if d['id'] == 'relaxed_tunic')
        self.assertEqual(tunic['measurements']['chest']['draft_finished_circumference_cm'], [100, 104])

    def test_missing_measurements_stay_missing(self):
        p = payload(); p['profile']['measurements'] = {}
        self.assertTrue(all(not d['measurements'] for d in recommend(p)['designs']))

    def test_review_required(self):
        p = payload(); p['fabric']['reviewed'] = False
        with self.assertRaises(ValueError): recommend(p)

    def test_invalid_known_property(self):
        for value in ('nan', -1, True, 3000):
            p = payload(); p['fabric']['properties']['gsm'] = {'value': value, 'source': 'user_measured', 'reference': 'record'}
            with self.assertRaises(ValueError): recommend(p)

    def test_stretch_zero_allowed_negative_rejected(self):
        p = payload()
        p['fabric']['properties']['stretch_percent'] = {'value': 0, 'source': 'user_measured', 'reference': 'test 1'}
        self.assertEqual(recommend(p)['fabric']['properties']['stretch_percent']['value'], 0)
        p['fabric']['properties']['stretch_percent']['value'] = -0.5
        with self.assertRaises(ValueError): recommend(p)

    def test_image_has_no_fabric_predictions(self):
        buffer = io.BytesIO(); Image.new('RGB', (128, 128), 'blue').save(buffer, 'PNG')
        result = inspect({'permission': True, 'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
        self.assertFalse(result['classifier']['trained'])
        self.assertEqual(result['predictions'], {})
        self.assertEqual(result['classifier']['status'], 'not_requested')

    def test_review_only_does_not_invoke_model(self):
        buffer = io.BytesIO(); Image.new('RGB', (128, 128), 'blue').save(buffer, 'PNG')
        with patch('backend.classifier.predict', side_effect=AssertionError('Model must not run')):
            result = inspect({'permission': True, 'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
            self.assertEqual(result['predictions'], {})

    def test_experimental_suggestions_never_become_properties(self):
        buffer = io.BytesIO(); Image.new('RGB', (128, 128), 'blue').save(buffer, 'PNG')
        with patch('backend.classifier.predict', return_value={'classifier': {'trained': True}, 'predictions': {'fabric_category': [{'category': 'denim', 'relative_score': 0.7}]}, 'notice': 'Review required'}):
            result = inspect({'permission': True, 'mode': 'experimental', 'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
        self.assertNotIn('properties', result)
        self.assertNotIn('material', result)
        self.assertEqual(result['status'], 'review_required')

    def test_category_alone_does_not_change_recommendations(self):
        a, b = payload(), payload()
        b['fabric']['fabric_category'] = 'denim'
        self.assertEqual(recommend(a)['designs'], recommend(b)['designs'])
        self.assertEqual(recommend(b)['fabric']['material'], 'cotton')

    def test_invalid_category_is_rejected(self):
        p = payload(); p['fabric']['fabric_category'] = 'handloom'
        with self.assertRaises(ValueError): recommend(p)

    def test_image_permission_and_invalid(self):
        for p in ({}, {'permission': True, 'image': 'data:image/png;base64,invalid!'}):
            with self.assertRaises(ValueError): inspect(p)

    def test_invalid_analysis_mode(self):
        with self.assertRaisesRegex(ValueError, 'mode'):
            inspect({'permission': True, 'mode': 'automatic_physical_properties'})

    def test_no_mutation(self):
        p = payload(); original = copy.deepcopy(p); recommend(p); self.assertEqual(p, original)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_health(self):
        with patch('backend.app.availability', return_value={'available': False, 'notice': 'fixture disabled'}):
            with urlopen(self.base + '/api/health') as response:
                self.assertFalse(json.load(response)['classifier_available'])

    def test_design_endpoint(self):
        req = Request(self.base + '/api/designs', data=json.dumps(payload()).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(req) as response:
            self.assertEqual(len(json.load(response)['designs']), 4)

    def test_invalid_json(self):
        req = Request(self.base + '/api/designs', data=b'{bad')
        with self.assertRaises(HTTPError) as ctx: urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_cross_origin(self):
        req = Request(self.base + '/api/designs', data=b'{}', headers={'Origin': 'https://example.com'})
        with self.assertRaises(HTTPError) as ctx: urlopen(req)
        self.assertEqual(ctx.exception.code, 403)

    def test_frontend(self):
        with urlopen(self.base + '/') as response:
            self.assertIn(b'Fabric to design', response.read())


if __name__ == '__main__': unittest.main()

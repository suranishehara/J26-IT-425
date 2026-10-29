import copy
import json
from pathlib import Path
import tempfile
import unittest
from backend.design_rules import load
from backend.engine import recommend
from backend.comparison import compare


class DesignRuleTests(unittest.TestCase):
    def test_existing_results_and_rule_identity(self):
        payload = {'fabric': {'reviewed': True, 'properties': {
            'drapability': {'value': 'structured', 'source': 'user_measured', 'reference': 'FICTIONAL TEST'},
            'weight_band': {'value': 'medium', 'source': 'user_measured', 'reference': 'FICTIONAL TEST'}}},
            'profile': {'measurements': {'chest': 90, 'waist': 75}}, 'style': 'office'}
        result = compare(payload)
        self.assertEqual(result['designs'][0]['rule_score'], 7)
        self.assertEqual(result['designs'][0]['measurements']['chest']['draft_finished_circumference_cm'], [96, 100])
        self.assertEqual(result['rule_config']['sha256'], load()['sha256'])
        self.assertEqual(result['rule_config']['review']['status'], 'pending')

    def test_invalid_rules_rejected(self):
        original = load(); original.pop('sha256')
        invalid = []
        def change(mutator):
            data = copy.deepcopy(original); mutator(data); invalid.append(data)
        change(lambda data: data['designs'].append(copy.deepcopy(data['designs'][0])))
        change(lambda data: data['review'].update(status='reviewed'))
        change(lambda data: data['weights'].update(style=True))
        change(lambda data: data['weights'].update(style=float('nan')))
        change(lambda data: data['designs'][0]['ease'].update(chest=[14, 10]))
        change(lambda data: data['designs'][0]['ease'].update(chest=[-2, 3]))
        change(lambda data: data['designs'][0].update(drapability=['invented']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rules.json'
            for data in invalid:
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load(path)

    def test_reviewed_metadata_and_changed_hash(self):
        original = load(); data = copy.deepcopy(original); data.pop('sha256')
        data['review'] = {'status': 'reviewed', 'reviewer': 'FICTIONAL TEST REVIEWER', 'reference': 'TEST ONLY'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rules.json'; path.write_text(json.dumps(data))
            result = load(path)
            self.assertNotEqual(result['sha256'], original['sha256'])
            self.assertEqual(result['review']['status'], 'reviewed')

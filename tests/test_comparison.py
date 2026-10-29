"""Fictional ablation fixtures; these are not validated fabric observations."""
from copy import deepcopy
import unittest
from backend.comparison import compare


class ComparisonTests(unittest.TestCase):
    def payload(self, properties=None):
        return {'fabric': {'reviewed': True, 'properties': properties or {}},
                'style': 'office', 'profile': {'unit': 'cm', 'measurements': {'chest': 90}}}

    def prop(self, value):
        return {'value': value, 'source': 'user_measured', 'reference': 'FICTIONAL TEST FIXTURE ONLY'}

    def test_unknown_properties_identical(self):
        comparison = compare(self.payload())['comparison']
        self.assertFalse(comparison['ranking_changed'])
        self.assertEqual(comparison['properties_used_for_ranking'], [])
        self.assertTrue(all(row['property_score_delta'] == 0 for row in comparison['rows']))

    def test_fluid_light_changes_top_choice(self):
        result = compare(self.payload({'drapability': self.prop('fluid'), 'weight_band': self.prop('light')}))
        self.assertEqual(result['designs'][0]['id'], 'flowing_skirt')
        self.assertTrue(result['comparison']['top_design_changed'])
        rows = {row['id']: row for row in result['comparison']['rows']}
        self.assertEqual(rows['flowing_skirt']['property_score_delta'], 5)
        self.assertEqual(rows['structured_shirt']['baseline_rank'], 1)
        self.assertEqual(rows['structured_shirt']['property_score_delta'], -1)

    def test_context_only_gsm_not_scored(self):
        result = compare(self.payload({'gsm': self.prop(120)}))['comparison']
        self.assertFalse(result['ranking_changed'])
        self.assertEqual(result['context_only_properties'], ['gsm'])

    def test_does_not_mutate_input(self):
        payload = self.payload({'drapability': self.prop('fluid')}); original = deepcopy(payload)
        compare(payload)
        self.assertEqual(payload, original)

    def test_rejects_unreviewed_or_unsourced(self):
        payload = self.payload(); payload['fabric']['reviewed'] = False
        with self.assertRaises(ValueError): compare(payload)
        with self.assertRaises(ValueError): compare(self.payload({'gsm': {'value': 120}}))

    def test_preferences_retained_in_baseline(self):
        payload = self.payload(); payload['silhouette'] = 'relaxed_tunic'
        result = compare(payload)
        rows = {row['id']: row for row in result['comparison']['rows']}
        self.assertEqual(rows['relaxed_tunic']['baseline_score'], 3)
        self.assertEqual(rows['structured_shirt']['baseline_score'], 2)

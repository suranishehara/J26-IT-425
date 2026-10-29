import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import specimens
from backend.engine import recommend, inspect_record
from backend.comparison import compare
from backend.typical_profiles import catalogue, fabric, resolve


class TypicalProfileTests(unittest.TestCase):
    def test_profiles_have_sources_and_are_separate(self):
        self.assertEqual(len(catalogue()), 3)
        for row in catalogue():
            record = inspect_record({'typical_profile_id': row['id'], 'reviewed': True})
            self.assertFalse(record['can_save_specimen'])
            self.assertEqual(record['fabric']['record_kind'], 'typical_profile')
            self.assertTrue(all(prop['source'] == 'typical_profile' for prop in record['fabric']['properties'].values()))

    def test_explicit_confirmation_and_allowlist(self):
        for value in ({'typical_profile_id': 'cotton_oxford'}, {'typical_profile_id': '../file', 'reviewed': True},
                      {'typical_profile_id': 'cotton_oxford', 'reviewed': True, 'specimen_id': 'saved'}):
            with self.assertRaises(ValueError): resolve(value)

    def test_client_overrides_not_used(self):
        result = recommend({'typical_profile_id': 'cotton_oxford', 'reviewed': True,
                            'fabric': {'properties': {'gsm': {'value': 999}}}, 'style': 'office'})
        self.assertEqual(result['fabric']['properties']['gsm']['value'], 194)
        self.assertEqual(result['property_basis'], 'typical_profile')
        self.assertEqual(result['designs'][0]['rule_score'], 7)

    def test_comparison_removes_preset_for_baseline(self):
        result = compare({'typical_profile_id': 'cotton_oxford', 'reviewed': True, 'style': 'office'})
        row = next(row for row in result['comparison']['rows'] if row['id'] == 'structured_shirt')
        self.assertEqual(row['baseline_score'], 2)
        self.assertEqual(row['property_aware_score'], 7)

    def test_cannot_save_or_import_as_documented_measurements(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(specimens, 'DATABASE', Path(tmp) / 'samples.sqlite3'):
            row = {'name': 'not a measured sample', 'reviewer': 'test', 'fabric': fabric('polyester_chiffon')}
            with self.assertRaises(ValueError): specimens.save(row)
            with self.assertRaises(ValueError): specimens.import_catalogue({'reviewed': True, 'catalogue': {'schema_version': 1, 'records': [row]}})
            self.assertEqual(specimens.listing(), [])

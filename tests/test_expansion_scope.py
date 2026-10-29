import json
from pathlib import Path
import unittest
from backend.classifier import CATEGORIES


class ExpansionScopeTests(unittest.TestCase):
    def test_new_scope_excludes_blends_without_relabelling_old_weights(self):
        path = Path(__file__).resolve().parents[1] / 'config/fabric_category_expansion.json'
        plan = json.loads(path.read_text())
        self.assertEqual(len(plan['labels']), 9)
        self.assertEqual(len(set(plan['labels'])), 9)
        self.assertNotIn('cotton_mixed', plan['labels'])
        self.assertIn('cotton_mixed', plan['excluded_labels'])
        self.assertFalse(plan['active_release_changed'])
        self.assertEqual(len(CATEGORIES), 7)
        self.assertIn('cotton_mixed', CATEGORIES)

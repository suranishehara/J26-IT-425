import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from scripts.audit_added_categories import audit, sample_key


class AddedCategoryAuditTests(unittest.TestCase):
    def test_related_view_names_grouped(self):
        self.assertEqual(sample_key('Leather/leather_16_im_1.png'), sample_key('Leather/leather_16_im_4.png'))
        self.assertNotEqual(sample_key('linen_17_im_1.png'), sample_key('linen_18_im_1.png'))

    def test_exact_duplicates_reported_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ('linen', 'Nylon', 'Leather'): (root / folder).mkdir()
            first = root / 'Leather/leather_1_im_1.png'; second = root / 'Leather/leather_2_im_1.png'
            Image.new('RGB', (64, 64), (60, 70, 80)).save(first)
            second.write_bytes(first.read_bytes())
            before = {str(p): p.read_bytes() for p in (first, second)}
            previous = root / 'old.json'; previous.write_text(json.dumps([]))
            report, rows = audit(root, previous)
            self.assertEqual(report['counts']['leather'], 2)
            self.assertEqual(len(report['duplicates']['file_sha256']), 1)
            self.assertTrue(all(row['split'] == 'unassigned' for row in rows))
            self.assertTrue(all(Path(path).read_bytes() == raw for path, raw in before.items()))

import copy
import json
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from catalogue import validate_catalogue

class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((ROOT / "data" / "garments.json").read_text())
    def test_demo_catalogue_validates(self):
        self.assertEqual(len(validate_catalogue(self.rows)), 7)
    def test_osariya_is_saree_with_its_own_drape(self):
        row = next(x for x in self.rows if x["id"] == "osariya")
        self.assertEqual((row["garment_type"], row["drape_style"]), ("saree","kandyan_osariya"))
    def test_duplicate_ids_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):
            validate_catalogue(self.rows)

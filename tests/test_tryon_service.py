import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from services.catalogue import load_catalogue
from services.tryon_service import TryOnService, ValidationError


class TryOnServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = TryOnService(load_catalogue(ROOT / "data" / "garments.json"))

    def test_routes_kandyan_saree_to_traditional_branch(self):
        result = self.service.create_job({"photoName": "person.jpg", "garmentId": "kandyan-saree"})
        self.assertEqual("traditional_draping", result["route"])
        self.assertIn("traditional pleat/front-fall representation", result["checks"])

    def test_rejects_missing_portrait(self):
        with self.assertRaises(ValidationError):
            self.service.create_job({"garmentId": "navy-dress"})


if __name__ == "__main__":
    unittest.main()

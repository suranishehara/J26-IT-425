import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from engine import recommend

class RankingTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            dict(id="a", name="Saree", price_lkr=5000, occasions=["wedding"], weather=["hot"], colours=["blue"], styles=["traditional"], fabric="cotton"),
            dict(id="b", name="Dress", price_lkr=7000, occasions=["wedding"], weather=["hot"], colours=["ivory"], styles=["western"], fabric="linen"),
        ]
    def test_budget_is_a_hard_constraint(self):
        rows = recommend(self.items, dict(budget=6000))["recommendations"]
        self.assertEqual([r["id"] for r in rows], ["a"])
    def test_preferences_can_outweigh_traditional_occasion_tags(self):
        rows = recommend(self.items, dict(budget=8000, occasion="wedding", styles=["western"], colours=["ivory"]))["recommendations"]
        self.assertEqual(rows[0]["id"], "b")
    def test_invalid_budget_rejected(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                recommend(self.items, dict(budget=value))

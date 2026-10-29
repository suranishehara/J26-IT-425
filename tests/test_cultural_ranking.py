import unittest
from engine import recommend
from ranking_metrics import ranking_metrics

class CulturalRankingTests(unittest.TestCase):
    def setUp(self):
        self.items=[dict(id='western',price_lkr=5000,garment_type='dress',colours=['blue'],styles=['western']),dict(id='osariya',price_lkr=5000,garment_type='saree',drape_style='kandyan_osariya')]
    def test_guidance_is_optional_and_does_not_remove_western_options(self):
        baseline=recommend(self.items,dict(budget=6000,occasion='wedding'))
        guided=recommend(self.items,dict(budget=6000,occasion='wedding',recommendation_mode='sri_lanka_guidance'))
        self.assertTrue(all(row['cultural_guidance']['bonus']==0 for row in baseline['recommendations']))
        self.assertEqual(guided['recommendations'][0]['id'],'osariya')
        self.assertEqual(len(guided['recommendations']),2)
    def test_explicit_preferences_can_outweigh_cultural_bonus(self):
        result=recommend(self.items,dict(budget=6000,occasion='wedding',recommendation_mode='sri_lanka_guidance',colours=['blue'],styles=['western']))
        self.assertEqual(result['recommendations'][0]['id'],'western')
    def test_metrics_known_ranking(self):
        metrics=ranking_metrics(['a','b','c'],{'a':3,'b':0,'c':2},2)
        self.assertEqual(metrics['precision_at_k'],.5)
        self.assertEqual(metrics['recall_at_k'],.5)
        self.assertGreater(metrics['ndcg_at_k'],0)
        self.assertLess(metrics['ndcg_at_k'],1)
    def test_perfect_ranking_and_missing_ratings(self):
        self.assertEqual(ranking_metrics(['a','b'],{'a':3,'b':2},2)['ndcg_at_k'],1)
        with self.assertRaises(ValueError):ranking_metrics(['a'],{},1)
        self.assertIsNone(ranking_metrics(['a'],{'a':0},1)['recall_at_k'])

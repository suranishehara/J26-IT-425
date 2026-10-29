import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_development import evaluate, ranking_metrics
from backend import dialogue
from backend.retrieval import load


class EvaluationTests(unittest.TestCase):
    def test_metrics_known_ranking(self):
        value = ranking_metrics(['wrong', 'a', 'b'], ['a', 'b'], 2)
        self.assertEqual(value['recall_at_k'], .5)
        self.assertEqual(value['mrr_at_k'], .5)
        self.assertAlmostEqual(value['ndcg_at_k'], .3868528072)

    def test_empty_relevance_not_perfect_recall(self):
        value = ranking_metrics(['irrelevant'], [], 3)
        self.assertIsNone(value['recall_at_k'])
        self.assertFalse(value['no_hit'])
        self.assertTrue(ranking_metrics([], [], 3)['no_hit'])

    def test_reproducible_and_unrated(self):
        suite = json.loads((ROOT / 'data/evaluation/development.json').read_text(encoding='utf-8'))
        sessions = set(dialogue.SESSIONS)
        first, reviews = evaluate(suite)
        second, _ = evaluate(suite)
        self.assertEqual(first, second)
        self.assertEqual(sessions, set(dialogue.SESSIONS))
        self.assertEqual(len(first['cases']), 13)
        self.assertTrue(all(all(value is None for value in row['ratings'].values()) for row in reviews))
        self.assertEqual(first['human_ratings_status'], 'not_collected')

    def test_invalid_labels_and_duplicate_cases(self):
        case = {'id': 'a', 'group': 'global', 'question': 'hello', 'relevant_ids': ['invented']}
        with self.assertRaises(ValueError): evaluate({'purpose': 'test', 'cases': [case]})
        with self.assertRaises(ValueError): evaluate({'purpose': 'test', 'cases': []}, 0)

    def test_expanded_sources_keep_scope(self):
        rows, _ = load()
        rows = {row['id']: row for row in rows}
        for identifier in ('sl_batik_materials', 'sl_handloom_products', 'global_wedding_context', 'global_black_tie'):
            row = rows[identifier]
            self.assertEqual(row['status'], 'source_checked_not_expert_reviewed')
            self.assertTrue(row['source']['locator'])
            self.assertTrue(row['scope'])


if __name__ == '__main__': unittest.main()

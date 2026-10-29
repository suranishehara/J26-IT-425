import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.retrieval import load, retrieve
from scripts.evaluate_development import evaluate


class KnowledgeExpansionTests(unittest.TestCase):
    def test_corpus_review_and_source_provenance(self):
        corpus = json.loads((ROOT / 'data/knowledge.json').read_text(encoding='utf-8'))
        rows, _ = load()
        self.assertEqual(corpus['version'], 'curated-0.3')
        self.assertEqual(corpus['review']['cultural_expert_review'], 'pending')
        self.assertEqual(len(rows), 23)
        for row in rows:
            self.assertIn(row['region'], ('sri_lankan', 'global', 'project'))
            self.assertTrue(row['scope'].strip())
            self.assertTrue(row['tags'])
            self.assertRegex(row['source']['checked'], r'^\d{4}-\d{2}-\d{2}$')
            if row['region'] != 'project':
                self.assertTrue(row['source']['url'].startswith('https://'))
                self.assertEqual(row['status'], 'source_checked_not_expert_reviewed')

    def test_specific_new_topics_retrievable(self):
        # Mechanics checks on explicit terminology, not natural-language accuracy.
        for query, expected in (
            ('satin weave silk fibre', 'global_satin_weave'),
            ('black tie optional dark suit', 'global_black_tie_optional'),
            ('handloom sarongs shawls', 'sl_handloom_garments'),
            ('garment washing care labels ironing', 'global_care_labels'),
        ):
            with self.subTest(query=query):
                rows, _ = retrieve(query)
                self.assertIn(expected, [row['id'] for row in rows])

    def test_expansion_suite_valid_and_human_scores_blank(self):
        suite = json.loads((ROOT / 'data/evaluation/knowledge_expansion.json').read_text(encoding='utf-8'))
        report, reviews = evaluate(suite)
        self.assertEqual(len(report['cases']), 16)
        self.assertEqual(report['human_ratings_status'], 'not_collected')
        self.assertTrue(all(value is None for row in reviews for value in row['ratings'].values()))
        self.assertEqual({row['group'] for row in report['cases']},
                         {'sri_lankan', 'global', 'mixed', 'insufficient_evidence'})

    def test_cultural_and_measurement_boundaries_retained(self):
        rows = {row['id']: row for row in load()[0]}
        self.assertIn('not a fibre test', rows['sl_handloom_materials']['scope'])
        self.assertIn('not automatic certification', rows['sl_handloom_craft']['text'])
        self.assertIn('not permission', rows['sl_handloom_garments']['scope'])
        self.assertIn('employer', rows['global_business_casual']['text'])
        self.assertIn('not measurements', rows['c3_rules']['text'])


if __name__ == '__main__':
    unittest.main()

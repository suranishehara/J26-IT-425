import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_simmc_acts import digest, examples, metrics, read_split, select_best
from scripts.compare_simmc_results import scores


def dialogue():
    return {'dialogue_data': [{'domain': 'fashion', 'dialogue_idx': 7, 'dialogue': [
        {'turn_idx': 0, 'transcript': 'Show blue shirts.', 'system_transcript': 'Here are two options.',
         'transcript_annotated': {'act': 'REQUEST:GET', 'act_attributes': {'slot_values': {'secret': 'hidden'}}}},
        {'turn_idx': 1, 'transcript': 'Compare them.', 'system_transcript': 'CURRENT_ANSWER_MUST_NOT_LEAK',
         'transcript_annotated': {'act': 'REQUEST:COMPARE'}}]}]}


class SimmcTrainingTests(unittest.TestCase):
    def test_features_use_previous_turn_not_current_answer_or_labels(self):
        data = dialogue(); original = copy.deepcopy(data)
        rows = examples(data)
        self.assertNotIn('Here are', rows[0]['text'])
        self.assertIn('Here are two options.', rows[1]['text'])
        self.assertNotIn('CURRENT_ANSWER', rows[1]['text'])
        self.assertNotIn('REQUEST:COMPARE', rows[1]['text'])
        self.assertNotIn('hidden', rows[0]['text'])
        self.assertTrue(rows[1]['text'].startswith('Current user: Compare them.'))
        self.assertEqual(data, original)

    def test_selection_is_validation_only_and_stable_on_ties(self):
        self.assertEqual(select_best([{'macro_f1': .4}, {'macro_f1': .6}, {'macro_f1': .6}]), 1)

    def test_source_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root / 'train.fashion.json'
            file.write_text(json.dumps(dialogue()), encoding='utf-8')
            audit = {'artifacts': [{'path': file.name, 'sha256': digest(file)}]}
            self.assertEqual(len(read_split(root, 'train', audit)), 2)
            file.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): read_split(root, 'train', audit)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Optional training dependencies absent')
    def test_fixed_label_metrics_include_missing_prediction_classes(self):
        result = metrics(['a', 'b'], ['a', 'a'], ['a', 'b'])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['per_class']['b']['recall'], 0)
        self.assertEqual(result['confusion_matrix'], [[1, 0], [1, 0]])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional training dependencies absent')
    def test_bootstrap_confusion_scores_match_classification_metrics(self):
        import numpy as np
        accuracy, f1 = scores(np.array([[1, 0], [1, 0]]))
        self.assertEqual(accuracy, .5)
        self.assertAlmostEqual(f1, 1 / 3)


if __name__ == '__main__':
    unittest.main()

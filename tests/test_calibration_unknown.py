"""Artificial unit fixtures prove mechanics only, never fabric/OOD performance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from backend.calibration import fit_temperature, probability_metrics
from backend.unknown_evaluation import metrics
from scripts.calibrate_fabric import partition
from scripts.evaluate_unknown_inputs import load_manifest


class CalibrationTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_temperature_fit_does_not_worsen_fit_nll(self):
        logits = torch.tensor([[9., 0.], [8., 0.], [0., 9.], [0., 8.]])
        truth = torch.tensor([0, 1, 1, 0])
        temperature = fit_temperature(logits, truth)
        before = probability_metrics(logits, truth)
        after = probability_metrics(logits, truth, temperature)
        self.assertLessEqual(after['negative_log_likelihood'], before['negative_log_likelihood'])
        self.assertEqual(after['accuracy'], before['accuracy'])
        self.assertTrue(.05 <= temperature <= 20)

    def test_perfect_uniform_metrics(self):
        result = probability_metrics(torch.zeros(2, 2), torch.tensor([0, 1]))
        self.assertAlmostEqual(result['multiclass_brier_sum'], .5)
        self.assertAlmostEqual(result['ece_equal_width'], 0.)
        self.assertEqual(sum(row['n'] for row in result['bins']), 2)

    def test_invalid_inputs_rejected(self):
        for logits, labels in ((torch.empty(0, 2), torch.tensor([], dtype=torch.long)),
                               (torch.tensor([[float('nan'), 0.]]), torch.tensor([0])),
                               (torch.zeros(1, 2), torch.tensor([2])),
                               (torch.zeros(1, 2), torch.tensor([.5]))):
            with self.assertRaises(ValueError): fit_temperature(logits, labels)
        for temperature in (0, -1, float('inf'), True):
            with self.assertRaises(ValueError): probability_metrics(torch.zeros(1, 2), torch.tensor([0]), temperature)

    def test_partition_group_exclusion_and_determinism(self):
        rows = [{'group_id': str(index // 2)} for index in range(20)]
        first, second = partition(rows)
        self.assertEqual((first, second), partition(rows))
        self.assertFalse({rows[i]['group_id'] for i in first} & {rows[i]['group_id'] for i in second})
        self.assertEqual(sorted(first + second), list(range(len(rows))))


class UnknownMetricTests(unittest.TestCase):
    def fixture(self):
        return [{'kind': 'supported', 'prediction': 'cotton', 'truth': 'cotton', 'max_score': .9, 'accepted': True},
                {'kind': 'unsupported_fabric', 'prediction': 'silk', 'truth': 'linen', 'max_score': .3, 'accepted': False}]

    def test_perfect_separation_and_ties(self):
        rows = self.fixture(); result = metrics(rows)
        self.assertEqual(result['unknown_detection_auroc_using_one_minus_max_score'], 1)
        self.assertEqual(result['unknown_false_quick_apply_rate'], 0)
        rows[1]['max_score'] = .9
        self.assertEqual(metrics(rows)['unknown_detection_auroc_using_one_minus_max_score'], .5)

    def test_no_known_accepted_has_null_accuracy(self):
        rows = self.fixture(); rows[0]['accepted'] = False
        self.assertIsNone(metrics(rows)['supported_accuracy_when_quick_apply_allowed'])

    def test_missing_or_invalid_scores_rejected(self):
        with self.assertRaises(ValueError): metrics(self.fixture()[:1])
        rows = self.fixture(); rows[0]['max_score'] = float('nan')
        with self.assertRaises(ValueError): metrics(rows)

    def test_manifest_permission_hash_scope_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / 'manifest.jsonl'
            rows = []
            for index, (kind, category) in enumerate((('supported', 'cotton'), ('unsupported_fabric', 'linen'))):
                image = root / f'{index}.png'; image.write_bytes(f'FIXTURE {index}'.encode())
                rows.append({'id': str(index), 'specimen_id': str(index), 'path': image.name, 'split': 'development',
                             'permission': True, 'reference': 'Fictional unit fixture', 'kind': kind, 'true_category': category,
                             'provenance': 'unknown', 'file_sha256': hashlib.sha256(image.read_bytes()).hexdigest()})
            def write(): manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            write(); self.assertEqual(len(load_manifest(manifest, root, set())), 2)
            with self.assertRaisesRegex(ValueError, 'overlap'): load_manifest(manifest, root, {rows[0]['file_sha256']})
            rows[0]['permission'] = False; write()
            with self.assertRaisesRegex(ValueError, 'permission'): load_manifest(manifest, root, set())
            rows[0]['permission'] = True; rows[1]['true_category'] = 'cotton'; write()
            with self.assertRaisesRegex(ValueError, 'inventory'): load_manifest(manifest, root, set())

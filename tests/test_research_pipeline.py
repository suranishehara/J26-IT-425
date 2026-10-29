"""Temporary synthetic fixtures only: never independent evaluation evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from backend.research_data import load_manifest, value
from backend.attribute_heads import targets, masked_loss, tasks_for, evaluate_head
from scripts.evaluate_research_release import final_receipt


class ResearchDataTests(unittest.TestCase):
    def test_final_cli_requires_acknowledgement_before_reading_data(self):
        from scripts.evaluate_research_release import main
        with patch('sys.argv', ['evaluate', '--stage', 'provider-final-test', '--manifest', 'absent.jsonl', '--image-root', 'absent', '--output', 'absent-output']):
            with self.assertRaisesRegex(ValueError, 'acknowledge-final-test'): main()

    def test_attribute_final_cli_requires_acknowledgement(self):
        from scripts.evaluate_attribute_heads import main
        with patch('sys.argv', ['evaluate', '--run', 'absent', '--manifest', 'absent.jsonl', '--image-root', 'absent', '--split', 'final_test', '--output', 'absent-output']):
            with self.assertRaisesRegex(ValueError, 'Acknowledge'): main()

    def fixture(self, root):
        rows = []
        for index, split in enumerate(('train', 'validation')):
            image = root / f'{index}.png'; Image.new('RGB', (64, 64), (50 + index, 70, 90)).save(image)
            rows.append({'id': str(index), 'path': image.name, 'specimen_id': str(index), 'source_group_id': str(index),
                'image_kind': 'real', 'permission_confirmed': True, 'independence_reviewed': True,
                'permission_reference': 'SYNTHETIC UNIT FIXTURE ONLY', 'source_reference': 'temporary fixture', 'reviewer': 'test',
                'split': split, 'kind': 'supported', 'category': 'cotton', 'region': 'unknown',
                'file_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                'labels': {'batik': {'value': True, 'reference': 'fixture', 'reviewer': 'test'}, 'handloom': None}})
        return rows

    def load(self, root, rows, forbidden=()):
        manifest = root / 'manifest.jsonl'; manifest.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return load_manifest(manifest, root, forbidden)

    def test_valid_fixture_unknown_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.load(root, self.fixture(root))
            self.assertTrue(value(rows[0], 'batik')); self.assertIsNone(value(rows[0], 'handloom'))

    def test_group_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.fixture(root); rows[1]['source_group_id'] = rows[0]['source_group_id']
            with self.assertRaisesRegex(ValueError, 'crosses splits'): self.load(root, rows)

    def test_permission_and_provider_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'overlap'): self.load(root, rows, {rows[0]['file_sha256']})
            rows[0]['permission_confirmed'] = False
            with self.assertRaisesRegex(ValueError, 'permission'): self.load(root, rows)

    def test_no_filename_labels_or_unreferenced_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.fixture(root); rows[0]['labels']['handloom'] = {'value': False}
            with self.assertRaisesRegex(ValueError, 'reference'): self.load(root, rows)
            rows[0]['labels']['handloom'] = {'value': 'false', 'reference': 'fixture', 'reviewer': 'test'}
            with self.assertRaisesRegex(ValueError, 'true, false or null'): self.load(root, rows)

    def test_final_attempt_consumed_once_even_when_order_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = {'checkpoint_sha256': 'a' * 64}
            rows = [{'file_sha256': '1'}, {'file_sha256': '2'}]
            final_receipt(temp, identity, rows, 'final-test')
            with self.assertRaises(FileExistsError): final_receipt(temp, identity, list(reversed(rows)), 'final-test')


class MaskedHeadTests(unittest.TestCase):
    def test_unknown_is_not_negative_and_both_techniques_can_be_true(self):
        rows = [{'split': split, 'labels': {'batik': {'value': v}, 'handloom': {'value': v}}}
                for split, v in [('train', False), ('train', True), ('validation', True)]]
        rows.append({'split': 'train', 'labels': {}})
        tasks, _ = tasks_for(rows)
        self.assertTrue(tasks['batik']['binary']); self.assertTrue(tasks['handloom']['binary'])
        self.assertEqual(targets(rows, 'batik', [False, True]).tolist(), [0, 1, 1, -1])

    def test_binary_mask_has_zero_unknown_gradient(self):
        logits = torch.tensor([[0.], [1.], [99.]], requires_grad=True)
        loss = masked_loss(logits, torch.tensor([0, 1, -1]), binary=True); loss.backward()
        self.assertEqual(float(logits.grad[2]), 0.)
        self.assertIsNone(masked_loss(logits, torch.tensor([-1, -1, -1]), True))

    def test_categorical_mask_and_out_of_scope_truth(self):
        logits = torch.tensor([[1., 0.], [0., 1.], [50., -50.]], requires_grad=True)
        loss = masked_loss(logits, torch.tensor([0, 1, -1])); loss.backward()
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(2)))
        rows = [{'labels': {'pattern_type': {'value': 'floral'}}}, {'labels': {}}]
        result = evaluate_head(torch.tensor([[1., 0.], [0., 1.]]), rows, 'pattern_type', {'labels': ['solid', 'striped'], 'binary': False})
        self.assertEqual(result['out_of_scope_truth_n'], 1); self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['unknown_masked_n'], 1)

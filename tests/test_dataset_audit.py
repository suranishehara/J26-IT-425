import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from backend.dataset_audit import LABELS, audit, original_key


class DatasetAuditTests(unittest.TestCase):
    def fixture(self, root):
        for split in ('train', 'valid', 'test'):
            for index, label in enumerate(LABELS):
                folder = root / split / label
                folder.mkdir(parents=True)
                Image.new('RGB', (64, 64), (index * 30, {'train': 30, 'valid': 60, 'test': 90}[split], 0)).save(folder / (split + '-' + label + '.rf.123.png'))

    def test_source_key_groups_roboflow_variants(self):
        self.assertEqual(original_key('IMG_jpg.rf.aaa.jpg'), original_key('IMG_jpg.rf.bbb.jpg'))

    def test_no_raw_mutation_and_no_test_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*.png')}
            report, records = audit(root)
            self.assertEqual(len(records), 21)
            self.assertEqual(report['errors'], [])
            self.assertNotIn('predictions', report)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*.png')})

    def test_reused_filename_is_flag_not_identity_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            Image.new('RGB', (64, 64), 'red').save(root / 'train/Cotton/shared.rf.a.png')
            Image.new('RGB', (64, 64), 'blue').save(root / 'valid/Silk/shared.rf.b.png')
            report, _ = audit(root)
            self.assertEqual(len(report['groups']['source_key']['cross_split']), 1)
            self.assertEqual(len(report['groups']['source_key']['label_conflicts']), 1)

    def test_corrupt_image_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            (root / 'train/Cotton/bad.jpg').write_bytes(b'not an image')
            report, _ = audit(root)
            self.assertFalse(report['structural_checks_pass'])
            self.assertEqual(len(report['errors']), 1)


class PreparationTests(unittest.TestCase):
    def test_heldout_priority_and_no_test_membership_changes(self):
        from scripts.prepare_textileclass import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit_dir = root / 'audit'; audit_dir.mkdir()
            rows = []
            for i, label in enumerate(LABELS.values()):
                for split in ('train', 'validation', 'test'):
                    key = label + '-' + split
                    rows.append({'path': key, 'split': split, 'label': label, 'source_key': key, 'decoded_sha256': key})
            # Extra training image related to a validation source must be removed.
            rows.append({'path': 'overlap', 'split': 'train', 'label': 'cotton', 'source_key': 'cotton-validation', 'decoded_sha256': 'different'})
            (audit_dir / 'inventory.json').write_text(json.dumps(rows))
            (audit_dir / 'report.json').write_text(json.dumps({'raw_root': str(root), 'dataset': 'fixture', 'errors': [], 'near_duplicate_triage': {'examples_first_300': [], 'pair_counts': {}}}))
            info = prepare(audit_dir, root / 'derived')
            self.assertEqual(info['manifest_sha256'], hashlib.sha256((root / 'derived/manifest.jsonl').read_bytes()).hexdigest())
            kept = [json.loads(line) for line in (root / 'derived/manifest.jsonl').read_text().splitlines()]
            self.assertEqual(info['excluded_count'], 1)
            self.assertEqual(len([r for r in kept if r['split'] == 'test']), 7)
            self.assertNotIn('overlap', [r['path'] for r in kept])
            self.assertFalse(info['test_evaluated'])


if __name__ == '__main__': unittest.main()
